#!/usr/bin/env python3
"""Prepare case-local kernel metrics CSVs for slowdown asset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from tools.data_prep.slowdown.build_ddp_slowdown_assets import (
    _REQUIRED_KERNEL_FEATURES,
    load_nvtx_rows,
    load_trace_metadata,
    overlap_ns,
)


def _require_existing_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field_name} must be an existing file: {path}")


def _require_existing_dir(path: Path, field_name: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory: {path}")


def _collect_required_kernels_by_rank(
    *,
    trace_dir: Path,
    nsys_sqlite: Path,
    label_prefix: str,
) -> Dict[str, List[str]]:
    backward_by_cmd_uid, _ = load_trace_metadata(trace_dir)
    parent_by_cmd_uid, compute_windows_by_cmd_uid, kernels_by_pid = load_nvtx_rows(nsys_sqlite, label_prefix)

    required: Dict[str, set] = {}
    for cmd_uid, backward_trace in backward_by_cmd_uid.items():
        if cmd_uid not in parent_by_cmd_uid:
            raise ValueError(f'Missing NVTX parent backward_step range for cmd_uid={cmd_uid}')
        parent = parent_by_cmd_uid[cmd_uid]
        compute_windows = list(compute_windows_by_cmd_uid.get(cmd_uid, []))
        if not compute_windows:
            raise ValueError(f'Missing phase=compute NVTX windows for cmd_uid={cmd_uid}')
        pid = int(parent['global_pid'])
        rank_key = str(int(backward_trace['rank']))
        required.setdefault(rank_key, set())
        for kernel in kernels_by_pid.get(pid, []):
            if bool(kernel['is_comm']):
                continue
            for compute_start_ns, compute_end_ns in compute_windows:
                ov_ns, _, _ = overlap_ns(
                    compute_start_ns,
                    compute_end_ns,
                    int(kernel['start_ns']),
                    int(kernel['end_ns']),
                )
                if ov_ns <= 0:
                    continue
                required[rank_key].add(str(kernel['kernel_name']))
                break
    return {rank: sorted(names) for rank, names in sorted(required.items(), key=lambda item: int(item[0]))}


def _load_candidate_csvs(candidate_csv_paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in candidate_csv_paths:
        _require_existing_file(csv_path, 'candidate_csv')
        frame = pd.read_csv(csv_path)
        if 'Kernel Name' not in frame.columns:
            raise ValueError(f"Candidate CSV is missing 'Kernel Name': {csv_path}")
        missing_columns = [name for name in _REQUIRED_KERNEL_FEATURES if name not in frame.columns]
        if missing_columns:
            raise ValueError(f'Candidate CSV is missing required columns {missing_columns}: {csv_path}')
        frame = frame.copy()
        frame['Kernel Name'] = frame['Kernel Name'].astype(str).str.strip()
        if frame['Kernel Name'].eq('').any() or frame['Kernel Name'].eq('nan').any():
            raise ValueError(f'Candidate CSV contains an empty Kernel Name: {csv_path}')
        frames.append(frame)
    if not frames:
        raise ValueError('At least one candidate CSV is required')
    return pd.concat(frames, ignore_index=True)


def _apply_alias_rows(dataframe: pd.DataFrame, aliases: Mapping[str, str]) -> pd.DataFrame:
    if not aliases:
        return dataframe
    aliased_frames = [dataframe]
    for source_name, target_name in aliases.items():
        if not source_name or not target_name:
            raise ValueError(f'Invalid alias mapping: {source_name!r} -> {target_name!r}')
        matched = dataframe[dataframe['Kernel Name'] == source_name]
        if matched.empty:
            continue
        duplicated = matched.copy()
        duplicated['Kernel Name'] = target_name
        aliased_frames.append(duplicated)
    return pd.concat(aliased_frames, ignore_index=True)


def prepare_case_kernel_metrics(
    *,
    trace_dir: Path,
    nsys_sqlite: Path,
    label_prefix: str,
    candidate_csv_paths: Sequence[Path],
    output_csv: Path,
    report_json: Optional[Path] = None,
    aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    trace_dir = Path(trace_dir)
    nsys_sqlite = Path(nsys_sqlite)
    output_csv = Path(output_csv)
    _require_existing_dir(trace_dir, 'trace_dir')
    _require_existing_file(nsys_sqlite, 'nsys_sqlite')
    if not isinstance(label_prefix, str) or not label_prefix:
        raise ValueError('label_prefix must be a non-empty string')

    required_by_rank = _collect_required_kernels_by_rank(
        trace_dir=trace_dir,
        nsys_sqlite=nsys_sqlite,
        label_prefix=label_prefix,
    )
    required_kernels = sorted({name for names in required_by_rank.values() for name in names})

    merged_df = _load_candidate_csvs([Path(path) for path in candidate_csv_paths])
    merged_df = _apply_alias_rows(merged_df, aliases or {})
    filtered_df = merged_df[merged_df['Kernel Name'].isin(required_kernels)].copy()
    if filtered_df.empty:
        raise ValueError('Prepared kernel metrics are empty after filtering to required kernels')

    available_kernels = sorted(set(filtered_df['Kernel Name'].astype(str)))
    missing_by_rank: Dict[str, List[str]] = {}
    for rank, kernel_names in required_by_rank.items():
        missing_by_rank[rank] = sorted(name for name in kernel_names if name not in available_kernels)
    missing_kernels = sorted({name for names in missing_by_rank.values() for name in names})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)

    report: Dict[str, object] = {
        'trace_dir': str(trace_dir),
        'nsys_sqlite': str(nsys_sqlite),
        'candidate_csv_paths': [str(Path(path)) for path in candidate_csv_paths],
        'output_csv': str(output_csv),
        'required_kernels_by_rank': required_by_rank,
        'required_kernels': required_kernels,
        'available_kernels': available_kernels,
        'missing_kernels_by_rank': missing_by_rank,
        'missing_kernels': missing_kernels,
        'aliases': dict(aliases or {}),
    }
    if report_json is not None:
        report_json = Path(report_json)
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report


def _parse_alias(alias_text: str) -> Tuple[str, str]:
    if '=' not in alias_text:
        raise ValueError(f"Invalid alias mapping {alias_text!r}, expected src=dst")
    source_name, target_name = alias_text.split('=', 1)
    source_name = source_name.strip()
    target_name = target_name.strip()
    if not source_name or not target_name:
        raise ValueError(f"Invalid alias mapping {alias_text!r}, empty source or target")
    return source_name, target_name


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare case-local kernel metrics CSV and report missing required kernels for slowdown assets.')
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--nsys-sqlite', required=True)
    parser.add_argument('--label-prefix', required=True)
    parser.add_argument('--candidate-csv', action='append', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--report-json', default=None)
    parser.add_argument('--alias', action='append', default=[])
    args = parser.parse_args()

    aliases = dict(_parse_alias(item) for item in args.alias)
    report = prepare_case_kernel_metrics(
        trace_dir=Path(args.trace_dir),
        nsys_sqlite=Path(args.nsys_sqlite),
        label_prefix=args.label_prefix,
        candidate_csv_paths=[Path(path) for path in args.candidate_csv],
        output_csv=Path(args.output_csv),
        report_json=Path(args.report_json) if args.report_json else None,
        aliases=aliases,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
