#!/usr/bin/env python3
"""Compare distributed/scaling compute-only kernel times from Nsight analysis JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_OPS = ("forward_step", "backward_step", "optimizer_step")
DEFAULT_RANKS = (0, 7)


def parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Empty rank list")
    return values


def parse_csv_strs(raw: str) -> List[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Empty op list")
    return values


def compute_trimmed_mean(values: List[float], trim_ratio: float) -> float:
    if not values:
        raise ValueError("trimmed mean requires non-empty values")
    if trim_ratio <= 0.0 or len(values) < 3:
        return mean(values)
    trim_each_side = int(len(values) * trim_ratio)
    if trim_each_side <= 0:
        trim_each_side = 1
    if trim_each_side * 2 >= len(values):
        trim_each_side = (len(values) - 1) // 2
    if trim_each_side <= 0:
        return mean(values)
    sorted_values = sorted(values)
    trimmed = sorted_values[trim_each_side : len(sorted_values) - trim_each_side]
    if not trimmed:
        return mean(values)
    return mean(trimmed)


def reduce_values(values: List[float], method: str, trim_ratio: float) -> float:
    if method == "mean":
        return mean(values)
    if method == "median":
        return median(values)
    if method == "trimmed_mean":
        return compute_trimmed_mean(values, trim_ratio=trim_ratio)
    raise ValueError(f"Unsupported reducer: {method}")


def _get_overlap_name_map(
    row: dict, shared_kernel_source: str, pure_phase: bool = False
) -> Dict[str, float]:
    if shared_kernel_source == "primary_stream":
        key = (
            "primary_stream_compute_pure_kernel_name_overlap_ms"
            if pure_phase
            else "primary_stream_compute_kernel_name_overlap_ms"
        )
    else:
        key = (
            "compute_pure_kernel_name_overlap_ms"
            if pure_phase
            else "compute_kernel_name_overlap_ms"
        )
    mapping = row.get(key, {})
    if not isinstance(mapping, dict):
        return {}
    result: Dict[str, float] = {}
    for raw_name, raw_value in mapping.items():
        if not isinstance(raw_name, str):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        result[raw_name] = value
    return result


def _metric_from_row(row: dict, compute_metric: str) -> float:
    if compute_metric == "overlap_sum":
        return float(row.get("compute_kernel_ms", 0.0))
    if compute_metric == "union":
        if "compute_kernel_union_ms" in row:
            return float(row.get("compute_kernel_union_ms", 0.0))
        return float(row.get("compute_kernel_ms", 0.0))
    if compute_metric == "primary_stream_union":
        if "compute_primary_stream_union_ms" in row:
            return float(row.get("compute_primary_stream_union_ms", 0.0))
        return float(row.get("compute_kernel_ms", 0.0))
    if compute_metric == "pure_union":
        if "compute_pure_union_ms" in row:
            return float(row.get("compute_pure_union_ms", 0.0))
        if "compute_kernel_union_ms" in row:
            return float(row.get("compute_kernel_union_ms", 0.0))
        return float(row.get("compute_kernel_ms", 0.0))
    if compute_metric == "pure_primary_union":
        if "compute_pure_primary_union_ms" in row:
            return float(row.get("compute_pure_primary_union_ms", 0.0))
        if "compute_primary_stream_union_ms" in row:
            return float(row.get("compute_primary_stream_union_ms", 0.0))
        return float(row.get("compute_kernel_ms", 0.0))
    raise ValueError(f"Unsupported compute metric: {compute_metric}")


def _shared_kernel_names(
    dist_samples: List[dict],
    scale_samples: List[dict],
    shared_kernel_source: str,
    pure_phase: bool = False,
) -> Set[str]:
    dist_names: Set[str] = set()
    scale_names: Set[str] = set()
    for sample in dist_samples:
        dist_names.update(
            _get_overlap_name_map(sample, shared_kernel_source, pure_phase=pure_phase).keys()
        )
    for sample in scale_samples:
        scale_names.update(
            _get_overlap_name_map(sample, shared_kernel_source, pure_phase=pure_phase).keys()
        )
    return dist_names & scale_names


def _shared_overlap_metric(
    row: dict, shared_names: Set[str], shared_kernel_source: str, pure_phase: bool = False
) -> float:
    mapping = _get_overlap_name_map(
        row, shared_kernel_source, pure_phase=pure_phase
    )
    if not mapping or not shared_names:
        return 0.0
    return sum(mapping.get(name, 0.0) for name in shared_names)


def _contamination_pct_from_row(row: dict) -> Optional[float]:
    if "contamination_pct" in row:
        try:
            return float(row.get("contamination_pct"))
        except (TypeError, ValueError):
            return None
    if "contamination_ms" not in row:
        return None
    try:
        contamination_ms = float(row.get("contamination_ms", 0.0))
    except (TypeError, ValueError):
        return None
    try:
        parent_compute_ms = float(row.get("compute_kernel_ms", 0.0))
    except (TypeError, ValueError):
        return None
    if parent_compute_ms <= 0:
        return 0.0
    return contamination_ms / parent_compute_ms * 100.0


def load_json_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("event_rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"event_rows must be list in {path}")
    return rows


def append_repeat_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_repeat_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def build_repeat_summary(records: List[dict], threshold_pct: float) -> Tuple[List[str], int]:
    grouped: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    for record in records:
        for row in record.get("rows", []):
            key = (row["rank"], row["op"], row["mg_state"])
            grouped[key].append(row["diff_pct"])
    lines: List[str] = []
    lines.append("| rank | op | mg_state | runs | median_diff_pct | status |")
    lines.append("|---:|---|---|---:|---:|---|")
    failed = 0
    for key in sorted(grouped.keys()):
        rank, op, mg_state = key
        med = median(grouped[key])
        status = "PASS" if med <= threshold_pct else "FAIL"
        if status == "FAIL":
            failed += 1
        lines.append(
            f"| {rank} | {op} | {mg_state} | {len(grouped[key])} | {med:.2f} | {status} |"
        )
    return lines, failed


def build_op_rank_median_summary(rows: List[dict], threshold_pct: float) -> Tuple[List[str], int]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[row["op"]].append(row["diff_pct"])
    lines: List[str] = []
    lines.append("| op | rank_samples | rank_median_diff_pct | rank_p75_diff_pct | status |")
    lines.append("|---|---:|---:|---:|---|")
    failed = 0
    for op in sorted(grouped.keys()):
        values = sorted(grouped[op])
        n = len(values)
        med = median(values)
        p75 = values[int((n - 1) * 0.75)]
        status = "PASS" if med <= threshold_pct else "FAIL"
        if status == "FAIL":
            failed += 1
        lines.append(f"| {op} | {n} | {med:.2f} | {p75:.2f} | {status} |")
    return lines, failed


def build_rank_total_summary(rows: List[dict], threshold_pct: float) -> Tuple[List[str], int]:
    grouped: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"dist_total": 0.0, "scale_total": 0.0, "rows": 0.0}
    )
    for row in rows:
        rank = int(row["rank"])
        grouped[rank]["dist_total"] += float(row["dist_compute_ms"])
        grouped[rank]["scale_total"] += float(row["scale_compute_ms"])
        grouped[rank]["rows"] += 1.0
    lines: List[str] = []
    lines.append("| rank | row_count | dist_total_compute_ms | scale_total_compute_ms | diff_pct | status |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    failed = 0
    for rank in sorted(grouped.keys()):
        dist_total = grouped[rank]["dist_total"]
        scale_total = grouped[rank]["scale_total"]
        if dist_total == 0:
            diff_pct = 0.0 if scale_total == 0 else 100.0
        else:
            diff_pct = abs(scale_total - dist_total) / dist_total * 100.0
        status = "PASS" if diff_pct <= threshold_pct else "FAIL"
        if status == "FAIL":
            failed += 1
        lines.append(
            f"| {rank} | {int(grouped[rank]['rows'])} | "
            f"{dist_total:.4f} | {scale_total:.4f} | {diff_pct:.2f} | {status} |"
        )
    return lines, failed


def build_repeat_op_rank_median_summary(
    records: List[dict], threshold_pct: float
) -> Tuple[List[str], int]:
    per_run_values: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in record.get("rows", []):
            grouped[row["op"]].append(row["diff_pct"])
        for op, values in grouped.items():
            per_run_values[op].append(median(values))
    lines: List[str] = []
    lines.append("| op | runs | median_of_run_rank_median_diff_pct | status |")
    lines.append("|---|---:|---:|---|")
    failed = 0
    for op in sorted(per_run_values.keys()):
        med = median(per_run_values[op])
        status = "PASS" if med <= threshold_pct else "FAIL"
        if status == "FAIL":
            failed += 1
        lines.append(f"| {op} | {len(per_run_values[op])} | {med:.2f} | {status} |")
    return lines, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare distributed/scaling compute-only kernel time from Nsight JSON."
    )
    parser.add_argument("--distributed-json", type=Path, required=True)
    parser.add_argument("--scaling-json", type=Path, required=True)
    parser.add_argument(
        "--ranks",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RANKS),
        help="Comma-separated rank list.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=",".join(DEFAULT_OPS),
        help="Comma-separated op list.",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Diff threshold percentage.",
    )
    parser.add_argument(
        "--no-align-by-state",
        action="store_true",
        help="Disable (op, mg_state) alignment; compare all states jointly.",
    )
    parser.add_argument(
        "--dist-reducer",
        type=str,
        default="trimmed_mean",
        choices=["mean", "median", "trimmed_mean"],
        help="Reducer for distributed samples.",
    )
    parser.add_argument(
        "--scale-reducer",
        type=str,
        default="median",
        choices=["mean", "median", "trimmed_mean"],
        help="Reducer for scaling samples.",
    )
    parser.add_argument(
        "--compute-metric",
        type=str,
        default="pure_primary_union",
        choices=[
            "overlap_sum",
            "union",
            "primary_stream_union",
            "pure_union",
            "pure_primary_union",
        ],
        help=(
            "Compute metric used before reducer. "
            "'union' avoids multi-stream overlap double counting; "
            "'primary_stream_union' focuses on dominant compute stream; "
            "'pure_*' uses phase-level compute windows when available."
        ),
    )
    parser.add_argument(
        "--kernel-scope",
        type=str,
        default="shared",
        choices=["all", "shared"],
        help=(
            "Kernel scope for compute metric. "
            "'shared' uses only kernel names present in both distributed and scaling samples "
            "to reduce distributed-only communication-side helper bias."
        ),
    )
    parser.add_argument(
        "--shared-kernel-source",
        type=str,
        default="primary_stream",
        choices=["all", "primary_stream"],
        help=(
            "Kernel-name source used when --kernel-scope=shared. "
            "'primary_stream' is recommended to align single-stream scaling semantics."
        ),
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.2,
        help="Trim ratio for trimmed_mean reducer.",
    )
    parser.add_argument(
        "--require-low-contamination-pct",
        type=float,
        default=None,
        help=(
            "Optional contamination threshold. "
            "When set, rows with dist/scale contamination pct above threshold are marked FAIL."
        ),
    )
    parser.add_argument("--repeat-report", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.distributed_json.exists():
        print(f"[ERROR] Missing distributed json: {args.distributed_json}")
        return 2
    if not args.scaling_json.exists():
        print(f"[ERROR] Missing scaling json: {args.scaling_json}")
        return 2
    try:
        ranks = parse_csv_ints(args.ranks)
    except ValueError as exc:
        print(f"[ERROR] Invalid --ranks: {exc}")
        return 2
    try:
        ops = parse_csv_strs(args.ops)
    except ValueError as exc:
        print(f"[ERROR] Invalid --ops: {exc}")
        return 2
    if args.trim_ratio < 0.0 or args.trim_ratio >= 0.5:
        print(f"[ERROR] Invalid --trim-ratio {args.trim_ratio}, expected 0 <= r < 0.5")
        return 2
    if args.require_low_contamination_pct is not None and args.require_low_contamination_pct < 0.0:
        print(
            "[ERROR] Invalid --require-low-contamination-pct "
            f"{args.require_low_contamination_pct}, expected >= 0."
        )
        return 2
    use_pure_phase_metrics = args.compute_metric in ("pure_union", "pure_primary_union")

    distributed_rows = load_json_rows(args.distributed_json)
    scaling_rows = load_json_rows(args.scaling_json)

    dist_grouped: Dict[Tuple[int, str, str], List[dict]] = defaultdict(list)
    scale_grouped: Dict[Tuple[int, str, str], List[dict]] = defaultdict(list)
    allowed_ranks = set(ranks)
    allowed_ops = set(ops)

    def _state_key(row: dict) -> str:
        return "ALL" if args.no_align_by_state else row.get("mg_state", "None")

    for row in distributed_rows:
        rank = row.get("rank")
        op = row.get("op")
        if rank not in allowed_ranks or op not in allowed_ops:
            continue
        dist_grouped[(rank, op, _state_key(row))].append(row)
    for row in scaling_rows:
        rank = row.get("rank")
        op = row.get("op")
        if rank not in allowed_ranks or op not in allowed_ops:
            continue
        scale_grouped[(rank, op, _state_key(row))].append(row)

    rows: List[str] = []
    row_records: List[dict] = []
    failed = 0
    rows.append(
        "| rank | op | mg_state | dist_samples | dist_compute_ms | dist_comm_ms | "
        "dist_contam_pct | scale_samples | scale_compute_ms | scale_comm_ms | "
        "scale_contam_pct | shared_kernels | diff_pct | status |"
    )
    rows.append(
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for rank in ranks:
        for op in ops:
            candidate_states = sorted(
                state
                for (grouped_rank, grouped_op, state) in (
                    set(dist_grouped.keys()) | set(scale_grouped.keys())
                )
                if grouped_op == op and grouped_rank == rank
            )
            if not candidate_states:
                candidate_states = ["ALL"] if args.no_align_by_state else ["None"]
            for state in candidate_states:
                key = (rank, op, state)
                dist_samples = dist_grouped.get(key, [])
                scale_samples = scale_grouped.get(key, [])
                if not dist_samples or not scale_samples:
                    rows.append(
                        f"| {rank} | {op} | {state} | "
                        f"{len(dist_samples)} | N/A | N/A | N/A | "
                        f"{len(scale_samples)} | N/A | N/A | N/A | N/A | N/A | FAIL (missing) |"
                    )
                    failed += 1
                    continue
                shared_names: Set[str] = set()
                if args.kernel_scope == "shared":
                    if not any(
                        _get_overlap_name_map(
                            sample,
                            args.shared_kernel_source,
                            pure_phase=use_pure_phase_metrics,
                        )
                        for sample in (dist_samples + scale_samples)
                    ):
                        print(
                            "[ERROR] --kernel-scope=shared requires kernel-name overlap maps in JSON. "
                            "Please regenerate JSON with updated "
                            "tests/performance/analyze_nsys_cmd_kernel_breakdown.py."
                        )
                        return 2
                    shared_names = _shared_kernel_names(
                        dist_samples,
                        scale_samples,
                        args.shared_kernel_source,
                        pure_phase=use_pure_phase_metrics,
                    )
                    dist_compute_values = [
                        _shared_overlap_metric(
                            row=sample,
                            shared_names=shared_names,
                            shared_kernel_source=args.shared_kernel_source,
                            pure_phase=use_pure_phase_metrics,
                        )
                        for sample in dist_samples
                    ]
                    scale_compute_values = [
                        _shared_overlap_metric(
                            row=sample,
                            shared_names=shared_names,
                            shared_kernel_source=args.shared_kernel_source,
                            pure_phase=use_pure_phase_metrics,
                        )
                        for sample in scale_samples
                    ]
                else:
                    dist_compute_values = [
                        _metric_from_row(sample, args.compute_metric)
                        for sample in dist_samples
                    ]
                    scale_compute_values = [
                        _metric_from_row(sample, args.compute_metric)
                        for sample in scale_samples
                    ]

                dist_compute = reduce_values(
                    dist_compute_values, method=args.dist_reducer, trim_ratio=args.trim_ratio
                )
                scale_compute = reduce_values(
                    scale_compute_values, method=args.scale_reducer, trim_ratio=args.trim_ratio
                )
                if args.compute_metric == "union":
                    dist_comm_values = [
                        float(sample.get("comm_kernel_union_ms", sample.get("comm_kernel_ms", 0.0)))
                        for sample in dist_samples
                    ]
                    scale_comm_values = [
                        float(sample.get("comm_kernel_union_ms", sample.get("comm_kernel_ms", 0.0)))
                        for sample in scale_samples
                    ]
                else:
                    dist_comm_values = [
                        float(sample.get("comm_kernel_ms", 0.0)) for sample in dist_samples
                    ]
                    scale_comm_values = [
                        float(sample.get("comm_kernel_ms", 0.0)) for sample in scale_samples
                    ]
                dist_comm = reduce_values(
                    dist_comm_values, method=args.dist_reducer, trim_ratio=args.trim_ratio
                )
                scale_comm = reduce_values(
                    scale_comm_values, method=args.scale_reducer, trim_ratio=args.trim_ratio
                )
                dist_contam_values = [
                    value
                    for value in (
                        _contamination_pct_from_row(sample) for sample in dist_samples
                    )
                    if value is not None
                ]
                scale_contam_values = [
                    value
                    for value in (
                        _contamination_pct_from_row(sample) for sample in scale_samples
                    )
                    if value is not None
                ]
                dist_contam_pct = (
                    reduce_values(
                        dist_contam_values,
                        method=args.dist_reducer,
                        trim_ratio=args.trim_ratio,
                    )
                    if dist_contam_values
                    else None
                )
                scale_contam_pct = (
                    reduce_values(
                        scale_contam_values,
                        method=args.scale_reducer,
                        trim_ratio=args.trim_ratio,
                    )
                    if scale_contam_values
                    else None
                )
                if dist_compute == 0:
                    diff_pct = 0.0 if scale_compute == 0 else 100.0
                else:
                    diff_pct = abs(scale_compute - dist_compute) / dist_compute * 100.0
                status = "PASS" if diff_pct <= args.threshold_pct else "FAIL"
                if args.require_low_contamination_pct is not None:
                    if dist_contam_pct is None or scale_contam_pct is None:
                        print(
                            "[ERROR] --require-low-contamination-pct requires contamination "
                            "fields in both distributed and scaling JSON rows."
                        )
                        return 2
                    if (
                        dist_contam_pct > args.require_low_contamination_pct
                        or scale_contam_pct > args.require_low_contamination_pct
                    ):
                        status = "FAIL (contamination)"
                if status.startswith("FAIL"):
                    failed += 1
                dist_contam_text = (
                    "N/A" if dist_contam_pct is None else f"{dist_contam_pct:.2f}"
                )
                scale_contam_text = (
                    "N/A" if scale_contam_pct is None else f"{scale_contam_pct:.2f}"
                )
                rows.append(
                    f"| {rank} | {op} | {state} | {len(dist_samples)} | {dist_compute:.4f} | "
                    f"{dist_comm:.4f} | {dist_contam_text} | "
                    f"{len(scale_samples)} | {scale_compute:.4f} | {scale_comm:.4f} | "
                    f"{scale_contam_text} | {len(shared_names)} | {diff_pct:.2f} | {status} |"
                )
                row_records.append(
                    {
                        "rank": rank,
                        "op": op,
                        "mg_state": state,
                        "dist_samples": len(dist_samples),
                        "dist_compute_ms": dist_compute,
                        "dist_comm_ms": dist_comm,
                        "dist_contamination_pct": dist_contam_pct,
                        "scale_samples": len(scale_samples),
                        "scale_compute_ms": scale_compute,
                        "scale_comm_ms": scale_comm,
                        "scale_contamination_pct": scale_contam_pct,
                        "shared_kernel_count": len(shared_names),
                        "diff_pct": diff_pct,
                        "status": status,
                    }
                )

    report_lines: List[str] = []
    report_lines.append(f"distributed_json={args.distributed_json}")
    report_lines.append(f"scaling_json={args.scaling_json}")
    report_lines.append(f"ranks={','.join(str(x) for x in ranks)}")
    report_lines.append(f"ops={','.join(ops)}")
    report_lines.append(f"align_by_state={not args.no_align_by_state}")
    report_lines.append(f"dist_reducer={args.dist_reducer}")
    report_lines.append(f"scale_reducer={args.scale_reducer}")
    report_lines.append(f"compute_metric={args.compute_metric}")
    report_lines.append(f"kernel_scope={args.kernel_scope}")
    report_lines.append(f"shared_kernel_source={args.shared_kernel_source}")
    report_lines.append(f"trim_ratio={args.trim_ratio:.4f}")
    report_lines.append(
        "require_low_contamination_pct="
        + (
            "None"
            if args.require_low_contamination_pct is None
            else f"{args.require_low_contamination_pct:.2f}"
        )
    )
    report_lines.append(f"threshold_pct={args.threshold_pct:.2f}")
    report_lines.extend(rows)
    report_lines.append("")
    report_lines.append("op_rank_median_aux_summary(recommended_for_paper):")
    op_summary_lines, _ = build_op_rank_median_summary(row_records, args.threshold_pct)
    report_lines.extend(op_summary_lines)
    report_lines.append("")
    report_lines.append("rank_total_comp_summary(new, sum of all selected fwd/bwd/optimizer rows):")
    rank_total_lines, rank_total_failed = build_rank_total_summary(
        row_records, args.threshold_pct
    )
    report_lines.extend(rank_total_lines)
    if rank_total_failed > 0:
        failed += rank_total_failed

    if args.repeat_report is not None:
        series_key = (
            f"dist={args.distributed_json}|scale={args.scaling_json}|"
            f"align={not args.no_align_by_state}|ranks={','.join(str(x) for x in ranks)}|"
            f"ops={','.join(ops)}|dist_reducer={args.dist_reducer}|"
            f"scale_reducer={args.scale_reducer}|metric={args.compute_metric}|"
            f"scope={args.kernel_scope}|shared_source={args.shared_kernel_source}|"
            f"trim={args.trim_ratio:.6f}|"
            f"contam={args.require_low_contamination_pct}"
        )
        run_record = {
            "series_key": series_key,
            "threshold_pct": args.threshold_pct,
            "run_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": row_records,
        }
        append_repeat_record(args.repeat_report, run_record)
        repeat_records = [
            x for x in load_repeat_records(args.repeat_report) if x.get("series_key") == series_key
        ]
        repeat_lines, repeat_failed = build_repeat_summary(repeat_records, args.threshold_pct)
        repeat_op_lines, _ = build_repeat_op_rank_median_summary(
            repeat_records, args.threshold_pct
        )
        report_lines.append("")
        report_lines.append("repeat_median_summary:")
        report_lines.extend(repeat_lines)
        report_lines.append("")
        report_lines.append("repeat_median_summary(op_rank_median_aux):")
        report_lines.extend(repeat_op_lines)
        print(f"[INFO] Repeat records in current series: {len(repeat_records)}")
        print(f"[INFO] Repeat report updated: {args.repeat_report}")
        if repeat_failed > 0:
            failed += repeat_failed

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(report_text + "\n")
        print(f"[INFO] Report written to {args.report_path}")

    if failed > 0:
        print(f"[RESULT] FAIL ({failed} checks above threshold or missing).")
        return 1
    print("[RESULT] PASS (all checks within threshold).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
