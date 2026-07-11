#!/usr/bin/env python3
"""Compare slowdown off/on simulation summaries against a real hardware reference trace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = PROJECT_ROOT / 'megatron-sim-engine'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from tools.data_prep.common.megatron_trace_utils import (
    build_ddp_comm_alignment_key,
    parse_trace_line,
)


def _find_rank_trace(trace_dir: Path, wrank_id: int) -> Path:
    candidates = sorted(trace_dir.glob(f'*rank{wrank_id}_*.txt'))
    if not candidates:
        candidates = sorted(trace_dir.glob(f'*{wrank_id}*.txt'))
    if not candidates:
        raise ValueError(f'No trace file found for wrank_id={wrank_id} under {trace_dir}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_pct_err(predicted: float, actual: float) -> Optional[float]:
    actual = float(actual)
    if math.isclose(actual, 0.0, abs_tol=1e-12):
        return None
    return abs(float(predicted) - actual) / abs(actual) * 100.0


def _mean_abs_error(pairs: Iterable[tuple[float, float]]) -> Optional[float]:
    values = [abs(float(a) - float(b)) for a, b in pairs]
    if not values:
        return None
    return mean(values)


def _collect_reference_trace_groups(trace_path: Path) -> dict:
    backward_by_cmd_uid: Dict[str, Dict[str, object]] = {}
    ddp_by_trigger_cmd_uid: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    finalize_wait_by_cmd_uid: Dict[str, Dict[str, object]] = {}
    ordered_backward_cmd_uids: List[str] = []

    for order_index, raw_line in enumerate(trace_path.read_text(encoding='utf-8').splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        _, event_name, fields = parse_trace_line(line)
        if event_name == 'backward_step':
            cmd_uid = fields.get('cmd_uid')
            if not isinstance(cmd_uid, str) or not cmd_uid:
                raise ValueError(f'backward_step in {trace_path} is missing cmd_uid')
            backward_by_cmd_uid[cmd_uid] = {
                'duration_ms': float(fields['duration']),
                'stage_id': int(fields['stage_id']),
                'mg_state': str(fields['mg_state']),
                'batch_id': int(fields['batch_id']),
                'timestamp_ms': float(fields['timestamp']),
                'order_index': order_index,
            }
            ordered_backward_cmd_uids.append(cmd_uid)
            continue

        if event_name == 'ddp_grad_comm':
            trigger_cmd_uid = fields.get('trigger_cmd_uid')
            if not isinstance(trigger_cmd_uid, str) or not trigger_cmd_uid:
                raise ValueError(f'ddp_grad_comm in {trace_path} is missing trigger_cmd_uid')
            alignment_key = build_ddp_comm_alignment_key(fields)
            if alignment_key in ddp_by_trigger_cmd_uid[trigger_cmd_uid]:
                raise ValueError(
                    f'Duplicate DDP comm alignment key {alignment_key!r} for trigger_cmd_uid={trigger_cmd_uid}'
                )
            launch_timestamp_ms = fields.get('launch_timestamp_ms')
            if launch_timestamp_ms is None:
                raise ValueError(
                    f'ddp_grad_comm {alignment_key} is missing launch_timestamp_ms in {trace_path}'
                )
            completion_timestamp_ms = fields.get('completion_observed_timestamp_ms')
            if completion_timestamp_ms is None:
                completion_timestamp_ms = fields.get('timestamp')
            if completion_timestamp_ms is None:
                raise ValueError(
                    f'ddp_grad_comm {alignment_key} is missing completion timestamp in {trace_path}'
                )
            ddp_by_trigger_cmd_uid[trigger_cmd_uid][alignment_key] = {
                'launch_time': float(launch_timestamp_ms),
                'finish_time': float(completion_timestamp_ms),
                'comm_uid': fields.get('comm_uid'),
                'wait_cmd_uid': fields.get('wait_cmd_uid'),
            }
            continue

        if event_name == 'dp_allreduce':
            cmd_uid = fields.get('cmd_uid')
            if not isinstance(cmd_uid, str) or not cmd_uid:
                raise ValueError(f'dp_allreduce in {trace_path} is missing cmd_uid')
            op_semantics = fields.get('op_semantics')
            if op_semantics not in {'wait_flush_only', 'metadata_placeholder'}:
                continue
            finalize_wait_by_cmd_uid[cmd_uid] = {
                'duration_ms': float(fields['duration']),
                'timestamp_ms': float(fields['timestamp']),
            }

    return {
        'backward_by_cmd_uid': backward_by_cmd_uid,
        'ddp_by_trigger_cmd_uid': ddp_by_trigger_cmd_uid,
        'finalize_wait_by_cmd_uid': finalize_wait_by_cmd_uid,
        'ordered_backward_cmd_uids': ordered_backward_cmd_uids,
    }


def _select_reference_backward_cmd_uid(
    *,
    target_backward: Mapping[str, object],
    backward_by_cmd_uid: Mapping[str, Mapping[str, object]],
    ddp_by_trigger_cmd_uid: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> str:
    target_stage_id = int(target_backward['stage_id'])
    target_mg_state = str(target_backward['mg_state'])
    target_keys = set(target_backward.get('ddp_comm_alignment_keys', []))
    target_comm_count = int(target_backward.get('ddp_comm_count', len(target_keys)))

    best_cmd_uid = None
    best_score = None
    for cmd_uid, backward in backward_by_cmd_uid.items():
        if int(backward['stage_id']) != target_stage_id:
            continue
        if str(backward['mg_state']) != target_mg_state:
            continue
        reference_keys = set(ddp_by_trigger_cmd_uid.get(cmd_uid, {}))
        shared_key_count = len(reference_keys & target_keys)
        score = (
            shared_key_count,
            int(len(reference_keys) == target_comm_count),
            -abs(len(reference_keys) - target_comm_count),
            -int(backward['order_index']),
        )
        if best_score is None or score > best_score:
            best_cmd_uid = cmd_uid
            best_score = score

    if best_cmd_uid is None:
        raise ValueError(
            f'No reference backward_step matches stage_id={target_stage_id}, mg_state={target_mg_state!r}'
        )
    if target_keys and best_score is not None and best_score[0] <= 0:
        raise ValueError(
            'Reference trace does not share any DDP comm alignment keys with the simulator target backward window.'
        )
    return best_cmd_uid


def load_reference_trace_summary(
    trace_dir: Path,
    wrank_id: int,
    target_backward: Mapping[str, object],
) -> dict:
    trace_path = _find_rank_trace(trace_dir, wrank_id)
    grouped = _collect_reference_trace_groups(trace_path)
    backward_by_cmd_uid = grouped['backward_by_cmd_uid']
    ddp_by_trigger_cmd_uid = grouped['ddp_by_trigger_cmd_uid']
    finalize_wait_by_cmd_uid = grouped['finalize_wait_by_cmd_uid']

    selected_backward_cmd_uid = _select_reference_backward_cmd_uid(
        target_backward=target_backward,
        backward_by_cmd_uid=backward_by_cmd_uid,
        ddp_by_trigger_cmd_uid=ddp_by_trigger_cmd_uid,
    )
    backward = backward_by_cmd_uid[selected_backward_cmd_uid]
    ddp_comm_ops = dict(ddp_by_trigger_cmd_uid.get(selected_backward_cmd_uid, {}))

    finalize_wait_duration_ms = None
    for comm in ddp_comm_ops.values():
        wait_cmd_uid = comm.get('wait_cmd_uid')
        if wait_cmd_uid in {None, 'None'}:
            continue
        finalize_entry = finalize_wait_by_cmd_uid.get(str(wait_cmd_uid))
        if finalize_entry is not None:
            finalize_wait_duration_ms = float(finalize_entry['duration_ms'])
            break

    return {
        'trace_path': str(trace_path),
        'wrank_id': int(wrank_id),
        'selected_backward_cmd_uid': selected_backward_cmd_uid,
        'backward_duration_ms': float(backward['duration_ms']),
        'backward_start_ms': float(backward['timestamp_ms']) - float(backward['duration_ms']),
        'finalize_wait_duration_ms': finalize_wait_duration_ms,
        'ddp_comm_ops': ddp_comm_ops,
    }


def build_error_summary(*, sim_summary: Mapping[str, object], reference_summary: Mapping[str, object]) -> dict:
    hardware_backward_duration_ms = float(reference_summary['backward_duration_ms'])
    backward_off = float(sim_summary['backward_duration_ms_off'])
    backward_on = float(sim_summary['backward_duration_ms_on'])

    reference_ddp = reference_summary['ddp_comm_ops']
    off_ddp = sim_summary['slowdown_off']['ddp_comm_ops']
    on_ddp = sim_summary['slowdown_on']['ddp_comm_ops']
    shared_comm_keys = sorted(set(reference_ddp) & set(off_ddp) & set(on_ddp))

    if not shared_comm_keys:
        raise ValueError(
            'Reference trace does not share any DDP comm alignment keys with the simulator target backward window.'
        )

    reference_backward_start_ms = float(reference_summary['backward_start_ms'])
    off_backward_start_ms = float(sim_summary['slowdown_off']['target_backward']['start_time_ms'])
    on_backward_start_ms = float(sim_summary['slowdown_on']['target_backward']['start_time_ms'])

    off_launch_pairs = [
        (
            float(off_ddp[key]['join_time']) - off_backward_start_ms,
            float(reference_ddp[key]['launch_time']) - reference_backward_start_ms,
        )
        for key in shared_comm_keys
    ]
    on_launch_pairs = [
        (
            float(on_ddp[key]['join_time']) - on_backward_start_ms,
            float(reference_ddp[key]['launch_time']) - reference_backward_start_ms,
        )
        for key in shared_comm_keys
    ]
    off_finish_pairs = [
        (
            float(off_ddp[key]['finish_time']) - off_backward_start_ms,
            float(reference_ddp[key]['finish_time']) - reference_backward_start_ms,
        )
        for key in shared_comm_keys
    ]
    on_finish_pairs = [
        (
            float(on_ddp[key]['finish_time']) - on_backward_start_ms,
            float(reference_ddp[key]['finish_time']) - reference_backward_start_ms,
        )
        for key in shared_comm_keys
    ]

    hardware_finalize_wait_ms = reference_summary.get('finalize_wait_duration_ms')
    off_finalize_wait_ms = sim_summary['slowdown_off'].get('finalize_wait_duration_ms')
    on_finalize_wait_ms = sim_summary['slowdown_on'].get('finalize_wait_duration_ms')

    return {
        'wrank_id': int(sim_summary['wrank_id']),
        'reference_trace_path': reference_summary['trace_path'],
        'reference_backward_cmd_uid': reference_summary['selected_backward_cmd_uid'],
        'shared_comm_uids_count': len(shared_comm_keys),
        'shared_comm_alignment_keys': shared_comm_keys,
        'hardware_backward_duration_ms': hardware_backward_duration_ms,
        'sim_backward_duration_ms_off': backward_off,
        'sim_backward_duration_ms_on': backward_on,
        'backward_abs_err_ms_off': abs(backward_off - hardware_backward_duration_ms),
        'backward_abs_err_ms_on': abs(backward_on - hardware_backward_duration_ms),
        'backward_pct_err_off': _safe_pct_err(backward_off, hardware_backward_duration_ms),
        'backward_pct_err_on': _safe_pct_err(backward_on, hardware_backward_duration_ms),
        'ddp_launch_mae_ms_off': _mean_abs_error(off_launch_pairs),
        'ddp_launch_mae_ms_on': _mean_abs_error(on_launch_pairs),
        'ddp_finish_mae_ms_off': _mean_abs_error(off_finish_pairs),
        'ddp_finish_mae_ms_on': _mean_abs_error(on_finish_pairs),
        'hardware_finalize_wait_duration_ms': hardware_finalize_wait_ms,
        'sim_finalize_wait_duration_ms_off': off_finalize_wait_ms,
        'sim_finalize_wait_duration_ms_on': on_finalize_wait_ms,
        'finalize_wait_abs_err_ms_off': None if hardware_finalize_wait_ms is None or off_finalize_wait_ms is None else abs(float(off_finalize_wait_ms) - float(hardware_finalize_wait_ms)),
        'finalize_wait_abs_err_ms_on': None if hardware_finalize_wait_ms is None or on_finalize_wait_ms is None else abs(float(on_finalize_wait_ms) - float(hardware_finalize_wait_ms)),
        'finalize_wait_pct_err_off': None if hardware_finalize_wait_ms is None or off_finalize_wait_ms is None else _safe_pct_err(float(off_finalize_wait_ms), float(hardware_finalize_wait_ms)),
        'finalize_wait_pct_err_on': None if hardware_finalize_wait_ms is None or on_finalize_wait_ms is None else _safe_pct_err(float(on_finalize_wait_ms), float(hardware_finalize_wait_ms)),
    }


def _render_md(rows: List[Mapping[str, object]], reference_nsys_sqlite: Optional[str]) -> str:
    lines = [
        '## DDP Slowdown vs Hardware Reference',
        '',
    ]
    if reference_nsys_sqlite:
        lines.append(f'- Reference nsys sqlite: `{reference_nsys_sqlite}`')
        lines.append('')
    lines.append('| wrank | backward hw | backward off | backward on | off abs err | on abs err | launch MAE off | launch MAE on | finalize err off | finalize err on |')
    lines.append('|-------|-------------|--------------|-------------|-------------|------------|----------------|---------------|------------------|-----------------|')
    for row in rows:
        lines.append(
            '| {wrank_id} | {hardware_backward_duration_ms:.6f} | {sim_backward_duration_ms_off:.6f} | {sim_backward_duration_ms_on:.6f} | {backward_abs_err_ms_off:.6f} | {backward_abs_err_ms_on:.6f} | {ddp_launch_mae_ms_off} | {ddp_launch_mae_ms_on} | {finalize_wait_abs_err_ms_off} | {finalize_wait_abs_err_ms_on} |'.format(
                wrank_id=row['wrank_id'],
                hardware_backward_duration_ms=row['hardware_backward_duration_ms'],
                sim_backward_duration_ms_off=row['sim_backward_duration_ms_off'],
                sim_backward_duration_ms_on=row['sim_backward_duration_ms_on'],
                backward_abs_err_ms_off=row['backward_abs_err_ms_off'],
                backward_abs_err_ms_on=row['backward_abs_err_ms_on'],
                ddp_launch_mae_ms_off='None' if row['ddp_launch_mae_ms_off'] is None else f"{row['ddp_launch_mae_ms_off']:.6f}",
                ddp_launch_mae_ms_on='None' if row['ddp_launch_mae_ms_on'] is None else f"{row['ddp_launch_mae_ms_on']:.6f}",
                finalize_wait_abs_err_ms_off='None' if row['finalize_wait_abs_err_ms_off'] is None else f"{row['finalize_wait_abs_err_ms_off']:.6f}",
                finalize_wait_abs_err_ms_on='None' if row['finalize_wait_abs_err_ms_on'] is None else f"{row['finalize_wait_abs_err_ms_on']:.6f}",
            )
        )
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare slowdown off/on summaries against a hardware reference trace.')
    parser.add_argument('--reference-trace-dir', required=True)
    parser.add_argument('--reference-nsys-sqlite', default=None)
    parser.add_argument('--sim-json', action='append', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args = parser.parse_args()

    rows = []
    for sim_json_path in args.sim_json:
        sim_summary = json.loads(Path(sim_json_path).read_text(encoding='utf-8'))
        wrank_id = int(sim_summary['wrank_id'])
        reference_summary = load_reference_trace_summary(
            Path(args.reference_trace_dir),
            wrank_id,
            target_backward=sim_summary['target_backward'],
        )
        rows.append(build_error_summary(sim_summary=sim_summary, reference_summary=reference_summary))
    rows = sorted(rows, key=lambda row: row['wrank_id'])

    payload = {
        'reference_trace_dir': args.reference_trace_dir,
        'reference_nsys_sqlite': args.reference_nsys_sqlite,
        'rows': rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    Path(args.output_md).write_text(_render_md(rows, args.reference_nsys_sqlite), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
