#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = PROJECT_ROOT / 'megatron-sim-engine'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import simu_main  # type: ignore
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager  # type: ignore
from src.core.static_graphs.rank_manager import RankManager  # type: ignore
from src.core.simu_engine import Operation, SimulatorEngine, SubOperation  # type: ignore
from tools.data_prep.common.megatron_trace_utils import build_ddp_comm_alignment_key


def _build_engine(cli_args: list[str]) -> SimulatorEngine:
    parsed = simu_main._parse_args(cli_args)
    config = simu_main._fail_fast_validate(parsed)
    running_mode = simu_main.RUN_MODE_MAP[config.mode]

    manager = ParallelGroupManager(
        local_size=config.local_size,
        world_size=config.world_size,
        pp_size=config.pp_size,
        tp_size=config.tp_size,
        exp_size=config.exp_size,
    )
    mpu_info = manager.get_mpu_info()
    all_groups = manager.get_all_groups()
    rank_manager = RankManager(
        mpu_info=mpu_info,
        gpus_per_node=config.local_size,
        all_groups=all_groups,
    )
    rank_instances = rank_manager.get_rank_zoos()

    simulator_engine = SimulatorEngine(
        trace_filepath=config.trace_dir,
        framwork=config.framework,
        strategy=config.strategy,
        running_mode=running_mode,
        torchgraph_filepath=config.database_dir,
        stages_scheduling_filepath=config.schedule_dir,
        cc_backend_name=config.cc_backend,
        cc_backend_options=config.cc_backend_options,
    )
    simulator_engine.simulator_config.slowdown = config.slowdown
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator_engine.validate_global_placement_requirements()
    simulator_engine.start_running()
    return simulator_engine


def _collect_rank_ops(simulator_engine: SimulatorEngine, wrank_id: int) -> list[Operation]:
    timeline = simulator_engine.timeline_manager.stages_timeline_process_dict[wrank_id]
    merged = list(getattr(timeline, 'final_merge_timeline', []))
    if merged:
        return [op for op in merged if isinstance(op, Operation) and not isinstance(op, SubOperation)]
    return [op for op in getattr(timeline, 'comp_timeline', []) if isinstance(op, Operation)] + [
        op for op in getattr(timeline, 'comm_timeline', []) if isinstance(op, Operation)
    ]


def _select_target_backward_op(
    simulator_engine: SimulatorEngine,
    wrank_id: int,
    preferred_cmd_uid: Optional[str] = None,
) -> Operation:
    ops = _collect_rank_ops(simulator_engine, wrank_id)
    backward_ops = [op for op in ops if op.name == 'backward_step']
    if not backward_ops:
        raise RuntimeError(f'No backward_step found for wrank {wrank_id}')

    if preferred_cmd_uid is not None:
        for op in backward_ops:
            if getattr(op, 'cmd_uid', None) == preferred_cmd_uid:
                return op
        raise RuntimeError(
            f'No backward_step with cmd_uid={preferred_cmd_uid!r} found for wrank {wrank_id}'
        )

    processed_cmd_uids = set(
        getattr(simulator_engine.timeline_manager, 'slowdown_processed_backward_cmd_uids', set())
    )
    for op in backward_ops:
        if getattr(op, 'cmd_uid', None) in processed_cmd_uids:
            return op
    return backward_ops[0]


def _extract_ddp_comm_by_alignment_key(
    simulator_engine: SimulatorEngine,
    wrank_id: int,
    trigger_cmd_uid: str,
) -> dict[str, dict[str, object]]:
    ops = _collect_rank_ops(simulator_engine, wrank_id)
    result: dict[str, dict[str, object]] = {}
    for op in ops:
        trace_metadata = dict(getattr(op, 'trace_metadata', {}) or {})
        if trace_metadata.get('trace_event_type') != 'ddp_grad_comm':
            continue
        if trace_metadata.get('trigger_cmd_uid') != trigger_cmd_uid:
            continue
        alignment_key = build_ddp_comm_alignment_key(trace_metadata)
        if alignment_key in result:
            raise RuntimeError(
                f'Duplicate DDP comm alignment key {alignment_key!r} found for wrank {wrank_id}'
            )
        result[alignment_key] = {
            'join_time': float(getattr(op, 'join_time', 0.0)),
            'duration': float(getattr(op, 'duration', 0.0)),
            'finish_time': float(getattr(op, 'finish_time', 0.0)),
            'comm_uid': getattr(op, 'cmd_uid', None),
            'wait_cmd_uid': trace_metadata.get('wait_cmd_uid'),
            'bucket_id': trace_metadata.get('bucket_id'),
            'buffer_id': trace_metadata.get('buffer_id'),
        }
    return result


def _extract_target_finalize_wait_duration(
    simulator_engine: SimulatorEngine,
    wrank_id: int,
    wait_cmd_uids: set[str],
) -> float | None:
    if not wait_cmd_uids:
        return None
    ops = _collect_rank_ops(simulator_engine, wrank_id)
    for op in ops:
        if op.name != 'dp_allreduce':
            continue
        if getattr(op, 'op_semantics', None) not in {'wait_flush_only', 'metadata_placeholder'}:
            continue
        if getattr(op, 'cmd_uid', None) in wait_cmd_uids:
            return float(op.duration)
    return None


def _summarize_engine(
    simulator_engine: SimulatorEngine,
    wrank_id: int,
    target_backward_cmd_uid: Optional[str] = None,
) -> dict:
    backward_op = _select_target_backward_op(
        simulator_engine,
        wrank_id,
        preferred_cmd_uid=target_backward_cmd_uid,
    )
    backward_cmd_uid = getattr(backward_op, 'cmd_uid', None)
    if not backward_cmd_uid:
        raise RuntimeError(f'Target backward_step for wrank {wrank_id} is missing cmd_uid')

    ddp_summary = _extract_ddp_comm_by_alignment_key(
        simulator_engine,
        wrank_id,
        trigger_cmd_uid=backward_cmd_uid,
    )
    wait_cmd_uids = {
        str(comm['wait_cmd_uid'])
        for comm in ddp_summary.values()
        if comm.get('wait_cmd_uid') not in {None, 'None'}
    }
    return {
        'target_backward': {
            'cmd_uid': backward_cmd_uid,
            'stage_id': int(getattr(backward_op, 'stage_id')),
            'mg_state': str(getattr(backward_op, 'mg_state')),
            'batch_id': int(getattr(backward_op, 'batch_id')),
            'start_time_ms': float(getattr(backward_op, 'join_time', 0.0)),
            'finish_time_ms': float(getattr(backward_op, 'finish_time', 0.0)),
            'duration_ms': float(getattr(backward_op, 'duration')),
            'slowdown_kernel_schedules': list(
                (getattr(backward_op, 'trace_metadata', {}) or {}).get(
                    'slowdown_kernel_schedules', []
                )
            ),
            'ddp_comm_alignment_keys': sorted(ddp_summary),
            'ddp_comm_count': len(ddp_summary),
        },
        'backward_duration_ms': float(getattr(backward_op, 'duration')),
        'finalize_wait_duration_ms': _extract_target_finalize_wait_duration(
            simulator_engine,
            wrank_id,
            wait_cmd_uids,
        ),
        'ddp_comm_ops': ddp_summary,
        'slowdown_runtime_comm_schedules_by_uid': dict(
            getattr(simulator_engine.timeline_manager, 'slowdown_runtime_comm_schedules_by_uid', {})
        ),
        'slowdown_processed_backward_cmd_uids': sorted(
            getattr(simulator_engine.timeline_manager, 'slowdown_processed_backward_cmd_uids', set())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare sim-engine slowdown off/on on a real trace-backed smoke case.')
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--database-dir', required=True)
    parser.add_argument('--schedule-dir', required=True)
    parser.add_argument('--slowdown-assets-dir', required=True)
    parser.add_argument('--slowdown-model-path', required=True)
    parser.add_argument('--slowdown-scaler-path', default=None)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--local-size', type=int, required=True)
    parser.add_argument('--pp-size', type=int, default=1)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--exp-size', type=int, default=1)
    parser.add_argument('--strategy', default='1F1B-none_interleaved')
    parser.add_argument('--wrank-id', type=int, default=0)
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    base_args = [
        '--framework', 'megatron-lm',
        '--mode', 'simulate',
        '--trace-dir', args.trace_dir,
        '--schedule-dir', args.schedule_dir,
        '--database-dir', args.database_dir,
        '--world-size', str(args.world_size),
        '--pp-size', str(args.pp_size),
        '--tp-size', str(args.tp_size),
        '--exp-size', str(args.exp_size),
        '--local-size', str(args.local_size),
        '--strategy', args.strategy,
        '--no-visualize',
    ]

    engine_on = _build_engine(
        base_args
        + [
            '--enable-slowdown',
            '--slowdown-assets-dir', args.slowdown_assets_dir,
            '--slowdown-model-path', args.slowdown_model_path,
        ]
        + ([] if not args.slowdown_scaler_path else ['--slowdown-scaler-path', args.slowdown_scaler_path])
    )
    on_summary = _summarize_engine(engine_on, args.wrank_id)

    if not on_summary['slowdown_processed_backward_cmd_uids']:
        raise RuntimeError('Slowdown-enabled simulator did not process any backward cmd_uid.')

    target_backward_cmd_uid = on_summary['target_backward']['cmd_uid']
    engine_off = _build_engine(base_args)
    off_summary = _summarize_engine(engine_off, args.wrank_id, target_backward_cmd_uid=target_backward_cmd_uid)

    backward_off = float(off_summary['backward_duration_ms'])
    backward_on = float(on_summary['backward_duration_ms'])
    if backward_on < backward_off:
        raise RuntimeError(
            f'Slowdown-enabled backward duration unexpectedly shrank: off={backward_off}, on={backward_on}'
        )

    shared_comm_keys = sorted(set(off_summary['ddp_comm_ops']).intersection(on_summary['ddp_comm_ops']))
    if not shared_comm_keys:
        raise RuntimeError('No shared DDP comm alignment key was found between slowdown off/on summaries.')

    delayed_comm_keys = []
    for comm_key in shared_comm_keys:
        off_join = float(off_summary['ddp_comm_ops'][comm_key]['join_time'])
        on_join = float(on_summary['ddp_comm_ops'][comm_key]['join_time'])
        if on_join > off_join:
            delayed_comm_keys.append(comm_key)

    summary = {
        'trace_dir': args.trace_dir,
        'database_dir': args.database_dir,
        'schedule_dir': args.schedule_dir,
        'slowdown_assets_dir': args.slowdown_assets_dir,
        'slowdown_model_path': args.slowdown_model_path,
        'slowdown_scaler_path': args.slowdown_scaler_path,
        'wrank_id': args.wrank_id,
        'target_backward': on_summary['target_backward'],
        'backward_duration_ms_off': backward_off,
        'backward_duration_ms_on': backward_on,
        'backward_duration_delta_ms': backward_on - backward_off,
        'shared_comm_uids': shared_comm_keys,
        'shared_comm_alignment_keys': shared_comm_keys,
        'delayed_comm_uids': delayed_comm_keys,
        'delayed_comm_alignment_keys': delayed_comm_keys,
        'slowdown_off': off_summary,
        'slowdown_on': on_summary,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(
        json.dumps(
            {
                'target_backward': on_summary['target_backward'],
                'backward_duration_ms_off': backward_off,
                'backward_duration_ms_on': backward_on,
                'backward_duration_delta_ms': backward_on - backward_off,
                'shared_comm_alignment_keys': shared_comm_keys,
                'delayed_comm_alignment_keys': delayed_comm_keys,
                'processed_backward_cmd_uids': on_summary['slowdown_processed_backward_cmd_uids'],
            },
            indent=2,
        )
    )

    if backward_on == backward_off and not delayed_comm_keys:
        raise RuntimeError(
            'Slowdown path executed but produced no visible backward/comm timing change on the chosen case.'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
