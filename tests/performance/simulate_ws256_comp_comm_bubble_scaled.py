#!/usr/bin/env python3
"""Run WS256 simulate-mode timeline with injected comp/comm duration scales.

The script explicitly reports per-rank decomposition:
- comp_execute_ms
- comm_execute_ms
- bubble_ms (waiting/synchronization)
- e2e_sum_ms = comp_execute + comm_execute + bubble
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List


def _collect_rank_breakdown(timelines) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for wrank_id in sorted(timelines.keys()):
        tl = timelines[wrank_id]
        comp_exec = sum(float(op.duration or 0.0) for op in tl.comp_timeline)
        comm_exec = sum(float(op.duration or 0.0) for op in tl.comm_timeline)
        comm_total = sum(float(op.finish_time - op.join_time) for op in tl.comm_timeline)
        bubble = comm_total - comm_exec
        e2e_sum = comp_exec + comm_exec + bubble
        rows.append(
            {
                "wrank": int(wrank_id),
                "stage_id": int(getattr(tl, "stage_id", -1)),
                "comp_execute_ms": float(round(comp_exec, 6)),
                "comm_execute_ms": float(round(comm_exec, 6)),
                "bubble_ms": float(round(bubble, 6)),
                "comm_total_ms": float(round(comm_total, 6)),
                "sum_ms": float(round(e2e_sum, 6)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaled simulate-mode decomposition for WS256")
    parser.add_argument("--comp-scale", type=float, default=1.0)
    parser.add_argument("--comm-scale", type=float, default=1.0)
    parser.add_argument(
        "--schedule-dir",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_256gpus_gpt175b_tp8_pp16_dp2/schedule"
        ),
    )
    parser.add_argument(
        "--database-dir",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_256gpus_gpt175b_tp8_pp16_dp2/database_profile"
        ),
    )
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument("--pp-size", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--exp-size", type=int, default=1)
    parser.add_argument("--local-size", type=int, default=8)
    parser.add_argument("--strategy", default="1F1B-none_interleaved")
    parser.add_argument(
        "--output-json",
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/scaled_simulation_result.json",
    )
    args = parser.parse_args()

    if args.comp_scale <= 0.0 or args.comm_scale <= 0.0:
        raise ValueError("comp-scale and comm-scale must be > 0")

    repo_root = Path(__file__).resolve().parents[2]
    engine_root = repo_root / "megatron-sim-engine"
    schedule_dir = (repo_root / args.schedule_dir).resolve()
    database_dir = (repo_root / args.database_dir).resolve()
    output_json = (repo_root / args.output_json).resolve()

    if not engine_root.exists():
        raise FileNotFoundError(f"engine root not found: {engine_root}")
    if not schedule_dir.exists():
        raise FileNotFoundError(f"schedule dir not found: {schedule_dir}")
    if not database_dir.exists():
        raise FileNotFoundError(f"database dir not found: {database_dir}")

    os.environ.setdefault("SIMULATOR_HARDWARE_TYPE", "H800_SXM")

    sys.path.insert(0, str(engine_root))

    from src.core.static_graphs.parallel_group_manager import MPUInfo, ParallelGroupManager
    from src.core.static_graphs.rank_manager import RankManager
    from src.core.simu_engine import MODE_SIMULATE, SimulatorEngine

    manager = ParallelGroupManager(
        local_size=args.local_size,
        world_size=args.world_size,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        exp_size=args.exp_size,
    )
    mpu_info: MPUInfo = manager.get_mpu_info()
    all_groups = manager.get_all_groups()

    rank_manager = RankManager(
        mpu_info=mpu_info,
        gpus_per_node=args.local_size,
        all_groups=all_groups,
    )
    rank_instances = rank_manager.get_rank_zoos()

    simulator_engine = SimulatorEngine(
        trace_filepath=None,
        framwork="megatron-lm",
        strategy=args.strategy,
        running_mode=MODE_SIMULATE,
        torchgraph_filepath=str(database_dir),
        stages_scheduling_filepath=str(schedule_dir),
        cc_backend_name="collective-sim",
        cc_backend_options={
            "collective-sim": {
                "repo_root": str((engine_root / "src/core/cc_backend/collective-sim").resolve()),
                "placement_mode": "group_size",
            }
        },
    )
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)

    load_t0 = time.time()
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator_engine.validate_global_placement_requirements()
    load_time = time.time() - load_t0

    # Inject comp scale to static op durations before timeline execution.
    timelines = simulator_engine.timeline_manager.stages_timeline_process_dict
    for tl in timelines.values():
        for op in tl.waiting_queue:
            if op.duration is None:
                continue
            if op.op_kind == "comp":
                op.duration = float(op.duration) * args.comp_scale

    # Inject comm scale at CC backend prediction boundary so all communication
    # operations (including ones re-calculated during scheduling) are scaled.
    if args.comm_scale != 1.0:
        cc_backend = simulator_engine.timeline_manager.cc_backend
        if cc_backend is None:
            raise RuntimeError("CC backend is not initialized, cannot apply comm scale.")
        original_predict = cc_backend.predict

        def _scaled_predict(request):  # noqa: ANN001 - runtime wrapper for backend API.
            return float(original_predict(request)) * args.comm_scale

        cc_backend.predict = _scaled_predict

    run_t0 = time.time()
    simulator_engine.start_running()
    run_time = time.time() - run_t0

    rows = _collect_rank_breakdown(simulator_engine.timeline_manager.stages_timeline_process_dict)
    if not rows:
        raise ValueError("No rank rows collected.")

    critical = max(rows, key=lambda r: r["sum_ms"])

    result = {
        "config": {
            "comp_scale": args.comp_scale,
            "comm_scale": args.comm_scale,
            "world_size": args.world_size,
            "pp_size": args.pp_size,
            "tp_size": args.tp_size,
            "exp_size": args.exp_size,
            "local_size": args.local_size,
            "strategy": args.strategy,
            "schedule_dir": str(schedule_dir),
            "database_dir": str(database_dir),
        },
        "timing": {
            "load_time_s": load_time,
            "execution_time_s": run_time,
        },
        "critical_rank": critical,
        "rank_summary": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
