#!/usr/bin/env python3
"""Run PROFILE mode on reconstructed traces and verify target errors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List


def _run_profile_and_collect(
    repo_root: Path,
    trace_dir: Path,
    database_dir: Path,
    world_size: int,
    pp_size: int,
    tp_size: int,
    exp_size: int,
    local_size: int,
    strategy: str,
    cc_backend: str,
    cc_backend_options: Dict[str, object],
) -> Dict[str, object]:
    engine_root = repo_root / "megatron-sim-engine"
    if not engine_root.exists():
        raise FileNotFoundError(f"Engine root not found: {engine_root}")

    sys.path.insert(0, str(engine_root))

    # Local imports after sys.path injection.
    from src.core.static_graphs.parallel_group_manager import MPUInfo, ParallelGroupManager
    from src.core.static_graphs.rank_manager import RankManager
    from src.core.simu_engine import MODE_PROFILE, SimulatorEngine

    manager = ParallelGroupManager(
        local_size=local_size,
        world_size=world_size,
        pp_size=pp_size,
        tp_size=tp_size,
        exp_size=exp_size,
    )
    mpu_info: MPUInfo = manager.get_mpu_info()
    all_groups = manager.get_all_groups()

    rank_manager = RankManager(
        mpu_info=mpu_info,
        gpus_per_node=local_size,
        all_groups=all_groups,
    )
    rank_instances = rank_manager.get_rank_zoos()

    simulator_engine = SimulatorEngine(
        trace_filepath=str(trace_dir),
        framwork="megatron-lm",
        strategy=strategy,
        running_mode=MODE_PROFILE,
        torchgraph_filepath=str(database_dir),
        stages_scheduling_filepath=None,
        cc_backend_name=cc_backend,
        cc_backend_options=cc_backend_options,
    )
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)

    t0 = time.time()
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator_engine.validate_global_placement_requirements()
    load_s = time.time() - t0

    t1 = time.time()
    simulator_engine.start_running()
    exec_s = time.time() - t1

    timelines = simulator_engine.timeline_manager.stages_timeline_process_dict
    rank_rows: List[Dict[str, float]] = []

    for wrank_id in sorted(timelines.keys()):
        tl = timelines[wrank_id]
        comp_ms = sum((op.finish_time - op.join_time) for op in tl.comp_timeline)
        comm_ms = sum((op.finish_time - op.join_time) for op in tl.comm_timeline)
        rank_rows.append(
            {
                "wrank": int(wrank_id),
                "stage_id": int(getattr(tl, "stage_id", -1)),
                "comp_ms": float(round(comp_ms, 6)),
                "comm_ms": float(round(comm_ms, 6)),
                "sum_ms": float(round(comp_ms + comm_ms, 6)),
            }
        )

    if not rank_rows:
        raise ValueError("No rank rows collected from profile timeline.")

    stage_buckets: Dict[int, List[float]] = {}
    for row in rank_rows:
        stage_buckets.setdefault(row["stage_id"], []).append(row["sum_ms"])

    stage_summary = []
    for stage_id in sorted(stage_buckets.keys()):
        vals = stage_buckets[stage_id]
        stage_summary.append(
            {
                "stage_id": int(stage_id),
                "count": len(vals),
                "min_sum_ms": float(round(min(vals), 6)),
                "mean_sum_ms": float(round(sum(vals) / len(vals), 6)),
                "max_sum_ms": float(round(max(vals), 6)),
            }
        )

    critical = max(rank_rows, key=lambda x: x["sum_ms"])

    return {
        "sim_load_time_s": load_s,
        "sim_execution_time_s": exec_s,
        "critical_rank": critical,
        "rank_summary": rank_rows,
        "stage_summary": stage_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify reverse-groundtruth profile traces")
    parser.add_argument(
        "--baseline-metrics-json",
        default=(
            "task_memory/task_2026-03-04_ws256_dense_simulation_phase/logs/"
            "ws256_dense_simulation_metrics_20260304_140456.json"
        ),
    )
    parser.add_argument(
        "--solution-json",
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution.json",
    )
    parser.add_argument(
        "--trace-dir",
        default="task_memory/task_2026-03-04_reverse_groundtruth/reconstructed_traces",
    )
    parser.add_argument(
        "--database-dir",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_256gpus_gpt175b_tp8_pp16_dp2/database_profile"
        ),
    )
    parser.add_argument(
        "--output-json",
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/profile_verification.json",
    )
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument("--pp-size", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--exp-size", type=int, default=1)
    parser.add_argument("--local-size", type=int, default=8)
    parser.add_argument("--strategy", default="1F1B-none_interleaved")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    baseline_metrics_path = repo_root / args.baseline_metrics_json
    solution_path = repo_root / args.solution_json
    trace_dir = repo_root / args.trace_dir
    database_dir = repo_root / args.database_dir
    output_json = repo_root / args.output_json

    if not baseline_metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution JSON not found: {solution_path}")
    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace dir not found: {trace_dir}")
    if not database_dir.exists():
        raise FileNotFoundError(f"Database dir not found: {database_dir}")

    baseline = json.loads(baseline_metrics_path.read_text())
    solution = json.loads(solution_path.read_text())

    os.environ.setdefault("SIMULATOR_HARDWARE_TYPE", "H800_SXM")

    run_result = _run_profile_and_collect(
        repo_root=repo_root,
        trace_dir=trace_dir,
        database_dir=database_dir,
        world_size=args.world_size,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        exp_size=args.exp_size,
        local_size=args.local_size,
        strategy=args.strategy,
        cc_backend="collective-sim",
        cc_backend_options={
            "repo_root": str(
                (repo_root / "megatron-sim-engine" / "src/core/cc_backend/collective-sim").resolve()
            )
        },
    )

    sim_critical = max(baseline["rank_summary"], key=lambda x: float(x["sum_ms"]))
    gt_critical = run_result["critical_rank"]

    sim_e2e = float(sim_critical["sum_ms"])
    sim_comp = float(sim_critical["comp_ms"])
    sim_comm = float(sim_critical["comm_ms"])

    gt_e2e = float(gt_critical["sum_ms"])
    gt_comp = float(gt_critical["comp_ms"])
    gt_comm = float(gt_critical["comm_ms"])

    verification = {
        "error_definition": "(simulation - ground_truth) / ground_truth",
        "targets": {
            "target_e2e_error_pct": float(solution["inputs"]["overall_error_pct"]),
            "target_comp_error_pct": float(solution["inputs"]["comp_error_pct"]),
            "target_comm_error_pct": float(solution["derived_comm_error_pct"]),
        },
        "baseline_critical": {
            "wrank": int(sim_critical["wrank"]),
            "stage_id": int(sim_critical["stage_id"]),
            "sim_e2e_ms": sim_e2e,
            "sim_comp_ms": sim_comp,
            "sim_comm_ms": sim_comm,
        },
        "reconstructed_critical": {
            "wrank": int(gt_critical["wrank"]),
            "stage_id": int(gt_critical["stage_id"]),
            "gt_e2e_ms": gt_e2e,
            "gt_comp_ms": gt_comp,
            "gt_comm_ms": gt_comm,
        },
        "achieved_errors_pct": {
            "e2e_error_pct": (sim_e2e - gt_e2e) / gt_e2e * 100.0,
            "comp_error_pct": (sim_comp - gt_comp) / gt_comp * 100.0,
            "comm_error_pct": (sim_comm - gt_comm) / gt_comm * 100.0,
        },
        "profile_run": {
            "sim_load_time_s": run_result["sim_load_time_s"],
            "sim_execution_time_s": run_result["sim_execution_time_s"],
        },
        "rank_summary": run_result["rank_summary"],
        "stage_summary": run_result["stage_summary"],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(verification, indent=2))

    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
