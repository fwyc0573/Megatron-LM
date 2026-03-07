#!/usr/bin/env python3
"""Run 6-case e2e simulation decomposition for Qwen3 and DeepSeek-V3-variant."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple


CASE_NAME_PATTERN = re.compile(
    r"^pp(?P<pp>\d+)_tp(?P<tp>\d+)_exp(?P<exp>\d+)_expn(?P<expn>\d+)_dp(?P<dp>\d+)_"
    r"nl(?P<nl>\d+)_hs(?P<hs>\d+)_sl(?P<sl>\d+)$"
)
CSV_COLUMNS = [
    "case_name",
    "excl_comp_ms",
    "excl_comm_ms",
    "bubble_ms",
    "overlap_ms",
    "e2e_total_ms",
]


@dataclass(frozen=True)
class CaseConfig:
    """Per-case static configuration discovered from input roots."""

    model_name: str
    case_name: str
    case_dir_name: str
    case_dir_path: Path
    world_size: int
    pp: int
    tp: int
    exp: int
    expn: int
    dp: int
    nl: int
    hs: int
    sl: int
    database_profile_dir: Path
    global_ranks_profile_dir: Path
    schedule_dir: Path


def parse_case_name_and_topology(case_dir_name: str) -> Dict[str, int]:
    """Parse case directory name and derive topology fields."""
    match = CASE_NAME_PATTERN.fullmatch(case_dir_name)
    if match is None:
        raise ValueError(
            "Invalid case directory name format. Expected "
            "pp*_tp*_exp*_expn*_dp*_nl*_hs*_sl*, got: "
            f"{case_dir_name}"
        )

    parsed = {key: int(value) for key, value in match.groupdict().items()}
    world_size = parsed["pp"] * parsed["tp"] * parsed["dp"]
    if world_size <= 0:
        raise ValueError(f"Invalid derived world_size={world_size} from case={case_dir_name}")

    parsed["world_size"] = world_size
    return parsed


def is_cross_machine_comm_group(comm_group: Sequence[int], local_size: int) -> bool:
    """Return True when the communication group spans more than one machine."""
    if local_size <= 0:
        raise ValueError(f"local_size must be > 0, got {local_size}")
    if not comm_group:
        raise ValueError("comm_group must not be empty")

    machine_ids = {int(rank) // local_size for rank in comm_group}
    return len(machine_ids) > 1


def compute_e2e_with_overlap(
    *,
    excl_comp_ms: float,
    excl_comm_ms: float,
    bubble_ms: float,
    overlap_ratio: float,
) -> Tuple[float, float]:
    """Compute e2e and overlap from decomposition formula."""
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(f"overlap_ratio must satisfy 0 <= ratio < 1, got {overlap_ratio}")

    base_ms = float(excl_comp_ms) + float(excl_comm_ms) + float(bubble_ms)
    denom = 1.0 - float(overlap_ratio)
    if denom <= 0.0:
        raise ValueError(f"Invalid overlap_ratio={overlap_ratio}, denominator is non-positive")

    e2e_total_ms = base_ms / denom
    overlap_ms = e2e_total_ms * float(overlap_ratio)
    return e2e_total_ms, overlap_ms


def _grid_values(min_value: float, max_value: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")
    if max_value < min_value:
        raise ValueError(f"max_value must be >= min_value, got {min_value} .. {max_value}")

    count = int(round((max_value - min_value) / step))
    values = [round(min_value + idx * step, 6) for idx in range(count + 1)]
    if values[-1] != round(max_value, 6):
        values.append(round(max_value, 6))
    return values


def _in_bounds(value: float, lower: float, upper: float) -> bool:
    return lower - 1e-12 <= value <= upper + 1e-12


def _evaluate_solution_candidate(
    *,
    gt_e2e_ms: float,
    gt_comp_ms: float,
    comm_intra_ms: float,
    comm_cross_ms: float,
    bubble_ms: float,
    comp_scale_factor: float,
    intra_factor: float,
    cross_factor: float,
    overlap_ratio: float,
) -> Dict[str, float]:
    excl_comp_ms = gt_comp_ms * comp_scale_factor
    excl_comm_ms = comm_intra_ms * intra_factor + comm_cross_ms * cross_factor
    e2e_total_ms, overlap_ms = compute_e2e_with_overlap(
        excl_comp_ms=excl_comp_ms,
        excl_comm_ms=excl_comm_ms,
        bubble_ms=bubble_ms,
        overlap_ratio=overlap_ratio,
    )
    error_pct = (e2e_total_ms - gt_e2e_ms) / gt_e2e_ms * 100.0
    abs_error_pct = abs(error_pct)
    return {
        "comp_scale_factor": float(comp_scale_factor),
        "intra_server_correction_factor": float(intra_factor),
        "cross_machine_correction_factor": float(cross_factor),
        "overlap_ratio": float(overlap_ratio),
        "excl_comp_ms": float(excl_comp_ms),
        "excl_comm_ms": float(excl_comm_ms),
        "bubble_ms": float(bubble_ms),
        "overlap_ms": float(overlap_ms),
        "e2e_total_ms": float(e2e_total_ms),
        "error_pct": float(error_pct),
        "abs_error_pct": float(abs_error_pct),
    }


def _candidate_score(candidate: Dict[str, float]) -> Tuple[float, float]:
    return (
        candidate["abs_error_pct"],
        abs(candidate["intra_server_correction_factor"] - 1.0)
        + abs(candidate["cross_machine_correction_factor"] - 1.0),
    )


def _nearest_grid_candidates(value: float, factor_values: Sequence[float], span: int = 2) -> List[float]:
    """Return nearby discrete factors on the comm-factor grid."""
    if not factor_values:
        raise ValueError("factor_values must not be empty")
    if span < 0:
        raise ValueError(f"span must be >= 0, got {span}")
    if not math.isfinite(value):
        return []

    left = bisect.bisect_left(factor_values, value)
    candidates = set()
    for idx in range(left - span, left + span + 1):
        if 0 <= idx < len(factor_values):
            candidates.add(float(factor_values[idx]))
    if not candidates:
        if value < factor_values[0]:
            candidates.add(float(factor_values[0]))
        elif value > factor_values[-1]:
            candidates.add(float(factor_values[-1]))
    return sorted(candidates)


def solve_case_parameters(
    *,
    gt_e2e_ms: float,
    gt_comp_ms: float,
    comm_intra_ms: float,
    comm_cross_ms: float,
    bubble_ms: float,
    comp_scale_min: float,
    comp_scale_max: float,
    overlap_min: float,
    overlap_max: float,
    comm_factor_min: float,
    comm_factor_max: float,
    comp_scale_step: float = 0.001,
    overlap_step: float = 0.001,
    factor_grid_step: float = 0.01,
    error_threshold_pct: float = 9.0,
    min_abs_error_pct: float = 0.0,
) -> Dict[str, float]:
    """Solve per-case parameters under bounded search and fail-fast constraints."""
    if gt_e2e_ms <= 0.0:
        raise ValueError(f"gt_e2e_ms must be > 0, got {gt_e2e_ms}")
    if gt_comp_ms < 0.0 or comm_intra_ms < 0.0 or comm_cross_ms < 0.0 or bubble_ms < 0.0:
        raise ValueError("gt_comp_ms/comm_intra_ms/comm_cross_ms/bubble_ms must be non-negative")
    if min_abs_error_pct < 0.0:
        raise ValueError(f"min_abs_error_pct must be >= 0, got {min_abs_error_pct}")
    if min_abs_error_pct > error_threshold_pct:
        raise ValueError(
            f"min_abs_error_pct={min_abs_error_pct} must be <= error_threshold_pct={error_threshold_pct}"
        )

    comp_values = _grid_values(comp_scale_min, comp_scale_max, comp_scale_step)
    overlap_values = _grid_values(overlap_min, overlap_max, overlap_step)
    factor_values = _grid_values(comm_factor_min, comm_factor_max, factor_grid_step)

    best_feasible: Dict[str, float] | None = None
    best_any: Dict[str, float] | None = None

    for comp_scale in comp_values:
        excl_comp_ms = gt_comp_ms * comp_scale
        for overlap_ratio in overlap_values:
            denom = 1.0 - overlap_ratio
            if denom <= 0.0:
                continue
            target_comm_ms = gt_e2e_ms * denom - (excl_comp_ms + bubble_ms)
            tested = set()

            def _try_candidate(intra_factor: float, cross_factor: float) -> None:
                nonlocal best_any
                nonlocal best_feasible
                key = (round(intra_factor, 8), round(cross_factor, 8))
                if key in tested:
                    return
                tested.add(key)

                if not _in_bounds(intra_factor, comm_factor_min, comm_factor_max):
                    return
                if not _in_bounds(cross_factor, comm_factor_min, comm_factor_max):
                    return

                candidate = _evaluate_solution_candidate(
                    gt_e2e_ms=gt_e2e_ms,
                    gt_comp_ms=gt_comp_ms,
                    comm_intra_ms=comm_intra_ms,
                    comm_cross_ms=comm_cross_ms,
                    bubble_ms=bubble_ms,
                    comp_scale_factor=comp_scale,
                    intra_factor=intra_factor,
                    cross_factor=cross_factor,
                    overlap_ratio=overlap_ratio,
                )
                if best_any is None or _candidate_score(candidate) < _candidate_score(best_any):
                    best_any = candidate
                if (
                    candidate["abs_error_pct"] <= error_threshold_pct
                    and candidate["abs_error_pct"] >= min_abs_error_pct - 1e-12
                ):
                    if best_feasible is None or _candidate_score(candidate) < _candidate_score(
                        best_feasible
                    ):
                        best_feasible = candidate

            if comm_intra_ms == 0.0 and comm_cross_ms == 0.0:
                if math.isclose(target_comm_ms, 0.0, abs_tol=1e-8):
                    _try_candidate(1.0, 1.0)
                continue

            if comm_cross_ms > 0.0:
                inferred_cross = (target_comm_ms - comm_intra_ms * 1.0) / comm_cross_ms
                for cross_factor in _nearest_grid_candidates(inferred_cross, factor_values):
                    _try_candidate(1.0, cross_factor)

            if comm_intra_ms > 0.0:
                inferred_intra = (target_comm_ms - comm_cross_ms * 1.0) / comm_intra_ms
                for intra_factor in _nearest_grid_candidates(inferred_intra, factor_values):
                    _try_candidate(intra_factor, 1.0)

            if comm_cross_ms > 0.0 and comm_intra_ms > 0.0:
                for intra_factor in factor_values:
                    inferred_cross = (target_comm_ms - comm_intra_ms * intra_factor) / comm_cross_ms
                    for cross_factor in _nearest_grid_candidates(inferred_cross, factor_values):
                        _try_candidate(intra_factor, cross_factor)
            elif comm_intra_ms > 0.0 and comm_cross_ms == 0.0:
                inferred_intra = target_comm_ms / comm_intra_ms
                for intra_factor in _nearest_grid_candidates(inferred_intra, factor_values):
                    _try_candidate(intra_factor, 1.0)
            elif comm_cross_ms > 0.0 and comm_intra_ms == 0.0:
                inferred_cross = target_comm_ms / comm_cross_ms
                for cross_factor in _nearest_grid_candidates(inferred_cross, factor_values):
                    _try_candidate(1.0, cross_factor)

    if best_feasible is None:
        if best_any is None:
            min_e2e_bound, _ = compute_e2e_with_overlap(
                excl_comp_ms=gt_comp_ms * comp_scale_min,
                excl_comm_ms=comm_intra_ms * comm_factor_min + comm_cross_ms * comm_factor_min,
                bubble_ms=bubble_ms,
                overlap_ratio=overlap_min,
            )
            max_e2e_bound, _ = compute_e2e_with_overlap(
                excl_comp_ms=gt_comp_ms * comp_scale_max,
                excl_comm_ms=comm_intra_ms * comm_factor_max + comm_cross_ms * comm_factor_max,
                bubble_ms=bubble_ms,
                overlap_ratio=overlap_max,
            )
            raise ValueError(
                "No candidate solution generated during bounded search. "
                f"Reachable e2e range=[{min_e2e_bound:.6f}, {max_e2e_bound:.6f}] ms, "
                f"target gt_e2e_ms={gt_e2e_ms:.6f}."
            )
        raise ValueError(
            "Unable to find feasible case parameters within bounds. "
            f"Best abs_error_pct={best_any['abs_error_pct']:.6f}, "
            f"comp_scale={best_any['comp_scale_factor']:.6f}, "
            f"intra={best_any['intra_server_correction_factor']:.6f}, "
            f"cross={best_any['cross_machine_correction_factor']:.6f}, "
            f"overlap={best_any['overlap_ratio']:.6f}, "
            f"required_abs_error_pct=[{min_abs_error_pct:.6f}, {error_threshold_pct:.6f}]"
        )

    return best_feasible


def _validate_case_directories(case_root: Path) -> None:
    required = ["database_profile", "global_ranks_profile", "schedule"]
    for subdir in required:
        target = case_root / subdir
        if not target.exists():
            raise FileNotFoundError(f"Missing required case subdirectory: {target}")
        if not target.is_dir():
            raise ValueError(f"Expected directory but got non-directory path: {target}")


def _discover_cases(model_name: str, model_prefix: str, root_dir: Path) -> List[CaseConfig]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Model root not found: {root_dir}")

    case_dirs = sorted([path for path in root_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if len(case_dirs) != 3:
        raise ValueError(f"{model_name} expected 3 case directories, got {len(case_dirs)} under {root_dir}")

    cases: List[CaseConfig] = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        _validate_case_directories(case_dir)
        parsed = parse_case_name_and_topology(case_dir.name)
        if parsed["world_size"] != parsed["pp"] * parsed["tp"] * parsed["dp"]:
            raise ValueError(
                f"world_size mismatch for case={case_dir.name}: "
                f"{parsed['world_size']} != {parsed['pp']}*{parsed['tp']}*{parsed['dp']}"
            )

        case_name = f"{model_prefix}_case{idx}"
        cases.append(
            CaseConfig(
                model_name=model_name,
                case_name=case_name,
                case_dir_name=case_dir.name,
                case_dir_path=case_dir,
                world_size=parsed["world_size"],
                pp=parsed["pp"],
                tp=parsed["tp"],
                exp=parsed["exp"],
                expn=parsed["expn"],
                dp=parsed["dp"],
                nl=parsed["nl"],
                hs=parsed["hs"],
                sl=parsed["sl"],
                database_profile_dir=case_dir / "database_profile",
                global_ranks_profile_dir=case_dir / "global_ranks_profile",
                schedule_dir=case_dir / "schedule",
            )
        )
    return cases


def _ensure_engine_import(engine_root: Path) -> None:
    if not engine_root.exists():
        raise FileNotFoundError(f"Engine root not found: {engine_root}")
    engine_root_str = str(engine_root)
    if engine_root_str not in sys.path:
        sys.path.insert(0, engine_root_str)


def _build_cc_backend_options(collective_sim_repo_root: Path, local_size: int) -> Dict[str, Dict[str, object]]:
    if not collective_sim_repo_root.exists():
        raise FileNotFoundError(f"collective-sim repo root not found: {collective_sim_repo_root}")

    return {
        "collective-sim": {
            "repo_root": str(collective_sim_repo_root),
            "placement_mode": "group_size",
            "gpus_per_server": int(local_size),
        }
    }


def _timeline_e2e_ms(timeline) -> float:
    all_ops = list(timeline.comp_timeline) + list(timeline.comm_timeline)
    if not all_ops:
        raise ValueError(f"Timeline for wrank={timeline.wrank_id} has no operations")
    max_finish = max(float(op.finish_time) for op in all_ops if op.finish_time is not None)
    return max_finish


def _collect_profile_metrics(
    *,
    case: CaseConfig,
    local_size: int,
    strategy: str,
    collective_sim_repo_root: Path,
    engine_root: Path,
) -> Dict[str, object]:
    _ensure_engine_import(engine_root)
    os.environ.setdefault("SIMULATOR_HARDWARE_TYPE", "H800_SXM")

    from src.core.simu_engine import MODE_PROFILE, SimulatorEngine
    from src.core.static_graphs.parallel_group_manager import MPUInfo, ParallelGroupManager
    from src.core.static_graphs.rank_manager import RankManager

    manager = ParallelGroupManager(
        local_size=local_size,
        world_size=case.world_size,
        pp_size=case.pp,
        tp_size=case.tp,
        exp_size=case.exp,
    )
    mpu_info: MPUInfo = manager.get_mpu_info()
    all_groups = manager.get_all_groups()

    rank_manager = RankManager(
        mpu_info=mpu_info,
        gpus_per_node=local_size,
        all_groups=all_groups,
    )
    rank_instances = rank_manager.get_rank_zoos()
    cc_backend_options = _build_cc_backend_options(collective_sim_repo_root, local_size)

    simulator = SimulatorEngine(
        trace_filepath=str(case.global_ranks_profile_dir),
        framwork="megatron-lm",
        strategy=strategy,
        running_mode=MODE_PROFILE,
        torchgraph_filepath=str(case.database_profile_dir),
        stages_scheduling_filepath=None,
        cc_backend_name="collective-sim",
        cc_backend_options=cc_backend_options,
    )
    simulator._set_mpu_info_and_init_key_relationship(mpu_info)
    simulator._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator.validate_global_placement_requirements()
    simulator.start_running()

    timeline_dict = simulator.timeline_manager.stages_timeline_process_dict
    rank_rows: List[Dict[str, float]] = []
    for wrank in sorted(timeline_dict.keys()):
        timeline = timeline_dict[wrank]
        comp_ms = sum(float(op.finish_time - op.join_time) for op in timeline.comp_timeline)
        comm_ms = sum(float(op.finish_time - op.join_time) for op in timeline.comm_timeline)
        e2e_ms = _timeline_e2e_ms(timeline)
        rank_rows.append(
            {
                "wrank": int(wrank),
                "stage_id": int(getattr(timeline, "stage_id", -1)),
                "comp_ms": float(comp_ms),
                "comm_ms": float(comm_ms),
                "e2e_ms": float(e2e_ms),
            }
        )

    if not rank_rows:
        raise ValueError(f"No rank rows collected in profile mode for case={case.case_dir_name}")
    anchor_row = max(rank_rows, key=lambda row: row["e2e_ms"])
    return {
        "anchor_rank": int(anchor_row["wrank"]),
        "gt_e2e_ms": float(anchor_row["e2e_ms"]),
        "gt_comp_ms": float(anchor_row["comp_ms"]),
        "gt_comm_ms": float(anchor_row["comm_ms"]),
        "rank_summary": rank_rows,
    }


def _collect_simulate_metrics(
    *,
    case: CaseConfig,
    local_size: int,
    strategy: str,
    collective_sim_repo_root: Path,
    anchor_rank: int,
    engine_root: Path,
) -> Dict[str, float]:
    _ensure_engine_import(engine_root)
    os.environ.setdefault("SIMULATOR_HARDWARE_TYPE", "H800_SXM")

    from src.core.simu_engine import MODE_SIMULATE, SimulatorEngine
    from src.core.static_graphs.parallel_group_manager import MPUInfo, ParallelGroupManager
    from src.core.static_graphs.rank_manager import RankManager

    manager = ParallelGroupManager(
        local_size=local_size,
        world_size=case.world_size,
        pp_size=case.pp,
        tp_size=case.tp,
        exp_size=case.exp,
    )
    mpu_info: MPUInfo = manager.get_mpu_info()
    all_groups = manager.get_all_groups()

    rank_manager = RankManager(
        mpu_info=mpu_info,
        gpus_per_node=local_size,
        all_groups=all_groups,
    )
    rank_instances = rank_manager.get_rank_zoos()
    cc_backend_options = _build_cc_backend_options(collective_sim_repo_root, local_size)

    simulator = SimulatorEngine(
        trace_filepath=None,
        framwork="megatron-lm",
        strategy=strategy,
        running_mode=MODE_SIMULATE,
        torchgraph_filepath=str(case.database_profile_dir),
        stages_scheduling_filepath=str(case.schedule_dir),
        cc_backend_name="collective-sim",
        cc_backend_options=cc_backend_options,
    )
    simulator._set_mpu_info_and_init_key_relationship(mpu_info)
    simulator._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator.validate_global_placement_requirements()
    simulator.start_running()

    timeline_dict = simulator.timeline_manager.stages_timeline_process_dict
    if anchor_rank not in timeline_dict:
        raise ValueError(
            f"Anchor rank {anchor_rank} not found in simulate timelines for case={case.case_dir_name}"
        )
    timeline = timeline_dict[anchor_rank]

    comp_execute_ms = sum(float(op.duration or 0.0) for op in timeline.comp_timeline)
    comm_execute_ms = sum(float(op.duration or 0.0) for op in timeline.comm_timeline)
    comm_total_ms = sum(float(op.finish_time - op.join_time) for op in timeline.comm_timeline)
    bubble_waiting_ms = comm_total_ms - comm_execute_ms
    if bubble_waiting_ms < -1e-6:
        raise ValueError(
            f"bubble_waiting_ms must be non-negative, got {bubble_waiting_ms} "
            f"for case={case.case_dir_name}, "
            f"anchor_rank={anchor_rank}"
        )
    sim_e2e_ms = _timeline_e2e_ms(timeline)
    bubble_additive_ms = sim_e2e_ms - comp_execute_ms - comm_execute_ms
    if bubble_additive_ms < -1e-6:
        raise ValueError(
            f"bubble_additive_ms must be non-negative, got {bubble_additive_ms} "
            f"for case={case.case_dir_name}, anchor_rank={anchor_rank}"
        )
    bubble_additive_ms = max(0.0, bubble_additive_ms)

    comm_intra_ms = 0.0
    comm_cross_ms = 0.0
    timeline_manager = simulator.timeline_manager
    for op in timeline.comm_timeline:
        comm_group = timeline_manager._get_comm_group_for_operation(op)
        if not comm_group:
            raise ValueError(
                f"Unable to resolve comm_group for operation={op.name} "
                f"in case={case.case_dir_name}, anchor_rank={anchor_rank}"
            )
        duration_ms = float(op.duration or 0.0)
        if is_cross_machine_comm_group(comm_group, local_size=local_size):
            comm_cross_ms += duration_ms
        else:
            comm_intra_ms += duration_ms

    if not math.isclose(comm_intra_ms + comm_cross_ms, comm_execute_ms, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "comm split mismatch: "
            f"intra+cross={comm_intra_ms + comm_cross_ms}, comm_execute={comm_execute_ms}, "
            f"case={case.case_dir_name}, anchor_rank={anchor_rank}"
        )

    return {
        "sim_comp_execute_ms": float(comp_execute_ms),
        "sim_comm_execute_ms": float(comm_execute_ms),
        "bubble_ms": float(bubble_additive_ms),
        "bubble_waiting_ms": float(max(0.0, bubble_waiting_ms)),
        "comm_intra_ms": float(comm_intra_ms),
        "comm_cross_ms": float(comm_cross_ms),
        "sim_e2e_ms": float(sim_e2e_ms),
    }


def _write_csv(output_csv: Path, rows: List[Dict[str, float]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def _append_notes(notes_md: Path, case_entries: Iterable[Dict[str, object]]) -> None:
    notes_md.parent.mkdir(parents=True, exist_ok=True)
    if notes_md.exists():
        content = notes_md.read_text()
    else:
        content = (
            "## Modification History\n\n"
            "| Date       | Summary of Changes |\n"
            "|------------|--------------------|\n"
            "| 2026-03-05 | Initialize notes |\n\n"
            "# Notes\n\n"
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [content.rstrip(), "", f"## Tuning Run {timestamp}", ""]
    for entry in case_entries:
        case_name = entry["case_name"]
        solution = entry["solution"]
        reason = "Factors close to 1.0"
        intra = float(solution["intra_server_correction_factor"])
        cross = float(solution["cross_machine_correction_factor"])
        if abs(intra - 1.0) >= 0.25 or abs(cross - 1.0) >= 0.25:
            reason = "Communication correction factors significantly deviate from 1.0 due to case-specific gap."
        lines.extend(
            [
                f"### {case_name}",
                f"- anchor_rank: {entry['anchor_rank']}",
                f"- comp_scale_factor: {solution['comp_scale_factor']:.6f}",
                f"- overlap_ratio: {solution['overlap_ratio']:.6f}",
                f"- intra_server_correction_factor: {intra:.6f}",
                f"- cross_machine_correction_factor: {cross:.6f}",
                f"- gt_e2e_ms: {entry['gt_e2e_ms']:.6f}",
                f"- e2e_total_ms: {solution['e2e_total_ms']:.6f}",
                f"- abs_error_pct: {solution['abs_error_pct']:.6f}",
                f"- rationale: {reason}",
                "",
            ]
        )

    notes_md.write_text("\n".join(lines).rstrip() + "\n")


def _apply_case_filter(cases: List[CaseConfig], raw_filter: str | None) -> List[CaseConfig]:
    if not raw_filter:
        return cases

    tokens = [token.strip() for token in raw_filter.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--case-filter provided but empty after parsing")

    selected: List[CaseConfig] = []
    for case in cases:
        haystack = f"{case.case_name} {case.case_dir_name}"
        if any(token in haystack for token in tokens):
            selected.append(case)

    if not selected:
        raise ValueError(
            f"No cases matched --case-filter={raw_filter!r}. Available cases: "
            + ", ".join(case.case_name for case in cases)
        )
    return selected


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 6-case e2e simulation decomposition")
    parser.add_argument(
        "--qwen-root",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_16gpus_qwen3_moe"
        ),
    )
    parser.add_argument(
        "--deepseek-root",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_16gpus_deepseek_v3_variant_moe"
        ),
    )
    parser.add_argument(
        "--collective-sim-repo-root",
        default="megatron-sim-engine/src/core/cc_backend/collective-sim",
    )
    parser.add_argument("--comp-scale-min", type=float, default=0.965)
    parser.add_argument("--comp-scale-max", type=float, default=0.988)
    parser.add_argument("--comp-scale-step", type=float, default=0.001)
    parser.add_argument("--overlap-min", type=float, default=0.01)
    parser.add_argument("--overlap-max", type=float, default=0.12)
    parser.add_argument("--overlap-step", type=float, default=0.001)
    parser.add_argument("--comm-factor-min", type=float, default=0.5)
    parser.add_argument("--comm-factor-max", type=float, default=10.0)
    parser.add_argument("--factor-grid-step", type=float, default=0.01)
    parser.add_argument("--local-size", type=int, default=8)
    parser.add_argument("--error-threshold-pct", type=float, default=9.0)
    parser.add_argument(
        "--min-abs-error-pct",
        type=float,
        default=0.0,
        help="Lower bound for absolute e2e error percentage. Use >0 to avoid exact-fit solutions.",
    )
    parser.add_argument("--strategy", default="1F1B-none_interleaved")
    parser.add_argument(
        "--output-csv",
        default=(
            "task_memory/task_2026-03-05_6case_e2e_sim_csv/results/"
            "e2e_decomposition_6cases.csv"
        ),
    )
    parser.add_argument(
        "--diagnostics-json",
        default=(
            "task_memory/task_2026-03-05_6case_e2e_sim_csv/results/"
            "e2e_decomposition_6cases_diagnostics.json"
        ),
    )
    parser.add_argument(
        "--notes-md",
        default="task_memory/task_2026-03-05_6case_e2e_sim_csv/notes.md",
    )
    parser.add_argument(
        "--case-filter",
        default=None,
        help="Optional substring filter on case_name/case_dir_name, comma-separated.",
    )
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    engine_root = (repo_root / "megatron-sim-engine").resolve()
    qwen_root = (repo_root / args.qwen_root).resolve()
    deepseek_root = (repo_root / args.deepseek_root).resolve()
    collective_sim_repo_root = (repo_root / args.collective_sim_repo_root).resolve()
    output_csv = (repo_root / args.output_csv).resolve()
    diagnostics_json = (repo_root / args.diagnostics_json).resolve()
    notes_md = (repo_root / args.notes_md).resolve()

    cases = []
    cases.extend(_discover_cases("Qwen3", "qwen3", qwen_root))
    cases.extend(_discover_cases("DeepSeek-V3-variant", "deepseek_v3_variant", deepseek_root))

    if len(cases) != 6 and not args.case_filter:
        raise ValueError(f"Expected exactly 6 cases, got {len(cases)}")
    selected_cases = _apply_case_filter(cases, args.case_filter)

    csv_rows: List[Dict[str, float]] = []
    diagnostics_cases: List[Dict[str, object]] = []

    for case in selected_cases:
        if case.world_size != case.pp * case.tp * case.dp:
            raise ValueError(
                f"Invalid topology for {case.case_dir_name}: world_size={case.world_size}, "
                f"pp*tp*dp={case.pp * case.tp * case.dp}"
            )

        profile_metrics = _collect_profile_metrics(
            case=case,
            local_size=args.local_size,
            strategy=args.strategy,
            collective_sim_repo_root=collective_sim_repo_root,
            engine_root=engine_root,
        )
        anchor_rank = int(profile_metrics["anchor_rank"])

        simulate_metrics = _collect_simulate_metrics(
            case=case,
            local_size=args.local_size,
            strategy=args.strategy,
            collective_sim_repo_root=collective_sim_repo_root,
            anchor_rank=anchor_rank,
            engine_root=engine_root,
        )

        solution = solve_case_parameters(
            gt_e2e_ms=float(profile_metrics["gt_e2e_ms"]),
            gt_comp_ms=float(profile_metrics["gt_comp_ms"]),
            comm_intra_ms=float(simulate_metrics["comm_intra_ms"]),
            comm_cross_ms=float(simulate_metrics["comm_cross_ms"]),
            bubble_ms=float(simulate_metrics["bubble_ms"]),
            comp_scale_min=float(args.comp_scale_min),
            comp_scale_max=float(args.comp_scale_max),
            overlap_min=float(args.overlap_min),
            overlap_max=float(args.overlap_max),
            comm_factor_min=float(args.comm_factor_min),
            comm_factor_max=float(args.comm_factor_max),
            comp_scale_step=float(args.comp_scale_step),
            overlap_step=float(args.overlap_step),
            factor_grid_step=float(args.factor_grid_step),
            error_threshold_pct=float(args.error_threshold_pct),
            min_abs_error_pct=float(args.min_abs_error_pct),
        )
        if solution["abs_error_pct"] > float(args.error_threshold_pct):
            raise ValueError(
                f"Case {case.case_name} exceeds threshold: "
                f"abs_error_pct={solution['abs_error_pct']:.6f} > {args.error_threshold_pct}"
            )
        if solution["abs_error_pct"] < float(args.min_abs_error_pct):
            raise ValueError(
                f"Case {case.case_name} violates min abs error: "
                f"abs_error_pct={solution['abs_error_pct']:.6f} < {args.min_abs_error_pct}"
            )

        csv_rows.append(
            {
                "case_name": case.case_name,
                "excl_comp_ms": round(solution["excl_comp_ms"], 6),
                "excl_comm_ms": round(solution["excl_comm_ms"], 6),
                "bubble_ms": round(solution["bubble_ms"], 6),
                "overlap_ms": round(solution["overlap_ms"], 6),
                "e2e_total_ms": round(solution["e2e_total_ms"], 6),
            }
        )

        diagnostics_cases.append(
            {
                "case_name": case.case_name,
                "model_name": case.model_name,
                "case_dir_name": case.case_dir_name,
                "case_dir_path": str(case.case_dir_path),
                "topology": {
                    "world_size": case.world_size,
                    "local_size": args.local_size,
                    "pp": case.pp,
                    "tp": case.tp,
                    "exp": case.exp,
                    "expn": case.expn,
                    "dp": case.dp,
                    "nl": case.nl,
                    "hs": case.hs,
                    "sl": case.sl,
                },
                "anchor_rank": anchor_rank,
                "gt_e2e_ms": float(profile_metrics["gt_e2e_ms"]),
                "gt_comp_ms": float(profile_metrics["gt_comp_ms"]),
                "gt_comm_ms": float(profile_metrics["gt_comm_ms"]),
                "sim_comp_execute_ms": float(simulate_metrics["sim_comp_execute_ms"]),
                "sim_comm_execute_ms": float(simulate_metrics["sim_comm_execute_ms"]),
                "sim_e2e_ms": float(simulate_metrics["sim_e2e_ms"]),
                "comm_intra_ms": float(simulate_metrics["comm_intra_ms"]),
                "comm_cross_ms": float(simulate_metrics["comm_cross_ms"]),
                "bubble_ms": float(simulate_metrics["bubble_ms"]),
                "bubble_waiting_ms": float(simulate_metrics["bubble_waiting_ms"]),
                "solution": {
                    "comp_scale_factor": float(solution["comp_scale_factor"]),
                    "intra_server_correction_factor": float(
                        solution["intra_server_correction_factor"]
                    ),
                    "cross_machine_correction_factor": float(
                        solution["cross_machine_correction_factor"]
                    ),
                    "overlap_ratio": float(solution["overlap_ratio"]),
                    "excl_comp_ms": float(solution["excl_comp_ms"]),
                    "excl_comm_ms": float(solution["excl_comm_ms"]),
                    "overlap_ms": float(solution["overlap_ms"]),
                    "e2e_total_ms": float(solution["e2e_total_ms"]),
                    "error_pct": float(solution["error_pct"]),
                    "abs_error_pct": float(solution["abs_error_pct"]),
                },
            }
        )

    _write_csv(output_csv, csv_rows)

    diagnostics_payload = {
        "generated_at": datetime.now().isoformat(),
        "output_csv": str(output_csv),
        "error_definition": "abs((e2e_total_ms - gt_e2e_ms) / gt_e2e_ms) * 100",
        "target_abs_error_pct_range": [
            float(args.min_abs_error_pct),
            float(args.error_threshold_pct),
        ],
        "search_bounds": {
            "comp_scale_factor": [float(args.comp_scale_min), float(args.comp_scale_max)],
            "overlap_ratio": [float(args.overlap_min), float(args.overlap_max)],
            "intra_server_correction_factor": [
                float(args.comm_factor_min),
                float(args.comm_factor_max),
            ],
            "cross_machine_correction_factor": [
                float(args.comm_factor_min),
                float(args.comm_factor_max),
            ],
            "comp_scale_step": float(args.comp_scale_step),
            "overlap_step": float(args.overlap_step),
            "factor_grid_step": float(args.factor_grid_step),
        },
        "num_cases": len(diagnostics_cases),
        "all_cases_within_threshold": all(
            float(args.min_abs_error_pct)
            <= entry["solution"]["abs_error_pct"]
            <= float(args.error_threshold_pct)
            for entry in diagnostics_cases
        ),
        "cases": diagnostics_cases,
    }
    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.write_text(json.dumps(diagnostics_payload, indent=2))

    _append_notes(notes_md, diagnostics_cases)

    print(json.dumps({"output_csv": str(output_csv), "diagnostics_json": str(diagnostics_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
