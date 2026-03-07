#!/usr/bin/env python3
"""Reconstruct WS256 ground-truth-like trace files using stage scheduling plans.

This script generates `global_ranks_profile` traces that are schedule-aligned:
- For each rank, operation sequence is copied from its PP stage scheduling file.
- Operation durations are synthesized from per-stage comp/comm targets solved from error constraints.
- forward/backward/loss_func include sub_operations templates to satisfy PROFILE-mode split logic.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


MG_COMP_OPS = {"forward_step", "backward_step", "optimizer_step", "get_batch", "loss_func"}
SPLIT_OPS = {"forward_step", "backward_step", "loss_func"}

PARENT_SPLIT_OVERHEAD_MS = 0.01
MIN_DURATION_MS = 0.001

COMM_WEIGHT_BY_OP = {
    "send_forward": 1.0,
    "recv_forward": 1.0,
    "send_backward": 1.0,
    "recv_backward": 1.0,
    "dp_allreduce": 4.0,
    "ep_allreduce": 2.5,
    "exp_dp_allreduce": 2.5,
    "ep_dp_allreduce": 2.5,
    "tp_allreduce": 1.5,
    "tp_reducescatter": 1.5,
    "dp_reducescatter": 1.5,
    "exp_reducescatter": 1.5,
    "cp_reducescatter": 1.5,
    "exp_all_to_all": 1.8,
    "exp_allgather": 1.8,
}

DEFAULT_COMP_DURATION_MS = {
    "get_batch": 0.20,
    "forward_step": 12.0,
    "backward_step": 21.0,
    "loss_func": 0.20,
    "optimizer_step": 25.0,
}


def _require_match(pattern: str, text: str, field: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Failed to parse {field} from: {text}")
    return match.group(1)


def _extract_suboperations_list(trace_line: str) -> List[str]:
    raw = _require_match(r"sub_operations=(\[.*\])\)$", trace_line, "sub_operations")
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError(f"sub_operations is not a list: {type(value)}")
    return [str(item) for item in value]


def _extract_numeric(trace_line: str, key: str) -> float:
    raw = _require_match(rf"{re.escape(key)}=(-?\d+(?:\.\d+)?)", trace_line, key)
    return float(raw)


def _extract_shape_literal(trace_line: str, key: str) -> str:
    return _require_match(rf"{re.escape(key)}=(\[[^\]]*\]|None)", trace_line, key)


def _extract_scalar_literal(trace_line: str, key: str) -> str:
    return _require_match(rf"{re.escape(key)}=([^,\)]+)", trace_line, key)


def _find_template_file(source_profile_dir: Path, rank_id: int) -> Path:
    candidates = sorted(source_profile_dir.glob(f"*rank{rank_id}_*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"Template profile for rank {rank_id} not found in {source_profile_dir}"
        )
    return candidates[-1]


def _parse_trace_line_header(trace_line: str) -> str:
    return _require_match(r"rank:\d+:([a-zA-Z0-9_]+)\(", trace_line, "op_name")


def _parse_subop_template(subop_text: str) -> Dict[str, str]:
    func_name_match = re.search(r"func_name=([^,]+)", subop_text)
    return {
        "trace_src_func": _require_match(r"trace_src_func=([^,]+)", subop_text, "trace_src_func"),
        "input_shape": _require_match(r"input__shape=(\[[^\]]+\]|None)", subop_text, "input__shape"),
        "input_dtype": _require_match(r"input__dtype=([^,]+)", subop_text, "input__dtype"),
        "func_name": func_name_match.group(1) if func_name_match else "",
        "group": _require_match(r"group=([^,]+)", subop_text, "group"),
        "comm_func": _require_match(r"comm_func=([^,]+)", subop_text, "comm_func"),
    }


def _load_stage_template(template_file: Path) -> Dict[str, object]:
    lines = [line.strip() for line in template_file.read_text().splitlines() if line.strip()]
    op_lines: Dict[str, str] = {}
    for line in lines:
        op_name = _parse_trace_line_header(line)
        if op_name not in op_lines:
            op_lines[op_name] = line

    required = {"forward_step", "backward_step", "optimizer_step", "dp_allreduce"}
    missing = sorted(required - set(op_lines.keys()))
    if missing:
        raise ValueError(f"Missing required ops {missing} in template file: {template_file}")

    op_duration: Dict[str, float] = {}
    for op_name in ("get_batch", "forward_step", "backward_step", "loss_func", "optimizer_step"):
        if op_name in op_lines:
            op_duration[op_name] = _extract_numeric(op_lines[op_name], "duration")

    subop_templates: Dict[str, Dict[str, str]] = {}
    for op_name in ("forward_step", "backward_step", "loss_func"):
        if op_name not in op_lines:
            continue
        subops = _extract_suboperations_list(op_lines[op_name])
        if not subops:
            continue
        subop_templates[op_name] = _parse_subop_template(subops[0])

    if "forward_step" not in subop_templates or "backward_step" not in subop_templates:
        raise ValueError(f"forward/backward sub_operations cannot be empty: {template_file}")

    tensor_meta: Dict[str, Dict[str, str]] = {}
    for op_name in ("dp_allreduce", "ep_allreduce", "exp_dp_allreduce", "ep_dp_allreduce"):
        if op_name in op_lines:
            tensor_meta[op_name] = {
                "shape": _extract_shape_literal(op_lines[op_name], "input__shape"),
                "dtype": _extract_scalar_literal(op_lines[op_name], "input__dtype"),
            }

    return {
        "op_duration": op_duration,
        "subop_templates": subop_templates,
        "tensor_meta": tensor_meta,
    }


def _find_stage_schedule_file(schedule_dir: Path, stage_id: int) -> Path:
    candidates = sorted(schedule_dir.glob(f"stage{stage_id}_*_scheduling_plan.txt"))
    if not candidates:
        raise FileNotFoundError(f"Scheduling file for stage {stage_id} not found in {schedule_dir}")
    return candidates[-1]


def _parse_schedule_line(line: str) -> Dict[str, object]:
    line = line.strip()
    header = re.match(r"^stage:(\d+):([a-zA-Z0-9_]+)\((.*)\)$", line)
    if not header:
        raise ValueError(f"Invalid schedule line: {line}")
    stage_id = int(header.group(1))
    op_name = header.group(2)
    body = header.group(3)

    batch_id = int(_require_match(r"batch_id=(\d+)", body, "batch_id"))
    mg_state = _require_match(r"mg_state=([^,]+)", body, "mg_state")
    group_kind_match = re.search(r"group_kind=([^,]+)", body)
    group_kind = group_kind_match.group(1) if group_kind_match else "None"
    shape_match = re.search(r"input__shape=(\[[^\]]*\]|None)", body)
    input_shape = shape_match.group(1) if shape_match else "None"
    dtype_match = re.search(r"input__dtype=([^,\)]+)", body)
    input_dtype = dtype_match.group(1) if dtype_match else "None"

    return {
        "stage_id": stage_id,
        "op_name": op_name,
        "batch_id": batch_id,
        "mg_state": mg_state,
        "group_kind": group_kind,
        "input_shape": input_shape,
        "input_dtype": input_dtype,
    }


def _load_stage_schedule(schedule_file: Path, expected_stage_id: int) -> List[Dict[str, object]]:
    lines = [line.strip() for line in schedule_file.read_text().splitlines() if line.strip()]
    parsed = [_parse_schedule_line(line) for line in lines]
    for op in parsed:
        if int(op["stage_id"]) != expected_stage_id:
            raise ValueError(
                f"Stage mismatch in {schedule_file}: expected {expected_stage_id}, got {op['stage_id']}"
            )
    return parsed


def _rank_stage(rank_id: int, tp_size: int, dp_size: int, pp_size: int) -> int:
    stage = rank_id // (tp_size * dp_size)
    if stage < 0 or stage >= pp_size:
        raise ValueError(f"Invalid stage computed for rank {rank_id}: stage={stage}")
    return stage


def _comp_base_duration(template: Dict[str, object], op_name: str) -> float:
    op_duration = template["op_duration"]
    value = op_duration.get(op_name)
    if value is None or float(value) <= 0:
        value = DEFAULT_COMP_DURATION_MS.get(op_name)
    if value is None or float(value) <= 0:
        raise ValueError(f"Missing comp duration template for op {op_name}")
    return float(value)


def _comm_weight(op_name: str) -> float:
    return float(COMM_WEIGHT_BY_OP.get(op_name, 1.0))


def _allocate_stage_durations(
    schedule_ops: List[Dict[str, object]],
    template: Dict[str, object],
    target_comp_ms: float,
    target_comm_ms: float,
) -> Dict[int, float]:
    if target_comp_ms <= 0 or target_comm_ms <= 0:
        raise ValueError(
            f"Non-positive stage targets: comp={target_comp_ms}, comm={target_comm_ms}"
        )

    split_count = 0
    comp_weights: Dict[int, float] = {}
    comm_weights: Dict[int, float] = {}

    for idx, op in enumerate(schedule_ops):
        op_name = str(op["op_name"])
        if op_name in MG_COMP_OPS:
            comp_weights[idx] = _comp_base_duration(template, op_name)
            if op_name in SPLIT_OPS:
                if op_name not in template["subop_templates"]:
                    raise ValueError(f"Missing subop template for split op {op_name}")
                split_count += 1
        else:
            comm_weights[idx] = _comm_weight(op_name)

    if not comp_weights:
        raise ValueError("No comp operations found in schedule stage.")
    if not comm_weights:
        raise ValueError("No comm operations found in schedule stage.")

    comp_budget = target_comp_ms - split_count * PARENT_SPLIT_OVERHEAD_MS
    if comp_budget <= 0:
        raise ValueError(
            f"Comp budget non-positive after split overhead: target_comp={target_comp_ms}, "
            f"split_count={split_count}, budget={comp_budget}"
        )

    comp_weight_sum = sum(comp_weights.values())
    comm_weight_sum = sum(comm_weights.values())

    durations: Dict[int, float] = {}
    for idx, weight in comp_weights.items():
        durations[idx] = max(MIN_DURATION_MS, comp_budget * weight / comp_weight_sum)
    for idx, weight in comm_weights.items():
        durations[idx] = max(MIN_DURATION_MS, target_comm_ms * weight / comm_weight_sum)
    return durations


def _build_suboperation_text(
    subop_template: Dict[str, str],
    op_start_ts: float,
    op_end_ts: float,
) -> str:
    midpoint = op_start_ts + (op_end_ts - op_start_ts) * 0.5
    fields = [
        f"trace_src_func={subop_template['trace_src_func']}",
        "duration=0.000000",
        f"timestamp={midpoint:.6f}",
        f"input__shape={subop_template['input_shape']}",
        f"input__dtype={subop_template['input_dtype']}",
    ]
    if subop_template.get("func_name"):
        fields.append(f"func_name={subop_template['func_name']}")
    fields.extend(
        [
            f"group={subop_template['group']}",
            f"comm_func={subop_template['comm_func']}",
        ]
    )
    return ",".join(fields)


def _render_trace_line(
    rank_id: int,
    stage_id: int,
    op: Dict[str, object],
    duration_ms: float,
    timestamp_ms: float,
    template: Dict[str, object],
) -> str:
    op_name = str(op["op_name"])
    batch_id = int(op["batch_id"])
    mg_state = str(op["mg_state"])
    group_kind = str(op["group_kind"])

    input_shape = str(op["input_shape"])
    input_dtype = str(op["input_dtype"])
    if input_shape == "None" or input_dtype == "None":
        fallback = template["tensor_meta"].get(op_name)
        if fallback:
            if input_shape == "None":
                input_shape = str(fallback["shape"])
            if input_dtype == "None":
                input_dtype = str(fallback["dtype"])

    sub_operations: List[str] = []
    if op_name in SPLIT_OPS:
        op_start = timestamp_ms - duration_ms
        sub_template = template["subop_templates"].get(op_name)
        if sub_template is None:
            raise ValueError(f"Missing subop template for split op {op_name}")
        sub_operations = [_build_suboperation_text(sub_template, op_start, timestamp_ms)]

    return (
        f"rank:{rank_id}:{op_name}("
        f"stage_id={stage_id},"
        f"batch_id={batch_id},"
        f"mg_state={mg_state},"
        f"duration={duration_ms:.6f},"
        f"description=reconstructed_groundtruth,"
        f"group_kind={group_kind},"
        f"input__shape={input_shape},"
        f"input__dtype={input_dtype},"
        f"timestamp={timestamp_ms:.6f},"
        f"sub_operations={repr(sub_operations)})"
    )


def _build_rank_trace_lines(
    rank_id: int,
    stage_id: int,
    schedule_ops: List[Dict[str, object]],
    op_duration_map: Dict[int, float],
    template: Dict[str, object],
) -> List[str]:
    lines: List[str] = []
    current_ts = 1_000_000.0 + rank_id * 10_000.0

    for idx, op in enumerate(schedule_ops):
        duration = float(op_duration_map[idx])
        current_ts += duration
        lines.append(
            _render_trace_line(
                rank_id=rank_id,
                stage_id=stage_id,
                op=op,
                duration_ms=duration,
                timestamp_ms=current_ts,
                template=template,
            )
        )

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct WS256 schedule-aligned trace files")
    parser.add_argument(
        "--baseline-metrics-json",
        default=(
            "task_memory/task_2026-03-04_ws256_dense_simulation_phase/logs/"
            "ws256_dense_simulation_metrics_20260304_140456.json"
        ),
    )
    parser.add_argument(
        "--source-profile-dir",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_256gpus_gpt175b_tp8_pp16_dp2/database_profile"
        ),
    )
    parser.add_argument(
        "--schedule-dir",
        default=(
            "megatron-sim-engine/simulation_inputs/megatron_operation_log/"
            "h800_256gpus_gpt175b_tp8_pp16_dp2/schedule"
        ),
    )
    parser.add_argument(
        "--output-trace-dir",
        default="task_memory/task_2026-03-04_reverse_groundtruth/reconstructed_traces",
    )
    parser.add_argument(
        "--solution-json",
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution.json",
    )
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument("--pp-size", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--overall-error-pct", type=float, default=-10.3)
    parser.add_argument("--comp-error-pct", type=float, default=-1.75)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    baseline_metrics_path = Path(args.baseline_metrics_json)
    source_profile_dir = Path(args.source_profile_dir)
    schedule_dir = Path(args.schedule_dir)
    output_trace_dir = Path(args.output_trace_dir)
    solution_json_path = Path(args.solution_json)

    if not baseline_metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")
    if not source_profile_dir.exists():
        raise FileNotFoundError(f"Source profile dir not found: {source_profile_dir}")
    if not schedule_dir.exists():
        raise FileNotFoundError(f"Schedule dir not found: {schedule_dir}")

    metrics = json.loads(baseline_metrics_path.read_text())
    rank_summary = metrics.get("rank_summary")
    if not isinstance(rank_summary, list) or not rank_summary:
        raise ValueError("baseline metrics missing non-empty rank_summary")

    stage_sim: Dict[int, Dict[str, float]] = {}
    critical = max(rank_summary, key=lambda r: float(r["sum_ms"]))
    for row in rank_summary:
        stage_id = int(row["stage_id"])
        if stage_id in stage_sim:
            continue
        stage_sim[stage_id] = {
            "comp_ms": float(row["comp_ms"]),
            "comm_ms": float(row["comm_ms"]),
            "sum_ms": float(row["sum_ms"]),
            "wrank": int(row["wrank"]),
        }
    if len(stage_sim) != args.pp_size:
        raise ValueError(f"Expected {args.pp_size} stages in baseline rank_summary, got {len(stage_sim)}")

    overall_error = args.overall_error_pct / 100.0
    comp_error = args.comp_error_pct / 100.0

    sim_e2e = float(critical["sum_ms"])
    sim_comp = float(critical["comp_ms"])
    sim_comm = float(critical["comm_ms"])

    comp_scale = 1.0 / (1.0 + comp_error)
    gt_e2e = sim_e2e / (1.0 + overall_error)
    gt_comp_critical = sim_comp * comp_scale
    gt_comm_critical = gt_e2e - gt_comp_critical
    if gt_comm_critical <= 0:
        raise ValueError(f"Solved gt_comm is non-positive: {gt_comm_critical}")
    comm_scale = gt_comm_critical / sim_comm

    stage_templates: Dict[int, Dict[str, object]] = {}
    stage_schedules: Dict[int, List[Dict[str, object]]] = {}
    stage_schedule_files: Dict[int, str] = {}
    for stage_id in range(args.pp_size):
        representative_rank = stage_id * args.tp_size * args.dp_size
        template_file = _find_template_file(source_profile_dir, representative_rank)
        stage_templates[stage_id] = _load_stage_template(template_file)

        schedule_file = _find_stage_schedule_file(schedule_dir, stage_id)
        stage_schedule_files[stage_id] = str(schedule_file)
        stage_schedules[stage_id] = _load_stage_schedule(schedule_file, stage_id)

    stage_duration_plans: Dict[int, Dict[int, float]] = {}
    for stage_id in range(args.pp_size):
        target_comp = stage_sim[stage_id]["comp_ms"] * comp_scale
        target_comm = stage_sim[stage_id]["comm_ms"] * comm_scale
        stage_duration_plans[stage_id] = _allocate_stage_durations(
            schedule_ops=stage_schedules[stage_id],
            template=stage_templates[stage_id],
            target_comp_ms=target_comp,
            target_comm_ms=target_comm,
        )

    output_trace_dir.mkdir(parents=True, exist_ok=True)
    existing_txt = sorted(output_trace_dir.glob("*.txt"))
    if existing_txt and not args.allow_overwrite:
        raise ValueError(
            f"Output trace dir is not empty ({len(existing_txt)} txt files): {output_trace_dir}. "
            "Please archive or remove existing files explicitly before reconstruction."
        )

    for rank_id in range(args.world_size):
        stage_id = _rank_stage(rank_id, args.tp_size, args.dp_size, args.pp_size)
        lines = _build_rank_trace_lines(
            rank_id=rank_id,
            stage_id=stage_id,
            schedule_ops=stage_schedules[stage_id],
            op_duration_map=stage_duration_plans[stage_id],
            template=stage_templates[stage_id],
        )
        output_file = output_trace_dir / (
            f"wd256_tp8_pp16_exp1_expNumNone_numl96_bs1_rank{rank_id}_reconstructed.txt"
        )
        output_file.write_text("\n".join(lines) + "\n")

    stage_schedule_line_counts = {
        str(stage_id): len(stage_schedules[stage_id]) for stage_id in range(args.pp_size)
    }
    output_stage_line_counts: Dict[str, int] = {}
    for stage_id in range(args.pp_size):
        representative_rank = stage_id * args.tp_size * args.dp_size
        file_path = output_trace_dir / (
            f"wd256_tp8_pp16_exp1_expNumNone_numl96_bs1_rank{representative_rank}_reconstructed.txt"
        )
        output_stage_line_counts[str(stage_id)] = len(
            [line for line in file_path.read_text().splitlines() if line.strip()]
        )

    solution = {
        "error_definition": "(simulation - ground_truth) / ground_truth",
        "inputs": {
            "overall_error_pct": args.overall_error_pct,
            "comp_error_pct": args.comp_error_pct,
            "critical_rank": int(critical["wrank"]),
            "sim_e2e_ms": sim_e2e,
            "sim_comp_ms": sim_comp,
            "sim_comm_ms": sim_comm,
        },
        "ground_truth": {
            "gt_e2e_ms": gt_e2e,
            "gt_comp_ms": gt_comp_critical,
            "gt_comm_ms": gt_comm_critical,
        },
        "scales": {
            "comp_scale_gt_over_sim": comp_scale,
            "comm_scale_gt_over_sim": comm_scale,
        },
        "derived_comm_error_pct": (sim_comm - gt_comm_critical) / gt_comm_critical * 100.0,
        "schedule_aligned": True,
        "schedule_dir": str(schedule_dir),
        "stage_schedule_files": stage_schedule_files,
        "stage_schedule_line_counts": stage_schedule_line_counts,
        "output_stage_line_counts": output_stage_line_counts,
        "output_trace_dir": str(output_trace_dir),
        "output_trace_file_count": len(list(output_trace_dir.glob("*.txt"))),
    }
    solution_json_path.parent.mkdir(parents=True, exist_ok=True)
    solution_json_path.write_text(json.dumps(solution, indent=2))

    print(json.dumps(solution, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
