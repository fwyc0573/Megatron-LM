"""Command-line entry point for Megatron simulation engine.

This module standardizes simulator execution with explicit CLI arguments while
keeping backward compatibility through optional presets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.static_graphs.parallel_group_manager import MPUInfo, ParallelGroupManager
from src.core.static_graphs.rank_manager import RankManager
from src.core.cc_backend import list_cc_backends
from src.core.comm_sim import nccl_comm
from src.core.simu_engine import MODE_MODEL, MODE_PROFILE, MODE_SIMULATE, RUNNING_MODE_OPTION
from src.core.simu_engine import SimulatorEngine
from src.core.simulator_config import OverlapConfig, SlowdownConfig


RUN_MODE_MAP = {
    "simulate": MODE_SIMULATE,
    "profile": MODE_PROFILE,
    "model": MODE_MODEL,
}

AE_REPORT_MODELS = {"gpt175b", "qwen3_a30b", "dsv3"}
AE_ARTIFACT_SOURCES = {"fresh", "prebaked"}
RANK0_REPORT_SCHEMA_VERSION = "sc26-ae-rank0-report-v1"


@dataclass
class CliConfig:
    framework: str
    mode: str
    trace_dir: Optional[str]
    schedule_dir: Optional[str]
    database_dir: Optional[str]
    world_size: int
    pp_size: int
    tp_size: int
    exp_size: int
    local_size: int
    strategy: str
    visualize_rank_start: int
    visualize_rank_end: Optional[int]
    no_visualize: bool
    cc_backend: str
    cc_backend_options: dict
    slowdown: SlowdownConfig
    overlap: OverlapConfig
    report_output_dir: Optional[str]
    report_model: Optional[str]
    artifact_source: Optional[str]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    backend_choices = list(list_cc_backends())

    parser = argparse.ArgumentParser(
        description="Run distributed training simulation with explicit topology/config inputs."
    )

    # Legacy compatibility flags (currently informational only).
    parser.add_argument(
        "--skip-coverage",
        dest="skip_coverage",
        action="store_true",
        help="Compatibility flag from legacy entry; currently unused.",
    )
    parser.add_argument(
        "--skip-accuracy",
        dest="skip_accuracy",
        action="store_true",
        help="Compatibility flag from legacy entry; currently unused.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        dest="config",
        default=None,
        help="Compatibility config path from legacy entry; currently unused.",
    )

    parser.add_argument(
        "--preset",
        choices=["moe_tiny_2pp_1tp_2dp"],
        default=None,
        help="Use a built-in preset for quick smoke testing.",
    )
    parser.add_argument(
        "--framework",
        choices=["megatron-lm", "deepspeed"],
        default=None,
        help="Framework trace/schema type.",
    )
    parser.add_argument(
        "--mode",
        choices=list(RUN_MODE_MAP.keys()),
        default=None,
        help="Execution mode.",
    )
    parser.add_argument("--trace-dir", default=None, help="Trace directory path.")
    parser.add_argument(
        "--schedule-dir",
        default=None,
        help="Scheduling plan directory path.",
    )
    parser.add_argument(
        "--database-dir",
        default=None,
        help="Single-GPU op database directory path.",
    )

    parser.add_argument("--world-size", type=int, default=None, help="Global world size.")
    parser.add_argument("--pp-size", type=int, default=None, help="Pipeline parallel size.")
    parser.add_argument("--tp-size", type=int, default=None, help="Tensor parallel size.")
    parser.add_argument(
        "--exp-size",
        type=int,
        default=1,
        help="Expert parallel size (default: 1).",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=None,
        help="GPUs per node used for rank mapping.",
    )
    parser.add_argument(
        "--strategy",
        default="1F1B-none_interleaved",
        help="Pipeline scheduling strategy.",
    )

    parser.add_argument(
        "--visualize-rank-start",
        type=int,
        default=0,
        help="Visualization rank range start (inclusive).",
    )
    parser.add_argument(
        "--visualize-rank-end",
        type=int,
        default=None,
        help="Visualization rank range end (exclusive). Defaults to world_size.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip timeline visualization.",
    )
    parser.add_argument(
        "--cc-backend",
        choices=backend_choices,
        default="collective-sim",
        help="Communication prediction backend.",
    )
    parser.add_argument(
        "--cc-backend-options-json",
        default=None,
        help="JSON object merged into selected backend options.",
    )
    parser.add_argument(
        "--collective-sim-repo-root",
        default=None,
        help=(
            "Path to collective-sim repository root (when --cc-backend collective-sim). "
            "Default: src/core/cc_backend/collective-sim"
        ),
    )
    parser.add_argument(
        "--enable-slowdown",
        action="store_true",
        help="Enable DDP backward slowdown prediction in simulate mode.",
    )
    parser.add_argument(
        "--slowdown-assets-dir",
        default=None,
        help="Directory containing manifest.json, kernel_features.json, and backward_kernel_blueprints.json.",
    )
    parser.add_argument(
        "--slowdown-model-path",
        default=None,
        help="Optional override for the slowdown XGBoost model path. Defaults to assets manifest model_path.",
    )
    parser.add_argument(
        "--slowdown-scaler-path",
        default=None,
        help="Optional override for the persisted slowdown scaler spec path. Defaults to assets manifest scaler_path.",
    )
    parser.add_argument(
        "--slowdown-max-iters",
        type=int,
        default=50,
        help="Maximum fixed-point iterations per kernel when slowdown is enabled.",
    )
    parser.add_argument(
        "--slowdown-tol-ms",
        type=float,
        default=1e-6,
        help="Fixed-point convergence tolerance in milliseconds for slowdown solving.",
    )
    parser.add_argument(
        "--overlap-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Trace-driven overlap policy. 'auto' enables overlap when the trace contains "
            "DDP overlap metadata, 'on' requires that metadata, and 'off' rejects it."
        ),
    )
    parser.add_argument(
        "--report-output-dir",
        default=None,
        help="Directory for SC26 AE rank0 report.json and report.md.",
    )
    parser.add_argument(
        "--report-model",
        choices=sorted(AE_REPORT_MODELS),
        default=None,
        help="SC26 AE model label for the rank0 report.",
    )
    parser.add_argument(
        "--artifact-source",
        choices=sorted(AE_ARTIFACT_SOURCES),
        default=None,
        help="Explicit SC26 AE artifact source recorded in the rank0 report.",
    )

    return parser.parse_args(argv)


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset != "moe_tiny_2pp_1tp_2dp":
        return args

    preset_values = {
        "framework": "megatron-lm",
        "mode": "simulate",
        "trace_dir": None,
        "schedule_dir": "simulation_inputs/megatron_operation_log/moe_tiny_2pp_1tp_2dp/schedule",
        "database_dir": "simulation_inputs/megatron_operation_log/moe_tiny_2pp_1tp_2dp/database_profile",
        "world_size": 4,
        "pp_size": 2,
        "tp_size": 1,
        "exp_size": 1,
        "local_size": 4,
    }

    for key, value in preset_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def _fail_fast_validate(args: argparse.Namespace) -> CliConfig:
    args = _apply_preset(args)
    cc_backend_options = {}

    if args.cc_backend_options_json is not None:
        try:
            cc_backend_options = json.loads(args.cc_backend_options_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--cc-backend-options-json must be valid JSON") from exc
        if not isinstance(cc_backend_options, dict):
            raise ValueError("--cc-backend-options-json must decode to a JSON object")

    if args.collective_sim_repo_root:
        if args.cc_backend == "collective-sim" and "collective-sim" not in cc_backend_options:
            cc_backend_options["repo_root"] = args.collective_sim_repo_root
        else:
            collective_sim_opts = cc_backend_options.setdefault("collective-sim", {})
            if not isinstance(collective_sim_opts, dict):
                raise ValueError(
                    "collective-sim backend options must be a JSON object when provided"
                )
            collective_sim_opts["repo_root"] = args.collective_sim_repo_root

    missing = []
    for field in ["framework", "mode", "database_dir", "world_size", "pp_size", "tp_size", "local_size"]:
        if getattr(args, field) is None:
            missing.append(field)

    if args.mode in {"simulate", "model"} and args.schedule_dir is None:
        missing.append("schedule_dir")
    if args.mode in {"profile", "model"} and args.trace_dir is None:
        # MODE_MODEL can still run without trace, but this project usually compares
        # with trace-backed data; keep this explicit to avoid hidden assumptions.
        if args.mode == "profile":
            missing.append("trace_dir")

    if args.enable_slowdown:
        if args.mode != "simulate":
            raise ValueError("Slowdown prediction is only supported in simulate mode.")
        if args.trace_dir is None:
            missing.append("trace_dir")

    if missing:
        missing_fields = ", ".join(sorted(set(missing)))
        raise ValueError(
            "Missing required arguments: "
            f"{missing_fields}. "
            "Use explicit flags or pass --preset moe_tiny_2pp_1tp_2dp for smoke testing."
        )

    for name in ["world_size", "pp_size", "tp_size", "exp_size", "local_size"]:
        value = getattr(args, name)
        if value is None or value <= 0:
            raise ValueError(
                f"Argument --{name.replace('_', '-')} must be a positive integer."
            )

    if args.mode not in RUN_MODE_MAP:
        raise ValueError(f"Unsupported mode {args.mode}. Available modes: {list(RUN_MODE_MAP)}")
    if RUN_MODE_MAP[args.mode] not in RUNNING_MODE_OPTION:
        raise ValueError(f"Internal mode mapping failed for {args.mode}")

    if args.world_size % args.local_size != 0:
        raise ValueError(
            f"world_size ({args.world_size}) must be divisible by local_size ({args.local_size})."
        )

    parallel_product = args.pp_size * args.tp_size * args.exp_size
    if args.world_size % parallel_product != 0:
        raise ValueError(
            "world_size must be divisible by pp_size * tp_size * exp_size. "
            f"Got world_size={args.world_size}, pp={args.pp_size}, tp={args.tp_size}, exp={args.exp_size}."
        )

    if args.visualize_rank_start < 0:
        raise ValueError("--visualize-rank-start must be >= 0")
    if args.visualize_rank_end is not None and args.visualize_rank_end <= args.visualize_rank_start:
        raise ValueError("--visualize-rank-end must be greater than --visualize-rank-start")
    if args.slowdown_max_iters <= 0:
        raise ValueError("--slowdown-max-iters must be > 0")
    if args.slowdown_tol_ms <= 0:
        raise ValueError("--slowdown-tol-ms must be > 0")

    report_values = [
        args.report_output_dir,
        args.report_model,
        args.artifact_source,
    ]
    if any(value is not None for value in report_values) and not all(
        value is not None for value in report_values
    ):
        raise ValueError(
            "--report-output-dir, --report-model, and --artifact-source must be provided together"
        )

    if args.cc_backend == "analytical" and args.local_size != nccl_comm.GPUS_PER_MACHINE:
        raise ValueError(
            "Analytical backend local-size mismatch: "
            f"--local-size={args.local_size}, "
            f"nccl_comm.GPUS_PER_MACHINE={nccl_comm.GPUS_PER_MACHINE}"
        )

    slowdown = SlowdownConfig(
        enabled=bool(args.enable_slowdown),
        assets_dir=args.slowdown_assets_dir,
        model_path=args.slowdown_model_path,
        scaler_path=args.slowdown_scaler_path,
        max_iters=args.slowdown_max_iters,
        tol_ms=args.slowdown_tol_ms,
    )
    overlap = OverlapConfig(mode=args.overlap_mode)

    return CliConfig(
        framework=args.framework,
        mode=args.mode,
        trace_dir=args.trace_dir,
        schedule_dir=args.schedule_dir,
        database_dir=args.database_dir,
        world_size=args.world_size,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        exp_size=args.exp_size,
        local_size=args.local_size,
        strategy=args.strategy,
        visualize_rank_start=args.visualize_rank_start,
        visualize_rank_end=args.visualize_rank_end,
        no_visualize=args.no_visualize,
        cc_backend=args.cc_backend,
        cc_backend_options=cc_backend_options,
        slowdown=slowdown,
        overlap=overlap,
        report_output_dir=args.report_output_dir,
        report_model=args.report_model,
        artifact_source=args.artifact_source,
    )


def _validated_duration(operation: object) -> float:
    name = getattr(operation, "name", "<unnamed>")
    join_time = getattr(operation, "join_time", None)
    finish_time = getattr(operation, "finish_time", None)
    if join_time is None or finish_time is None:
        raise ValueError(f"Operation {name} is missing a join_time or finish_time timestamp")
    if isinstance(join_time, bool) or isinstance(finish_time, bool):
        raise ValueError(f"Operation {name} timestamps must be numeric")
    try:
        join_time = float(join_time)
        finish_time = float(finish_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Operation {name} timestamps must be numeric") from exc
    if not math.isfinite(join_time) or not math.isfinite(finish_time):
        raise ValueError(f"Operation {name} timestamps must be finite")
    if finish_time < join_time:
        raise ValueError(
            f"Operation {name} finish_time must be greater than or equal to join_time"
        )
    return finish_time - join_time


def _validated_runtime_seconds(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite nonnegative number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite nonnegative number") from exc
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{field_name} must be a finite nonnegative number")
    return round(numeric_value, 6)


def build_rank0_report(
    simulator_engine: SimulatorEngine,
    model: str,
    artifact_source: str,
    load_time_s: float,
    execution_time_s: float,
) -> dict:
    """Build the SC26 AE report directly from the completed rank0 timeline."""
    if model not in AE_REPORT_MODELS:
        raise ValueError(f"Unsupported report model: {model}")
    if artifact_source not in AE_ARTIFACT_SOURCES:
        raise ValueError(f"Unsupported artifact_source: {artifact_source}")

    timeline_manager = getattr(simulator_engine, "timeline_manager", None)
    timelines = getattr(timeline_manager, "stages_timeline_process_dict", None)
    if not isinstance(timelines, dict) or 0 not in timelines:
        raise ValueError("Simulator timeline manager does not contain rank0")

    rank0 = timelines[0]
    final_timeline = list(getattr(rank0, "final_merge_timeline", []) or [])
    comp_timeline = list(getattr(rank0, "comp_timeline", []) or [])
    comm_timeline = list(getattr(rank0, "comm_timeline", []) or [])
    if not final_timeline:
        raise ValueError("rank0 final timeline is empty")

    for operation in final_timeline:
        _validated_duration(operation)
    final_start = min(float(operation.join_time) for operation in final_timeline)
    final_finish = max(float(operation.finish_time) for operation in final_timeline)
    step_time_ms = final_finish - final_start
    if not math.isfinite(step_time_ms) or step_time_ms <= 0:
        raise ValueError("rank0 timeline span must be finite and strictly positive")

    operation_sums = {}
    for operation_name in ("forward_step", "backward_step", "optimizer_step"):
        matching_operations = [
            operation for operation in comp_timeline if getattr(operation, "name", None) == operation_name
        ]
        if not matching_operations:
            raise ValueError(f"rank0 comp timeline is missing exact operation {operation_name}")
        operation_sums[operation_name] = sum(
            _validated_duration(operation) for operation in matching_operations
        )

    diagnostic_ms = sum(
        _validated_duration(operation) for operation in comp_timeline + comm_timeline
    )
    load_time = _validated_runtime_seconds(load_time_s, "load_time_s")
    execution_time = _validated_runtime_seconds(execution_time_s, "execution_time_s")

    return {
        "schema_version": RANK0_REPORT_SCHEMA_VERSION,
        "model": model,
        "artifact_source": artifact_source,
        "rank_id": 0,
        "rank0_step_time_ms": round(step_time_ms, 6),
        "rank0_forward_step_duration_sum_ms": round(operation_sums["forward_step"], 6),
        "rank0_backward_step_duration_sum_ms": round(operation_sums["backward_step"], 6),
        "rank0_optimizer_step_duration_sum_ms": round(operation_sums["optimizer_step"], 6),
        "rank0_comp_plus_comm_diagnostic_ms": round(diagnostic_ms, 6),
        "simulator_load_time_s": load_time,
        "simulator_execution_time_s": execution_time,
        "simulator_wall_clock_s": round(load_time + execution_time, 6),
    }


def write_rank0_report(report: dict, output_dir: Path) -> None:
    """Write matching JSON and Markdown renderings of a validated rank0 report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_lines = [
        "# SC26 AE Rank0 Simulation Report",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    markdown_lines.extend(f"| `{key}` | `{value}` |" for key, value in report.items())
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def run_simulation(config: CliConfig) -> dict:
    running_mode = RUN_MODE_MAP[config.mode]
    manager = ParallelGroupManager(
        local_size=config.local_size,
        world_size=config.world_size,
        pp_size=config.pp_size,
        tp_size=config.tp_size,
        exp_size=config.exp_size,
    )
    mpu_info: MPUInfo = manager.get_mpu_info()
    all_groups = manager.get_all_groups()

    print(
        f"[Config] framework={config.framework}, mode={config.mode}, "
        f"strategy={config.strategy}, cc_backend={config.cc_backend}"
    )
    print(f"[Topology] {mpu_info}")

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
    simulator_engine.simulator_config.overlap = config.overlap
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)

    time_load_start = time.time()
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator_engine.validate_global_placement_requirements()
    load_time = time.time() - time_load_start

    time_execution_start = time.time()
    simulator_engine.start_running()
    execution_time = time.time() - time_execution_start

    print(f"sim load time: {load_time:.6f}s")
    print(
        f"world_size: {config.world_size}, sim load time: {load_time:.6f}s, "
        f"sim execution time: {execution_time:.6f}s"
    )

    report = None
    if config.report_output_dir is not None:
        report = build_rank0_report(
            simulator_engine,
            model=config.report_model,
            artifact_source=config.artifact_source,
            load_time_s=load_time,
            execution_time_s=execution_time,
        )
        write_rank0_report(report, Path(config.report_output_dir))

    if not config.no_visualize:
        rank_end = config.visualize_rank_end if config.visualize_rank_end is not None else config.world_size
        simulator_engine.visualize_timelines(
            wrank_id_start_end=[config.visualize_rank_start, rank_end],
            output_dir="./log/visualization_outputs",
        )

    if report is not None:
        return {"world_size": config.world_size, **report}
    return {
        "world_size": config.world_size,
        "load_time": load_time,
        "execution_time": execution_time,
    }


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
        config = _fail_fast_validate(args)
        run_simulation(config)
    except Exception as exc:  # noqa: BLE001 - explicit fail-fast reporting for CLI users.
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
