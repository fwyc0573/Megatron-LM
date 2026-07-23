#!/usr/bin/env python3
"""Cross-validate communication cost backends on synthetic + trace-derived samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.cc_backend import CommunicationPredictionRequest, create_cc_backend
from src.core.cc_backend.op_mapping import infer_collective_kind, infer_domain_dims
from src.core.simulator_config import create_h800_sxm_ib_config
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager
from src.core.static_graphs.rank_manager import RankManager
from src.utils.message_size_calculator import calculate_comm_message_size


COMM_OPS = {
    "send_forward",
    "recv_forward",
    "send_backward",
    "recv_backward",
    "tp_allreduce",
    "dp_allreduce",
    "ep_allreduce",
    "exp_dp_allreduce",
    "ep_dp_allreduce",
    "exp_all_to_all",
    "exp_allgather",
    "tp_allgather",
    "tp_reducescatter",
    "cp_allgather",
    "cp_reducescatter",
}

TOP_LEVEL_PATTERN = re.compile(r"^rank:(?P<rank>\d+):(?P<op>[a-zA-Z0-9_]+)\((?P<body>.*)\)$")
GROUP_KIND_PATTERN = re.compile(r"group_kind=([^,]+)")
SHAPE_PATTERN = re.compile(r"input__shape=(.*?),input__dtype=")
DTYPE_PATTERN = re.compile(r"input__dtype=([^,]+)")


@dataclass
class Sample:
    source: str
    op_name: str
    group_kind: str
    rank_id: int
    comm_group: Tuple[int, ...]
    data_size_bytes: int
    tensor_shape: Optional[str]
    tensor_dtype: Optional[str]
    note: str
    mpu_info: Optional[object] = None


@dataclass
class ResultRow:
    source: str
    op_name: str
    collective_kind: str
    group_kind: str
    domain_dims: Tuple[str, ...]
    group_size: int
    data_size_bytes: int
    analytical_ms: Optional[float]
    collective_sim_ms: Optional[float]
    placement_aware_ms: Optional[float]
    baseline_ms: Optional[float]
    delta_ms: Optional[float]
    delta_pct: Optional[float]
    ratio_collective_over_analytical: Optional[float]
    abs_pct_diff: Optional[float]
    status: str
    note: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate analytical and collective-sim communication predictors."
    )
    parser.add_argument(
        "--trace-dir",
        default="simulation_inputs/megatron_operation_log/moe_6.7b_2pp_1tp_2dp/global_ranks_profile",
        help="Trace directory used to extract communication samples.",
    )
    parser.add_argument(
        "--max-trace-samples",
        type=int,
        default=24,
        help="Maximum number of communication samples extracted from trace files.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="World size for group reconstruction.",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=2,
        help="Pipeline parallel degree.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel degree.",
    )
    parser.add_argument(
        "--exp-size",
        type=int,
        default=1,
        help="Expert parallel degree (for topology manager).",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=4,
        help="GPUs per node for rank mapping.",
    )
    parser.add_argument(
        "--collective-sim-repo-root",
        default="src/core/cc_backend/collective-sim",
        help="collective-sim repo root path.",
    )
    parser.add_argument(
        "--pp-domain-dim",
        default="DP",
        help=(
            "Explicit surrogate domain dim for pipeline p2p operations. "
            "Required because PP is not a native collective-sim dimension."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path with per-sample results.",
    )
    parser.add_argument(
        "--ab-csv-out",
        default=None,
        help=(
            "Optional CSV output for placement-aware on/off A/B charting. "
            "Columns: scenario, collective, placement_aware_latency_ms, baseline_latency_ms, delta_ms, delta_pct."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=(
            "task_memory/task_2026-02-27_sim_restructure/"
            "test_report_2026-02-28_cc_backend_cross_validation.md"
        ),
        help="Markdown report output path.",
    )
    return parser.parse_args()


def _build_rank_zoos(
    world_size: int,
    pp_size: int,
    tp_size: int,
    exp_size: int,
    local_size: int,
) -> Tuple[Dict[int, object], object]:
    manager = ParallelGroupManager(
        local_size=local_size,
        world_size=world_size,
        pp_size=pp_size,
        tp_size=tp_size,
        exp_size=exp_size,
    )
    mpu_info = manager.get_mpu_info()
    rank_manager = RankManager(
        mpu_info=mpu_info,
        gpus_per_node=local_size,
        all_groups=manager.get_all_groups(),
    )
    return rank_manager.get_rank_zoos(), mpu_info


def _peer_for_pp(op_name: str, rank_id: int, rank_zoo: object) -> Optional[int]:
    if "send_forward" in op_name or "recv_backward" in op_name:
        return rank_zoo._get_pp_next_world_rank()
    if "recv_forward" in op_name or "send_backward" in op_name:
        return rank_zoo._get_pp_previous_world_rank()
    return None


def _group_for_operation(
    op_name: str,
    group_kind: str,
    rank_id: int,
    rank_zoos: Dict[int, object],
) -> Tuple[int, ...]:
    rank_zoo = rank_zoos[rank_id]
    normalized = group_kind.lower()

    if normalized == "pp":
        peer = _peer_for_pp(op_name, rank_id, rank_zoo)
        if peer is None:
            raise ValueError(f"Cannot infer PP peer for op={op_name}, rank={rank_id}")
        return tuple(sorted({rank_id, peer}))
    if normalized == "dp":
        return tuple(rank_zoo.dp_groups or [])
    if normalized == "tp":
        return tuple(rank_zoo.tp_groups or [])
    if normalized == "ep":
        return tuple(rank_zoo.ep_groups or [])
    if normalized == "cp":
        return tuple(rank_zoo.cp_groups or [])
    if normalized in {"exp", "exp_dp"}:
        if normalized == "exp_dp" and rank_zoo.dp_modulo_exp_groups:
            return tuple(rank_zoo.dp_modulo_exp_groups)
        return tuple(rank_zoo.exp_groups or [])
    raise ValueError(f"Unsupported group_kind={group_kind} for op={op_name}")


def _extract_comm_samples_from_trace(
    trace_dir: Path,
    rank_zoos: Dict[int, object],
    mpu_info: object,
    max_samples: int,
) -> List[Sample]:
    samples: List[Sample] = []
    if not trace_dir.exists():
        raise FileNotFoundError(f"trace_dir not found: {trace_dir}")

    for trace_file in sorted(trace_dir.glob("*.txt")):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            match = TOP_LEVEL_PATTERN.match(line)
            if match is None:
                continue

            op_name = match.group("op").strip()
            if op_name not in COMM_OPS:
                continue

            rank_id = int(match.group("rank"))
            body = match.group("body")

            group_match = GROUP_KIND_PATTERN.search(body)
            group_kind = group_match.group(1).strip() if group_match else ""
            if not group_kind:
                continue

            shape_match = SHAPE_PATTERN.search(body)
            dtype_match = DTYPE_PATTERN.search(body)
            if shape_match is None or dtype_match is None:
                continue

            tensor_shape = shape_match.group(1).strip()
            tensor_dtype = dtype_match.group(1).strip()
            if tensor_shape in {"None", "[]"} or tensor_dtype == "None":
                continue

            comm_group = _group_for_operation(op_name, group_kind, rank_id, rank_zoos)
            if len(comm_group) < 2:
                continue

            data_size_bytes = int(
                calculate_comm_message_size(tensor_shape, tensor_dtype, op_name, len(comm_group))
            )
            if data_size_bytes <= 0:
                continue

            samples.append(
                Sample(
                    source="trace",
                    op_name=op_name,
                    group_kind=group_kind,
                    rank_id=rank_id,
                    comm_group=comm_group,
                    data_size_bytes=data_size_bytes,
                    tensor_shape=tensor_shape,
                    tensor_dtype=tensor_dtype,
                    note=f"trace_file={trace_file.name}",
                    mpu_info=mpu_info,
                )
            )
            if len(samples) >= max_samples:
                return samples
    return samples


def _build_synthetic_samples(mpu_info: object) -> List[Sample]:
    samples: List[Sample] = []

    dp_group = tuple(int(rank) for rank in (getattr(mpu_info, "dp_groups", [[0]])[0]))
    if len(dp_group) >= 2:
        samples.append(
            Sample(
                source="synthetic",
                op_name="dp_allreduce",
                group_kind="dp",
                rank_id=int(dp_group[0]),
                comm_group=dp_group,
                data_size_bytes=64 * 1024 * 1024,
                tensor_shape=None,
                tensor_dtype=None,
                note=f"DP allreduce group_size={len(dp_group)}",
                mpu_info=mpu_info,
            )
        )

    tp_group = tuple(int(rank) for rank in (getattr(mpu_info, "tp_groups", [[0]])[0]))
    if len(tp_group) >= 2:
        samples.append(
            Sample(
                source="synthetic",
                op_name="tp_allgather",
                group_kind="tp",
                rank_id=int(tp_group[0]),
                comm_group=tp_group,
                data_size_bytes=4 * 1024 * 1024,
                tensor_shape=None,
                tensor_dtype=None,
                note=f"TP allgather group_size={len(tp_group)}",
                mpu_info=mpu_info,
            )
        )
        samples.append(
            Sample(
                source="synthetic",
                op_name="tp_reducescatter",
                group_kind="tp",
                rank_id=int(tp_group[0]),
                comm_group=tp_group,
                data_size_bytes=4 * 1024 * 1024,
                tensor_shape=None,
                tensor_dtype=None,
                note=f"TP reducescatter group_size={len(tp_group)}",
                mpu_info=mpu_info,
            )
        )

    exp_group = tuple(int(rank) for rank in (getattr(mpu_info, "exp_groups", [[0]])[0]))
    if len(exp_group) >= 2:
        samples.append(
            Sample(
                source="synthetic",
                op_name="exp_all_to_all",
                group_kind="exp",
                rank_id=int(exp_group[0]),
                comm_group=exp_group,
                data_size_bytes=32 * 1024 * 1024,
                tensor_shape=None,
                tensor_dtype=None,
                note=f"EXP alltoall group_size={len(exp_group)}",
                mpu_info=mpu_info,
            )
        )

    pp_group = tuple(int(rank) for rank in (getattr(mpu_info, "pp_groups", [[0]])[0]))
    if len(pp_group) >= 2:
        pair = (int(pp_group[0]), int(pp_group[1]))
        samples.append(
            Sample(
                source="synthetic",
                op_name="send_forward",
                group_kind="pp",
                rank_id=pair[0],
                comm_group=pair,
                data_size_bytes=4 * 1024 * 1024,
                tensor_shape=None,
                tensor_dtype=None,
                note="PP p2p send_forward pair",
                mpu_info=mpu_info,
            )
        )

    return samples


def _evaluate_samples(
    samples: Sequence[Sample],
    collective_repo_root: Path,
    pp_domain_dim: str,
) -> List[ResultRow]:
    analytical = create_cc_backend("analytical", create_h800_sxm_ib_config())

    placement_cfg = create_h800_sxm_ib_config()
    placement_cfg.communication.backend_options.setdefault("collective-sim", {})
    placement_cfg.communication.backend_options["collective-sim"].update(
        {
            "repo_root": str(collective_repo_root),
            "pp_domain_dim": pp_domain_dim,
            "placement_mode": "global",
            "strict_mpu_alignment": True,
        }
    )
    collective_placement = create_cc_backend("collective-sim", placement_cfg)

    baseline_cfg = create_h800_sxm_ib_config()
    baseline_cfg.communication.backend_options.setdefault("collective-sim", {})
    baseline_cfg.communication.backend_options["collective-sim"].update(
        {
            "repo_root": str(collective_repo_root),
            "pp_domain_dim": pp_domain_dim,
            "placement_mode": "group_size",
            "strict_mpu_alignment": False,
        }
    )
    collective_baseline = create_cc_backend("collective-sim", baseline_cfg)

    results: List[ResultRow] = []
    for sample in samples:
        try:
            if sample.group_kind.lower() == "pp":
                normalized_pp_domain = str(pp_domain_dim).strip().upper()
                if not normalized_pp_domain:
                    raise ValueError(
                        "pp_domain_dim must be explicitly provided for pipeline p2p samples"
                    )
                domain_dims = (normalized_pp_domain,)
            else:
                domain_dims = infer_domain_dims(sample.group_kind, sample.op_name)
            collective_kind = infer_collective_kind(sample.op_name)
        except Exception as exc:  # noqa: BLE001
            results.append(
                ResultRow(
                    source=sample.source,
                    op_name=sample.op_name,
                    collective_kind="unknown",
                    group_kind=sample.group_kind,
                    domain_dims=(),
                    group_size=len(sample.comm_group),
                    data_size_bytes=sample.data_size_bytes,
                    analytical_ms=None,
                    collective_sim_ms=None,
                    placement_aware_ms=None,
                    baseline_ms=None,
                    delta_ms=None,
                    delta_pct=None,
                    ratio_collective_over_analytical=None,
                    abs_pct_diff=None,
                    status="mapping_error",
                    note=f"{sample.note}; mapping_error={exc}",
                )
            )
            continue

        request = CommunicationPredictionRequest.from_raw(
            comm_group=sample.comm_group,
            op_name=sample.op_name,
            data_size_bytes=sample.data_size_bytes,
            group_kind=sample.group_kind,
            domain_dims=domain_dims,
            tensor_shape=sample.tensor_shape,
            tensor_dtype=sample.tensor_dtype,
            mpu_info=sample.mpu_info,
        )

        analytical_ms: Optional[float] = None
        placement_aware_ms: Optional[float] = None
        baseline_ms: Optional[float] = None
        status_parts: List[str] = []
        note = sample.note

        try:
            analytical_ms = float(analytical.predict(request))
        except Exception as exc:  # noqa: BLE001
            status_parts.append("analytical_error")
            note = f"{note}; analytical_error={exc}"

        try:
            placement_aware_ms = float(collective_placement.predict(request))
        except Exception as exc:  # noqa: BLE001
            status_parts.append("placement_aware_error")
            note = f"{note}; placement_aware_error={exc}"

        try:
            baseline_ms = float(collective_baseline.predict(request))
        except Exception as exc:  # noqa: BLE001
            status_parts.append("baseline_error")
            note = f"{note}; baseline_error={exc}"

        status = "+".join(status_parts) if status_parts else "ok"

        ratio = None
        abs_pct_diff = None
        if analytical_ms is not None and placement_aware_ms is not None and analytical_ms > 0:
            ratio = placement_aware_ms / analytical_ms
            abs_pct_diff = abs(placement_aware_ms - analytical_ms) / analytical_ms * 100.0

        delta_ms = None
        delta_pct = None
        if placement_aware_ms is not None and baseline_ms is not None:
            delta_ms = placement_aware_ms - baseline_ms
            if baseline_ms != 0.0:
                delta_pct = (delta_ms / baseline_ms) * 100.0

        if (
            placement_aware_ms is not None
            and placement_aware_ms == 0.0
            and sample.data_size_bytes > 0
            and status == "ok"
        ):
            status = "semantic_risk_zero_collective"

        results.append(
            ResultRow(
                source=sample.source,
                op_name=sample.op_name,
                collective_kind=collective_kind,
                group_kind=sample.group_kind,
                domain_dims=domain_dims,
                group_size=len(sample.comm_group),
                data_size_bytes=sample.data_size_bytes,
                analytical_ms=analytical_ms,
                collective_sim_ms=placement_aware_ms,
                placement_aware_ms=placement_aware_ms,
                baseline_ms=baseline_ms,
                delta_ms=delta_ms,
                delta_pct=delta_pct,
                ratio_collective_over_analytical=ratio,
                abs_pct_diff=abs_pct_diff,
                status=status,
                note=note,
            )
        )
    return results


def _p90(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def _summarize(results: Sequence[ResultRow]) -> Dict[str, object]:
    ok_rows = [row for row in results if row.status == "ok" and row.abs_pct_diff is not None]
    diffs = [row.abs_pct_diff for row in ok_rows if row.abs_pct_diff is not None]
    ratios = [row.ratio_collective_over_analytical for row in ok_rows if row.ratio_collective_over_analytical is not None]
    ab_rows = [row for row in results if row.status == "ok" and row.delta_pct is not None]
    ab_deltas = [row.delta_pct for row in ab_rows if row.delta_pct is not None]
    status_hist: Dict[str, int] = {}
    for row in results:
        status_hist[row.status] = status_hist.get(row.status, 0) + 1

    by_collective: Dict[str, Dict[str, float]] = {}
    collectives = sorted({row.collective_kind for row in ok_rows})
    for name in collectives:
        rows = [row for row in ok_rows if row.collective_kind == name]
        row_diffs = [row.abs_pct_diff for row in rows if row.abs_pct_diff is not None]
        row_ratios = [
            row.ratio_collective_over_analytical
            for row in rows
            if row.ratio_collective_over_analytical is not None
        ]
        row_ab_deltas = [row.delta_pct for row in rows if row.delta_pct is not None]
        by_collective[name] = {
            "count": float(len(rows)),
            "median_abs_pct_diff": float(median(row_diffs)) if row_diffs else 0.0,
            "p90_abs_pct_diff": _p90(row_diffs),
            "median_ratio_collective_over_analytical": float(median(row_ratios)) if row_ratios else 0.0,
            "median_ab_delta_pct": float(median(row_ab_deltas)) if row_ab_deltas else 0.0,
        }

    return {
        "total_samples": len(results),
        "ok_samples": len(ok_rows),
        "ok_ab_samples": len(ab_rows),
        "status_histogram": status_hist,
        "median_abs_pct_diff": float(median(diffs)) if diffs else 0.0,
        "p90_abs_pct_diff": _p90(diffs),
        "max_abs_pct_diff": float(max(diffs)) if diffs else 0.0,
        "median_ratio_collective_over_analytical": float(median(ratios)) if ratios else 0.0,
        "median_ab_delta_pct": float(median(ab_deltas)) if ab_deltas else 0.0,
        "p90_ab_delta_pct": _p90(ab_deltas),
        "max_abs_ab_delta_pct": float(max((abs(v) for v in ab_deltas), default=0.0)),
        "by_collective_kind": by_collective,
    }


def _write_report(
    report_path: Path,
    args: argparse.Namespace,
    results: Sequence[ResultRow],
    summary: Dict[str, object],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    status_hist = summary["status_histogram"]
    risky_rows = [row for row in results if row.status != "ok"]
    no_zero_anomaly = status_hist.get("semantic_risk_zero_collective", 0) == 0

    lines = [
        "## Modification History",
        "",
        "| Date       | Summary of Changes |",
        "|------------|--------------------|",
        f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | Added CC backend cross-validation report for collective-sim default rollout |",
        "",
        "# Test Report: CC Backend Cross Validation",
        "",
        f"**Date**: {now_utc}",
        "",
        "## Test Script Information",
        f"- Script: `tests/performance/compare_cc_backends.py`",
        "- Commands:",
        "```bash",
        "python tests/performance/compare_cc_backends.py \\",
        f"  --trace-dir {args.trace_dir} \\",
        f"  --max-trace-samples {args.max_trace_samples} \\",
        f"  --world-size {args.world_size} --pp-size {args.pp_size} --tp-size {args.tp_size} --exp-size {args.exp_size} --local-size {args.local_size} \\",
        f"  --pp-domain-dim {args.pp_domain_dim} \\",
        f"  --collective-sim-repo-root {args.collective_sim_repo_root} \\",
        f"  --report-path {args.report_path}",
        "```",
        "",
        "## Validation Criteria",
        "- Op mapping should be resolvable for all sampled ops (`allreduce`/`alltoall`/`allgather`/`reducescatter`/`p2p`).",
        "- Domain mapping should be valid for sampled `group_kind` values (TP/DP/EP/CP plus explicit PP surrogate dim).",
        "- Non-zero payloads should not silently produce zero predicted time in collective-sim backend.",
        "- Backend invocation should return finite numeric duration or explicit fail-fast error.",
        "",
        "## Test Results",
        f"- Total samples: **{summary['total_samples']}** (synthetic + trace-derived)",
        f"- OK samples: **{summary['ok_samples']}**",
        f"- OK A/B samples (placement-aware on/off): **{summary['ok_ab_samples']}**",
        f"- Status histogram: `{status_hist}`",
        f"- Median abs diff (collective-sim vs analytical): **{summary['median_abs_pct_diff']:.2f}%**",
        f"- P90 abs diff: **{summary['p90_abs_pct_diff']:.2f}%**",
        f"- Max abs diff: **{summary['max_abs_pct_diff']:.2f}%**",
        f"- Median ratio (collective-sim/analytical): **{summary['median_ratio_collective_over_analytical']:.4f}**",
        f"- Median delta pct (placement-aware vs baseline): **{summary['median_ab_delta_pct']:.2f}%**",
        f"- P90 delta pct (placement-aware vs baseline): **{summary['p90_ab_delta_pct']:.2f}%**",
        f"- Max |delta pct| (placement-aware vs baseline): **{summary['max_abs_ab_delta_pct']:.2f}%**",
        "",
        "### Per Collective Kind",
    ]

    by_collective = summary["by_collective_kind"]
    if not by_collective:
        lines.append("- No successful rows were produced.")
    else:
        lines.extend(
            [
                "",
                "| Collective | Count | Median Abs Diff % | P90 Abs Diff % | Median Ratio | Median A/B Delta % |",
                "|------------|------:|------------------:|---------------:|-------------:|-------------------:|",
            ]
        )
        for name, values in sorted(by_collective.items()):
            lines.append(
                f"| {name} | {int(values['count'])} | "
                f"{values['median_abs_pct_diff']:.2f} | {values['p90_abs_pct_diff']:.2f} | "
                f"{values['median_ratio_collective_over_analytical']:.4f} | {values['median_ab_delta_pct']:.2f} |"
            )

    ab_rows = [
        row
        for row in results
        if row.placement_aware_ms is not None and row.baseline_ms is not None
    ]
    lines.extend(["", "### Placement-aware On/Off A/B (Chart-ready)"])
    if not ab_rows:
        lines.append("- No rows available for placement-aware A/B comparison.")
    else:
        lines.extend(
            [
                "",
                "| Scenario | Collective | Placement Aware Latency (ms) | Baseline Latency (ms) | Delta (ms) | Delta (%) |",
                "|----------|------------|-----------------------------:|----------------------:|-----------:|----------:|",
            ]
        )
        for row in ab_rows[:50]:
            delta_ms = row.delta_ms if row.delta_ms is not None else 0.0
            delta_pct = row.delta_pct if row.delta_pct is not None else 0.0
            lines.append(
                f"| {row.note[:48]} | {row.collective_kind} | {row.placement_aware_ms:.6f} | "
                f"{row.baseline_ms:.6f} | {delta_ms:.6f} | {delta_pct:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Evidence and Risk Check",
            f"- Non-zero payload -> zero collective-sim duration anomaly: **{'NOT FOUND' if no_zero_anomaly else 'FOUND'}**",
            f"- Non-OK rows: **{len(risky_rows)}**",
        ]
    )
    if risky_rows:
        lines.extend(
            [
                "",
                "| Source | Op | Group | Size (bytes) | Status | Note |",
                "|--------|----|-------|-------------:|--------|------|",
            ]
        )
        for row in risky_rows[:20]:
            lines.append(
                f"| {row.source} | {row.op_name} | {row.group_kind} | {row.data_size_bytes} | "
                f"{row.status} | {row.note[:120]} |"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "- Backend wiring is functional (both backends invoked on mixed sample set).",
            "- Numerical parity is not expected to be strict; the report focuses on semantic consistency and anomaly detection.",
            "- If anomalies appear in this report, fix op semantics or topology/domain mapping before claiming accuracy.",
            "",
            "## Coverage Gaps",
            "- This validation does not cover very large multi-node TP/EP layouts from production traces.",
            "- This validation does not include `broadcast` because collective-sim backend currently raises `NotImplementedError` for broadcast.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ab_csv(csv_path: Path, results: Sequence[ResultRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "collective",
                "placement_aware_latency_ms",
                "baseline_latency_ms",
                "delta_ms",
                "delta_pct",
                "op_name",
                "group_kind",
                "data_size_bytes",
                "status",
            ],
        )
        writer.writeheader()
        for row in results:
            if row.placement_aware_ms is None or row.baseline_ms is None:
                continue
            writer.writerow(
                {
                    "scenario": row.note,
                    "collective": row.collective_kind,
                    "placement_aware_latency_ms": f"{row.placement_aware_ms:.9f}",
                    "baseline_latency_ms": f"{row.baseline_ms:.9f}",
                    "delta_ms": f"{(row.delta_ms or 0.0):.9f}",
                    "delta_pct": f"{(row.delta_pct or 0.0):.6f}",
                    "op_name": row.op_name,
                    "group_kind": row.group_kind,
                    "data_size_bytes": row.data_size_bytes,
                    "status": row.status,
                }
            )


def main() -> int:
    args = _parse_args()
    trace_dir = Path(args.trace_dir)
    collective_repo_root = Path(args.collective_sim_repo_root)
    if not collective_repo_root.exists():
        raise FileNotFoundError(f"collective-sim repo root not found: {collective_repo_root}")

    rank_zoos, mpu_info = _build_rank_zoos(
        world_size=args.world_size,
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        exp_size=args.exp_size,
        local_size=args.local_size,
    )
    trace_samples = _extract_comm_samples_from_trace(
        trace_dir=trace_dir,
        rank_zoos=rank_zoos,
        mpu_info=mpu_info,
        max_samples=args.max_trace_samples,
    )
    synthetic_samples = _build_synthetic_samples(mpu_info)
    samples = synthetic_samples + trace_samples
    if not samples:
        raise RuntimeError("No communication samples were collected for cross validation.")

    results = _evaluate_samples(
        samples=samples,
        collective_repo_root=collective_repo_root,
        pp_domain_dim=args.pp_domain_dim,
    )
    summary = _summarize(results)

    if args.json_out:
        json_out_path = Path(args.json_out)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "summary": summary,
                    "results": [asdict(row) for row in results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    report_path = Path(args.report_path)
    _write_report(report_path=report_path, args=args, results=results, summary=summary)
    if args.ab_csv_out:
        _write_ab_csv(Path(args.ab_csv_out), results)

    print(f"[CC-CROSS-VALIDATION] samples={len(samples)} ok={summary['ok_samples']}")
    print(f"[CC-CROSS-VALIDATION] status_histogram={summary['status_histogram']}")
    print(f"[CC-CROSS-VALIDATION] report={report_path}")
    if args.json_out:
        print(f"[CC-CROSS-VALIDATION] json={args.json_out}")
    if args.ab_csv_out:
        print(f"[CC-CROSS-VALIDATION] ab_csv={args.ab_csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
