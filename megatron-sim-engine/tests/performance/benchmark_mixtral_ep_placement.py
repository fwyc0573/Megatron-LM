#!/usr/bin/env python3
"""Benchmark placement-aware vs baseline collective-sim on Mixtral/EP-heavy scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

from src.core.cc_backend import CommunicationPredictionRequest, create_cc_backend
from src.core.simulator_config import create_h800_sxm_ib_config
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    world_size: int
    pp_size: int
    tp_size: int
    exp_size: int
    local_size: int


@dataclass
class BenchmarkRow:
    scenario: str
    collective: str
    payload_bytes: int
    comm_group: Tuple[int, ...]
    placement_aware_latency_ms: float
    baseline_latency_ms: float
    delta_ms: float
    delta_pct: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Mixtral/EP-heavy communication with placement-aware vs baseline collective-sim."
    )
    parser.add_argument(
        "--collective-sim-repo-root",
        default="src/core/cc_backend/collective-sim",
        help="Path to collective-sim repository root.",
    )
    parser.add_argument(
        "--pp-domain-dim",
        default="DP",
        help="Pipeline p2p surrogate domain dim used by collective-sim.",
    )
    parser.add_argument(
        "--payload-mb",
        default="1,2,4,8,16,32,64",
        help="Comma separated payload sizes in MB for sweep.",
    )
    parser.add_argument(
        "--groups-per-collective",
        type=int,
        default=2,
        help="Maximum number of comm groups sampled per collective per scenario.",
    )
    parser.add_argument(
        "--acceptable-threshold-pct",
        type=float,
        default=35.0,
        help="Acceptable max |delta_pct| threshold used in summary.",
    )
    parser.add_argument(
        "--convergence-span-threshold-pct",
        type=float,
        default=5.0,
        help="Convergence threshold on last-3 payload delta_pct span.",
    )
    parser.add_argument(
        "--csv-out",
        default=(
            "task_memory/task_2026-02-27_sim_restructure/"
            "mixtral_ep_placement_ab_2026-02-28.csv"
        ),
        help="CSV output for charting.",
    )
    parser.add_argument(
        "--json-out",
        default=(
            "task_memory/task_2026-02-27_sim_restructure/"
            "mixtral_ep_placement_ab_2026-02-28.json"
        ),
        help="JSON output path.",
    )
    parser.add_argument(
        "--report-path",
        default=(
            "task_memory/task_2026-02-27_sim_restructure/"
            "test_report_2026-02-28_mixtral_ep_placement_aware.md"
        ),
        help="Markdown report output path.",
    )
    return parser.parse_args()


def _parse_payloads(payload_mb: str) -> List[int]:
    payloads: List[int] = []
    for token in payload_mb.split(","):
        token = token.strip()
        if not token:
            continue
        value_mb = int(token)
        if value_mb <= 0:
            raise ValueError(f"payload value must be positive, got {token}")
        payloads.append(value_mb * 1024 * 1024)
    if not payloads:
        raise ValueError("at least one payload is required")
    return sorted(payloads)


def _build_backends(repo_root: Path, pp_domain_dim: str):
    placement_cfg = create_h800_sxm_ib_config()
    placement_cfg.communication.backend_options["collective-sim"].update(
        {
            "repo_root": str(repo_root),
            "pp_domain_dim": pp_domain_dim,
            "placement_mode": "global",
            "strict_mpu_alignment": True,
        }
    )

    baseline_cfg = create_h800_sxm_ib_config()
    baseline_cfg.communication.backend_options["collective-sim"].update(
        {
            "repo_root": str(repo_root),
            "pp_domain_dim": pp_domain_dim,
            "placement_mode": "group_size",
            "strict_mpu_alignment": False,
        }
    )

    return (
        create_cc_backend("collective-sim", placement_cfg),
        create_cc_backend("collective-sim", baseline_cfg),
    )


def _sample_groups(groups: Sequence[Sequence[int]], limit: int) -> List[Tuple[int, ...]]:
    sampled: List[Tuple[int, ...]] = []
    for group in groups:
        if len(group) < 2:
            continue
        sampled.append(tuple(int(rank) for rank in group))
        if len(sampled) >= limit:
            break
    return sampled


def _build_rows_for_scenario(
    scenario: ScenarioConfig,
    payloads: Sequence[int],
    groups_per_collective: int,
    placement_backend,
    baseline_backend,
) -> List[BenchmarkRow]:
    manager = ParallelGroupManager(
        local_size=scenario.local_size,
        world_size=scenario.world_size,
        pp_size=scenario.pp_size,
        tp_size=scenario.tp_size,
        exp_size=scenario.exp_size,
    )
    mpu_info = manager.get_mpu_info()

    rows: List[BenchmarkRow] = []

    exp_groups = _sample_groups(mpu_info.exp_groups or [], groups_per_collective)
    pp_pairs: List[Tuple[int, ...]] = []
    for pp_group in (mpu_info.pp_groups or []):
        if len(pp_group) >= 2:
            pp_pairs.append((int(pp_group[0]), int(pp_group[1])))
        if len(pp_pairs) >= groups_per_collective:
            break

    for payload in payloads:
        for exp_group in exp_groups:
            request = CommunicationPredictionRequest.from_raw(
                comm_group=exp_group,
                op_name="exp_all_to_all",
                data_size_bytes=payload,
                group_kind="exp",
                mpu_info=mpu_info,
            )
            placement_latency = float(placement_backend.predict(request))
            baseline_latency = float(baseline_backend.predict(request))
            delta_ms = placement_latency - baseline_latency
            delta_pct = (delta_ms / baseline_latency) * 100.0 if baseline_latency != 0.0 else 0.0
            rows.append(
                BenchmarkRow(
                    scenario=scenario.name,
                    collective="alltoall",
                    payload_bytes=payload,
                    comm_group=exp_group,
                    placement_aware_latency_ms=placement_latency,
                    baseline_latency_ms=baseline_latency,
                    delta_ms=delta_ms,
                    delta_pct=delta_pct,
                )
            )

        for pp_pair in pp_pairs:
            request = CommunicationPredictionRequest.from_raw(
                comm_group=pp_pair,
                op_name="send_forward",
                data_size_bytes=payload,
                group_kind="pp",
                mpu_info=mpu_info,
                metadata={"p2p_src_index": 0, "p2p_dst_index": 1, "p2p_direction": "0->1"},
            )
            placement_latency = float(placement_backend.predict(request))
            baseline_latency = float(baseline_backend.predict(request))
            delta_ms = placement_latency - baseline_latency
            delta_pct = (delta_ms / baseline_latency) * 100.0 if baseline_latency != 0.0 else 0.0
            rows.append(
                BenchmarkRow(
                    scenario=scenario.name,
                    collective="p2p_sendrecv",
                    payload_bytes=payload,
                    comm_group=pp_pair,
                    placement_aware_latency_ms=placement_latency,
                    baseline_latency_ms=baseline_latency,
                    delta_ms=delta_ms,
                    delta_pct=delta_pct,
                )
            )

    return rows


def _write_csv(path: Path, rows: Sequence[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "collective",
                "payload_bytes",
                "placement_aware_latency_ms",
                "baseline_latency_ms",
                "delta_ms",
                "delta_pct",
                "comm_group",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": row.scenario,
                    "collective": row.collective,
                    "payload_bytes": row.payload_bytes,
                    "placement_aware_latency_ms": f"{row.placement_aware_latency_ms:.9f}",
                    "baseline_latency_ms": f"{row.baseline_latency_ms:.9f}",
                    "delta_ms": f"{row.delta_ms:.9f}",
                    "delta_pct": f"{row.delta_pct:.6f}",
                    "comm_group": list(row.comm_group),
                }
            )


def _summarize(rows: Sequence[BenchmarkRow], convergence_span_threshold_pct: float) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str], List[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.collective)].append(row)

    summaries = []
    for (scenario, collective), bucket in sorted(grouped.items()):
        bucket = sorted(bucket, key=lambda item: item.payload_bytes)
        delta_values = [item.delta_pct for item in bucket]
        abs_delta_values = [abs(item.delta_pct) for item in bucket]
        last_three = delta_values[-3:] if len(delta_values) >= 3 else delta_values
        if last_three:
            span_last_three = max(last_three) - min(last_three)
        else:
            span_last_three = 0.0

        summaries.append(
            {
                "scenario": scenario,
                "collective": collective,
                "samples": len(bucket),
                "median_delta_pct": float(median(delta_values)) if delta_values else 0.0,
                "max_abs_delta_pct": float(max(abs_delta_values)) if abs_delta_values else 0.0,
                "convergence_span_last3_pct": float(span_last_three),
                "converged": span_last_three <= convergence_span_threshold_pct,
            }
        )

    all_abs = [abs(row.delta_pct) for row in rows]
    return {
        "total_rows": len(rows),
        "median_abs_delta_pct": float(median(all_abs)) if all_abs else 0.0,
        "max_abs_delta_pct": float(max(all_abs)) if all_abs else 0.0,
        "per_collective_summary": summaries,
    }


def _write_report(
    report_path: Path,
    rows: Sequence[BenchmarkRow],
    summary: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "## Modification History",
        "",
        "| Date       | Summary of Changes |",
        "|------------|--------------------|",
        f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | Added Mixtral/EP-heavy placement-aware benchmark report |",
        "",
        "# Test Report: Mixtral / EP-heavy Placement-aware Benchmark",
        "",
        f"**Date**: {now_utc}",
        "",
        "## Test Script Information",
        "- Script: `tests/performance/benchmark_mixtral_ep_placement.py`",
        "- Command:",
        "```bash",
        "python tests/performance/benchmark_mixtral_ep_placement.py \\",
        f"  --collective-sim-repo-root {args.collective_sim_repo_root} \\",
        f"  --payload-mb {args.payload_mb} \\",
        f"  --groups-per-collective {args.groups_per_collective} \\",
        f"  --csv-out {args.csv_out} \\",
        f"  --json-out {args.json_out} \\",
        f"  --report-path {args.report_path}",
        "```",
        "",
        "## Validation Criteria",
        "- Placement-aware path must use exact comm-group participant ranks from global topology.",
        "- Focus collectives: EP alltoall and PP p2p send/recv.",
        "- Report delta between placement-aware and placement-unaware baseline for charting and trend analysis.",
        "",
        "## Aggregate Results",
        f"- Total A/B rows: **{summary['total_rows']}**",
        f"- Median |delta_pct|: **{summary['median_abs_delta_pct']:.2f}%**",
        f"- Max |delta_pct|: **{summary['max_abs_delta_pct']:.2f}%**",
        f"- Acceptable threshold (max |delta_pct|): **{args.acceptable_threshold_pct:.2f}%**",
        f"- Threshold check: **{'PASS' if summary['max_abs_delta_pct'] <= args.acceptable_threshold_pct else 'FAIL'}**",
        "",
        "## Convergence Summary",
        "| Scenario | Collective | Samples | Median Delta % | Max |Delta| % | Last-3 Span % | Converged |",
        "|----------|------------|--------:|---------------:|--------------:|--------------:|----------:|",
    ]

    for item in summary["per_collective_summary"]:
        lines.append(
            f"| {item['scenario']} | {item['collective']} | {item['samples']} | "
            f"{item['median_delta_pct']:.2f} | {item['max_abs_delta_pct']:.2f} | "
            f"{item['convergence_span_last3_pct']:.2f} | {'YES' if item['converged'] else 'NO'} |"
        )

    lines.extend(
        [
            "",
            "## A/B Detail (first 40 rows)",
            "| Scenario | Collective | Payload (bytes) | Placement-aware (ms) | Baseline (ms) | Delta (ms) | Delta (%) |",
            "|----------|------------|----------------:|---------------------:|--------------:|-----------:|----------:|",
        ]
    )

    for row in list(rows)[:40]:
        lines.append(
            f"| {row.scenario} | {row.collective} | {row.payload_bytes} | "
            f"{row.placement_aware_latency_ms:.6f} | {row.baseline_latency_ms:.6f} | "
            f"{row.delta_ms:.6f} | {row.delta_pct:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "- EP-heavy alltoall and PP p2p paths are benchmarked with placement-aware on/off A/B outputs.",
            "- Use CSV for charting and verify whether max |delta_pct| meets your acceptance bar.",
            "- If threshold fails, inspect participant-rank mapping and topology assumptions before tuning models.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.collective_sim_repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"collective-sim repo root not found: {repo_root}")

    payloads = _parse_payloads(args.payload_mb)
    placement_backend, baseline_backend = _build_backends(repo_root, args.pp_domain_dim)

    scenarios = [
        ScenarioConfig(
            name="mixtral_16gpu_2node_pp2_tp1_dp8_exp8",
            world_size=16,
            pp_size=2,
            tp_size=1,
            exp_size=8,
            local_size=8,
        ),
        ScenarioConfig(
            name="mixtral_16gpu_2node_pp4_tp1_dp4_exp4",
            world_size=16,
            pp_size=4,
            tp_size=1,
            exp_size=4,
            local_size=8,
        ),
    ]

    rows: List[BenchmarkRow] = []
    for scenario in scenarios:
        rows.extend(
            _build_rows_for_scenario(
                scenario=scenario,
                payloads=payloads,
                groups_per_collective=args.groups_per_collective,
                placement_backend=placement_backend,
                baseline_backend=baseline_backend,
            )
        )

    if not rows:
        raise RuntimeError("No benchmark rows produced for Mixtral/EP-heavy placement test")

    summary = _summarize(rows, args.convergence_span_threshold_pct)

    csv_path = Path(args.csv_out)
    json_path = Path(args.json_out)
    report_path = Path(args.report_path)

    _write_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "summary": summary,
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, rows, summary, args)

    print(f"[MIXTRAL-PLACEMENT] rows={len(rows)}")
    print(f"[MIXTRAL-PLACEMENT] csv={csv_path}")
    print(f"[MIXTRAL-PLACEMENT] json={json_path}")
    print(f"[MIXTRAL-PLACEMENT] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
