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
from typing import Dict, List, Optional, Tuple


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
        "--trim-ratio",
        type=float,
        default=0.2,
        help="Trim ratio for trimmed_mean reducer.",
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
        "scale_samples | scale_compute_ms | scale_comm_ms | diff_pct | status |"
    )
    rows.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

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
                        f"{len(dist_samples)} | N/A | N/A | {len(scale_samples)} | N/A | N/A | N/A | FAIL (missing) |"
                    )
                    failed += 1
                    continue
                dist_compute = reduce_values(
                    [x["compute_kernel_ms"] for x in dist_samples],
                    method=args.dist_reducer,
                    trim_ratio=args.trim_ratio,
                )
                scale_compute = reduce_values(
                    [x["compute_kernel_ms"] for x in scale_samples],
                    method=args.scale_reducer,
                    trim_ratio=args.trim_ratio,
                )
                dist_comm = reduce_values(
                    [x["comm_kernel_ms"] for x in dist_samples],
                    method=args.dist_reducer,
                    trim_ratio=args.trim_ratio,
                )
                scale_comm = reduce_values(
                    [x["comm_kernel_ms"] for x in scale_samples],
                    method=args.scale_reducer,
                    trim_ratio=args.trim_ratio,
                )
                if dist_compute == 0:
                    diff_pct = 0.0 if scale_compute == 0 else 100.0
                else:
                    diff_pct = abs(scale_compute - dist_compute) / dist_compute * 100.0
                status = "PASS" if diff_pct <= args.threshold_pct else "FAIL"
                if status == "FAIL":
                    failed += 1
                rows.append(
                    f"| {rank} | {op} | {state} | {len(dist_samples)} | {dist_compute:.4f} | "
                    f"{dist_comm:.4f} | {len(scale_samples)} | {scale_compute:.4f} | "
                    f"{scale_comm:.4f} | {diff_pct:.2f} | {status} |"
                )
                row_records.append(
                    {
                        "rank": rank,
                        "op": op,
                        "mg_state": state,
                        "dist_samples": len(dist_samples),
                        "dist_compute_ms": dist_compute,
                        "dist_comm_ms": dist_comm,
                        "scale_samples": len(scale_samples),
                        "scale_compute_ms": scale_compute,
                        "scale_comm_ms": scale_comm,
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
    report_lines.append(f"trim_ratio={args.trim_ratio:.4f}")
    report_lines.append(f"threshold_pct={args.threshold_pct:.2f}")
    report_lines.extend(rows)
    report_lines.append("")
    report_lines.append("op_rank_median_aux_summary(recommended_for_paper):")
    op_summary_lines, _ = build_op_rank_median_summary(row_records, args.threshold_pct)
    report_lines.extend(op_summary_lines)

    if args.repeat_report is not None:
        series_key = (
            f"dist={args.distributed_json}|scale={args.scaling_json}|"
            f"align={not args.no_align_by_state}|ranks={','.join(str(x) for x in ranks)}|"
            f"ops={','.join(ops)}|dist_reducer={args.dist_reducer}|"
            f"scale_reducer={args.scale_reducer}|trim={args.trim_ratio:.6f}"
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
