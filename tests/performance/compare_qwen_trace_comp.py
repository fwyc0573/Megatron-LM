#!/usr/bin/env python3
"""Compare distributed/scaling trace comp durations with robust pairing and repeat median."""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple


DEFAULT_DISTRIBUTED_DIR = Path(
    "realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl128"
)
DEFAULT_SCALING_DIR = Path(
    "profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl128"
)
DEFAULT_RANKS = (0, 7)
DEFAULT_OPS = ("forward_step", "backward_step", "optimizer_step")

LINE_PATTERN = re.compile(r"^rank:(?P<rank>\d+):(?P<op>\w+)\((?P<body>.*)\)$")
DURATION_PATTERN = re.compile(r"duration=([0-9.]+)")
SUB_OPS_PATTERN = re.compile(r"sub_operations=(\[.*\])")
STATE_PATTERN = re.compile(r"mg_state=([^,]+)")
TS_PATTERN = re.compile(r"_rank(?P<rank>\d+)_(?P<ts>\d{14})\.txt$")


@dataclass
class OpStats:
    total_ms: float
    comm_ms: float
    comp_ms: float
    mg_state: str
    sub_op_count: int


@dataclass
class BucketStats:
    total_ms: float
    comm_ms: float
    comp_ms: float
    sub_op_count: float


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


def _extract_timestamp(path: Path) -> Optional[str]:
    match = TS_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group("ts")


def find_latest_rank_file(trace_dir: Path, rank: int, pair_timestamp: Optional[str] = None) -> Path:
    best: Tuple[str, Path] = ("", Path())
    for candidate in trace_dir.glob(f"*rank{rank}_*.txt"):
        ts = _extract_timestamp(candidate)
        if ts is None:
            continue
        if pair_timestamp is not None and ts > pair_timestamp:
            continue
        if ts > best[0]:
            best = (ts, candidate)
    if not best[1]:
        extra = (
            f" with timestamp <= {pair_timestamp}"
            if pair_timestamp is not None
            else ""
        )
        raise FileNotFoundError(f"No trace file found for rank {rank} in {trace_dir}{extra}")
    return best[1]


def parse_trace_file(path: Path, subtract_comm: bool) -> Dict[str, List[OpStats]]:
    result: Dict[str, List[OpStats]] = defaultdict(list)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            continue
        op = match.group("op")
        body = match.group("body")
        duration_match = DURATION_PATTERN.search(body)
        sub_ops_match = SUB_OPS_PATTERN.search(body)
        if duration_match is None or sub_ops_match is None:
            continue
        total_ms = float(duration_match.group(1))
        sub_ops = ast.literal_eval(sub_ops_match.group(1))
        state_match = STATE_PATTERN.search(body)
        mg_state = state_match.group(1) if state_match is not None else "None"

        comm_ms = 0.0
        for sub_op in sub_ops:
            if "comm_func=" not in sub_op:
                continue
            sub_duration_match = DURATION_PATTERN.search(sub_op)
            if sub_duration_match is not None:
                comm_ms += float(sub_duration_match.group(1))

        comp_ms = total_ms - comm_ms if subtract_comm else total_ms

        result[op].append(
            OpStats(
                total_ms=total_ms,
                comm_ms=comm_ms,
                comp_ms=comp_ms,
                mg_state=mg_state,
                sub_op_count=len(sub_ops),
            )
        )
    return result


def aggregate_by_state(op_stats: List[OpStats]) -> Dict[str, BucketStats]:
    values: Dict[str, List[OpStats]] = defaultdict(list)
    for stat in op_stats:
        values[stat.mg_state].append(stat)
    buckets: Dict[str, BucketStats] = {}
    for state, state_stats in values.items():
        buckets[state] = BucketStats(
            total_ms=mean([x.total_ms for x in state_stats]),
            comm_ms=mean([x.comm_ms for x in state_stats]),
            comp_ms=mean([x.comp_ms for x in state_stats]),
            sub_op_count=mean([x.sub_op_count for x in state_stats]),
        )
    return buckets


def aggregate_all(op_stats: List[OpStats]) -> Dict[str, BucketStats]:
    return {
        "ALL": BucketStats(
            total_ms=mean([x.total_ms for x in op_stats]),
            comm_ms=mean([x.comm_ms for x in op_stats]),
            comp_ms=mean([x.comp_ms for x in op_stats]),
            sub_op_count=mean([x.sub_op_count for x in op_stats]),
        )
    }


def append_repeat_record(repeat_path: Path, record: dict) -> None:
    repeat_path.parent.mkdir(parents=True, exist_ok=True)
    with repeat_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_repeat_records(repeat_path: Path) -> List[dict]:
    records: List[dict] = []
    if not repeat_path.exists():
        return records
    for raw in repeat_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def build_repeat_summary(
    records: List[dict], threshold_pct: float
) -> Tuple[List[str], int]:
    grouped: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    for record in records:
        for row in record.get("rows", []):
            key = (row["rank"], row["op"], row["mg_state"])
            grouped[key].append(row["diff_pct"])

    lines: List[str] = []
    lines.append(
        "| rank | op | mg_state | runs | median_diff_pct | status |"
    )
    lines.append("|---:|---|---|---:|---:|---|")
    failed_checks = 0
    for key in sorted(grouped.keys()):
        rank, op, state = key
        med = median(grouped[key])
        status = "PASS" if med <= threshold_pct else "FAIL"
        if status == "FAIL":
            failed_checks += 1
        lines.append(
            f"| {rank} | {op} | {state} | {len(grouped[key])} | {med:.2f} | {status} |"
        )
    return lines, failed_checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare distributed/scaling comp durations with configurable ranks/ops."
        )
    )
    parser.add_argument("--distributed-dir", type=Path, default=DEFAULT_DISTRIBUTED_DIR)
    parser.add_argument("--scaling-dir", type=Path, default=DEFAULT_SCALING_DIR)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument(
        "--ranks",
        type=str,
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
        help="Comma-separated rank list, e.g. 0,5.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=",".join(DEFAULT_OPS),
        help="Comma-separated op list, e.g. forward_step,backward_step,optimizer_step.",
    )
    parser.add_argument(
        "--distributed-subtract-comm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract comm sub-op duration from distributed total to derive distributed comp.",
    )
    parser.add_argument(
        "--scaling-subtract-comm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Subtract comm sub-op duration from scaling total to derive scaling comp.",
    )
    parser.add_argument(
        "--pair-timestamp",
        type=str,
        default=None,
        help=(
            "Timestamp cap in YYYYMMDDHHMMSS. "
            "For each rank, choose the latest trace file with timestamp <= cap."
        ),
    )
    parser.add_argument(
        "--repeat-report",
        type=Path,
        default=None,
        help=(
            "JSONL path used to append current run and generate median summary "
            "across repeated runs."
        ),
    )
    parser.add_argument(
        "--no-align-by-state",
        action="store_true",
        help="Disable (op, mg_state) bucket alignment and compare all states jointly.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

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

    if args.pair_timestamp is not None:
        try:
            datetime.strptime(args.pair_timestamp, "%Y%m%d%H%M%S")
        except ValueError:
            print(f"[ERROR] Invalid --pair-timestamp: {args.pair_timestamp}")
            return 2

    missing_dirs = [str(p) for p in (args.distributed_dir, args.scaling_dir) if not p.exists()]
    if missing_dirs:
        print(f"[ERROR] Missing trace directory: {', '.join(missing_dirs)}")
        return 2

    rows: List[str] = []
    row_records: List[dict] = []
    failed_checks = 0

    header = (
        "| rank | op | mg_state | "
        "dist_total_ms | dist_comm_ms | dist_comp_ms | dist_subops | "
        "scale_total_ms | scale_comm_ms | scale_comp_ms | scale_subops | "
        "diff_pct | status |"
    )
    sep = "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    rows.append(header)
    rows.append(sep)

    for rank in ranks:
        dist_file = find_latest_rank_file(
            args.distributed_dir, rank, pair_timestamp=args.pair_timestamp
        )
        scale_file = find_latest_rank_file(
            args.scaling_dir, rank, pair_timestamp=args.pair_timestamp
        )
        dist_ops = parse_trace_file(dist_file, subtract_comm=args.distributed_subtract_comm)
        scale_ops = parse_trace_file(scale_file, subtract_comm=args.scaling_subtract_comm)

        print(f"[INFO] rank {rank} distributed file: {dist_file}")
        print(f"[INFO] rank {rank} scaling file:     {scale_file}")

        for op in ops:
            if op not in dist_ops or op not in scale_ops:
                rows.append(
                    f"| {rank} | {op} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FAIL (missing op) |"
                )
                failed_checks += 1
                continue

            if args.no_align_by_state:
                dist_buckets = aggregate_all(dist_ops[op])
                scale_buckets = aggregate_all(scale_ops[op])
            else:
                dist_buckets = aggregate_by_state(dist_ops[op])
                scale_buckets = aggregate_by_state(scale_ops[op])
            common_states = sorted(set(dist_buckets.keys()) & set(scale_buckets.keys()))

            if not common_states:
                rows.append(
                    f"| {rank} | {op} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FAIL (no common mg_state) |"
                )
                failed_checks += 1
                continue

            for state in common_states:
                dist_bucket = dist_buckets[state]
                scale_bucket = scale_buckets[state]
                if dist_bucket.comp_ms == 0:
                    diff_pct = 0.0 if scale_bucket.comp_ms == 0 else 100.0
                else:
                    diff_pct = (
                        abs(scale_bucket.comp_ms - dist_bucket.comp_ms)
                        / dist_bucket.comp_ms
                        * 100.0
                    )
                status = "PASS" if diff_pct <= args.threshold_pct else "FAIL"
                if status == "FAIL":
                    failed_checks += 1

                rows.append(
                    "| "
                    f"{rank} | {op} | {state} | "
                    f"{dist_bucket.total_ms:.4f} | {dist_bucket.comm_ms:.4f} | {dist_bucket.comp_ms:.4f} | {dist_bucket.sub_op_count:.2f} | "
                    f"{scale_bucket.total_ms:.4f} | {scale_bucket.comm_ms:.4f} | {scale_bucket.comp_ms:.4f} | {scale_bucket.sub_op_count:.2f} | "
                    f"{diff_pct:.2f} | {status} |"
                )

                row_records.append(
                    {
                        "rank": rank,
                        "op": op,
                        "mg_state": state,
                        "dist_total_ms": dist_bucket.total_ms,
                        "dist_comm_ms": dist_bucket.comm_ms,
                        "dist_comp_ms": dist_bucket.comp_ms,
                        "dist_subops": dist_bucket.sub_op_count,
                        "scale_total_ms": scale_bucket.total_ms,
                        "scale_comm_ms": scale_bucket.comm_ms,
                        "scale_comp_ms": scale_bucket.comp_ms,
                        "scale_subops": scale_bucket.sub_op_count,
                        "diff_pct": diff_pct,
                        "status": status,
                    }
                )

    report_lines: List[str] = []
    report_lines.append(f"threshold_pct={args.threshold_pct:.2f}")
    report_lines.append(f"distributed_dir={args.distributed_dir}")
    report_lines.append(f"scaling_dir={args.scaling_dir}")
    report_lines.append(f"ranks={','.join(str(rank) for rank in ranks)}")
    report_lines.append(f"ops={','.join(ops)}")
    report_lines.append(f"distributed_subtract_comm={args.distributed_subtract_comm}")
    report_lines.append(f"scaling_subtract_comm={args.scaling_subtract_comm}")
    report_lines.append(f"align_by_state={not args.no_align_by_state}")
    report_lines.append(f"pair_timestamp={args.pair_timestamp}")
    report_lines.extend(rows)

    if args.repeat_report is not None:
        series_key = (
            f"distributed={args.distributed_dir}|scaling={args.scaling_dir}|"
            f"align={not args.no_align_by_state}|ranks={','.join(str(rank) for rank in ranks)}|"
            f"ops={','.join(ops)}|dist_subtract={args.distributed_subtract_comm}|"
            f"scale_subtract={args.scaling_subtract_comm}"
        )
        run_record = {
            "series_key": series_key,
            "threshold_pct": args.threshold_pct,
            "run_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": row_records,
        }
        append_repeat_record(args.repeat_report, run_record)
        all_records = [
            record
            for record in load_repeat_records(args.repeat_report)
            if record.get("series_key") == series_key
        ]
        repeat_lines, repeat_failed = build_repeat_summary(all_records, args.threshold_pct)
        report_lines.append("")
        report_lines.append("repeat_median_summary:")
        report_lines.extend(repeat_lines)
        if repeat_failed > 0:
            failed_checks += repeat_failed
        print(f"[INFO] Repeat records in current series: {len(all_records)}")
        print(f"[INFO] Repeat report updated: {args.repeat_report}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(report_text + "\n")
        print(f"[INFO] Report written to {args.report_path}")

    if failed_checks > 0:
        print(f"[RESULT] FAIL ({failed_checks} checks above threshold or missing).")
        return 1

    print("[RESULT] PASS (all checks within threshold).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
