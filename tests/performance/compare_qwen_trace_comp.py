#!/usr/bin/env python3
"""Compare latest Qwen3 distributed/scaling trace comp durations for rank0/rank7."""

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


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


def find_latest_rank_file(trace_dir: Path, rank: int) -> Path:
    best: Tuple[str, Path] = ("", Path())
    for candidate in trace_dir.glob(f"*rank{rank}_*.txt"):
        match = TS_PATTERN.search(candidate.name)
        if not match:
            continue
        ts = match.group("ts")
        if ts > best[0]:
            best = (ts, candidate)
    if not best[1]:
        raise FileNotFoundError(f"No trace file found for rank {rank} in {trace_dir}")
    return best[1]


def parse_trace_file(path: Path) -> Dict[str, List[OpStats]]:
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
            sub_duration_match = DURATION_PATTERN.search(sub_op)
            if sub_duration_match is not None:
                comm_ms += float(sub_duration_match.group(1))
        result[op].append(
            OpStats(
                total_ms=total_ms,
                comm_ms=comm_ms,
                comp_ms=total_ms - comm_ms,
                mg_state=mg_state,
            )
        )
    return result


def aggregate_comp_by_state(op_stats: List[OpStats]) -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for stat in op_stats:
        values[stat.mg_state].append(stat.comp_ms)
    return {state: mean(comp_values) for state, comp_values in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare latest distributed/scaling comp durations for rank0/rank7 (forward/backward/optimizer)."
    )
    parser.add_argument("--distributed-dir", type=Path, default=DEFAULT_DISTRIBUTED_DIR)
    parser.add_argument("--scaling-dir", type=Path, default=DEFAULT_SCALING_DIR)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument(
        "--no-align-by-state",
        action="store_true",
        help="Disable (op, mg_state) bucket alignment and compare all states jointly.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    missing_dirs = [str(p) for p in (args.distributed_dir, args.scaling_dir) if not p.exists()]
    if missing_dirs:
        print(f"[ERROR] Missing trace directory: {', '.join(missing_dirs)}")
        return 2

    rows: List[str] = []
    failed_checks = 0

    header = (
        "| rank | op | mg_state | distributed_comp_ms | scaling_comp_ms | diff_pct | status |"
    )
    sep = "|---:|---|---|---:|---:|---:|---|"
    rows.append(header)
    rows.append(sep)

    for rank in DEFAULT_RANKS:
        dist_file = find_latest_rank_file(args.distributed_dir, rank)
        scale_file = find_latest_rank_file(args.scaling_dir, rank)
        dist_ops = parse_trace_file(dist_file)
        scale_ops = parse_trace_file(scale_file)

        print(f"[INFO] rank {rank} distributed file: {dist_file}")
        print(f"[INFO] rank {rank} scaling file:     {scale_file}")

        for op in DEFAULT_OPS:
            if op not in dist_ops or op not in scale_ops:
                rows.append(
                    f"| {rank} | {op} | N/A | N/A | N/A | N/A | FAIL (missing op) |"
                )
                failed_checks += 1
                continue

            if args.no_align_by_state:
                dist_buckets = {"ALL": mean([stat.comp_ms for stat in dist_ops[op]])}
                scale_buckets = {"ALL": mean([stat.comp_ms for stat in scale_ops[op]])}
            else:
                dist_buckets = aggregate_comp_by_state(dist_ops[op])
                scale_buckets = aggregate_comp_by_state(scale_ops[op])
            common_states = sorted(set(dist_buckets.keys()) & set(scale_buckets.keys()))

            if not common_states:
                rows.append(
                    f"| {rank} | {op} | N/A | N/A | N/A | N/A | FAIL (no common mg_state) |"
                )
                failed_checks += 1
                continue

            for state in common_states:
                dist_comp = dist_buckets[state]
                scale_comp = scale_buckets[state]
                if dist_comp == 0:
                    diff_pct = 0.0 if scale_comp == 0 else 100.0
                else:
                    diff_pct = abs(scale_comp - dist_comp) / dist_comp * 100.0
                status = "PASS" if diff_pct <= args.threshold_pct else "FAIL"
                if status == "FAIL":
                    failed_checks += 1
                rows.append(
                    f"| {rank} | {op} | {state} | {dist_comp:.4f} | {scale_comp:.4f} | {diff_pct:.2f} | {status} |"
                )

    report_lines: List[str] = []
    report_lines.append(f"threshold_pct={args.threshold_pct:.2f}")
    report_lines.append(f"distributed_dir={args.distributed_dir}")
    report_lines.append(f"scaling_dir={args.scaling_dir}")
    report_lines.append(f"align_by_state={not args.no_align_by_state}")
    report_lines.extend(rows)

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
