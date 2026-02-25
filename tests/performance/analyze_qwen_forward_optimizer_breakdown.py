#!/usr/bin/env python3
"""Analyze per-rank forward/optimizer decomposition for distributed vs scaling traces."""

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

LINE_PATTERN = re.compile(r"^rank:(?P<rank>\d+):(?P<op>\w+)\((?P<body>.*)\)$")
DURATION_PATTERN = re.compile(r"duration=([0-9.]+)")
SUB_OPS_PATTERN = re.compile(r"sub_operations=(\[.*\])")
TS_PATTERN = re.compile(r"_rank(?P<rank>\d+)_(?P<ts>\d{14})\.txt$")

TARGET_OPS = ("forward_step", "optimizer_step")


@dataclass
class OpRecord:
    total_ms: float
    comm_ms: float
    comp_ms: float
    subops: List[str]


def _extract_timestamp(path: Path) -> Optional[str]:
    match = TS_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group("ts")


def find_latest_rank_file(trace_dir: Path, rank: int, pair_timestamp: Optional[str]) -> Path:
    best: Tuple[str, Optional[Path]] = ("", None)
    for candidate in trace_dir.glob(f"*rank{rank}_*.txt"):
        ts = _extract_timestamp(candidate)
        if ts is None:
            continue
        if pair_timestamp is not None and ts > pair_timestamp:
            continue
        if ts > best[0]:
            best = (ts, candidate)
    if best[1] is None:
        extra = f" with timestamp <= {pair_timestamp}" if pair_timestamp else ""
        raise FileNotFoundError(f"No trace file for rank {rank} in {trace_dir}{extra}")
    return best[1]


def parse_trace(path: Path, subtract_comm: bool) -> Dict[str, List[OpRecord]]:
    records: Dict[str, List[OpRecord]] = defaultdict(list)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if match is None:
            continue
        op = match.group("op")
        if op not in TARGET_OPS:
            continue
        body = match.group("body")
        duration_match = DURATION_PATTERN.search(body)
        sub_ops_match = SUB_OPS_PATTERN.search(body)
        if duration_match is None or sub_ops_match is None:
            continue
        total_ms = float(duration_match.group(1))
        subops = ast.literal_eval(sub_ops_match.group(1))
        comm_ms = 0.0
        for sub_op in subops:
            if "comm_func=" not in sub_op:
                continue
            sub_duration = DURATION_PATTERN.search(sub_op)
            if sub_duration is not None:
                comm_ms += float(sub_duration.group(1))
        comp_ms = total_ms - comm_ms if subtract_comm else total_ms
        records[op].append(OpRecord(total_ms=total_ms, comm_ms=comm_ms, comp_ms=comp_ms, subops=subops))
    return records


def aggregate_subops(op_records: List[OpRecord]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not op_records:
        return {}, {}
    by_src = defaultdict(float)
    by_comm = defaultdict(float)
    for record in op_records:
        for sub_op in record.subops:
            duration_match = DURATION_PATTERN.search(sub_op)
            duration = float(duration_match.group(1)) if duration_match else 0.0
            src_name = sub_op.split("trace_src_func=", 1)[1].split(",", 1)[0] if "trace_src_func=" in sub_op else "unknown"
            by_src[src_name] += duration
            if "comm_func=" in sub_op:
                comm_name = sub_op.split("comm_func=", 1)[1].split(",", 1)[0]
                by_comm[comm_name] += duration
    sample_count = len(op_records)
    src_avg = {key: value / sample_count for key, value in sorted(by_src.items())}
    comm_avg = {key: value / sample_count for key, value in sorted(by_comm.items())}
    return src_avg, comm_avg


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze forward/optimizer decomposition for Qwen traces.")
    parser.add_argument("--distributed-dir", type=Path, required=True)
    parser.add_argument("--scaling-dir", type=Path, required=True)
    parser.add_argument("--pair-timestamp", type=str, default=None)
    parser.add_argument("--ranks", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    ranks = [int(token.strip()) for token in args.ranks.split(",") if token.strip()]
    lines: List[str] = []

    lines.append(f"distributed_dir={args.distributed_dir}")
    lines.append(f"scaling_dir={args.scaling_dir}")
    lines.append(f"pair_timestamp={args.pair_timestamp}")
    lines.append(f"ranks={','.join(str(rank) for rank in ranks)}")
    lines.append("")

    summary_header = (
        "| rank | op | dist_samples | dist_total_ms | dist_comm_ms | dist_comp_ms | "
        "scale_samples | scale_total_ms | scale_comm_ms | scale_comp_ms | comp_diff_pct |"
    )
    summary_sep = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(summary_header)
    lines.append(summary_sep)

    for rank in ranks:
        dist_file = find_latest_rank_file(args.distributed_dir, rank, args.pair_timestamp)
        scale_file = find_latest_rank_file(args.scaling_dir, rank, args.pair_timestamp)
        dist_records = parse_trace(dist_file, subtract_comm=True)
        scale_records = parse_trace(scale_file, subtract_comm=False)

        lines.append("")
        lines.append(f"[files] rank {rank} dist={dist_file}")
        lines.append(f"[files] rank {rank} scale={scale_file}")

        for op in TARGET_OPS:
            d_records = dist_records.get(op, [])
            s_records = scale_records.get(op, [])
            if not d_records or not s_records:
                lines.append(f"| {rank} | {op} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            dist_total = mean([item.total_ms for item in d_records])
            dist_comm = mean([item.comm_ms for item in d_records])
            dist_comp = mean([item.comp_ms for item in d_records])
            scale_total = mean([item.total_ms for item in s_records])
            scale_comm = mean([item.comm_ms for item in s_records])
            scale_comp = mean([item.comp_ms for item in s_records])
            comp_diff = abs(scale_comp - dist_comp) / dist_comp * 100.0 if dist_comp else 0.0

            lines.append(
                f"| {rank} | {op} | {len(d_records)} | {dist_total:.4f} | {dist_comm:.4f} | {dist_comp:.4f} | "
                f"{len(s_records)} | {scale_total:.4f} | {scale_comm:.4f} | {scale_comp:.4f} | {comp_diff:.2f} |"
            )

            d_src, d_comm_breakdown = aggregate_subops(d_records)
            s_src, s_comm_breakdown = aggregate_subops(s_records)
            lines.append(f"  - {op} dist_trace_src_avg_ms={d_src}")
            lines.append(f"  - {op} scale_trace_src_avg_ms={s_src}")
            lines.append(f"  - {op} dist_comm_func_avg_ms={d_comm_breakdown}")
            lines.append(f"  - {op} scale_comm_func_avg_ms={s_comm_breakdown}")

    report_text = "\n".join(lines) + "\n"
    print(report_text)

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(report_text)
        print(f"[INFO] Report written to {args.report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
