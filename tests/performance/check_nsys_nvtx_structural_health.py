#!/usr/bin/env python3
"""Check NVTX CMD structural health in Nsight Systems sqlite traces.

This gate detects two common structural issues that can corrupt per-op attribution:
1) Open-ended CMD windows (e.g., forward_step ranges ending at session max timestamp).
2) Cross-op overlap between forward_step and backward_step CMD windows on the same rank.

It is intended to run before fidelity compare/gating so contaminated traces can fail fast.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


GLOBAL_TID_LOCAL_MASK = (1 << 24) - 1


@dataclass(frozen=True)
class NvtxCmdRange:
    start_ns: int
    end_ns: int
    global_pid: int
    rank: int
    op: str
    mg_state: str
    stage_id: str
    batch_id: str
    iter_id: str
    label: str


def parse_csv_ints(raw: str) -> Optional[List[int]]:
    if raw.strip() == "":
        return None
    values: List[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        return None
    return values


def parse_csv_strs(raw: str) -> List[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Empty op list")
    return values


def derive_global_pid(global_tid: int) -> int:
    return global_tid & (~GLOBAL_TID_LOCAL_MASK)


def parse_cmd_nvtx_label(label: str, prefix: str) -> Optional[Dict[str, str]]:
    prefix_token = f"{prefix}|"
    if not label.startswith(prefix_token):
        return None
    fields = label[len(prefix_token) :].split("|")
    mapping: Dict[str, str] = {}
    for field in fields:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        mapping[key] = value
    required = ("rank", "op", "state", "stage", "batch", "iter")
    if any(key not in mapping for key in required):
        return None
    return mapping


def load_parent_cmd_ranges(
    conn: sqlite3.Connection,
    label_prefix: str,
    allowed_ops: Sequence[str],
    rank_filter: Optional[Sequence[int]],
) -> List[NvtxCmdRange]:
    rows = conn.execute(
        """
        SELECT start, end, text, globalTid
        FROM NVTX_EVENTS
        WHERE text LIKE ?
          AND start IS NOT NULL
          AND end IS NOT NULL
          AND end > start
        ORDER BY start ASC
        """,
        (f"{label_prefix}|%",),
    ).fetchall()

    op_set = set(allowed_ops)
    rank_set = set(rank_filter) if rank_filter is not None else None
    ranges: List[NvtxCmdRange] = []

    for start_ns, end_ns, label, global_tid in rows:
        parsed = parse_cmd_nvtx_label(label, label_prefix)
        if parsed is None:
            continue
        if parsed.get("phase") in ("compute", "comm"):
            continue
        op = parsed["op"]
        if op not in op_set:
            continue
        rank = int(parsed["rank"])
        if rank_set is not None and rank not in rank_set:
            continue
        ranges.append(
            NvtxCmdRange(
                start_ns=int(start_ns),
                end_ns=int(end_ns),
                global_pid=derive_global_pid(int(global_tid)),
                rank=rank,
                op=op,
                mg_state=parsed["state"],
                stage_id=parsed["stage"],
                batch_id=parsed["batch"],
                iter_id=parsed["iter"],
                label=label,
            )
        )
    return ranges


def query_global_max_end_ns(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(end) FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def count_open_cmd_ranges(
    cmd_ranges: Sequence[NvtxCmdRange], global_max_end_ns: int
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for nvtx_range in cmd_ranges:
        if nvtx_range.end_ns != global_max_end_ns:
            continue
        counts[nvtx_range.op] = counts.get(nvtx_range.op, 0) + 1
    return counts


def compute_forward_backward_overlap_stats(
    cmd_ranges: Sequence[NvtxCmdRange],
) -> Tuple[int, float, Dict[int, int], Dict[int, float]]:
    by_rank: Dict[int, List[NvtxCmdRange]] = {}
    for nvtx_range in cmd_ranges:
        by_rank.setdefault(nvtx_range.rank, []).append(nvtx_range)

    overlap_count = 0
    overlap_ms = 0.0
    overlap_count_by_rank: Dict[int, int] = {}
    overlap_ms_by_rank: Dict[int, float] = {}

    for rank, ranges in by_rank.items():
        ranges_sorted = sorted(ranges, key=lambda item: item.start_ns)
        for idx, left in enumerate(ranges_sorted):
            if left.op not in ("forward_step", "backward_step"):
                continue
            for right in ranges_sorted[idx + 1 :]:
                if right.op not in ("forward_step", "backward_step"):
                    continue
                if left.op == right.op:
                    continue
                overlap_ns = min(left.end_ns, right.end_ns) - max(left.start_ns, right.start_ns)
                if overlap_ns <= 0:
                    continue
                overlap_count += 1
                overlap_ms += overlap_ns / 1_000_000.0
                overlap_count_by_rank[rank] = overlap_count_by_rank.get(rank, 0) + 1
                overlap_ms_by_rank[rank] = overlap_ms_by_rank.get(rank, 0.0) + (
                    overlap_ns / 1_000_000.0
                )

    return overlap_count, overlap_ms, overlap_count_by_rank, overlap_ms_by_rank


def evaluate_gate(
    open_counts: Dict[str, int],
    overlap_count: int,
    max_open_forward: int,
    max_open_backward: int,
    max_overlap_count: int,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    open_forward = open_counts.get("forward_step", 0)
    open_backward = open_counts.get("backward_step", 0)

    if open_forward > max_open_forward:
        failures.append(
            f"open_forward_step={open_forward} exceeds max_open_forward={max_open_forward}"
        )
    if open_backward > max_open_backward:
        failures.append(
            f"open_backward_step={open_backward} exceeds max_open_backward={max_open_backward}"
        )
    if overlap_count > max_overlap_count:
        failures.append(
            f"forward_backward_overlap_count={overlap_count} exceeds max_overlap_count={max_overlap_count}"
        )
    return len(failures) == 0, failures


def run(args: argparse.Namespace) -> int:
    if not args.sqlite.exists():
        print(f"[ERROR] Missing sqlite file: {args.sqlite}")
        return 2

    try:
        allowed_ops = parse_csv_strs(args.ops)
    except ValueError as exc:
        print(f"[ERROR] Invalid --ops: {exc}")
        return 2

    rank_filter = parse_csv_ints(args.ranks)

    conn = sqlite3.connect(str(args.sqlite))
    try:
        cmd_ranges = load_parent_cmd_ranges(
            conn=conn,
            label_prefix=args.label_prefix,
            allowed_ops=allowed_ops,
            rank_filter=rank_filter,
        )
        global_max_end_ns = query_global_max_end_ns(conn)
    finally:
        conn.close()

    open_counts = count_open_cmd_ranges(cmd_ranges, global_max_end_ns)
    (
        overlap_count,
        overlap_ms,
        overlap_count_by_rank,
        overlap_ms_by_rank,
    ) = compute_forward_backward_overlap_stats(cmd_ranges)

    passed, failures = evaluate_gate(
        open_counts=open_counts,
        overlap_count=overlap_count,
        max_open_forward=args.max_open_forward,
        max_open_backward=args.max_open_backward,
        max_overlap_count=args.max_overlap_count,
    )

    lines: List[str] = []
    lines.append(f"sqlite={args.sqlite}")
    lines.append(f"label_prefix={args.label_prefix}")
    lines.append(f"ops={','.join(allowed_ops)}")
    lines.append(
        "ranks=" + ("ALL" if rank_filter is None else ",".join(str(x) for x in rank_filter))
    )
    lines.append(f"global_max_end_ns={global_max_end_ns}")
    lines.append(f"parent_cmd_ranges={len(cmd_ranges)}")
    lines.append(f"open_forward_step={open_counts.get('forward_step', 0)}")
    lines.append(f"open_backward_step={open_counts.get('backward_step', 0)}")
    lines.append(f"forward_backward_overlap_count={overlap_count}")
    lines.append(f"forward_backward_overlap_ms={overlap_ms:.3f}")

    if overlap_count_by_rank:
        lines.append("overlap_count_by_rank:")
        for rank in sorted(overlap_count_by_rank.keys()):
            lines.append(f"  rank={rank}: {overlap_count_by_rank[rank]}")
    if overlap_ms_by_rank:
        lines.append("overlap_ms_by_rank:")
        for rank in sorted(overlap_ms_by_rank.keys()):
            lines.append(f"  rank={rank}: {overlap_ms_by_rank[rank]:.3f}")

    if failures:
        lines.append("gate_failures:")
        for failure in failures:
            lines.append(f"  - {failure}")

    report = "\n".join(lines)
    print(report)

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(report + "\n")
        print(f"[INFO] Report written to {args.report_path}")

    if passed:
        print("[RESULT] PASS (NVTX structural gate)")
        return 0
    print("[RESULT] FAIL (NVTX structural gate)")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check NVTX CMD structural health for attribution safety."
    )
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--label-prefix", type=str, default="cmd_trace")
    parser.add_argument(
        "--ops",
        type=str,
        default="forward_step,backward_step",
        help="Comma-separated CMD ops used for structural check.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="",
        help="Optional rank filter, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--max-open-forward",
        type=int,
        default=0,
        help="Maximum allowed count of forward_step ranges ending at global max end.",
    )
    parser.add_argument(
        "--max-open-backward",
        type=int,
        default=0,
        help="Maximum allowed count of backward_step ranges ending at global max end.",
    )
    parser.add_argument(
        "--max-overlap-count",
        type=int,
        default=0,
        help="Maximum allowed count of forward/backward overlaps on same rank.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    if args.max_open_forward < 0 or args.max_open_backward < 0 or args.max_overlap_count < 0:
        print("[ERROR] max-open-forward/max-open-backward/max-overlap-count must be >= 0")
        return 2

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
