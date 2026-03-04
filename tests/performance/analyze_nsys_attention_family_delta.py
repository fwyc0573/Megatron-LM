#!/usr/bin/env python3
"""Diagnose attention-family residual between distributed and scaling Nsight traces.

This script focuses on one phase window slice (default: backward_step/steady/stage1/phase=compute)
and reports:
- paired-window primary-stream compute totals
- fmha_cutlassB contribution and launch statistics
- top kernel-name deltas (scale - dist)
- launch-config parity check for fmha kernels
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


GLOBAL_TID_LOCAL_MASK = (1 << 24) - 1


@dataclass(frozen=True)
class PhaseWindow:
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


@dataclass(frozen=True)
class KernelRecord:
    start_ns: int
    end_ns: int
    stream_id: int
    name: str
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    registers_per_thread: int
    static_shared_memory: int
    dynamic_shared_memory: int


def parse_csv_ints(raw: str) -> List[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Empty rank list")
    return values


def parse_csv_strs(raw: str) -> List[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Empty value list")
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


def overlap_ns(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0, hi - lo)


def merged_length_ns(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals_sorted = sorted(intervals)
    total = 0
    cur_start, cur_end = intervals_sorted[0]
    for start, end in intervals_sorted[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
            continue
        total += cur_end - cur_start
        cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def _resolve_name(candidate: Optional[str], fallback: Optional[int]) -> str:
    if candidate is not None and candidate != "":
        return candidate
    if fallback is None:
        return "UNKNOWN"
    return f"id:{fallback}"


def load_phase_windows(
    conn: sqlite3.Connection,
    label_prefix: str,
    ranks: Sequence[int],
    op: str,
    mg_state: str,
    stage_id: str,
    phase: str,
    segment_key: Optional[str],
    segment_values: Optional[Sequence[str]],
) -> List[PhaseWindow]:
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

    rank_set = set(ranks)
    segment_set = set(segment_values) if segment_values else None
    windows: List[PhaseWindow] = []
    for start_ns, end_ns, label, global_tid in rows:
        parsed = parse_cmd_nvtx_label(label, label_prefix)
        if parsed is None:
            continue
        if parsed.get("phase") != phase:
            continue
        if parsed["op"] != op:
            continue
        if parsed["state"] != mg_state:
            continue
        if parsed["stage"] != str(stage_id):
            continue
        if segment_key is not None:
            segment_value = parsed.get(segment_key)
            if segment_value is None:
                continue
            if segment_set is not None and segment_value not in segment_set:
                continue
        rank = int(parsed["rank"])
        if rank not in rank_set:
            continue
        windows.append(
            PhaseWindow(
                start_ns=int(start_ns),
                end_ns=int(end_ns),
                global_pid=derive_global_pid(int(global_tid)),
                rank=rank,
                op=parsed["op"],
                mg_state=parsed["state"],
                stage_id=parsed["stage"],
                batch_id=parsed["batch"],
                iter_id=parsed["iter"],
                label=label,
            )
        )
    return windows


def load_kernels_for_windows(
    conn: sqlite3.Connection, windows: Sequence[PhaseWindow]
) -> Dict[int, List[KernelRecord]]:
    if not windows:
        return {}
    min_start = min(window.start_ns for window in windows)
    max_end = max(window.end_ns for window in windows)
    pids = sorted(set(window.global_pid for window in windows))
    placeholders = ",".join("?" for _ in pids)
    query = f"""
    SELECT
        k.start,
        k.end,
        k.streamId,
        k.globalPid,
        s_short.value AS short_name,
        s_dem.value AS demangled_name,
        k.shortName,
        k.demangledName,
        k.gridX,
        k.gridY,
        k.gridZ,
        k.blockX,
        k.blockY,
        k.blockZ,
        k.registersPerThread,
        k.staticSharedMemory,
        k.dynamicSharedMemory
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
    LEFT JOIN StringIds AS s_short ON k.shortName = s_short.id
    LEFT JOIN StringIds AS s_dem ON k.demangledName = s_dem.id
    WHERE k.end > ?
      AND k.start < ?
      AND k.globalPid IN ({placeholders})
    ORDER BY k.globalPid ASC, k.start ASC
    """
    params: List[int] = [min_start, max_end, *pids]
    rows = conn.execute(query, params).fetchall()

    grouped: Dict[int, List[KernelRecord]] = defaultdict(list)
    for (
        start_ns,
        end_ns,
        stream_id,
        global_pid,
        short_name,
        demangled_name,
        short_name_id,
        demangled_name_id,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        registers_per_thread,
        static_shared_memory,
        dynamic_shared_memory,
    ) in rows:
        kernel_name = _resolve_name(demangled_name, demangled_name_id)
        if kernel_name.startswith("id:"):
            kernel_name = _resolve_name(short_name, short_name_id)
        grouped[int(global_pid)].append(
            KernelRecord(
                start_ns=int(start_ns),
                end_ns=int(end_ns),
                stream_id=int(stream_id),
                name=kernel_name,
                grid_x=int(grid_x),
                grid_y=int(grid_y),
                grid_z=int(grid_z),
                block_x=int(block_x),
                block_y=int(block_y),
                block_z=int(block_z),
                registers_per_thread=int(registers_per_thread),
                static_shared_memory=int(static_shared_memory),
                dynamic_shared_memory=int(dynamic_shared_memory),
            )
        )
    return grouped


def _is_comm(name: str) -> bool:
    return "nccl" in name.lower()


def _safe_numeric_key(raw: str) -> Tuple[int, object]:
    """Sort helper that prefers numeric ordering but preserves non-numeric values."""
    try:
        return (0, int(raw))
    except ValueError:
        return (1, raw)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def interquartile_range(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(float(item) for item in values)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    return q3 - q1


def summarize_window(
    window: PhaseWindow,
    kernels_by_pid: Dict[int, List[KernelRecord]],
    small_kernel_threshold_us: float,
    adjacency_window_us: float,
) -> dict:
    kernels = kernels_by_pid.get(window.global_pid, [])
    stream_intervals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    stream_name_ms: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stream_small_kernel_count: Dict[int, int] = defaultdict(int)
    stream_small_kernel_ms: Dict[int, float] = defaultdict(float)
    fmha_kernel_durations_us: List[float] = []
    fmha_launch_configs: List[Tuple[int, ...]] = []
    fmha_stream_ids: List[int] = []
    overlap_kernel_records: List[Tuple[int, int, str, int]] = []
    stream_overlap_records: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)
    adjacency_window_ns = int(adjacency_window_us * 1_000.0)
    small_kernel_threshold_ns = int(small_kernel_threshold_us * 1_000.0)

    for kernel in kernels:
        if kernel.end_ns <= window.start_ns:
            continue
        if kernel.start_ns >= window.end_ns:
            break
        ov = overlap_ns(window.start_ns, window.end_ns, kernel.start_ns, kernel.end_ns)
        if ov <= 0:
            continue
        if _is_comm(kernel.name):
            continue
        overlap_start = max(window.start_ns, kernel.start_ns)
        overlap_end = min(window.end_ns, kernel.end_ns)
        overlap_duration_ns = overlap_end - overlap_start
        stream_intervals[kernel.stream_id].append((overlap_start, overlap_end))
        stream_name_ms[kernel.stream_id][kernel.name] += ov / 1_000_000.0
        overlap_kernel_records.append(
            (overlap_start, overlap_end, kernel.name, kernel.stream_id)
        )
        stream_overlap_records[kernel.stream_id].append(
            (overlap_start, overlap_end, kernel.name)
        )
        if overlap_duration_ns <= small_kernel_threshold_ns:
            stream_small_kernel_count[kernel.stream_id] += 1
            stream_small_kernel_ms[kernel.stream_id] += overlap_duration_ns / 1_000_000.0
        if "fmha_cutlassB" in kernel.name:
            # Use overlapped duration so stage/phase window semantics stay strict.
            fmha_kernel_durations_us.append(ov / 1_000.0)
            fmha_stream_ids.append(kernel.stream_id)
            fmha_launch_configs.append(
                (
                    kernel.grid_x,
                    kernel.grid_y,
                    kernel.grid_z,
                    kernel.block_x,
                    kernel.block_y,
                    kernel.block_z,
                    kernel.registers_per_thread,
                    kernel.static_shared_memory,
                    kernel.dynamic_shared_memory,
                )
            )

    primary_stream_id: Optional[int] = None
    primary_union_ms = 0.0
    primary_name_ms: Dict[str, float] = {}
    primary_small_kernel_count = 0
    primary_small_kernel_ms = 0.0
    if stream_intervals:
        primary_stream_id = max(
            stream_intervals,
            key=lambda stream_id: merged_length_ns(stream_intervals[stream_id]),
        )
        primary_union_ms = merged_length_ns(stream_intervals[primary_stream_id]) / 1_000_000.0
        primary_name_ms = dict(stream_name_ms[primary_stream_id])
        primary_small_kernel_count = stream_small_kernel_count.get(primary_stream_id, 0)
        primary_small_kernel_ms = stream_small_kernel_ms.get(primary_stream_id, 0.0)

    small_kernel_adjacent_pre_count = 0
    small_kernel_adjacent_post_count = 0
    small_kernel_adjacent_pre_ms = 0.0
    small_kernel_adjacent_post_ms = 0.0
    small_kernel_adjacent_pre_name_count: Dict[str, int] = defaultdict(int)
    small_kernel_adjacent_post_name_count: Dict[str, int] = defaultdict(int)
    small_kernel_adjacent_pre_name_ms: Dict[str, float] = defaultdict(float)
    small_kernel_adjacent_post_name_ms: Dict[str, float] = defaultdict(float)
    fmha_overlap_records = [
        (start_ns, end_ns)
        for start_ns, end_ns, name, _stream_id in overlap_kernel_records
        if "fmha_cutlassB" in name
    ]
    non_fmha_small_records = [
        (start_ns, end_ns, name)
        for start_ns, end_ns, name, _stream_id in overlap_kernel_records
        if "fmha_cutlassB" not in name and (end_ns - start_ns) <= small_kernel_threshold_ns
    ]
    for kernel_start_ns, kernel_end_ns, kernel_name in non_fmha_small_records:
        kernel_ms = (kernel_end_ns - kernel_start_ns) / 1_000_000.0
        has_pre_adjacent = False
        has_post_adjacent = False
        for fmha_start_ns, fmha_end_ns in fmha_overlap_records:
            if kernel_end_ns <= fmha_start_ns:
                delta_ns = fmha_start_ns - kernel_end_ns
                if delta_ns <= adjacency_window_ns:
                    has_pre_adjacent = True
            if kernel_start_ns >= fmha_end_ns:
                delta_ns = kernel_start_ns - fmha_end_ns
                if delta_ns <= adjacency_window_ns:
                    has_post_adjacent = True
            if has_pre_adjacent and has_post_adjacent:
                break
        if has_pre_adjacent:
            small_kernel_adjacent_pre_count += 1
            small_kernel_adjacent_pre_ms += kernel_ms
            small_kernel_adjacent_pre_name_count[kernel_name] += 1
            small_kernel_adjacent_pre_name_ms[kernel_name] += kernel_ms
        if has_post_adjacent:
            small_kernel_adjacent_post_count += 1
            small_kernel_adjacent_post_ms += kernel_ms
            small_kernel_adjacent_post_name_count[kernel_name] += 1
            small_kernel_adjacent_post_name_ms[kernel_name] += kernel_ms

    small_kernel_immediate_pre_count = 0
    small_kernel_immediate_post_count = 0
    small_kernel_immediate_pre_ms = 0.0
    small_kernel_immediate_post_ms = 0.0
    small_kernel_immediate_pre_name_count: Dict[str, int] = defaultdict(int)
    small_kernel_immediate_post_name_count: Dict[str, int] = defaultdict(int)
    small_kernel_immediate_pre_name_ms: Dict[str, float] = defaultdict(float)
    small_kernel_immediate_post_name_ms: Dict[str, float] = defaultdict(float)
    for stream_records in stream_overlap_records.values():
        stream_records.sort(key=lambda item: (item[0], item[1]))
        for idx, (kernel_start_ns, kernel_end_ns, kernel_name) in enumerate(stream_records):
            if "fmha_cutlassB" not in kernel_name:
                continue

            prev_idx = idx - 1
            while prev_idx >= 0:
                prev_start_ns, prev_end_ns, prev_name = stream_records[prev_idx]
                if prev_end_ns <= kernel_start_ns:
                    prev_duration_ns = prev_end_ns - prev_start_ns
                    prev_delta_ns = kernel_start_ns - prev_end_ns
                    if (
                        "fmha_cutlassB" not in prev_name
                        and prev_duration_ns <= small_kernel_threshold_ns
                        and prev_delta_ns <= adjacency_window_ns
                    ):
                        prev_ms = prev_duration_ns / 1_000_000.0
                        small_kernel_immediate_pre_count += 1
                        small_kernel_immediate_pre_ms += prev_ms
                        small_kernel_immediate_pre_name_count[prev_name] += 1
                        small_kernel_immediate_pre_name_ms[prev_name] += prev_ms
                    break
                prev_idx -= 1

            next_idx = idx + 1
            while next_idx < len(stream_records):
                next_start_ns, next_end_ns, next_name = stream_records[next_idx]
                if next_start_ns >= kernel_end_ns:
                    next_duration_ns = next_end_ns - next_start_ns
                    next_delta_ns = next_start_ns - kernel_end_ns
                    if (
                        "fmha_cutlassB" not in next_name
                        and next_duration_ns <= small_kernel_threshold_ns
                        and next_delta_ns <= adjacency_window_ns
                    ):
                        next_ms = next_duration_ns / 1_000_000.0
                        small_kernel_immediate_post_count += 1
                        small_kernel_immediate_post_ms += next_ms
                        small_kernel_immediate_post_name_count[next_name] += 1
                        small_kernel_immediate_post_name_ms[next_name] += next_ms
                    break
                next_idx += 1

    return {
        "rank": window.rank,
        "iter_id": window.iter_id,
        "batch_id": window.batch_id,
        "primary_stream_id": primary_stream_id,
        "compute_stream_count": len(stream_intervals),
        "primary_union_ms": primary_union_ms,
        "primary_name_ms": primary_name_ms,
        "primary_small_kernel_count": primary_small_kernel_count,
        "primary_small_kernel_ms": primary_small_kernel_ms,
        "small_kernel_adjacent_pre_count": small_kernel_adjacent_pre_count,
        "small_kernel_adjacent_post_count": small_kernel_adjacent_post_count,
        "small_kernel_adjacent_pre_ms": small_kernel_adjacent_pre_ms,
        "small_kernel_adjacent_post_ms": small_kernel_adjacent_post_ms,
        "small_kernel_adjacent_pre_name_count": dict(small_kernel_adjacent_pre_name_count),
        "small_kernel_adjacent_post_name_count": dict(small_kernel_adjacent_post_name_count),
        "small_kernel_adjacent_pre_name_ms": dict(small_kernel_adjacent_pre_name_ms),
        "small_kernel_adjacent_post_name_ms": dict(small_kernel_adjacent_post_name_ms),
        "small_kernel_immediate_pre_count": small_kernel_immediate_pre_count,
        "small_kernel_immediate_post_count": small_kernel_immediate_post_count,
        "small_kernel_immediate_pre_ms": small_kernel_immediate_pre_ms,
        "small_kernel_immediate_post_ms": small_kernel_immediate_post_ms,
        "small_kernel_immediate_pre_name_count": dict(small_kernel_immediate_pre_name_count),
        "small_kernel_immediate_post_name_count": dict(small_kernel_immediate_post_name_count),
        "small_kernel_immediate_pre_name_ms": dict(small_kernel_immediate_pre_name_ms),
        "small_kernel_immediate_post_name_ms": dict(small_kernel_immediate_post_name_ms),
        "fmha_stream_ids": fmha_stream_ids,
        "fmha_kernel_durations_us": fmha_kernel_durations_us,
        "fmha_launch_configs": fmha_launch_configs,
    }


def pair_windows(dist_rows: Sequence[dict], scale_rows: Sequence[dict]) -> Tuple[List[Tuple[dict, dict]], int]:
    grouped_dist: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    grouped_scale: Dict[Tuple[int, str], List[dict]] = defaultdict(list)

    for row in dist_rows:
        grouped_dist[(int(row["rank"]), str(row["iter_id"]))].append(row)
    for row in scale_rows:
        grouped_scale[(int(row["rank"]), str(row["iter_id"]))].append(row)

    for key in grouped_dist:
        grouped_dist[key].sort(key=lambda item: _safe_numeric_key(str(item["batch_id"])))
    for key in grouped_scale:
        grouped_scale[key].sort(key=lambda item: _safe_numeric_key(str(item["batch_id"])))

    pairs: List[Tuple[dict, dict]] = []
    missing = 0
    for key in sorted(set(grouped_dist) | set(grouped_scale)):
        dist_list = grouped_dist.get(key, [])
        scale_list = grouped_scale.get(key, [])
        pair_count = min(len(dist_list), len(scale_list))
        if pair_count == 0:
            missing += max(len(dist_list), len(scale_list))
            continue
        missing += abs(len(dist_list) - len(scale_list))
        for idx in range(pair_count):
            pairs.append((dist_list[idx], scale_list[idx]))
    return pairs, missing


def build_report(pairs: Sequence[Tuple[dict, dict]], missing: int) -> Tuple[str, dict]:
    delta_by_name: Dict[str, float] = defaultdict(float)
    fmha_dist_ms = 0.0
    fmha_scale_ms = 0.0
    total_dist_ms = 0.0
    total_scale_ms = 0.0

    fmha_us_dist_by_rank: Dict[int, List[float]] = defaultdict(list)
    fmha_us_scale_by_rank: Dict[int, List[float]] = defaultdict(list)
    fmha_cfg_dist: List[Tuple[int, ...]] = []
    fmha_cfg_scale: List[Tuple[int, ...]] = []
    primary_stream_id_mismatch_pairs = 0
    fmha_stream_set_mismatch_pairs = 0
    dist_compute_stream_count = 0
    scale_compute_stream_count = 0
    dist_small_kernel_count = 0
    scale_small_kernel_count = 0
    dist_small_kernel_ms = 0.0
    scale_small_kernel_ms = 0.0
    dist_small_adj_pre_count = 0
    scale_small_adj_pre_count = 0
    dist_small_adj_post_count = 0
    scale_small_adj_post_count = 0
    dist_small_adj_pre_ms = 0.0
    scale_small_adj_pre_ms = 0.0
    dist_small_adj_post_ms = 0.0
    scale_small_adj_post_ms = 0.0
    dist_small_immediate_pre_count = 0
    scale_small_immediate_pre_count = 0
    dist_small_immediate_post_count = 0
    scale_small_immediate_post_count = 0
    dist_small_immediate_pre_ms = 0.0
    scale_small_immediate_pre_ms = 0.0
    dist_small_immediate_post_ms = 0.0
    scale_small_immediate_post_ms = 0.0
    dist_small_adj_pre_name_count: Dict[str, int] = defaultdict(int)
    scale_small_adj_pre_name_count: Dict[str, int] = defaultdict(int)
    dist_small_adj_pre_name_ms: Dict[str, float] = defaultdict(float)
    scale_small_adj_pre_name_ms: Dict[str, float] = defaultdict(float)
    dist_small_adj_post_name_count: Dict[str, int] = defaultdict(int)
    scale_small_adj_post_name_count: Dict[str, int] = defaultdict(int)
    dist_small_adj_post_name_ms: Dict[str, float] = defaultdict(float)
    scale_small_adj_post_name_ms: Dict[str, float] = defaultdict(float)
    dist_small_immediate_pre_name_count: Dict[str, int] = defaultdict(int)
    scale_small_immediate_pre_name_count: Dict[str, int] = defaultdict(int)
    dist_small_immediate_pre_name_ms: Dict[str, float] = defaultdict(float)
    scale_small_immediate_pre_name_ms: Dict[str, float] = defaultdict(float)
    dist_small_immediate_post_name_count: Dict[str, int] = defaultdict(int)
    scale_small_immediate_post_name_count: Dict[str, int] = defaultdict(int)
    dist_small_immediate_post_name_ms: Dict[str, float] = defaultdict(float)
    scale_small_immediate_post_name_ms: Dict[str, float] = defaultdict(float)

    for dist_row, scale_row in pairs:
        total_dist_ms += float(dist_row["primary_union_ms"])
        total_scale_ms += float(scale_row["primary_union_ms"])
        dist_compute_stream_count += int(dist_row.get("compute_stream_count", 0))
        scale_compute_stream_count += int(scale_row.get("compute_stream_count", 0))
        dist_small_kernel_count += int(dist_row.get("primary_small_kernel_count", 0))
        scale_small_kernel_count += int(scale_row.get("primary_small_kernel_count", 0))
        dist_small_kernel_ms += float(dist_row.get("primary_small_kernel_ms", 0.0))
        scale_small_kernel_ms += float(scale_row.get("primary_small_kernel_ms", 0.0))
        dist_small_adj_pre_count += int(dist_row.get("small_kernel_adjacent_pre_count", 0))
        scale_small_adj_pre_count += int(scale_row.get("small_kernel_adjacent_pre_count", 0))
        dist_small_adj_post_count += int(dist_row.get("small_kernel_adjacent_post_count", 0))
        scale_small_adj_post_count += int(scale_row.get("small_kernel_adjacent_post_count", 0))
        dist_small_adj_pre_ms += float(dist_row.get("small_kernel_adjacent_pre_ms", 0.0))
        scale_small_adj_pre_ms += float(scale_row.get("small_kernel_adjacent_pre_ms", 0.0))
        dist_small_adj_post_ms += float(dist_row.get("small_kernel_adjacent_post_ms", 0.0))
        scale_small_adj_post_ms += float(scale_row.get("small_kernel_adjacent_post_ms", 0.0))
        dist_small_immediate_pre_count += int(dist_row.get("small_kernel_immediate_pre_count", 0))
        scale_small_immediate_pre_count += int(
            scale_row.get("small_kernel_immediate_pre_count", 0)
        )
        dist_small_immediate_post_count += int(dist_row.get("small_kernel_immediate_post_count", 0))
        scale_small_immediate_post_count += int(
            scale_row.get("small_kernel_immediate_post_count", 0)
        )
        dist_small_immediate_pre_ms += float(dist_row.get("small_kernel_immediate_pre_ms", 0.0))
        scale_small_immediate_pre_ms += float(
            scale_row.get("small_kernel_immediate_pre_ms", 0.0)
        )
        dist_small_immediate_post_ms += float(
            dist_row.get("small_kernel_immediate_post_ms", 0.0)
        )
        scale_small_immediate_post_ms += float(
            scale_row.get("small_kernel_immediate_post_ms", 0.0)
        )
        for name, value in dist_row.get("small_kernel_adjacent_pre_name_count", {}).items():
            dist_small_adj_pre_name_count[str(name)] += int(value)
        for name, value in scale_row.get("small_kernel_adjacent_pre_name_count", {}).items():
            scale_small_adj_pre_name_count[str(name)] += int(value)
        for name, value in dist_row.get("small_kernel_adjacent_pre_name_ms", {}).items():
            dist_small_adj_pre_name_ms[str(name)] += float(value)
        for name, value in scale_row.get("small_kernel_adjacent_pre_name_ms", {}).items():
            scale_small_adj_pre_name_ms[str(name)] += float(value)
        for name, value in dist_row.get("small_kernel_adjacent_post_name_count", {}).items():
            dist_small_adj_post_name_count[str(name)] += int(value)
        for name, value in scale_row.get("small_kernel_adjacent_post_name_count", {}).items():
            scale_small_adj_post_name_count[str(name)] += int(value)
        for name, value in dist_row.get("small_kernel_adjacent_post_name_ms", {}).items():
            dist_small_adj_post_name_ms[str(name)] += float(value)
        for name, value in scale_row.get("small_kernel_adjacent_post_name_ms", {}).items():
            scale_small_adj_post_name_ms[str(name)] += float(value)
        for name, value in dist_row.get("small_kernel_immediate_pre_name_count", {}).items():
            dist_small_immediate_pre_name_count[str(name)] += int(value)
        for name, value in scale_row.get("small_kernel_immediate_pre_name_count", {}).items():
            scale_small_immediate_pre_name_count[str(name)] += int(value)
        for name, value in dist_row.get("small_kernel_immediate_pre_name_ms", {}).items():
            dist_small_immediate_pre_name_ms[str(name)] += float(value)
        for name, value in scale_row.get("small_kernel_immediate_pre_name_ms", {}).items():
            scale_small_immediate_pre_name_ms[str(name)] += float(value)
        for name, value in dist_row.get("small_kernel_immediate_post_name_count", {}).items():
            dist_small_immediate_post_name_count[str(name)] += int(value)
        for name, value in scale_row.get("small_kernel_immediate_post_name_count", {}).items():
            scale_small_immediate_post_name_count[str(name)] += int(value)
        for name, value in dist_row.get("small_kernel_immediate_post_name_ms", {}).items():
            dist_small_immediate_post_name_ms[str(name)] += float(value)
        for name, value in scale_row.get("small_kernel_immediate_post_name_ms", {}).items():
            scale_small_immediate_post_name_ms[str(name)] += float(value)

        dist_name_ms = dist_row["primary_name_ms"]
        scale_name_ms = scale_row["primary_name_ms"]
        all_names = set(dist_name_ms) | set(scale_name_ms)
        for name in all_names:
            delta_by_name[name] += float(scale_name_ms.get(name, 0.0)) - float(
                dist_name_ms.get(name, 0.0)
            )

        dist_fmha_ms = sum(
            value for name, value in dist_name_ms.items() if "fmha_cutlassB" in name
        )
        scale_fmha_ms = sum(
            value for name, value in scale_name_ms.items() if "fmha_cutlassB" in name
        )
        fmha_dist_ms += dist_fmha_ms
        fmha_scale_ms += scale_fmha_ms

        rank = int(dist_row["rank"])
        fmha_us_dist_by_rank[rank].extend(dist_row["fmha_kernel_durations_us"])
        fmha_us_scale_by_rank[rank].extend(scale_row["fmha_kernel_durations_us"])
        fmha_cfg_dist.extend(dist_row["fmha_launch_configs"])
        fmha_cfg_scale.extend(scale_row["fmha_launch_configs"])
        if dist_row.get("primary_stream_id") != scale_row.get("primary_stream_id"):
            primary_stream_id_mismatch_pairs += 1
        if set(dist_row.get("fmha_stream_ids", [])) != set(scale_row.get("fmha_stream_ids", [])):
            fmha_stream_set_mismatch_pairs += 1

    gap_ms = total_scale_ms - total_dist_ms
    fmha_gap_ms = fmha_scale_ms - fmha_dist_ms
    fmha_gap_share_pct = (fmha_gap_ms / gap_ms * 100.0) if gap_ms != 0 else 0.0

    top_delta = sorted(delta_by_name.items(), key=lambda item: abs(item[1]), reverse=True)
    def _build_top_kernel_entries(
        name_ms: Dict[str, float], name_count: Dict[str, int], top_k: int = 15
    ) -> List[dict]:
        rows = [
            {
                "name": name,
                "ms": float(ms),
                "count": int(name_count.get(name, 0)),
            }
            for name, ms in name_ms.items()
        ]
        rows.sort(key=lambda item: abs(item["ms"]), reverse=True)
        return rows[:top_k]

    dist_pre_top = _build_top_kernel_entries(dist_small_adj_pre_name_ms, dist_small_adj_pre_name_count)
    scale_pre_top = _build_top_kernel_entries(
        scale_small_adj_pre_name_ms, scale_small_adj_pre_name_count
    )
    dist_post_top = _build_top_kernel_entries(
        dist_small_adj_post_name_ms, dist_small_adj_post_name_count
    )
    scale_post_top = _build_top_kernel_entries(
        scale_small_adj_post_name_ms, scale_small_adj_post_name_count
    )
    dist_immediate_pre_top = _build_top_kernel_entries(
        dist_small_immediate_pre_name_ms, dist_small_immediate_pre_name_count
    )
    scale_immediate_pre_top = _build_top_kernel_entries(
        scale_small_immediate_pre_name_ms, scale_small_immediate_pre_name_count
    )
    dist_immediate_post_top = _build_top_kernel_entries(
        dist_small_immediate_post_name_ms, dist_small_immediate_post_name_count
    )
    scale_immediate_post_top = _build_top_kernel_entries(
        scale_small_immediate_post_name_ms, scale_small_immediate_post_name_count
    )

    lines: List[str] = []
    lines.append("# Attention-family residual diagnosis")
    lines.append("")
    lines.append(f"paired_windows={len(pairs)}")
    lines.append(f"missing_windows={missing}")
    lines.append("")
    lines.append("## Primary-stream totals (paired windows)")
    lines.append(f"- dist_total_ms: {total_dist_ms:.3f}")
    lines.append(f"- scale_total_ms: {total_scale_ms:.3f}")
    lines.append(f"- gap_ms (scale-dist): {gap_ms:.3f}")
    lines.append(f"- fmha_dist_ms: {fmha_dist_ms:.3f}")
    lines.append(f"- fmha_scale_ms: {fmha_scale_ms:.3f}")
    lines.append(f"- fmha_gap_ms: {fmha_gap_ms:.3f}")
    lines.append(f"- fmha_gap_share_pct: {fmha_gap_share_pct:.2f}%")
    lines.append("")
    lines.append("## Stream and small-kernel diagnostics")
    lines.append(f"- primary_stream_id_mismatch_pairs: {primary_stream_id_mismatch_pairs}")
    lines.append(f"- fmha_stream_set_mismatch_pairs: {fmha_stream_set_mismatch_pairs}")
    lines.append(f"- dist_compute_stream_count_total: {dist_compute_stream_count}")
    lines.append(f"- scale_compute_stream_count_total: {scale_compute_stream_count}")
    lines.append(f"- dist_primary_small_kernel_count_total: {dist_small_kernel_count}")
    lines.append(f"- scale_primary_small_kernel_count_total: {scale_small_kernel_count}")
    lines.append(f"- dist_primary_small_kernel_ms_total: {dist_small_kernel_ms:.3f}")
    lines.append(f"- scale_primary_small_kernel_ms_total: {scale_small_kernel_ms:.3f}")
    lines.append(f"- dist_small_adjacent_pre_count: {dist_small_adj_pre_count}")
    lines.append(f"- scale_small_adjacent_pre_count: {scale_small_adj_pre_count}")
    lines.append(f"- dist_small_adjacent_post_count: {dist_small_adj_post_count}")
    lines.append(f"- scale_small_adjacent_post_count: {scale_small_adj_post_count}")
    lines.append(f"- dist_small_adjacent_pre_ms: {dist_small_adj_pre_ms:.3f}")
    lines.append(f"- scale_small_adjacent_pre_ms: {scale_small_adj_pre_ms:.3f}")
    lines.append(f"- dist_small_adjacent_post_ms: {dist_small_adj_post_ms:.3f}")
    lines.append(f"- scale_small_adjacent_post_ms: {scale_small_adj_post_ms:.3f}")
    lines.append(f"- dist_small_immediate_pre_count: {dist_small_immediate_pre_count}")
    lines.append(f"- scale_small_immediate_pre_count: {scale_small_immediate_pre_count}")
    lines.append(f"- dist_small_immediate_post_count: {dist_small_immediate_post_count}")
    lines.append(f"- scale_small_immediate_post_count: {scale_small_immediate_post_count}")
    lines.append(f"- dist_small_immediate_pre_ms: {dist_small_immediate_pre_ms:.3f}")
    lines.append(f"- scale_small_immediate_pre_ms: {scale_small_immediate_pre_ms:.3f}")
    lines.append(f"- dist_small_immediate_post_ms: {dist_small_immediate_post_ms:.3f}")
    lines.append(f"- scale_small_immediate_post_ms: {scale_small_immediate_post_ms:.3f}")
    lines.append("")
    lines.append("## Pre-fmha adjacent small-kernel top names")
    lines.append("### distributed")
    if dist_pre_top:
        for entry in dist_pre_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")
    lines.append("### scaling")
    if scale_pre_top:
        for entry in scale_pre_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Post-fmha adjacent small-kernel top names")
    lines.append("### distributed")
    if dist_post_top:
        for entry in dist_post_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")
    lines.append("### scaling")
    if scale_post_top:
        for entry in scale_post_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Immediate same-stream adjacent small-kernel top names")
    lines.append("### pre-fmha distributed")
    if dist_immediate_pre_top:
        for entry in dist_immediate_pre_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")
    lines.append("### pre-fmha scaling")
    if scale_immediate_pre_top:
        for entry in scale_immediate_pre_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")
    lines.append("### post-fmha distributed")
    if dist_immediate_post_top:
        for entry in dist_immediate_post_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")
    lines.append("### post-fmha scaling")
    if scale_immediate_post_top:
        for entry in scale_immediate_post_top:
            lines.append(
                f"- {entry['ms']:.3f} ms | {entry['count']} kernels | {entry['name']}"
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## fmha launch-config parity")
    lines.append(f"- dist_unique_cfg: {len(set(fmha_cfg_dist))}")
    lines.append(f"- scale_unique_cfg: {len(set(fmha_cfg_scale))}")
    lines.append(f"- cfg_sets_equal: {set(fmha_cfg_dist) == set(fmha_cfg_scale)}")
    lines.append("")
    lines.append("## fmha duration stats by rank (us)")
    lines.append("| rank | dist_count | dist_mean_us | dist_p50_us | scale_count | scale_mean_us | scale_p50_us |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    ranks = sorted(set(fmha_us_dist_by_rank) | set(fmha_us_scale_by_rank))
    fmha_duration_stats_by_rank: Dict[int, Dict[str, float]] = {}
    for rank in ranks:
        dist_vals = fmha_us_dist_by_rank.get(rank, [])
        scale_vals = fmha_us_scale_by_rank.get(rank, [])
        dist_mean = statistics.mean(dist_vals) if dist_vals else 0.0
        dist_p50 = statistics.median(dist_vals) if dist_vals else 0.0
        scale_mean = statistics.mean(scale_vals) if scale_vals else 0.0
        scale_p50 = statistics.median(scale_vals) if scale_vals else 0.0
        dist_iqr = interquartile_range(dist_vals)
        scale_iqr = interquartile_range(scale_vals)
        fmha_duration_stats_by_rank[rank] = {
            "dist_count": len(dist_vals),
            "dist_mean_us": dist_mean,
            "dist_p50_us": dist_p50,
            "dist_iqr_us": dist_iqr,
            "scale_count": len(scale_vals),
            "scale_mean_us": scale_mean,
            "scale_p50_us": scale_p50,
            "scale_iqr_us": scale_iqr,
        }
        lines.append(
            f"| {rank} | {len(dist_vals)} | {dist_mean:.1f} | {dist_p50:.1f} | "
            f"{len(scale_vals)} | {scale_mean:.1f} | {scale_p50:.1f} |"
        )

    lines.append("")
    lines.append("## fmha duration IQR by rank (us)")
    lines.append("| rank | dist_iqr_us | scale_iqr_us |")
    lines.append("|---:|---:|---:|")
    for rank in ranks:
        stats_by_rank = fmha_duration_stats_by_rank[rank]
        lines.append(
            f"| {rank} | {stats_by_rank['dist_iqr_us']:.1f} | {stats_by_rank['scale_iqr_us']:.1f} |"
        )

    lines.append("")
    lines.append("## Top kernel deltas (scale - dist, ms)")
    for name, delta_ms in top_delta[:15]:
        lines.append(f"- {delta_ms:+.3f} ms | {name}")

    payload = {
        "paired_windows": len(pairs),
        "missing_windows": missing,
        "total_dist_ms": total_dist_ms,
        "total_scale_ms": total_scale_ms,
        "gap_ms": gap_ms,
        "fmha_dist_ms": fmha_dist_ms,
        "fmha_scale_ms": fmha_scale_ms,
        "fmha_gap_ms": fmha_gap_ms,
        "fmha_gap_share_pct": fmha_gap_share_pct,
        "fmha_cfg_set_dist": sorted(set(fmha_cfg_dist)),
        "fmha_cfg_set_scale": sorted(set(fmha_cfg_scale)),
        "fmha_duration_stats_by_rank": fmha_duration_stats_by_rank,
        "primary_stream_id_mismatch_pairs": primary_stream_id_mismatch_pairs,
        "fmha_stream_set_mismatch_pairs": fmha_stream_set_mismatch_pairs,
        "dist_compute_stream_count_total": dist_compute_stream_count,
        "scale_compute_stream_count_total": scale_compute_stream_count,
        "dist_primary_small_kernel_count_total": dist_small_kernel_count,
        "scale_primary_small_kernel_count_total": scale_small_kernel_count,
        "dist_primary_small_kernel_ms_total": dist_small_kernel_ms,
        "scale_primary_small_kernel_ms_total": scale_small_kernel_ms,
        "dist_small_adjacent_pre_count": dist_small_adj_pre_count,
        "scale_small_adjacent_pre_count": scale_small_adj_pre_count,
        "dist_small_adjacent_post_count": dist_small_adj_post_count,
        "scale_small_adjacent_post_count": scale_small_adj_post_count,
        "dist_small_adjacent_pre_ms": dist_small_adj_pre_ms,
        "scale_small_adjacent_pre_ms": scale_small_adj_pre_ms,
        "dist_small_adjacent_post_ms": dist_small_adj_post_ms,
        "scale_small_adjacent_post_ms": scale_small_adj_post_ms,
        "dist_small_adjacent_pre_top_names": dist_pre_top,
        "scale_small_adjacent_pre_top_names": scale_pre_top,
        "dist_small_adjacent_post_top_names": dist_post_top,
        "scale_small_adjacent_post_top_names": scale_post_top,
        "dist_small_immediate_pre_count": dist_small_immediate_pre_count,
        "scale_small_immediate_pre_count": scale_small_immediate_pre_count,
        "dist_small_immediate_post_count": dist_small_immediate_post_count,
        "scale_small_immediate_post_count": scale_small_immediate_post_count,
        "dist_small_immediate_pre_ms": dist_small_immediate_pre_ms,
        "scale_small_immediate_pre_ms": scale_small_immediate_pre_ms,
        "dist_small_immediate_post_ms": dist_small_immediate_post_ms,
        "scale_small_immediate_post_ms": scale_small_immediate_post_ms,
        "dist_small_immediate_pre_top_names": dist_immediate_pre_top,
        "scale_small_immediate_pre_top_names": scale_immediate_pre_top,
        "dist_small_immediate_post_top_names": dist_immediate_post_top,
        "scale_small_immediate_post_top_names": scale_immediate_post_top,
        "top_kernel_delta": top_delta[:50],
    }
    return "\n".join(lines), payload


def run(args: argparse.Namespace) -> int:
    ranks = parse_csv_ints(args.ranks)
    segment_values = parse_csv_strs(args.segment_values) if args.segment_values else None

    dist_conn = sqlite3.connect(str(args.dist_sqlite))
    scale_conn = sqlite3.connect(str(args.scale_sqlite))
    try:
        dist_windows = load_phase_windows(
            dist_conn,
            label_prefix=args.label_prefix,
            ranks=ranks,
            op=args.op,
            mg_state=args.mg_state,
            stage_id=args.stage_id,
            phase=args.phase,
            segment_key=args.segment_key,
            segment_values=segment_values,
        )
        scale_windows = load_phase_windows(
            scale_conn,
            label_prefix=args.label_prefix,
            ranks=ranks,
            op=args.op,
            mg_state=args.mg_state,
            stage_id=args.stage_id,
            phase=args.phase,
            segment_key=args.segment_key,
            segment_values=segment_values,
        )
        dist_kernels = load_kernels_for_windows(dist_conn, dist_windows)
        scale_kernels = load_kernels_for_windows(scale_conn, scale_windows)
    finally:
        dist_conn.close()
        scale_conn.close()

    dist_rows = [
        summarize_window(
            window,
            dist_kernels,
            small_kernel_threshold_us=args.small_kernel_threshold_us,
            adjacency_window_us=args.adjacency_window_us,
        )
        for window in dist_windows
    ]
    scale_rows = [
        summarize_window(
            window,
            scale_kernels,
            small_kernel_threshold_us=args.small_kernel_threshold_us,
            adjacency_window_us=args.adjacency_window_us,
        )
        for window in scale_windows
    ]
    pairs, missing = pair_windows(dist_rows, scale_rows)

    report_text, payload = build_report(pairs, missing)
    if args.segment_key is not None:
        report_text += (
            "\n\n"
            f"segment_filter={args.segment_key}:{args.segment_values if args.segment_values else 'ANY'}"
        )
    print(report_text)

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(report_text + "\n")
        print(f"[INFO] Report written to {args.report_path}")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] JSON written to {args.json_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze attention-family residual from sqlite traces")
    parser.add_argument("--dist-sqlite", type=Path, required=True)
    parser.add_argument("--scale-sqlite", type=Path, required=True)
    parser.add_argument("--label-prefix", type=str, default="cmd_trace")
    parser.add_argument("--ranks", type=str, default="4,5,6,7")
    parser.add_argument("--op", type=str, default="backward_step")
    parser.add_argument("--mg-state", type=str, default="steady")
    parser.add_argument("--stage-id", type=str, default="1")
    parser.add_argument("--phase", type=str, default="compute")
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--segment-values", type=str, default=None)
    parser.add_argument("--small-kernel-threshold-us", type=float, default=60.0)
    parser.add_argument("--adjacency-window-us", type=float, default=200.0)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--json-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.dist_sqlite.exists():
        print(f"[ERROR] Missing dist sqlite: {args.dist_sqlite}")
        return 2
    if not args.scale_sqlite.exists():
        print(f"[ERROR] Missing scale sqlite: {args.scale_sqlite}")
        return 2

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
