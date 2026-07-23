#!/usr/bin/env python3
"""Build offline slowdown assets for DDP backward-only slowdown replay."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


_TRACE_LINE_RE = re.compile(r"^rank:(\d+):([^\(]+)\((.*)\)$")
_REQUIRED_KERNEL_FEATURES = (
    "Compute throughput",
    "Memory throughput",
    "DRAM throughput",
    "Achieved occupancy",
    "Maximum occupancy",
    "L1 hit rate",
    "L2 hit rate",
)
_GLOBAL_TID_LOCAL_MASK = (1 << 24) - 1


def _default_model_path() -> str:
    return str(
        Path(__file__).resolve().parents[4]
        / "Echo-slowdown"
        / "training_testing"
        / "output"
        / "xgb_model.json"
    )


def _default_scaler_path() -> str:
    return str(
        Path(__file__).resolve().parents[4]
        / "Echo-slowdown"
        / "training_testing"
        / "output"
        / "standard_scaler.json"
    )


def _require_existing_dir(path: Path, field_name: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory: {path}")


def _require_existing_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field_name} must be an existing file: {path}")


def _is_top_level_trace_field_boundary(params: str, comma_index: int) -> bool:
    cursor = comma_index + 1
    while cursor < len(params) and params[cursor].isspace():
        cursor += 1
    if cursor >= len(params):
        return False
    if not (params[cursor].isalpha() or params[cursor] == "_"):
        return False

    cursor += 1
    while cursor < len(params) and (params[cursor].isalnum() or params[cursor] == "_"):
        cursor += 1
    while cursor < len(params) and params[cursor].isspace():
        cursor += 1
    return cursor < len(params) and params[cursor] == "="


def _split_top_level_trace_fields(raw_fields: str) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    index = 0
    length = len(raw_fields)

    while index < length:
        while index < length and raw_fields[index] in {",", " ", "\t", "\n"}:
            index += 1
        if index >= length:
            break

        key_start = index
        while index < length and raw_fields[index] != "=":
            index += 1
        if index >= length:
            raise ValueError(f"Malformed Megatron trace field: {raw_fields[key_start:]}")

        key = raw_fields[key_start:index].strip()
        if not key:
            raise ValueError(f"Empty Megatron trace field key in: {raw_fields}")

        index += 1
        value_start = index
        bracket_depth = 0
        brace_depth = 0
        paren_depth = 0
        quote_char = None

        while index < length:
            char = raw_fields[index]
            if quote_char is not None:
                if char == "\\":
                    index += 2
                    continue
                if char == quote_char:
                    quote_char = None
                index += 1
                continue

            if char in {"'", '"'}:
                quote_char = char
                index += 1
                continue

            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif (
                char == ","
                and bracket_depth == 0
                and brace_depth == 0
                and paren_depth == 0
                and _is_top_level_trace_field_boundary(raw_fields, index)
            ):
                break

            index += 1

        value = raw_fields[value_start:index].strip()
        fields.append((key, value))

        if index < length and raw_fields[index] == ",":
            index += 1

    return fields


def _parse_trace_value(raw_value: str):
    value = raw_value.strip()
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value):
        return float(value)
    if (
        (value.startswith("[") and value.endswith("]"))
        or (value.startswith("{") and value.endswith("}"))
        or (value.startswith("(") and value.endswith(")"))
        or (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def parse_trace_line(line: str) -> Tuple[int, str, Dict[str, object]]:
    match = _TRACE_LINE_RE.match(line.strip())
    if match is None:
        raise ValueError(f"Invalid trace line: {line[:120]}")
    rank = int(match.group(1))
    event_name = match.group(2).strip()
    raw_fields = match.group(3)
    parsed_fields: Dict[str, object] = {}
    for key, value in _split_top_level_trace_fields(raw_fields):
        parsed_fields[key] = _parse_trace_value(value)
    return rank, event_name, parsed_fields


def load_trace_metadata(trace_dir: Path):
    _require_existing_dir(trace_dir, "trace_dir")
    trace_files = sorted(trace_dir.glob("*.txt"))
    if not trace_files:
        raise ValueError(f"trace_dir contains no .txt trace files: {trace_dir}")

    backward_by_cmd_uid: Dict[str, Dict[str, object]] = {}
    ddp_markers_by_cmd_uid: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    seen_comm_uids_by_cmd: Dict[str, set] = defaultdict(set)

    for trace_file in trace_files:
        for raw_line in trace_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rank, event_name, fields = parse_trace_line(line)
            if event_name == "backward_step":
                cmd_uid = fields.get("cmd_uid")
                if not isinstance(cmd_uid, str) or not cmd_uid:
                    raise ValueError(f"backward_step is missing cmd_uid in {trace_file}")
                if cmd_uid in backward_by_cmd_uid:
                    raise ValueError(f"Duplicate backward_step cmd_uid={cmd_uid} in {trace_file}")
                timestamp = fields.get("timestamp")
                duration = fields.get("duration")
                mg_state = fields.get("mg_state")
                if timestamp is None or duration is None:
                    raise ValueError(f"backward_step {cmd_uid} is missing timestamp/duration in {trace_file}")
                if not isinstance(mg_state, str) or not mg_state:
                    raise ValueError(f"backward_step {cmd_uid} is missing mg_state in {trace_file}")
                backward_by_cmd_uid[cmd_uid] = {
                    "cmd_uid": cmd_uid,
                    "rank": rank,
                    "stage_id": int(fields.get("stage_id")),
                    "batch_id": int(fields.get("batch_id")),
                    "mg_state": mg_state,
                    "baseline_duration_ms": round(float(duration), 6),
                    "start_timestamp_ms": round(float(timestamp) - float(duration), 6),
                    "end_timestamp_ms": round(float(timestamp), 6),
                }
                continue
            if event_name != "ddp_grad_comm":
                continue

            trigger_cmd_uid = fields.get("trigger_cmd_uid")
            comm_uid = fields.get("comm_uid")
            launch_timestamp_ms = fields.get("launch_timestamp_ms")
            if not isinstance(trigger_cmd_uid, str) or not trigger_cmd_uid:
                raise ValueError(f"ddp_grad_comm in {trace_file} is missing trigger_cmd_uid")
            if not isinstance(comm_uid, str) or not comm_uid:
                raise ValueError(f"ddp_grad_comm in {trace_file} is missing comm_uid")
            if launch_timestamp_ms is None:
                raise ValueError(f"ddp_grad_comm {comm_uid} is missing launch_timestamp_ms in {trace_file}")
            if comm_uid in seen_comm_uids_by_cmd[trigger_cmd_uid]:
                raise ValueError(
                    f"Duplicate ddp_grad_comm comm_uid={comm_uid} for trigger_cmd_uid={trigger_cmd_uid}"
                )
            seen_comm_uids_by_cmd[trigger_cmd_uid].add(comm_uid)
            ddp_markers_by_cmd_uid[trigger_cmd_uid].append(
                {
                    "comm_uid": comm_uid,
                    "baseline_launch_timestamp_ms": round(float(launch_timestamp_ms), 6),
                    "bucket_id": fields.get("bucket_id"),
                    "buffer_id": fields.get("buffer_id"),
                }
            )
    return backward_by_cmd_uid, ddp_markers_by_cmd_uid


def derive_global_pid(global_tid: int) -> int:
    return global_tid & (~_GLOBAL_TID_LOCAL_MASK)


def parse_nvtx_label(label: str, prefix: str) -> Optional[Dict[str, str]]:
    prefix_token = f"{prefix}|"
    if not label.startswith(prefix_token):
        return None
    mapping: Dict[str, str] = {}
    for field in label[len(prefix_token) :].split("|"):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        mapping[key] = value
    required = ("rank", "op", "state", "stage", "batch", "iter", "cmd_uid")
    if any(key not in mapping for key in required):
        return None
    return mapping


def load_nvtx_rows(sqlite_path: Path, label_prefix: str):
    _require_existing_file(sqlite_path, "nsys_sqlite")
    if not label_prefix:
        raise ValueError("label_prefix must be a non-empty string")

    conn = sqlite3.connect(str(sqlite_path))
    try:
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

        parent_by_cmd_uid: Dict[str, Dict[str, object]] = {}
        compute_windows_by_cmd_uid: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for start_ns, end_ns, label, global_tid in rows:
            parsed = parse_nvtx_label(label, label_prefix)
            if parsed is None:
                continue
            cmd_uid = parsed["cmd_uid"]
            if parsed["op"] != "backward_step":
                continue
            row = {
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "global_pid": derive_global_pid(int(global_tid)),
                "rank": int(parsed["rank"]),
                "op": parsed["op"],
                "mg_state": parsed["state"],
                "stage_id": int(parsed["stage"]),
                "batch_id": int(parsed["batch"]),
                "iter_id": int(parsed["iter"]),
                "cmd_uid": cmd_uid,
                "label": label,
                "phase": parsed.get("phase"),
            }
            if row["phase"] == "compute":
                compute_windows_by_cmd_uid[cmd_uid].append((row["start_ns"], row["end_ns"]))
                continue
            if row["phase"] is not None:
                continue
            if cmd_uid in parent_by_cmd_uid:
                raise ValueError(f"Duplicate parent NVTX backward_step range for cmd_uid={cmd_uid}")
            parent_by_cmd_uid[cmd_uid] = row

        if not parent_by_cmd_uid:
            raise ValueError(f"No backward_step NVTX rows found in {sqlite_path}")

        pid_values = sorted({int(row["global_pid"]) for row in parent_by_cmd_uid.values()})
        placeholders = ",".join("?" for _ in pid_values)
        min_start_ns = min(int(row["start_ns"]) for row in parent_by_cmd_uid.values())
        max_end_ns = max(int(row["end_ns"]) for row in parent_by_cmd_uid.values())
        kernel_rows = conn.execute(
            f"""
            SELECT
                k.start,
                k.end,
                k.streamId,
                k.globalPid,
                s_short.value AS short_name,
                s_dem.value AS demangled_name,
                k.shortName,
                k.demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            LEFT JOIN StringIds AS s_short ON k.shortName = s_short.id
            LEFT JOIN StringIds AS s_dem ON k.demangledName = s_dem.id
            WHERE k.end > ?
              AND k.start < ?
              AND k.globalPid IN ({placeholders})
            ORDER BY k.globalPid ASC, k.start ASC
            """,
            [min_start_ns, max_end_ns, *pid_values],
        ).fetchall()
    finally:
        conn.close()

    kernels_by_pid: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for start_ns, end_ns, stream_id, global_pid, short_name, demangled_name, short_name_id, demangled_name_id in kernel_rows:
        kernel_name = short_name if short_name not in (None, "") else demangled_name
        if kernel_name in (None, ""):
            kernel_name = f"id:{demangled_name_id if demangled_name_id is not None else short_name_id}"
        kernels_by_pid[int(global_pid)].append(
            {
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "stream_id": int(stream_id),
                "kernel_name": str(kernel_name),
                "is_comm": "nccl" in str(kernel_name).lower(),
            }
        )
    return parent_by_cmd_uid, compute_windows_by_cmd_uid, kernels_by_pid


def load_kernel_features(ncu_metrics_csv: Path) -> Dict[str, Dict[str, float]]:
    _require_existing_file(ncu_metrics_csv, "ncu_metrics_csv")
    df = pd.read_csv(ncu_metrics_csv)
    if "Kernel Name" not in df.columns:
        raise ValueError(f"NCU metrics CSV is missing 'Kernel Name': {ncu_metrics_csv}")
    missing = [name for name in _REQUIRED_KERNEL_FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"NCU metrics CSV is missing required columns: {missing}")

    kernel_names = df["Kernel Name"].astype(str).str.strip()
    if kernel_names.eq("").any() or kernel_names.eq("nan").any():
        raise ValueError("NCU metrics CSV contains an empty Kernel Name")
    df = df.assign(**{"Kernel Name": kernel_names})

    for feature_name in _REQUIRED_KERNEL_FEATURES:
        df[feature_name] = pd.to_numeric(df[feature_name], errors="raise")

    grouped = (
        df.groupby("Kernel Name", sort=True, as_index=False)[list(_REQUIRED_KERNEL_FEATURES)]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    kernel_features: Dict[str, Dict[str, float]] = {}
    for _, row in grouped.iterrows():
        kernel_name = str(row["Kernel Name"])
        kernel_features[kernel_name] = {
            feature_name: round(float(row[feature_name]), 6)
            for feature_name in _REQUIRED_KERNEL_FEATURES
        }
    return kernel_features


def overlap_ns(window_start: int, window_end: int, kernel_start: int, kernel_end: int) -> Tuple[int, int, int]:
    start = max(window_start, kernel_start)
    end = min(window_end, kernel_end)
    return max(0, end - start), start, end



def collect_required_kernel_names(
    *,
    trace_dir: Path,
    nsys_sqlite: Path,
    label_prefix: str,
) -> List[str]:
    backward_by_cmd_uid, ddp_markers_by_cmd_uid = load_trace_metadata(trace_dir)
    parent_by_cmd_uid, compute_windows_by_cmd_uid, kernels_by_pid = load_nvtx_rows(nsys_sqlite, label_prefix)

    orphan_cmd_uids = sorted(set(ddp_markers_by_cmd_uid) - set(backward_by_cmd_uid))
    if orphan_cmd_uids:
        raise ValueError(
            f"Found ddp_grad_comm markers without matching backward_step trace cmd_uid values: {orphan_cmd_uids}"
        )

    required_kernel_names = set()
    for cmd_uid, backward_trace in backward_by_cmd_uid.items():
        if cmd_uid not in parent_by_cmd_uid:
            raise ValueError(f"Missing NVTX parent backward_step range for cmd_uid={cmd_uid}")
        parent = parent_by_cmd_uid[cmd_uid]
        compute_windows = list(compute_windows_by_cmd_uid.get(cmd_uid, []))
        if not compute_windows:
            raise ValueError(
                f"Missing phase=compute NVTX windows for backward_step cmd_uid={cmd_uid}; rerun with --trace-kernel-ground-truth-phase"
            )
        if int(parent["rank"]) != int(backward_trace["rank"]):
            raise ValueError(f"Rank mismatch for cmd_uid={cmd_uid}")
        if int(parent["stage_id"]) != int(backward_trace["stage_id"]):
            raise ValueError(f"Stage mismatch for cmd_uid={cmd_uid}")
        if int(parent["batch_id"]) != int(backward_trace["batch_id"]):
            raise ValueError(f"Batch mismatch for cmd_uid={cmd_uid}")

        pid = int(parent["global_pid"])
        kernels = list(kernels_by_pid.get(pid, []))
        found_kernel = False
        for kernel in kernels:
            if bool(kernel["is_comm"]):
                continue
            for compute_start_ns, compute_end_ns in compute_windows:
                ov_ns, _, _ = overlap_ns(
                    compute_start_ns,
                    compute_end_ns,
                    int(kernel["start_ns"]),
                    int(kernel["end_ns"]),
                )
                if ov_ns <= 0:
                    continue
                required_kernel_names.add(str(kernel["kernel_name"]))
                found_kernel = True
                break
        if not found_kernel:
            raise ValueError(f"No compute kernels found for backward_step cmd_uid={cmd_uid}")

    return sorted(required_kernel_names)

def build_blueprints(
    backward_by_cmd_uid: Mapping[str, Mapping[str, object]],
    ddp_markers_by_cmd_uid: Mapping[str, Sequence[Mapping[str, object]]],
    parent_by_cmd_uid: Mapping[str, Mapping[str, object]],
    compute_windows_by_cmd_uid: Mapping[str, Sequence[Tuple[int, int]]],
    kernels_by_pid: Mapping[int, Sequence[Mapping[str, object]]],
    kernel_features_all: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, float]]]:
    orphan_cmd_uids = sorted(set(ddp_markers_by_cmd_uid) - set(backward_by_cmd_uid))
    if orphan_cmd_uids:
        raise ValueError(
            f"Found ddp_grad_comm markers without matching backward_step trace cmd_uid values: {orphan_cmd_uids}"
        )

    blueprints: Dict[str, Dict[str, object]] = {}
    used_kernel_features: Dict[str, Dict[str, float]] = {}
    for cmd_uid, backward_trace in backward_by_cmd_uid.items():
        if cmd_uid not in parent_by_cmd_uid:
            raise ValueError(f"Missing NVTX parent backward_step range for cmd_uid={cmd_uid}")
        parent = parent_by_cmd_uid[cmd_uid]
        compute_windows = list(compute_windows_by_cmd_uid.get(cmd_uid, []))
        if not compute_windows:
            raise ValueError(
                f"Missing phase=compute NVTX windows for backward_step cmd_uid={cmd_uid}; rerun with --trace-kernel-ground-truth-phase"
            )
        if int(parent["rank"]) != int(backward_trace["rank"]):
            raise ValueError(f"Rank mismatch for cmd_uid={cmd_uid}")
        if int(parent["stage_id"]) != int(backward_trace["stage_id"]):
            raise ValueError(f"Stage mismatch for cmd_uid={cmd_uid}")
        if int(parent["batch_id"]) != int(backward_trace["batch_id"]):
            raise ValueError(f"Batch mismatch for cmd_uid={cmd_uid}")

        pid = int(parent["global_pid"])
        kernels = list(kernels_by_pid.get(pid, []))
        source_kernel_entries: List[Dict[str, object]] = []
        for kernel in kernels:
            if bool(kernel["is_comm"]):
                continue
            for compute_start_ns, compute_end_ns in compute_windows:
                ov_ns, ov_start_ns, _ = overlap_ns(
                    compute_start_ns,
                    compute_end_ns,
                    int(kernel["start_ns"]),
                    int(kernel["end_ns"]),
                )
                if ov_ns <= 0:
                    continue
                kernel_name = str(kernel["kernel_name"])
                if kernel_name not in kernel_features_all:
                    raise ValueError(
                        f"Missing NCU kernel features for kernel_name={kernel_name!r} referenced by cmd_uid={cmd_uid}"
                    )
                used_kernel_features[kernel_name] = dict(kernel_features_all[kernel_name])
                source_kernel_entries.append(
                    {
                        "kernel_name": kernel_name,
                        "stream_id": int(kernel["stream_id"]),
                        "source_start_ns": int(ov_start_ns),
                        "source_end_ns": int(ov_start_ns + ov_ns),
                    }
                )
        if not source_kernel_entries:
            raise ValueError(f"No compute kernels found for backward_step cmd_uid={cmd_uid}")

        source_kernel_entries.sort(
            key=lambda item: (
                int(item["source_start_ns"]),
                int(item["stream_id"]),
                int(item["source_end_ns"]),
                str(item["kernel_name"]),
            )
        )
        parent_start_ns = int(parent["start_ns"])
        serial_cursor_ns = None
        kernel_entries: List[Dict[str, object]] = []
        for source_entry in source_kernel_entries:
            source_start_ns = int(source_entry["source_start_ns"])
            source_end_ns = int(source_entry["source_end_ns"])
            projected_start_ns = source_start_ns
            if serial_cursor_ns is not None:
                projected_start_ns = max(projected_start_ns, serial_cursor_ns)
            if projected_start_ns >= source_end_ns:
                raise ValueError(
                    "Compute kernel interval is fully covered by earlier compute intervals: "
                    f"cmd_uid={cmd_uid}, kernel_name={source_entry['kernel_name']!r}, "
                    f"stream_id={source_entry['stream_id']}"
                )

            source_duration_ns = source_end_ns - source_start_ns
            projected_duration_ns = source_end_ns - projected_start_ns
            trimmed_ns = projected_start_ns - source_start_ns
            kernel_entries.append(
                {
                    "kernel_name": str(source_entry["kernel_name"]),
                    "stream_id": int(source_entry["stream_id"]),
                    "source_start_offset_ms": round(
                        (source_start_ns - parent_start_ns) / 1_000_000.0,
                        6,
                    ),
                    "source_baseline_duration_ms": round(source_duration_ns / 1_000_000.0, 6),
                    "serial_projection_trimmed_ms": round(trimmed_ns / 1_000_000.0, 6),
                    "start_offset_ms": round(
                        (projected_start_ns - parent_start_ns) / 1_000_000.0,
                        6,
                    ),
                    "baseline_duration_ms": round(projected_duration_ns / 1_000_000.0, 6),
                }
            )
            serial_cursor_ns = source_end_ns

        launch_markers: List[Dict[str, object]] = []
        seen_comm_uids = set()
        backward_start_timestamp_ms = float(backward_trace["start_timestamp_ms"])
        for marker in ddp_markers_by_cmd_uid.get(cmd_uid, []):
            comm_uid = marker["comm_uid"]
            if comm_uid in seen_comm_uids:
                raise ValueError(f"Duplicate launch marker comm_uid={comm_uid} for cmd_uid={cmd_uid}")
            seen_comm_uids.add(comm_uid)
            offset_ms = round(float(marker["baseline_launch_timestamp_ms"]) - backward_start_timestamp_ms, 6)
            if offset_ms < -0.05:
                raise ValueError(
                    f"Negative launch marker offset {offset_ms} for cmd_uid={cmd_uid}, comm_uid={comm_uid}"
                )
            launch_markers.append(
                {
                    "comm_uid": comm_uid,
                    "baseline_offset_ms": max(0.0, offset_ms),
                    "bucket_id": marker.get("bucket_id"),
                    "buffer_id": marker.get("buffer_id"),
                }
            )
        launch_markers.sort(key=lambda item: (float(item["baseline_offset_ms"]), str(item["comm_uid"])))
        blueprints[cmd_uid] = {
            "rank": int(backward_trace["rank"]),
            "stage_id": int(backward_trace["stage_id"]),
            "batch_id": int(backward_trace["batch_id"]),
            "iter_id": int(parent["iter_id"]),
            "mg_state": str(backward_trace["mg_state"]),
            "baseline_duration_ms": round(float(backward_trace["baseline_duration_ms"]), 6),
            "kernels": kernel_entries,
            "launch_markers": launch_markers,
        }
    return blueprints, used_kernel_features


def write_assets(
    output_dir: Path,
    *,
    model_path: str,
    scaler_path: str,
    label_prefix: str,
    trace_dir: Path,
    nsys_sqlite: Path,
    ncu_metrics_csv: Path,
    kernel_features: Mapping[str, Mapping[str, float]],
    backward_kernel_blueprints: Mapping[str, Mapping[str, object]],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "ddp_backward_only",
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "label_prefix": str(label_prefix),
        "clip_negative_slowdown": True,
        "generator_version": "v3",
        "kernel_timeline_model": "serial_interval_union_projection_v1",
        "source_trace_dir": str(trace_dir),
        "source_nsys_sqlite": str(nsys_sqlite),
        "source_ncu_metrics": str(ncu_metrics_csv),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "kernel_features.json").write_text(
        json.dumps(kernel_features, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "backward_kernel_blueprints.json").write_text(
        json.dumps(backward_kernel_blueprints, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_assets(args: argparse.Namespace) -> None:
    trace_dir = Path(args.trace_dir)
    nsys_sqlite = Path(args.nsys_sqlite)
    ncu_metrics_csv = Path(args.ncu_metrics_csv)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)

    _require_existing_dir(trace_dir, "trace_dir")
    _require_existing_file(nsys_sqlite, "nsys_sqlite")
    _require_existing_file(ncu_metrics_csv, "ncu_metrics_csv")
    _require_existing_file(model_path, "model_path")
    _require_existing_file(scaler_path, "scaler_path")
    if not args.label_prefix:
        raise ValueError("label_prefix must be a non-empty string")

    backward_by_cmd_uid, ddp_markers_by_cmd_uid = load_trace_metadata(trace_dir)
    parent_by_cmd_uid, compute_windows_by_cmd_uid, kernels_by_pid = load_nvtx_rows(
        nsys_sqlite, args.label_prefix
    )
    kernel_features_all = load_kernel_features(ncu_metrics_csv)
    blueprints, used_kernel_features = build_blueprints(
        backward_by_cmd_uid=backward_by_cmd_uid,
        ddp_markers_by_cmd_uid=ddp_markers_by_cmd_uid,
        parent_by_cmd_uid=parent_by_cmd_uid,
        compute_windows_by_cmd_uid=compute_windows_by_cmd_uid,
        kernels_by_pid=kernels_by_pid,
        kernel_features_all=kernel_features_all,
    )
    write_assets(
        output_dir=output_dir,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        label_prefix=args.label_prefix,
        trace_dir=trace_dir,
        nsys_sqlite=nsys_sqlite,
        ncu_metrics_csv=ncu_metrics_csv,
        kernel_features=used_kernel_features,
        backward_kernel_blueprints=blueprints,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DDP slowdown assets for sim-engine")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--nsys-sqlite", required=True)
    parser.add_argument("--ncu-metrics-csv", required=True)
    parser.add_argument("--label-prefix", default="cmd_trace")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=_default_model_path())
    parser.add_argument("--scaler-path", default=_default_scaler_path())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_assets(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
