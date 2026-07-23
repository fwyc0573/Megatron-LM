#!/usr/bin/env python3
"""Build a replayable PP schedule from compressed top-level Megatron traces."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_prep.common.megatron_trace_utils import parse_trace_line


_REQUIRED_BASE_OPS_BY_STAGE_POSITION = {
    "first": ("get_batch", "forward_step", "backward_step", "dp_allreduce", "optimizer_step"),
    "last": ("get_batch", "forward_step", "loss_func", "backward_step", "dp_allreduce", "optimizer_step"),
}
_TOP_LEVEL_KEEP_OPS = {
    "get_batch",
    "forward_step",
    "loss_func",
    "backward_step",
    "dp_allreduce",
    "ep_allreduce",
    "optimizer_step",
}


@dataclass(frozen=True)
class TraceEvent:
    wrank_id: int
    stage_id: int
    event_name: str
    fields: Mapping[str, object]


def _require_existing_dir(path: Path, field_name: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory: {path}")


def _normalize_dtype(dtype: str) -> str:
    value = str(dtype).strip()
    if not value:
        raise ValueError("pipeline_dtype must be a non-empty string")
    return value


def _format_field_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _format_stage_line(stage_id: int, event_name: str, fields: Mapping[str, object]) -> str:
    ordered_keys = [
        "batch_id",
        "mg_state",
        "duration",
        "description",
        "group_kind",
        "op_semantics",
        "finalize_base_duration_ms",
        "input__shape",
        "input__dtype",
    ]
    rendered = []
    for key in ordered_keys:
        rendered.append(f"{key}={_format_field_value(fields.get(key))}")
    for key in sorted(fields):
        if key in ordered_keys:
            continue
        rendered.append(f"{key}={_format_field_value(fields[key])}")
    return f"stage:{stage_id}:{event_name}(" + ", ".join(rendered) + ")"


def _collect_trace_events_by_wrank(trace_dir: Path) -> Dict[int, List[TraceEvent]]:
    _require_existing_dir(trace_dir, "trace_dir")
    trace_files = sorted(trace_dir.glob("*.txt"))
    if not trace_files:
        raise ValueError(f"trace_dir contains no .txt files: {trace_dir}")

    events_by_wrank: Dict[int, List[TraceEvent]] = defaultdict(list)
    for trace_file in trace_files:
        for raw_line in trace_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            wrank_id, event_name, fields = parse_trace_line(line)
            if event_name == "ddp_grad_comm":
                continue
            if event_name not in _TOP_LEVEL_KEEP_OPS:
                continue
            stage_id = fields.get("stage_id")
            if stage_id is None:
                raise ValueError(f"Trace event {event_name} in {trace_file} is missing stage_id")
            events_by_wrank[wrank_id].append(
                TraceEvent(
                    wrank_id=wrank_id,
                    stage_id=int(stage_id),
                    event_name=event_name,
                    fields=dict(fields),
                )
            )
    if not events_by_wrank:
        raise ValueError(f"No top-level trace events found in {trace_dir}")
    return events_by_wrank


def _pick_representative_wranks(events_by_wrank: Mapping[int, Sequence[TraceEvent]], pp_size: int) -> Dict[int, int]:
    if pp_size != 2:
        raise ValueError(f"build_trace_shaped_pp_schedule v1 only supports pp_size=2, got {pp_size}")

    stage_to_wranks: Dict[int, List[int]] = defaultdict(list)
    for wrank_id, events in events_by_wrank.items():
        stage_ids = {event.stage_id for event in events}
        if len(stage_ids) != 1:
            raise ValueError(f"Trace file for wrank={wrank_id} mixes stage ids: {sorted(stage_ids)}")
        stage_to_wranks[next(iter(stage_ids))].append(wrank_id)

    expected_stage_ids = list(range(pp_size))
    missing_stage_ids = [stage_id for stage_id in expected_stage_ids if stage_id not in stage_to_wranks]
    if missing_stage_ids:
        raise ValueError(f"Trace directory is missing stage ids: {missing_stage_ids}")

    return {stage_id: min(stage_to_wranks[stage_id]) for stage_id in expected_stage_ids}


def _extract_stage_backbone(events: Sequence[TraceEvent], *, stage_position: str) -> List[TraceEvent]:
    required_ops = list(_REQUIRED_BASE_OPS_BY_STAGE_POSITION[stage_position])
    seen_ops = set()
    backbone: List[TraceEvent] = []
    for event in events:
        if event.event_name not in _TOP_LEVEL_KEEP_OPS:
            continue
        if event.event_name in seen_ops:
            continue
        backbone.append(event)
        seen_ops.add(event.event_name)
        if event.event_name == "optimizer_step":
            break

    missing = [op_name for op_name in required_ops if op_name not in seen_ops]
    if missing:
        raise ValueError(f"Stage trace is missing required top-level op(s): {missing}")
    return backbone


def _find_event(backbone: Sequence[TraceEvent], event_name: str) -> TraceEvent:
    for event in backbone:
        if event.event_name == event_name:
            return event
    raise ValueError(f"Backbone is missing event {event_name}")


def _synth_pp_fields(*, batch_id: int, mg_state: str, tensor_shape: Sequence[int], tensor_dtype: str) -> Dict[str, object]:
    return {
        "batch_id": int(batch_id),
        "mg_state": str(mg_state),
        "duration": None,
        "description": None,
        "group_kind": "pp",
        "op_semantics": None,
        "finalize_base_duration_ms": None,
        "input__shape": list(tensor_shape),
        "input__dtype": str(tensor_dtype),
    }


def _trace_event_to_schedule_fields(event: TraceEvent) -> Dict[str, object]:
    fields = dict(event.fields)
    fields["duration"] = None
    return {
        "batch_id": fields.get("batch_id"),
        "mg_state": fields.get("mg_state"),
        "duration": None,
        "description": fields.get("description"),
        "group_kind": fields.get("group_kind"),
        "op_semantics": fields.get("op_semantics"),
        "finalize_base_duration_ms": fields.get("finalize_base_duration_ms"),
        "input__shape": fields.get("input__shape"),
        "input__dtype": fields.get("input__dtype"),
    }


def build_stage_schedule_lines(
    *,
    stage_id: int,
    representative_events: Sequence[TraceEvent],
    pp_size: int,
    seq_length: int,
    micro_batch_size: int,
    hidden_size: int,
    pipeline_dtype: str,
) -> List[str]:
    if stage_id not in {0, pp_size - 1}:
        raise ValueError(f"v1 only supports first/last stage replay, got stage_id={stage_id}")
    stage_position = "first" if stage_id == 0 else "last"
    backbone = _extract_stage_backbone(representative_events, stage_position=stage_position)
    tensor_shape = [int(seq_length), int(micro_batch_size), int(hidden_size)]

    get_batch_event = _find_event(backbone, "get_batch")
    forward_event = _find_event(backbone, "forward_step")
    backward_event = _find_event(backbone, "backward_step")

    lines: List[str] = []
    finalize_events = [event for event in backbone if event.event_name in {"dp_allreduce", "ep_allreduce", "optimizer_step"}]

    if stage_position == "first":
        lines.append(_format_stage_line(stage_id, get_batch_event.event_name, _trace_event_to_schedule_fields(get_batch_event)))
        lines.append(_format_stage_line(stage_id, forward_event.event_name, _trace_event_to_schedule_fields(forward_event)))
        lines.append(
            _format_stage_line(
                stage_id,
                "send_forward",
                _synth_pp_fields(
                    batch_id=int(forward_event.fields.get("batch_id", 0)),
                    mg_state=str(forward_event.fields.get("mg_state")),
                    tensor_shape=tensor_shape,
                    tensor_dtype=pipeline_dtype,
                ),
            )
        )
        lines.append(
            _format_stage_line(
                stage_id,
                "recv_backward",
                _synth_pp_fields(
                    batch_id=int(backward_event.fields.get("batch_id", 0)),
                    mg_state=str(backward_event.fields.get("mg_state")),
                    tensor_shape=tensor_shape,
                    tensor_dtype=pipeline_dtype,
                ),
            )
        )
        lines.append(_format_stage_line(stage_id, backward_event.event_name, _trace_event_to_schedule_fields(backward_event)))
    else:
        lines.append(
            _format_stage_line(
                stage_id,
                "recv_forward",
                _synth_pp_fields(
                    batch_id=int(get_batch_event.fields.get("batch_id", 0)),
                    mg_state="help",
                    tensor_shape=tensor_shape,
                    tensor_dtype=pipeline_dtype,
                ),
            )
        )
        lines.append(_format_stage_line(stage_id, get_batch_event.event_name, _trace_event_to_schedule_fields(get_batch_event)))
        lines.append(_format_stage_line(stage_id, forward_event.event_name, _trace_event_to_schedule_fields(forward_event)))
        loss_event = _find_event(backbone, "loss_func")
        lines.append(_format_stage_line(stage_id, loss_event.event_name, _trace_event_to_schedule_fields(loss_event)))
        lines.append(_format_stage_line(stage_id, backward_event.event_name, _trace_event_to_schedule_fields(backward_event)))
        lines.append(
            _format_stage_line(
                stage_id,
                "send_backward",
                _synth_pp_fields(
                    batch_id=int(backward_event.fields.get("batch_id", 0)),
                    mg_state=str(backward_event.fields.get("mg_state")),
                    tensor_shape=tensor_shape,
                    tensor_dtype=pipeline_dtype,
                ),
            )
        )

    for event in finalize_events:
        lines.append(_format_stage_line(stage_id, event.event_name, _trace_event_to_schedule_fields(event)))
    return lines


def build_trace_shaped_schedule(
    *,
    trace_dir: Path,
    output_dir: Path,
    pp_size: int,
    seq_length: int,
    micro_batch_size: int,
    hidden_size: int,
    pipeline_dtype: str,
) -> List[Path]:
    events_by_wrank = _collect_trace_events_by_wrank(trace_dir)
    representative_wranks = _pick_representative_wranks(events_by_wrank, pp_size=pp_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: List[Path] = []
    for stage_id in range(pp_size):
        wrank_id = representative_wranks[stage_id]
        lines = build_stage_schedule_lines(
            stage_id=stage_id,
            representative_events=events_by_wrank[wrank_id],
            pp_size=pp_size,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            hidden_size=hidden_size,
            pipeline_dtype=_normalize_dtype(pipeline_dtype),
        )
        output_path = output_dir / f"stage{stage_id}_trace_shaped_scheduling_plan.txt"
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written_files.append(output_path)
    return written_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trace-shaped PP schedule from compressed traces.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--pipeline-dtype", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    written_files = build_trace_shaped_schedule(
        trace_dir=Path(args.trace_dir),
        output_dir=Path(args.output_dir),
        pp_size=int(args.pp_size),
        seq_length=int(args.seq_length),
        micro_batch_size=int(args.micro_batch_size),
        hidden_size=int(args.hidden_size),
        pipeline_dtype=str(args.pipeline_dtype),
    )
    for path in written_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
