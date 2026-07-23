#!/usr/bin/env python3
"""Build a trace-shaped replayable schedule for compressed Megatron pp2 traces."""

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


_TRACE_LINE_RE = re.compile(r"^rank:(\d+):([^\(]+)\((.*)\)$")


@dataclass(frozen=True)
class TraceScheduleOp:
    rank: int
    stage_id: int
    batch_id: int
    mg_state: str
    op_name: str


def _require_existing_dir(path: Path, field_name: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{field_name} must be an existing directory: {path}")


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
    fields: Dict[str, object] = {}
    for key, value in _split_top_level_trace_fields(match.group(3)):
        fields[key] = _parse_trace_value(value)
    return rank, event_name, fields


def _format_schedule_line(
    *,
    stage_id: int,
    op_name: str,
    batch_id: int,
    mg_state: str,
    description: object,
    group_kind: object,
    input_shape: object,
    input_dtype: object,
) -> str:
    return (
        f"stage:{stage_id}:{op_name}(batch_id={batch_id}, mg_state={mg_state}, duration=None, "
        f"description={description}, group_kind={group_kind}, input__shape={input_shape}, input__dtype={input_dtype})"
    )


def _normalized_description(op_name: str):
    if op_name == "dp_allreduce":
        return "model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas"
    if op_name == "ep_allreduce":
        return "_allreduce_word_embedding_grads"
    return None


def _normalized_group_kind(op_name: str):
    if op_name in {"send_forward", "recv_forward", "send_backward", "recv_backward"}:
        return "pp"
    if op_name == "dp_allreduce":
        return "dp"
    if op_name == "ep_allreduce":
        return "ep"
    return None


def _collect_stage_sequences(trace_dir: Path) -> Dict[int, Dict[int, List[TraceScheduleOp]]]:
    stage_sequences: Dict[int, Dict[int, List[TraceScheduleOp]]] = defaultdict(dict)
    trace_files = sorted(trace_dir.glob("*.txt"))
    if not trace_files:
        raise ValueError(f"trace_dir contains no .txt files: {trace_dir}")

    for trace_file in trace_files:
        file_ops: List[TraceScheduleOp] = []
        for raw_line in trace_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rank, event_name, fields = parse_trace_line(line)
            if event_name == "ddp_grad_comm":
                continue
            stage_id = fields.get("stage_id")
            batch_id = fields.get("batch_id")
            mg_state = fields.get("mg_state")
            if stage_id is None or batch_id is None or not isinstance(mg_state, str) or not mg_state:
                raise ValueError(f"Missing required top-level trace fields in {trace_file}: {line}")
            file_ops.append(
                TraceScheduleOp(
                    rank=rank,
                    stage_id=int(stage_id),
                    batch_id=int(batch_id),
                    mg_state=str(mg_state),
                    op_name=event_name,
                )
            )
        if not file_ops:
            continue
        stage_ids = {op.stage_id for op in file_ops}
        if len(stage_ids) != 1:
            raise ValueError(f"Trace file {trace_file} mixes multiple stage_ids: {sorted(stage_ids)}")
        batch_ids = {op.batch_id for op in file_ops}
        if len(batch_ids) != 1:
            raise ValueError(f"Trace file {trace_file} mixes multiple batch_id values: {sorted(batch_ids)}")
        stage_id = next(iter(stage_ids))
        rank = file_ops[0].rank
        stage_sequences[stage_id][rank] = file_ops

    if not stage_sequences:
        raise ValueError(f"No top-level trace operations found in {trace_dir}")
    return dict(stage_sequences)


def _validate_stage_consistency(stage_sequences: Mapping[int, Mapping[int, Sequence[TraceScheduleOp]]]) -> None:
    for stage_id, rank_to_ops in stage_sequences.items():
        reference_rank = min(rank_to_ops)
        reference_signature = [(op.op_name, op.mg_state, op.batch_id) for op in rank_to_ops[reference_rank]]
        for rank, ops in rank_to_ops.items():
            signature = [(op.op_name, op.mg_state, op.batch_id) for op in ops]
            if signature != reference_signature:
                raise ValueError(
                    f"Inconsistent top-level op sequence for stage {stage_id}: reference rank {reference_rank} != rank {rank}"
                )


def _representative_stage_ops(stage_sequences: Mapping[int, Mapping[int, Sequence[TraceScheduleOp]]]) -> Dict[int, List[TraceScheduleOp]]:
    return {stage_id: list(rank_to_ops[min(rank_to_ops)]) for stage_id, rank_to_ops in stage_sequences.items()}


def _build_stage_lines(
    *,
    stage_id: int,
    stage_ops: Sequence[TraceScheduleOp],
    pp_size: int,
    pp_shape: Sequence[int],
    pipeline_dtype: str,
) -> List[str]:
    if not stage_ops:
        raise ValueError(f"Stage {stage_id} has no top-level operations")
    if stage_id not in {0, 1}:
        raise ValueError(f"pp2 schedule builder only supports stage_id 0/1, got {stage_id}")

    forward_op = next((op for op in stage_ops if op.op_name == "forward_step"), None)
    backward_op = next((op for op in stage_ops if op.op_name == "backward_step"), None)
    if forward_op is None or backward_op is None:
        raise ValueError(f"Stage {stage_id} must contain forward_step and backward_step")

    lines: List[str] = []
    if stage_id > 0:
        lines.append(
            _format_schedule_line(
                stage_id=stage_id,
                op_name="recv_forward",
                batch_id=stage_ops[0].batch_id,
                mg_state="help",
                description=None,
                group_kind="pp",
                input_shape=list(pp_shape),
                input_dtype=pipeline_dtype,
            )
        )

    for op in stage_ops:
        if stage_id == 0 and op.op_name == "backward_step":
            lines.append(
                _format_schedule_line(
                    stage_id=stage_id,
                    op_name="recv_backward",
                    batch_id=op.batch_id,
                    mg_state=op.mg_state,
                    description=None,
                    group_kind="pp",
                    input_shape=list(pp_shape),
                    input_dtype=pipeline_dtype,
                )
            )

        lines.append(
            _format_schedule_line(
                stage_id=stage_id,
                op_name=op.op_name,
                batch_id=op.batch_id,
                mg_state=op.mg_state,
                description=_normalized_description(op.op_name),
                group_kind=_normalized_group_kind(op.op_name),
                input_shape=None,
                input_dtype=None,
            )
        )

        if stage_id == 0 and op.op_name == "forward_step":
            lines.append(
                _format_schedule_line(
                    stage_id=stage_id,
                    op_name="send_forward",
                    batch_id=op.batch_id,
                    mg_state=op.mg_state,
                    description=None,
                    group_kind="pp",
                    input_shape=list(pp_shape),
                    input_dtype=pipeline_dtype,
                )
            )
        if stage_id == pp_size - 1 and op.op_name == "backward_step":
            lines.append(
                _format_schedule_line(
                    stage_id=stage_id,
                    op_name="send_backward",
                    batch_id=op.batch_id,
                    mg_state=op.mg_state,
                    description=None,
                    group_kind="pp",
                    input_shape=list(pp_shape),
                    input_dtype=pipeline_dtype,
                )
            )
    return lines


def build_trace_shaped_pp2_schedule(
    *,
    trace_dir: Path,
    output_dir: Path,
    pp_size: int,
    seq_length: int,
    micro_batch_size: int,
    hidden_size: int,
    pipeline_dtype: str,
) -> Dict[int, Path]:
    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir)
    _require_existing_dir(trace_dir, "trace_dir")
    if pp_size != 2:
        raise ValueError(f"Only pp_size=2 is supported for trace-shaped compressed schedule generation, got {pp_size}")
    if seq_length <= 0 or micro_batch_size <= 0 or hidden_size <= 0:
        raise ValueError("seq_length, micro_batch_size, and hidden_size must all be > 0")
    if not isinstance(pipeline_dtype, str) or not pipeline_dtype:
        raise ValueError("pipeline_dtype must be a non-empty string")

    stage_sequences = _collect_stage_sequences(trace_dir)
    _validate_stage_consistency(stage_sequences)
    stage_ops = _representative_stage_ops(stage_sequences)
    missing_stages = [stage_id for stage_id in (0, 1) if stage_id not in stage_ops]
    if missing_stages:
        raise ValueError(f"Missing representative compressed traces for stages: {missing_stages}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pp_shape = [int(seq_length), int(micro_batch_size), int(hidden_size)]
    output_paths: Dict[int, Path] = {}
    for stage_id in (0, 1):
        lines = _build_stage_lines(
            stage_id=stage_id,
            stage_ops=stage_ops[stage_id],
            pp_size=pp_size,
            pp_shape=pp_shape,
            pipeline_dtype=pipeline_dtype,
        )
        path = output_dir / f"stage{stage_id}_trace_shaped_scheduling_plan.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output_paths[stage_id] = path
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build trace-shaped replayable pp2 schedule files from compressed Megatron trace files.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--pipeline-dtype", default="torch.float16")
    args = parser.parse_args()

    output_paths = build_trace_shaped_pp2_schedule(
        trace_dir=Path(args.trace_dir),
        output_dir=Path(args.output_dir),
        pp_size=args.pp_size,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        hidden_size=args.hidden_size,
        pipeline_dtype=args.pipeline_dtype,
    )
    for stage_id, path in sorted(output_paths.items()):
        print(f"stage{stage_id}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
