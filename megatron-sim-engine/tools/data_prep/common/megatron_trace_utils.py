"""Shared Megatron trace parsing helpers for data-prep utilities."""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Mapping, Tuple


_TRACE_LINE_RE = re.compile(r"^rank:(\d+):([^\(]+)\((.*)\)$")
_STABLE_DDP_COMM_ALIGNMENT_FIELDS: Tuple[str, ...] = (
    "stage_id",
    "mg_state",
    "comm_func",
    "buffer_id",
    "bucket_id",
    "bucket_offset",
    "bucket_numel_unpadded",
    "bucket_numel",
    "param_count",
    "logical_stream_role",
    "grad_dtype",
)


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


def split_top_level_trace_fields(raw_fields: str) -> List[Tuple[str, str]]:
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


def parse_trace_value(raw_value: str):
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
    for key, value in split_top_level_trace_fields(raw_fields):
        parsed_fields[key] = parse_trace_value(value)
    return rank, event_name, parsed_fields


def build_ddp_comm_alignment_key(fields: Mapping[str, object]) -> str:
    parts: List[str] = []
    for field_name in _STABLE_DDP_COMM_ALIGNMENT_FIELDS:
        value = fields.get(field_name)
        if field_name == "bucket_numel_unpadded" and value is None:
            value = fields.get("bucket_numel")
        if field_name == "bucket_numel" and value is None:
            value = fields.get("bucket_numel_unpadded")
        if value is None:
            raise ValueError(f"ddp_grad_comm is missing stable alignment field {field_name!r}")
        parts.append(f"{field_name}={value}")
    return "|".join(parts)
