"""Type definitions for communication-cost (CC) backends."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


def _normalize_int_sequence(raw: Optional[Any], field_name: str) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    parsed = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            raise ValueError(f"{field_name} must be a sequence of integers, got: {raw!r}")

    if not isinstance(parsed, Iterable) or isinstance(parsed, (bytes, str)):
        raise ValueError(f"{field_name} must be a sequence of integers, got: {type(parsed)!r}")

    try:
        return tuple(int(item) for item in parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains non-integer values: {parsed!r}") from exc


@dataclass(frozen=True)
class CommunicationPredictionRequest:
    """Normalized input required by all CC backends.

    Attributes:
        comm_group: Global ranks participating in this communication operation.
        op_name: Original operation name from trace/schedule (e.g., dp_allreduce).
        data_size_bytes: Message size in bytes for backend prediction.
        group_kind: Optional high-level group tag (dp/tp/ep/pp/exp/...)
        domain_dims: Optional explicit domain dims for topology-aware backends.
        tensor_shape: Optional tensor shape metadata from source trace/schedule.
        tensor_dtype: Optional tensor dtype string from source trace/schedule.
        mpu_info: Optional simulator topology object.
        metadata: Extra backend-specific metadata.
    """

    comm_group: Tuple[int, ...]
    op_name: str
    data_size_bytes: int
    group_kind: Optional[str] = None
    domain_dims: Tuple[str, ...] = ()
    tensor_shape: Optional[Tuple[int, ...]] = None
    tensor_dtype: Optional[str] = None
    mpu_info: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        *,
        comm_group: Sequence[int],
        op_name: str,
        data_size_bytes: int,
        group_kind: Optional[str] = None,
        domain_dims: Sequence[str] = (),
        tensor_shape: Optional[Sequence[int]] = None,
        tensor_dtype: Optional[str] = None,
        mpu_info: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "CommunicationPredictionRequest":
        normalized_group = _normalize_int_sequence(comm_group, "comm_group")
        if not normalized_group:
            raise ValueError("comm_group must not be empty")
        if data_size_bytes < 0:
            raise ValueError("data_size_bytes must be >= 0")
        return cls(
            comm_group=normalized_group,
            op_name=str(op_name),
            data_size_bytes=int(data_size_bytes),
            group_kind=group_kind,
            domain_dims=tuple(domain_dims),
            tensor_shape=_normalize_int_sequence(tensor_shape, "tensor_shape"),
            tensor_dtype=tensor_dtype,
            mpu_info=mpu_info,
            metadata=dict(metadata or {}),
        )

    @property
    def group_size(self) -> int:
        return len(self.comm_group)
