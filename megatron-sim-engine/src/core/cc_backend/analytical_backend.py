"""Analytical CC backend based on `comm_sim.nccl_comm`."""

from __future__ import annotations

from src.core.comm_sim.nccl_comm import get_comm_op_exc_time

from .base import CCBackend
from .op_mapping import infer_collective_kind, normalize_analytical_comm_func
from .registry import register_cc_backend
from .types import CommunicationPredictionRequest


@register_cc_backend("analytical")
class AnalyticalCCBackend(CCBackend):
    backend_name = "analytical"

    def predict(self, request: CommunicationPredictionRequest) -> float:
        self.validate_request(request)

        collective_kind = infer_collective_kind(request.op_name)
        comm_func = normalize_analytical_comm_func(collective_kind)

        duration_ms = get_comm_op_exc_time(
            comm_group=list(request.comm_group),
            data_size=int(request.data_size_bytes),
            comm_func=comm_func,
        )
        return float(duration_ms)
