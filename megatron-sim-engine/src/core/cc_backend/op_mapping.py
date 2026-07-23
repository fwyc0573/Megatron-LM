"""Operation-name and domain mapping helpers for CC backends."""

from __future__ import annotations

from typing import Optional, Tuple


_P2P_OPS = {
    "send_forward",
    "recv_forward",
    "send_backward",
    "recv_backward",
    "send_recv",
    "sendgrad",
    "recvgrad",
    "sendactivation",
    "recvactivation",
}

_GROUP_KIND_TO_DOMAIN = {
    "tp": "TP",
    "dp": "DP",
    "exp_dp": "DP",
    "ep": "EP",
    "exp": "EP",
    "cp": "CP",
}


def infer_collective_kind(op_name: str) -> str:
    """Map simulator operation names to canonical collective kinds.

    Returns one of: allreduce, allgather, reducescatter, alltoall, p2p, broadcast.
    """

    normalized = op_name.lower()

    if normalized in _P2P_OPS or normalized.startswith("send_") or normalized.startswith("recv_"):
        return "p2p"
    if "all_to_all" in normalized or "alltoall" in normalized:
        return "alltoall"
    if "reduce_scatter" in normalized or "reducescatter" in normalized:
        return "reducescatter"
    if "allgather" in normalized:
        return "allgather"
    if "allreduce" in normalized:
        return "allreduce"
    if "broadcast" in normalized:
        return "broadcast"

    raise ValueError(f"Unsupported communication operation name: {op_name}")


def infer_domain_dims(group_kind: Optional[str], op_name: str) -> Tuple[str, ...]:
    """Infer domain dims for topology-aware backends.

    Domain dims must align with collective-sim schema: TP/CP/DP/EP.
    """

    normalized_group = (group_kind or "").lower().strip()
    if normalized_group in _GROUP_KIND_TO_DOMAIN:
        return (_GROUP_KIND_TO_DOMAIN[normalized_group],)

    normalized_name = op_name.lower()
    if "tp" in normalized_name:
        return ("TP",)
    if "exp" in normalized_name or "ep" in normalized_name:
        return ("EP",)
    if "dp" in normalized_name:
        return ("DP",)

    collective_kind = infer_collective_kind(op_name)
    if collective_kind == "p2p":
        raise ValueError(
            f"Cannot infer domain dims for p2p op={op_name}, group_kind={group_kind}. "
            "Please provide explicit domain dims. For pipeline p2p traffic, set "
            "communication.backend_options.collective-sim.pp_domain_dim."
        )

    raise ValueError(
        f"Cannot infer domain dims for op={op_name}, group_kind={group_kind}. "
        "Please provide explicit domain dims in backend request/config."
    )


def normalize_analytical_comm_func(collective_kind: str) -> str:
    """Map canonical collective kinds to `comm_sim.nccl_comm` function names."""

    if collective_kind == "allreduce":
        return "allreduce"
    if collective_kind == "allgather":
        return "allgather"
    if collective_kind == "reducescatter":
        return "reduce_scatter"
    if collective_kind == "alltoall":
        return "all_to_all"
    if collective_kind == "p2p":
        return "send_recv"
    if collective_kind == "broadcast":
        return "broadcast"

    raise ValueError(f"Unsupported collective kind for analytical backend: {collective_kind}")
