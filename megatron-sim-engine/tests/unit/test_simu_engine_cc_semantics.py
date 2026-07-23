"""Unit tests for TimelinesManager communication semantic alignment."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import Operation, TimelinesManager


class _CaptureBackend:
    def __init__(self) -> None:
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return 3.25


def _build_manager_for_unit_tests(backend: _CaptureBackend) -> TimelinesManager:
    manager = TimelinesManager.__new__(TimelinesManager)
    manager.cc_backend = backend
    manager.cc_estimator = None
    manager.mpu_info = SimpleNamespace(world_size=2)
    manager.optimization_enabled = False
    manager.is_moe_model = False
    manager.selected_ranks = set()
    manager.comm_matching_relationship = {}
    return manager


def test_get_comm_kind_recognizes_reducescatter() -> None:
    manager = TimelinesManager.__new__(TimelinesManager)
    op = Operation(
        name="tp_reducescatter",
        op_kind="comm",
        group_kind=None,
        mg_state="steady",
    )

    comm_kind, parallel_kind = manager._get_comm_operation_kind_and_parallel_dimension(op)

    assert comm_kind == "reducescatter"
    assert parallel_kind == "tp"


@pytest.mark.parametrize(
    ("name", "expected_kind", "expected_group"),
    [
        ("tp_all_to_all", "all_to_all", "tp"),
        ("tp_allgather", "allgather", "tp"),
        ("tp_reduce_scatter", "reducescatter", "tp"),
        ("tp_broadcast", "broadcast", "tp"),
    ],
)
def test_get_comm_kind_accepts_trace_collective_aliases(name, expected_kind, expected_group) -> None:
    manager = TimelinesManager.__new__(TimelinesManager)
    operation = Operation(name=name, op_kind="comm", group_kind="tp", mg_state="steady")

    comm_kind, parallel_kind = manager._get_comm_operation_kind_and_parallel_dimension(operation)

    assert (comm_kind, parallel_kind) == (expected_kind, expected_group)


def test_calculate_comm_duration_passes_standardized_p2p_metadata() -> None:
    backend = _CaptureBackend()
    manager = _build_manager_for_unit_tests(backend)
    manager._get_comm_group_for_operation = lambda _op: None
    manager._get_comm_data_size = lambda _op, _size: 4096

    send_op = Operation(
        name="send_forward",
        op_kind="comm",
        group_kind="pp",
        wrank_id=4,
        mg_state="warmup",
        tensor_shape=[8, 16],
        tensor_dtype="torch.float16",
    )
    recv_op = Operation(
        name="recv_forward",
        op_kind="comm",
        group_kind="pp",
        wrank_id=12,
        mg_state="warmup",
        tensor_shape=[8, 16],
        tensor_dtype="torch.float16",
    )

    manager._calculate_comm_duration([send_op, recv_op])

    assert backend.requests, "cc backend should receive one request"
    request = backend.requests[-1]
    assert request.comm_group == (4, 12)
    assert request.metadata["p2p_src_index"] == 0
    assert request.metadata["p2p_dst_index"] == 1
    assert request.metadata["p2p_direction"] == "0->1"
    assert request.metadata["mg_state"] == "warmup"
    assert send_op.duration == 3.25
    assert recv_op.duration == 3.25


def test_get_comm_matching_for_tp_reducescatter_without_explicit_mapping() -> None:
    backend = _CaptureBackend()
    manager = _build_manager_for_unit_tests(backend)

    operation = Operation(
        name="tp_reducescatter",
        op_kind="comm",
        group_kind="tp",
        wrank_id=0,
        mg_state="steady",
    )
    timeline = SimpleNamespace(
        wrank_id=0,
        stage_rank=SimpleNamespace(tp_groups=[0, 1, 2, 3]),
    )

    matching_name, matching_ranks, stage_offset = manager._get_comm_matching_operation_name_and_wrank_id(
        operation=operation,
        comm_kind="reducescatter",
        parallel_kind="tp",
        timeline=timeline,
    )

    assert matching_name == "tp_reducescatter"
    assert matching_ranks == [1, 2, 3]
    assert stage_offset is None


def test_get_comm_group_supports_flat_ep_and_exp_groups() -> None:
    manager = TimelinesManager.__new__(TimelinesManager)
    stage_rank = SimpleNamespace(
        dp_groups=[0, 4, 8, 12],
        pp_groups=[0, 8],
        tp_groups=[0, 1, 2, 3],
        ep_groups=[0, 4, 8, 12],
        exp_groups=[0, 4, 8, 12],
        cp_groups=None,
        dp_modulo_exp_groups=[0, 8],
    )
    manager.stages_timeline_process_dict = {0: SimpleNamespace(wrank_id=0, stage_rank=stage_rank)}

    ep_op = Operation(
        name="ep_allreduce",
        op_kind="comm",
        group_kind="ep",
        wrank_id=0,
        mg_state="steady",
    )
    exp_op = Operation(
        name="exp_all_to_all",
        op_kind="comm",
        group_kind="exp",
        wrank_id=0,
        mg_state="steady",
    )
    exp_dp_op = Operation(
        name="exp_dp_allreduce",
        op_kind="comm",
        group_kind="exp_dp",
        wrank_id=0,
        mg_state="steady",
    )
    pp_op = Operation(
        name="send_forward",
        op_kind="comm",
        group_kind="pp",
        wrank_id=0,
        mg_state="warmup",
    )

    assert manager._get_comm_group_for_operation(ep_op) == [0, 4, 8, 12]
    assert manager._get_comm_group_for_operation(exp_op) == [0, 4, 8, 12]
    assert manager._get_comm_group_for_operation(exp_dp_op) == [0, 8]
    assert manager._get_comm_group_for_operation(pp_op) == [0, 8]
