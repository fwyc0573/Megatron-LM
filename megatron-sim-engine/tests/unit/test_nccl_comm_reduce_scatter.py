"""Regression tests for analytical reduce-scatter communication timing."""

from src.core.comm_sim.nccl_comm import get_comm_op_exc_time


def test_reduce_scatter_uses_ring_profile_and_is_positive():
    duration_ms = get_comm_op_exc_time([0, 1, 2, 3], 1024, "reduce_scatter")
    assert duration_ms > 0.0


def test_reduce_scatter_is_distinct_from_allreduce_for_same_payload():
    reduce_scatter_ms = get_comm_op_exc_time([0, 1, 2, 3], 1 << 20, "reduce_scatter")
    allreduce_ms = get_comm_op_exc_time([0, 1, 2, 3], 1 << 20, "allreduce")
    assert reduce_scatter_ms == allreduce_ms * 0.5
