"""Unit tests for MoE model detection and simulate fail-fast behavior."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import MODE_SIMULATE, SimulatorEngine


def _build_engine(running_mode: str = MODE_SIMULATE, exp_size: int = 1) -> SimulatorEngine:
    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.running_mode = running_mode
    engine.mpu = SimpleNamespace(
        exp_size=exp_size,
        ep_size=1,
        pp_size=1,
        tp_size=1,
        dp_size=1,
        world_size=1,
    )
    return engine


def test_detect_model_type_uses_topology_signal_in_simulate_mode() -> None:
    engine = _build_engine(exp_size=2)

    is_moe = engine._detect_model_type(
        trace_stages_dict=None,
        stages_or_wranks_dict=None,
        torch_graph_stage_op_dict=None,
    )

    assert is_moe is True


def test_detect_model_type_uses_stage_operation_signal() -> None:
    engine = _build_engine(exp_size=1)
    fake_stage = SimpleNamespace(
        operations_list=[
            SimpleNamespace(
                name="exp_all_to_all",
                group_kind=None,
                sub_operations=[],
            )
        ]
    )

    is_moe = engine._detect_model_type(
        trace_stages_dict=None,
        stages_or_wranks_dict={0: fake_stage},
        torch_graph_stage_op_dict=None,
    )

    assert is_moe is True


def test_detect_model_type_uses_database_profile_signal() -> None:
    engine = _build_engine(exp_size=1)
    torch_graph_stage_op_dict = {
        0: {
            "exp_allgather": {"duration": 0.1},
        }
    }

    is_moe = engine._detect_model_type(
        trace_stages_dict=None,
        stages_or_wranks_dict=None,
        torch_graph_stage_op_dict=torch_graph_stage_op_dict,
    )

    assert is_moe is True


def test_detect_model_type_returns_dense_without_moe_signal() -> None:
    engine = _build_engine(exp_size=1)

    is_moe = engine._detect_model_type(
        trace_stages_dict={},
        stages_or_wranks_dict={},
        torch_graph_stage_op_dict={},
    )

    assert is_moe is False


def test_init_3d_parallel_fail_fast_when_exp_size_gt_one_but_detected_dense() -> None:
    engine = _build_engine(exp_size=2)
    engine.optimization_enabled = True

    rank_instances_dict = {0: object()}
    stages_or_wranks_dict = {0: object()}

    # Force detector output to validate fail-fast branch directly.
    engine._detect_model_type = lambda *args, **kwargs: False

    with pytest.raises(ValueError, match="exp_size > 1"):
        engine._init_3d_parallel_all_ranks(
            stages_or_wranks_dict=stages_or_wranks_dict,
            rank_instances_dict=rank_instances_dict,
            trace_stages_dict={},
            torch_graph_stage_op_dict={},
        )
