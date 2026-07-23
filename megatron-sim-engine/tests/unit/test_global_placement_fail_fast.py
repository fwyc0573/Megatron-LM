"""Unit tests for global placement fail-fast validation in SimulatorEngine."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import Operation, SimulatorEngine


def test_global_placement_fail_fast_reports_missing_fields() -> None:
    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.cc_backend = SimpleNamespace(backend_name="collective-sim", placement_mode="global")
    engine.mpu = SimpleNamespace(
        world_size=32,
        tp_size=4,
        dp_size=8,
        pp_size=2,
        exp_size=8,
        dp_groups=[[0, 4, 8, 12, 16, 20, 24, 28]],
        tp_groups=[[0, 1, 2, 3]],
        pp_groups=[[0, 16]],
        ep_groups=[[0, 16]],
        exp_groups=[[0, 4, 8, 12, 16, 20, 24, 28]],
        dp_modulo_exp_groups=[[0]],
        cp_groups=None,
    )

    cp_op = Operation(
        name="cp_reducescatter",
        op_kind="comm",
        group_kind="cp",
        wrank_id=0,
        mg_state="steady",
    )

    engine.timeline_manager = SimpleNamespace(
        stages_timeline_process_dict={0: SimpleNamespace(waiting_queue=[cp_op])},
        _get_comm_operation_kind_and_parallel_dimension=lambda _op: ("reducescatter", "cp"),
        _get_comm_group_for_operation=lambda _op: None,
    )

    with pytest.raises(ValueError) as exc:
        engine.validate_global_placement_requirements()

    msg = str(exc.value)
    assert "placement_mode=global" in msg
    assert "mpu_info.cp_groups" in msg
    assert "comm_group.cp_ranks" in msg


def test_global_placement_validation_passes_when_metadata_complete() -> None:
    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.cc_backend = SimpleNamespace(backend_name="collective-sim", placement_mode="global")
    engine.mpu = SimpleNamespace(
        world_size=32,
        tp_size=4,
        dp_size=8,
        pp_size=1,
        exp_size=1,
        dp_groups=[[0, 4, 8, 12, 16, 20, 24, 28]],
        tp_groups=[[0, 1, 2, 3]],
        pp_groups=[[0]],
        ep_groups=[[0]],
        exp_groups=[[0]],
        dp_modulo_exp_groups=[[0]],
        cp_groups=[[0]],
    )

    dp_op = Operation(
        name="dp_allreduce",
        op_kind="comm",
        group_kind="dp",
        wrank_id=0,
        mg_state="steady",
    )

    engine.timeline_manager = SimpleNamespace(
        stages_timeline_process_dict={0: SimpleNamespace(waiting_queue=[dp_op])},
        _get_comm_operation_kind_and_parallel_dimension=lambda _op: ("allreduce", "dp"),
        _get_comm_group_for_operation=lambda _op: [0, 4, 8, 12, 16, 20, 24, 28],
    )

    engine.validate_global_placement_requirements()
