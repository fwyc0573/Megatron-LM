"""Integration checks for MoE simulate all-rank selection and TP barrier behavior."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import MODE_SIMULATE, Operation, SimulatorEngine, TimelinesManager


def test_moe_simulate_uses_all_ranks_and_keeps_tp_barrier() -> None:
    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.running_mode = MODE_SIMULATE
    engine.optimization_enabled = True
    engine.is_moe_model = True
    engine.mpu = SimpleNamespace(pp_size=2, tp_size=2, dp_size=2)

    rank_instances_dict = {rank_id: object() for rank_id in range(8)}
    selected_ranks = engine._select_optimization_ranks(rank_instances_dict)

    assert selected_ranks == set(rank_instances_dict.keys())

    manager = TimelinesManager.__new__(TimelinesManager)
    manager.running_mode = MODE_SIMULATE
    manager.optimization_enabled = True
    manager.is_moe_model = True
    manager.selected_ranks = set(rank_instances_dict.keys())
    manager.comm_matching_relationship = {}
    manager.global_waiting_pool = {}

    timeline = SimpleNamespace(
        wrank_id=0,
        stage_rank=SimpleNamespace(tp_groups=[0, 1, 2, 3]),
    )
    operation = Operation(
        name="tp_allreduce",
        op_kind="comm",
        wrank_id=0,
        batch_id=0,
        mg_state="steady",
    )

    matching_name, matching_ranks, stage_offset = manager._get_comm_matching_operation_name_and_wrank_id(
        operation=operation,
        comm_kind="allreduce",
        parallel_kind="tp",
        timeline=timeline,
    )

    assert matching_name == "tp_allreduce"
    assert matching_ranks == [1, 2, 3]
    assert stage_offset is None

    # TP barrier must remain active in MoE simulate mode.
    can_start, _, _ = manager._can_start_matching_comm_group_process(
        operation_list=[operation],
        comm_kind="allreduce",
        parallel_kind="tp",
        timeline=timeline,
    )
    assert can_start is False
