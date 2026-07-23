"""Unit tests for MoE rank selection behavior in simulator optimization."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import MODE_SIMULATE, SimulatorEngine


def _build_engine() -> SimulatorEngine:
    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.running_mode = MODE_SIMULATE
    engine.optimization_enabled = True
    engine.mpu = SimpleNamespace(pp_size=2, tp_size=1, dp_size=4)
    return engine


def test_select_optimization_ranks_dense_keeps_representative_pp_ranks() -> None:
    engine = _build_engine()
    engine.is_moe_model = False

    rank_instances_dict = {rank_id: object() for rank_id in range(8)}
    selected = engine._select_optimization_ranks(rank_instances_dict)

    assert selected == {0, 4}


def test_select_optimization_ranks_moe_uses_all_ranks() -> None:
    engine = _build_engine()
    engine.is_moe_model = True

    rank_instances_dict = {rank_id: object() for rank_id in range(8)}
    selected = engine._select_optimization_ranks(rank_instances_dict)

    assert selected == set(rank_instances_dict.keys())


def test_select_optimization_ranks_returns_all_when_optimization_disabled() -> None:
    engine = _build_engine()
    engine.optimization_enabled = False
    engine.is_moe_model = False

    rank_instances_dict = {rank_id: object() for rank_id in range(8)}
    selected = engine._select_optimization_ranks(rank_instances_dict)

    assert selected == set(rank_instances_dict.keys())
