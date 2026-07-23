"""Unit tests for parallel topology group construction."""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.static_graphs.parallel_group_manager import ParallelGroupManager


def test_parallel_group_shapes_basic() -> None:
    manager = ParallelGroupManager(
        local_size=4,
        world_size=8,
        pp_size=2,
        tp_size=2,
        exp_size=1,
    )

    groups = manager.get_all_groups()
    assert len(groups["pp_groups"]) == 4
    assert all(len(group) == 2 for group in groups["pp_groups"])

    assert len(groups["tp_groups"]) == 4
    assert all(len(group) == 2 for group in groups["tp_groups"])

    assert len(groups["dp_groups"]) == 4
    assert all(len(group) == 2 for group in groups["dp_groups"])


def test_parallel_group_invalid_exp_size_fail_fast() -> None:
    try:
        ParallelGroupManager(
            local_size=4,
            world_size=8,
            pp_size=2,
            tp_size=2,
            exp_size=3,
        )
    except ValueError as exc:
        assert "dp_size" in str(exc)
        return

    raise AssertionError("Expected ValueError for invalid exp_size divisibility")
