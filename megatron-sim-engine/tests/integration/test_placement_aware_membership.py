"""Placement-aware membership validation for communication groups and backend scenarios."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cc_backend.collective_sim_backend import CollectiveSimCCBackend
from src.core.cc_backend.types import CommunicationPredictionRequest
from src.core.simulator_config import create_h800_sxm_ib_config
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager


def _node_distribution(ranks: list[int] | tuple[int, ...], gpus_per_node: int) -> dict[int, int]:
    distribution: dict[int, int] = {}
    for rank in ranks:
        node_id = int(rank) // int(gpus_per_node)
        distribution[node_id] = distribution.get(node_id, 0) + 1
    return distribution


def _mocked_backend(monkeypatch):
    captured = {}

    def fake_predictor(scenario, repo_root=None):
        captured["scenario"] = scenario
        captured["repo_root"] = repo_root
        return {"predicted_time_ms": 5.0}

    monkeypatch.setattr(
        CollectiveSimCCBackend,
        "_resolve_repo_root",
        lambda self: Path(PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim"),
    )
    monkeypatch.setattr(
        CollectiveSimCCBackend,
        "_load_predictor",
        staticmethod(lambda _repo_root: fake_predictor),
    )

    config = create_h800_sxm_ib_config()
    config.communication.backend_options["collective-sim"].update(
        {
            "placement_mode": "global",
            "strict_mpu_alignment": True,
        }
    )
    return CollectiveSimCCBackend(config), captured


def test_dp_group_32gpus_uses_4_ranks_per_node_across_8_nodes() -> None:
    manager = ParallelGroupManager(
        local_size=4,
        world_size=32,
        pp_size=1,
        tp_size=1,
        exp_size=1,
    )
    mpu_info = manager.get_mpu_info()
    dp_group = list(mpu_info.dp_groups[0])

    distribution = _node_distribution(dp_group, gpus_per_node=4)

    assert len(dp_group) == 32
    assert sorted(distribution.keys()) == list(range(8))
    assert set(distribution.values()) == {4}


def test_dp_group_tp4_world32_uses_one_rank_per_node() -> None:
    manager = ParallelGroupManager(
        local_size=4,
        world_size=32,
        pp_size=1,
        tp_size=4,
        exp_size=1,
    )
    mpu_info = manager.get_mpu_info()

    for dp_group in mpu_info.dp_groups:
        distribution = _node_distribution(dp_group, gpus_per_node=4)
        assert len(dp_group) == 8
        assert sorted(distribution.keys()) == list(range(8))
        assert set(distribution.values()) == {1}


def test_collective_sim_global_mode_keeps_exact_dp_participants(monkeypatch) -> None:
    backend, captured = _mocked_backend(monkeypatch)

    manager = ParallelGroupManager(
        local_size=4,
        world_size=32,
        pp_size=1,
        tp_size=4,
        exp_size=1,
    )
    mpu_info = manager.get_mpu_info()
    dp_group = tuple(int(rank) for rank in mpu_info.dp_groups[0])

    request = CommunicationPredictionRequest.from_raw(
        comm_group=dp_group,
        op_name="dp_allreduce",
        data_size_bytes=8 * 1024 * 1024,
        group_kind="dp",
        mpu_info=mpu_info,
    )

    duration = backend.predict(request)

    assert duration == 5.0
    assert captured["scenario"]["collective"]["participant_ranks"] == list(dp_group)
    distribution = _node_distribution(dp_group, gpus_per_node=4)
    assert set(distribution.values()) == {1}


def test_collective_sim_global_mode_keeps_exact_exp_participants(monkeypatch) -> None:
    backend, captured = _mocked_backend(monkeypatch)

    manager = ParallelGroupManager(
        local_size=4,
        world_size=32,
        pp_size=1,
        tp_size=4,
        exp_size=8,
    )
    mpu_info = manager.get_mpu_info()
    exp_group = tuple(int(rank) for rank in mpu_info.exp_groups[0])

    request = CommunicationPredictionRequest.from_raw(
        comm_group=exp_group,
        op_name="exp_all_to_all",
        data_size_bytes=8 * 1024 * 1024,
        group_kind="exp",
        mpu_info=mpu_info,
    )

    duration = backend.predict(request)

    assert duration == 5.0
    assert captured["scenario"]["collective"]["participant_ranks"] == list(exp_group)
    distribution = _node_distribution(exp_group, gpus_per_node=4)
    assert sorted(distribution.keys()) == list(range(8))
    assert set(distribution.values()) == {1}


def test_collective_sim_global_mode_keeps_exact_pp_pair_for_p2p(monkeypatch) -> None:
    backend, captured = _mocked_backend(monkeypatch)

    manager = ParallelGroupManager(
        local_size=8,
        world_size=16,
        pp_size=2,
        tp_size=1,
        exp_size=8,
    )
    mpu_info = manager.get_mpu_info()
    pp_pair = tuple(int(rank) for rank in mpu_info.pp_groups[0])

    request = CommunicationPredictionRequest.from_raw(
        comm_group=pp_pair,
        op_name="send_forward",
        data_size_bytes=4 * 1024 * 1024,
        group_kind="pp",
        mpu_info=mpu_info,
        metadata={
            "p2p_src_index": 0,
            "p2p_dst_index": 1,
            "p2p_direction": "0->1",
        },
    )

    duration = backend.predict(request)

    assert duration == 5.0
    assert captured["scenario"]["collective"]["kind"] == "p2p"
    assert captured["scenario"]["collective"]["participant_ranks"] == list(pp_pair)
