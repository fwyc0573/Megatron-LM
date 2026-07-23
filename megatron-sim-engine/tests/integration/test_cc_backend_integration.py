"""Integration checks for SimulatorEngine cc backend wiring."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cc_backend.collective_sim_backend import CollectiveSimCCBackend
from src.core.cc_backend.types import CommunicationPredictionRequest
from src.core.simu_engine import MODE_SIMULATE, SimulatorEngine
from src.core.simulator_config import create_h800_sxm_ib_config
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager


def test_simulator_engine_initializes_analytical_cc_backend() -> None:
    engine = SimulatorEngine(
        trace_filepath=None,
        framwork="megatron-lm",
        strategy="1F1B-none_interleaved",
        running_mode=MODE_SIMULATE,
        torchgraph_filepath=None,
        stages_scheduling_filepath=None,
        cc_backend_name="analytical",
    )

    assert engine.cc_backend is not None
    assert engine.cc_backend.backend_name == "analytical"


def test_collective_sim_backend_uses_global_placement_from_mpu(monkeypatch) -> None:
    captured = {}

    def fake_predictor(scenario, repo_root=None):
        captured["scenario"] = scenario
        captured["repo_root"] = repo_root
        return {"predicted_time_ms": 6.0}

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

    backend = CollectiveSimCCBackend(config)

    manager = ParallelGroupManager(
        local_size=8,
        world_size=32,
        pp_size=2,
        tp_size=2,
        dp_size=8,
        exp_size=1,
        cp_size=1,
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

    duration_ms = backend.predict(request)

    assert duration_ms == 6.0
    assert captured["scenario"]["cluster"] == {"servers": 4, "gpus_per_server": 8}
    assert captured["scenario"]["parallelism"] == {"tp": 2, "cp": 1, "dp": 16, "ep": 1}
    assert captured["scenario"]["collective"]["participant_ranks"] == list(dp_group)
    assert captured["scenario"]["collective"]["domain_dims"] == ["DP"]


def test_collective_sim_backend_reducescatter_flow(monkeypatch) -> None:
    captured = {}

    def fake_predictor(scenario, repo_root=None):
        captured["scenario"] = scenario
        captured["repo_root"] = repo_root
        return {"predicted_time_ms": 4.5}

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
    backend = CollectiveSimCCBackend(config)

    request = CommunicationPredictionRequest.from_raw(
        comm_group=[0, 1, 2, 3],
        op_name="tp_reducescatter",
        data_size_bytes=2 * 1024 * 1024,
        group_kind="tp",
    )

    duration_ms = backend.predict(request)

    assert duration_ms == 4.5
    assert captured["scenario"]["collective"]["kind"] == "reducescatter"
    assert captured["scenario"]["collective"]["domain_dims"] == ["TP"]
