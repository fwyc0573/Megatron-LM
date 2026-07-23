"""Unit tests for collective-sim backend adapter (with mocked predictor)."""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cc_backend.collective_sim_backend import CollectiveSimCCBackend
from src.core.cc_backend.types import CommunicationPredictionRequest
from src.core.simulator_config import create_h800_sxm_ib_config


def _write_sendrecv_profile(profile_dir: Path, name: str, rows) -> Path:
    path = profile_dir / name
    lines = ["# nccl-tests compatible output"]
    for size_bytes, time_us in rows:
        lines.append(
            f"{int(size_bytes):>12d} {int(size_bytes // 4):>12d} float none none {float(time_us):>8.2f} 0.00 0.00 N/A"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_collective_sim_backend_builds_valid_scenario(monkeypatch) -> None:
    captured = {}

    def fake_predictor(scenario, repo_root=None):
        captured["scenario"] = scenario
        captured["repo_root"] = repo_root
        return {"predicted_time_ms": 12.5}

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
    config.communication.backend_options["collective-sim"] = {
        "repo_root": str(PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim"),
        "placement_order": ["TP", "CP", "DP", "EP"],
    }
    backend = CollectiveSimCCBackend(config)

    request = CommunicationPredictionRequest.from_raw(
        comm_group=[0, 1, 2, 3],
        op_name="exp_all_to_all",
        data_size_bytes=16 * 1024 * 1024,
        group_kind="exp",
    )

    duration_ms = backend.predict(request)

    assert duration_ms == 12.5
    assert captured["scenario"]["collective"]["kind"] == "alltoall"
    assert captured["scenario"]["collective"]["domain_dims"] == ["EP"]
    assert captured["scenario"]["parallelism"]["ep"] == 4


def test_collective_sim_backend_pp_p2p_requires_explicit_domain_mapping(monkeypatch) -> None:
    def fake_predictor(_scenario, repo_root=None):
        _ = repo_root
        return {"predicted_time_ms": 1.0}

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
    config.communication.backend_options["collective-sim"] = {
        "repo_root": str(PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim"),
        "placement_order": ["TP", "CP", "DP", "EP"],
        "pp_domain_dim": None,
    }
    backend = CollectiveSimCCBackend(config)
    request = CommunicationPredictionRequest.from_raw(
        comm_group=[0, 1],
        op_name="send_forward",
        data_size_bytes=8 * 1024 * 1024,
        group_kind="pp",
    )

    with pytest.raises(ValueError, match="pp_domain_dim"):
        backend.predict(request)


def test_collective_sim_backend_p2p_propagates_semantics(monkeypatch) -> None:
    captured = {}

    def fake_predictor(scenario, repo_root=None):
        captured["scenario"] = scenario
        captured["repo_root"] = repo_root
        return {"predicted_time_ms": 2.0}

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
    config.communication.backend_options["collective-sim"] = {
        "repo_root": str(PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim"),
        "placement_order": ["TP", "CP", "DP", "EP"],
        "pp_domain_dim": "DP",
    }
    backend = CollectiveSimCCBackend(config)
    request = CommunicationPredictionRequest.from_raw(
        comm_group=[12, 13],
        op_name="send_forward",
        data_size_bytes=4 * 1024 * 1024,
        group_kind="pp",
        metadata={"p2p_src_index": 0, "p2p_dst_index": 1, "p2p_direction": "0->1"},
    )

    duration_ms = backend.predict(request)

    assert duration_ms == pytest.approx(2.0)
    assert captured["scenario"]["collective"]["kind"] == "p2p"
    assert captured["scenario"]["collective"]["domain_dims"] == ["DP"]
    assert captured["scenario"]["collective"]["p2p_src_index"] == 0
    assert captured["scenario"]["collective"]["p2p_dst_index"] == 1
    assert captured["scenario"]["collective"]["p2p_direction"] == "0->1"


def test_collective_sim_backend_fail_fast_on_nonzero_zero_duration(monkeypatch) -> None:
    def fake_predictor(_scenario, repo_root=None):
        _ = repo_root
        return {"predicted_time_ms": 0.0}

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
        op_name="exp_all_to_all",
        data_size_bytes=2 * 1024 * 1024,
        group_kind="exp",
    )

    with pytest.raises(RuntimeError, match="non-positive predicted_time_ms"):
        backend.predict(request)


def test_collective_sim_backend_p2p_uses_measured_profile(monkeypatch, tmp_path) -> None:
    profile_dir = tmp_path / "sendrecv"
    profile_dir.mkdir(parents=True, exist_ok=True)
    _write_sendrecv_profile(
        profile_dir,
        "single_node_sendrecv_20250818_151722.txt",
        rows=[(1024, 100.0), (2048, 200.0)],
    )
    _write_sendrecv_profile(
        profile_dir,
        "multi_node_sendrecv_20250818_150945.txt",
        rows=[(1024, 300.0), (2048, 600.0)],
    )

    monkeypatch.setattr(
        CollectiveSimCCBackend,
        "_resolve_repo_root",
        lambda self: Path(PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim"),
    )
    predictor_calls = {"count": 0}

    def _unexpected_predictor(_scenario, repo_root=None):
        _ = repo_root
        predictor_calls["count"] += 1
        raise AssertionError("htsim should not run when p2p_profile_dir is configured")

    monkeypatch.setattr(
        CollectiveSimCCBackend,
        "_load_predictor",
        staticmethod(lambda _repo_root: _unexpected_predictor),
    )

    config = create_h800_sxm_ib_config()
    config.communication.backend_options["collective-sim"].update(
        {
            "p2p_profile_dir": str(profile_dir),
            "p2p_unidir_scale": 1.0,
        }
    )
    backend = CollectiveSimCCBackend(config)

    intra_request = CommunicationPredictionRequest.from_raw(
        comm_group=[0, 1],
        op_name="send_forward",
        data_size_bytes=2048,
        group_kind="pp",
        metadata={"p2p_src_index": 0, "p2p_dst_index": 1, "p2p_direction": "0->1"},
    )
    inter_request = CommunicationPredictionRequest.from_raw(
        comm_group=[0, 8],
        op_name="send_forward",
        data_size_bytes=2048,
        group_kind="pp",
        metadata={"p2p_src_index": 0, "p2p_dst_index": 1, "p2p_direction": "0->1"},
    )

    assert backend.predict(intra_request) == pytest.approx(0.2)
    assert backend.predict(inter_request) == pytest.approx(0.6)
    assert predictor_calls["count"] == 0


def test_collective_sim_backend_p2p_profile_dir_missing_fail_fast(monkeypatch) -> None:
    monkeypatch.setattr(
        CollectiveSimCCBackend,
        "_resolve_repo_root",
        lambda self: Path(PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim"),
    )
    monkeypatch.setattr(
        CollectiveSimCCBackend,
        "_load_predictor",
        staticmethod(lambda _repo_root: lambda _scenario, repo_root=None: {"predicted_time_ms": 1.0}),
    )

    config = create_h800_sxm_ib_config()
    config.communication.backend_options["collective-sim"].update(
        {
            "p2p_profile_dir": str(PROJECT_ROOT / "data" / "does_not_exist"),
        }
    )
    with pytest.raises(FileNotFoundError, match="p2p_profile_dir"):
        CollectiveSimCCBackend(config)
