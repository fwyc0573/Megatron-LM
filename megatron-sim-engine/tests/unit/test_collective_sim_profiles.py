"""Unit tests for collective-sim built-in scenario profiles."""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
COLLECTIVE_SIM_PYTHON = PROJECT_ROOT / "src" / "core" / "cc_backend" / "collective-sim" / "python"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(COLLECTIVE_SIM_PYTHON) not in sys.path:
    sys.path.insert(0, str(COLLECTIVE_SIM_PYTHON))

from collective_sim_core import get_scenario_profile_builder, list_scenario_profiles
from collective_sim_core.schema import CollectiveConfig, ParallelismConfig


def test_collective_sim_profile_registry_covers_production_gpu_families() -> None:
    profiles = set(list_scenario_profiles())
    expected = {
        "h100_rail",
        "h100_fattree",
        "h800_rail",
        "h800_fattree",
        "a100_rail",
        "a100_fattree",
        "a800_rail",
        "a800_fattree",
    }
    assert expected.issubset(profiles)


def test_collective_sim_a100_and_a800_profiles_build_valid_scenarios() -> None:
    collective = CollectiveConfig(
        kind="allreduce",
        tensor_bytes=64 * 1024 * 1024,
        domain_dims=("DP",),
        placement_order=("TP", "CP", "DP", "EP"),
    )
    parallelism = ParallelismConfig(tp=8, cp=1, dp=16, ep=1)

    for profile_name in ("a100_rail", "a800_rail"):
        scenario = get_scenario_profile_builder(profile_name)(
            servers=16,
            collective=collective,
            parallelism=parallelism,
        )
        scenario.validate()
        assert scenario.intra_server.model == "nvlink_analytic"
        assert scenario.network.linkspeed_mbps > 0
