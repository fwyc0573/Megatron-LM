"""Regression tests for the CPU-only simulator hardware selector."""

from __future__ import annotations

import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simulator_config import load_config_from_env


def test_cpu_hardware_selector_uses_deterministic_analytical_profile(monkeypatch) -> None:
    """CPU-only execution must still select a deterministic target profile."""

    monkeypatch.setenv("SIMULATOR_HARDWARE_TYPE", "cpu")

    config = load_config_from_env()

    assert config.hardware.gpu_type.value == "CPU"
    assert config.hardware.gpus_per_node == 8
    assert config.communication.backend == "collective-sim"
    assert config.communication.backend_options["collective-sim"]["gpu_profile"] == "h800_sxm"

