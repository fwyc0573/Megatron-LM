#!/usr/bin/env python3
"""Simulator configuration module."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


import subprocess

class HardwareType:
    # CPU denotes the local execution host.  Communication timings still use
    # the deterministic H800 analytical target profile below.
    CPU = "CPU"
    A100_SXM = "A100-SXM"
    A800_SXM = "A800-SXM"
    H100_SXM = "H100-SXM"
    H800_SXM = "H800-SXM"


@dataclass
class GPUTypeValue:
    value: str


@dataclass
class HardwareConfig:
    gpu_type: GPUTypeValue
    gpus_per_node: int


@dataclass
class CommunicationConfig:
    # Legacy fields (kept for backward compatibility)
    use_cc_estimator: bool = True
    use_ml_predictor: bool = True
    cache_predictions: bool = True
    cache_size_limit: int = 10000
    sendrecv_dataset_path: Optional[str] = None

    # New pluggable backend fields
    backend: str = "analytical"
    backend_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlowdownConfig:
    enabled: bool = False
    assets_dir: Optional[str] = None
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    max_iters: int = 50
    tol_ms: float = 1e-6


@dataclass
class OverlapConfig:
    mode: str = "auto"


@dataclass
class SimulatorConfig:
    hardware: HardwareConfig
    communication: CommunicationConfig
    slowdown: SlowdownConfig = field(default_factory=SlowdownConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    exp_size: int = 1


def _default_collective_sim_repo_root() -> str:
    return str(Path(__file__).resolve().parent / "cc_backend" / "collective-sim")


def _default_h800_sendrecv_profile_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "data" / "h800_dgx_roce_sendrecv")


def _collective_sim_defaults_for_gpu(gpu_type: str) -> Dict[str, Any]:
    normalized = gpu_type.strip().upper()
    if normalized == HardwareType.H100_SXM:
        return {
            "gpu_profile": "h100_sxm",
            "network": {"linkspeed_mbps": 400000, "mtu": 9216, "q": 64, "cwnd": 64},
            "intra_server": {
                "model": "nvlink_analytic",
                "nvlink_one_way_bw_GBps": 450.0,
                "nvlink_latency_us": 0.5,
                "nvlink_efficiency": 0.8,
            },
        }
    if normalized == HardwareType.H800_SXM:
        return {
            "gpu_profile": "h800_sxm",
            "network": {"linkspeed_mbps": 400000, "mtu": 9216, "q": 64, "cwnd": 64},
            "intra_server": {
                "model": "nvlink_analytic",
                "nvlink_one_way_bw_GBps": 200.0,
                "nvlink_latency_us": 0.6,
                "nvlink_efficiency": 0.8,
            },
            "p2p_profile_dir": _default_h800_sendrecv_profile_dir(),
        }
    if normalized == HardwareType.A100_SXM:
        return {
            "gpu_profile": "a100_sxm",
            "network": {"linkspeed_mbps": 200000, "mtu": 9216, "q": 48, "cwnd": 32},
            "intra_server": {
                "model": "nvlink_analytic",
                "nvlink_one_way_bw_GBps": 300.0,
                "nvlink_latency_us": 0.8,
                "nvlink_efficiency": 0.78,
            },
        }
    if normalized == HardwareType.A800_SXM:
        return {
            "gpu_profile": "a800_sxm",
            "network": {"linkspeed_mbps": 200000, "mtu": 9216, "q": 48, "cwnd": 32},
            "intra_server": {
                "model": "nvlink_analytic",
                "nvlink_one_way_bw_GBps": 180.0,
                "nvlink_latency_us": 0.9,
                "nvlink_efficiency": 0.76,
            },
        }
    raise ValueError(f"Unsupported gpu_type for collective-sim defaults: {gpu_type}")


def _collective_backend_options(gpu_type: str) -> Dict[str, Any]:
    defaults = _collective_sim_defaults_for_gpu(gpu_type)
    return {
        "repo_root": _default_collective_sim_repo_root(),
        "placement_order": ["TP", "CP", "DP", "EP"],
        # PP is not a native collective-sim dimension. This explicit mapping is required
        # for pipeline send/recv p2p semantics.
        "pp_domain_dim": "DP",
        "exclude_intra_server": True,
        "enforce_nonzero_duration": True,
        "placement_mode": "auto",
        "strict_mpu_alignment": True,
        **defaults,
    }


def create_h800_sxm_ib_config() -> SimulatorConfig:
    """Create H800-SXM + InfiniBand configuration."""

    return SimulatorConfig(
        hardware=HardwareConfig(
            gpu_type=GPUTypeValue(HardwareType.H800_SXM),
            gpus_per_node=8,
        ),
        communication=CommunicationConfig(
            use_cc_estimator=True,
            use_ml_predictor=True,
            cache_predictions=True,
            cache_size_limit=10000,
            sendrecv_dataset_path="moe_mg/sendrecv/pytorch_sendrecv_results",
            backend="collective-sim",
            backend_options={
                "collective-sim": _collective_backend_options(HardwareType.H800_SXM)
            },
        ),
    )


def create_cpu_config() -> SimulatorConfig:
    """Create a CPU-host configuration with the deterministic H800 target profile.

    The simulator executes its analytical model on the host CPU; ``CPU`` is
    therefore an execution-host selector, not a replacement communication
    topology.  Keeping the H800 profile explicit preserves stable fake-level
    communication estimates without requiring a local GPU.
    """

    config = create_h800_sxm_ib_config()
    config.hardware = HardwareConfig(
        gpu_type=GPUTypeValue(HardwareType.CPU),
        gpus_per_node=8,
    )
    return config


def create_a100_sxm_ib_config() -> SimulatorConfig:
    """Create A100-SXM + InfiniBand configuration."""

    return SimulatorConfig(
        hardware=HardwareConfig(
            gpu_type=GPUTypeValue(HardwareType.A100_SXM),
            gpus_per_node=8,
        ),
        communication=CommunicationConfig(
            use_cc_estimator=True,
            use_ml_predictor=False,
            cache_predictions=True,
            cache_size_limit=10000,
            sendrecv_dataset_path=None,
            backend="collective-sim",
            backend_options={
                "collective-sim": _collective_backend_options(HardwareType.A100_SXM)
            },
        ),
    )


def create_h100_sxm_ib_config() -> SimulatorConfig:
    """Create H100-SXM + InfiniBand/RoCE configuration."""

    return SimulatorConfig(
        hardware=HardwareConfig(
            gpu_type=GPUTypeValue(HardwareType.H100_SXM),
            gpus_per_node=8,
        ),
        communication=CommunicationConfig(
            use_cc_estimator=True,
            use_ml_predictor=True,
            cache_predictions=True,
            cache_size_limit=10000,
            sendrecv_dataset_path="moe_mg/sendrecv/pytorch_sendrecv_results",
            backend="collective-sim",
            backend_options={"collective-sim": _collective_backend_options(HardwareType.H100_SXM)},
        ),
    )


def create_a800_sxm_ib_config() -> SimulatorConfig:
    """Create A800-SXM + InfiniBand/RoCE configuration."""

    return SimulatorConfig(
        hardware=HardwareConfig(
            gpu_type=GPUTypeValue(HardwareType.A800_SXM),
            gpus_per_node=8,
        ),
        communication=CommunicationConfig(
            use_cc_estimator=True,
            use_ml_predictor=False,
            cache_predictions=True,
            cache_size_limit=10000,
            sendrecv_dataset_path=None,
            backend="collective-sim",
            backend_options={"collective-sim": _collective_backend_options(HardwareType.A800_SXM)},
        ),
    )


def _detect_local_hardware_type() -> str:
    try:
        raw_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Failed to detect local GPU type via nvidia-smi. "
            "Please set SIMULATOR_HARDWARE_TYPE explicitly."
        ) from exc

    lines = [line.strip() for line in raw_output.decode("utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            "nvidia-smi returned no GPU names. Please set SIMULATOR_HARDWARE_TYPE explicitly."
        )

    first_name = lines[0].upper()
    if "A800" in first_name:
        return "A800_SXM"
    if "A100" in first_name:
        return "A100_SXM"
    if "H800" in first_name:
        return "H800_SXM"
    if "H100" in first_name:
        return "H100_SXM"

    raise ValueError(
        f"Unsupported auto-detected GPU name {lines[0]!r}. "
        "Please set SIMULATOR_HARDWARE_TYPE explicitly."
    )


def get_default_config() -> SimulatorConfig:
    return create_h800_sxm_ib_config()


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_config_from_env() -> SimulatorConfig:
    hardware_type = os.environ.get("SIMULATOR_HARDWARE_TYPE")
    if hardware_type is None:
        hardware_type = _detect_local_hardware_type()

    hardware_type = hardware_type.strip().upper()

    if hardware_type == HardwareType.CPU:
        config = create_cpu_config()
    elif hardware_type == "A100_SXM":
        config = create_a100_sxm_ib_config()
    elif hardware_type == "A800_SXM":
        config = create_a800_sxm_ib_config()
    elif hardware_type == "H100_SXM":
        config = create_h100_sxm_ib_config()
    elif hardware_type == "H800_SXM":
        config = create_h800_sxm_ib_config()
    else:
        raise ValueError(
            "Unsupported SIMULATOR_HARDWARE_TYPE. "
            "Expected one of: cpu | A100_SXM | A800_SXM | H100_SXM | H800_SXM, "
            f"got {hardware_type!r}."
        )

    cc_backend = os.environ.get("SIMULATOR_CC_BACKEND")
    if cc_backend:
        config.communication.backend = cc_backend

    collective_repo_root = os.environ.get("SIMULATOR_COLLECTIVE_SIM_REPO_ROOT")
    if collective_repo_root:
        collective_options = config.communication.backend_options.setdefault("collective-sim", {})
        collective_options["repo_root"] = collective_repo_root

    raw_backend_options = os.environ.get("SIMULATOR_CC_BACKEND_OPTIONS")
    if raw_backend_options:
        try:
            options = json.loads(raw_backend_options)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SIMULATOR_CC_BACKEND_OPTIONS must be a valid JSON object"
            ) from exc
        if not isinstance(options, dict):
            raise ValueError("SIMULATOR_CC_BACKEND_OPTIONS must decode to a JSON object")
        _deep_update(config.communication.backend_options, options)

    return config
