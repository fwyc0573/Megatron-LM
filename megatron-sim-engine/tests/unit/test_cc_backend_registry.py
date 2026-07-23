"""Unit tests for CC backend registry and analytical backend."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cc_backend import CommunicationPredictionRequest, create_cc_backend, list_cc_backends
from src.core.cc_backend.op_mapping import infer_domain_dims
from src.core.simulator_config import create_a100_sxm_ib_config, create_h800_sxm_ib_config


def test_builtin_backends_are_registered() -> None:
    backends = set(list_cc_backends())
    assert "analytical" in backends
    assert "collective-sim" in backends
    assert "profiling" in backends


def test_create_unknown_backend_fails_fast() -> None:
    config = create_h800_sxm_ib_config()
    with pytest.raises(ValueError, match="Unknown CC backend"):
        create_cc_backend("unknown-backend", config)


def test_analytical_backend_predict() -> None:
    config = create_h800_sxm_ib_config()
    backend = create_cc_backend("analytical", config)

    request = CommunicationPredictionRequest.from_raw(
        comm_group=list(range(8)),
        op_name="dp_allreduce",
        data_size_bytes=1024 * 1024,
        group_kind="dp",
    )
    duration_ms = backend.predict(request)

    assert duration_ms > 0.0


def test_default_config_uses_collective_sim_backend() -> None:
    config = create_h800_sxm_ib_config()
    assert config.communication.backend == "collective-sim"

    backend = create_cc_backend(config.communication.backend, config)
    assert backend.backend_name == "collective-sim"


def test_default_collective_sim_options_fail_fast_for_pipeline_p2p() -> None:
    config = create_h800_sxm_ib_config()
    options = config.communication.backend_options["collective-sim"]
    assert options["pp_domain_dim"] == "DP"
    assert options["enforce_nonzero_duration"] is True
    assert options["intra_server"]["model"] == "nvlink_analytic"


def test_a100_default_collective_sim_network_profile() -> None:
    config = create_a100_sxm_ib_config()
    options = config.communication.backend_options["collective-sim"]
    assert options["gpu_profile"] == "a100_sxm"
    assert options["network"]["linkspeed_mbps"] == 200000


def test_request_from_raw_parses_string_tensor_shape() -> None:
    request = CommunicationPredictionRequest.from_raw(
        comm_group="[0, 1]",
        op_name="dp_allreduce",
        data_size_bytes=1024,
        tensor_shape="[8, 16]",
        tensor_dtype="torch.float16",
    )
    assert request.comm_group == (0, 1)
    assert request.tensor_shape == (8, 16)


def test_infer_domain_dims_pipeline_p2p_requires_explicit_config() -> None:
    with pytest.raises(ValueError, match="pp_domain_dim"):
        infer_domain_dims("pp", "send_forward")
