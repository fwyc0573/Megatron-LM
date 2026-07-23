"""Unit tests for profiling-table CC backend."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cc_backend import CommunicationPredictionRequest, create_cc_backend
from src.core.simulator_config import create_h800_sxm_ib_config


def test_profiling_backend_predict_exact_match() -> None:
    config = create_h800_sxm_ib_config()
    config.communication.backend_options["profiling"] = {
        "table": {
            "allreduce|DP|8|1048576": 3.25,
        }
    }

    backend = create_cc_backend("profiling", config)

    request = CommunicationPredictionRequest.from_raw(
        comm_group=list(range(8)),
        op_name="dp_allreduce",
        data_size_bytes=1048576,
        group_kind="dp",
    )

    assert backend.predict(request) == pytest.approx(3.25)


def test_profiling_backend_missing_record_fails_fast() -> None:
    config = create_h800_sxm_ib_config()
    config.communication.backend_options["profiling"] = {
        "table": {
            "allreduce|DP|4|1048576": 1.0,
        }
    }

    backend = create_cc_backend("profiling", config)
    request = CommunicationPredictionRequest.from_raw(
        comm_group=list(range(8)),
        op_name="dp_allreduce",
        data_size_bytes=1048576,
        group_kind="dp",
    )

    with pytest.raises(KeyError, match="No profiling record"):
        backend.predict(request)
