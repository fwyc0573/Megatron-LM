"""Synthetic unit coverage for the Task2 metrics contract.

These tests exercise only the local contract implementation.  They do not
qualify the required two-GPU Echo run or claim a real dataset.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[2] / "SC26-AE/tools/echo_metrics.py"
SPEC = importlib.util.spec_from_file_location("sc26_ae_echo_metrics", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(root: Path) -> dict[str, Path]:
    dataset = root / "dataset.csv"
    dataset.write_text(
        "ground_truth,Compute throughput,slowdown\n1,2,0.1\n2,3,0.2\n",
        encoding="utf-8",
    )
    model = root / "xgb_model.json"
    model.write_text(
        json.dumps(
            {
                "format": "sc26-ae-synthetic-xgb-v1",
                "weights": [0.1, 0.2],
                "bias": 0.3,
            }
        ),
        encoding="utf-8",
    )
    scaler = root / "standard_scaler.json"
    scaler.write_text(
        json.dumps(
            {
                "feature_names": ["ground_truth", "Compute throughput"],
                "mean": [0.0, 0.0],
                "scale": [1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    log = root / "run_all.log"
    log.write_text(
        "MSE for each fold (validation set): [1.0, 2.0, 3.0, 4.0, 5.0]\n"
        "Average MSE (validation set): 3.0\n"
        "Test MSE: 0.5\n",
        encoding="utf-8",
    )
    return {"dataset": dataset, "model": model, "scaler": scaler, "log": log}


def test_synthetic_metrics_are_numeric_and_explicit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metrics = MODULE.build_metrics(
        dataset_path=paths["dataset"],
        model_path=paths["model"],
        scaler_path=paths["scaler"],
        log_path=paths["log"],
        elapsed_seconds="1.25",
        predictor_run_id="synthetic-run",
        synthetic=True,
    )
    assert metrics["schema_version"] == "sc26-ae-echo-metrics-v1"
    assert metrics["dataset_row_count"] == 2
    assert metrics["validation_mse_by_fold"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert metrics["average_validation_mse"] == 3.0
    assert metrics["model_reload_max_abs_prediction_delta"] == 0.0
    assert metrics["scaler_nonzero_scale_count"] == 2
    assert metrics["execution_evidence"] == "local_synthetic_not_two_gpu_qualification"
    assert all(value == value for value in metrics["prediction_sample"].values())


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda paths: paths["log"].write_text("Test MSE: 1.0\n", encoding="utf-8"), "five validation"),
        (
            lambda paths: paths["scaler"].write_text(
                json.dumps(
                    {
                        "feature_names": ["ground_truth", "Compute throughput"],
                        "mean": [0.0, 0.0],
                        "scale": [1.0, 0.0],
                    }
                ),
                encoding="utf-8",
            ),
            "non-zero",
        ),
    ],
)
def test_invalid_fold_or_scaler_fails_fast(tmp_path: Path, mutator, expected: str) -> None:
    paths = _fixture(tmp_path)
    mutator(paths)
    with pytest.raises(MODULE.MetricsContractError, match=expected):
        MODULE.build_metrics(
            dataset_path=paths["dataset"],
            model_path=paths["model"],
            scaler_path=paths["scaler"],
            log_path=paths["log"],
            elapsed_seconds="1.0",
            synthetic=True,
        )


def test_nonpositive_elapsed_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(MODULE.MetricsContractError, match="strictly positive"):
        MODULE.build_metrics(
            dataset_path=paths["dataset"],
            model_path=paths["model"],
            scaler_path=paths["scaler"],
            log_path=paths["log"],
            elapsed_seconds="0",
            synthetic=True,
        )


def test_average_mismatch_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["log"].write_text(
        "MSE for each fold (validation set): [1.0, 2.0, 3.0, 4.0, 5.0]\n"
        "Average MSE (validation set): 99.0\n"
        "Test MSE: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(MODULE.MetricsContractError, match="differs"):
        MODULE.build_metrics(
            dataset_path=paths["dataset"],
            model_path=paths["model"],
            scaler_path=paths["scaler"],
            log_path=paths["log"],
            elapsed_seconds="1.0",
            synthetic=True,
        )
