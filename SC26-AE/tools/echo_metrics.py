#!/usr/bin/env python3
"""Validate and report the portable Echo Task2 predictor bundle.

The upstream Echo scripts print useful diagnostics, but they do not emit a
stable machine-readable contract and their historical ``prediction`` files
are tracked in the source repository.  This wrapper intentionally owns the
Task2 evidence contract: it reads only the files passed on the command line,
re-loads the predictor twice, and writes a deterministic JSON/Markdown report.

``--synthetic`` is an explicit test-only mode.  It accepts the tiny linear
model fixture used by the repository tests; it is never selected implicitly
and the resulting report is marked as local synthetic evidence rather than a
two-GPU qualification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "sc26-ae-echo-metrics-v1"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_FOLD_RE = re.compile(
    rf"MSE\s+for\s+each\s+fold\s*\(validation\s+set\)\s*:\s*\[([^\]]+)\]",
    re.IGNORECASE,
)
_AVERAGE_RE = re.compile(
    rf"Average\s+MSE\s*\(validation\s+set\)\s*:\s*({_NUMBER})",
    re.IGNORECASE,
)
_TEST_RE = re.compile(rf"Test\s+MSE\s*:\s*({_NUMBER})", re.IGNORECASE)


class MetricsContractError(ValueError):
    """Raised when a Task2 evidence input violates the frozen contract."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricsContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise MetricsContractError(f"{name} must be finite")
    return number


def _finite_nonnegative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise MetricsContractError(f"{name} must be non-negative")
    return number


def _require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise MetricsContractError(f"{name} does not exist: {path}")
    return path


def _parse_elapsed(elapsed_seconds: str | None, timing_path: Path | None) -> float:
    if elapsed_seconds is None and timing_path is None:
        raise MetricsContractError("one of --elapsed-seconds or --timing-path is required")
    if elapsed_seconds is not None:
        return _finite_nonnegative(elapsed_seconds, "task2_run_all_elapsed_seconds")
    assert timing_path is not None
    _require_file(timing_path, "timing file")
    try:
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricsContractError(f"invalid timing JSON: {timing_path}") from exc
    if not isinstance(payload, Mapping) or "elapsed_seconds" not in payload:
        raise MetricsContractError("timing JSON must contain elapsed_seconds")
    return _finite_nonnegative(payload["elapsed_seconds"], "task2_run_all_elapsed_seconds")


def _read_dataset(path: Path) -> tuple[int, list[str]]:
    _require_file(path, "dataset")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if "slowdown" not in fieldnames:
                raise MetricsContractError("dataset must contain a slowdown column")
            rows = list(reader)
    except OSError as exc:
        raise MetricsContractError(f"cannot read dataset: {path}") from exc
    if not rows:
        raise MetricsContractError("dataset row count must be positive")
    for index, row in enumerate(rows, start=1):
        try:
            _finite(row.get("slowdown"), f"dataset slowdown row {index}")
        except MetricsContractError:
            raise
    return len(rows), [name for name in fieldnames if name != "slowdown"]


def _parse_training_log(path: Path) -> tuple[list[float], float, float]:
    _require_file(path, "run log")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MetricsContractError(f"cannot read run log: {path}") from exc
    fold_match = _FOLD_RE.search(text)
    if fold_match is None:
        raise MetricsContractError("run log is missing the five validation fold MSE values")
    folds = [_finite_nonnegative(token, "validation fold MSE") for token in re.findall(_NUMBER, fold_match.group(1))]
    if len(folds) != 5:
        raise MetricsContractError(f"expected exactly five validation folds, found {len(folds)}")
    average_match = _AVERAGE_RE.search(text)
    if average_match is None:
        raise MetricsContractError("run log is missing Average MSE (validation set)")
    logged_average = _finite_nonnegative(average_match.group(1), "logged average validation MSE")
    average = sum(folds) / len(folds)
    if not math.isclose(average, logged_average, rel_tol=1e-12, abs_tol=1e-12):
        raise MetricsContractError(
            f"logged average validation MSE {logged_average} differs from fold mean {average}"
        )
    test_match = _TEST_RE.search(text)
    if test_match is None:
        raise MetricsContractError("run log is missing Test MSE")
    test_mse = _finite_nonnegative(test_match.group(1), "test MSE")
    return folds, average, test_mse


def _read_scaler(path: Path, expected_features: Sequence[str]) -> dict[str, Any]:
    _require_file(path, "scaler")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricsContractError(f"invalid scaler JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise MetricsContractError("scaler JSON must be an object")
    names = payload.get("feature_names")
    means = payload.get("mean")
    scales = payload.get("scale")
    if not isinstance(names, list) or not isinstance(means, list) or not isinstance(scales, list):
        raise MetricsContractError("scaler must contain feature_names, mean, and scale arrays")
    if not names or not (len(names) == len(means) == len(scales)):
        raise MetricsContractError("scaler feature_names/mean/scale arrays must have equal positive length")
    if list(names) != list(expected_features):
        raise MetricsContractError(
            f"scaler feature_names do not match dataset columns: {names!r} != {list(expected_features)!r}"
        )
    finite_means = [_finite(value, f"scaler mean[{index}]") for index, value in enumerate(means)]
    finite_scales = [_finite(value, f"scaler scale[{index}]") for index, value in enumerate(scales)]
    if any(scale == 0 for scale in finite_scales):
        raise MetricsContractError("every scaler scale must be non-zero")
    return {
        "feature_names": list(names),
        "mean": finite_means,
        "scale": finite_scales,
        "feature_count": len(names),
        "mean_count": len(means),
        "scale_count": len(scales),
        "nonzero_scale_count": sum(scale != 0 for scale in finite_scales),
    }


def _sample_values(feature_names: Sequence[str]) -> dict[str, float]:
    canonical = {
        "ground_truth": 1.0,
        "Compute throughput": 50.0,
        "Memory throughput": 60.0,
        "DRAM throughput": 70.0,
        "Achieved occupancy": 80.0,
        "Maximum occupancy": 90.0,
        "L1 hit rate": 95.0,
        "L2 hit rate": 85.0,
    }
    # Echo's canonical feature names are the entries above.  A deterministic
    # index value keeps the test-only fixture useful for a deliberately tiny
    # feature schema while still rejecting missing scaler columns elsewhere.
    return {name: float(canonical.get(name, index + 1)) for index, name in enumerate(feature_names)}


def _normalised_values(values: Mapping[str, float], scaler: Mapping[str, Any]) -> list[float]:
    return [
        (float(values[name]) - float(mean)) / float(scale)
        for name, mean, scale in zip(scaler["feature_names"], scaler["mean"], scaler["scale"])
    ]


def _synthetic_model_prediction(model_payload: Mapping[str, Any], row: Sequence[float]) -> float:
    if model_payload.get("format") != "sc26-ae-synthetic-xgb-v1":
        raise MetricsContractError(
            "synthetic mode requires a model with format sc26-ae-synthetic-xgb-v1"
        )
    weights = model_payload.get("weights")
    if not isinstance(weights, list) or len(weights) != len(row):
        raise MetricsContractError("synthetic model weights must match scaler feature count")
    bias = _finite(model_payload.get("bias", 0.0), "synthetic model bias")
    return _finite(bias + sum(_finite(weight, "synthetic model weight") * value for weight, value in zip(weights, row)), "synthetic prediction")


def _load_real_predictor(model_path: Path, scaler: Mapping[str, Any]):
    try:
        import numpy as np  # type: ignore
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise MetricsContractError("real Echo metric mode requires numpy and xgboost") from exc
    model_a = xgb.Booster()
    model_b = xgb.Booster()
    try:
        model_a.load_model(str(model_path))
        model_b.load_model(str(model_path))
    except Exception as exc:  # xgboost exposes version-specific exception classes
        raise MetricsContractError(f"cannot load XGBoost model: {model_path}") from exc
    values = _sample_values(scaler["feature_names"])
    row = np.asarray([_normalised_values(values, scaler)], dtype=float)
    try:
        matrix = xgb.DMatrix(row, feature_names=list(scaler["feature_names"]))
        prediction_a = float(model_a.predict(matrix)[0])
        prediction_b = float(model_b.predict(matrix)[0])
    except Exception as exc:
        raise MetricsContractError(
            "XGBoost prediction failed for deterministic sample: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return values, prediction_a, prediction_b


def _load_synthetic_predictor(model_path: Path, scaler: Mapping[str, Any]):
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricsContractError(f"invalid synthetic model JSON: {model_path}") from exc
    values = _sample_values(scaler["feature_names"])
    row = _normalised_values(values, scaler)
    prediction_a = _synthetic_model_prediction(payload, row)
    # Deliberately parse a second JSON object so the reload check exercises two
    # independent model instances, rather than reusing one in-memory value.
    payload_b = json.loads(model_path.read_text(encoding="utf-8"))
    prediction_b = _synthetic_model_prediction(payload_b, row)
    return values, prediction_a, prediction_b


def _prediction_sample(values: Mapping[str, float], prediction: float, overlap_ratio: float) -> dict[str, float]:
    ground_truth = _finite(values.get("ground_truth"), "prediction ground_truth")
    overlap = _finite(overlap_ratio, "prediction overlap ratio")
    if overlap < 0 or overlap > 1:
        raise MetricsContractError("prediction overlap ratio must be between zero and one")
    clipped = max(0.0, prediction)
    result = {
        "original_execution_time": ground_truth,
        "predicted_execution_time": (1 - overlap) * ground_truth + overlap * ground_truth * (1 + prediction),
        "predicted_execution_time_clipped": (1 - overlap) * ground_truth + overlap * ground_truth * (1 + clipped),
        "predicted_slowdown_factor": prediction,
        "predicted_slowdown_factor_clipped": clipped,
    }
    for name, value in result.items():
        result[name] = _finite(value, f"prediction sample {name}")
    if result["predicted_slowdown_factor_clipped"] < 0:
        raise MetricsContractError("clipped slowdown must be non-negative")
    return result


def build_metrics(
    *,
    dataset_path: Path,
    model_path: Path,
    scaler_path: Path,
    log_path: Path,
    elapsed_seconds: str | None = None,
    timing_path: Path | None = None,
    predictor_run_id: str | None = None,
    synthetic: bool = False,
    overlap_ratio: float = 0.5,
) -> dict[str, Any]:
    elapsed = _parse_elapsed(elapsed_seconds, timing_path)
    if elapsed <= 0:
        raise MetricsContractError("task2_run_all_elapsed_seconds must be strictly positive")
    row_count, feature_names = _read_dataset(dataset_path)
    folds, average, test_mse = _parse_training_log(log_path)
    scaler = _read_scaler(scaler_path, feature_names)
    _require_file(model_path, "model")
    if synthetic:
        values, prediction_a, prediction_b = _load_synthetic_predictor(model_path, scaler)
    else:
        values, prediction_a, prediction_b = _load_real_predictor(model_path, scaler)
    prediction = _prediction_sample(values, prediction_a, overlap_ratio)
    prediction_b_sample = _prediction_sample(values, prediction_b, overlap_ratio)
    reload_delta = max(
        abs(prediction[name] - prediction_b_sample[name]) for name in prediction
    )
    reload_delta = _finite_nonnegative(reload_delta, "model reload max absolute prediction delta")
    if reload_delta > 1e-12:
        raise MetricsContractError(
            f"independently loaded model predictions differ by {reload_delta}, expected <= 1e-12"
        )
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task2_run_all_elapsed_seconds": elapsed,
        "dataset_row_count": row_count,
        "validation_mse_by_fold": folds,
        "average_validation_mse": average,
        "test_mse": test_mse,
        "model_reload_max_abs_prediction_delta": reload_delta,
        "scaler_feature_count": scaler["feature_count"],
        "scaler_mean_count": scaler["mean_count"],
        "scaler_scale_count": scaler["scale_count"],
        "scaler_nonzero_scale_count": scaler["nonzero_scale_count"],
        "prediction_sample": prediction,
    }
    if predictor_run_id is not None:
        metrics["predictor_run_id"] = predictor_run_id
    if synthetic:
        metrics["execution_evidence"] = "local_synthetic_not_two_gpu_qualification"
    else:
        metrics["execution_evidence"] = "echo_runtime_measurement_pending_external_qualification"
    return metrics


def write_metrics(metrics: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    # ``sort_keys`` makes checksums reproducible while preserving the schema's
    # nested shape for readers.
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sample = metrics["prediction_sample"]
    folds = metrics["validation_mse_by_fold"]
    lines = [
        "# Echo Task2 Metrics",
        "",
        f"- Schema: `{metrics['schema_version']}`",
        f"- Predictor run: `{metrics.get('predictor_run_id', 'not-bound')}`",
        f"- Evidence class: `{metrics.get('execution_evidence', 'unknown')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Run elapsed seconds | {metrics['task2_run_all_elapsed_seconds']:.12g} |",
        f"| Dataset rows | {metrics['dataset_row_count']} |",
        f"| Validation MSE fold 1 | {folds[0]:.12g} |",
        f"| Validation MSE fold 2 | {folds[1]:.12g} |",
        f"| Validation MSE fold 3 | {folds[2]:.12g} |",
        f"| Validation MSE fold 4 | {folds[3]:.12g} |",
        f"| Validation MSE fold 5 | {folds[4]:.12g} |",
        f"| Average validation MSE | {metrics['average_validation_mse']:.12g} |",
        f"| Test MSE | {metrics['test_mse']:.12g} |",
        f"| Model reload max abs delta | {metrics['model_reload_max_abs_prediction_delta']:.12g} |",
        f"| Scaler feature count | {metrics['scaler_feature_count']} |",
        f"| Scaler mean count | {metrics['scaler_mean_count']} |",
        f"| Scaler scale count | {metrics['scaler_scale_count']} |",
        f"| Scaler nonzero scale count | {metrics['scaler_nonzero_scale_count']} |",
        "",
        "## Deterministic prediction sample",
        "",
        "| Field | Value |",
        "| --- | ---: |",
    ]
    for name in (
        "original_execution_time",
        "predicted_execution_time",
        "predicted_execution_time_clipped",
        "predicted_slowdown_factor",
        "predicted_slowdown_factor_clipped",
    ):
        lines.append(f"| {name} | {sample[name]:.12g} |")
    lines.extend(
        [
            "",
            "Synthetic reports are local contract evidence only and do not qualify the required two-GPU Echo run.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--dataset-path", dest="dataset_path", type=Path, required=True)
    parser.add_argument("--model", "--model-path", dest="model_path", type=Path, required=True)
    parser.add_argument("--scaler", "--scaler-path", dest="scaler_path", type=Path, required=True)
    parser.add_argument("--log", "--log-path", dest="log_path", type=Path, required=True)
    parser.add_argument("--elapsed-seconds")
    parser.add_argument("--timing-path", type=Path)
    parser.add_argument("--predictor-run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the explicit local synthetic model fixture; never a GPU qualification path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output_dir is not None:
        json_path = args.output_dir / "metrics.json"
        markdown_path = args.output_dir / "metrics.md"
    else:
        if args.output_json is None or args.output_md is None:
            raise MetricsContractError("--output-dir or both --output-json and --output-md are required")
        json_path = args.output_json
        markdown_path = args.output_md
    metrics = build_metrics(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        log_path=args.log_path,
        elapsed_seconds=args.elapsed_seconds,
        timing_path=args.timing_path,
        predictor_run_id=args.predictor_run_id,
        synthetic=args.synthetic,
        overlap_ratio=args.overlap_ratio,
    )
    write_metrics(metrics, json_path, markdown_path)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MetricsContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
