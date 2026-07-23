"""Slowdown prediction helpers for DDP backward overlap replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REQUIRED_SLOWDOWN_FEATURES: Tuple[str, ...] = (
    "ground_truth",
    "Compute throughput",
    "Memory throughput",
    "DRAM throughput",
    "Achieved occupancy",
    "Maximum occupancy",
    "L1 hit rate",
    "L2 hit rate",
)

REQUIRED_KERNEL_FEATURES: Tuple[str, ...] = REQUIRED_SLOWDOWN_FEATURES[1:]
_REQUIRED_MANIFEST_KEYS: Tuple[str, ...] = (
    "scope",
    "model_path",
    "scaler_path",
    "label_prefix",
    "clip_negative_slowdown",
    "generator_version",
    "source_trace_dir",
    "source_nsys_sqlite",
    "source_ncu_metrics",
)


def _require_non_negative_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float-compatible value") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got {parsed}")
    return parsed


def _require_overlap_ratio(value: object) -> float:
    overlap_ratio = _require_non_negative_float(value, "input_overlap_ratio")
    if overlap_ratio > 1.0:
        raise ValueError(f"input_overlap_ratio must be <= 1, got {overlap_ratio}")
    return overlap_ratio


def _validate_feature_mapping(features: Mapping[str, object], required: Sequence[str]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"Missing slowdown features: {missing}")
    for feature_name in required:
        normalized[feature_name] = _require_non_negative_float(features[feature_name], feature_name)
    return normalized


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    sorted_intervals = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not sorted_intervals:
        return []
    merged: List[Tuple[float, float]] = []
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def compute_overlap_time_ms(
    kernel_start_ms: float,
    kernel_duration_ms: float,
    active_intervals_ms: Sequence[Tuple[float, float]],
) -> float:
    if kernel_duration_ms <= 0:
        return 0.0
    kernel_end_ms = float(kernel_start_ms) + float(kernel_duration_ms)
    overlap_intervals: List[Tuple[float, float]] = []
    for interval_start, interval_end in active_intervals_ms:
        start = max(float(kernel_start_ms), float(interval_start))
        end = min(kernel_end_ms, float(interval_end))
        if end > start:
            overlap_intervals.append((start, end))
    return round(sum(end - start for start, end in _merge_intervals(overlap_intervals)), 6)


def solve_kernel_slowdown_duration(
    *,
    ground_truth_ms: float,
    slowdown_factor_clipped: float,
    kernel_start_ms: float,
    active_intervals_ms: Sequence[Tuple[float, float]],
    max_iters: int,
    tol_ms: float,
) -> float:
    ground_truth_ms = _require_non_negative_float(ground_truth_ms, "ground_truth_ms")
    slowdown_factor_clipped = _require_non_negative_float(
        slowdown_factor_clipped, "slowdown_factor_clipped"
    )
    if max_iters <= 0:
        raise ValueError(f"max_iters must be > 0, got {max_iters}")
    tol_ms = _require_non_negative_float(tol_ms, "tol_ms")

    if ground_truth_ms == 0 or slowdown_factor_clipped == 0 or not active_intervals_ms:
        return round(ground_truth_ms, 6)

    predicted_duration_ms = float(ground_truth_ms)
    for _ in range(max_iters):
        overlap_time_ms = compute_overlap_time_ms(
            kernel_start_ms=kernel_start_ms,
            kernel_duration_ms=predicted_duration_ms,
            active_intervals_ms=active_intervals_ms,
        )
        if overlap_time_ms <= 0:
            return round(ground_truth_ms, 6)
        overlap_ratio = min(1.0, overlap_time_ms / predicted_duration_ms)
        next_duration_ms = ground_truth_ms * (1.0 + overlap_ratio * slowdown_factor_clipped)
        if abs(next_duration_ms - predicted_duration_ms) <= tol_ms:
            return round(next_duration_ms, 6)
        predicted_duration_ms = next_duration_ms
    return round(predicted_duration_ms, 6)


def _load_scaler_spec(path: Path) -> Dict[str, object]:
    payload = _load_json_file(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Slowdown scaler spec must be a JSON object: {path}")
    for required_key in ("feature_names", "mean", "scale"):
        if required_key not in payload:
            raise RuntimeError(
                f"Slowdown scaler spec is missing required key {required_key!r}: {path}"
            )
    feature_names = payload["feature_names"]
    mean = payload["mean"]
    scale = payload["scale"]
    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("Slowdown scaler spec feature_names must be a non-empty list")
    if not isinstance(mean, list) or not isinstance(scale, list):
        raise RuntimeError("Slowdown scaler spec mean/scale must be lists")
    if not (len(feature_names) == len(mean) == len(scale)):
        raise RuntimeError("Slowdown scaler spec feature_names/mean/scale length mismatch")
    normalized_names = []
    normalized_mean = []
    normalized_scale = []
    for index, feature_name in enumerate(feature_names):
        if not isinstance(feature_name, str) or not feature_name:
            raise RuntimeError("Slowdown scaler spec contains an invalid feature name")
        normalized_names.append(feature_name)
        normalized_mean.append(float(mean[index]))
        scale_value = float(scale[index])
        if scale_value == 0:
            raise RuntimeError(
                f"Slowdown scaler spec scale for feature {feature_name!r} must be non-zero"
            )
        normalized_scale.append(scale_value)
    return {
        "feature_names": normalized_names,
        "mean": normalized_mean,
        "scale": normalized_scale,
    }


def _apply_scaler_spec(
    features: Mapping[str, float],
    scaler_spec: Mapping[str, object],
) -> Dict[str, float]:
    feature_names = list(scaler_spec["feature_names"])
    means = list(scaler_spec["mean"])
    scales = list(scaler_spec["scale"])
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"Missing slowdown features required by scaler: {missing}")
    normalized = {}
    for index, feature_name in enumerate(feature_names):
        normalized[feature_name] = (
            _require_non_negative_float(features[feature_name], feature_name) - float(means[index])
        ) / float(scales[index])
    return normalized


class EchoSlowdownPredictor:
    """Thin adapter mirroring `Echo-slowdown/training_testing/prediction_api.py`."""

    def __init__(self, model_path: str, scaler_path: str | None = None):
        self.model_path = str(model_path)
        resolved_scaler_path = scaler_path
        if resolved_scaler_path is None:
            resolved_scaler_path = str(Path(model_path).with_name("standard_scaler.json"))
        self.scaler_path = str(resolved_scaler_path)
        try:
            import xgboost as xgb  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Slowdown prediction requires `xgboost`; install it before enabling slowdown."
            ) from exc
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Slowdown prediction requires `pandas`; install it before enabling slowdown."
            ) from exc
        self._xgb = xgb
        self._pd = pd
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        self.scaler_spec = _load_scaler_spec(Path(self.scaler_path))

    def predict_slowdown(self, input_features: Mapping[str, object], input_overlap_ratio: float) -> dict:
        normalized = _validate_feature_mapping(input_features, REQUIRED_SLOWDOWN_FEATURES)
        overlap_ratio = _require_overlap_ratio(input_overlap_ratio)
        scaled_features = _apply_scaler_spec(normalized, self.scaler_spec)

        df_single_row = self._pd.DataFrame(
            [scaled_features],
            columns=list(self.scaler_spec["feature_names"]),
        )
        dtest = self._xgb.DMatrix(df_single_row)
        slowdown_predictions = self.model.predict(dtest)
        slowdown_predictions = self._pd.Series(slowdown_predictions)
        slowdown_predictions_clipped = slowdown_predictions.clip(lower=0)

        predicted_slowdown_factor = float(slowdown_predictions.iloc[0])
        predicted_slowdown_factor_clipped = float(slowdown_predictions_clipped.iloc[0])
        ground_truth = normalized["ground_truth"]
        predicted_execution_time = (1 - overlap_ratio) * ground_truth + overlap_ratio * ground_truth * (
            1 + predicted_slowdown_factor
        )
        predicted_execution_time_clipped = (1 - overlap_ratio) * ground_truth + overlap_ratio * ground_truth * (
            1 + predicted_slowdown_factor_clipped
        )
        return {
            "original_execution_time": ground_truth,
            "predicted_execution_time": predicted_execution_time,
            "predicted_execution_time_clipped": predicted_execution_time_clipped,
            "predicted_slowdown_factor": predicted_slowdown_factor,
            "predicted_slowdown_factor_clipped": predicted_slowdown_factor_clipped,
        }

    def predict_clipped_slowdown_factor(self, input_features: Mapping[str, object]) -> float:
        result = self.predict_slowdown(input_features=input_features, input_overlap_ratio=1.0)
        return float(result["predicted_slowdown_factor_clipped"])

    def predict_slowdown_factor(
        self,
        kernel_name: str,
        ground_truth_ms: float,
        feature_row: Mapping[str, object],
    ) -> float:
        del kernel_name
        input_features = dict(feature_row)
        if "ground_truth" in input_features:
            normalized_ground_truth = _require_non_negative_float(
                input_features["ground_truth"], "ground_truth"
            )
            requested_ground_truth = _require_non_negative_float(ground_truth_ms, "ground_truth_ms")
            if normalized_ground_truth != requested_ground_truth:
                raise ValueError(
                    "feature_row['ground_truth'] does not match ground_truth_ms passed to predict_slowdown_factor"
                )
        else:
            input_features["ground_truth"] = _require_non_negative_float(
                ground_truth_ms, "ground_truth_ms"
            )
        return self.predict_clipped_slowdown_factor(input_features)


@dataclass(frozen=True)
class SlowdownAssets:
    manifest: Mapping[str, object]
    kernel_features: Mapping[str, Mapping[str, float]]
    backward_kernel_blueprints: Mapping[str, Mapping[str, object]]


def _load_json_file(path: Path) -> object:
    def _reject_duplicate_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise RuntimeError(f"Duplicate JSON key {key!r} found in slowdown asset file: {path}")
            obj[key] = value
        return obj

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing slowdown asset file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in slowdown asset file: {path}") from exc


def _validate_manifest(manifest: Mapping[str, object]) -> Dict[str, object]:
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise RuntimeError(f"Slowdown manifest is missing required keys: {missing}")
    if manifest.get("scope") != "ddp_backward_only":
        raise RuntimeError(
            f"Unsupported slowdown asset scope {manifest.get('scope')!r}; expected 'ddp_backward_only'"
        )
    if manifest.get("clip_negative_slowdown") is not True:
        raise RuntimeError("Slowdown manifest must set clip_negative_slowdown=true")
    model_path = manifest.get("model_path")
    scaler_path = manifest.get("scaler_path")
    label_prefix = manifest.get("label_prefix")
    generator_version = manifest.get("generator_version")
    if not isinstance(model_path, str) or not model_path:
        raise RuntimeError("Slowdown manifest model_path must be a non-empty string")
    if not isinstance(scaler_path, str) or not scaler_path:
        raise RuntimeError("Slowdown manifest scaler_path must be a non-empty string")
    if not isinstance(label_prefix, str) or not label_prefix:
        raise RuntimeError("Slowdown manifest label_prefix must be a non-empty string")
    if not isinstance(generator_version, str) or not generator_version:
        raise RuntimeError("Slowdown manifest generator_version must be a non-empty string")
    return dict(manifest)


def load_slowdown_assets(assets_dir: str) -> SlowdownAssets:
    base_dir = Path(assets_dir)
    manifest_path = base_dir / "manifest.json"
    kernel_features_path = base_dir / "kernel_features.json"
    blueprints_path = base_dir / "backward_kernel_blueprints.json"

    manifest = _load_json_file(manifest_path)
    kernel_features = _load_json_file(kernel_features_path)
    backward_kernel_blueprints = _load_json_file(blueprints_path)

    if not isinstance(manifest, dict):
        raise RuntimeError("Slowdown manifest.json must contain a JSON object")
    if not isinstance(kernel_features, dict):
        raise RuntimeError("kernel_features.json must contain a JSON object")
    if not isinstance(backward_kernel_blueprints, dict):
        raise RuntimeError("backward_kernel_blueprints.json must contain a JSON object")

    normalized_manifest = _validate_manifest(manifest)

    normalized_kernel_features: Dict[str, Dict[str, float]] = {}
    for kernel_name, raw_features in kernel_features.items():
        if not isinstance(kernel_name, str) or not kernel_name:
            raise RuntimeError("kernel_features.json contains an invalid kernel name key")
        if not isinstance(raw_features, dict):
            raise RuntimeError(f"kernel_features[{kernel_name!r}] must be a JSON object")
        normalized_kernel_features[kernel_name] = _validate_feature_mapping(
            raw_features,
            REQUIRED_KERNEL_FEATURES,
        )

    normalized_blueprints: Dict[str, Dict[str, object]] = {}
    for cmd_uid, raw_blueprint in backward_kernel_blueprints.items():
        if not isinstance(cmd_uid, str) or not cmd_uid:
            raise RuntimeError("backward_kernel_blueprints.json contains an invalid cmd_uid key")
        if not isinstance(raw_blueprint, dict):
            raise RuntimeError(f"backward_kernel_blueprints[{cmd_uid!r}] must be a JSON object")
        for required_key in (
            "rank",
            "stage_id",
            "batch_id",
            "iter_id",
            "mg_state",
            "baseline_duration_ms",
            "kernels",
            "launch_markers",
        ):
            if required_key not in raw_blueprint:
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] is missing required key {required_key!r}"
                )
        if not isinstance(raw_blueprint.get("mg_state"), str) or not raw_blueprint["mg_state"]:
            raise RuntimeError(
                f"backward_kernel_blueprints[{cmd_uid!r}] must provide a non-empty mg_state"
            )
        _require_non_negative_float(raw_blueprint.get("baseline_duration_ms"), "baseline_duration_ms")

        kernels = raw_blueprint["kernels"]
        if not isinstance(kernels, list) or not kernels:
            raise RuntimeError(
                f"backward_kernel_blueprints[{cmd_uid!r}]['kernels'] must be a non-empty list"
            )
        launch_markers = raw_blueprint["launch_markers"]
        if not isinstance(launch_markers, list):
            raise RuntimeError(
                f"backward_kernel_blueprints[{cmd_uid!r}]['launch_markers'] must be a list"
            )

        seen_comm_uids = set()
        normalized_kernels: List[Dict[str, object]] = []
        for kernel in kernels:
            if not isinstance(kernel, dict):
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] kernel entry must be an object"
                )
            kernel_name = kernel.get("kernel_name")
            if not isinstance(kernel_name, str) or not kernel_name:
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] kernel entry is missing kernel_name"
                )
            if kernel_name not in normalized_kernel_features:
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] references missing kernel features for {kernel_name!r}"
                )
            normalized_kernels.append(
                {
                    "kernel_name": kernel_name,
                    "start_offset_ms": _require_non_negative_float(
                        kernel.get("start_offset_ms"), "start_offset_ms"
                    ),
                    "baseline_duration_ms": _require_non_negative_float(
                        kernel.get("baseline_duration_ms"), "baseline_duration_ms"
                    ),
                }
            )

        normalized_launch_markers: List[Dict[str, object]] = []
        for marker in launch_markers:
            if not isinstance(marker, dict):
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] launch marker must be an object"
                )
            for required_key in ("comm_uid", "baseline_offset_ms", "bucket_id", "buffer_id"):
                if required_key not in marker:
                    raise RuntimeError(
                        f"backward_kernel_blueprints[{cmd_uid!r}] launch marker is missing {required_key!r}"
                    )
            comm_uid = marker.get("comm_uid")
            if not isinstance(comm_uid, str) or not comm_uid:
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] launch marker is missing comm_uid"
                )
            if comm_uid in seen_comm_uids:
                raise RuntimeError(
                    f"backward_kernel_blueprints[{cmd_uid!r}] contains duplicate comm_uid {comm_uid!r}"
                )
            seen_comm_uids.add(comm_uid)
            normalized_launch_markers.append(
                {
                    "comm_uid": comm_uid,
                    "baseline_offset_ms": _require_non_negative_float(
                        marker.get("baseline_offset_ms"), "baseline_offset_ms"
                    ),
                    "bucket_id": marker.get("bucket_id"),
                    "buffer_id": marker.get("buffer_id"),
                }
            )

        normalized_blueprints[cmd_uid] = {
            "rank": raw_blueprint["rank"],
            "stage_id": raw_blueprint["stage_id"],
            "batch_id": raw_blueprint["batch_id"],
            "iter_id": raw_blueprint["iter_id"],
            "mg_state": raw_blueprint["mg_state"],
            "baseline_duration_ms": _require_non_negative_float(
                raw_blueprint["baseline_duration_ms"], "baseline_duration_ms"
            ),
            "kernels": normalized_kernels,
            "launch_markers": normalized_launch_markers,
        }

    return SlowdownAssets(
        manifest=normalized_manifest,
        kernel_features=normalized_kernel_features,
        backward_kernel_blueprints=normalized_blueprints,
    )
