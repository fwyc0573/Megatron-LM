#!/usr/bin/env python3
"""Seal externally qualified AE artifacts without relabelling unqualified data.

The producer manifest remains immutable.  This tool accepts a separately
attested PASS result, copies the producer tree to a new destination, records
the attestation and qualification payloads under ``provenance/``, and writes a
new manifest plus an out-of-tree receipt.  It never chooses another source,
rewrites an existing destination, or infers real hardware evidence.

The qualification-metrics file is intentionally an integrity-boundary payload,
not a task-specific semantic validator.  This wrapper requires a regular,
non-empty file and binds its exact bytes to the attestation, sealed provenance,
and receipt.  The external qualification issuer owns the task-specific metrics
schema and acceptance semantics; adding those semantics here would require a
separately versioned contract for every task.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tempfile
import uuid
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "sc26-ae-qualification-attestation-v1"
RESULT_SCHEMA_VERSION = "sc26-ae-qualification-result-v1"
RECEIPT_SCHEMA_VERSION = "sc26-ae-qualification-seal-receipt-v1"
IMAGE_REFERENCE = "hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
COMMAND_STATUS_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")

ATTESTATION_KEYS = {
    "schema_version",
    "attestation_id",
    "status",
    "target_task",
    "target_model",
    "target_execution_evidence",
    "source_commits",
    "source_manifest_sha256",
    "image_ref",
    "image_digest",
    "gpu_model",
    "gpu_count",
    "gpu_uuids",
    "qualification_result_path",
    "qualification_result_sha256",
    "qualification_metrics_path",
    "qualification_metrics_sha256",
}
QUALIFICATION_RESULT_KEYS = {
    "schema_version",
    "status",
    "task",
    "model",
    "image_ref",
    "image_digest",
    "source_commits",
    "gpu_model",
    "gpu_count",
    "gpu_uuids",
    "command_status",
    "issuer",
    "qualification_source",
}
RECEIPT_KEYS = {
    "schema_version",
    "status",
    "attestation_id",
    "target_task",
    "target_model",
    "execution_evidence",
    "source_manifest_sha256",
    "sealed_manifest_sha256",
    "qualification_attestation_sha256",
    "qualification_result_sha256",
    "qualification_metrics_sha256",
    "source_commits",
    "image_ref",
    "image_digest",
    "gpu_model",
    "gpu_count",
    "gpu_uuids",
    "qualification_result_issuer",
    "qualification_source",
    "sealed_file_count",
    "sealed_total_bytes",
}
SOURCE_COMMIT_KEYS = {"megatron_lm", "echo_slowdown", "megatron_sim_engine"}
ALLOWED_MODELS = {"gpt175b", "qwen3_a30b", "dsv3", "shared_task2"}

# These are the only producer labels that can be promoted by an external
# attestation.  Synthetic labels and already-qualified labels are terminal for
# this operation and are never silently upgraded or re-sealed.
PROMOTABLE_SOURCE_EVIDENCE = {
    "task1": "runtime_measurement_requires_external_single_gpu_qualification",
    "task2": "runtime_measurement_requires_external_two_gpu_qualification",
    "task3": "runtime_measurement_requires_external_single_gpu_qualification",
}
TARGET_EVIDENCE = {
    "task1": "real_single_h800_qualified",
    "task2": "real_exact_two_h800_qualified",
    "task3": "real_single_h800_qualified",
}
EXPECTED_GPU_COUNT = {"task1": 1, "task2": 2, "task3": 1}


def _regular_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("{} must be a regular directory: {}".format(label, path))
    return path.resolve(strict=True)


def _regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("{} must be a regular file: {}".format(label, path))
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("{} must be a regular file: {}".format(label, path))
    return path.resolve(strict=True)


def _safe_relative(value: object, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{} must be a non-empty POSIX relative path".format(label))
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError("{} is unsafe: {}".format(label, value))
    return path


def _sha256(path: pathlib.Path, label: str = "hash input") -> str:
    path = _regular_file(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: pathlib.Path, label: str) -> MutableMapping[str, Any]:
    path = _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("{} is invalid JSON: {}".format(label, path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("{} must be a JSON object: {}".format(label, path))
    return payload


def _load_artifact_module() -> Any:
    module_path = pathlib.Path(__file__).resolve().with_name("artifact_manifest.py")
    _regular_file(module_path, "artifact manifest tool")
    specification = importlib.util.spec_from_file_location(
        "sc26_ae_seal_artifact_manifest", module_path
    )
    if specification is None or specification.loader is None:
        raise ValueError("cannot load artifact manifest tool")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _require_exact_keys(name: str, payload: Mapping[str, Any], expected: set[str]) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            "{} keys must be exactly {}; got {}".format(
                name, sorted(expected), sorted(observed)
            )
        )


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be a path-free identifier".format(name))
    return value


def _require_hash(name: str, value: object) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA256".format(name))
    return value


def _require_audit_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("{} must be a non-empty printable string".format(name))
    return value


def _assert_no_symlink_components(
    path: pathlib.Path, root: pathlib.Path, label: str
) -> None:
    """Reject symlinks in every component of a path under ``root``.

    Resolving first would erase the evidence that the requested payload was a
    symlink.  ``lstat`` is therefore used on the lexical path before any file
    is opened or hashed.
    """

    path = pathlib.Path(path)
    root = pathlib.Path(root)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("{} escapes its root: {}".format(label, path)) from exc
    current = root
    parts = relative.parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValueError("{} is not accessible: {}".format(label, current)) from exc
        if stat.S_ISLNK(mode):
            raise ValueError("{} contains a symlink: {}".format(label, current))
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ValueError("{} contains a non-directory component: {}".format(label, current))


def _manifest_entries(manifest: Mapping[str, Any]) -> list[tuple[pathlib.PurePosixPath, int, str]]:
    """Return the source manifest's exact declared payload entries."""

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source artifact manifest files must be a non-empty list")
    entries: list[tuple[pathlib.PurePosixPath, int, str]] = []
    observed: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("source artifact manifest has an invalid file entry")
        relative = _safe_relative(entry["path"], "source manifest file path")
        if relative.as_posix() != entry["path"] or relative.as_posix() in observed:
            raise ValueError("source artifact manifest has a duplicate or unsafe file path")
        observed.add(relative.as_posix())
        size = entry["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("source manifest size_bytes is invalid for {}".format(relative))
        digest = _require_hash("source manifest sha256 for {}".format(relative), entry["sha256"])
        entries.append((relative, size, digest))
    if [relative.as_posix() for relative, _size, _digest in entries] != sorted(observed):
        raise ValueError("source artifact manifest files must be sorted by path")
    return entries


def _file_digest_and_size(path: pathlib.Path, label: str) -> tuple[int, str]:
    path = pathlib.Path(path)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError("{} must be a regular file: {}".format(label, path))
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("{} cannot be read: {}".format(label, path)) from exc
    return size, digest.hexdigest()


def _verify_declared_files(
    root: pathlib.Path,
    entries: Sequence[tuple[pathlib.PurePosixPath, int, str]],
    label: str,
) -> None:
    root = _regular_directory(root, label)
    expected_paths = [relative.as_posix() for relative, _size, _digest in entries]
    observed_paths = _inventory(root)
    if observed_paths != expected_paths:
        raise ValueError(
            "{} inventory differs from manifest: expected {}, got {}".format(
                label, expected_paths, observed_paths
            )
        )
    for relative, expected_size, expected_digest in entries:
        path = root / pathlib.PurePosixPath(relative)
        actual_size, actual_digest = _file_digest_and_size(path, "{} file".format(label))
        if actual_size != expected_size or actual_digest != expected_digest:
            raise ValueError(
                "{} changed for {} (expected {} bytes/{}, got {} bytes/{})".format(
                    label,
                    relative,
                    expected_size,
                    expected_digest,
                    actual_size,
                    actual_digest,
                )
            )


def _inventory(root: pathlib.Path) -> list[str]:
    root = _regular_directory(root, "artifact root")
    paths: list[str] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = pathlib.Path(current_root)
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError("artifact tree contains a non-directory or symlink: {}".format(path))
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError("artifact tree contains a non-regular file or symlink: {}".format(path))
            relative = path.relative_to(root).as_posix()
            if relative != "artifact_manifest.json":
                paths.append(relative)
    return sorted(paths)


def _assert_not_inside(path: pathlib.Path, root: pathlib.Path, label: str) -> None:
    candidate = pathlib.Path(path).resolve(strict=False)
    root = pathlib.Path(root).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    raise ValueError("{} must not be inside source root: {}".format(label, path))


def _resolve_attestation_payload(
    attestation_path: pathlib.Path, relative: object, label: str
) -> pathlib.Path:
    relative_path = _safe_relative(relative, label)
    parent = attestation_path.parent.resolve(strict=True)
    candidate = parent / pathlib.Path(relative_path)
    _assert_no_symlink_components(candidate, parent, label)
    return _regular_file(candidate, label)


def _target_contract(manifest: Mapping[str, Any]) -> tuple[str, str, str, int]:
    task = manifest.get("task")
    model = manifest.get("model")
    if task not in TARGET_EVIDENCE:
        raise ValueError("qualification sealing does not support task: {}".format(task))
    if model not in ALLOWED_MODELS:
        raise ValueError("source manifest model is invalid: {}".format(model))
    if task == "task2" and model != "shared_task2":
        raise ValueError("Task2 source manifest model must be shared_task2")
    if task != "task2" and model == "shared_task2":
        raise ValueError("{} source manifest cannot use shared_task2".format(task))
    return task, model, TARGET_EVIDENCE[task], EXPECTED_GPU_COUNT[task]


def _validate_result_payload(
    result_payload: Mapping[str, Any],
    *,
    task: str,
    model: str,
    attestation: Mapping[str, Any],
) -> tuple[str, str]:
    """Validate the canonical external qualification-result wrapper.

    Historical worker payloads are intentionally not accepted directly.  The
    external qualification orchestrator must emit this normalized wrapper so
    every hardware/image/source fact used for promotion is explicit and
    auditable.
    """

    _require_exact_keys("qualification result", result_payload, QUALIFICATION_RESULT_KEYS)
    if result_payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("qualification result schema_version is invalid")
    if result_payload["status"] != "PASS":
        raise ValueError("qualification result status must be PASS")
    if result_payload["task"] != task:
        raise ValueError("qualification result task differs from attestation")
    if result_payload["model"] != model:
        raise ValueError("qualification result model differs from attestation")

    for field in (
        "image_ref",
        "image_digest",
        "gpu_model",
        "gpu_count",
        "gpu_uuids",
    ):
        if result_payload[field] != attestation[field]:
            raise ValueError("qualification result {} differs from attestation".format(field))
    if result_payload["image_ref"] != IMAGE_REFERENCE:
        raise ValueError(
            "qualification result image_ref must equal {}".format(IMAGE_REFERENCE)
        )
    if not isinstance(result_payload["image_digest"], str) or IMAGE_DIGEST_PATTERN.fullmatch(
        result_payload["image_digest"]
    ) is None:
        raise ValueError("qualification result image_digest must be immutable")
    if result_payload["gpu_model"] != "NVIDIA H800":
        raise ValueError("qualification result gpu_model must be NVIDIA H800")
    count = result_payload["gpu_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("qualification result gpu_count must be a positive integer")
    uuids = result_payload["gpu_uuids"]
    if not isinstance(uuids, list) or len(uuids) != count:
        raise ValueError("qualification result gpu_uuids must match gpu_count")
    if any(
        not isinstance(value, str) or GPU_UUID_PATTERN.fullmatch(value) is None
        for value in uuids
    ):
        raise ValueError("qualification result gpu_uuids contains an invalid identity")
    if len(set(uuids)) != len(uuids):
        raise ValueError("qualification result gpu_uuids must be distinct")

    source_commits = result_payload["source_commits"]
    if not isinstance(source_commits, Mapping):
        raise ValueError("qualification result source_commits must be an object")
    _require_exact_keys(
        "qualification result source_commits", source_commits, SOURCE_COMMIT_KEYS
    )
    if dict(source_commits) != dict(attestation["source_commits"]):
        raise ValueError("qualification result source_commits differs from attestation")
    for key, value in source_commits.items():
        if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("qualification result source_commits.{} is invalid".format(key))

    command_status = result_payload["command_status"]
    if not isinstance(command_status, Mapping) or not command_status:
        raise ValueError("qualification result command_status must be a non-empty object")
    for command, exit_code in command_status.items():
        if not isinstance(command, str) or COMMAND_STATUS_KEY_PATTERN.fullmatch(command) is None:
            raise ValueError("qualification result command_status has an invalid command key")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
            raise ValueError(
                "qualification result command_status.{} must be integer zero".format(command)
            )

    issuer = _require_audit_string("qualification result issuer", result_payload["issuer"])
    qualification_source = _require_audit_string(
        "qualification result qualification_source", result_payload["qualification_source"]
    )
    return issuer, qualification_source


def validate_receipt(
    receipt_payload: Mapping[str, Any], expected: Mapping[str, Any] | None = None
) -> None:
    """Validate one on-disk seal receipt before it is trusted by a caller."""

    _require_exact_keys("qualification seal receipt", receipt_payload, RECEIPT_KEYS)
    if receipt_payload["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("qualification seal receipt schema_version is invalid")
    if receipt_payload["status"] != "PASS":
        raise ValueError("qualification seal receipt status must be PASS")
    _require_id("attestation_id", receipt_payload["attestation_id"])
    task = receipt_payload["target_task"]
    model = receipt_payload["target_model"]
    if task not in TARGET_EVIDENCE or model not in ALLOWED_MODELS:
        raise ValueError("qualification seal receipt target is invalid")
    if task == "task2" and model != "shared_task2":
        raise ValueError("Task2 receipt model must be shared_task2")
    if task != "task2" and model == "shared_task2":
        raise ValueError("{} receipt cannot use shared_task2".format(task))
    if receipt_payload["execution_evidence"] != TARGET_EVIDENCE[task]:
        raise ValueError("qualification seal receipt execution evidence is invalid")
    for field in (
        "source_manifest_sha256",
        "sealed_manifest_sha256",
        "qualification_attestation_sha256",
        "qualification_result_sha256",
        "qualification_metrics_sha256",
    ):
        _require_hash("qualification seal receipt {}".format(field), receipt_payload[field])
    commits = receipt_payload["source_commits"]
    if not isinstance(commits, Mapping):
        raise ValueError("qualification seal receipt source_commits must be an object")
    _require_exact_keys("qualification seal receipt source_commits", commits, SOURCE_COMMIT_KEYS)
    for key, value in commits.items():
        if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("qualification seal receipt source_commits.{} is invalid".format(key))
    if receipt_payload["image_ref"] != IMAGE_REFERENCE:
        raise ValueError("qualification seal receipt image_ref is invalid")
    if not isinstance(receipt_payload["image_digest"], str) or IMAGE_DIGEST_PATTERN.fullmatch(
        receipt_payload["image_digest"]
    ) is None:
        raise ValueError("qualification seal receipt image_digest is invalid")
    if receipt_payload["gpu_model"] != "NVIDIA H800":
        raise ValueError("qualification seal receipt gpu_model is invalid")
    expected_gpu_count = EXPECTED_GPU_COUNT[task]
    gpu_count = receipt_payload["gpu_count"]
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count != expected_gpu_count:
        raise ValueError("qualification seal receipt gpu_count is invalid")
    gpu_uuids = receipt_payload["gpu_uuids"]
    if not isinstance(gpu_uuids, list) or len(gpu_uuids) != gpu_count:
        raise ValueError("qualification seal receipt gpu_uuids is invalid")
    if any(not isinstance(value, str) or GPU_UUID_PATTERN.fullmatch(value) is None for value in gpu_uuids):
        raise ValueError("qualification seal receipt gpu_uuids is invalid")
    if len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("qualification seal receipt gpu_uuids must be distinct")
    _require_audit_string(
        "qualification seal receipt qualification_result_issuer",
        receipt_payload["qualification_result_issuer"],
    )
    _require_audit_string(
        "qualification seal receipt qualification_source",
        receipt_payload["qualification_source"],
    )
    for field in ("sealed_file_count", "sealed_total_bytes"):
        value = receipt_payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("qualification seal receipt {} is invalid".format(field))
    if expected is not None and dict(receipt_payload) != dict(expected):
        raise ValueError("qualification seal receipt differs from prepared receipt")


def validate_attestation(
    attestation_path: pathlib.Path,
    source_root: pathlib.Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete external attestation against one source manifest."""

    attestation_path = _regular_file(attestation_path, "qualification attestation")
    source_root = _regular_directory(source_root, "source root")
    payload = _load_json(attestation_path, "qualification attestation")
    _require_exact_keys("qualification attestation", payload, ATTESTATION_KEYS)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("qualification attestation schema_version is invalid")
    _require_id("attestation_id", payload["attestation_id"])
    if payload["status"] != "PASS":
        raise ValueError("qualification attestation status must be PASS")

    task, model, target_evidence, expected_gpu_count = _target_contract(source_manifest)
    if task == "task1":
        # Keep the external promotion boundary fail-closed for MoE QUICK
        # captures.  Synthetic/local bundles are not in this evidence set and
        # remain usable for smoke testing; only pending/qualified promotion
        # inputs are required to carry the exact full-rank inventory.
        artifact_module = _load_artifact_module()
        artifact_module.validate_task1_rank_promotion_scope(source_manifest)
    if payload["target_task"] != task:
        raise ValueError("attestation target_task differs from source manifest")
    if payload["target_model"] != model:
        raise ValueError("attestation target_model differs from source manifest")
    if payload["target_execution_evidence"] != target_evidence:
        raise ValueError("attestation target execution evidence is invalid")

    source_manifest_path = source_root / "artifact_manifest.json"
    actual_source_manifest_sha256 = _sha256(source_manifest_path, "source artifact manifest")
    declared_source_manifest_sha256 = _require_hash(
        "source_manifest_sha256", payload["source_manifest_sha256"]
    )
    if declared_source_manifest_sha256 != actual_source_manifest_sha256:
        raise ValueError("source manifest SHA256 does not match source artifact manifest")

    commits = payload["source_commits"]
    if not isinstance(commits, Mapping):
        raise ValueError("source_commits must be an object")
    _require_exact_keys("source_commits", commits, SOURCE_COMMIT_KEYS)
    source_commits = source_manifest.get("source_commits")
    if not isinstance(source_commits, Mapping):
        raise ValueError("source manifest source_commits must be an object")
    _require_exact_keys("source manifest source_commits", source_commits, SOURCE_COMMIT_KEYS)
    for key in sorted(SOURCE_COMMIT_KEYS):
        if not isinstance(commits[key], str) or COMMIT_PATTERN.fullmatch(commits[key]) is None:
            raise ValueError("source_commits.{} is not a 40-hex commit".format(key))
        if commits[key] != source_commits[key]:
            raise ValueError("source commit {} differs from source manifest".format(key))

    if payload["image_ref"] != IMAGE_REFERENCE:
        raise ValueError("image_ref must equal {}".format(IMAGE_REFERENCE))
    if not isinstance(payload["image_digest"], str) or IMAGE_DIGEST_PATTERN.fullmatch(
        payload["image_digest"]
    ) is None:
        raise ValueError("image_digest must be an immutable sha256 digest")
    if payload["gpu_model"] != "NVIDIA H800":
        raise ValueError("gpu_model must be NVIDIA H800")
    count = payload["gpu_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_gpu_count:
        raise ValueError(
            "gpu_count must equal {} for {}".format(expected_gpu_count, task)
        )
    uuids = payload["gpu_uuids"]
    if not isinstance(uuids, list) or len(uuids) != count:
        raise ValueError("gpu_uuids must list exactly gpu_count devices")
    if any(not isinstance(value, str) or GPU_UUID_PATTERN.fullmatch(value) is None for value in uuids):
        raise ValueError("gpu_uuids contains an invalid device identity")
    if len(set(uuids)) != len(uuids):
        raise ValueError("gpu_uuids must be distinct")

    source_evidence = source_manifest.get("execution_evidence")
    if source_evidence != PROMOTABLE_SOURCE_EVIDENCE[task]:
        raise ValueError(
            "source execution evidence is not promotable; synthetic or already-qualified evidence cannot be upgraded"
        )

    result_path = _resolve_attestation_payload(
        attestation_path, payload["qualification_result_path"], "qualification result path"
    )
    metrics_path = _resolve_attestation_payload(
        attestation_path, payload["qualification_metrics_path"], "qualification metrics path"
    )
    if result_path in {metrics_path, attestation_path} or metrics_path == attestation_path:
        raise ValueError(
            "qualification result, metrics, and attestation must be different files"
        )
    result_sha256 = _require_hash(
        "qualification_result_sha256", payload["qualification_result_sha256"]
    )
    metrics_sha256 = _require_hash(
        "qualification_metrics_sha256", payload["qualification_metrics_sha256"]
    )
    if result_sha256 != _sha256(result_path, "qualification result"):
        raise ValueError("qualification result SHA256 does not match payload")
    if metrics_sha256 != _sha256(metrics_path, "qualification metrics"):
        raise ValueError("qualification metrics SHA256 does not match payload")
    result_payload = _load_json(result_path, "qualification result")
    issuer, qualification_source = _validate_result_payload(
        result_payload,
        task=task,
        model=model,
        attestation=payload,
    )
    if metrics_path.stat().st_size <= 0:
        raise ValueError("qualification metrics must be non-empty")

    return {
        "attestation": payload,
        "attestation_path": attestation_path,
        "result_path": result_path,
        "metrics_path": metrics_path,
        "source_manifest_sha256": actual_source_manifest_sha256,
        "qualification_result_sha256": result_sha256,
        "qualification_metrics_sha256": metrics_sha256,
        "task": task,
        "model": model,
        "target_execution_evidence": target_evidence,
        "result_payload": result_payload,
        "issuer": issuer,
        "qualification_source": qualification_source,
    }


def _copy_verified_file(
    source: pathlib.Path,
    destination: pathlib.Path,
    expected_size: int,
    expected_digest: str,
    label: str,
) -> None:
    """Copy one regular file while hashing the exact bytes that were copied."""

    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    source_mode = source.lstat().st_mode
    if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
        raise ValueError("{} source must be a regular file: {}".format(label, source))
    if destination.exists() or destination.is_symlink():
        raise ValueError("duplicate sealed payload path: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                destination_handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    except OSError as exc:
        raise ValueError("{} copy failed: {}".format(label, source)) from exc
    actual_digest = digest.hexdigest()
    if size != expected_size or actual_digest != expected_digest:
        raise ValueError(
            "{} changed while copying {} (expected {} bytes/{}, got {} bytes/{})".format(
                label, source, expected_size, expected_digest, size, actual_digest
            )
        )


def _copy_source_tree(
    source_root: pathlib.Path,
    destination: pathlib.Path,
    entries: Sequence[tuple[pathlib.PurePosixPath, int, str]],
) -> None:
    _regular_directory(destination, "sealed staging root")
    for relative, expected_size, expected_digest in entries:
        source_path = source_root / pathlib.PurePosixPath(relative)
        target = destination / pathlib.PurePosixPath(relative)
        _copy_verified_file(
            source_path,
            target,
            expected_size,
            expected_digest,
            "source payload",
        )


def _copy_provenance_file(
    source: pathlib.Path,
    destination: pathlib.Path,
    label: str,
    expected_size: int | None = None,
    expected_digest: str | None = None,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("sealed provenance destination already exists: {}".format(destination))
    source = _regular_file(source, label)
    if expected_size is None or expected_digest is None:
        expected_size, expected_digest = _file_digest_and_size(source, label)
    _copy_verified_file(source, destination, expected_size, expected_digest, label)


def _prepare_json_file(path: pathlib.Path, payload: Mapping[str, Any]) -> pathlib.Path:
    """Write a durable sibling temporary file without touching the destination."""

    path = pathlib.Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("qualification seal receipt already exists: {}".format(path))
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(parent, "receipt parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.tmp-".format(path.name),
        dir=str(parent),
        text=True,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_json_file(temporary: pathlib.Path, destination: pathlib.Path) -> None:
    """Publish a prepared JSON file atomically and never replace a destination."""

    temporary = _regular_file(temporary, "prepared receipt")
    destination = pathlib.Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("qualification seal receipt already exists: {}".format(destination))
    try:
        # A hard-link publication is atomic and has no replace semantics. Both
        # paths are in the same parent filesystem because the temporary file is
        # created there.
        os.link(temporary, destination, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("cannot atomically publish qualification seal receipt") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _make_staging_root(destination: pathlib.Path) -> pathlib.Path:
    destination = pathlib.Path(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(parent, "sealed destination parent")
    for _attempt in range(32):
        candidate = parent / ".{}-staging-{}".format(destination.name, uuid.uuid4().hex)
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("cannot allocate an exclusive staging directory")


def _publish_staging_root(staging: pathlib.Path, destination: pathlib.Path) -> None:
    """Publish a complete staging tree under an exclusive sibling lock."""

    staging = _regular_directory(staging, "sealed staging root")
    destination = pathlib.Path(destination)
    parent = destination.parent
    lock = parent / ".{}-publish-lock".format(destination.name)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("sealed destination already exists: {}".format(destination))
        # Rename is atomic within the parent. The exclusive lock serializes
        # publishers using this tool; no destination is replaced.
        os.rename(staging, destination)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _remove_owned_directory(
    path: pathlib.Path, expected_stat: os.stat_result | None = None
) -> None:
    path = pathlib.Path(path)
    if not path.exists() or path.is_symlink() or not path.is_dir():
        return
    if expected_stat is not None:
        actual_stat = path.stat()
        if (actual_stat.st_dev, actual_stat.st_ino) != (
            expected_stat.st_dev,
            expected_stat.st_ino,
        ):
            return
    shutil.rmtree(path)


def _remove_owned_file(path: pathlib.Path, expected_stat: os.stat_result | None = None) -> None:
    path = pathlib.Path(path)
    if not path.exists() or path.is_symlink() or not path.is_file():
        return
    if expected_stat is not None:
        actual_stat = path.stat()
        if (actual_stat.st_dev, actual_stat.st_ino) != (
            expected_stat.st_dev,
            expected_stat.st_ino,
        ):
            return
    path.unlink()


def _verify_expected_file(
    path: pathlib.Path, expected_size: int, expected_digest: str, label: str
) -> None:
    actual_size, actual_digest = _file_digest_and_size(path, label)
    if actual_size != expected_size or actual_digest != expected_digest:
        raise ValueError(
            "{} mismatch (expected {} bytes/{}, got {} bytes/{})".format(
                label, expected_size, expected_digest, actual_size, actual_digest
            )
        )


def seal_qualification(
    source_root: pathlib.Path,
    attestation_path: pathlib.Path,
    destination_root: pathlib.Path,
    receipt_path: pathlib.Path,
) -> dict[str, Any]:
    """Create one immutable, qualification-sealed copy of a producer bundle."""

    source_root = _regular_directory(source_root, "source root")
    attestation_path = _regular_file(attestation_path, "qualification attestation")
    destination_root = pathlib.Path(destination_root)
    receipt_path = pathlib.Path(receipt_path)
    if destination_root.exists() or destination_root.is_symlink():
        raise FileExistsError("sealed destination already exists: {}".format(destination_root))
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("qualification seal receipt already exists: {}".format(receipt_path))
    _assert_not_inside(destination_root, source_root, "sealed destination")
    _assert_not_inside(receipt_path, source_root, "seal receipt")
    _assert_not_inside(attestation_path, source_root, "qualification attestation")
    destination_candidate = destination_root.resolve(strict=False)
    receipt_candidate = receipt_path.resolve(strict=False)
    try:
        receipt_candidate.relative_to(destination_candidate)
    except ValueError:
        pass
    else:
        raise ValueError("seal receipt must be outside sealed destination")

    source_manifest_path = _regular_file(source_root / "artifact_manifest.json", "source artifact manifest")
    source_manifest = _load_json(source_manifest_path, "source artifact manifest")
    artifact_module = _load_artifact_module()
    try:
        artifact_module.verify_manifest(source_root, source_manifest)
    except Exception as exc:
        raise ValueError("source artifact manifest verification failed: {}".format(exc)) from exc
    source_entries = _manifest_entries(source_manifest)
    _verify_declared_files(source_root, source_entries, "source payload")
    source_manifest_sha256 = _sha256(source_manifest_path, "source artifact manifest")
    evidence = validate_attestation(attestation_path, source_root, source_manifest)

    attestation_size, attestation_sha256 = _file_digest_and_size(
        attestation_path, "qualification attestation"
    )
    result_size, result_sha256 = _file_digest_and_size(
        evidence["result_path"], "qualification result"
    )
    metrics_size, metrics_sha256 = _file_digest_and_size(
        evidence["metrics_path"], "qualification metrics"
    )
    if result_sha256 != evidence["qualification_result_sha256"]:
        raise ValueError("qualification result changed after validation")
    if metrics_sha256 != evidence["qualification_metrics_sha256"]:
        raise ValueError("qualification metrics changed after validation")

    reserved_paths = {
        "artifact_manifest.json",
        "provenance/source_artifact_manifest.json",
        "provenance/qualification_attestation.json",
        "provenance/qualification_result.json",
        "provenance/qualification_metrics.json",
    }
    collisions = sorted(
        relative.as_posix() for relative, _size, _digest in source_entries if relative.as_posix() in reserved_paths
    )
    if collisions:
        raise ValueError("source payload collides with sealed provenance paths: {}".format(collisions))

    staging_root = _make_staging_root(destination_root)
    receipt_temporary: pathlib.Path | None = None
    published_destination_stat: os.stat_result | None = None
    published_receipt_stat: os.stat_result | None = None
    try:
        _copy_source_tree(source_root, staging_root, source_entries)

        # Verify the producer did not change while it was copied.  The copied
        # bytes are checked independently below, so a manifest-valid source
        # cannot silently turn into a different sealed bundle.
        _verify_declared_files(source_root, source_entries, "source payload after copy")
        if _sha256(source_manifest_path, "source artifact manifest") != source_manifest_sha256:
            raise ValueError("source artifact manifest changed while sealing")

        provenance_root = staging_root / "provenance"
        _copy_provenance_file(
            source_manifest_path,
            provenance_root / "source_artifact_manifest.json",
            "source artifact manifest",
            expected_size=source_manifest_path.stat().st_size,
            expected_digest=source_manifest_sha256,
        )
        _copy_provenance_file(
            attestation_path,
            provenance_root / "qualification_attestation.json",
            "qualification attestation",
            expected_size=attestation_size,
            expected_digest=attestation_sha256,
        )
        _copy_provenance_file(
            evidence["result_path"],
            provenance_root / "qualification_result.json",
            "qualification result",
            expected_size=result_size,
            expected_digest=result_sha256,
        )
        _copy_provenance_file(
            evidence["metrics_path"],
            provenance_root / "qualification_metrics.json",
            "qualification metrics",
            expected_size=metrics_size,
            expected_digest=metrics_sha256,
        )

        # Re-check every source and external evidence file after copying to
        # catch a producer-side TOCTOU mutation.
        _verify_declared_files(source_root, source_entries, "source payload final snapshot")
        if _sha256(source_manifest_path, "source artifact manifest") != source_manifest_sha256:
            raise ValueError("source artifact manifest changed during provenance copy")
        _verify_expected_file(
            attestation_path,
            attestation_size,
            attestation_sha256,
            "qualification attestation final snapshot",
        )
        _verify_expected_file(
            evidence["result_path"],
            result_size,
            result_sha256,
            "qualification result final snapshot",
        )
        _verify_expected_file(
            evidence["metrics_path"],
            metrics_size,
            metrics_sha256,
            "qualification metrics final snapshot",
        )
        for relative, expected_size, expected_digest in source_entries:
            _verify_expected_file(
                staging_root / pathlib.PurePosixPath(relative),
                expected_size,
                expected_digest,
                "sealed source payload {}".format(relative),
            )
        _verify_expected_file(
            provenance_root / "source_artifact_manifest.json",
            source_manifest_path.stat().st_size,
            source_manifest_sha256,
            "sealed source artifact manifest",
        )
        _verify_expected_file(
            provenance_root / "qualification_attestation.json",
            attestation_size,
            attestation_sha256,
            "sealed qualification attestation",
        )
        _verify_expected_file(
            provenance_root / "qualification_result.json",
            result_size,
            result_sha256,
            "sealed qualification result",
        )
        _verify_expected_file(
            provenance_root / "qualification_metrics.json",
            metrics_size,
            metrics_sha256,
            "sealed qualification metrics",
        )

        metadata = {key: value for key, value in source_manifest.items() if key != "files"}
        attestation_payload = evidence["attestation"]
        metadata.update(
            {
                "execution_evidence": evidence["target_execution_evidence"],
                "qualification_status": "PASS",
                "source_manifest_sha256": source_manifest_sha256,
                "qualification_attestation_sha256": attestation_sha256,
                "qualification_result_sha256": result_sha256,
                "qualification_metrics_sha256": metrics_sha256,
                "qualification_image_ref": attestation_payload["image_ref"],
                "qualification_image_digest": attestation_payload["image_digest"],
                "qualification_gpu_model": attestation_payload["gpu_model"],
                "qualification_gpu_count": attestation_payload["gpu_count"],
                "qualification_gpu_uuids": list(attestation_payload["gpu_uuids"]),
                "qualification_result_issuer": evidence["issuer"],
                "qualification_source": evidence["qualification_source"],
                "qualification_attestation_schema": SCHEMA_VERSION,
                "qualification_result_schema": RESULT_SCHEMA_VERSION,
            }
        )
        manifest = artifact_module.create_manifest(
            staging_root, metadata, _inventory(staging_root)
        )
        manifest_path = staging_root / "artifact_manifest.json"
        _stable_json(manifest_path, manifest)
        reloaded_manifest = _load_json(manifest_path, "sealed artifact manifest")
        artifact_module.verify_manifest(staging_root, reloaded_manifest)

        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "PASS",
            "attestation_id": attestation_payload["attestation_id"],
            "target_task": evidence["task"],
            "target_model": evidence["model"],
            "execution_evidence": evidence["target_execution_evidence"],
            "source_manifest_sha256": source_manifest_sha256,
            "sealed_manifest_sha256": _sha256(manifest_path, "sealed artifact manifest"),
            "qualification_attestation_sha256": attestation_sha256,
            "qualification_result_sha256": result_sha256,
            "qualification_metrics_sha256": metrics_sha256,
            "source_commits": dict(attestation_payload["source_commits"]),
            "image_ref": attestation_payload["image_ref"],
            "image_digest": attestation_payload["image_digest"],
            "gpu_model": attestation_payload["gpu_model"],
            "gpu_count": attestation_payload["gpu_count"],
            "gpu_uuids": list(attestation_payload["gpu_uuids"]),
            "qualification_result_issuer": evidence["issuer"],
            "qualification_source": evidence["qualification_source"],
            "sealed_file_count": len(reloaded_manifest["files"]),
            "sealed_total_bytes": sum(
                entry["size_bytes"] for entry in reloaded_manifest["files"]
            ),
        }
        validate_receipt(receipt)
        receipt_temporary = _prepare_json_file(receipt_path, receipt)
        _publish_staging_root(staging_root, destination_root)
        published_destination_stat = destination_root.stat()
        _publish_json_file(receipt_temporary, receipt_path)
        receipt_temporary = None
        published_receipt_stat = receipt_path.stat()

        final_manifest = _load_json(
            destination_root / "artifact_manifest.json", "published artifact manifest"
        )
        artifact_module.verify_manifest(destination_root, final_manifest)
        final_receipt = _load_json(receipt_path, "published qualification seal receipt")
        validate_receipt(final_receipt, expected=receipt)
        if _sha256(destination_root / "artifact_manifest.json", "published artifact manifest") != receipt[
            "sealed_manifest_sha256"
        ]:
            raise ValueError("published artifact manifest hash changed after publication")
        return {
            **receipt,
            "sealed_root": str(destination_root),
            "receipt_path": str(receipt_path),
        }
    except BaseException:
        if receipt_temporary is not None:
            receipt_temporary.unlink(missing_ok=True)
        _remove_owned_directory(staging_root)
        if published_destination_stat is not None:
            _remove_owned_directory(destination_root, published_destination_stat)
        if published_receipt_stat is not None:
            _remove_owned_file(receipt_path, published_receipt_stat)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal", help="seal one externally qualified producer bundle")
    seal.add_argument("--source-root", type=pathlib.Path, required=True)
    seal.add_argument("--attestation", type=pathlib.Path, required=True)
    seal.add_argument("--destination-root", type=pathlib.Path, required=True)
    seal.add_argument("--receipt", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal":
        result = seal_qualification(
            args.source_root, args.attestation, args.destination_root, args.receipt
        )
        print("SEAL_STATUS=verified")
        print("SEAL_TASK={}".format(result["target_task"]))
        print("SEAL_MODEL={}".format(result["target_model"]))
        print("SEAL_EXECUTION_EVIDENCE={}".format(result["execution_evidence"]))
        print("SEAL_FILE_COUNT={}".format(result["sealed_file_count"]))
        print("SEAL_TOTAL_BYTES={}".format(result["sealed_total_bytes"]))
        print("SEAL_MANIFEST_SHA256={}".format(result["sealed_manifest_sha256"]))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
