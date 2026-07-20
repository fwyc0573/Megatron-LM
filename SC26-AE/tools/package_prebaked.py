#!/usr/bin/env python3
"""Build and verify a portable SC'26 AE prebaked distribution.

The packager is intentionally a sealing step, not an evidence promotion step.
It accepts only fresh bundles whose producers already carry the exact real
qualification evidence classes.  It never changes an evidence class, chooses
another source, or overwrites an existing staging root.
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
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


MODELS: Dict[str, Dict[str, Any]] = {
    "gpt175b": {
        "profile": "175",
        "topology": {
            "world_size": 1024,
            "local_size": 8,
            "pp": 8,
            "tp": 8,
            "dp": 16,
            "exp": 1,
        },
    },
    "qwen3_a30b": {
        "profile": "full",
        "topology": {
            "world_size": 256,
            "local_size": 8,
            "pp": 4,
            "tp": 8,
            "dp": 8,
            "exp": 8,
        },
    },
    "dsv3": {
        "profile": "smoke",
        "topology": {
            "world_size": 256,
            "local_size": 8,
            "pp": 4,
            "tp": 8,
            "dp": 8,
            "exp": 8,
        },
    },
}
ALL_BUNDLES = set(MODELS) | {"shared_task2"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

TASK1_REAL_EVIDENCE = "real_single_h800_qualified"
TASK2_REAL_EVIDENCE = "real_exact_two_h800_qualified"
TASK3_REAL_EVIDENCE = "real_single_h800_qualified"
TASK1_SYNTHETIC_EVIDENCE = "local_synthetic_not_gpu_qualification"
TASK2_SYNTHETIC_EVIDENCE = "local_synthetic_not_two_gpu_qualification"


def _fail(message: str) -> None:
    raise ValueError(message)


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be a path-free identifier: {}".format(label, value))
    return value


def _safe_relative(value: object, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("{} must be a non-empty POSIX relative path".format(label))
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError("{} is unsafe: {}".format(label, value))
    return path


def _regular_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("{} must be a regular directory: {}".format(label, path))
    return path.resolve(strict=True)


def _regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("{} must be a regular file: {}".format(label, path))
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("{} must be a regular file: {}".format(label, path))
    return path.resolve(strict=True)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "hash input").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: pathlib.Path, label: str) -> MutableMapping[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("{} is invalid JSON: {}".format(label, path)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object: {}".format(label, path))
    return value


def _git(repo_root: pathlib.Path, *args: str) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("git command failed in {}: {}".format(repo_root, args)) from exc
    if not COMMIT_PATTERN.fullmatch(value):
        raise ValueError("git command returned a non-commit identity: {}".format(value))
    return value


def _source_commits(repo_root: pathlib.Path) -> Dict[str, str]:
    return {
        "megatron_lm": _git(repo_root, "rev-parse", "HEAD"),
        "echo_slowdown": _git(repo_root, "rev-parse", "HEAD:Echo-slowdown"),
        "megatron_sim_engine": _git(repo_root, "rev-parse", "HEAD:megatron-sim-engine"),
    }


def _load_artifact_module(repo_root: pathlib.Path) -> Any:
    module_path = repo_root / "SC26-AE" / "tools" / "artifact_manifest.py"
    _regular_file(module_path, "artifact manifest tool")
    specification = importlib.util.spec_from_file_location(
        "sc26_ae_package_artifact_manifest", module_path
    )
    if specification is None or specification.loader is None:
        raise ValueError("cannot load artifact manifest tool")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _manifest_files(root: pathlib.Path) -> list[str]:
    root = _regular_directory(root, "manifest root")
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


def _manifest_expectations(root: pathlib.Path) -> Dict[str, tuple[int, str]]:
    """Return expected payload size/digest values when a source manifest exists."""

    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return {}
    manifest = _load_json(manifest_path, "source artifact manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("source artifact manifest files must be an array")
    expectations: Dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("source artifact manifest file entry is invalid")
        relative = _safe_relative(entry.get("path"), "source artifact path").as_posix()
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("source artifact manifest checksum entry is invalid: {}".format(relative))
        if relative in expectations:
            raise ValueError("source artifact manifest contains duplicate path: {}".format(relative))
        expectations[relative] = (size, digest)
    return expectations


def _copy_verified_file(
    source: pathlib.Path,
    target: pathlib.Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Copy one regular file and close the source/destination TOCTOU window."""

    source = _regular_file(source, "copy source file")
    target = pathlib.Path(target)
    if target.exists() or target.is_symlink():
        raise ValueError("copy destination already exists: {}".format(target))
    before_size = source.stat().st_size
    before_digest = _sha256(source)
    if expected_size is not None and before_size != expected_size:
        raise ValueError("source size does not match expected manifest: {}".format(source))
    if expected_sha256 is not None and before_digest != expected_sha256:
        raise ValueError("source checksum does not match expected manifest: {}".format(source))

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied_size = target.stat().st_size
    copied_digest = _sha256(target)
    after_size = source.stat().st_size
    after_digest = _sha256(source)
    if after_size != before_size or after_digest != before_digest:
        raise ValueError("source changed during copy: {}".format(source))
    if copied_size != before_size or copied_digest != before_digest:
        raise ValueError("destination bytes differ after copy: {}".format(target))


def copy_regular_tree(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Copy a complete regular-file tree without following links."""

    source = _regular_directory(source, "copy source")
    destination = pathlib.Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("copy destination already exists: {}".format(destination))
    destination.mkdir(parents=True)
    expectations = _manifest_expectations(source)
    for relative in _manifest_files(source):
        source_path = source / pathlib.PurePosixPath(relative)
        target = destination / pathlib.PurePosixPath(relative)
        expected = expectations.get(relative)
        _copy_verified_file(
            source_path,
            target,
            expected_size=expected[0] if expected else None,
            expected_sha256=expected[1] if expected else None,
        )


def _copy_tree_without_manifest(source: pathlib.Path, destination: pathlib.Path) -> None:
    source = _regular_directory(source, "payload source")
    destination.mkdir(parents=True, exist_ok=False)
    expectations = _manifest_expectations(source)
    for relative in _manifest_files(source):
        source_path = source / pathlib.PurePosixPath(relative)
        target = destination / pathlib.PurePosixPath(relative)
        expected = expectations.get(relative)
        _copy_verified_file(
            source_path,
            target,
            expected_size=expected[0] if expected else None,
            expected_sha256=expected[1] if expected else None,
        )


def _resolve_under(root: pathlib.Path, relative: object, label: str, directory: bool = False) -> pathlib.Path:
    rel = _safe_relative(
        relative.as_posix() if isinstance(relative, pathlib.PurePosixPath) else relative,
        label,
    )
    root = _regular_directory(root, "artifact root")
    candidate = (root / pathlib.Path(rel)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("{} escapes its root: {}".format(label, relative)) from exc
    return _regular_directory(candidate, label) if directory else _regular_file(candidate, label)


def _verify_manifest(module: Any, root: pathlib.Path, path: pathlib.Path, label: str) -> Dict[str, Any]:
    manifest = _load_json(path, label)
    try:
        module.verify_manifest(root, manifest)
    except Exception as exc:
        raise ValueError("{} verification failed: {}".format(label, exc)) from exc
    return dict(manifest)


def _read_marker(root: pathlib.Path, relative: str, schema: str, label: str) -> Dict[str, Any]:
    marker = _load_json(root / relative, label)
    if marker.get("schema_version") != schema or marker.get("verified") is not True:
        raise ValueError("{} is not a verified {} marker".format(label, schema))
    return dict(marker)


def _source_run_from_marker(
    output_root: pathlib.Path, marker: Mapping[str, Any], expected_prefix: str, label: str
) -> pathlib.Path:
    relative = _safe_relative(marker.get("run_path"), "{} run_path".format(label))
    if not relative.as_posix().startswith(expected_prefix.rstrip("/") + "/"):
        raise ValueError("{} run_path is outside the canonical prefix".format(label))
    return _resolve_under(output_root, relative, "{} run".format(label), directory=True)


def _require_exact_evidence(manifest: Mapping[str, Any], expected: str, label: str) -> None:
    observed = manifest.get("execution_evidence")
    if observed != expected:
        raise ValueError(
            "{} requires execution_evidence={}, observed {}".format(label, expected, observed)
        )


def _require_release_distribution_evidence(distribution: Mapping[str, Any]) -> None:
    """Keep the production distribution verifier release-strict.

    Contract tests may patch this function in-process while exercising the
    copying and inventory mechanics with explicitly synthetic fixtures.  The
    command-line verifier never patches it and therefore rejects such bundles.
    """

    observed = distribution.get("execution_evidence")
    if observed != "real_prebaked_qualified":
        raise ValueError(
            "distribution requires execution_evidence=real_prebaked_qualified, observed {}".format(
                observed
            )
        )


def _derive_distribution_evidence(
    shared_evidence: object, model_evidence: Iterable[object]
) -> str:
    """Propagate evidence without upgrading a synthetic source to real.

    The strict production path accepts only the real classes.  This explicit
    propagation is useful for in-process contract tests and prevents metadata
    relabeling if a synthetic fixture is ever used there.
    """

    observed = [shared_evidence, *list(model_evidence)]
    if all(value == "real_single_h800_qualified" for value in observed[1:]) and observed[0] == "real_exact_two_h800_qualified":
        return "real_prebaked_qualified"
    if all(value == "local_synthetic_not_gpu_qualification" for value in observed[1:]) and observed[0] == "local_synthetic_not_two_gpu_qualification":
        return "local_synthetic_not_gpu_qualification"
    return "mixed_evidence_not_release_qualified"


def _derive_model_bundle_evidence(task1_evidence: object, task3_evidence: object) -> str:
    """Derive model-bundle evidence from both compute and simulation producers.

    Task3 evidence is intentionally independent from Task1.  A bundle is real
    only when both source manifests carry the sealed single-H800 class; a
    synthetic pair remains explicitly synthetic, and every mixed/unknown pair
    is terminally non-release-qualified.
    """

    if task1_evidence == TASK1_REAL_EVIDENCE and task3_evidence == TASK3_REAL_EVIDENCE:
        return TASK1_REAL_EVIDENCE
    if (
        task1_evidence == TASK1_SYNTHETIC_EVIDENCE
        and task3_evidence == TASK1_SYNTHETIC_EVIDENCE
    ):
        return TASK1_SYNTHETIC_EVIDENCE
    return "mixed_evidence_not_release_qualified"


def _manifest_digest(path: pathlib.Path) -> str:
    digest = _sha256(path)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError("invalid manifest digest: {}".format(path))
    return digest


def _validate_marker_manifest(marker: Mapping[str, Any], manifest_path: pathlib.Path, label: str) -> None:
    digest = _manifest_digest(manifest_path)
    if (
        marker.get("manifest_sha256") != digest
        or marker.get("artifact_manifest_sha256") != digest
    ):
        raise ValueError("{} manifest checksum does not match its marker".format(label))


def _copy_selected(source: pathlib.Path, destination: pathlib.Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        rel = _safe_relative(relative, "selected payload path")
        source_path = _resolve_under(source, rel, "selected payload")
        target = destination / pathlib.Path(rel)
        if target.exists() or target.is_symlink():
            raise ValueError("duplicate package payload path: {}".format(rel))
        _copy_verified_file(source_path, target)


def _copy_task3_assets(task3_root: pathlib.Path, model_root: pathlib.Path) -> None:
    assets = _resolve_under(task3_root, "slowdown_assets", "Task3 slowdown assets", directory=True)
    target = model_root / "slowdown_assets"
    copy_regular_tree(assets, target)
    for required in ("manifest.json", "kernel_features.json", "backward_kernel_blueprints.json"):
        _regular_file(target / required, "required slowdown asset")


def _copy_task3_metadata(task3_root: pathlib.Path, model_root: pathlib.Path) -> None:
    """Copy portable scheduler/report metadata without source absolute paths."""

    metadata_root = model_root / "simulation"
    metadata_root.mkdir(parents=True, exist_ok=False)
    for relative in ("schedule", "report.json", "report.md"):
        source = task3_root / relative
        if source.is_dir():
            copy_regular_tree(source, metadata_root / relative)
        else:
            _regular_file(source, "Task3 simulation metadata")
            target = metadata_root / relative
            _copy_verified_file(source, target)


def _build_shared_bundle(
    module: Any,
    source_root: pathlib.Path,
    destination: pathlib.Path,
    source_manifest: Mapping[str, Any],
    producer_commits: Mapping[str, str],
    predictor_run_id: str,
) -> Dict[str, Any]:
    _copy_tree_without_manifest(source_root, destination)
    provenance = destination / "provenance/source_task2_manifest.json"
    _stable_json(provenance, dict(source_manifest))
    metadata: Dict[str, Any] = {
        "schema_version": module.SCHEMA_VERSION,
        "model": "shared_task2",
        "task": "task2",
        "artifact_source": "prebaked",
        "predictor_run_id": predictor_run_id,
        "source_commits": dict(producer_commits),
        "execution_evidence": source_manifest["execution_evidence"],
        "source_manifest_sha256": _manifest_digest(source_root / "artifact_manifest.json"),
    }
    manifest = module.create_manifest(destination, metadata, _manifest_files(destination))
    manifest_path = destination / "artifact_manifest.json"
    _stable_json(manifest_path, manifest)
    module.verify_manifest(destination, manifest)
    return {
        "root": "bundles/shared_task2",
        "manifest": "bundles/shared_task2/artifact_manifest.json",
        "manifest_sha256": _manifest_digest(manifest_path),
        "predictor_run_id": predictor_run_id,
        "producer_commits": dict(producer_commits),
        "execution_evidence": source_manifest["execution_evidence"],
    }


def _build_model_bundle(
    module: Any,
    model: str,
    specification: Mapping[str, Any],
    output_root: pathlib.Path,
    destination: pathlib.Path,
    task1_root: pathlib.Path,
    task1_manifest: Mapping[str, Any],
    task3_root: pathlib.Path,
    task3_manifest: Mapping[str, Any],
    producer_commits: Mapping[str, str],
    capture_id: str,
    predictor_run_id: str,
) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    bundle_evidence = _derive_model_bundle_evidence(
        task1_manifest.get("execution_evidence"),
        task3_manifest.get("execution_evidence"),
    )

    task1_payload = [entry["path"] for entry in task1_manifest.get("files", [])]
    _copy_selected(task1_root, destination, task1_payload)
    _copy_task3_assets(task3_root, destination)
    _copy_task3_metadata(task3_root, destination)

    # Keep the producer manifests as explicit provenance payloads, but never
    # copy either source artifact_manifest.json into the package root.
    _stable_json(destination / "provenance/source_task1_manifest.json", dict(task1_manifest))
    _stable_json(destination / "provenance/source_task3_manifest.json", dict(task3_manifest))
    configuration = {
        "schema_version": "sc26-ae-prebaked-configuration-v1",
        "model": model,
        "profile": specification["profile"],
        "simulation_topology": specification["topology"],
        "capture_id": capture_id,
        "predictor_run_id": predictor_run_id,
        "source_task1_manifest_sha256": _manifest_digest(task1_root / "artifact_manifest.json"),
        "source_task3_manifest_sha256": _manifest_digest(task3_root / "artifact_manifest.json"),
        "producer_commits": dict(producer_commits),
        "execution_evidence": bundle_evidence,
        "artifact_source": "prebaked",
        "paths_are_bundle_relative": True,
    }
    _stable_json(destination / "configuration.json", configuration)

    metadata: Dict[str, Any] = {
        "schema_version": module.SCHEMA_VERSION,
        "model": model,
        "task": "prebaked",
        "artifact_source": "prebaked",
        "capture_id": capture_id,
        "predictor_run_id": predictor_run_id,
        "simulation_run_id": "prebaked-{}".format(capture_id),
        "source_commits": dict(producer_commits),
        "simulation_topology": dict(specification["topology"]),
        "profile": specification["profile"],
        "precision": "bf16",
        "ddp_overlap": True,
        "communication_backend": "analytical",
        "overlap_mode": "on",
        "database_is_trace_dir": True,
        "execution_evidence": bundle_evidence,
        "source_task1_manifest_sha256": _manifest_digest(task1_root / "artifact_manifest.json"),
        "source_task3_manifest_sha256": _manifest_digest(task3_root / "artifact_manifest.json"),
    }
    manifest = module.create_manifest(destination, metadata, _manifest_files(destination))
    manifest_path = destination / "artifact_manifest.json"
    _stable_json(manifest_path, manifest)
    module.verify_manifest(destination, manifest)
    return {
        "root": "bundles/{}".format(model),
        "manifest": "bundles/{}/artifact_manifest.json".format(model),
        "manifest_sha256": _manifest_digest(manifest_path),
        "model": model,
        "profile": specification["profile"],
        "simulation_topology": dict(specification["topology"]),
        "capture_id": capture_id,
        "predictor_run_id": predictor_run_id,
        "producer_commits": dict(producer_commits),
        "source_task1_manifest_sha256": _manifest_digest(task1_root / "artifact_manifest.json"),
        "source_task3_manifest_sha256": _manifest_digest(task3_root / "artifact_manifest.json"),
        "execution_evidence": bundle_evidence,
    }


def _inventory(root: pathlib.Path) -> list[str]:
    root = _regular_directory(root, "distribution root")
    paths: list[str] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = pathlib.Path(current_root)
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError("distribution contains a non-directory or symlink: {}".format(path))
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError("distribution contains a non-regular file or symlink: {}".format(path))
            paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def _distribution_files(root: pathlib.Path) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    for relative in _inventory(root):
        if relative == "distribution_manifest.json":
            continue
        path = root / pathlib.PurePosixPath(relative)
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _distribution_total_size(root: pathlib.Path) -> int:
    """Return the byte size of every regular file in a staged distribution."""

    return sum(
        (root / pathlib.PurePosixPath(relative)).stat().st_size
        for relative in _inventory(root)
    )


def _finalize_distribution_manifest(
    module: Any,
    staging_root: pathlib.Path,
    manifest_path: pathlib.Path,
    distribution: MutableMapping[str, Any],
) -> tuple[int, str]:
    """Write summary fields until they describe the bytes just written.

    The manifest is part of the staged tree, so changing its summary fields can
    change its own size.  A bounded fixed-point loop makes the self-referential
    size explicit and fails closed if serialization does not converge.
    """

    max_iterations = 16
    for _ in range(max_iterations):
        _stable_json(manifest_path, distribution)
        actual_total = _distribution_total_size(staging_root)
        actual_medium = module.evaluate_distribution_gate(staging_root)
        if (
            distribution.get("total_size_bytes") == actual_total
            and distribution.get("distribution_medium") == actual_medium
        ):
            return actual_total, actual_medium
        distribution["total_size_bytes"] = actual_total
        distribution["distribution_medium"] = actual_medium
    raise ValueError(
        "distribution manifest summary did not converge after {} iterations".format(
            max_iterations
        )
    )


def _validate_distribution_files(root: pathlib.Path, distribution: Mapping[str, Any]) -> None:
    entries = distribution.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("distribution manifest files must be a non-empty array")
    paths = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("distribution file entry schema is invalid")
        relative = _safe_relative(entry["path"], "distribution file path").as_posix()
        if relative == "distribution_manifest.json" or relative in paths:
            raise ValueError("distribution file path is duplicated or self-referential")
        if not isinstance(entry["size_bytes"], int) or isinstance(entry["size_bytes"], bool) or entry["size_bytes"] < 0:
            raise ValueError("distribution file size is invalid: {}".format(relative))
        if not isinstance(entry["sha256"], str) or SHA256_PATTERN.fullmatch(entry["sha256"]) is None:
            raise ValueError("distribution file SHA256 is invalid: {}".format(relative))
        path = _resolve_under(root, relative, "distribution file")
        if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError("distribution file bytes/checksum mismatch: {}".format(relative))
        paths.append(relative)
    if paths != sorted(paths):
        raise ValueError("distribution files must be sorted by path")
    inventory = set(_inventory(root)) - {"distribution_manifest.json"}
    if set(paths) != inventory:
        raise ValueError("distribution file inventory does not match the staged tree")
    size_mib = distribution.get("file_size_mib")
    if not isinstance(size_mib, Mapping) or set(size_mib) != set(paths):
        raise ValueError("distribution file_size_mib inventory does not match files")
    for relative in paths:
        value = size_mib[relative]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError("distribution file_size_mib value is invalid: {}".format(relative))


def _validate_bundle_entry(
    module: Any,
    root: pathlib.Path,
    key: str,
    entry: Mapping[str, Any],
    expected_commits: Mapping[str, str],
) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("distribution bundle entry must be an object: {}".format(key))
    for field in ("root", "manifest", "manifest_sha256"):
        if field not in entry:
            raise ValueError("distribution bundle entry is missing {}: {}".format(field, key))
    bundle_root_rel = _safe_relative(entry["root"], "{} bundle root".format(key))
    manifest_rel = _safe_relative(entry["manifest"], "{} manifest".format(key))
    if manifest_rel != bundle_root_rel / "artifact_manifest.json":
        raise ValueError("{} nested manifest path is not canonical".format(key))
    bundle_root = _resolve_under(root, bundle_root_rel, "{} bundle root".format(key), directory=True)
    manifest_path = _resolve_under(root, manifest_rel, "{} manifest".format(key))
    if entry["manifest_sha256"] != _manifest_digest(manifest_path):
        raise ValueError("{} nested manifest checksum mismatch".format(key))
    manifest = _verify_manifest(module, bundle_root, manifest_path, "{} manifest".format(key))
    if key == "shared_task2":
        if manifest.get("model") != "shared_task2" or manifest.get("task") != "task2":
            raise ValueError("shared Task2 manifest identity is invalid")
        if manifest.get("artifact_source") != "prebaked":
            raise ValueError("shared Task2 manifest must be prebaked")
        _require_exact_evidence(manifest, "real_exact_two_h800_qualified", "shared Task2 manifest")
    else:
        specification = MODELS[key]
        if manifest.get("model") != key or manifest.get("task") != "prebaked":
            raise ValueError("{} model manifest identity is invalid".format(key))
        if manifest.get("artifact_source") != "prebaked":
            raise ValueError("{} model manifest must be prebaked".format(key))
        if manifest.get("simulation_topology") != specification["topology"]:
            raise ValueError("{} model topology is invalid".format(key))
        if manifest.get("profile") != specification["profile"]:
            raise ValueError("{} model profile is invalid".format(key))
        _require_exact_evidence(manifest, "real_single_h800_qualified", "{} model manifest".format(key))
        if manifest.get("execution_evidence") == TASK1_REAL_EVIDENCE:
            source_task1_manifest_path = bundle_root / "provenance/source_task1_manifest.json"
            source_task1_manifest = _load_json(
                source_task1_manifest_path,
                "{} source Task1 manifest".format(key),
            )
            module.validate_task1_rank_promotion_scope(source_task1_manifest)
        files = {entry["path"] for entry in manifest.get("files", [])}
        traces = [path for path in files if path.endswith(".txt") and (path.startswith("trace/") or path.startswith("runtime/profiler_log/"))]
        if not traces:
            raise ValueError("{} model manifest contains no trace files".format(key))
        sqlite = [path for path in files if path.endswith(".sqlite")]
        if len(sqlite) != 1:
            raise ValueError("{} model manifest must contain exactly one SQLite artifact".format(key))
        required_assets = {
            "slowdown_assets/manifest.json",
            "slowdown_assets/kernel_features.json",
            "slowdown_assets/backward_kernel_blueprints.json",
        }
        if not required_assets.issubset(files):
            raise ValueError("{} model slowdown assets are incomplete".format(key))
    if manifest.get("source_commits") != dict(expected_commits):
        raise ValueError("{} producer commits differ from the declared distribution".format(key))
    return manifest


def verify_distribution(repo_root: pathlib.Path, prebaked_root: pathlib.Path) -> Dict[str, Any]:
    repo_root = _regular_directory(pathlib.Path(repo_root), "repository root")
    prebaked_root = _regular_directory(pathlib.Path(prebaked_root), "prebaked root")
    module = _load_artifact_module(repo_root)
    distribution_path = prebaked_root / "distribution_manifest.json"
    distribution = _load_json(distribution_path, "distribution manifest")
    if distribution.get("schema_version") != "sc26-ae-distribution-manifest-v1":
        raise ValueError("distribution manifest schema is invalid")
    if distribution.get("artifact_source") != "prebaked":
        raise ValueError("distribution artifact_source must be prebaked")
    _require_release_distribution_evidence(distribution)
    distribution_id = _safe_id(distribution.get("distribution_id"), "distribution_id")
    expected_commits = _source_commits(repo_root)
    compatible = distribution.get("compatible_commits")
    if compatible != {
        "echo_slowdown": expected_commits["echo_slowdown"],
        "megatron_sim_engine": expected_commits["megatron_sim_engine"],
    }:
        raise ValueError("distribution compatible commits differ from current gitlinks")
    _validate_distribution_files(prebaked_root, distribution)
    declared_total = distribution.get("total_size_bytes")
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total < 0
        or declared_total != _distribution_total_size(prebaked_root)
    ):
        raise ValueError("distribution total_size_bytes does not match staged bytes")
    actual_medium = module.evaluate_distribution_gate(prebaked_root)
    if distribution.get("distribution_medium") != actual_medium:
        raise ValueError("distribution_medium does not match staged bytes")
    bundles = distribution.get("bundles")
    if not isinstance(bundles, Mapping) or set(bundles) != ALL_BUNDLES:
        raise ValueError("distribution bundles must contain all three models and shared_task2")
    manifests: Dict[str, Dict[str, Any]] = {}
    for key in sorted(ALL_BUNDLES):
        manifests[key] = _validate_bundle_entry(module, prebaked_root, key, bundles[key], distribution["producer_commits"][key] if key in MODELS else distribution["producer_commits"]["shared_task2"])
    predictor_ids = {manifests[key].get("predictor_run_id") for key in MODELS}
    if len(predictor_ids) != 1 or None in predictor_ids:
        raise ValueError("model bundles do not share one predictor_run_id")
    predictor_run_id = next(iter(predictor_ids))
    if manifests["shared_task2"].get("predictor_run_id") != predictor_run_id:
        raise ValueError("shared predictor_run_id differs from model bundles")
    captures = {manifests[key].get("capture_id") for key in MODELS}
    if len(captures) != len(MODELS) or None in captures:
        raise ValueError("model capture_ids must be distinct and present")
    for key in MODELS:
        if manifests[key].get("capture_id") == predictor_run_id:
            raise ValueError("{} capture_id must differ from predictor_run_id".format(key))
    return {
        "distribution_id": distribution_id,
        "predictor_run_id": predictor_run_id,
        "bundle_count": len(bundles),
        "distribution_file_count": len(distribution["files"]),
        "total_size_bytes": int(distribution["total_size_bytes"]),
        "distribution_medium": distribution["distribution_medium"],
    }


def _source_manifest_and_run(
    module: Any,
    output_root: pathlib.Path,
    model: str,
    expected_commits: Mapping[str, str],
) -> tuple[pathlib.Path, Dict[str, Any], pathlib.Path, Dict[str, Any], pathlib.Path, Dict[str, Any]]:
    task1_dir = _resolve_under(output_root, "{}/task1".format(model), "{} Task1 directory".format(model), directory=True)
    capture_marker = _read_marker(task1_dir, "capture_marker.json", "sc26-ae-task1-capture-marker-v1", "{} Task1 capture".format(model))
    if capture_marker.get("model") != model:
        raise ValueError("{} Task1 capture marker model mismatch".format(model))
    task1_root = _source_run_from_marker(task1_dir, capture_marker, "runs", "{} Task1".format(model))
    task1_manifest_path = task1_root / "artifact_manifest.json"
    task1_manifest = _verify_manifest(module, task1_root, task1_manifest_path, "{} Task1 manifest".format(model))
    _validate_marker_manifest(capture_marker, task1_manifest_path, "{} Task1 capture".format(model))
    _require_exact_evidence(task1_manifest, "real_single_h800_qualified", "{} Task1 manifest".format(model))
    if task1_manifest.get("model") != model or task1_manifest.get("task") != "task1" or task1_manifest.get("artifact_source") != "fresh":
        raise ValueError("{} Task1 manifest identity is invalid".format(model))
    module.validate_task1_rank_promotion_scope(task1_manifest)
    if task1_manifest.get("source_commits") != dict(expected_commits):
        raise ValueError("{} Task1 source commits differ from current checkout".format(model))
    if task1_manifest.get("capture_id") != capture_marker.get("capture_id"):
        raise ValueError("{} Task1 capture_id differs from marker".format(model))

    task3_dir = _resolve_under(output_root, "{}/task3".format(model), "{} Task3 directory".format(model), directory=True)
    run_marker = _read_marker(task3_dir, "run_marker.json", "sc26-ae-task3-run-marker-v1", "{} Task3 run".format(model))
    if run_marker.get("model") != model or run_marker.get("artifact_source") != "fresh":
        raise ValueError("{} Task3 run marker identity is invalid".format(model))
    simulation_run_id = _safe_id(
        run_marker.get("simulation_run_id"), "{} Task3 simulation_run_id".format(model)
    )
    if run_marker.get("run_path") != "runs/{}".format(simulation_run_id):
        raise ValueError(
            "{} Task3 run_path does not match simulation_run_id".format(model)
        )
    task3_root = _source_run_from_marker(task3_dir, run_marker, "runs", "{} Task3".format(model))
    task3_manifest_path = task3_root / "artifact_manifest.json"
    task3_manifest = _verify_manifest(module, task3_root, task3_manifest_path, "{} Task3 manifest".format(model))
    _validate_marker_manifest(run_marker, task3_manifest_path, "{} Task3 run".format(model))
    if run_marker.get("execution_evidence") != task3_manifest.get("execution_evidence"):
        raise ValueError("{} Task3 marker evidence differs from its manifest".format(model))
    if task3_manifest.get("model") != model or task3_manifest.get("task") != "task3" or task3_manifest.get("artifact_source") != "fresh":
        raise ValueError("{} Task3 manifest identity is invalid".format(model))
    _require_exact_evidence(task3_manifest, TASK3_REAL_EVIDENCE, "{} Task3 manifest".format(model))
    if task3_manifest.get("source_commits") != dict(expected_commits):
        raise ValueError("{} Task3 source commits differ from current checkout".format(model))
    if task3_manifest.get("simulation_run_id") != simulation_run_id:
        raise ValueError("{} Task3 simulation_run_id differs from marker".format(model))
    if task3_manifest.get("capture_id") != run_marker.get("capture_id"):
        raise ValueError("{} Task3 capture_id differs from marker".format(model))
    if task3_manifest.get("capture_id") != task1_manifest.get("capture_id"):
        raise ValueError("{} Task3 capture_id differs from Task1".format(model))
    if task3_manifest.get("predictor_run_id") != run_marker.get("predictor_run_id"):
        raise ValueError("{} Task3 predictor_run_id differs from marker".format(model))
    return task1_root, task1_manifest, task3_root, task3_manifest, task1_manifest_path, task3_manifest_path


def build_distribution(
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
    staging_root: pathlib.Path,
    distribution_id: str,
    result_json: pathlib.Path,
) -> Dict[str, Any]:
    repo_root = _regular_directory(pathlib.Path(repo_root), "repository root")
    output_root = _regular_directory(pathlib.Path(output_root), "fresh output root")
    staging_root = pathlib.Path(staging_root)
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError("staging destination already exists: {}".format(staging_root))
    _safe_id(distribution_id, "distribution_id")
    module = _load_artifact_module(repo_root)
    expected_commits = _source_commits(repo_root)

    pointer_path = output_root / "_shared/task2/predictor_marker.json"
    pointer = _read_marker(output_root / "_shared/task2", "predictor_marker.json", "sc26-ae-task2-shared-pointer-v1", "shared Task2 pointer")
    predictor_run_id = _safe_id(pointer.get("predictor_run_id"), "predictor_run_id")
    shared_root = _source_run_from_marker(output_root, pointer, "_shared/task2/runs", "shared Task2")
    shared_manifest_path = shared_root / "artifact_manifest.json"
    shared_manifest = _verify_manifest(module, shared_root, shared_manifest_path, "shared Task2 manifest")
    _validate_marker_manifest(pointer, shared_manifest_path, "shared Task2 pointer")
    _require_exact_evidence(shared_manifest, "real_exact_two_h800_qualified", "shared Task2 manifest")
    if shared_manifest.get("model") != "shared_task2" or shared_manifest.get("task") != "task2" or shared_manifest.get("artifact_source") != "fresh":
        raise ValueError("shared Task2 manifest identity is invalid")
    if shared_manifest.get("predictor_run_id") != predictor_run_id:
        raise ValueError("shared Task2 predictor_run_id differs from pointer")
    if shared_manifest.get("source_commits") != expected_commits:
        raise ValueError("shared Task2 source commits differ from current checkout")

    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=False)
    bundles_root = staging_root / "bundles"
    bundles_root.mkdir()
    bundle_entries: Dict[str, Dict[str, Any]] = {}
    bundle_entries["shared_task2"] = _build_shared_bundle(
        module,
        shared_root,
        bundles_root / "shared_task2",
        shared_manifest,
        expected_commits,
        predictor_run_id,
    )

    for model, specification in MODELS.items():
        task1_root, task1_manifest, task3_root, task3_manifest, _, _ = _source_manifest_and_run(
            module, output_root, model, expected_commits
        )
        if task3_manifest.get("predictor_run_id") != predictor_run_id:
            raise ValueError("{} Task3 predictor_run_id differs from shared Task2".format(model))
        bundle_entries[model] = _build_model_bundle(
            module,
            model,
            specification,
            output_root,
            bundles_root / model,
            task1_root,
            task1_manifest,
            task3_root,
            task3_manifest,
            expected_commits,
            _safe_id(task1_manifest["capture_id"], "capture_id"),
            predictor_run_id,
        )

    entries = _distribution_files(staging_root)
    file_size_mib = {entry["path"]: round(entry["size_bytes"] / 1048576.0, 6) for entry in entries}
    distribution: Dict[str, Any] = {
        "schema_version": "sc26-ae-distribution-manifest-v1",
        "artifact_source": "prebaked",
        "distribution_id": distribution_id,
        "compatible_commits": {
            "echo_slowdown": expected_commits["echo_slowdown"],
            "megatron_sim_engine": expected_commits["megatron_sim_engine"],
        },
        "producer_commits": {
            **{model: dict(expected_commits) for model in MODELS},
            "shared_task2": dict(expected_commits),
        },
        "bundles": bundle_entries,
        "files": entries,
        "file_size_mib": file_size_mib,
        "execution_evidence": _derive_distribution_evidence(
            shared_manifest["execution_evidence"],
            [bundle_entries[model]["execution_evidence"] for model in MODELS],
        ),
    }
    manifest_path = staging_root / "distribution_manifest.json"
    _finalize_distribution_manifest(module, staging_root, manifest_path, distribution)
    verified = verify_distribution(repo_root, staging_root)
    result = {
        **verified,
        "distribution_manifest": {
            "path": "distribution_manifest.json",
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _manifest_digest(manifest_path),
        },
        "file_size_mib": file_size_mib,
    }
    result_path = pathlib.Path(result_json)
    try:
        result_path.resolve().relative_to(staging_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("result_json must be outside staging_root")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _stable_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repo-root", type=pathlib.Path, required=True)
    build.add_argument("--output-root", type=pathlib.Path, required=True)
    build.add_argument("--staging-root", type=pathlib.Path, required=True)
    build.add_argument("--distribution-id", required=True)
    build.add_argument("--result-json", type=pathlib.Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repo-root", type=pathlib.Path, required=True)
    verify.add_argument("--prebaked-root", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_distribution(
            args.repo_root,
            args.output_root,
            args.staging_root,
            args.distribution_id,
            args.result_json,
        )
        print("DISTRIBUTION_STATUS=verified")
        print("DISTRIBUTION_MEDIUM={}".format(result["distribution_medium"]))
        print("DISTRIBUTION_BUNDLE_COUNT={}".format(result["bundle_count"]))
        print("DISTRIBUTION_FILE_COUNT={}".format(result["distribution_file_count"]))
        print("DISTRIBUTION_TOTAL_SIZE_BYTES={}".format(result["total_size_bytes"]))
        return 0
    if args.command == "verify":
        result = verify_distribution(args.repo_root, args.prebaked_root)
        print("DISTRIBUTION_STATUS=verified")
        print("DISTRIBUTION_ID={}".format(result["distribution_id"]))
        print("DISTRIBUTION_BUNDLE_COUNT={}".format(result["bundle_count"]))
        print("DISTRIBUTION_FILE_COUNT={}".format(result["distribution_file_count"]))
        print("DISTRIBUTION_TOTAL_SIZE_BYTES={}".format(result["total_size_bytes"]))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
