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
            "pp": 8,
            "tp": 8,
            "dp": 4,
            "exp": 4,
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
SOURCE_COMMIT_KEYS = {"megatron_lm", "echo_slowdown", "megatron_sim_engine"}

TASK1_REAL_EVIDENCE = "real_single_h800_qualified"
TASK2_REAL_EVIDENCE = "real_exact_two_h800_qualified"
TASK3_REAL_EVIDENCE = "real_single_h800_qualified"
TASK1_SYNTHETIC_EVIDENCE = "local_synthetic_not_gpu_qualification"
TASK2_SYNTHETIC_EVIDENCE = "local_synthetic_not_two_gpu_qualification"

# Functional fake-level bundles are deliberately separate from the release
# sealer.  They carry the real producer outputs and checksums, but their
# runtime evidence is still pending external qualification.  The explicit
# class prevents a fake/local workflow from being relabeled as real.
FUNCTIONAL_DISTRIBUTION_SCHEMA = "sc26-ae-functional-distribution-manifest-v1"
FUNCTIONAL_DISTRIBUTION_EVIDENCE = "functional_prebaked_not_release_qualified"
TASK1_PENDING_EVIDENCE = "runtime_measurement_requires_external_single_gpu_qualification"
TASK2_PENDING_EVIDENCE = "runtime_measurement_requires_external_two_gpu_qualification"
FUNCTIONAL_MODELS = {"gpt175b", "qwen3_a30b"}
FUNCTIONAL_BUNDLES = FUNCTIONAL_MODELS | {"shared_task2"}


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


def _validated_source_commits(value: object, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != SOURCE_COMMIT_KEYS:
        raise ValueError("{} source_commits schema is invalid".format(label))
    commits = dict(value)
    for key, commit in commits.items():
        if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError("{} source_commits.{} is invalid".format(label, key))
    return commits


def _validate_functional_source_compatibility(value: object, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"policy", "task1", "task2"}:
        raise ValueError("{} source compatibility schema is invalid".format(label))
    compatibility = dict(value)
    policy = compatibility["policy"]
    if policy == "exact_or_simulator_only_ancestor_v1":
        allowed = {"exact", "simulator_only_reuse"}
        if compatibility["task1"] not in allowed or compatibility["task2"] not in allowed:
            raise ValueError("{} source compatibility modes are invalid for {}".format(label, policy))
    elif policy == "task_specific_source_compatibility_v2":
        if compatibility["task1"] not in {
            "exact",
            "simulator_only_reuse",
            "task1_consumer_only_reuse",
        } or compatibility["task2"] not in {
            "exact",
            "task2_producer_equivalent_reuse",
        }:
            raise ValueError("{} source compatibility modes are invalid for {}".format(label, policy))
    else:
        raise ValueError("{} source compatibility policy is unsupported: {}".format(label, policy))
    return compatibility


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


def _manifest_entry_expectations(
    manifest: Mapping[str, Any], label: str
) -> Dict[str, tuple[int, str]]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("{} files must be an array".format(label))
    expectations: Dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("{} file entry is invalid".format(label))
        relative = _safe_relative(entry.get("path"), "{} path".format(label)).as_posix()
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("{} checksum entry is invalid: {}".format(label, relative))
        if relative in expectations:
            raise ValueError("{} contains duplicate path: {}".format(label, relative))
        expectations[relative] = (size, digest)
    return expectations


def _manifest_expectations(root: pathlib.Path) -> Dict[str, tuple[int, str]]:
    """Return expected payload size/digest values when a source manifest exists."""

    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return {}
    manifest = _load_json(manifest_path, "source artifact manifest")
    return _manifest_entry_expectations(manifest, "source artifact manifest")


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


def _copy_manifest_file(
    source: pathlib.Path,
    target: pathlib.Path,
    relative: str,
    expectations: Mapping[str, tuple[int, str]],
    label: str,
) -> None:
    rel = _safe_relative(relative, "{} path".format(label))
    relative_text = rel.as_posix()
    expected = expectations.get(relative_text)
    if expected is None:
        raise ValueError(
            "{} lacks a verified manifest expectation: {}".format(label, relative_text)
        )
    source_path = _resolve_under(source, rel, label)
    if target.exists() or target.is_symlink():
        raise ValueError("duplicate package payload path: {}".format(target))
    _copy_verified_file(
        source_path,
        target,
        expected_size=expected[0],
        expected_sha256=expected[1],
    )


def _copy_tree_without_manifest(
    source: pathlib.Path,
    destination: pathlib.Path,
    expectations: Mapping[str, tuple[int, str]],
) -> None:
    source = _regular_directory(source, "payload source")
    destination.mkdir(parents=True, exist_ok=False)
    inventory = _manifest_files(source)
    if set(inventory) != set(expectations):
        raise ValueError("payload inventory differs from the verified source manifest")
    for relative in inventory:
        _copy_manifest_file(
            source,
            destination / pathlib.PurePosixPath(relative),
            relative,
            expectations,
            "payload source",
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


def _validate_marker_manifest(
    marker: Mapping[str, Any], manifest_path: pathlib.Path, label: str
) -> str:
    digest = _manifest_digest(manifest_path)
    if (
        marker.get("manifest_sha256") != digest
        or marker.get("artifact_manifest_sha256") != digest
    ):
        raise ValueError("{} manifest checksum does not match its marker".format(label))
    return digest


def _validate_functional_task3_provenance(
    model: str,
    task1_manifest: Mapping[str, Any],
    task1_manifest_path: pathlib.Path,
    task2_manifest: Mapping[str, Any],
    task2_manifest_path: pathlib.Path,
    task3_manifest: Mapping[str, Any],
    resolved_inputs_path: pathlib.Path,
) -> Dict[str, Any]:
    """Validate the exact Task1/Task2 inputs sealed by one Fresh Task3 run."""

    label = "{} Task3".format(model)
    expectations = _manifest_entry_expectations(task3_manifest, "{} manifest".format(label))
    source_paths = {
        "provenance/task1_manifest.json": task1_manifest_path,
        "provenance/task2_manifest.json": task2_manifest_path,
        "provenance/resolved_inputs.json": resolved_inputs_path,
    }
    for relative, actual_path in source_paths.items():
        expected = expectations.get(relative)
        actual = (_regular_file(actual_path, "{} provenance".format(label)).stat().st_size, _sha256(actual_path))
        if expected != actual:
            raise ValueError("{} provenance differs from its manifest: {}".format(label, relative))

    if _load_json(task1_manifest_path, "{} embedded Task1 manifest".format(label)) != dict(task1_manifest):
        raise ValueError("{} embedded Task1 manifest differs from the selected Task1".format(label))
    if _load_json(task2_manifest_path, "{} embedded Task2 manifest".format(label)) != dict(task2_manifest):
        raise ValueError("{} embedded Task2 manifest differs from the selected Task2".format(label))

    resolved = _load_json(resolved_inputs_path, "{} resolved inputs".format(label))
    if resolved.get("schema_version") != "sc26-ae-task3-resolved-inputs-v1":
        raise ValueError("{} resolved input schema is invalid".format(label))
    if resolved.get("artifact_source") != "fresh":
        raise ValueError("{} resolved input source is invalid".format(label))
    if resolved.get("capture_id") != task1_manifest.get("capture_id"):
        raise ValueError("{} resolved capture_id differs from Task1".format(label))
    if resolved.get("predictor_run_id") != task2_manifest.get("predictor_run_id"):
        raise ValueError("{} resolved predictor_run_id differs from Task2".format(label))

    task1_commits = _validated_source_commits(
        task1_manifest.get("source_commits"), "{} Task1".format(model)
    )
    task2_commits = _validated_source_commits(
        task2_manifest.get("source_commits"), "{} Task2".format(model)
    )
    if resolved.get("task1_source_commits") != task1_commits:
        raise ValueError("{} resolved Task1 commits differ from Task1 manifest".format(label))
    if resolved.get("task2_source_commits") != task2_commits:
        raise ValueError("{} resolved Task2 commits differ from Task2 manifest".format(label))
    _validate_functional_source_compatibility(
        resolved.get("source_compatibility"), label
    )

    input_expectations = resolved.get("input_expectations")
    if not isinstance(input_expectations, Mapping):
        raise ValueError("{} input expectations are missing".format(label))
    for key, manifest_path in (
        ("task1_manifest", task1_manifest_path),
        ("task2_manifest", task2_manifest_path),
    ):
        entry = input_expectations.get(key)
        if not isinstance(entry, Mapping) or (
            entry.get("size_bytes") != manifest_path.stat().st_size
            or entry.get("sha256") != _sha256(manifest_path)
        ):
            raise ValueError("{} {} expectation differs from the selected manifest".format(label, key))
    return dict(resolved)


def _copy_selected(
    source: pathlib.Path,
    destination: pathlib.Path,
    relative_paths: Iterable[str],
    expectations: Mapping[str, tuple[int, str]],
) -> None:
    for relative in relative_paths:
        rel = _safe_relative(relative, "selected payload path")
        target = destination / pathlib.Path(rel)
        _copy_manifest_file(
            source,
            target,
            rel.as_posix(),
            expectations,
            "selected payload",
        )


def _copy_task3_assets(
    task3_root: pathlib.Path,
    model_root: pathlib.Path,
    expectations: Mapping[str, tuple[int, str]],
) -> None:
    asset_paths = sorted(
        relative
        for relative in expectations
        if pathlib.PurePosixPath(relative).parts[0] == "slowdown_assets"
    )
    if not asset_paths:
        raise ValueError("Task3 slowdown asset manifest inventory is empty")
    _copy_selected(task3_root, model_root, asset_paths, expectations)
    for required in ("manifest.json", "kernel_features.json", "backward_kernel_blueprints.json"):
        _regular_file(model_root / "slowdown_assets" / required, "required slowdown asset")


def _copy_task3_metadata(
    task3_root: pathlib.Path,
    model_root: pathlib.Path,
    expectations: Mapping[str, tuple[int, str]],
) -> None:
    """Copy portable scheduler/report metadata without source absolute paths."""

    metadata_root = model_root / "simulation"
    metadata_root.mkdir(parents=True, exist_ok=False)
    metadata_paths = sorted(
        relative
        for relative in expectations
        if relative in {"report.json", "report.md"}
        or pathlib.PurePosixPath(relative).parts[0] == "schedule"
    )
    if not {"report.json", "report.md"}.issubset(metadata_paths) or not any(
        pathlib.PurePosixPath(relative).parts[0] == "schedule"
        for relative in metadata_paths
    ):
        raise ValueError("Task3 simulation metadata manifest inventory is incomplete")
    for relative in metadata_paths:
        _copy_manifest_file(
            task3_root,
            metadata_root / pathlib.PurePosixPath(relative),
            relative,
            expectations,
            "Task3 simulation metadata",
        )


def _build_shared_bundle(
    module: Any,
    source_root: pathlib.Path,
    destination: pathlib.Path,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    producer_commits: Mapping[str, str],
    predictor_run_id: str,
    evidence_override: str | None = None,
    source_artifact_commits: Mapping[str, Mapping[str, str]] | None = None,
) -> Dict[str, Any]:
    source_expectations = _manifest_entry_expectations(
        source_manifest, "verified shared Task2 manifest"
    )
    _copy_tree_without_manifest(source_root, destination, source_expectations)
    provenance = destination / "provenance/source_task2_manifest.json"
    _stable_json(provenance, dict(source_manifest))
    metadata: Dict[str, Any] = {
        "schema_version": module.SCHEMA_VERSION,
        "model": "shared_task2",
        "task": "task2",
        "artifact_source": "prebaked",
        "predictor_run_id": predictor_run_id,
        "source_commits": dict(producer_commits),
        "execution_evidence": evidence_override or source_manifest["execution_evidence"],
        "source_manifest_sha256": source_manifest_sha256,
    }
    if source_artifact_commits is not None:
        metadata["source_artifact_commits"] = {
            key: dict(value) for key, value in source_artifact_commits.items()
        }
    manifest = module.create_manifest(destination, metadata, _manifest_files(destination))
    manifest_path = destination / "artifact_manifest.json"
    _stable_json(manifest_path, manifest)
    module.verify_manifest(destination, manifest)
    entry = {
        "root": "bundles/shared_task2",
        "manifest": "bundles/shared_task2/artifact_manifest.json",
        "manifest_sha256": _manifest_digest(manifest_path),
        "predictor_run_id": predictor_run_id,
        "producer_commits": dict(producer_commits),
        "execution_evidence": evidence_override or source_manifest["execution_evidence"],
    }
    if source_artifact_commits is not None:
        entry["source_artifact_commits"] = {
            key: dict(value) for key, value in source_artifact_commits.items()
        }
    return entry


def _build_model_bundle(
    module: Any,
    model: str,
    specification: Mapping[str, Any],
    output_root: pathlib.Path,
    destination: pathlib.Path,
    task1_root: pathlib.Path,
    task1_manifest: Mapping[str, Any],
    task1_manifest_sha256: str,
    task3_root: pathlib.Path,
    task3_manifest: Mapping[str, Any],
    task3_manifest_sha256: str,
    producer_commits: Mapping[str, str],
    capture_id: str,
    predictor_run_id: str,
    evidence_override: str | None = None,
    source_artifact_commits: Mapping[str, Mapping[str, str]] | None = None,
    task3_resolved_inputs_path: pathlib.Path | None = None,
) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    bundle_evidence = evidence_override or _derive_model_bundle_evidence(
        task1_manifest.get("execution_evidence"),
        task3_manifest.get("execution_evidence"),
    )

    task1_expectations = _manifest_entry_expectations(
        task1_manifest, "verified Task1 manifest"
    )
    task3_expectations = _manifest_entry_expectations(
        task3_manifest, "verified Task3 manifest"
    )
    task1_payload = list(task1_expectations)
    _copy_selected(task1_root, destination, task1_payload, task1_expectations)
    _copy_task3_assets(task3_root, destination, task3_expectations)
    _copy_task3_metadata(task3_root, destination, task3_expectations)

    # Keep the producer manifests as explicit provenance payloads, but never
    # copy either source artifact_manifest.json into the package root.
    _stable_json(destination / "provenance/source_task1_manifest.json", dict(task1_manifest))
    _stable_json(destination / "provenance/source_task3_manifest.json", dict(task3_manifest))
    if task3_resolved_inputs_path is not None:
        _copy_manifest_file(
            task3_root,
            destination / "provenance/source_task3_resolved_inputs.json",
            "provenance/resolved_inputs.json",
            task3_expectations,
            "Task3 resolved input provenance",
        )
    configuration = {
        "schema_version": "sc26-ae-prebaked-configuration-v1",
        "model": model,
        "profile": specification["profile"],
        "simulation_topology": specification["topology"],
        "capture_id": capture_id,
        "predictor_run_id": predictor_run_id,
        "source_task1_manifest_sha256": task1_manifest_sha256,
        "source_task3_manifest_sha256": task3_manifest_sha256,
        "producer_commits": dict(producer_commits),
        "execution_evidence": bundle_evidence,
        "artifact_source": "prebaked",
        "paths_are_bundle_relative": True,
    }
    if source_artifact_commits is not None:
        configuration["source_artifact_commits"] = {
            key: dict(value) for key, value in source_artifact_commits.items()
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
        "source_task1_manifest_sha256": task1_manifest_sha256,
        "source_task3_manifest_sha256": task3_manifest_sha256,
    }
    if source_artifact_commits is not None:
        metadata["source_artifact_commits"] = {
            key: dict(value) for key, value in source_artifact_commits.items()
        }
    manifest = module.create_manifest(destination, metadata, _manifest_files(destination))
    manifest_path = destination / "artifact_manifest.json"
    _stable_json(manifest_path, manifest)
    module.verify_manifest(destination, manifest)
    entry = {
        "root": "bundles/{}".format(model),
        "manifest": "bundles/{}/artifact_manifest.json".format(model),
        "manifest_sha256": _manifest_digest(manifest_path),
        "model": model,
        "profile": specification["profile"],
        "simulation_topology": dict(specification["topology"]),
        "capture_id": capture_id,
        "predictor_run_id": predictor_run_id,
        "producer_commits": dict(producer_commits),
        "source_task1_manifest_sha256": task1_manifest_sha256,
        "source_task3_manifest_sha256": task3_manifest_sha256,
        "execution_evidence": bundle_evidence,
    }
    if source_artifact_commits is not None:
        entry["source_artifact_commits"] = {
            key: dict(value) for key, value in source_artifact_commits.items()
        }
    return entry


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


def _functional_source_evidence(value: object, *, task2: bool = False) -> bool:
    allowed = {
        TASK1_REAL_EVIDENCE,
        TASK1_PENDING_EVIDENCE,
        TASK1_SYNTHETIC_EVIDENCE,
    }
    if task2:
        allowed = {
            TASK2_REAL_EVIDENCE,
            TASK2_PENDING_EVIDENCE,
            TASK2_SYNTHETIC_EVIDENCE,
        }
    return value in allowed


def _validate_functional_task1_source(
    model: str, task1_manifest: Mapping[str, Any]
) -> None:
    """Validate the model-local Task1 evidence required by functional bundles."""

    if not _functional_source_evidence(task1_manifest.get("execution_evidence")):
        raise ValueError("{} functional source Task1 evidence is invalid".format(model))
    files = {entry["path"] for entry in task1_manifest.get("files", [])}
    if "ncu/kernel_metric_output.csv" not in files:
        raise ValueError(
            "{} functional source Task1 is missing ncu/kernel_metric_output.csv".format(
                model
            )
        )

    ncu_provenance = task1_manifest.get("ncu_feature_provenance")
    if not isinstance(ncu_provenance, Mapping):
        raise ValueError("{} functional source Task1 lacks NCU provenance".format(model))
    if (
        ncu_provenance.get("rank_scope") != "global_rank_0"
        or ncu_provenance.get("rank_ids") != [0]
        or ncu_provenance.get("physical_gpu_count") != 1
        or ncu_provenance.get("missing_kernel_count") != 0
    ):
        raise ValueError("{} functional source Task1 NCU scope is invalid".format(model))

    capture_summary = task1_manifest.get("capture_summary")
    if not isinstance(capture_summary, Mapping):
        raise ValueError(
            "{} functional source Task1 capture summary is missing".format(model)
        )
    if model == "gpt175b":
        expected_ranks = [0, 128, 256, 384, 512, 640, 768, 896]
        expected_scope = "representative"
    elif model == "qwen3_a30b":
        # Qwen3-A3B records one representative fake rank per PP stage
        # and EP group; TP/DP peers have equivalent execution graphs.
        expected_ranks = [
            pp * 8 * 4 + exp * 8 for pp in range(8) for exp in range(4)
        ]
        expected_scope = "representative_ep"
    else:
        raise ValueError("unsupported functional model: {}".format(model))
    if (
        capture_summary.get("capture_scope") != expected_scope
        or capture_summary.get("selected_rank_ids") != expected_ranks
        or capture_summary.get("selected_rank_count") != len(expected_ranks)
        or capture_summary.get("trace_file_count") != len(expected_ranks)
        or capture_summary.get("memory_json_count") != len(expected_ranks)
    ):
        raise ValueError(
            "{} functional source Task1 rank inventory is invalid".format(model)
        )


def _validate_functional_bundle_entry(
    module: Any,
    root: pathlib.Path,
    key: str,
    entry: Mapping[str, Any],
    expected_commits: Mapping[str, str],
    producer_record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate one non-release functional bundle without promoting evidence."""

    if not isinstance(entry, Mapping):
        raise ValueError("functional bundle entry must be an object: {}".format(key))
    bundle_root_rel = _safe_relative(entry.get("root"), "{} bundle root".format(key))
    manifest_rel = _safe_relative(entry.get("manifest"), "{} manifest".format(key))
    if manifest_rel != bundle_root_rel / "artifact_manifest.json":
        raise ValueError("{} nested manifest path is not canonical".format(key))
    bundle_root = _resolve_under(root, bundle_root_rel, "{} bundle root".format(key), directory=True)
    manifest_path = _resolve_under(root, manifest_rel, "{} manifest".format(key))
    if entry.get("manifest_sha256") != _manifest_digest(manifest_path):
        raise ValueError("{} nested manifest checksum mismatch".format(key))
    manifest = _verify_manifest(module, bundle_root, manifest_path, "{} manifest".format(key))
    if manifest.get("artifact_source") != "prebaked":
        raise ValueError("{} functional manifest must be prebaked".format(key))
    if manifest.get("execution_evidence") != FUNCTIONAL_DISTRIBUTION_EVIDENCE:
        raise ValueError("{} functional manifest evidence is invalid".format(key))
    expected_record_keys = {"bundle", "task2"} if key == "shared_task2" else {
        "bundle",
        "task1",
        "task3",
    }
    if not isinstance(producer_record, Mapping) or set(producer_record) != expected_record_keys:
        raise ValueError("{} functional producer record schema is invalid".format(key))
    bundle_commits = _validated_source_commits(
        producer_record.get("bundle"), "{} functional bundle".format(key)
    )
    if bundle_commits != dict(expected_commits):
        raise ValueError("{} functional bundle producer differs from the current checkout".format(key))
    source_artifact_commits = {
        task: _validated_source_commits(
            producer_record.get(task), "{} functional {}".format(key, task)
        )
        for task in expected_record_keys - {"bundle"}
    }
    if manifest.get("source_commits") != bundle_commits:
        raise ValueError("{} functional producer commits differ from the current checkout".format(key))
    if entry.get("producer_commits") != bundle_commits:
        raise ValueError("{} bundle producer differs from its nested manifest".format(key))
    if entry.get("source_artifact_commits") != source_artifact_commits:
        raise ValueError("{} bundle source producers differ from the distribution".format(key))
    if manifest.get("source_artifact_commits") != source_artifact_commits:
        raise ValueError("{} manifest source producers differ from the distribution".format(key))

    files = {entry["path"] for entry in manifest.get("files", [])}
    if key == "shared_task2":
        if manifest.get("model") != "shared_task2" or manifest.get("task") != "task2":
            raise ValueError("functional shared Task2 identity is invalid")
        for required in (
            "merge/input/kernel_metric_output.csv",
            "training_testing/output/xgb_model.json",
            "training_testing/output/standard_scaler.json",
            "provenance/source_task2_manifest.json",
        ):
            if required not in files:
                raise ValueError("functional shared Task2 is missing {}".format(required))
        source_manifest = _load_json(
            bundle_root / "provenance/source_task2_manifest.json",
            "functional source Task2 manifest",
        )
        if not _functional_source_evidence(source_manifest.get("execution_evidence"), task2=True):
            raise ValueError("functional source Task2 evidence is invalid")
        if source_manifest.get("source_commits") != source_artifact_commits["task2"]:
            raise ValueError("functional source Task2 producer identity is invalid")
    else:
        specification = MODELS[key]
        if manifest.get("model") != key or manifest.get("task") != "prebaked":
            raise ValueError("{} functional model identity is invalid".format(key))
        if manifest.get("simulation_topology") != specification["topology"]:
            raise ValueError("{} functional model topology is invalid".format(key))
        if manifest.get("profile") != specification["profile"]:
            raise ValueError("{} functional model profile is invalid".format(key))
        for required in (
            "provenance/source_task1_manifest.json",
            "provenance/source_task3_manifest.json",
            "provenance/source_task3_resolved_inputs.json",
            "slowdown_assets/manifest.json",
            "slowdown_assets/kernel_features.json",
            "slowdown_assets/backward_kernel_blueprints.json",
            "ncu/kernel_metric_output.csv",
        ):
            if required not in files:
                raise ValueError("{} functional model is missing {}".format(key, required))
        traces = [path for path in files if path.endswith(".txt")]
        if not traces:
            raise ValueError("{} functional model contains no trace files".format(key))
        if len([path for path in files if path.endswith(".sqlite")]) != 1:
            raise ValueError("{} functional model must contain exactly one SQLite artifact".format(key))
        task1_source = _load_json(
            bundle_root / "provenance/source_task1_manifest.json",
            "{} functional source Task1 manifest".format(key),
        )
        task3_source = _load_json(
            bundle_root / "provenance/source_task3_manifest.json",
            "{} functional source Task3 manifest".format(key),
        )
        _validate_functional_task1_source(key, task1_source)
        if not _functional_source_evidence(task3_source.get("execution_evidence")):
            raise ValueError("{} functional source Task3 evidence is invalid".format(key))
        if task1_source.get("source_commits") != source_artifact_commits["task1"]:
            raise ValueError("{} functional source Task1 producer identity is invalid".format(key))
        if task3_source.get("source_commits") != source_artifact_commits["task3"]:
            raise ValueError("{} functional source Task3 producer identity is invalid".format(key))
    return dict(manifest)


def verify_functional_distribution(repo_root: pathlib.Path, prebaked_root: pathlib.Path) -> Dict[str, Any]:
    """Verify a two-model functional bundle while retaining its non-release class."""

    repo_root = _regular_directory(pathlib.Path(repo_root), "repository root")
    prebaked_root = _regular_directory(pathlib.Path(prebaked_root), "functional prebaked root")
    module = _load_artifact_module(repo_root)
    distribution = _load_json(prebaked_root / "distribution_manifest.json", "functional distribution manifest")
    if distribution.get("schema_version") != FUNCTIONAL_DISTRIBUTION_SCHEMA:
        raise ValueError("functional distribution manifest schema is invalid")
    if distribution.get("artifact_source") != "prebaked":
        raise ValueError("functional distribution artifact_source must be prebaked")
    if distribution.get("execution_evidence") != FUNCTIONAL_DISTRIBUTION_EVIDENCE:
        raise ValueError("functional distribution evidence is invalid")
    expected_commits = _source_commits(repo_root)
    if distribution.get("compatible_commits") != {
        "echo_slowdown": expected_commits["echo_slowdown"],
        "megatron_sim_engine": expected_commits["megatron_sim_engine"],
    }:
        raise ValueError("functional distribution compatible commits differ from current gitlinks")
    _validate_distribution_files(prebaked_root, distribution)
    bundles = distribution.get("bundles")
    if not isinstance(bundles, Mapping) or set(bundles) != FUNCTIONAL_BUNDLES:
        raise ValueError("functional distribution bundles must contain gpt175b, qwen3_a30b, and shared_task2")
    producer_records = distribution.get("producer_commits")
    if not isinstance(producer_records, Mapping) or set(producer_records) != FUNCTIONAL_BUNDLES:
        raise ValueError("functional distribution producer records are invalid")
    manifests = {
        key: _validate_functional_bundle_entry(
            module,
            prebaked_root,
            key,
            bundles[key],
            expected_commits,
            producer_records[key],
        )
        for key in sorted(FUNCTIONAL_BUNDLES)
    }
    shared_entry = bundles["shared_task2"]
    shared_root = _resolve_under(
        prebaked_root,
        _safe_relative(shared_entry.get("root"), "shared Task2 bundle root"),
        "shared Task2 bundle root",
        directory=True,
    )
    task2_manifest_path = shared_root / "provenance/source_task2_manifest.json"
    task2_manifest = _load_json(task2_manifest_path, "functional source Task2 manifest")
    for model in sorted(FUNCTIONAL_MODELS):
        model_entry = bundles[model]
        model_root = _resolve_under(
            prebaked_root,
            _safe_relative(model_entry.get("root"), "{} bundle root".format(model)),
            "{} bundle root".format(model),
            directory=True,
        )
        task1_manifest_path = model_root / "provenance/source_task1_manifest.json"
        task3_manifest_path = model_root / "provenance/source_task3_manifest.json"
        resolved_inputs_path = model_root / "provenance/source_task3_resolved_inputs.json"
        _validate_functional_task3_provenance(
            model,
            _load_json(task1_manifest_path, "{} functional source Task1 manifest".format(model)),
            task1_manifest_path,
            task2_manifest,
            task2_manifest_path,
            _load_json(task3_manifest_path, "{} functional source Task3 manifest".format(model)),
            resolved_inputs_path,
        )
    predictor_ids = {manifests[key].get("predictor_run_id") for key in FUNCTIONAL_MODELS}
    if len(predictor_ids) != 1 or None in predictor_ids:
        raise ValueError("functional model bundles do not share one predictor_run_id")
    predictor_run_id = next(iter(predictor_ids))
    if manifests["shared_task2"].get("predictor_run_id") != predictor_run_id:
        raise ValueError("functional shared predictor_run_id differs from model bundles")
    captures = {manifests[key].get("capture_id") for key in FUNCTIONAL_MODELS}
    if len(captures) != len(FUNCTIONAL_MODELS) or None in captures:
        raise ValueError("functional model capture_ids must be distinct and present")
    return {
        "distribution_id": _safe_id(distribution.get("distribution_id"), "distribution_id"),
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
    *,
    require_real_evidence: bool = True,
    require_current_commits: bool = True,
) -> tuple[
    pathlib.Path,
    Dict[str, Any],
    pathlib.Path,
    Dict[str, Any],
    str,
    str,
    pathlib.Path,
    pathlib.Path,
]:
    task1_dir = _resolve_under(output_root, "{}/task1".format(model), "{} Task1 directory".format(model), directory=True)
    capture_marker = _read_marker(task1_dir, "capture_marker.json", "sc26-ae-task1-capture-marker-v1", "{} Task1 capture".format(model))
    if capture_marker.get("model") != model:
        raise ValueError("{} Task1 capture marker model mismatch".format(model))
    task1_root = _source_run_from_marker(task1_dir, capture_marker, "runs", "{} Task1".format(model))
    task1_manifest_path = task1_root / "artifact_manifest.json"
    task1_manifest = _verify_manifest(module, task1_root, task1_manifest_path, "{} Task1 manifest".format(model))
    task1_manifest_sha256 = _validate_marker_manifest(
        capture_marker, task1_manifest_path, "{} Task1 capture".format(model)
    )
    if require_real_evidence:
        _require_exact_evidence(task1_manifest, "real_single_h800_qualified", "{} Task1 manifest".format(model))
    elif task1_manifest.get("execution_evidence") not in {
        TASK1_REAL_EVIDENCE,
        TASK1_PENDING_EVIDENCE,
        TASK1_SYNTHETIC_EVIDENCE,
    }:
        raise ValueError(
            "{} Task1 manifest has unsupported functional evidence: {}".format(
                model, task1_manifest.get("execution_evidence")
            )
        )
    if task1_manifest.get("model") != model or task1_manifest.get("task") != "task1" or task1_manifest.get("artifact_source") != "fresh":
        raise ValueError("{} Task1 manifest identity is invalid".format(model))
    module.validate_task1_rank_promotion_scope(task1_manifest)
    if require_current_commits and task1_manifest.get("source_commits") != dict(expected_commits):
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
    task3_manifest_sha256 = _validate_marker_manifest(
        run_marker, task3_manifest_path, "{} Task3 run".format(model)
    )
    if run_marker.get("execution_evidence") != task3_manifest.get("execution_evidence"):
        raise ValueError("{} Task3 marker evidence differs from its manifest".format(model))
    if task3_manifest.get("model") != model or task3_manifest.get("task") != "task3" or task3_manifest.get("artifact_source") != "fresh":
        raise ValueError("{} Task3 manifest identity is invalid".format(model))
    if require_real_evidence:
        _require_exact_evidence(task3_manifest, TASK3_REAL_EVIDENCE, "{} Task3 manifest".format(model))
    elif task3_manifest.get("execution_evidence") not in {
        TASK3_REAL_EVIDENCE,
        TASK1_PENDING_EVIDENCE,
        TASK1_SYNTHETIC_EVIDENCE,
    }:
        raise ValueError(
            "{} Task3 manifest has unsupported functional evidence: {}".format(
                model, task3_manifest.get("execution_evidence")
            )
        )
    if require_current_commits and task3_manifest.get("source_commits") != dict(expected_commits):
        raise ValueError("{} Task3 source commits differ from current checkout".format(model))
    if task3_manifest.get("simulation_run_id") != simulation_run_id:
        raise ValueError("{} Task3 simulation_run_id differs from marker".format(model))
    if task3_manifest.get("capture_id") != run_marker.get("capture_id"):
        raise ValueError("{} Task3 capture_id differs from marker".format(model))
    if task3_manifest.get("capture_id") != task1_manifest.get("capture_id"):
        raise ValueError("{} Task3 capture_id differs from Task1".format(model))
    if task3_manifest.get("predictor_run_id") != run_marker.get("predictor_run_id"):
        raise ValueError("{} Task3 predictor_run_id differs from marker".format(model))
    return (
        task1_root,
        task1_manifest,
        task3_root,
        task3_manifest,
        task1_manifest_sha256,
        task3_manifest_sha256,
        task1_manifest_path,
        task3_manifest_path,
    )


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
    shared_manifest_sha256 = _validate_marker_manifest(
        pointer, shared_manifest_path, "shared Task2 pointer"
    )
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
        shared_manifest_sha256,
        expected_commits,
        predictor_run_id,
    )

    for model, specification in MODELS.items():
        (
            task1_root,
            task1_manifest,
            task3_root,
            task3_manifest,
            task1_manifest_sha256,
            task3_manifest_sha256,
            _,
            _,
        ) = _source_manifest_and_run(module, output_root, model, expected_commits)
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
            task1_manifest_sha256,
            task3_root,
            task3_manifest,
            task3_manifest_sha256,
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


def build_functional_distribution(
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
    staging_root: pathlib.Path,
    distribution_id: str,
    result_json: pathlib.Path,
) -> Dict[str, Any]:
    """Package the two official fake-level models without release promotion."""

    repo_root = _regular_directory(pathlib.Path(repo_root), "repository root")
    output_root = _regular_directory(pathlib.Path(output_root), "fresh output root")
    staging_root = pathlib.Path(staging_root)
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError("staging destination already exists: {}".format(staging_root))
    _safe_id(distribution_id, "distribution_id")
    module = _load_artifact_module(repo_root)
    expected_commits = _source_commits(repo_root)

    pointer = _read_marker(
        output_root / "_shared/task2",
        "predictor_marker.json",
        "sc26-ae-task2-shared-pointer-v1",
        "shared Task2 pointer",
    )
    predictor_run_id = _safe_id(pointer.get("predictor_run_id"), "predictor_run_id")
    shared_root = _source_run_from_marker(output_root, pointer, "_shared/task2/runs", "shared Task2")
    shared_manifest_path = shared_root / "artifact_manifest.json"
    shared_manifest = _verify_manifest(module, shared_root, shared_manifest_path, "shared Task2 manifest")
    shared_manifest_sha256 = _validate_marker_manifest(pointer, shared_manifest_path, "shared Task2 pointer")
    if not _functional_source_evidence(shared_manifest.get("execution_evidence"), task2=True):
        raise ValueError(
            "shared Task2 manifest has unsupported functional evidence: {}".format(
                shared_manifest.get("execution_evidence")
            )
        )
    if (
        shared_manifest.get("model") != "shared_task2"
        or shared_manifest.get("task") != "task2"
        or shared_manifest.get("artifact_source") != "fresh"
        or shared_manifest.get("predictor_run_id") != predictor_run_id
    ):
        raise ValueError("shared Task2 functional source identity is invalid")
    task2_source_commits = _validated_source_commits(
        shared_manifest.get("source_commits"), "shared Task2"
    )

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
        shared_manifest_sha256,
        expected_commits,
        predictor_run_id,
        evidence_override=FUNCTIONAL_DISTRIBUTION_EVIDENCE,
        source_artifact_commits={"task2": task2_source_commits},
    )

    for model in sorted(FUNCTIONAL_MODELS):
        specification = MODELS[model]
        (
            task1_root,
            task1_manifest,
            task3_root,
            task3_manifest,
            task1_manifest_sha256,
            task3_manifest_sha256,
            task1_manifest_path,
            _,
        ) = _source_manifest_and_run(
            module,
            output_root,
            model,
            expected_commits,
            require_real_evidence=False,
            require_current_commits=False,
        )
        _validate_functional_task1_source(model, task1_manifest)
        if task3_manifest.get("predictor_run_id") != predictor_run_id:
            raise ValueError("{} Task3 predictor_run_id differs from shared Task2".format(model))
        resolved_inputs_path = task3_root / "provenance/resolved_inputs.json"
        _validate_functional_task3_provenance(
            model,
            task1_manifest,
            task1_manifest_path,
            shared_manifest,
            shared_manifest_path,
            task3_manifest,
            resolved_inputs_path,
        )
        source_artifact_commits = {
            "task1": _validated_source_commits(
                task1_manifest.get("source_commits"), "{} Task1".format(model)
            ),
            "task3": _validated_source_commits(
                task3_manifest.get("source_commits"), "{} Task3".format(model)
            ),
        }
        bundle_entries[model] = _build_model_bundle(
            module,
            model,
            specification,
            output_root,
            bundles_root / model,
            task1_root,
            task1_manifest,
            task1_manifest_sha256,
            task3_root,
            task3_manifest,
            task3_manifest_sha256,
            expected_commits,
            _safe_id(task1_manifest["capture_id"], "capture_id"),
            predictor_run_id,
            evidence_override=FUNCTIONAL_DISTRIBUTION_EVIDENCE,
            source_artifact_commits=source_artifact_commits,
            task3_resolved_inputs_path=resolved_inputs_path,
        )

    entries = _distribution_files(staging_root)
    file_size_mib = {entry["path"]: round(entry["size_bytes"] / 1048576.0, 6) for entry in entries}
    distribution: Dict[str, Any] = {
        "schema_version": FUNCTIONAL_DISTRIBUTION_SCHEMA,
        "artifact_source": "prebaked",
        "distribution_id": distribution_id,
        "compatible_commits": {
            "echo_slowdown": expected_commits["echo_slowdown"],
            "megatron_sim_engine": expected_commits["megatron_sim_engine"],
        },
        "producer_commits": {
            key: {
                "bundle": dict(entry["producer_commits"]),
                **{
                    task: dict(commits)
                    for task, commits in entry["source_artifact_commits"].items()
                },
            }
            for key, entry in bundle_entries.items()
        },
        "bundles": bundle_entries,
        "files": entries,
        "file_size_mib": file_size_mib,
        "execution_evidence": FUNCTIONAL_DISTRIBUTION_EVIDENCE,
    }
    manifest_path = staging_root / "distribution_manifest.json"
    _finalize_distribution_manifest(module, staging_root, manifest_path, distribution)
    verified = verify_functional_distribution(repo_root, staging_root)
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
    functional = subparsers.add_parser(
        "build-functional",
        help="build a two-model fake-level bundle with explicit non-release evidence",
    )
    functional.add_argument("--repo-root", type=pathlib.Path, required=True)
    functional.add_argument("--output-root", type=pathlib.Path, required=True)
    functional.add_argument("--staging-root", type=pathlib.Path, required=True)
    functional.add_argument("--distribution-id", required=True)
    functional.add_argument("--result-json", type=pathlib.Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repo-root", type=pathlib.Path, required=True)
    verify.add_argument("--prebaked-root", type=pathlib.Path, required=True)
    verify_functional_parser = subparsers.add_parser(
        "verify-functional",
        help="verify a two-model fake-level bundle without release promotion",
    )
    verify_functional_parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    verify_functional_parser.add_argument("--prebaked-root", type=pathlib.Path, required=True)
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
    if args.command == "build-functional":
        result = build_functional_distribution(
            args.repo_root,
            args.output_root,
            args.staging_root,
            args.distribution_id,
            args.result_json,
        )
        print("DISTRIBUTION_STATUS=verified")
        print("DISTRIBUTION_EVIDENCE={}".format(FUNCTIONAL_DISTRIBUTION_EVIDENCE))
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
    if args.command == "verify-functional":
        result = verify_functional_distribution(args.repo_root, args.prebaked_root)
        print("DISTRIBUTION_STATUS=verified")
        print("DISTRIBUTION_EVIDENCE={}".format(FUNCTIONAL_DISTRIBUTION_EVIDENCE))
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
