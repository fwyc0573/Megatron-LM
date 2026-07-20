#!/usr/bin/env python3
"""Create and verify portable SC26 AE artifact manifests."""

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set


SCHEMA_VERSION = "sc26-ae-artifact-manifest-v1"
ALLOWED_MODELS = {"gpt175b", "qwen3_a30b", "dsv3", "shared_task2"}
ALLOWED_TASKS = {"task1", "task2", "task3", "prebaked"}
ALLOWED_SOURCES = {"fresh", "prebaked"}
ALLOWED_PROFILES = {"175", "full", "smoke", "shared"}
SOURCE_COMMIT_KEYS = {"megatron_lm", "echo_slowdown", "megatron_sim_engine"}
TOPOLOGY_KEYS = {"world_size", "local_size", "pp", "tp", "dp", "exp"}
CAPTURE_RUNTIME_KEYS = {
    "physical_gpu_count",
    "fake_gpus_per_node",
    "scaling_min_warmup_iters",
    "scaling_profile_iters",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# A real Task1 qualification for either MoE model is meaningful only when the
# producer captured every fake rank.  QUICK remains a valid local/synthetic
# scope; this gate is applied only when a caller is crossing the external
# qualification boundary.
TASK1_MOE_FULL_RANK_MODELS = {"qwen3_a30b", "dsv3"}
TASK1_MOE_FULL_RANK_COUNT = 256
TASK1_PROMOTION_EVIDENCE = {
    "runtime_measurement_requires_external_single_gpu_qualification",
    "real_single_h800_qualified",
}


def sha256_file(path: pathlib.Path) -> str:
    """Return the lowercase SHA256 digest of one regular file."""

    path = pathlib.Path(path)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError("Cannot hash a symlink: {}".format(path))
    if not stat.S_ISREG(mode):
        raise ValueError("Cannot hash a non-regular file: {}".format(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(name: str, value: Mapping[str, object], expected: Set[str]) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            "{} keys must be exactly {}; got {}".format(
                name, sorted(expected), sorted(observed)
            )
        )


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def validate_task1_rank_promotion_scope(metadata: Mapping[str, object]) -> None:
    """Reject a partial MoE capture at the real qualification boundary.

    The ordinary Task1 producer may emit a QUICK subset for local smoke
    validation.  Only callers that are about to accept a pending external
    qualification input or an already-qualified real manifest invoke this
    boundary check.  The exact ordered rank vector and both artifact counts
    are checked so a claimed 256-rank run cannot be backed by a four-rank
    subset.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("Task1 promotion metadata must be an object")
    if metadata.get("task") != "task1":
        return
    model = metadata.get("model")
    if model not in TASK1_MOE_FULL_RANK_MODELS:
        return
    if metadata.get("execution_evidence") not in TASK1_PROMOTION_EVIDENCE:
        return

    topology = metadata.get("simulation_topology")
    if not isinstance(topology, Mapping) or topology.get("world_size") != TASK1_MOE_FULL_RANK_COUNT:
        raise ValueError(
            "Task1 MoE promotion requires simulation_topology.world_size=256"
        )

    summary = metadata.get("capture_summary")
    if not isinstance(summary, Mapping):
        raise ValueError(
            "Task1 MoE promotion requires capture_summary rank inventory"
        )
    if summary.get("capture_scope") != "full":
        raise ValueError(
            "Task1 MoE promotion requires capture_scope=full; QUICK is not promotable"
        )

    selected = summary.get("selected_rank_ids")
    expected = list(range(TASK1_MOE_FULL_RANK_COUNT))
    if not isinstance(selected, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in selected
    ) or selected != expected:
        raise ValueError(
            "Task1 MoE promotion requires the exact full rank inventory 0..255"
        )
    for field in ("selected_rank_count", "trace_file_count", "memory_json_count"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != TASK1_MOE_FULL_RANK_COUNT:
            raise ValueError(
                "Task1 MoE promotion requires {}=256".format(field)
            )


def _validate_identity(name: str, value: object) -> None:
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        raise ValueError("{} must be a non-empty path-free string".format(name))


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("Manifest metadata must be an object")
    if "files" in metadata:
        raise ValueError("Manifest metadata must not contain files")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Invalid schema_version")

    model = metadata.get("model")
    if model not in ALLOWED_MODELS:
        raise ValueError("Invalid model")
    task = metadata.get("task")
    if task not in ALLOWED_TASKS:
        raise ValueError("Invalid task")
    source = metadata.get("artifact_source")
    if source not in ALLOWED_SOURCES:
        raise ValueError("Invalid artifact_source")

    source_commits = metadata.get("source_commits")
    if not isinstance(source_commits, Mapping):
        raise ValueError("source_commits must be an object")
    _require_exact_keys("source_commits", source_commits, SOURCE_COMMIT_KEYS)
    for key, value in source_commits.items():
        if not isinstance(value, str) or not COMMIT_PATTERN.fullmatch(value):
            raise ValueError("source_commits.{} must be a 40-hex commit".format(key))

    if task in {"task1", "task3", "prebaked"}:
        topology = metadata.get("simulation_topology")
        if not isinstance(topology, Mapping):
            raise ValueError("simulation_topology must be an object")
        _require_exact_keys("simulation_topology", topology, TOPOLOGY_KEYS)
        values = {
            key: _require_positive_int("simulation_topology.{}".format(key), topology[key])
            for key in TOPOLOGY_KEYS
        }
        if values["world_size"] != values["pp"] * values["tp"] * values["dp"]:
            raise ValueError("simulation_topology world_size must equal pp * tp * dp")
        if values["local_size"] != 8:
            raise ValueError("simulation_topology.local_size must equal 8")

    if task == "task1":
        if source != "fresh":
            raise ValueError("Task1 artifact_source must be fresh")
        _validate_identity("capture_id", metadata.get("capture_id"))
        if "predictor_run_id" in metadata:
            raise ValueError("Task1 manifest must omit predictor_run_id")
        capture_runtime = metadata.get("capture_runtime")
        if not isinstance(capture_runtime, Mapping):
            raise ValueError("capture_runtime must be an object")
        _require_exact_keys("capture_runtime", capture_runtime, CAPTURE_RUNTIME_KEYS)
        if _require_positive_int(
            "capture_runtime.physical_gpu_count", capture_runtime["physical_gpu_count"]
        ) != 1:
            raise ValueError("capture_runtime.physical_gpu_count must equal 1")
        _require_positive_int(
            "capture_runtime.fake_gpus_per_node", capture_runtime["fake_gpus_per_node"]
        )
        if capture_runtime["scaling_min_warmup_iters"] != 3:
            raise ValueError("capture_runtime.scaling_min_warmup_iters must equal 3")
        if capture_runtime["scaling_profile_iters"] != 1:
            raise ValueError("capture_runtime.scaling_profile_iters must equal 1")
        if metadata.get("profile") not in ALLOWED_PROFILES:
            raise ValueError("Invalid profile")
        if metadata.get("precision") != "bf16":
            raise ValueError("Task1 precision must be bf16")
        if metadata.get("mock_data") is not True:
            raise ValueError("Task1 mock_data must be true")
        if metadata.get("ddp_overlap") is not True:
            raise ValueError("Task1 ddp_overlap must be true")

    if task == "task2":
        _validate_identity("predictor_run_id", metadata.get("predictor_run_id"))
        if "capture_id" in metadata:
            raise ValueError("Task2 manifest must omit capture_id")

    if task in {"task3", "prebaked"}:
        _validate_identity("capture_id", metadata.get("capture_id"))
        _validate_identity("predictor_run_id", metadata.get("predictor_run_id"))
        if metadata.get("profile") not in ALLOWED_PROFILES - {"shared"}:
            raise ValueError("Invalid profile")
        if metadata.get("precision") != "bf16":
            raise ValueError("{} precision must be bf16".format(task))
        if metadata.get("ddp_overlap") is not True:
            raise ValueError("{} ddp_overlap must be true".format(task))


def _normalize_relative_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Artifact path must be a non-empty relative path")
    if "\\" in relative_path:
        raise ValueError("Artifact path must use POSIX separators: {}".format(relative_path))
    pure_path = pathlib.PurePosixPath(relative_path)
    if pure_path.is_absolute() or relative_path in {".", ""}:
        raise ValueError("Artifact path must be relative: {}".format(relative_path))
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError("Artifact path contains an unsafe path component: {}".format(relative_path))
    normalized = pure_path.as_posix()
    if normalized != relative_path:
        raise ValueError("Artifact path is not normalized: {}".format(relative_path))
    if normalized == "artifact_manifest.json":
        raise ValueError("artifact_manifest.json must not list itself")
    return normalized


def _inventory_regular_files(root: pathlib.Path) -> Set[str]:
    root = root.resolve(strict=True)
    regular_files = set()
    for current_root, directory_names, file_names in os.walk(str(root), followlinks=False):
        current = pathlib.Path(current_root)
        for name in list(directory_names):
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Artifact tree contains a symlink: {}".format(path))
            if not stat.S_ISDIR(mode):
                raise ValueError("Artifact tree contains a special entry: {}".format(path))
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Artifact tree contains a symlink: {}".format(path))
            if not stat.S_ISREG(mode):
                raise ValueError("Artifact tree contains a special file: {}".format(path))
            regular_files.add(path.relative_to(root).as_posix())
    return regular_files


def _validate_file_entries(files: object) -> List[Mapping[str, object]]:
    if not isinstance(files, list) or not files:
        raise ValueError("files must be a non-empty array")
    normalized_entries = []
    observed_paths = set()
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("Each files entry must contain path, size_bytes, and sha256")
        path = _normalize_relative_path(entry["path"])
        if path in observed_paths:
            raise ValueError("Duplicate artifact path: {}".format(path))
        observed_paths.add(path)
        size = entry["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Invalid size_bytes for {}".format(path))
        digest = entry["sha256"]
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError("Invalid sha256 for {}".format(path))
        normalized_entries.append(entry)
    if [entry["path"] for entry in normalized_entries] != sorted(observed_paths):
        raise ValueError("files entries must be sorted by path")
    return normalized_entries


def create_manifest(
    root: pathlib.Path, metadata: dict, relative_files: List[str]
) -> Dict[str, object]:
    """Create an in-memory manifest after validating a complete payload tree."""

    root = pathlib.Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Manifest root must be a directory: {}".format(root))
    _validate_metadata(metadata)
    normalized_paths = sorted(_normalize_relative_path(path) for path in relative_files)
    if not normalized_paths:
        raise ValueError("At least one payload file is required")
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("Artifact file list contains duplicates")

    inventory = _inventory_regular_files(root)
    permitted_inventory = set(normalized_paths)
    if "artifact_manifest.json" in inventory:
        permitted_inventory.add("artifact_manifest.json")
    missing = set(normalized_paths) - inventory
    if missing:
        raise ValueError("Missing artifact files: {}".format(sorted(missing)))
    unexpected = inventory - permitted_inventory
    if unexpected:
        raise ValueError("Unexpected artifact files: {}".format(sorted(unexpected)))

    entries = []
    for relative_path in normalized_paths:
        path = root / pathlib.PurePosixPath(relative_path)
        entries.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = dict(metadata)
    manifest["files"] = entries
    return manifest


def verify_manifest(root: pathlib.Path, manifest: dict) -> None:
    """Fail if a manifest or any artifact differs from the declared bundle."""

    root = pathlib.Path(root).resolve(strict=True)
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be an object")
    metadata = {key: value for key, value in manifest.items() if key != "files"}
    _validate_metadata(metadata)
    entries = _validate_file_entries(manifest.get("files"))
    inventory = _inventory_regular_files(root)
    listed_paths = {entry["path"] for entry in entries}
    permitted_inventory = set(listed_paths)
    if "artifact_manifest.json" in inventory:
        permitted_inventory.add("artifact_manifest.json")
    missing = listed_paths - inventory
    if missing:
        raise ValueError("Missing artifact files: {}".format(sorted(missing)))
    unexpected = inventory - permitted_inventory
    if unexpected:
        raise ValueError("Unexpected artifact files: {}".format(sorted(unexpected)))

    for entry in entries:
        path = root / pathlib.PurePosixPath(entry["path"])
        actual_size = path.stat().st_size
        if actual_size != entry["size_bytes"]:
            raise ValueError(
                "Artifact size mismatch for {}: expected {}, got {}".format(
                    entry["path"], entry["size_bytes"], actual_size
                )
            )
        actual_digest = sha256_file(path)
        if actual_digest != entry["sha256"]:
            raise ValueError(
                "Artifact SHA256 mismatch for {}: expected {}, got {}".format(
                    entry["path"], entry["sha256"], actual_digest
                )
            )


def evaluate_distribution_gate(
    distribution_root: pathlib.Path,
    per_file_limit_bytes: int = 52_428_800,
    bundle_limit_bytes: int = 524_288_000,
) -> str:
    """Return the required distribution medium for a complete staged tree."""

    root = pathlib.Path(distribution_root).resolve(strict=True)
    _require_positive_int("per_file_limit_bytes", per_file_limit_bytes)
    _require_positive_int("bundle_limit_bytes", bundle_limit_bytes)
    inventory = _inventory_regular_files(root)
    total_bytes = 0
    per_file_exceeded = False
    for relative_path in sorted(inventory):
        size_bytes = (root / pathlib.PurePosixPath(relative_path)).stat().st_size
        total_bytes += size_bytes
        if size_bytes >= per_file_limit_bytes:
            per_file_exceeded = True
    if per_file_exceeded or total_bytes > bundle_limit_bytes:
        return "github_release"
    return "regular_git"


def _load_json(path: pathlib.Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON document must be an object: {}".format(path))
    return payload


def _write_stable_json(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--root", type=pathlib.Path, required=True)
    create_parser.add_argument("--metadata-json", type=pathlib.Path, required=True)
    create_parser.add_argument("--file-list", type=pathlib.Path, required=True)
    create_parser.add_argument("--output", type=pathlib.Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=pathlib.Path, required=True)
    verify_parser.add_argument("--manifest", type=pathlib.Path, required=True)

    size_parser = subparsers.add_parser("size-gate")
    size_parser.add_argument("--root", type=pathlib.Path, required=True)
    size_parser.add_argument("--per-file-limit-bytes", type=int, default=52_428_800)
    size_parser.add_argument("--bundle-limit-bytes", type=int, default=524_288_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "create":
        root = args.root.resolve(strict=True)
        expected_output = root / "artifact_manifest.json"
        if args.output.resolve(strict=False) != expected_output:
            raise ValueError("Manifest output must be {}".format(expected_output))
        metadata = _load_json(args.metadata_json)
        relative_files = [
            line.strip()
            for line in args.file_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = create_manifest(root, metadata, relative_files)
        _write_stable_json(expected_output, manifest)
        verify_manifest(root, manifest)
        print("MANIFEST_STATUS=verified")
        print("MANIFEST_FILE_COUNT={}".format(len(manifest["files"])))
        return 0
    if args.command == "verify":
        manifest = _load_json(args.manifest)
        verify_manifest(args.root, manifest)
        print("MANIFEST_STATUS=verified")
        print("MANIFEST_FILE_COUNT={}".format(len(manifest["files"])))
        return 0
    if args.command == "size-gate":
        result = evaluate_distribution_gate(
            args.root,
            per_file_limit_bytes=args.per_file_limit_bytes,
            bundle_limit_bytes=args.bundle_limit_bytes,
        )
        print(result)
        return 0
    raise AssertionError("Unhandled command: {}".format(args.command))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("[ERROR] {}".format(error), file=sys.stderr)
        raise SystemExit(1)
