import importlib.util
import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "SC26-AE" / "tools" / "artifact_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sc26_ae_artifact_manifest", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task1_metadata(**overrides):
    metadata = {
        "schema_version": "sc26-ae-artifact-manifest-v1",
        "model": "qwen3_a30b",
        "task": "task1",
        "artifact_source": "fresh",
        "capture_id": "qwen3_a30b-20260718T000000Z",
        "source_commits": {
            "megatron_lm": "1" * 40,
            "echo_slowdown": "2" * 40,
            "megatron_sim_engine": "3" * 40,
        },
        "simulation_topology": {
            "world_size": 256,
            "local_size": 8,
            "pp": 4,
            "tp": 8,
            "dp": 8,
            "exp": 8,
        },
        "capture_runtime": {
            "physical_gpu_count": 1,
            "fake_gpus_per_node": 256,
            "scaling_min_warmup_iters": 3,
            "scaling_profile_iters": 1,
        },
        "profile": "full",
        "precision": "bf16",
        "mock_data": True,
        "ddp_overlap": True,
    }
    metadata.update(overrides)
    return metadata


def task3_metadata(task: str = "task3", **overrides):
    metadata = {
        "schema_version": "sc26-ae-artifact-manifest-v1",
        "model": "qwen3_a30b",
        "task": task,
        "artifact_source": "prebaked",
        "capture_id": "qwen3_a30b-20260718T000000Z",
        "predictor_run_id": "predictor-20260718T000000Z",
        "source_commits": {
            "megatron_lm": "1" * 40,
            "echo_slowdown": "2" * 40,
            "megatron_sim_engine": "3" * 40,
        },
        "simulation_topology": {
            "world_size": 256,
            "local_size": 8,
            "pp": 4,
            "tp": 8,
            "dp": 8,
            "exp": 8,
        },
        "profile": "full",
        "precision": "bf16",
        "ddp_overlap": True,
    }
    metadata.update(overrides)
    return metadata


def moe_promotion_summary(**overrides):
    summary = {
        "capture_scope": "full",
        "selected_rank_ids": list(range(256)),
        "selected_rank_count": 256,
        "trace_file_count": 256,
        "memory_json_count": 256,
    }
    summary.update(overrides)
    return summary


def moe_promotion_metadata(model: str = "qwen3_a30b", **summary_overrides):
    profile = "full" if model == "qwen3_a30b" else "smoke"
    return task1_metadata(
        model=model,
        profile=profile,
        execution_evidence="runtime_measurement_requires_external_single_gpu_qualification",
        capture_summary=moe_promotion_summary(**summary_overrides),
    )


def build_manifest(module, root: Path, **metadata_overrides):
    payload = root / "runtime" / "profiler_log" / "rank0.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("rank:0:forward_step(duration=1.0)\n", encoding="utf-8")
    manifest = module.create_manifest(
        root,
        task1_metadata(**metadata_overrides),
        ["runtime/profiler_log/rank0.txt"],
    )
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_sha256_and_manifest_file_order_are_deterministic(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "z.txt"
    second = tmp_path / "a.txt"
    first.write_bytes(b"z")
    second.write_bytes(b"a")

    manifest = module.create_manifest(
        tmp_path,
        task1_metadata(),
        ["z.txt", "a.txt"],
    )

    assert module.sha256_file(first) == (
        "594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06"
    )
    assert [entry["path"] for entry in manifest["files"]] == ["a.txt", "z.txt"]


@pytest.mark.parametrize(
    "relative_path",
    ["/absolute.txt", "../outside.txt", "nested/../../outside.txt", "", "."],
)
def test_create_rejects_unsafe_relative_paths(tmp_path: Path, relative_path: str) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="relative|path"):
        module.create_manifest(tmp_path, task1_metadata(), [relative_path])


def test_verify_rejects_missing_extra_size_and_hash_drift(tmp_path: Path) -> None:
    module = load_module()

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_manifest = build_manifest(module, missing_root)
    missing_manifest["files"][0]["path"] = "runtime/profiler_log/missing.txt"
    with pytest.raises(ValueError, match="Missing"):
        module.verify_manifest(missing_root, missing_manifest)

    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    extra_manifest = build_manifest(module, extra_root)
    (extra_root / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        module.verify_manifest(extra_root, extra_manifest)

    size_root = tmp_path / "size"
    size_root.mkdir()
    size_manifest = build_manifest(module, size_root)
    payload = size_root / "runtime" / "profiler_log" / "rank0.txt"
    payload.write_text("different-size", encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        module.verify_manifest(size_root, size_manifest)

    hash_root = tmp_path / "hash"
    hash_root.mkdir()
    hash_manifest = build_manifest(module, hash_root)
    payload = hash_root / "runtime" / "profiler_log" / "rank0.txt"
    original_size = payload.stat().st_size
    payload.write_bytes(b"x" * original_size)
    with pytest.raises(ValueError, match="SHA256"):
        module.verify_manifest(hash_root, hash_manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "schema"),
        ("model", "unknown", "model"),
        ("artifact_source", "automatic", "artifact_source"),
        ("profile", "unknown", "profile"),
        ("precision", "fp16", "precision"),
        ("mock_data", False, "mock_data"),
        ("ddp_overlap", False, "ddp_overlap"),
    ],
)
def test_create_rejects_invalid_task1_metadata(
    tmp_path: Path, field: str, value, message: str
) -> None:
    module = load_module()
    payload = tmp_path / "payload.txt"
    payload.write_text("payload", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.create_manifest(tmp_path, task1_metadata(**{field: value}), ["payload.txt"])


@pytest.mark.parametrize(
    ("capture_runtime", "message"),
    [
        (
            {
                "physical_gpu_count": 1,
                "fake_gpus_per_node": 256,
                "scaling_min_warmup_iters": 2,
                "scaling_profile_iters": 1,
            },
            "scaling_min_warmup_iters",
        ),
        (
            {
                "physical_gpu_count": 1,
                "fake_gpus_per_node": 256,
                "scaling_min_warmup_iters": 3,
                "scaling_profile_iters": 2,
            },
            "scaling_profile_iters",
        ),
        (
            {
                "physical_gpu_count": 2,
                "fake_gpus_per_node": 256,
                "scaling_min_warmup_iters": 3,
                "scaling_profile_iters": 1,
            },
            "physical_gpu_count",
        ),
    ],
)
def test_task1_capture_runtime_is_frozen(
    tmp_path: Path, capture_runtime: dict, message: str
) -> None:
    module = load_module()
    payload = tmp_path / "payload.txt"
    payload.write_text("payload", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.create_manifest(
            tmp_path,
            task1_metadata(capture_runtime=capture_runtime),
            ["payload.txt"],
        )


@pytest.mark.parametrize("model", ["qwen3_a30b", "dsv3"])
def test_task1_moe_promotion_accepts_exact_full_rank_inventory(model: str) -> None:
    module = load_module()

    # This is a control-plane fixture only.  It proves the exact promotion
    # predicate accepts the intended full inventory; it does not qualify GPU
    # hardware or change the evidence class of any artifact.
    module.validate_task1_rank_promotion_scope(moe_promotion_metadata(model))


@pytest.mark.parametrize(
    ("case", "metadata", "message"),
    [
        (
            "missing-capture-summary",
            task1_metadata(
                model="qwen3_a30b",
                profile="full",
                execution_evidence="runtime_measurement_requires_external_single_gpu_qualification",
            ),
            "capture_summary",
        ),
        (
            "quick-scope",
            moe_promotion_metadata(capture_scope="quick"),
            "capture_scope",
        ),
        (
            "missing-one-rank",
            moe_promotion_metadata(selected_rank_ids=list(range(255))),
            "exact full rank inventory",
        ),
        (
            "duplicate-rank",
            moe_promotion_metadata(selected_rank_ids=[*range(255), 254]),
            "exact full rank inventory",
        ),
        (
            "wrong-order",
            moe_promotion_metadata(selected_rank_ids=[1, 0, *range(2, 256)]),
            "exact full rank inventory",
        ),
        (
            "boolean-rank",
            moe_promotion_metadata(selected_rank_ids=[True, *range(1, 256)]),
            "exact full rank inventory",
        ),
        (
            "selected-count",
            moe_promotion_metadata(selected_rank_count=255),
            "selected_rank_count",
        ),
        (
            "trace-count",
            moe_promotion_metadata(trace_file_count=255),
            "trace_file_count",
        ),
        (
            "memory-count",
            moe_promotion_metadata(memory_json_count=255),
            "memory_json_count",
        ),
        (
            "wrong-topology",
            task1_metadata(
                model="qwen3_a30b",
                profile="full",
                execution_evidence="runtime_measurement_requires_external_single_gpu_qualification",
                simulation_topology={
                    "world_size": 255,
                    "local_size": 8,
                    "pp": 4,
                    "tp": 8,
                    "dp": 8,
                    "exp": 8,
                },
                capture_summary=moe_promotion_summary(),
            ),
            "world_size",
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_task1_moe_promotion_rejects_malformed_inventory(
    case: str, metadata: dict, message: str
) -> None:
    module = load_module()

    with pytest.raises(ValueError, match=message):
        module.validate_task1_rank_promotion_scope(metadata)


def test_task1_gpt_representative_scope_is_not_subject_to_moe_full_rank_gate() -> None:
    module = load_module()
    metadata = task1_metadata(
        model="gpt175b",
        profile="175",
        execution_evidence="real_single_h800_qualified",
        capture_summary={
            "capture_scope": "representative",
            "selected_rank_ids": [0, 128, 256, 384, 512, 640, 768, 896],
            "selected_rank_count": 8,
            "trace_file_count": 8,
            "memory_json_count": 8,
        },
    )

    module.validate_task1_rank_promotion_scope(metadata)


@pytest.mark.parametrize("metadata", [None, [], "not-an-object"])
def test_task1_moe_promotion_rejects_non_object_metadata(metadata) -> None:
    module = load_module()

    with pytest.raises(ValueError, match="metadata must be an object"):
        module.validate_task1_rank_promotion_scope(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        task1_metadata(
            model="qwen3_a30b",
            execution_evidence="local_synthetic_not_gpu_qualification",
        ),
        task1_metadata(
            model="qwen3_a30b",
            execution_evidence="real_single_h800_qualified",
            task="task3",
        ),
    ],
)
def test_task1_moe_promotion_gate_skips_non_promotion_inputs(metadata: dict) -> None:
    module = load_module()

    # QUICK/local and non-Task1 metadata remain valid inputs to their own
    # lifecycle paths; this helper only guards the requested promotion seam.
    module.validate_task1_rank_promotion_scope(metadata)


def test_verify_rejects_topology_and_manifest_metadata_drift(tmp_path: Path) -> None:
    module = load_module()
    manifest = build_manifest(module, tmp_path)

    bad_topology = json.loads(json.dumps(manifest))
    bad_topology["simulation_topology"]["world_size"] = 255
    with pytest.raises(ValueError, match="simulation_topology"):
        module.verify_manifest(tmp_path, bad_topology)

    bad_capture_id = json.loads(json.dumps(manifest))
    bad_capture_id["capture_id"] = ""
    with pytest.raises(ValueError, match="capture_id"):
        module.verify_manifest(tmp_path, bad_capture_id)


@pytest.mark.parametrize("task", ["task3", "prebaked"])
def test_task3_and_prebaked_metadata_require_bf16_overlap_profile(
    tmp_path: Path, task: str
) -> None:
    module = load_module()
    payload = tmp_path / "payload.txt"
    payload.write_text("payload", encoding="utf-8")

    manifest = module.create_manifest(
        tmp_path,
        task3_metadata(task=task),
        ["payload.txt"],
    )
    assert manifest["profile"] == "full"
    assert manifest["precision"] == "bf16"
    assert manifest["ddp_overlap"] is True

    for field, value, message in (
        ("profile", "shared", "profile"),
        ("precision", "fp16", "precision"),
        ("ddp_overlap", False, "ddp_overlap"),
    ):
        with pytest.raises(ValueError, match=message):
            module.create_manifest(
                tmp_path,
                task3_metadata(task=task, **{field: value}),
                ["payload.txt"],
            )


def test_verify_rejects_symlinks(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "target.txt"
    target.write_text("payload", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        module.create_manifest(tmp_path, task1_metadata(), ["link.txt"])


def test_distribution_gate_boundaries_and_nested_files(tmp_path: Path) -> None:
    module = load_module()
    under_limit = tmp_path / "under"
    under_limit.mkdir()
    (under_limit / "payload.bin").write_bytes(b"x")
    assert (
        module.evaluate_distribution_gate(
            under_limit, per_file_limit_bytes=2, bundle_limit_bytes=1
        )
        == "regular_git"
    )

    exact_file_limit = tmp_path / "file-limit"
    exact_file_limit.mkdir()
    (exact_file_limit / "artifact_manifest.json").write_bytes(b"xx")
    assert (
        module.evaluate_distribution_gate(
            exact_file_limit, per_file_limit_bytes=2, bundle_limit_bytes=100
        )
        == "github_release"
    )

    over_total = tmp_path / "total-limit"
    (over_total / "nested").mkdir(parents=True)
    (over_total / "nested" / "artifact_manifest.json").write_bytes(b"xx")
    (over_total / "distribution_manifest.json").write_bytes(b"x")
    assert (
        module.evaluate_distribution_gate(
            over_total, per_file_limit_bytes=10, bundle_limit_bytes=2
        )
        == "github_release"
    )


def test_distribution_gate_rejects_symlink(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        module.evaluate_distribution_gate(tmp_path)


def test_cli_create_and_verify_write_stable_json(tmp_path: Path) -> None:
    module = load_module()
    payload = tmp_path / "payload.txt"
    payload.write_text("payload", encoding="utf-8")
    metadata_path = tmp_path.parent / f"{tmp_path.name}-metadata.json"
    file_list_path = tmp_path.parent / f"{tmp_path.name}-files.txt"
    metadata_path.write_text(json.dumps(task1_metadata()), encoding="utf-8")
    file_list_path.write_text("payload.txt\n", encoding="utf-8")
    output = tmp_path / "artifact_manifest.json"

    assert (
        module.main(
            [
                "create",
                "--root",
                os.fspath(tmp_path),
                "--metadata-json",
                os.fspath(metadata_path),
                "--file-list",
                os.fspath(file_list_path),
                "--output",
                os.fspath(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert (
        module.main(
            [
                "verify",
                "--root",
                os.fspath(tmp_path),
                "--manifest",
                os.fspath(output),
            ]
        )
        == 0
    )
