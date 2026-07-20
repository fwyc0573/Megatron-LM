"""Contract tests for external qualification evidence sealing.

These tests use small local fixtures only.  They prove the control-plane
contract; they do not qualify a GPU, image, runtime, or release dataset.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "SC26-AE" / "tools" / "seal_qualification.py"
ARTIFACT_TOOL_PATH = REPO_ROOT / "SC26-AE" / "tools" / "artifact_manifest.py"
IMAGE_REF = "hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae"
IMAGE_DIGEST = "sha256:" + "a" * 64
RESULT_SCHEMA_VERSION = "sc26-ae-qualification-result-v1"
COMMITS = {
    "megatron_lm": "1" * 40,
    "echo_slowdown": "2" * 40,
    "megatron_sim_engine": "3" * 40,
}


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def stable_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(
    tmp_path: Path,
    *,
    task: str = "task1",
    model: str = "gpt175b",
    execution_evidence: str | None = None,
) -> Path:
    artifact = load_module(ARTIFACT_TOOL_PATH, "sc26_ae_seal_fixture_artifact")
    if execution_evidence is None:
        execution_evidence = (
            "runtime_measurement_requires_external_two_gpu_qualification"
            if task == "task2"
            else "runtime_measurement_requires_external_single_gpu_qualification"
        )
    root = tmp_path / "source"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime/trace_rank0.txt").write_text("rank:0:forward_step(duration=1.0)\n", encoding="utf-8")
    metadata = {
        "schema_version": artifact.SCHEMA_VERSION,
        "model": model,
        "task": task,
        "artifact_source": "fresh",
        "source_commits": COMMITS,
        "execution_evidence": execution_evidence,
    }
    if task in {"task1", "task3"}:
        metadata.update(
            {
                "capture_id": "{}-capture-001".format(task),
                "simulation_topology": {
                    "world_size": 1024,
                    "local_size": 8,
                    "pp": 8,
                    "tp": 8,
                    "dp": 16,
                    "exp": 1,
                },
                "capture_runtime": {
                    "physical_gpu_count": 1,
                    "fake_gpus_per_node": 8,
                    "scaling_min_warmup_iters": 3,
                    "scaling_profile_iters": 1,
                },
                "profile": "175",
                "precision": "bf16",
                "mock_data": True,
                "ddp_overlap": True,
            }
        )
        if task == "task3":
            metadata.update(
                {
                    "model": "qwen3_a30b",
                    "predictor_run_id": "predictor-001",
                    "profile": "full",
                }
            )
    elif task == "task2":
        metadata["model"] = "shared_task2"
        metadata["predictor_run_id"] = "predictor-001"
        (root / "metrics.json").write_text("{\"rows\": 2}\n", encoding="utf-8")
    else:
        raise AssertionError(task)
    manifest = artifact.create_manifest(
        root,
        metadata,
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()),
    )
    stable_json(root / "artifact_manifest.json", manifest)
    artifact.verify_manifest(root, manifest)
    return root


def make_moe_promotion_source(tmp_path: Path, model: str, *, full: bool) -> Path:
    """Build a local control-plane fixture for the Task1 MoE promotion gate."""

    artifact = load_module(
        ARTIFACT_TOOL_PATH, "sc26_ae_seal_moe_fixture_artifact_{}".format(model)
    )
    source = make_source(tmp_path, task="task1", model=model)
    manifest_path = source / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = "full" if model == "qwen3_a30b" else "smoke"
    manifest["simulation_topology"] = {
        "world_size": 256,
        "local_size": 8,
        "pp": 4,
        "tp": 8,
        "dp": 8,
        "exp": 8,
    }
    manifest["capture_runtime"]["fake_gpus_per_node"] = 256
    manifest["execution_evidence"] = (
        "runtime_measurement_requires_external_single_gpu_qualification"
    )
    if full:
        manifest["capture_summary"] = {
            "capture_scope": "full",
            "selected_rank_ids": list(range(256)),
            "selected_rank_count": 256,
            "trace_file_count": 256,
            "memory_json_count": 256,
        }
    else:
        manifest["capture_summary"] = {
            "capture_scope": "quick",
            "selected_rank_ids": [0, 64, 128, 192],
            "selected_rank_count": 4,
            "trace_file_count": 4,
            "memory_json_count": 4,
        }
    stable_json(manifest_path, manifest)
    artifact.verify_manifest(source, manifest)
    return source


def make_attestation(
    tmp_path: Path,
    source: Path,
    *,
    task: str = "task1",
    model: str = "gpt175b",
    gpu_count: int = 1,
    gpu_uuids: list[str] | None = None,
    status: str = "PASS",
    image_ref: str = IMAGE_REF,
    image_digest: str = IMAGE_DIGEST,
    source_manifest_sha256: str | None = None,
    result_status: str = "PASS",
    result_overrides: dict[str, object] | None = None,
) -> Path:
    if gpu_uuids is None:
        gpu_uuids = ["GPU-1111"] if gpu_count == 1 else ["GPU-1111", "GPU-2222"]
    result = tmp_path / "qualification_result.json"
    result_payload: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": result_status,
        "task": task,
        "model": "shared_task2" if task == "task2" else model,
        "image_ref": image_ref,
        "image_digest": image_digest,
        "source_commits": dict(COMMITS),
        "gpu_model": "NVIDIA H800",
        "gpu_count": gpu_count,
        "gpu_uuids": list(gpu_uuids),
        "command_status": {"predict_only": 0, "qualification": 0},
        "issuer": "external_h800_qualification_authority",
        "qualification_source": "external_worker_attestation",
    }
    if result_overrides:
        result_payload.update(result_overrides)
    stable_json(result, result_payload)
    metrics = tmp_path / "qualification_metrics.json"
    stable_json(metrics, {"rows": 2, "test_mse": 0.25})
    source_manifest = source / "artifact_manifest.json"
    target_evidence = (
        "real_exact_two_h800_qualified"
        if task == "task2"
        else "real_single_h800_qualified"
    )
    payload = {
        "schema_version": "sc26-ae-qualification-attestation-v1",
        "attestation_id": "attestation-001",
        "status": status,
        "target_task": task,
        "target_model": "shared_task2" if task == "task2" else model,
        "target_execution_evidence": target_evidence,
        "source_commits": COMMITS,
        "source_manifest_sha256": source_manifest_sha256 or digest(source_manifest),
        "image_ref": image_ref,
        "image_digest": image_digest,
        "gpu_model": "NVIDIA H800",
        "gpu_count": gpu_count,
        "gpu_uuids": gpu_uuids,
        "qualification_result_path": result.name,
        "qualification_result_sha256": digest(result),
        "qualification_metrics_path": metrics.name,
        "qualification_metrics_sha256": digest(metrics),
    }
    attestation = tmp_path / "attestation.json"
    stable_json(attestation, payload)
    return attestation


def test_seal_copies_pending_source_and_rebuilds_manifest_and_receipt(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_qualification")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)
    destination = tmp_path / "sealed" / "gpt175b"
    receipt = tmp_path / "receipts" / "gpt175b.json"

    result = module.seal_qualification(source, attestation, destination, receipt)

    source_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    sealed_manifest_path = destination / "artifact_manifest.json"
    sealed_manifest = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
    assert source_manifest["execution_evidence"].startswith("runtime_measurement_requires_external")
    assert sealed_manifest["execution_evidence"] == "real_single_h800_qualified"
    assert sealed_manifest["qualification_status"] == "PASS"
    assert sealed_manifest["source_manifest_sha256"] == digest(source / "artifact_manifest.json")
    assert (destination / "provenance/source_artifact_manifest.json").is_file()
    assert (destination / "provenance/qualification_attestation.json").is_file()
    assert (destination / "provenance/qualification_result.json").is_file()
    assert (destination / "provenance/qualification_metrics.json").is_file()
    assert digest(destination / "provenance/qualification_attestation.json") == digest(attestation)
    assert result["execution_evidence"] == "real_single_h800_qualified"
    assert result["sealed_manifest_sha256"] == digest(sealed_manifest_path)
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "PASS"

    artifact = load_module(ARTIFACT_TOOL_PATH, "sc26_ae_seal_verify_artifact")
    artifact.verify_manifest(destination, sealed_manifest)
    assert digest(source / "artifact_manifest.json") != digest(sealed_manifest_path)


@pytest.mark.parametrize("model", ["qwen3_a30b", "dsv3"])
def test_seal_rejects_moe_quick_capture_before_external_promotion(
    tmp_path: Path, model: str
) -> None:
    """A QUICK MoE source cannot enter the external qualification seal path."""

    module = load_module(TOOL_PATH, "sc26_ae_seal_moe_quick_{}".format(model))
    source = make_moe_promotion_source(tmp_path, model, full=False)
    attestation = make_attestation(tmp_path, source, model=model)
    destination = tmp_path / "sealed-{}".format(model)

    with pytest.raises(
        ValueError,
        match="capture_scope|full rank inventory|rank inventory",
    ):
        module.seal_qualification(
            source,
            attestation,
            destination,
            tmp_path / "receipt-{}.json".format(model),
        )
    assert not destination.exists()


@pytest.mark.parametrize("model", ["qwen3_a30b", "dsv3"])
def test_seal_accepts_moe_full_capture_control_plane_fixture(
    tmp_path: Path, model: str
) -> None:
    """The same boundary accepts the exact full-rank MoE inventory."""

    module = load_module(TOOL_PATH, "sc26_ae_seal_moe_full_{}".format(model))
    source = make_moe_promotion_source(tmp_path, model, full=True)
    attestation = make_attestation(tmp_path, source, model=model)
    destination = tmp_path / "sealed-{}".format(model)
    receipt = tmp_path / "receipt-{}.json".format(model)

    result = module.seal_qualification(source, attestation, destination, receipt)

    assert result["target_task"] == "task1"
    assert result["target_model"] == model
    assert result["execution_evidence"] == "real_single_h800_qualified"
    assert destination.is_dir()
    assert receipt.is_file()


def test_seal_rejects_synthetic_source_without_creating_destination(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_synthetic")
    source = make_source(
        tmp_path,
        execution_evidence="local_synthetic_not_gpu_qualification",
    )
    attestation = make_attestation(tmp_path, source)
    destination = tmp_path / "sealed"
    with pytest.raises(ValueError, match="synthetic|promotable"):
        module.seal_qualification(source, attestation, destination, tmp_path / "receipt.json")
    assert not destination.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "PENDING", "status"),
        ("image_ref", "wrong/image:tag", "image_ref"),
        ("image_digest", "sha256:bad", "image_digest"),
        ("gpu_uuids", ["GPU-1111", "GPU-1111"], "gpu_count|distinct"),
        ("gpu_uuids", ["not-a-gpu"], "gpu_uuids"),
        ("source_manifest_sha256", "0" * 64, "source manifest"),
    ],
)
def test_seal_rejects_invalid_attestation_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_invalid_fields")
    source = make_source(tmp_path)
    kwargs = {field: value}
    attestation = make_attestation(tmp_path, source, **kwargs)
    with pytest.raises(ValueError, match=message):
        module.seal_qualification(source, attestation, tmp_path / "sealed", tmp_path / "receipt.json")


def test_seal_rejects_noncanonical_qualification_result(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_result_schema")
    source = make_source(tmp_path)
    attestation = make_attestation(
        tmp_path,
        source,
        result_overrides={"image_ref": "wrong/image:tag"},
    )
    with pytest.raises(ValueError, match="qualification result image_ref"):
        module.seal_qualification(source, attestation, tmp_path / "sealed", tmp_path / "receipt.json")

    result = tmp_path / "qualification_result.json"
    result_payload = json.loads(result.read_text(encoding="utf-8"))
    result_payload.pop("command_status")
    stable_json(result, result_payload)
    attestation_payload = json.loads(attestation.read_text(encoding="utf-8"))
    attestation_payload["qualification_result_sha256"] = digest(result)
    stable_json(attestation, attestation_payload)
    with pytest.raises(ValueError, match="qualification result keys"):
        module.seal_qualification(
            source, attestation, tmp_path / "sealed-missing", tmp_path / "receipt-missing.json"
        )


def test_seal_rejects_nonzero_result_command_status(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_result_command_status")
    source = make_source(tmp_path)
    attestation = make_attestation(
        tmp_path,
        source,
        result_overrides={"command_status": {"qualification": 1}},
    )
    with pytest.raises(ValueError, match="command_status"):
        module.seal_qualification(source, attestation, tmp_path / "sealed", tmp_path / "receipt.json")


def test_seal_supports_task3_real_single_gpu_branch(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_task3")
    source = make_source(tmp_path, task="task3", model="qwen3_a30b")
    attestation = make_attestation(tmp_path, source, task="task3", model="qwen3_a30b")
    destination = tmp_path / "sealed-task3"
    receipt = tmp_path / "receipt-task3.json"
    result = module.seal_qualification(source, attestation, destination, receipt)
    assert result["target_task"] == "task3"
    assert result["target_model"] == "qwen3_a30b"
    assert result["execution_evidence"] == "real_single_h800_qualified"
    assert json.loads((destination / "artifact_manifest.json").read_text(encoding="utf-8"))[
        "qualification_result_schema"
    ] == RESULT_SCHEMA_VERSION


def test_seal_requires_two_distinct_h800s_for_task2(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_task2")
    source = make_source(tmp_path, task="task2", model="shared_task2")
    attestation = make_attestation(
        tmp_path,
        source,
        task="task2",
        model="shared_task2",
        gpu_count=1,
        gpu_uuids=["GPU-1111"],
    )
    with pytest.raises(ValueError, match="gpu_count|exactly two|UUID"):
        module.seal_qualification(source, attestation, tmp_path / "sealed", tmp_path / "receipt.json")


def test_seal_accepts_task2_exact_two_h800s(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_task2_valid")
    source = make_source(tmp_path, task="task2", model="shared_task2")
    attestation = make_attestation(
        tmp_path,
        source,
        task="task2",
        model="shared_task2",
        gpu_count=2,
        gpu_uuids=["GPU-1111", "GPU-2222"],
    )
    result = module.seal_qualification(
        source,
        attestation,
        tmp_path / "sealed-task2",
        tmp_path / "receipt-task2.json",
    )
    assert result["target_task"] == "task2"
    assert result["target_model"] == "shared_task2"
    assert result["execution_evidence"] == "real_exact_two_h800_qualified"
    assert result["gpu_count"] == 2


def test_seal_rejects_existing_destination_and_receipt(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_existing")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)
    destination = tmp_path / "sealed"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="destination"):
        module.seal_qualification(source, attestation, destination, tmp_path / "receipt.json")

    destination.rmdir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("already exists\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="receipt"):
        module.seal_qualification(source, attestation, destination, receipt)


def test_seal_rejects_result_status_failure_and_hash_mismatch(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_result")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source, result_status="FAIL")
    with pytest.raises(ValueError, match="qualification result.*PASS"):
        module.seal_qualification(source, attestation, tmp_path / "sealed", tmp_path / "receipt.json")

    source = make_source(tmp_path / "hash", task="task1")
    attestation = make_attestation(tmp_path / "hash", source)
    result = tmp_path / "hash" / "qualification_result.json"
    result.write_text(result.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="qualification result.*SHA256"):
        module.seal_qualification(
            source, attestation, tmp_path / "sealed-hash", tmp_path / "receipt-hash.json"
        )


@pytest.mark.parametrize(
    ("path_field", "hash_field", "payload_name"),
    [
        (
            "qualification_result_path",
            "qualification_result_sha256",
            "qualification_result.json",
        ),
        (
            "qualification_metrics_path",
            "qualification_metrics_sha256",
            "qualification_metrics.json",
        ),
    ],
)
def test_seal_rejects_attestation_payload_symlink(
    tmp_path: Path, path_field: str, hash_field: str, payload_name: str
) -> None:
    """Qualification payloads must be regular files, not in-tree symlinks."""

    module = load_module(TOOL_PATH, "sc26_ae_seal_payload_symlink")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    target = tmp_path / (payload_name + ".target")
    original = tmp_path / payload_name
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target.name)
    payload[path_field] = original.name
    payload[hash_field] = digest(target)
    stable_json(attestation, payload)

    with pytest.raises(ValueError, match="symlink|regular file"):
        module.seal_qualification(
            source,
            attestation,
            tmp_path / "sealed",
            tmp_path / "receipt.json",
        )


def test_seal_rejects_attestation_payload_intermediate_symlink(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_intermediate_symlink")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    target_dir = tmp_path / "payload-target"
    target_dir.mkdir()
    target = target_dir / "result.json"
    target.write_bytes((tmp_path / "qualification_result.json").read_bytes())
    link_dir = tmp_path / "payload-link"
    link_dir.symlink_to(target_dir, target_is_directory=True)
    payload["qualification_result_path"] = "payload-link/result.json"
    payload["qualification_result_sha256"] = digest(target)
    stable_json(attestation, payload)

    with pytest.raises(ValueError, match="symlink"):
        module.seal_qualification(
            source,
            attestation,
            tmp_path / "sealed",
            tmp_path / "receipt.json",
        )


def test_seal_cleans_staging_when_source_changes_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_source_toctou")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)
    original_copy = module._copy_source_tree

    def copy_then_mutate(source_root, destination, entries):
        original_copy(source_root, destination, entries)
        trace = source_root / "runtime/trace_rank0.txt"
        trace.write_text(
            trace.read_text(encoding="utf-8") + "tampered" + chr(10),
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "_copy_source_tree", copy_then_mutate)
    destination = tmp_path / "sealed-toctou"
    receipt = tmp_path / "receipt-toctou.json"
    with pytest.raises(ValueError, match="changed|manifest"):
        module.seal_qualification(source, attestation, destination, receipt)
    assert not destination.exists()
    assert not receipt.exists()
    assert not list(destination.parent.glob(".sealed-toctou-staging-*"))


def test_seal_cleans_staging_after_provenance_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_partial_cleanup")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)

    def fail_copy(*_args, **_kwargs):
        raise OSError("injected provenance copy failure")

    monkeypatch.setattr(module, "_copy_provenance_file", fail_copy)
    destination = tmp_path / "sealed-partial"
    receipt = tmp_path / "receipt-partial.json"
    with pytest.raises(OSError, match="injected provenance"):
        module.seal_qualification(source, attestation, destination, receipt)
    assert not destination.exists()
    assert not receipt.exists()
    assert not list(destination.parent.glob(".sealed-partial-staging-*"))


def test_seal_rejects_reserved_provenance_collision(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_reserved_collision")
    artifact = load_module(ARTIFACT_TOOL_PATH, "sc26_ae_seal_reserved_artifact")
    source = make_source(tmp_path)
    collision = source / "provenance/source_artifact_manifest.json"
    collision.parent.mkdir(parents=True)
    collision.write_text("collision" + chr(10), encoding="utf-8")
    old_manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    metadata = {key: value for key, value in old_manifest.items() if key != "files"}
    manifest = artifact.create_manifest(
        source,
        metadata,
        sorted(
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
    )
    stable_json(source / "artifact_manifest.json", manifest)
    artifact.verify_manifest(source, manifest)
    attestation = make_attestation(tmp_path, source)
    with pytest.raises(ValueError, match="collides"):
        module.seal_qualification(
            source, attestation, tmp_path / "sealed", tmp_path / "receipt.json"
        )


def test_seal_rejects_source_symlink_before_publication(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_source_symlink")
    source = make_source(tmp_path)
    (source / "runtime/source-link.txt").symlink_to("trace_rank0.txt")
    attestation = make_attestation(tmp_path, source)
    with pytest.raises(ValueError, match="symlink|manifest"):
        module.seal_qualification(
            source, attestation, tmp_path / "sealed", tmp_path / "receipt.json"
        )
    assert not (tmp_path / "sealed").exists()


def test_seal_cleans_staging_when_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_publish_failure")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)

    def fail_publish(*_args, **_kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(module, "_publish_staging_root", fail_publish)
    destination = tmp_path / "sealed-publish-failure"
    receipt = tmp_path / "receipt-publish-failure.json"
    with pytest.raises(OSError, match="injected publication"):
        module.seal_qualification(source, attestation, destination, receipt)
    assert not destination.exists()
    assert not receipt.exists()
    assert not list(destination.parent.glob(".sealed-publish-failure-staging-*"))


def test_seal_cleans_published_destination_when_receipt_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_receipt_failure")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)

    def fail_receipt(*_args, **_kwargs):
        raise OSError("injected receipt publication failure")

    monkeypatch.setattr(module, "_publish_json_file", fail_receipt)
    destination = tmp_path / "sealed-receipt-failure"
    receipt = tmp_path / "receipt-receipt-failure.json"
    with pytest.raises(OSError, match="injected receipt"):
        module.seal_qualification(source, attestation, destination, receipt)
    assert not destination.exists()
    assert not receipt.exists()
    assert not list(destination.parent.glob(".sealed-receipt-failure-staging-*"))


def test_seal_receipt_validator_rejects_tampering(tmp_path: Path) -> None:
    module = load_module(TOOL_PATH, "sc26_ae_seal_receipt_tamper")
    source = make_source(tmp_path)
    attestation = make_attestation(tmp_path, source)
    receipt_path = tmp_path / "receipt.json"
    module.seal_qualification(source, attestation, tmp_path / "sealed", receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "PENDING"
    with pytest.raises(ValueError, match="status"):
        module.validate_receipt(receipt)
