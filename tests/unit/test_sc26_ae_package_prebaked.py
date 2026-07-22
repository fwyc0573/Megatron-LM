import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "SC26-AE" / "tools" / "package_prebaked.py"
ARTIFACT_MODULE_PATH = REPO_ROOT / "SC26-AE" / "tools" / "artifact_manifest.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "integration" / "fixtures" / "sc26_ae_task3_fixture.py"
MODELS = {
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


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def stable_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def source_commits() -> dict[str, str]:
    return {
        "megatron_lm": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "echo_slowdown": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD:Echo-slowdown"], text=True
        ).strip(),
        "megatron_sim_engine": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD:megatron-sim-engine"],
            text=True,
        ).strip(),
    }


def build_functional_fixture(root: Path) -> None:
    fixture = load_module(FIXTURE_PATH, "sc26_ae_functional_fixture")
    fixture.build_functional_source(REPO_ROOT, root)


def create_manifest(artifact_module, root: Path, metadata: dict[str, object]) -> Path:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = artifact_module.create_manifest(root, metadata, files)
    output = root / "artifact_manifest.json"
    stable_json(output, manifest)
    artifact_module.verify_manifest(root, manifest)
    return output


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_marker_manifest_digest(marker_path: Path, manifest_path: Path) -> None:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = marker["manifest_sha256"]
    stable_json(marker_path, marker)


def _functional_producer_commits(seed: str) -> dict[str, str]:
    simulator_seed = {"1": "2", "3": "4", "5": "6", "7": "8", "9": "b"}[seed]
    return {
        "megatron_lm": seed * 40,
        "echo_slowdown": "a" * 40,
        "megatron_sim_engine": simulator_seed * 40,
    }


def set_heterogeneous_functional_provenance(root: Path) -> dict[str, object]:
    """Make each synthetic source artifact carry a distinct producer identity."""

    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, "sc26_ae_artifact_manifest_heterogeneous_fixture"
    )
    task2_commits = _functional_producer_commits("1")
    task2_marker_path = root / "_shared/task2/predictor_marker.json"
    task2_marker = json.loads(task2_marker_path.read_text(encoding="utf-8"))
    task2_root = root / task2_marker["run_path"]
    task2_manifest_path = task2_root / "artifact_manifest.json"
    task2_manifest = json.loads(task2_manifest_path.read_text(encoding="utf-8"))
    task2_manifest["source_commits"] = task2_commits
    stable_json(task2_manifest_path, task2_manifest)
    _rewrite_manifest(artifact_module, task2_root)
    _update_marker_manifest_digest(task2_marker_path, task2_manifest_path)
    task2_manifest = json.loads(task2_manifest_path.read_text(encoding="utf-8"))

    producers: dict[str, object] = {"shared_task2": {"task2": task2_commits}}
    model_settings = {
        "gpt175b": {
            "task1": _functional_producer_commits("3"),
            "task3": _functional_producer_commits("5"),
            "compatibility": {
                "policy": "task_specific_source_compatibility_v2",
                "task1": "task1_consumer_only_reuse",
                "task2": "task2_producer_equivalent_reuse",
            },
        },
        "qwen3_a30b": {
            "task1": _functional_producer_commits("7"),
            "task3": _functional_producer_commits("9"),
            "compatibility": {
                "policy": "exact_or_simulator_only_ancestor_v1",
                "task1": "simulator_only_reuse",
                "task2": "simulator_only_reuse",
            },
        },
    }
    for model, settings in model_settings.items():
        task1_marker_path = root / model / "task1/capture_marker.json"
        task1_marker = json.loads(task1_marker_path.read_text(encoding="utf-8"))
        task1_root = task1_marker_path.parent / task1_marker["run_path"]
        task1_manifest_path = task1_root / "artifact_manifest.json"
        task1_manifest = json.loads(task1_manifest_path.read_text(encoding="utf-8"))
        task1_manifest["source_commits"] = settings["task1"]
        stable_json(task1_manifest_path, task1_manifest)
        _rewrite_manifest(artifact_module, task1_root)
        _update_marker_manifest_digest(task1_marker_path, task1_manifest_path)
        task1_manifest = json.loads(task1_manifest_path.read_text(encoding="utf-8"))

        task3_marker_path = root / model / "task3/run_marker.json"
        task3_marker = json.loads(task3_marker_path.read_text(encoding="utf-8"))
        task3_root = task3_marker_path.parent / task3_marker["run_path"]
        task3_manifest_path = task3_root / "artifact_manifest.json"
        shutil.copy2(task1_manifest_path, task3_root / "provenance/task1_manifest.json")
        shutil.copy2(task2_manifest_path, task3_root / "provenance/task2_manifest.json")
        stable_json(
            task3_root / "provenance/resolved_inputs.json",
            {
                "schema_version": "sc26-ae-task3-resolved-inputs-v1",
                "artifact_source": "fresh",
                "capture_id": task1_manifest["capture_id"],
                "predictor_run_id": task2_manifest["predictor_run_id"],
                "task1_source_commits": settings["task1"],
                "task2_source_commits": task2_commits,
                "source_compatibility": settings["compatibility"],
                "input_expectations": {
                    "task1_manifest": {
                        "sha256": sha256(task1_manifest_path),
                        "size_bytes": task1_manifest_path.stat().st_size,
                    },
                    "task2_manifest": {
                        "sha256": sha256(task2_manifest_path),
                        "size_bytes": task2_manifest_path.stat().st_size,
                    },
                },
            },
        )
        task3_manifest = json.loads(task3_manifest_path.read_text(encoding="utf-8"))
        task3_manifest["source_commits"] = settings["task3"]
        stable_json(task3_manifest_path, task3_manifest)
        _rewrite_manifest(artifact_module, task3_root)
        _update_marker_manifest_digest(task3_marker_path, task3_manifest_path)
        producers[model] = {
            "task1": settings["task1"],
            "task3": settings["task3"],
        }
    return producers


def mutate_functional_task3_resolved_input(
    root: Path, model: str, key: str, value: object
) -> None:
    """Mutate sealed Task3 input metadata while keeping fixture checksums valid."""

    artifact_module = load_module(
        ARTIFACT_MODULE_PATH,
        "sc26_ae_artifact_manifest_resolved_input_{}_{}".format(model, key),
    )
    marker_path = root / model / "task3/run_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    task3_root = marker_path.parent / marker["run_path"]
    resolved_path = task3_root / "provenance/resolved_inputs.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved[key] = value
    stable_json(resolved_path, resolved)
    manifest_path = _rewrite_manifest(artifact_module, task3_root)
    _update_marker_manifest_digest(marker_path, manifest_path)


def set_functional_task1_execution_evidence(
    root: Path, model: str, evidence: str
) -> None:
    """Update Task1 evidence and keep its sealed Task3 provenance synchronized."""

    artifact_module = load_module(
        ARTIFACT_MODULE_PATH,
        "sc26_ae_artifact_manifest_task1_evidence_{}".format(model),
    )
    task1_marker_path = root / model / "task1/capture_marker.json"
    task1_marker = json.loads(task1_marker_path.read_text(encoding="utf-8"))
    task1_root = task1_marker_path.parent / task1_marker["run_path"]
    task1_manifest_path = task1_root / "artifact_manifest.json"
    task1_manifest = json.loads(task1_manifest_path.read_text(encoding="utf-8"))
    task1_manifest["execution_evidence"] = evidence
    stable_json(task1_manifest_path, task1_manifest)
    _rewrite_manifest(artifact_module, task1_root)
    _update_marker_manifest_digest(task1_marker_path, task1_manifest_path)

    task3_marker_path = root / model / "task3/run_marker.json"
    task3_marker = json.loads(task3_marker_path.read_text(encoding="utf-8"))
    task3_root = task3_marker_path.parent / task3_marker["run_path"]
    shutil.copy2(task1_manifest_path, task3_root / "provenance/task1_manifest.json")
    resolved_path = task3_root / "provenance/resolved_inputs.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["input_expectations"]["task1_manifest"] = {
        "sha256": sha256(task1_manifest_path),
        "size_bytes": task1_manifest_path.stat().st_size,
    }
    stable_json(resolved_path, resolved)
    task3_manifest_path = _rewrite_manifest(artifact_module, task3_root)
    _update_marker_manifest_digest(task3_marker_path, task3_manifest_path)


def full_moe_capture_summary() -> dict[str, object]:
    return {
        "capture_scope": "full",
        "selected_rank_ids": list(range(256)),
        "selected_rank_count": 256,
        "trace_file_count": 256,
        "memory_json_count": 256,
    }


def set_task1_promotion_metadata(
    source_root: Path, model: str, *, capture_summary: dict[str, object], evidence: str
) -> None:
    task1_root = source_root / model / "task1" / "runs" / f"{model}-contract-capture"
    manifest_path = task1_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_evidence"] = evidence
    manifest["capture_summary"] = capture_summary
    stable_json(manifest_path, manifest)

    marker_path = source_root / model / "task1" / "capture_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = sha256(manifest_path)
    stable_json(marker_path, marker)


def _rewrite_manifest(artifact_module, root: Path) -> Path:
    """Recompute a fixture manifest after changing its payload or metadata."""

    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = dict(manifest)
    metadata.pop("files", None)
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    rewritten = artifact_module.create_manifest(root, metadata, files)
    stable_json(manifest_path, rewritten)
    artifact_module.verify_manifest(root, rewritten)
    return manifest_path


def functional_capture_summary(model: str) -> dict[str, object]:
    """Return the exact fake-level rank inventory required by the functional path."""

    if model == "gpt175b":
        selected = [0, 128, 256, 384, 512, 640, 768, 896]
        scope = "representative"
    elif model == "qwen3_a30b":
        selected = [pp * 8 * 4 + exp * 8 for pp in range(8) for exp in range(4)]
        scope = "representative_ep"
    else:  # pragma: no cover - callers intentionally use the two functional models
        raise AssertionError(model)
    return {
        "capture_scope": scope,
        "selected_rank_ids": selected,
        "selected_rank_count": len(selected),
        "trace_file_count": len(selected),
        "memory_json_count": len(selected),
    }


def augment_functional_contract_source(
    root: Path, *, models_without_ncu: set[str] | None = None
) -> None:
    """Add the model-local NCU feature and rank-scope metadata to a fixture.

    ``build_contract_source`` remains the legacy three-model fixture used by
    strict packaging tests.  Functional packaging has a narrower contract: the
    two official models each need their own rank-0 NCU CSV and an explicit
    capture inventory.  This helper layers those fields onto only GPT/DSV3 so
    the legacy fixture and tests remain unchanged.
    """

    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, "sc26_ae_artifact_manifest_functional_fixture"
    )
    models_without_ncu = models_without_ncu or set()
    for model in ("gpt175b", "qwen3_a30b"):
        task1_dir = root / model / "task1"
        task1_root = task1_dir / "runs" / f"{model}-contract-capture"
        if model not in models_without_ncu:
            write_text(
                task1_root / "ncu/kernel_metric_output.csv",
                "Kernel Name,SM,rank_scope\nfixture_kernel,1,global_rank_0\n",
            )
        manifest_path = task1_root / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["capture_summary"] = functional_capture_summary(model)
        manifest["ncu_feature_provenance"] = {
            "rank_scope": "global_rank_0",
            "rank_ids": [0],
            "physical_gpu_count": 1,
            "missing_kernel_count": 0,
        }
        stable_json(manifest_path, manifest)
        _rewrite_manifest(artifact_module, task1_root)

        marker_path = task1_dir / "capture_marker.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["manifest_sha256"] = sha256(manifest_path)
        marker["artifact_manifest_sha256"] = sha256(manifest_path)
        stable_json(marker_path, marker)

        task2_manifest_path = (
            root
            / "_shared/task2/runs/predictor-contract-001/artifact_manifest.json"
        )
        task2_manifest = json.loads(task2_manifest_path.read_text(encoding="utf-8"))
        task3_dir = root / model / "task3"
        task3_root = task3_dir / "runs" / f"{model}-contract-task3"
        shutil.copy2(manifest_path, task3_root / "provenance/task1_manifest.json")
        shutil.copy2(
            task2_manifest_path, task3_root / "provenance/task2_manifest.json"
        )
        stable_json(
            task3_root / "provenance/resolved_inputs.json",
            {
                "schema_version": "sc26-ae-task3-resolved-inputs-v1",
                "artifact_source": "fresh",
                "capture_id": manifest["capture_id"],
                "predictor_run_id": task2_manifest["predictor_run_id"],
                "task1_source_commits": manifest["source_commits"],
                "task2_source_commits": task2_manifest["source_commits"],
                "source_compatibility": {
                    "policy": "task_specific_source_compatibility_v2",
                    "task1": "exact",
                    "task2": "exact",
                },
                "input_expectations": {
                    "task1_manifest": {
                        "sha256": sha256(manifest_path),
                        "size_bytes": manifest_path.stat().st_size,
                    },
                    "task2_manifest": {
                        "sha256": sha256(task2_manifest_path),
                        "size_bytes": task2_manifest_path.stat().st_size,
                    },
                },
            },
        )
        task3_manifest_path = _rewrite_manifest(artifact_module, task3_root)
        _update_marker_manifest_digest(
            task3_dir / "run_marker.json", task3_manifest_path
        )


@contextmanager
def allow_synthetic_contract_evidence(module):
    """Permit synthetic labels only while exercising packaging mechanics in-process.

    The production CLI remains strict.  This seam lets the structural packaging
    tests use honest synthetic evidence labels without ever presenting them as a
    real qualification result.
    """

    task1_synthetic = "local_synthetic_not_gpu_qualification"
    task2_synthetic = "local_synthetic_not_two_gpu_qualification"

    def require_evidence(manifest, expected, label):
        observed = manifest.get("execution_evidence")
        allowed = {
            "real_single_h800_qualified": {"real_single_h800_qualified", task1_synthetic},
            "real_exact_two_h800_qualified": {
                "real_exact_two_h800_qualified",
                task2_synthetic,
            },
        }[expected]
        if observed not in allowed:
            raise ValueError(
                "{} requires execution_evidence={}, observed {}".format(
                    label, expected, observed
                )
            )

    def require_distribution(manifest):
        observed = manifest.get("execution_evidence")
        if observed not in {"real_prebaked_qualified", "local_synthetic_not_gpu_qualification"}:
            raise ValueError(
                "distribution execution evidence is invalid: {}".format(observed)
            )

    with patch.object(module, "_require_exact_evidence", require_evidence), patch.object(
        module, "_require_release_distribution_evidence", require_distribution
    ):
        yield


def build_contract_source(
    root: Path,
    task2_evidence: str = "local_synthetic_not_two_gpu_qualification",
) -> None:
    """Create an explicitly synthetic source tree for contract-only tests.

    This helper must never emit real qualification evidence.  Real producer
    manifests are supplied only by the external H800 qualification workflow.
    """

    artifact_module = load_module(ARTIFACT_MODULE_PATH, "sc26_ae_test_artifact_manifest")
    commits = source_commits()
    predictor_run_id = "predictor-contract-001"
    task2_root = root / "_shared/task2/runs" / predictor_run_id
    write_text(task2_root / "merge/input/kernel_metric_output.csv", "Kernel Name,SM\nqualified,1\n")
    stable_json(task2_root / "training_testing/output/xgb_model.json", {"model": "fixture"})
    stable_json(task2_root / "training_testing/output/standard_scaler.json", {"scale": [1.0]})
    stable_json(
        task2_root / "metrics.json",
        {
            "schema_version": "sc26-ae-echo-metrics-v1",
            "predictor_run_id": predictor_run_id,
            "dataset_row_count": 2,
        },
    )
    task2_manifest = create_manifest(
        artifact_module,
        task2_root,
        {
            "schema_version": "sc26-ae-artifact-manifest-v1",
            "model": "shared_task2",
            "task": "task2",
            "artifact_source": "fresh",
            "predictor_run_id": predictor_run_id,
            "source_commits": commits,
            "execution_evidence": task2_evidence,
        },
    )
    stable_json(
        root / "_shared/task2/predictor_marker.json",
        {
            "schema_version": "sc26-ae-task2-shared-pointer-v1",
            "predictor_run_id": predictor_run_id,
            "run_path": f"_shared/task2/runs/{predictor_run_id}",
            "manifest_sha256": sha256(task2_manifest),
            "artifact_manifest_sha256": sha256(task2_manifest),
            "verified": True,
        },
    )

    trace_text = (
        "rank:0:forward_step(stage_id=0,batch_id=0,mg_state=steady,duration=4.0,timestamp=1.0,cmd_uid=fwd-0)\n"
        "rank:0:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=6.0,timestamp=5.0,cmd_uid=bwd-0)\n"
        "rank:0:ddp_grad_comm(stage_id=0,batch_id=0,mg_state=steady,duration=0.5,timestamp=6.0,trigger_cmd_uid=bwd-0)\n"
        "rank:0:optimizer_step(stage_id=0,batch_id=0,mg_state=finalize,duration=1.5,timestamp=11.0,cmd_uid=opt-0)\n"
    )
    for model, specification in MODELS.items():
        capture_id = f"{model}-contract-capture"
        task1_dir = root / model / "task1"
        task1_root = task1_dir / "runs" / capture_id
        write_text(task1_root / "runtime/profiler_log/config/rank0.txt", trace_text)
        write_text(task1_root / f"nsys/{model}.sqlite", "synthetic sqlite fixture\n")
        task1_manifest = create_manifest(
            artifact_module,
            task1_root,
            {
                "schema_version": "sc26-ae-artifact-manifest-v1",
                "model": model,
                "task": "task1",
                "artifact_source": "fresh",
                "capture_id": capture_id,
                "source_commits": commits,
                "simulation_topology": specification["topology"],
                "capture_runtime": {
                    "physical_gpu_count": 1,
                    "fake_gpus_per_node": 8,
                    "scaling_min_warmup_iters": 3,
                    "scaling_profile_iters": 1,
                },
                "profile": specification["profile"],
                "precision": "bf16",
                "mock_data": True,
                "ddp_overlap": True,
                "execution_evidence": "local_synthetic_not_gpu_qualification",
            },
        )
        stable_json(
            task1_dir / "capture_marker.json",
            {
                "schema_version": "sc26-ae-task1-capture-marker-v1",
                "model": model,
                "capture_id": capture_id,
                "run_path": f"runs/{capture_id}",
                "manifest_sha256": sha256(task1_manifest),
                "artifact_manifest_sha256": sha256(task1_manifest),
                "verified": True,
            },
        )

        simulation_run_id = f"{model}-contract-task3"
        task3_dir = root / model / "task3"
        task3_root = task3_dir / "runs" / simulation_run_id
        stable_json(task3_root / "slowdown_assets/manifest.json", {"cmd_uids": ["bwd-0"]})
        stable_json(task3_root / "slowdown_assets/kernel_features.json", {"bwd-0": {"x": 1}})
        stable_json(
            task3_root / "slowdown_assets/backward_kernel_blueprints.json",
            {"bwd-0": ["kernel"]},
        )
        for stage in range(specification["topology"]["pp"]):
            write_text(
                task3_root / f"schedule/stage{stage}_scheduling_plan.txt",
                "dtype=torch.bfloat16\n",
            )
        stable_json(task3_root / "report.json", {"rank_id": 0, "rank0_step_time_ms": 1.0})
        write_text(task3_root / "report.md", "# Synthetic contract fixture\n")
        stable_json(
            task3_root / "provenance/input_evidence.json",
            {
                "schema_version": "sc26-ae-task3-input-evidence-v1",
                "model": model,
                "artifact_source": "fresh",
                "capture_id": capture_id,
                "predictor_run_id": predictor_run_id,
            },
        )
        shutil.copy2(task1_manifest, task3_root / "provenance/task1_manifest.json")
        shutil.copy2(task2_manifest, task3_root / "provenance/task2_manifest.json")
        stable_json(
            task3_root / "provenance/resolved_inputs.json",
            {
                "schema_version": "sc26-ae-task3-resolved-inputs-v1",
                "artifact_source": "fresh",
                "capture_id": capture_id,
                "predictor_run_id": predictor_run_id,
                "task1_source_commits": commits,
                "task2_source_commits": commits,
                "source_compatibility": {
                    "policy": "task_specific_source_compatibility_v2",
                    "task1": "exact",
                    "task2": "exact",
                },
                "input_expectations": {
                    "task1_manifest": {
                        "sha256": sha256(task1_manifest),
                        "size_bytes": task1_manifest.stat().st_size,
                    },
                    "task2_manifest": {
                        "sha256": sha256(task2_manifest),
                        "size_bytes": task2_manifest.stat().st_size,
                    },
                },
            },
        )
        task3_manifest = create_manifest(
            artifact_module,
            task3_root,
            {
                "schema_version": "sc26-ae-artifact-manifest-v1",
                "model": model,
                "task": "task3",
                "artifact_source": "fresh",
                "capture_id": capture_id,
                "predictor_run_id": predictor_run_id,
                "simulation_run_id": simulation_run_id,
                "source_commits": commits,
                "simulation_topology": specification["topology"],
                "profile": specification["profile"],
                "precision": "bf16",
                "ddp_overlap": True,
                "communication_backend": "analytical",
                "overlap_mode": "on",
                "database_is_trace_dir": True,
                "execution_evidence": "local_synthetic_not_gpu_qualification",
            },
        )
        stable_json(
            task3_dir / "run_marker.json",
            {
                "schema_version": "sc26-ae-task3-run-marker-v1",
                "task": "task3",
                "model": model,
                "artifact_source": "fresh",
                "simulation_run_id": simulation_run_id,
                "capture_id": capture_id,
                "predictor_run_id": predictor_run_id,
                "run_path": f"runs/{simulation_run_id}",
                "manifest_sha256": sha256(task3_manifest),
                "artifact_manifest_sha256": sha256(task3_manifest),
                "execution_evidence": "local_synthetic_not_gpu_qualification",
                "verified": True,
            },
        )


def test_build_verify_and_relocate_complete_distribution(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    result_path = tmp_path / "work" / "package_result.json"

    with allow_synthetic_contract_evidence(module):
        result = module.build_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=staging_root,
            distribution_id="contract-fixture-001",
            result_json=result_path,
        )

    assert result["distribution_medium"] == "regular_git"
    assert result["bundle_count"] == 4
    assert result["distribution_file_count"] > 20
    assert result["total_size_bytes"] > result["distribution_manifest"]["size_bytes"] > 0
    distribution = json.loads((staging_root / "distribution_manifest.json").read_text())
    assert distribution["execution_evidence"] == "local_synthetic_not_gpu_qualification"
    assert "real_prebaked_qualified" not in json.dumps(distribution, sort_keys=True)
    assert set(distribution["bundles"]) == set(MODELS) | {"shared_task2"}
    assert set(distribution["file_size_mib"]) == {
        entry["path"] for entry in distribution["files"]
    }
    for model in MODELS:
        entry = distribution["bundles"][model]
        assert entry["capture_id"] != entry["predictor_run_id"]
        assert entry["predictor_run_id"] == "predictor-contract-001"
        assert entry["source_task1_manifest_sha256"] == sha256(
            source_root
            / model
            / "task1/runs"
            / entry["capture_id"]
            / "artifact_manifest.json"
        )
        assert (staging_root / entry["root"] / "configuration.json").is_file()

    relocated = tmp_path / "relocated" / "deep" / "prebaked"
    relocated.parent.mkdir(parents=True)
    shutil.copytree(staging_root, relocated)
    with allow_synthetic_contract_evidence(module):
        verified = module.verify_distribution(REPO_ROOT, relocated)
    assert verified["distribution_id"] == "contract-fixture-001"
    assert verified["predictor_run_id"] == "predictor-contract-001"
    assert str(source_root) not in json.dumps(distribution, sort_keys=True)


def test_build_and_verify_functional_two_model_distribution(tmp_path: Path) -> None:
    """Fake-level packaging accepts only GPT/Qwen3 and keeps non-release evidence."""

    module = load_module(MODULE_PATH, "sc26_ae_package_functional_positive")
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    staging_root = tmp_path / "staging" / "functional"
    result = module.build_functional_distribution(
        repo_root=REPO_ROOT,
        output_root=source_root,
        staging_root=staging_root,
        distribution_id="functional-fixture-001",
        result_json=tmp_path / "result.json",
    )

    assert result["bundle_count"] == 3
    assert result["distribution_medium"] == "regular_git"
    distribution = json.loads((staging_root / "distribution_manifest.json").read_text())
    assert distribution["schema_version"] == module.FUNCTIONAL_DISTRIBUTION_SCHEMA
    assert distribution["execution_evidence"] == module.FUNCTIONAL_DISTRIBUTION_EVIDENCE
    assert set(distribution["bundles"]) == {"gpt175b", "qwen3_a30b", "shared_task2"}
    for model in ("gpt175b", "qwen3_a30b"):
        entry = distribution["bundles"][model]
        manifest = json.loads((staging_root / entry["manifest"]).read_text())
        paths = {item["path"] for item in manifest["files"]}
        assert "ncu/kernel_metric_output.csv" in paths
    verified = module.verify_functional_distribution(REPO_ROOT, staging_root)
    assert verified["bundle_count"] == 3
    with pytest.raises(ValueError, match="distribution manifest schema"):
        module.verify_distribution(REPO_ROOT, staging_root)


def test_functional_distribution_preserves_heterogeneous_source_producers(
    tmp_path: Path,
) -> None:
    """Functional packaging must preserve the producers Task3 actually consumed."""

    module = load_module(MODULE_PATH, "sc26_ae_package_functional_heterogeneous")
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    source_producers = set_heterogeneous_functional_provenance(source_root)
    staging_root = tmp_path / "staging" / "functional"

    module.build_functional_distribution(
        repo_root=REPO_ROOT,
        output_root=source_root,
        staging_root=staging_root,
        distribution_id="functional-heterogeneous-001",
        result_json=tmp_path / "work" / "result.json",
    )

    distribution = json.loads(
        (staging_root / "distribution_manifest.json").read_text(encoding="utf-8")
    )
    bundle_producer = source_commits()
    for key, source_artifacts in source_producers.items():
        assert distribution["producer_commits"][key] == {
            "bundle": bundle_producer,
            **source_artifacts,
        }
        entry = distribution["bundles"][key]
        bundle_root = staging_root / entry["root"]
        bundle_manifest = json.loads(
            (bundle_root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        assert bundle_manifest["source_commits"] == bundle_producer
        assert bundle_manifest["source_artifact_commits"] == source_artifacts
        assert entry["source_artifact_commits"] == source_artifacts
    assert (
        staging_root
        / "bundles/gpt175b/provenance/source_task3_resolved_inputs.json"
    ).is_file()
    assert (
        staging_root
        / "bundles/qwen3_a30b/provenance/source_task3_resolved_inputs.json"
    ).is_file()
    assert module.verify_functional_distribution(REPO_ROOT, staging_root)[
        "distribution_id"
    ] == "functional-heterogeneous-001"


def test_functional_distribution_accepts_pending_qwen_representative_scope(
    tmp_path: Path,
) -> None:
    """Functional packaging uses the 32-rank Qwen contract, not release promotion."""

    module = load_module(MODULE_PATH, "sc26_ae_package_functional_pending_qwen")
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    set_functional_task1_execution_evidence(
        source_root,
        "qwen3_a30b",
        "runtime_measurement_requires_external_single_gpu_qualification",
    )

    result = module.build_functional_distribution(
        repo_root=REPO_ROOT,
        output_root=source_root,
        staging_root=tmp_path / "staging" / "functional",
        distribution_id="functional-pending-qwen-001",
        result_json=tmp_path / "work" / "result.json",
    )

    assert result["bundle_count"] == 3


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "task1_source_commits",
            _functional_producer_commits("3"),
            "resolved Task1 commits differ",
        ),
        (
            "task2_source_commits",
            _functional_producer_commits("5"),
            "resolved Task2 commits differ",
        ),
        (
            "source_compatibility",
            {
                "policy": "unsupported_descendant_reuse",
                "task1": "exact",
                "task2": "exact",
            },
            "source compatibility policy is unsupported",
        ),
    ],
)
def test_functional_distribution_rejects_unsealed_task3_provenance(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    """Functional packaging rejects Task3 inputs outside the sealed contract."""

    module = load_module(
        MODULE_PATH, "sc26_ae_package_functional_resolved_input_{}".format(key)
    )
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    mutate_functional_task3_resolved_input(source_root, "gpt175b", key, value)

    with pytest.raises(ValueError, match=message):
        module.build_functional_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=tmp_path / "staging" / "functional",
            distribution_id="functional-invalid-provenance-001",
            result_json=tmp_path / "work" / "result.json",
        )


def test_functional_verifier_rejects_tampered_nested_source_producer(
    tmp_path: Path,
) -> None:
    """Offline verification rejects distribution-level producer substitution."""

    module = load_module(MODULE_PATH, "sc26_ae_package_functional_producer_tamper")
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    staging_root = tmp_path / "staging" / "functional"
    module.build_functional_distribution(
        repo_root=REPO_ROOT,
        output_root=source_root,
        staging_root=staging_root,
        distribution_id="functional-producer-tamper-001",
        result_json=tmp_path / "work" / "result.json",
    )
    distribution_path = staging_root / "distribution_manifest.json"
    distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
    distribution["producer_commits"]["gpt175b"]["task1"] = (
        _functional_producer_commits("7")
    )
    stable_json(distribution_path, distribution)

    with pytest.raises(ValueError, match="bundle source producers differ"):
        module.verify_functional_distribution(REPO_ROOT, staging_root)


def test_functional_model_scope_replaces_deepseek_with_qwen3() -> None:
    """The active functional bundle follows the Qwen3 replacement decision."""

    module = load_module(MODULE_PATH, "sc26_ae_package_qwen3_scope")
    assert module.FUNCTIONAL_MODELS == {"gpt175b", "qwen3_a30b"}


def test_functional_distribution_rejects_missing_model_ncu_feature(tmp_path: Path) -> None:
    """A model may not borrow the shared Task2 NCU CSV as a fallback."""

    module = load_module(MODULE_PATH, "sc26_ae_package_functional_missing_ncu")
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    ncu_path = (
        source_root
        / "gpt175b/task1/runs/synthetic-gpt175b-functional-capture/ncu/kernel_metric_output.csv"
    )
    ncu_path.unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="missing|verified manifest|NCU|ncu"):
        module.build_functional_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=tmp_path / "staging" / "functional",
            distribution_id="functional-missing-ncu-001",
            result_json=tmp_path / "result.json",
        )


def test_functional_distribution_rejects_qwen3_incomplete_rank_subset(tmp_path: Path) -> None:
    """Qwen3 functional packaging requires all PP×EP representative ranks."""

    module = load_module(MODULE_PATH, "sc26_ae_package_functional_rank_gate")
    source_root = tmp_path / "functional-source"
    build_functional_fixture(source_root)
    task1_root = source_root / "qwen3_a30b/task1/runs/synthetic-qwen3_a30b-functional-capture"
    manifest_path = task1_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["capture_summary"] = {
        "capture_scope": "representative_ep",
        "selected_rank_ids": [0, 8, 16, 24],
        "selected_rank_count": 4,
        "trace_file_count": 4,
        "memory_json_count": 4,
    }
    stable_json(manifest_path, manifest)
    marker_path = source_root / "qwen3_a30b/task1/capture_marker.json"
    marker = json.loads(marker_path.read_text())
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = marker["manifest_sha256"]
    stable_json(marker_path, marker)
    with pytest.raises(ValueError, match="rank inventory"):
        module.build_functional_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=tmp_path / "staging" / "functional",
            distribution_id="functional-rank-gate-001",
            result_json=tmp_path / "result.json",
        )


def test_build_and_verify_functional_distribution_uses_two_models_and_rank0_ncu(
    tmp_path: Path,
) -> None:
    """Functional prebaking seals the fake-level two-model contract only."""

    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_functional")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    augment_functional_contract_source(source_root)
    staging_root = tmp_path / "staging" / "functional"
    result_path = tmp_path / "work" / "functional_result.json"

    result = module.build_functional_distribution(
        repo_root=REPO_ROOT,
        output_root=source_root,
        staging_root=staging_root,
        distribution_id="functional-fixture-001",
        result_json=result_path,
    )

    assert result["bundle_count"] == 3
    assert result["predictor_run_id"] == "predictor-contract-001"
    distribution = json.loads(
        (staging_root / "distribution_manifest.json").read_text(encoding="utf-8")
    )
    assert distribution["schema_version"] == module.FUNCTIONAL_DISTRIBUTION_SCHEMA
    assert distribution["execution_evidence"] == module.FUNCTIONAL_DISTRIBUTION_EVIDENCE
    assert set(distribution["bundles"]) == {"gpt175b", "qwen3_a30b", "shared_task2"}

    verified = module.verify_functional_distribution(REPO_ROOT, staging_root)
    assert verified["distribution_id"] == "functional-fixture-001"
    assert verified["bundle_count"] == 3

    # The release verifier intentionally does not accept the functional schema.
    with pytest.raises(ValueError, match="distribution manifest schema"):
        module.verify_distribution(REPO_ROOT, staging_root)

    for model in ("gpt175b", "qwen3_a30b"):
        entry = distribution["bundles"][model]
        model_root = staging_root / entry["root"]
        files = {
            item["path"]
            for item in json.loads(
                (model_root / "artifact_manifest.json").read_text(encoding="utf-8")
            )["files"]
        }
        assert "ncu/kernel_metric_output.csv" in files
        source_manifest = json.loads(
            (model_root / "provenance/source_task1_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert source_manifest["ncu_feature_provenance"] == {
            "rank_scope": "global_rank_0",
            "rank_ids": [0],
            "physical_gpu_count": 1,
            "missing_kernel_count": 0,
        }
        assert source_manifest["capture_summary"] == functional_capture_summary(model)


@pytest.mark.parametrize("model", ["gpt175b", "qwen3_a30b"])
def test_functional_build_rejects_invalid_rank_inventory(
    tmp_path: Path, model: str
) -> None:
    """The functional path must enforce model-specific Task1 rank vectors."""

    module = load_module(
        MODULE_PATH, f"sc26_ae_package_prebaked_functional_rank_{model}"
    )
    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, f"sc26_ae_artifact_manifest_functional_rank_{model}"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    augment_functional_contract_source(source_root)

    task1_root = source_root / model / "task1" / "runs" / f"{model}-contract-capture"
    manifest_path = task1_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = functional_capture_summary(model)
    summary["selected_rank_ids"] = [0]
    summary["selected_rank_count"] = 1
    summary["trace_file_count"] = 1
    summary["memory_json_count"] = 1
    manifest["capture_summary"] = summary
    stable_json(manifest_path, manifest)
    _rewrite_manifest(artifact_module, task1_root)
    marker_path = source_root / model / "task1" / "capture_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = sha256(manifest_path)
    stable_json(marker_path, marker)

    with pytest.raises(ValueError, match="rank inventory"):
        module.build_functional_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=tmp_path / "staging" / "functional-invalid-rank",
            distribution_id=f"functional-invalid-rank-{model}",
            result_json=tmp_path / "work" / "result.json",
        )


def test_functional_build_requires_model_local_ncu_csv(tmp_path: Path) -> None:
    """A shared Task2 CSV cannot satisfy the per-model Task1 NCU requirement."""

    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_functional_ncu")
    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, "sc26_ae_artifact_manifest_functional_ncu"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    augment_functional_contract_source(source_root, models_without_ncu={"gpt175b"})

    model = "gpt175b"
    task1_root = source_root / model / "task1" / "runs" / f"{model}-contract-capture"
    _rewrite_manifest(artifact_module, task1_root)
    manifest_path = task1_root / "artifact_manifest.json"
    marker_path = source_root / model / "task1" / "capture_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = sha256(manifest_path)
    stable_json(marker_path, marker)

    with pytest.raises(ValueError, match="missing ncu/kernel_metric_output.csv"):
        module.build_functional_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=tmp_path / "staging" / "functional-missing-ncu",
            distribution_id="functional-missing-ncu",
            result_json=tmp_path / "work" / "result.json",
        )


def test_functional_verifier_rejects_extra_deepseek_bundle(tmp_path: Path) -> None:
    """The functional distribution is intentionally limited to GPT and Qwen3."""

    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_functional_extra_model")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    augment_functional_contract_source(source_root)
    staging_root = tmp_path / "staging" / "functional-extra-model"
    module.build_functional_distribution(
        repo_root=REPO_ROOT,
        output_root=source_root,
        staging_root=staging_root,
        distribution_id="functional-extra-model",
        result_json=tmp_path / "work" / "result.json",
    )

    distribution_path = staging_root / "distribution_manifest.json"
    distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
    distribution["bundles"]["dsv3"] = dict(distribution["bundles"]["gpt175b"])
    stable_json(distribution_path, distribution)
    with pytest.raises(ValueError, match="bundles must contain gpt175b, qwen3_a30b, and shared_task2"):
        module.verify_functional_distribution(REPO_ROOT, staging_root)


def test_functional_build_rejects_non_rank0_ncu_provenance(tmp_path: Path) -> None:
    """Functional model NCU features must be collected only for global rank 0."""

    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_functional_ncu_scope")
    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, "sc26_ae_artifact_manifest_functional_ncu_scope"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    augment_functional_contract_source(source_root)

    model = "gpt175b"
    task1_root = source_root / model / "task1" / "runs" / f"{model}-contract-capture"
    manifest_path = task1_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ncu_feature_provenance"]["rank_ids"] = [1]
    stable_json(manifest_path, manifest)
    _rewrite_manifest(artifact_module, task1_root)
    marker_path = source_root / model / "task1" / "capture_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = sha256(manifest_path)
    stable_json(marker_path, marker)

    with pytest.raises(ValueError, match="NCU scope"):
        module.build_functional_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=tmp_path / "staging" / "functional-invalid-ncu-scope",
            distribution_id="functional-invalid-ncu-scope",
            result_json=tmp_path / "work" / "result.json",
        )


@pytest.mark.parametrize("model", ["qwen3_a30b", "dsv3"])
def test_real_moe_task1_promotion_rejects_quick_rank_subset(
    tmp_path: Path, model: str
) -> None:
    """A QUICK MoE capture must not cross the real qualification boundary."""

    package_module = load_module(
        MODULE_PATH, f"sc26_ae_package_prebaked_rank_gate_{model}"
    )
    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, f"sc26_ae_artifact_manifest_rank_gate_{model}"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    task1_root = source_root / model / "task1" / "runs" / f"{model}-contract-capture"
    manifest_path = task1_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_evidence"] = "real_single_h800_qualified"
    manifest["capture_summary"] = {
        "capture_scope": "quick",
        "selected_rank_ids": [0, 64, 128, 192],
        "selected_rank_count": 4,
        "trace_file_count": 4,
        "memory_json_count": 4,
    }
    stable_json(manifest_path, manifest)
    marker_path = source_root / model / "task1" / "capture_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = sha256(manifest_path)
    stable_json(marker_path, marker)

    # The fixture intentionally bypasses the unrelated evidence-class check so
    # this test reaches the rank-scope promotion boundary under examination.
    with patch.object(package_module, "_require_exact_evidence", lambda *_args: None):
        with pytest.raises(
            ValueError,
            match="capture_scope|full-rank|rank inventory|selected_rank",
        ):
            package_module._source_manifest_and_run(
                artifact_module,
                source_root,
                model,
                source_commits(),
            )


@pytest.mark.parametrize("model", ["qwen3_a30b", "dsv3"])
def test_real_moe_task1_promotion_accepts_exact_full_rank_inventory(
    tmp_path: Path, model: str
) -> None:
    """A complete MoE inventory is allowed to reach the packaging boundary."""

    package_module = load_module(
        MODULE_PATH, f"sc26_ae_package_prebaked_full_rank_{model}"
    )
    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, f"sc26_ae_artifact_manifest_full_rank_{model}"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    set_task1_promotion_metadata(
        source_root,
        model,
        capture_summary=full_moe_capture_summary(),
        evidence="real_single_h800_qualified",
    )

    # The fixture's remaining Task1/Task3 evidence is synthetic and is
    # intentionally bypassed; this test isolates the exact MoE rank gate.
    with patch.object(package_module, "_require_exact_evidence", lambda *_args: None):
        task1_root, manifest, task3_root, task3_manifest, *_ = (
            package_module._source_manifest_and_run(
                artifact_module,
                source_root,
                model,
                source_commits(),
            )
        )

    assert task1_root.is_dir()
    assert task3_root.is_dir()
    assert manifest["capture_summary"]["capture_scope"] == "full"
    assert manifest["capture_summary"]["selected_rank_ids"] == list(range(256))
    assert manifest["capture_summary"]["selected_rank_count"] == 256
    assert manifest["capture_summary"]["trace_file_count"] == 256
    assert manifest["capture_summary"]["memory_json_count"] == 256
    assert task3_manifest["model"] == model


def test_build_records_exact_final_distribution_size(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_size")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"

    with allow_synthetic_contract_evidence(module):
        result = module.build_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=staging_root,
            distribution_id="contract-fixture-size-001",
            result_json=tmp_path / "work" / "package_result.json",
        )

    actual_total = sum(
        path.stat().st_size
        for path in staging_root.rglob("*")
        if path.is_file()
    )
    distribution = json.loads(
        (staging_root / "distribution_manifest.json").read_text(encoding="utf-8")
    )
    assert distribution["total_size_bytes"] == actual_total
    assert result["total_size_bytes"] == actual_total


def test_verify_rejects_distribution_with_stale_total_size(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_stale_size")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"

    with allow_synthetic_contract_evidence(module):
        module.build_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=staging_root,
            distribution_id="contract-fixture-stale-size-001",
            result_json=tmp_path / "work" / "package_result.json",
        )
        distribution_path = staging_root / "distribution_manifest.json"
        distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
        distribution["total_size_bytes"] += 1
        stable_json(distribution_path, distribution)

        with pytest.raises(ValueError, match="total_size_bytes"):
            module.verify_distribution(REPO_ROOT, staging_root)


def test_build_fails_if_distribution_summary_does_not_converge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_nonconvergent")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    calls = 0

    def alternating_medium(_root: Path) -> str:
        nonlocal calls
        calls += 1
        return "regular_git" if calls % 2 else "github_release"

    artifact_module = load_module(
        ARTIFACT_MODULE_PATH, "sc26_ae_test_artifact_manifest_nonconvergent"
    )
    monkeypatch.setattr(module, "_load_artifact_module", lambda _root: artifact_module)
    monkeypatch.setattr(artifact_module, "evaluate_distribution_gate", alternating_medium)
    with allow_synthetic_contract_evidence(module):
        with pytest.raises(ValueError, match="did not converge"):
            module.build_distribution(
                repo_root=REPO_ROOT,
                output_root=source_root,
                staging_root=staging_root,
                distribution_id="contract-fixture-nonconvergent-001",
                result_json=tmp_path / "work" / "package_result.json",
            )


def test_build_rejects_nonqualified_task2_and_does_not_publish_manifest(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_reject")
    source_root = tmp_path / "fresh-output"
    build_contract_source(
        source_root,
        task2_evidence="runtime_measurement_requires_external_two_gpu_qualification",
    )
    staging_root = tmp_path / "staging" / "prebaked"

    with pytest.raises(ValueError, match="real_exact_two_h800_qualified"):
        module.build_distribution(
            repo_root=REPO_ROOT,
            output_root=source_root,
            staging_root=staging_root,
            distribution_id="must-fail-001",
            result_json=tmp_path / "result.json",
        )
    assert not (staging_root / "distribution_manifest.json").exists()


def test_build_rejects_task3_without_execution_evidence(tmp_path: Path) -> None:
    """Task3 evidence must be consumed independently of Task1 evidence."""

    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_task3_evidence")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    for model in MODELS:
        task3_dir = source_root / model / "task3"
        task3_root = task3_dir / "runs" / f"{model}-contract-task3"
        task3_manifest_path = task3_root / "artifact_manifest.json"
        task3_manifest = json.loads(task3_manifest_path.read_text(encoding="utf-8"))
        task3_manifest.pop("execution_evidence")
        stable_json(task3_manifest_path, task3_manifest)
        marker_path = task3_dir / "run_marker.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.pop("execution_evidence")
        marker["manifest_sha256"] = sha256(task3_manifest_path)
        marker["artifact_manifest_sha256"] = sha256(task3_manifest_path)
        stable_json(marker_path, marker)

    with allow_synthetic_contract_evidence(module):
        with pytest.raises(ValueError, match=r"Task3.*execution[_ ]evidence"):
            module.build_distribution(
                repo_root=REPO_ROOT,
                output_root=source_root,
                staging_root=staging_root,
                distribution_id="task3-evidence-required-001",
                result_json=tmp_path / "result.json",
            )
    assert not (staging_root / "distribution_manifest.json").exists()


@pytest.mark.parametrize(
    "field",
    ["simulation_run_id", "capture_id", "predictor_run_id"],
)
def test_build_rejects_task3_marker_manifest_identity_split_brain(
    tmp_path: Path, field: str
) -> None:
    module = load_module(
        MODULE_PATH, f"sc26_ae_package_prebaked_task3_marker_identity_{field}"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    marker_path = source_root / "gpt175b/task3/run_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = f"split-brain-{field}"
    stable_json(marker_path, marker)

    with allow_synthetic_contract_evidence(module):
        with pytest.raises(ValueError, match=field):
            module.build_distribution(
                repo_root=REPO_ROOT,
                output_root=source_root,
                staging_root=staging_root,
                distribution_id=f"task3-marker-identity-{field}",
                result_json=tmp_path / "result.json",
            )
    assert not (staging_root / "distribution_manifest.json").exists()


def test_build_rejects_task3_manifest_simulation_run_id_split_brain(
    tmp_path: Path,
) -> None:
    module = load_module(
        MODULE_PATH, "sc26_ae_package_prebaked_task3_manifest_simulation_identity"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    manifest_path = (
        source_root
        / "gpt175b/task3/runs/gpt175b-contract-task3/artifact_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["simulation_run_id"] = "split-brain-manifest-simulation-run-id"
    stable_json(manifest_path, manifest)
    marker_path = source_root / "gpt175b/task3/run_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256(manifest_path)
    marker["artifact_manifest_sha256"] = sha256(manifest_path)
    stable_json(marker_path, marker)

    with allow_synthetic_contract_evidence(module):
        with pytest.raises(ValueError, match="simulation_run_id"):
            module.build_distribution(
                repo_root=REPO_ROOT,
                output_root=source_root,
                staging_root=staging_root,
                distribution_id="task3-manifest-simulation-identity",
                result_json=tmp_path / "result.json",
            )
    assert not (staging_root / "distribution_manifest.json").exists()


def test_build_rejects_unsafe_task3_marker_simulation_run_id(tmp_path: Path) -> None:
    module = load_module(
        MODULE_PATH, "sc26_ae_package_prebaked_task3_unsafe_simulation_identity"
    )
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    marker_path = source_root / "gpt175b/task3/run_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["simulation_run_id"] = "../unsafe-simulation-run-id"
    stable_json(marker_path, marker)

    with allow_synthetic_contract_evidence(module):
        with pytest.raises(ValueError, match="path-free identifier"):
            module.build_distribution(
                repo_root=REPO_ROOT,
                output_root=source_root,
                staging_root=staging_root,
                distribution_id="task3-unsafe-simulation-identity",
                result_json=tmp_path / "result.json",
            )
    assert not (staging_root / "distribution_manifest.json").exists()


def test_build_rejects_existing_destination(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_existing")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "existing"
    staging_root.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        with allow_synthetic_contract_evidence(module):
            module.build_distribution(
                repo_root=REPO_ROOT,
                output_root=source_root,
                staging_root=staging_root,
                distribution_id="existing-001",
                result_json=tmp_path / "result.json",
            )


def test_copy_tree_rejects_symlink(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_symlink")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    write_text(source / "payload.txt", "payload\n")
    (source / "link.txt").symlink_to(source / "payload.txt")
    with pytest.raises(ValueError, match="symlink"):
        module.copy_regular_tree(source, destination)


def test_build_rejects_selected_payload_drift_after_manifest_verification(
    tmp_path: Path,
) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_selected_drift")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    original_copy_selected = module._copy_selected
    mutated_path = None

    def mutate_before_selected_copy(source, destination, relative_paths, *args, **kwargs):
        nonlocal mutated_path
        paths = list(relative_paths)
        candidate = Path(source) / "runtime/profiler_log/config/rank0.txt"
        if mutated_path is None and candidate.is_file():
            candidate.write_text(
                candidate.read_text(encoding="utf-8")
                + "coherent post-verification drift\n",
                encoding="utf-8",
            )
            mutated_path = candidate
        return original_copy_selected(source, destination, paths, *args, **kwargs)

    with patch.object(module, "_copy_selected", mutate_before_selected_copy):
        with allow_synthetic_contract_evidence(module):
            with pytest.raises(
                ValueError,
                match=r"source (size|checksum) does not match expected manifest",
            ):
                module.build_distribution(
                    repo_root=REPO_ROOT,
                    output_root=source_root,
                    staging_root=staging_root,
                    distribution_id="selected-payload-drift-001",
                    result_json=tmp_path / "result.json",
                )

    assert mutated_path is not None
    assert not (staging_root / "distribution_manifest.json").exists()


@pytest.mark.parametrize(
    "field_to_corrupt",
    ["manifest_sha256", "artifact_manifest_sha256"],
)
def test_marker_requires_both_manifest_checksum_aliases(
    tmp_path: Path, field_to_corrupt: str
) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_marker_alias")
    manifest = tmp_path / "artifact_manifest.json"
    write_text(manifest, "manifest\n")
    digest = sha256(manifest)
    marker = {
        "manifest_sha256": digest,
        "artifact_manifest_sha256": digest,
    }
    marker[field_to_corrupt] = "0" * 64
    with pytest.raises(ValueError, match="manifest checksum"):
        module._validate_marker_manifest(marker, manifest, "marker")


def test_marker_requires_manifest_checksum_aliases_to_be_present_and_equal(
    tmp_path: Path,
) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_marker_alias_missing")
    manifest = tmp_path / "artifact_manifest.json"
    write_text(manifest, "manifest\n")
    digest = sha256(manifest)
    with pytest.raises(ValueError, match="manifest checksum"):
        module._validate_marker_manifest(
            {"manifest_sha256": digest}, manifest, "marker"
        )
    with pytest.raises(ValueError, match="manifest checksum"):
        module._validate_marker_manifest(
            {"manifest_sha256": digest, "artifact_manifest_sha256": "1" * 64},
            manifest,
            "marker",
        )


def test_copy_tree_rejects_source_mutation_during_copy(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_copy_race")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    payload = source / "payload.txt"
    write_text(payload, "stable\n")
    original_copyfile = module.shutil.copyfile

    def mutate_after_copy(source_path, target_path):
        result = original_copyfile(source_path, target_path)
        Path(source_path).write_text("changed after copy\n", encoding="utf-8")
        return result

    with patch.object(module.shutil, "copyfile", mutate_after_copy):
        with pytest.raises(ValueError, match="changed during copy"):
            module.copy_regular_tree(source, destination)


def test_production_build_rejects_synthetic_evidence(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_strict")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "build",
            "--repo-root",
            str(REPO_ROOT),
            "--output-root",
            str(source_root),
            "--staging-root",
            str(staging_root),
            "--distribution-id",
            "strict-synthetic-001",
            "--result-json",
            str(tmp_path / "result.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "real_exact_two_h800_qualified" in completed.stderr
    assert not (staging_root / "distribution_manifest.json").exists()


def test_cli_contract_fixture_reports_numeric_inventory(tmp_path: Path) -> None:
    module = load_module(MODULE_PATH, "sc26_ae_package_prebaked_cli")
    source_root = tmp_path / "fresh-output"
    build_contract_source(source_root)
    staging_root = tmp_path / "staging" / "prebaked"
    result_path = tmp_path / "work" / "result.json"
    with allow_synthetic_contract_evidence(module):
        build_stdout = StringIO()
        with redirect_stdout(build_stdout):
            assert module.main(
                [
                    "build",
                    "--repo-root",
                    str(REPO_ROOT),
                    "--output-root",
                    str(source_root),
                    "--staging-root",
                    str(staging_root),
                    "--distribution-id",
                    "cli-contract-001",
                    "--result-json",
                    str(result_path),
                ]
            ) == 0
        verify_stdout = StringIO()
        with redirect_stdout(verify_stdout):
            assert module.main(
                [
                    "verify",
                    "--repo-root",
                    str(REPO_ROOT),
                    "--prebaked-root",
                    str(staging_root),
                ]
            ) == 0
    assert "DISTRIBUTION_STATUS=verified" in verify_stdout.getvalue()
    assert "DISTRIBUTION_BUNDLE_COUNT=4" in verify_stdout.getvalue()
    assert "DISTRIBUTION_FILE_COUNT=" in verify_stdout.getvalue()

    strict_verify = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "verify",
            "--repo-root",
            str(REPO_ROOT),
            "--prebaked-root",
            str(staging_root),
        ],
        text=True,
        capture_output=True,
    )
    assert strict_verify.returncode != 0
    assert "real_prebaked_qualified" in strict_verify.stderr
