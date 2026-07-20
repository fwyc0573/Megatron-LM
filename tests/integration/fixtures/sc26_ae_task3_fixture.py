#!/usr/bin/env python3
"""Build strict local-only fixtures for the SC26 AE Task3 shell tests.

The generated artifacts intentionally use the synthetic evidence classes.  They
exercise manifests, checksums, provenance, scheduling, and report plumbing, but
they are not GPU qualification evidence and must never be published as a real
pre-dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import resource
import stat
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Mapping


MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "gpt175b": {
        "profile": "175",
        "topology": {"world_size": 1024, "local_size": 8, "pp": 8, "tp": 8, "dp": 16, "exp": 1},
    },
    "qwen3_a30b": {
        "profile": "full",
        "topology": {"world_size": 256, "local_size": 8, "pp": 4, "tp": 8, "dp": 8, "exp": 8},
    },
    "dsv3": {
        "profile": "smoke",
        "topology": {"world_size": 256, "local_size": 8, "pp": 4, "tp": 8, "dp": 8, "exp": 8},
    },
}


def stable_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: pathlib.Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(repo_root: pathlib.Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *arguments], text=True
    ).strip()


def source_commits(repo_root: pathlib.Path) -> Dict[str, str]:
    return {
        "megatron_lm": git_output(repo_root, "rev-parse", "HEAD"),
        "echo_slowdown": git_output(repo_root, "rev-parse", "HEAD:Echo-slowdown"),
        "megatron_sim_engine": git_output(repo_root, "rev-parse", "HEAD:megatron-sim-engine"),
    }


def load_manifest_module(repo_root: pathlib.Path):
    tool = repo_root / "SC26-AE/tools/artifact_manifest.py"
    specification = importlib.util.spec_from_file_location("sc26_ae_fixture_manifest", tool)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the canonical artifact manifest module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def create_manifest(module: Any, root: pathlib.Path, metadata: Dict[str, Any]) -> pathlib.Path:
    relative_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = module.create_manifest(root, metadata, relative_files)
    output = root / "artifact_manifest.json"
    stable_json(output, manifest)
    module.verify_manifest(root, manifest)
    return output


def trace_text(rank: int) -> str:
    return "\n".join(
        [
            f"rank:{rank}:forward_step(stage_id=0,batch_id=0,mg_state=steady,duration=4.000,timestamp=100.000,cmd_uid=fwd-{rank})",
            f"rank:{rank}:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=6.000,timestamp=104.000,cmd_uid=bwd-{rank})",
            f"rank:{rank}:ddp_grad_comm(stage_id=0,batch_id=0,mg_state=steady,duration=0.500,timestamp=105.000,trigger_cmd_uid=bwd-{rank})",
            f"rank:{rank}:optimizer_step(stage_id=0,batch_id=0,mg_state=finalize,duration=1.500,timestamp=110.000,cmd_uid=opt-{rank})",
        ]
    ) + "\n"


def slowdown_assets(root: pathlib.Path, model: str) -> None:
    stable_json(
        root / "manifest.json",
        {
            "schema_version": "sc26-ae-synthetic-slowdown-assets-v1",
            "model": model,
            "backward_cmd_uids": ["bwd-0"],
            "execution_evidence": "local_synthetic_fixture",
        },
    )
    stable_json(root / "kernel_features.json", {"bwd-0": {"synthetic_feature": 1.0}})
    stable_json(root / "backward_kernel_blueprints.json", {"bwd-0": ["synthetic_kernel"]})


def build_prebaked(repo_root: pathlib.Path, output_root: pathlib.Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    module = load_manifest_module(repo_root)
    compatible = source_commits(repo_root)
    producer = {
        "megatron_lm": "1" * 40,
        "echo_slowdown": compatible["echo_slowdown"],
        "megatron_sim_engine": compatible["megatron_sim_engine"],
    }
    predictor_run_id = "synthetic-predictor-001"
    bundles: Dict[str, Dict[str, Any]] = {}

    shared_root = output_root / "bundles/shared_task2"
    write_text(shared_root / "merge/input/kernel_metric_output.csv", "Kernel Name,SM\nsynthetic,1\n")
    stable_json(
        shared_root / "training_testing/output/xgb_model.json",
        {"format": "sc26-ae-synthetic-xgb-v1", "weights": [0.1, 0.2], "bias": 0.3},
    )
    stable_json(
        shared_root / "training_testing/output/standard_scaler.json",
        {"feature_names": ["ground_truth", "Compute throughput"], "mean": [0, 0], "scale": [1, 1]},
    )
    shared_manifest = create_manifest(
        module,
        shared_root,
        {
            "schema_version": "sc26-ae-artifact-manifest-v1",
            "model": "shared_task2",
            "task": "task2",
            "artifact_source": "prebaked",
            "predictor_run_id": predictor_run_id,
            "source_commits": producer,
            "execution_evidence": "local_synthetic_not_two_gpu_qualification",
        },
    )
    bundles["shared_task2"] = {
        "root": "bundles/shared_task2",
        "manifest": "bundles/shared_task2/artifact_manifest.json",
        "manifest_sha256": sha256(shared_manifest),
        "predictor_run_id": predictor_run_id,
        "producer_commits": producer,
    }

    for model, specification in MODEL_SPECS.items():
        capture_id = f"synthetic-{model}-capture"
        model_root = output_root / "bundles" / model
        write_text(model_root / "trace/rank0.txt", trace_text(0))
        write_text(model_root / "nsys/capture.sqlite", "synthetic sqlite fixture\n")
        slowdown_assets(model_root / "slowdown_assets", model)
        model_manifest = create_manifest(
            module,
            model_root,
            {
                "schema_version": "sc26-ae-artifact-manifest-v1",
                "model": model,
                "task": "prebaked",
                "artifact_source": "prebaked",
                "capture_id": capture_id,
                "predictor_run_id": predictor_run_id,
                "source_commits": producer,
                "simulation_topology": specification["topology"],
                "profile": specification["profile"],
                "precision": "bf16",
                "ddp_overlap": True,
                "execution_evidence": "local_synthetic_fixture",
            },
        )
        bundles[model] = {
            "root": f"bundles/{model}",
            "manifest": f"bundles/{model}/artifact_manifest.json",
            "manifest_sha256": sha256(model_manifest),
            "model": model,
            "profile": specification["profile"],
            "simulation_topology": specification["topology"],
            "capture_id": capture_id,
            "predictor_run_id": predictor_run_id,
            "producer_commits": producer,
        }

    inventory = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "distribution_manifest.json"
    )
    distribution = {
        "schema_version": "sc26-ae-distribution-manifest-v1",
        "artifact_source": "prebaked",
        "distribution_id": "synthetic-local-distribution-001",
        "compatible_commits": {
            "echo_slowdown": compatible["echo_slowdown"],
            "megatron_sim_engine": compatible["megatron_sim_engine"],
        },
        "bundles": bundles,
        "files": [
            {
                "path": relative,
                "size_bytes": (output_root / relative).stat().st_size,
                "sha256": sha256(output_root / relative),
            }
            for relative in inventory
        ],
        "execution_evidence": "local_synthetic_fixture",
    }
    stable_json(output_root / "distribution_manifest.json", distribution)
    return {
        "prebaked_root": str(output_root),
        "distribution_file_count": len(inventory),
        "predictor_run_id": predictor_run_id,
    }


def build_fresh_inputs(repo_root: pathlib.Path, output_root: pathlib.Path, model: str) -> Dict[str, Any]:
    module = load_manifest_module(repo_root)
    commits = source_commits(repo_root)
    specification = MODEL_SPECS[model]
    capture_id = f"synthetic-{model}-fresh"
    predictor_run_id = "synthetic-fresh-predictor"

    task1_dir = output_root / model / "task1"
    task1_root = task1_dir / "runs" / capture_id
    write_text(task1_root / "runtime/profiler_log/synthetic/rank0.txt", trace_text(0))
    write_text(task1_root / "nsys/capture.sqlite", "synthetic sqlite fixture\n")
    task1_manifest = create_manifest(
        module,
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
            "execution_evidence": "local_synthetic_fixture",
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

    task2_root = output_root / "_shared/task2/runs" / predictor_run_id
    write_text(task2_root / "merge/input/kernel_metric_output.csv", "Kernel Name,SM\nsynthetic,1\n")
    stable_json(task2_root / "training_testing/output/xgb_model.json", {"synthetic": True})
    stable_json(task2_root / "training_testing/output/standard_scaler.json", {"synthetic": True})
    task2_manifest = create_manifest(
        module,
        task2_root,
        {
            "schema_version": "sc26-ae-artifact-manifest-v1",
            "model": "shared_task2",
            "task": "task2",
            "artifact_source": "fresh",
            "predictor_run_id": predictor_run_id,
            "source_commits": commits,
            "execution_evidence": "local_synthetic_not_two_gpu_qualification",
        },
    )
    stable_json(
        output_root / "_shared/task2/predictor_marker.json",
        {
            "schema_version": "sc26-ae-task2-shared-pointer-v1",
            "predictor_run_id": predictor_run_id,
            "run_path": f"_shared/task2/runs/{predictor_run_id}",
            "manifest_sha256": sha256(task2_manifest),
            "artifact_manifest_sha256": sha256(task2_manifest),
            "verified": True,
        },
    )
    return {
        "output_root": str(output_root),
        "model": model,
        "capture_id": capture_id,
        "predictor_run_id": predictor_run_id,
    }


BUILDER_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--trace-dir", required=True)
parser.add_argument("--nsys-sqlite", required=True)
parser.add_argument("--ncu-metrics-csv", required=True)
parser.add_argument("--label-prefix", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--scaler-path", required=True)
args = parser.parse_args()
if os.environ.get("TASK3_FIXTURE_BUILDER_FAIL") == "1":
    print("synthetic builder root-cause sentinel", file=sys.stderr)
    raise SystemExit(37)
for value in (args.nsys_sqlite, args.ncu_metrics_csv, args.model_path, args.scaler_path):
    if not pathlib.Path(value).is_file():
        raise SystemExit(f"missing required input: {value}")
trace_paths = sorted(pathlib.Path(args.trace_dir).glob("*.txt"))
if not trace_paths:
    raise SystemExit("no trace files")
backward = set()
triggers = set()
for path in trace_paths:
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":backward_step(" in line:
            match = re.search(r"(?:^|,)cmd_uid=([^,)]+)", line)
            if match:
                backward.add(match.group(1))
        if ":ddp_grad_comm(" in line:
            match = re.search(r"(?:^|,)trigger_cmd_uid=([^,)]+)", line)
            if match:
                triggers.add(match.group(1))
if not backward or backward != triggers:
    raise SystemExit(f"backward/trigger coverage mismatch: backward={sorted(backward)}, triggers={sorted(triggers)}")
root = pathlib.Path(args.output_dir)
root.mkdir(parents=True, exist_ok=False)
payloads = {
    "manifest.json": {
        "schema_version": "sc26-ae-synthetic-slowdown-assets-v1",
        "backward_cmd_uids": sorted(backward),
        "execution_evidence": "local_synthetic_fixture",
    },
    "kernel_features.json": {value: {"synthetic_feature": 1.0} for value in sorted(backward)},
    "backward_kernel_blueprints.json": {value: ["synthetic_kernel"] for value in sorted(backward)},
}
for name, payload in payloads.items():
    (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if os.environ.get("TASK3_FIXTURE_BUILDER_LOG"):
    pathlib.Path(os.environ["TASK3_FIXTURE_BUILDER_LOG"]).write_text(
        json.dumps({"argv": sys.argv[1:], "backward_cmd_uids": sorted(backward)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
'''


SCHEDULER_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--tensor-model-parallel-size", type=int, required=True)
parser.add_argument("--pipeline-model-parallel-size", type=int, required=True)
parser.add_argument("--expert-model-parallel-size", type=int, required=True)
parser.add_argument("--num-experts", type=int, required=True)
parser.add_argument("--world-size", type=int, required=True)
parser.add_argument("--local-size", type=int, required=True)
parser.add_argument("--micro-batch-size", type=int, required=True)
parser.add_argument("--global-batch-size", type=int, required=True)
parser.add_argument("--seq-length", type=int, required=True)
parser.add_argument("--hidden-size", type=int, required=True)
parser.add_argument("--model-size", required=True)
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--train-iters", type=int, required=True)
parser.add_argument("--trace-start", type=int, required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--untie-embeddings-and-output-weights", action="store_true")
args = parser.parse_args()
if args.local_size != 8 or not args.bf16:
    raise SystemExit("scheduler contract requires local_size=8 and bf16")
parallel_product = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
if args.world_size % parallel_product != 0:
    raise SystemExit("world size must be divisible by TP times PP")
data_parallel_size = args.world_size // parallel_product
microbatch_denominator = args.micro_batch_size * data_parallel_size
if args.global_batch_size % microbatch_denominator != 0:
    raise SystemExit("global batch size must be divisible by micro batch size times DP")
expected_microbatches = args.global_batch_size // microbatch_denominator
mode = os.environ.get("TASK3_FIXTURE_SCHEDULER_MODE", "valid")
if mode in {"valid", "wrong_shape", "wrong_dtype", "no_pp"}:
    emitted_microbatches = expected_microbatches
elif mode == "underspecified":
    emitted_microbatches = 1
else:
    raise SystemExit(f"unsupported TASK3_FIXTURE_SCHEDULER_MODE: {mode}")
shape = [args.seq_length, args.micro_batch_size, args.hidden_size]
if mode == "wrong_shape":
    shape[0] += 1
pipeline_dtype = "torch.float16" if mode == "wrong_dtype" else "torch.bfloat16"
root = pathlib.Path(args.output_dir)
root.mkdir(parents=True, exist_ok=True)
for stage in range(args.pipeline_model_parallel_size):
    records = []
    if mode != "no_pp":
        pp_operation = "send_forward" if stage == 0 else "recv_forward"
        records.append(
            f"stage:{stage}:{pp_operation}(batch_id=0, mg_state=steady, duration=None, "
            f"description=None, group_kind=pp, input__shape={shape}, "
            f"input__dtype={pipeline_dtype})"
        )
    records.extend(
        f"stage:{stage}:forward_step(batch_id={batch_id}, mg_state=steady, duration=None, "
        "description=None, group_kind=None, input__shape=None, input__dtype=None)"
        for batch_id in range(emitted_microbatches)
    )
    records.extend(
        f"stage:{stage}:backward_step(batch_id={batch_id}, mg_state=steady, duration=None, "
        "description=None, group_kind=None, input__shape=None, input__dtype=None)"
        for batch_id in range(emitted_microbatches)
    )
    records.append(
        f"stage:{stage}:optimizer_step(batch_id=0, mg_state=finalize, duration=None, "
        "description=None, group_kind=None, input__shape=None, input__dtype=None)"
    )
    (root / f"stage{stage}_scheduling_plan.txt").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )
if os.environ.get("TASK3_FIXTURE_SCHEDULER_LOG"):
    with pathlib.Path(os.environ["TASK3_FIXTURE_SCHEDULER_LOG"]).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": sys.argv[1:], "local_size": args.local_size}, sort_keys=True) + "\n")
'''


SIMULATOR_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--framework", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--trace-dir", required=True)
parser.add_argument("--database-dir", required=True)
parser.add_argument("--schedule-dir", required=True)
parser.add_argument("--world-size", type=int, required=True)
parser.add_argument("--local-size", type=int, required=True)
parser.add_argument("--pp-size", type=int, required=True)
parser.add_argument("--tp-size", type=int, required=True)
parser.add_argument("--exp-size", type=int, required=True)
parser.add_argument("--strategy", required=True)
parser.add_argument("--cc-backend", required=True)
parser.add_argument("--enable-slowdown", action="store_true")
parser.add_argument("--overlap-mode", required=True)
parser.add_argument("--slowdown-assets-dir", required=True)
parser.add_argument("--slowdown-model-path", required=True)
parser.add_argument("--slowdown-scaler-path", required=True)
parser.add_argument("--no-visualize", action="store_true")
parser.add_argument("--report-output-dir", required=True)
parser.add_argument("--report-model", required=True)
parser.add_argument("--artifact-source", required=True)
args = parser.parse_args()
trace = pathlib.Path(args.trace_dir).resolve(strict=True)
database = pathlib.Path(args.database_dir).resolve(strict=True)
if trace != database:
    raise SystemExit("DATABASE_DIR and TRACE_DIR differ")
if args.local_size != 8 or args.cc_backend != "analytical" or args.overlap_mode != "on":
    raise SystemExit("simulator topology/backend/overlap contract mismatch")
for path in (args.slowdown_assets_dir, args.slowdown_model_path, args.slowdown_scaler_path):
    if not pathlib.Path(path).exists():
        raise SystemExit(f"missing slowdown input: {path}")
values = {
    "gpt175b": (18.5, 5.0, 9.0, 2.0),
    "qwen3_a30b": (22.5, 6.0, 11.0, 2.5),
    "dsv3": (24.5, 6.5, 12.0, 3.0),
}
step, forward, backward, optimizer = values[args.report_model]
report = {
    "schema_version": "sc26-ae-rank0-report-v1",
    "model": args.report_model,
    "artifact_source": args.artifact_source,
    "rank_id": 0,
    "rank0_step_time_ms": step,
    "rank0_forward_step_duration_sum_ms": forward,
    "rank0_backward_step_duration_sum_ms": backward,
    "rank0_optimizer_step_duration_sum_ms": optimizer,
    "rank0_comp_plus_comm_diagnostic_ms": forward + backward + optimizer + 0.25,
    "simulator_load_time_s": 0.125,
    "simulator_execution_time_s": 0.375,
    "simulator_wall_clock_s": 0.5,
}
mode = os.environ.get("TASK3_FIXTURE_REPORT_MODE", "valid")
if mode == "missing_optimizer":
    report.pop("rank0_optimizer_step_duration_sum_ms")
elif mode == "boolean_rank":
    report["rank_id"] = False
output = pathlib.Path(args.report_output_dir)
output.mkdir(parents=True, exist_ok=True)
(output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(output / "report.md").write_text(
    "# Synthetic rank0 report\n\n"
    + "\n".join(f"| `{key}` | `{value}` |" for key, value in report.items())
    + "\n",
    encoding="utf-8",
)
if os.environ.get("TASK3_FIXTURE_SIMULATOR_LOG"):
    with pathlib.Path(os.environ["TASK3_FIXTURE_SIMULATOR_LOG"]).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "argv": sys.argv[1:],
            "trace_dir": str(trace),
            "database_dir": str(database),
            "local_size": args.local_size,
            "backend": args.cc_backend,
            "overlap_mode": args.overlap_mode,
        }, sort_keys=True) + "\n")
'''


TORCHRUN_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
rank_id=""
previous=""
for argument in "$@"; do
    if [[ "${previous}" == "--fake-current-rank-id" ]]; then
        rank_id=${argument}
        break
    fi
    previous=${argument}
done
[[ "${rank_id}" =~ ^[0-9]+$ ]]
mkdir -p "${PWD}/profiler_log/synthetic" "${PWD}/memory_traces_scaling"
cat > "${PWD}/profiler_log/synthetic/trace_rank${rank_id}.txt" <<EOF
rank:${rank_id}:forward_step(stage_id=0,batch_id=0,mg_state=steady,duration=4.000,timestamp=100.000,cmd_uid=fwd-${rank_id})
rank:${rank_id}:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=6.000,timestamp=104.000,cmd_uid=bwd-${rank_id})
rank:${rank_id}:ddp_grad_comm(stage_id=0,batch_id=0,mg_state=steady,duration=0.500,timestamp=105.000,trigger_cmd_uid=bwd-${rank_id})
rank:${rank_id}:optimizer_step(stage_id=0,batch_id=0,mg_state=finalize,duration=1.500,timestamp=110.000,cmd_uid=opt-${rank_id})
EOF
cat > "${PWD}/memory_traces_scaling/memory_trace_rank${rank_id}.json" <<JSON
{"0":{"samples":[{"timestamp_s":0.1,"reserved_memory_MB":100.0,"allocated_memory_MB":80.0}],"peak_allocated_MB":90.0,"theoretical_memory_MB":120.0}}
JSON
'''


NSYS_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
subcommand=$1
shift
case "${subcommand}" in
    profile)
        output_base=""
        while (($#)); do
            case "$1" in
                --output) output_base=$2; shift 2 ;;
                bash) bash "$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        [[ -n "${output_base}" ]]
        printf 'synthetic nsys report\n' > "${output_base}.nsys-rep"
        ;;
    export)
        output_path=""
        input_path=""
        while (($#)); do
            case "$1" in
                -o) output_path=$2; shift 2 ;;
                -t|--force-overwrite) shift 2 ;;
                *) input_path=$1; shift ;;
            esac
        done
        [[ -s "${input_path}" && -n "${output_path}" ]]
        printf 'synthetic sqlite fixture\n' > "${output_path}"
        ;;
    *) exit 92 ;;
esac
'''


TASK2_GENERATOR_SCRIPT = r'''from pathlib import Path
import json

root = Path('.')
(root / 'training_testing/output').mkdir(parents=True, exist_ok=True)
(root / 'merge/input').mkdir(parents=True, exist_ok=True)
(root / 'training_testing/output/train_dataset.csv').write_text(
    'ground_truth,Compute throughput,slowdown\n1,2,0.1\n2,3,0.2\n', encoding='utf-8')
(root / 'training_testing/output/xgb_model.json').write_text(json.dumps({
    'format': 'sc26-ae-synthetic-xgb-v1', 'weights': [0.1, 0.2], 'bias': 0.3}), encoding='utf-8')
(root / 'training_testing/output/standard_scaler.json').write_text(json.dumps({
    'feature_names': ['ground_truth', 'Compute throughput'], 'mean': [0, 0], 'scale': [1, 1]}), encoding='utf-8')
(root / 'merge/input/kernel_metric_output.csv').write_text('Kernel Name,SM\nsynthetic,1\n', encoding='utf-8')
print('MSE for each fold (validation set): [1.0, 2.0, 3.0, 4.0, 5.0]')
print('Average MSE (validation set): 3.0')
print('Test MSE: 0.5')
'''


def write_tools(root: pathlib.Path) -> Dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    paths = {
        "builder": root / "builder.py",
        "scheduler": root / "scheduler.py",
        "simulator": root / "simulator.py",
        "torchrun": root / "torchrun",
        "nsys": root / "nsys",
        "task2_generator": root / "task2_generate.py",
    }
    write_text(paths["builder"], BUILDER_SCRIPT, executable=True)
    write_text(paths["scheduler"], SCHEDULER_SCRIPT, executable=True)
    write_text(paths["simulator"], SIMULATOR_SCRIPT, executable=True)
    write_text(paths["torchrun"], TORCHRUN_SCRIPT, executable=True)
    write_text(paths["nsys"], NSYS_SCRIPT, executable=True)
    write_text(paths["task2_generator"], TASK2_GENERATOR_SCRIPT)
    return {key: str(value) for key, value in paths.items()}


def measure(output: pathlib.Path, allocation_mib: int, command: Iterable[str]) -> int:
    if allocation_mib <= 0:
        raise ValueError("allocation_mib must be positive")
    allocation = bytearray(allocation_mib * 1024 * 1024)
    for offset in range(0, len(allocation), 4096):
        allocation[offset] = 1
    start = time.perf_counter()
    result = subprocess.run(list(command), check=False)
    elapsed = time.perf_counter() - start
    max_rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    payload = {
        "schema_version": "sc26-ae-synthetic-process-measurement-v1",
        "exit_code": result.returncode,
        "wall_clock_s": round(elapsed, 6),
        "peak_rss_kib": max_rss_kib,
        "peak_rss_gib": round(max_rss_kib / 1048576.0, 9),
        "tested_host_allocation_bytes": len(allocation),
        "tested_host_allocation_mib": allocation_mib,
        "execution_evidence": "local_synthetic_fixture",
    }
    stable_json(output, payload)
    return result.returncode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    prebaked = subparsers.add_parser("prebaked")
    prebaked.add_argument("--repo-root", type=pathlib.Path, required=True)
    prebaked.add_argument("--output-root", type=pathlib.Path, required=True)

    fresh = subparsers.add_parser("fresh-inputs")
    fresh.add_argument("--repo-root", type=pathlib.Path, required=True)
    fresh.add_argument("--output-root", type=pathlib.Path, required=True)
    fresh.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)

    tools = subparsers.add_parser("tools")
    tools.add_argument("--output-root", type=pathlib.Path, required=True)

    measurement = subparsers.add_parser("measure")
    measurement.add_argument("--output", type=pathlib.Path, required=True)
    measurement.add_argument("--allocation-mib", type=int, default=32)
    measurement.add_argument("remainder", nargs=argparse.REMAINDER)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prebaked":
        print(json.dumps(build_prebaked(args.repo_root.resolve(), args.output_root), sort_keys=True))
        return 0
    if args.command == "fresh-inputs":
        args.output_root.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                build_fresh_inputs(args.repo_root.resolve(), args.output_root, args.model),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "tools":
        print(json.dumps(write_tools(args.output_root), sort_keys=True))
        return 0
    if args.command == "measure":
        command = list(args.remainder)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise ValueError("measure requires a command after --")
        return measure(args.output, args.allocation_mib, command)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)
