#!/usr/bin/env bash
# Full local synthetic Task1 -> Task2 -> Task3 public-entry chain.
# It verifies orchestration/provenance only and is not H800 qualification.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
FIXTURE_HELPER="${REPO_ROOT}/tests/integration/fixtures/sc26_ae_task3_fixture.py"
MANIFEST_TOOL="${REPO_ROOT}/SC26-AE/tools/artifact_manifest.py"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/data/ycfeng/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-fresh-chain.XXXXXX")
TOOLS_ROOT="${TEST_ROOT}/tools"
OUTPUT_ROOT="${TEST_ROOT}/output"
MODEL=qwen3_a30b
CAPTURE_ID=qwen3_a30b-fresh-chain-synthetic
PREDICTOR_RUN_ID=fresh-chain-synthetic-predictor
SIMULATION_RUN_ID=fresh-chain-synthetic-simulation

python3 "${FIXTURE_HELPER}" tools --output-root "${TOOLS_ROOT}" \
    >"${TEST_ROOT}/tools.json"

PATH="${TOOLS_ROOT}:${PATH}" \
AE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
AE_TASK1_TEST_MODE=1 \
AE_TASK1_TEST_CAPTURE_ID="${CAPTURE_ID}" \
AE_TASK1_TORCHRUN="${TOOLS_ROOT}/torchrun" \
AE_MEGATRON_PYTHON="$(command -v python3)" \
AE_NSYS_BIN="${TOOLS_ROOT}/nsys" \
QUICK=0 \
SCALE_GPU=0 \
CAPTURE_NSYS=1 \
    bash "${REPO_ROOT}/SC26-AE/task1_qwen3_a30b.sh" \
    >"${TEST_ROOT}/task1.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1 \
AE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
TASK2_SOURCE_REPO="${REPO_ROOT}/Echo-slowdown" \
TASK2_EXECUTION_MODE=synthetic \
TASK2_META_PYTHON=python3 \
TASK2_PYTHON=python3 \
TASK2_RUN_COMMAND="python3 '${TOOLS_ROOT}/task2_generate.py'" \
TASK2_SKIP_UPDATE_CONFIGS=1 \
TASK2_SKIP_PREDICT=1 \
TASK2_SKIP_HARDWARE_CHECK=1 \
REBUILD=1 \
PREDICTOR_RUN_ID="${PREDICTOR_RUN_ID}" \
    bash "${REPO_ROOT}/SC26-AE/task2_qwen3_a30b.sh" \
    >"${TEST_ROOT}/task2.log" 2>&1

MEASUREMENT="${TEST_ROOT}/task3-measurement.json"
env \
    AE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    ARTIFACT_SOURCE=fresh \
    SIMULATOR_HARDWARE_TYPE=H800_SXM \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    TASK3_SIMULATOR_PYTHON=python3 \
    TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}" \
    TASK3_BUILDER="${TOOLS_ROOT}/builder.py" \
    TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py" \
    TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py" \
    TASK3_SIMULATION_RUN_ID="${SIMULATION_RUN_ID}" \
    TASK3_FIXTURE_BUILDER_LOG="${TEST_ROOT}/builder.json" \
    TASK3_FIXTURE_SCHEDULER_LOG="${TEST_ROOT}/scheduler.jsonl" \
    TASK3_FIXTURE_SIMULATOR_LOG="${TEST_ROOT}/simulator.jsonl" \
    python3 "${FIXTURE_HELPER}" measure \
        --output "${MEASUREMENT}" \
        --allocation-mib 32 \
        -- bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh" \
        >"${TEST_ROOT}/task3.log" 2>&1

TASK1_RUN="${OUTPUT_ROOT}/${MODEL}/task1/runs/${CAPTURE_ID}"
TASK2_RUN="${OUTPUT_ROOT}/_shared/task2/runs/${PREDICTOR_RUN_ID}"
TASK3_RUN="${OUTPUT_ROOT}/${MODEL}/task3/runs/${SIMULATION_RUN_ID}"

for run_root in "${TASK1_RUN}" "${TASK2_RUN}" "${TASK3_RUN}"; do
    python3 -B "${MANIFEST_TOOL}" verify \
        --root "${run_root}" \
        --manifest "${run_root}/artifact_manifest.json" \
        >>"${TEST_ROOT}/manifest-verification.log"
done

python3 - \
    "${REPO_ROOT}" "${OUTPUT_ROOT}" "${TEST_ROOT}" \
    "${CAPTURE_ID}" "${PREDICTOR_RUN_ID}" "${SIMULATION_RUN_ID}" \
    "${MEASUREMENT}" <<'PY'
import hashlib
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[2])
test_root = pathlib.Path(sys.argv[3])
capture_id, predictor_run_id, simulation_run_id = sys.argv[4:7]
measurement_path = pathlib.Path(sys.argv[7])
model = "qwen3_a30b"

main_commit = __import__("subprocess").check_output(
    ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
).strip()
echo_commit = (repo_root / "Echo-slowdown/.source_commit").read_text(encoding="utf-8").strip()
sim_commit = (repo_root / "megatron-sim-engine/.source_commit").read_text(encoding="utf-8").strip()
expected_commits = {
    "megatron_lm": main_commit,
    "echo_slowdown": echo_commit,
    "megatron_sim_engine": sim_commit,
}

task1_dir = output_root / model / "task1"
task1_root = task1_dir / "runs" / capture_id
task1_marker = json.loads((task1_dir / "capture_marker.json").read_text(encoding="utf-8"))
task1_manifest_path = task1_root / "artifact_manifest.json"
task1_manifest = json.loads(task1_manifest_path.read_text(encoding="utf-8"))
assert task1_marker["capture_id"] == capture_id
assert task1_marker["manifest_sha256"] == hashlib.sha256(task1_manifest_path.read_bytes()).hexdigest()
assert task1_marker["artifact_manifest_sha256"] == task1_marker["manifest_sha256"]
assert task1_manifest["source_commits"] == expected_commits
assert task1_manifest["profile"] == "full"
assert task1_manifest["precision"] == "bf16"
assert task1_manifest["mock_data"] is True
assert task1_manifest["ddp_overlap"] is True
assert task1_manifest["simulation_topology"] == {
    "world_size": 256, "local_size": 8, "pp": 8, "tp": 8, "dp": 4, "exp": 4
}
assert task1_manifest["capture_runtime"]["physical_gpu_count"] == 1
assert task1_manifest["capture_runtime"]["scaling_min_warmup_iters"] == 3
assert task1_manifest["capture_runtime"]["scaling_profile_iters"] == 1
task1_summary = task1_manifest["capture_summary"]
assert task1_summary["selected_rank_count"] == 32
assert task1_summary["trace_file_count"] == 32
assert task1_summary["memory_json_count"] == 32
assert task1_summary["capture_nsys"] is True
assert task1_summary["d16_gate_applicable"] is True
assert task1_summary["estimate_basis_rank"] == 0
assert task1_summary["estimate_rank_count"] == 256
assert task1_summary["single_rank_elapsed_seconds"] > 0
assert (
    task1_summary["estimated_full_seconds"]
    == task1_summary["single_rank_elapsed_seconds"] * 256
)
assert task1_summary["fresh_capture_gate_threshold_seconds"] == 7200
expected_d16_result = (
    "pass"
    if task1_summary["estimated_full_seconds"] <= 7200
    else "prebaked_required"
)
assert task1_summary["fresh_capture_gate_result"] == expected_d16_result

task2_root = output_root / "_shared/task2/runs" / predictor_run_id
task2_marker_path = output_root / "_shared/task2/predictor_marker.json"
task2_marker = json.loads(task2_marker_path.read_text(encoding="utf-8"))
task2_manifest_path = task2_root / "artifact_manifest.json"
task2_manifest = json.loads(task2_manifest_path.read_text(encoding="utf-8"))
assert task2_marker["predictor_run_id"] == predictor_run_id
assert task2_marker["manifest_sha256"] == hashlib.sha256(task2_manifest_path.read_bytes()).hexdigest()
assert task2_marker["artifact_manifest_sha256"] == task2_marker["manifest_sha256"]
assert task2_manifest["source_commits"] == expected_commits
assert task2_manifest["execution_evidence"] == "local_synthetic_not_two_gpu_qualification"
metrics = json.loads((task2_root / "metrics.json").read_text(encoding="utf-8"))
assert metrics["dataset_row_count"] == 2
assert metrics["validation_mse_by_fold"] == [1.0, 2.0, 3.0, 4.0, 5.0]
assert metrics["average_validation_mse"] == 3.0
assert metrics["test_mse"] == 0.5
assert metrics["model_reload_max_abs_prediction_delta"] == 0.0

task3_dir = output_root / model / "task3"
task3_root = task3_dir / "runs" / simulation_run_id
task3_marker = json.loads((task3_dir / "run_marker.json").read_text(encoding="utf-8"))
task3_manifest_path = task3_root / "artifact_manifest.json"
task3_manifest = json.loads(task3_manifest_path.read_text(encoding="utf-8"))
assert task3_marker["verified"] is True
assert task3_marker["artifact_source"] == "fresh"
assert task3_marker["capture_id"] == capture_id
assert task3_marker["predictor_run_id"] == predictor_run_id
assert task3_marker["manifest_sha256"] == hashlib.sha256(task3_manifest_path.read_bytes()).hexdigest()
assert task3_marker["artifact_manifest_sha256"] == task3_marker["manifest_sha256"]
assert task3_manifest["source_commits"] == expected_commits
assert task3_manifest["artifact_source"] == "fresh"
assert task3_manifest["capture_id"] == capture_id
assert task3_manifest["predictor_run_id"] == predictor_run_id
assert task3_manifest["simulation_run_id"] == simulation_run_id
assert task3_manifest["database_is_trace_dir"] is True
assert task3_manifest["communication_backend"] == "analytical"
assert task3_manifest["overlap_mode"] == "on"

resolved = json.loads((task3_root / "provenance/resolved_inputs.json").read_text(encoding="utf-8"))
assert resolved["artifact_source"] == "fresh"
assert resolved["capture_id"] == capture_id
assert resolved["predictor_run_id"] == predictor_run_id
assert resolved["distribution_manifest"] == ""
assert resolved["source_assets_dir"] == ""
assert pathlib.Path(resolved["task1_manifest"]).resolve() == task1_manifest_path.resolve()
assert pathlib.Path(resolved["task2_manifest"]).resolve() == task2_manifest_path.resolve()

input_evidence = json.loads((task3_root / "provenance/input_evidence.json").read_text(encoding="utf-8"))
assert input_evidence["artifact_source"] == "fresh"
assert input_evidence["trace_file_count"] == 32
assert "distribution_manifest" not in input_evidence
assert all(entry["size_bytes"] > 0 and len(entry["sha256"]) == 64 for entry in input_evidence["trace_files"])

builder = json.loads((test_root / "builder.json").read_text(encoding="utf-8"))
assert set(builder["backward_cmd_uids"]) == {"bwd-0"}
assets = json.loads((task3_root / "slowdown_assets/manifest.json").read_text(encoding="utf-8"))
assert set(assets["backward_cmd_uids"]) == {"bwd-0"}
expected_backward = {
    f"bwd-{pp_stage * 8 * 4 + exp_rank * 8}"
    for pp_stage in range(8)
    for exp_rank in range(4)
}
blueprints = json.loads(
    (task3_root / "slowdown_assets/backward_kernel_blueprints.json").read_text(
        encoding="utf-8"
    )
)
assert set(blueprints) == expected_backward
expansion = json.loads(
    (task3_root / "provenance/slowdown_blueprint_expansion.json").read_text(
        encoding="utf-8"
    )
)
assert expansion["source_blueprint_count"] == 1
assert expansion["expanded_blueprint_count"] == 32
assert set(expansion["expanded_trigger_cmd_uids"]) == expected_backward

report = json.loads((task3_root / "report.json").read_text(encoding="utf-8"))
assert report["rank0_step_time_ms"] == 22.5
assert report["rank0_forward_step_duration_sum_ms"] == 6.0
assert report["rank0_backward_step_duration_sum_ms"] == 11.0
assert report["rank0_optimizer_step_duration_sum_ms"] == 2.5
assert report["simulator_load_time_s"] == 0.125
assert report["simulator_execution_time_s"] == 0.375
assert report["simulator_wall_clock_s"] == 0.5

measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
assert measurement["exit_code"] == 0
assert measurement["wall_clock_s"] > 0
assert measurement["peak_rss_kib"] > 0
assert measurement["peak_rss_gib"] > 0
assert measurement["tested_host_allocation_mib"] == 32

print(f"TASK1_TRACE_FILES={task1_manifest['capture_summary']['trace_file_count']}")
print(f"TASK1_MEMORY_JSON={task1_manifest['capture_summary']['memory_json_count']}")
print(f"TASK2_DATASET_ROWS={metrics['dataset_row_count']}")
print(f"TASK2_AVERAGE_VALIDATION_MSE={metrics['average_validation_mse']}")
print(f"TASK2_TEST_MSE={metrics['test_mse']}")
print(
    "TASK2_MODEL_RELOAD_MAX_ABS_PREDICTION_DELTA="
    f"{metrics['model_reload_max_abs_prediction_delta']}"
)
print(f"TASK3_BACKWARD_CMD_UID_COUNT={len(expected_backward)}")
print(f"TASK3_RANK0_STEP_MS={report['rank0_step_time_ms']}")
print(f"TASK3_FORWARD_MS={report['rank0_forward_step_duration_sum_ms']}")
print(f"TASK3_BACKWARD_MS={report['rank0_backward_step_duration_sum_ms']}")
print(f"TASK3_OPTIMIZER_MS={report['rank0_optimizer_step_duration_sum_ms']}")
print(f"TASK3_LOAD_S={report['simulator_load_time_s']}")
print(f"TASK3_EXECUTION_S={report['simulator_execution_time_s']}")
print(f"TASK3_SIM_WALL_S={report['simulator_wall_clock_s']}")
print(f"TASK3_PROCESS_WALL_S={measurement['wall_clock_s']}")
print(f"TASK3_PEAK_RSS_KIB={measurement['peak_rss_kib']}")
print(f"TASK3_PEAK_RSS_GIB={measurement['peak_rss_gib']}")
print(f"TESTED_HOST_ALLOCATION_MIB={measurement['tested_host_allocation_mib']}")
PY

grep -Fq 'MANIFEST_STATUS=verified' "${TEST_ROOT}/manifest-verification.log"
grep -Fq 'TASK1_STATUS=verified' "${TEST_ROOT}/task1.log"
grep -Fq '[PASS] Task2 qwen3_a30b' "${TEST_ROOT}/task2.log"
grep -Fq '[PASS] Task3 model=qwen3_a30b artifact_source=fresh' "${TEST_ROOT}/task3.log"

printf 'PASS: Task1 capture, shared Task2 predictor, and fresh Task3 simulation formed one verified synthetic chain.\n'
printf 'CHAIN_PASS_COUNT=1\n'
printf 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification\n'
printf 'EVIDENCE_ROOT=%s\n' "${TEST_ROOT}"
