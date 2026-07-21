#!/usr/bin/env bash
# Strict local synthetic integration coverage for the Task3 public workflow.
# Passing this script is not real-GPU or pre-dataset qualification evidence.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
FIXTURE_HELPER="${REPO_ROOT}/tests/integration/fixtures/sc26_ae_task3_fixture.py"
MANIFEST_TOOL="${REPO_ROOT}/SC26-AE/tools/artifact_manifest.py"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task3-integration.XXXXXX")
TOOLS_ROOT="${TEST_ROOT}/tools"
PREBAKED_SOURCE="${TEST_ROOT}/prebaked-source"
PREBAKED_ROOT="${TEST_ROOT}/relocated/prebaked"
OUTPUT_ROOT="${TEST_ROOT}/output"
SCHEDULER_LOG="${TEST_ROOT}/scheduler.jsonl"
SIMULATOR_LOG="${TEST_ROOT}/simulator.jsonl"
BUILDER_LOG="${TEST_ROOT}/builder.json"
PASS_COUNT=0

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$1"
}

expect_failure() {
    local expected_status=$1
    local needle=$2
    local output_path=$3
    shift 3
    local status
    set +e
    "$@" >"${output_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -eq ${expected_status} ]] || {
        cat "${output_path}" >&2
        fail "expected exit ${expected_status}, got ${status}: $*"
    }
    grep -Fq -- "${needle}" "${output_path}" || {
        cat "${output_path}" >&2
        fail "missing failure text '${needle}'"
    }
}

python3 "${FIXTURE_HELPER}" tools --output-root "${TOOLS_ROOT}" \
    >"${TEST_ROOT}/tools.json"
python3 "${FIXTURE_HELPER}" prebaked \
    --repo-root "${REPO_ROOT}" \
    --output-root "${PREBAKED_SOURCE}" \
    >"${TEST_ROOT}/prebaked.json"
mkdir -p "$(dirname -- "${PREBAKED_ROOT}")"
cp -a -- "${PREBAKED_SOURCE}" "${PREBAKED_ROOT}"

COMMON_ENV=(
    AE_OUTPUT_ROOT="${OUTPUT_ROOT}"
    ARTIFACT_SOURCE=prebaked
    PREBAKED_ROOT="${PREBAKED_ROOT}"
    SIMULATOR_HARDWARE_TYPE=H800_SXM
    TASK3_EXECUTION_MODE=synthetic
    TASK3_META_PYTHON=python3
    TASK3_SIMULATOR_PYTHON=python3
    TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}"
    TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py"
    TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py"
    TASK3_FIXTURE_SCHEDULER_LOG="${SCHEDULER_LOG}"
    TASK3_FIXTURE_SIMULATOR_LOG="${SIMULATOR_LOG}"
)

env "${COMMON_ENV[@]}" \
    TASK3_SIMULATION_RUN_ID=integration-qwen-prebaked \
    bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh" \
    >"${TEST_ROOT}/success.log" 2>&1

RUN_ROOT="${OUTPUT_ROOT}/qwen3_a30b/task3/runs/integration-qwen-prebaked"
MARKER="${OUTPUT_ROOT}/qwen3_a30b/task3/run_marker.json"
python3 -B "${MANIFEST_TOOL}" verify \
    --root "${RUN_ROOT}" \
    --manifest "${RUN_ROOT}/artifact_manifest.json" \
    >"${TEST_ROOT}/outer-manifest-verify.log"

python3 - \
    "${RUN_ROOT}" "${MARKER}" "${PREBAKED_ROOT}" \
    "${SCHEDULER_LOG}" "${SIMULATOR_LOG}" <<'PY'
import hashlib
import json
import math
import pathlib
import shlex
import sys

run_root = pathlib.Path(sys.argv[1])
marker_path = pathlib.Path(sys.argv[2])
prebaked_root = pathlib.Path(sys.argv[3]).resolve(strict=True)
scheduler_rows = [json.loads(line) for line in pathlib.Path(sys.argv[4]).read_text(encoding="utf-8").splitlines()]
simulator_rows = [json.loads(line) for line in pathlib.Path(sys.argv[5]).read_text(encoding="utf-8").splitlines()]
assert len(scheduler_rows) == 1
assert len(simulator_rows) == 1

commands = {}
for line in (run_root / "logs/commands.log").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    label, command = line.split("=", 1)
    commands[label] = shlex.split(command)
assert set(commands) == {"slowdown_trace", "scheduler", "simulator"}


def value(arguments, flag):
    index = arguments.index(flag)
    return arguments[index + 1]


scheduler = commands["scheduler"]
simulator = commands["simulator"]
slowdown_trace = commands["slowdown_trace"]
assert slowdown_trace[0] == "task3_prepare_rank0_slowdown_trace"
assert pathlib.Path(slowdown_trace[1]).is_dir()
assert pathlib.Path(slowdown_trace[2]).is_dir()
assert pathlib.Path(slowdown_trace[2]).name == "slowdown_trace_rank0"
for flag in (
    "--tensor-model-parallel-size",
    "--pipeline-model-parallel-size",
    "--expert-model-parallel-size",
    "--num-experts",
    "--world-size",
    "--local-size",
    "--micro-batch-size",
    "--global-batch-size",
    "--seq-length",
    "--hidden-size",
    "--model-size",
    "--bf16",
    "--train-iters",
    "--trace-start",
    "--output-dir",
):
    assert flag in scheduler, flag
assert value(scheduler, "--local-size") == "8"
assert value(scheduler, "--world-size") == "256"
assert value(scheduler, "--model-size") == "qwen3_a30b"

for flag in (
    "--framework",
    "--mode",
    "--trace-dir",
    "--database-dir",
    "--schedule-dir",
    "--world-size",
    "--local-size",
    "--pp-size",
    "--tp-size",
    "--exp-size",
    "--strategy",
    "--cc-backend",
    "--enable-slowdown",
    "--overlap-mode",
    "--slowdown-assets-dir",
    "--slowdown-model-path",
    "--slowdown-scaler-path",
    "--no-visualize",
    "--report-output-dir",
    "--report-model",
    "--artifact-source",
):
    assert flag in simulator, flag
assert pathlib.Path(value(simulator, "--trace-dir")).resolve(strict=True) == pathlib.Path(
    value(simulator, "--database-dir")
).resolve(strict=True)
assert value(simulator, "--local-size") == "8"
assert value(simulator, "--cc-backend") == "analytical"
assert value(simulator, "--overlap-mode") == "on"
assert value(simulator, "--artifact-source") == "prebaked"
assert pathlib.Path(value(simulator, "--slowdown-model-path")).is_file()
assert pathlib.Path(value(simulator, "--slowdown-scaler-path")).is_file()
assert simulator_rows[0]["trace_dir"] == simulator_rows[0]["database_dir"]

report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
assert report["rank0_step_time_ms"] == 22.5
assert report["rank0_forward_step_duration_sum_ms"] == 6.0
assert report["rank0_backward_step_duration_sum_ms"] == 11.0
assert report["rank0_optimizer_step_duration_sum_ms"] == 2.5
assert report["simulator_load_time_s"] == 0.125
assert report["simulator_execution_time_s"] == 0.375
assert report["simulator_wall_clock_s"] == 0.5
assert all(math.isfinite(float(value)) and value >= 0 for key, value in report.items() if key not in {
    "schema_version", "model", "artifact_source"
})
markdown = (run_root / "report.md").read_text(encoding="utf-8")
for key, item in report.items():
    assert f"| `{key}` | `{item}` |" in markdown

marker = json.loads(marker_path.read_text(encoding="utf-8"))
manifest_path = run_root / "artifact_manifest.json"
assert marker["verified"] is True
assert marker["artifact_source"] == "prebaked"
assert marker["execution_evidence"] == "local_synthetic_not_gpu_qualification"
assert marker["run_path"] == "runs/integration-qwen-prebaked"
assert marker["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert marker["artifact_manifest_sha256"] == marker["manifest_sha256"]
assert marker["slowdown_trace_scope"] == "global_rank_0"
assert marker["slowdown_trace_rank_ids"] == [0]
assert marker["ncu_metrics_source"] == "synthetic_fixture_compatibility"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["execution_evidence"] == "local_synthetic_not_gpu_qualification"
assert manifest["slowdown_trace_scope"] == "global_rank_0"
assert manifest["slowdown_trace_rank_ids"] == [0]
assert manifest["ncu_metrics_source"] == "synthetic_fixture_compatibility"
assert manifest["database_is_trace_dir"] is True
assert manifest["communication_backend"] == "analytical"
assert manifest["overlap_mode"] == "on"
assert manifest["simulation_topology"] == {
    "world_size": 256, "local_size": 8, "pp": 8, "tp": 8, "dp": 4, "exp": 4
}
entries = {row["path"]: row for row in manifest["files"]}
assert all(len(row["sha256"]) == 64 and row["size_bytes"] >= 0 for row in entries.values())
required_nonempty = {
    "logs/commands.log",
    "report.json",
    "report.md",
    "provenance/resolved_inputs.json",
    "provenance/input_evidence.json",
    "provenance/task1_manifest.json",
    "provenance/task2_manifest.json",
    "provenance/distribution_manifest.json",
    "slowdown_trace_rank0/rank0.txt",
    "logs/slowdown_trace.log",
    "slowdown_assets/manifest.json",
    "slowdown_assets/kernel_features.json",
    "slowdown_assets/backward_kernel_blueprints.json",
    "schedule/stage0_scheduling_plan.txt",
    "schedule/stage1_scheduling_plan.txt",
    "schedule/stage2_scheduling_plan.txt",
    "schedule/stage3_scheduling_plan.txt",
    "schedule/stage4_scheduling_plan.txt",
    "schedule/stage5_scheduling_plan.txt",
    "schedule/stage6_scheduling_plan.txt",
    "schedule/stage7_scheduling_plan.txt",
}
assert required_nonempty.issubset(entries)
assert all(entries[path]["size_bytes"] > 0 for path in required_nonempty)
assert entries["logs/scheduler.log"]["size_bytes"] == 0
assert entries["logs/simulator.log"]["size_bytes"] == 0
resolved = json.loads((run_root / "provenance/resolved_inputs.json").read_text(encoding="utf-8"))
assert pathlib.Path(resolved["distribution_manifest"]).resolve(strict=True).parent == prebaked_root
assert resolved["artifact_source"] == "prebaked"
assert len(list((run_root / "schedule").glob("stage*_scheduling_plan.txt"))) == 8
PY
pass "prebaked resolver remains portable and emits every explicit schedule/simulator flag"

grep -Fq 'MANIFEST_STATUS=verified' "${TEST_ROOT}/outer-manifest-verify.log"
test -s "${RUN_ROOT}/provenance/input_evidence.json"
test -s "${RUN_ROOT}/provenance/distribution_manifest.json"
pass "rank0 report, provenance, outer manifest, and verified marker close in order"

MALFORMED_OUTPUT="${TEST_ROOT}/malformed-schedule-output"
expect_failure 1 'schedule operation count mismatch' "${TEST_ROOT}/malformed-schedule.log" \
    env "${COMMON_ENV[@]}" \
        AE_OUTPUT_ROOT="${MALFORMED_OUTPUT}" \
        TASK3_FIXTURE_SCHEDULER_MODE=underspecified \
        TASK3_SIMULATION_RUN_ID=integration-malformed-schedule \
        bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh"
[[ ! -e "${MALFORMED_OUTPUT}/qwen3_a30b/task3/run_marker.json" ]] || \
    fail "malformed schedule published a marker"
pass "underspecified schedule fails exact microbatch validation before simulator execution"

WRONG_SHAPE_OUTPUT="${TEST_ROOT}/wrong-shape-schedule-output"
expect_failure 1 'PP schedule tensor shape mismatch' "${TEST_ROOT}/wrong-shape-schedule.log" \
    env "${COMMON_ENV[@]}" \
        AE_OUTPUT_ROOT="${WRONG_SHAPE_OUTPUT}" \
        TASK3_FIXTURE_SCHEDULER_MODE=wrong_shape \
        TASK3_SIMULATION_RUN_ID=integration-wrong-shape-schedule \
        bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh"
[[ ! -e "${WRONG_SHAPE_OUTPUT}/qwen3_a30b/task3/run_marker.json" ]] || \
    fail "wrong-shape schedule published a marker"
pass "PP schedule shape mismatch fails before simulator execution"

WRONG_DTYPE_OUTPUT="${TEST_ROOT}/wrong-dtype-schedule-output"
expect_failure 1 'PP schedule dtype is not torch.bfloat16' "${TEST_ROOT}/wrong-dtype-schedule.log" \
    env "${COMMON_ENV[@]}" \
        AE_OUTPUT_ROOT="${WRONG_DTYPE_OUTPUT}" \
        TASK3_FIXTURE_SCHEDULER_MODE=wrong_dtype \
        TASK3_SIMULATION_RUN_ID=integration-wrong-dtype-schedule \
        bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh"
[[ ! -e "${WRONG_DTYPE_OUTPUT}/qwen3_a30b/task3/run_marker.json" ]] || \
    fail "wrong-dtype schedule published a marker"
pass "PP schedule dtype mismatch fails before simulator execution"

NO_PP_OUTPUT="${TEST_ROOT}/no-pp-schedule-output"
expect_failure 1 'schedule contains no PP communication records' "${TEST_ROOT}/no-pp-schedule.log" \
    env "${COMMON_ENV[@]}" \
        AE_OUTPUT_ROOT="${NO_PP_OUTPUT}" \
        TASK3_FIXTURE_SCHEDULER_MODE=no_pp \
        TASK3_SIMULATION_RUN_ID=integration-no-pp-schedule \
        bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh"
[[ ! -e "${NO_PP_OUTPUT}/qwen3_a30b/task3/run_marker.json" ]] || \
    fail "schedule without PP records published a marker"
pass "missing PP schedule records fail before simulator execution"

MARKER_SHA_BEFORE=$(sha256sum "${MARKER}" | awk '{print $1}')
expect_failure 1 'already exists' "${TEST_ROOT}/reuse.log" \
    env "${COMMON_ENV[@]}" \
        TASK3_SIMULATION_RUN_ID=integration-qwen-prebaked \
        bash "${REPO_ROOT}/SC26-AE/task3_qwen3_a30b.sh"
MARKER_SHA_AFTER=$(sha256sum "${MARKER}" | awk '{print $1}')
[[ "${MARKER_SHA_BEFORE}" == "${MARKER_SHA_AFTER}" ]] || fail "existing-run rejection rewrote the verified marker"
pass "existing simulation_run_id fails without stale-output or marker reuse"

CORRUPT_ROOT="${TEST_ROOT}/corrupt-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${CORRUPT_ROOT}"
printf 'tamper\n' >>"${CORRUPT_ROOT}/bundles/dsv3/trace/rank0.txt"
CORRUPT_OUTPUT="${TEST_ROOT}/corrupt-output"
expect_failure 1 'distribution file size mismatch' "${TEST_ROOT}/corrupt.log" \
    env "${COMMON_ENV[@]}" \
        AE_OUTPUT_ROOT="${CORRUPT_OUTPUT}" \
        PREBAKED_ROOT="${CORRUPT_ROOT}" \
        TASK3_SIMULATION_RUN_ID=integration-corrupt \
        bash "${REPO_ROOT}/SC26-AE/task3_dsv3.sh"
[[ ! -e "${CORRUPT_OUTPUT}/dsv3/task3/run_marker.json" ]] || fail "corrupt source published a marker"
pass "prebaked checksum corruption fails before simulator execution and marker publication"

DRIFT_ROOT="${TEST_ROOT}/post-resolution-drift-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${DRIFT_ROOT}"
DRIFT_OUTPUT="${TEST_ROOT}/post-resolution-drift-output"
set +e
env "${COMMON_ENV[@]}" \
    AE_OUTPUT_ROOT="${DRIFT_OUTPUT}" \
    PREBAKED_ROOT="${DRIFT_ROOT}" \
    TASK3_SIMULATION_RUN_ID=integration-post-resolution-drift \
    REPO_ROOT="${REPO_ROOT}" \
    bash -c '
        set -euo pipefail
        source "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"
        original_definition=$(declare -f task3_write_input_evidence)
        original_definition=${original_definition/task3_write_input_evidence/task3_write_input_evidence_original}
        eval "${original_definition}"
        task3_write_input_evidence() {
            printf "coherent post-resolution drift\n" >>"${TASK3_TRACE_DIR}/rank0.txt"
            task3_write_input_evidence_original "$@"
        }
        ae_run_task3 qwen3_a30b
    ' >"${TEST_ROOT}/post-resolution-drift.log" 2>&1
DRIFT_STATUS=$?
set -e
[[ ${DRIFT_STATUS} -eq 1 ]] || {
    cat "${TEST_ROOT}/post-resolution-drift.log" >&2
    fail "Task3 accepted source drift after resolver validation"
}
grep -Fq 'Task3 input expectation drift' \
    "${TEST_ROOT}/post-resolution-drift.log" || {
    cat "${TEST_ROOT}/post-resolution-drift.log" >&2
    fail "Task3 drift rejection did not report the expectation mismatch"
}
[[ ! -e "${DRIFT_OUTPUT}/qwen3_a30b/task3/run_marker.json" ]] || \
    fail "post-resolution source drift published a marker"
pass "post-resolution source drift fails before input evidence and marker publication"

FRESH_OUTPUT="${TEST_ROOT}/fresh-output"
python3 "${FIXTURE_HELPER}" fresh-inputs \
    --repo-root "${REPO_ROOT}" \
    --output-root "${FRESH_OUTPUT}" \
    --model gpt175b \
    >"${TEST_ROOT}/fresh-inputs.json"
set +e
env \
    AE_OUTPUT_ROOT="${FRESH_OUTPUT}" \
    ARTIFACT_SOURCE=fresh \
    SIMULATOR_HARDWARE_TYPE=H800_SXM \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    TASK3_SIMULATOR_PYTHON=python3 \
    TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}" \
    TASK3_BUILDER="${TOOLS_ROOT}/builder.py" \
    TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py" \
    TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py" \
    TASK3_FIXTURE_BUILDER_FAIL=1 \
    TASK3_SIMULATION_RUN_ID=integration-builder-failure \
    bash "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh" \
    >"${TEST_ROOT}/builder-failure.log" 2>&1
BUILDER_STATUS=$?
set -e
[[ ${BUILDER_STATUS} -eq 37 ]] || {
    cat "${TEST_ROOT}/builder-failure.log" >&2
    fail "builder exit 37 was masked as ${BUILDER_STATUS}"
}
grep -Fq 'synthetic builder root-cause sentinel' \
    "${FRESH_OUTPUT}/gpt175b/task3/runs/integration-builder-failure/logs/builder.log"
[[ ! -e "${FRESH_OUTPUT}/gpt175b/task3/run_marker.json" ]] || fail "builder failure published a marker"
pass "fresh builder root-cause exit status propagates unchanged without fallback"

INVALID_OUTPUT="${TEST_ROOT}/invalid-report-output"
expect_failure 1 'Task3 report fields mismatch' "${TEST_ROOT}/invalid-report.log" \
    env "${COMMON_ENV[@]}" \
        AE_OUTPUT_ROOT="${INVALID_OUTPUT}" \
        TASK3_FIXTURE_REPORT_MODE=missing_optimizer \
        TASK3_SIMULATION_RUN_ID=integration-invalid-report \
        bash "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
[[ ! -e "${INVALID_OUTPUT}/gpt175b/task3/run_marker.json" ]] || fail "invalid report published a marker"
pass "missing exact optimizer metric fails report validation and marker publication"

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 11 ]] || fail "expected 11 integration cases, got ${PASS_COUNT}"
printf 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification\n'
printf 'EVIDENCE_ROOT=%s\n' "${TEST_ROOT}"
