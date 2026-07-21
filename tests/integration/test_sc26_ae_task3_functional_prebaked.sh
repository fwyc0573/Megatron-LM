#!/usr/bin/env bash
# Functional fake-level Task3 coverage.  This is CPU-only synthetic evidence;
# it must never be treated as H800 or release qualification.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
FIXTURE_HELPER="${REPO_ROOT}/tests/integration/fixtures/sc26_ae_task3_fixture.py"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task3-functional.XXXXXX")
SOURCE_ROOT="${TEST_ROOT}/fresh-source"
PACKAGE_ROOT="${TEST_ROOT}/functional-prebaked"
TOOLS_ROOT="${TEST_ROOT}/tools"
OUTPUT_ROOT="${TEST_ROOT}/output"
RESULT_JSON="${TEST_ROOT}/package-result.json"
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
    local needle=$1
    local output_path=$2
    shift 2
    local status
    set +e
    "$@" >"${output_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || {
        cat "${output_path}" >&2
        fail "expected failure: $*"
    }
    grep -Fq -- "${needle}" "${output_path}" || {
        cat "${output_path}" >&2
        fail "missing failure text '${needle}'"
    }
}

python3 "${FIXTURE_HELPER}" functional-source \
    --repo-root "${REPO_ROOT}" --output-root "${SOURCE_ROOT}" \
    >"${TEST_ROOT}/functional-source.json"
python3 SC26-AE/tools/package_prebaked.py build-functional \
    --repo-root "${REPO_ROOT}" \
    --output-root "${SOURCE_ROOT}" \
    --staging-root "${PACKAGE_ROOT}" \
    --distribution-id "functional-fixture-001" \
    --result-json "${RESULT_JSON}" \
    >"${TEST_ROOT}/package-build.log"
python3 "${FIXTURE_HELPER}" tools --output-root "${TOOLS_ROOT}" \
    >"${TEST_ROOT}/tools.json"

python3 - "${PACKAGE_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
distribution = json.loads((root / "distribution_manifest.json").read_text())
assert distribution["schema_version"] == "sc26-ae-functional-distribution-manifest-v1"
assert distribution["execution_evidence"] == "functional_prebaked_not_release_qualified"
assert set(distribution["bundles"]) == {"gpt175b", "qwen3_a30b", "shared_task2"}
for model in ("gpt175b", "qwen3_a30b"):
    entry = distribution["bundles"][model]
    manifest = json.loads((root / entry["manifest"]).read_text())
    paths = {item["path"] for item in manifest["files"]}
    assert "ncu/kernel_metric_output.csv" in paths, model
    assert (root / entry["root"] / "ncu/kernel_metric_output.csv").is_file()
print("FUNCTIONAL_MODEL_LOCAL_NCU=verified")
PY
pass "functional package contains GPT and Qwen3 model-local rank-0 NCU features"

COMMON_ENV=(
    AE_OUTPUT_ROOT="${OUTPUT_ROOT}"
    ARTIFACT_SOURCE=prebaked
    PREBAKED_ROOT="${PACKAGE_ROOT}"
    SIMULATOR_HARDWARE_TYPE=cpu
    TASK3_META_PYTHON=python3
    TASK3_SIMULATOR_PYTHON=python3
    TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}"
    TASK3_BUILDER="${TOOLS_ROOT}/builder.py"
    TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py"
    TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py"
)

expect_failure \
    'functional prebaked input requires TASK3_ALLOW_FUNCTIONAL_PREBAKED=1' \
    "${TEST_ROOT}/missing-opt-in.log" \
    env "${COMMON_ENV[@]}" \
        TASK3_EXECUTION_MODE=synthetic \
        TASK3_ALLOW_FUNCTIONAL_PREBAKED=0 \
        TASK3_SIMULATION_RUN_ID=functional-no-opt-in \
        bash "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass "functional prebaked Task3 fails without explicit opt-in"

expect_failure \
    'TASK3_META_PYTHON overrides are forbidden in real mode' \
    "${TEST_ROOT}/real-mode.log" \
    env "${COMMON_ENV[@]}" \
        TASK3_EXECUTION_MODE=real \
        TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
        TASK3_SIMULATION_RUN_ID=functional-real-mode \
        bash "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass "functional prebaked Task3 rejects real execution mode"

for model in gpt175b qwen3_a30b; do
    env "${COMMON_ENV[@]}" \
        TASK3_EXECUTION_MODE=synthetic \
        TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
        TASK3_SIMULATION_RUN_ID="functional-${model}" \
        bash "${REPO_ROOT}/SC26-AE/task3_${model}.sh" \
        >"${TEST_ROOT}/${model}.log" 2>&1
    run_root="${OUTPUT_ROOT}/${model}/task3/runs/functional-${model}"
    test -s "${run_root}/report.json"
    test -s "${run_root}/report.md"
    test -s "${run_root}/artifact_manifest.json"
    test -s "${OUTPUT_ROOT}/${model}/task3/run_marker.json"
    python3 - "${run_root}" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
report = json.loads((run_root / "report.json").read_text())
manifest = json.loads((run_root / "artifact_manifest.json").read_text())
marker = json.loads((run_root.parent.parent / "run_marker.json").read_text())
assert report["artifact_source"] == "prebaked"
assert report["simulator_wall_clock_s"] > 0
assert manifest["execution_evidence"] == "local_synthetic_not_gpu_qualification"
assert manifest["ncu_metrics_source"] == "task1_rank0"
assert marker["ncu_metrics_source"] == "task1_rank0"
assert marker["slowdown_trace_scope"] == "global_rank_0"
assert marker["slowdown_trace_rank_ids"] == [0]
PY
    pass "${model} functional CPU-only Task3 emits report, manifest, marker, and rank-0 NCU provenance"
done

printf 'FUNCTIONAL_MODEL_PASS_COUNT=2\n'
printf 'EVIDENCE_CLASS=functional_prebaked_not_release_qualified\n'
printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
