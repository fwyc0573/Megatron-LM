#!/usr/bin/env bash
# Task3 contract tests.  These tests intentionally exercise only synthetic
# fixtures; a successful result is not a real GPU qualification claim.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
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
    shift
    local output status
    set +e
    output=$("$@" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "expected failure: $*"
    grep -Fq -- "${needle}" <<<"${output}" || {
        printf '%s\n' "${output}" >&2
        fail "missing failure text '${needle}'"
    }
}

source "${REPO_ROOT}/SC26-AE/lib/common.sh"
source "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"

[[ "$(TASK3_EXECUTION_MODE=synthetic task3_execution_evidence)" == \
    "local_synthetic_not_gpu_qualification" ]] || \
    fail 'synthetic Task3 evidence mapping is incorrect'
pass 'synthetic Task3 mode emits the non-qualification evidence class'

[[ "$(TASK3_EXECUTION_MODE=real task3_execution_evidence)" == \
    "runtime_measurement_requires_external_single_gpu_qualification" ]] || \
    fail 'real Task3 evidence mapping is incorrect'
pass 'real Task3 mode emits the pending external-sealing evidence class'

expect_failure 'Unsupported Task3 execution mode' env \
    TASK3_EXECUTION_MODE=unexpected bash -c \
    'source "$1"; task3_execution_evidence' _ \
    "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"
pass 'unknown Task3 execution mode fails without an evidence fallback'

for model in gpt175b qwen3_a30b dsv3; do
    entry="${REPO_ROOT}/SC26-AE/task3_${model}.sh"
    [[ -x "${entry}" ]] || fail "Task3 entry is not executable: ${entry}"
    grep -Fq -- 'lib/task3_simulation.sh' "${entry}" || fail "entry does not use shared Task3 library: ${entry}"
done
[[ -f "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh" ]] || fail 'shared Task3 library is missing'
pass 'three public entries and shared library exist'

TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task3-unit.XXXXXX")
OUT="${ROOT}/output"
SIM_ENGINE_FIXTURE="${ROOT}/sim-engine"
TASK3_SCHEDULER_FIXTURE="${ROOT}/scheduler.py"
TASK3_SIMULATOR_FIXTURE="${ROOT}/simulator.py"
TASK3_BUILDER_FIXTURE="${ROOT}/builder.py"
mkdir -p "${SIM_ENGINE_FIXTURE}"
: >"${TASK3_SCHEDULER_FIXTURE}"
: >"${TASK3_SIMULATOR_FIXTURE}"
: >"${TASK3_BUILDER_FIXTURE}"

expect_failure 'ARTIFACT_SOURCE' env \
    AE_OUTPUT_ROOT="${OUT}" \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    TASK3_SIMULATOR_PYTHON=python3 \
    "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass 'missing artifact source fails before execution'

expect_failure 'ARTIFACT_SOURCE' env \
    AE_OUTPUT_ROOT="${OUT}" \
    ARTIFACT_SOURCE=automatic \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    TASK3_SIMULATOR_PYTHON=python3 \
    "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass 'invalid artifact source fails without fallback'

expect_failure 'PREBAKED_ROOT' env \
    AE_OUTPUT_ROOT="${OUT}" \
    ARTIFACT_SOURCE=prebaked \
    PREBAKED_ROOT="${ROOT}/does-not-exist" \
    SIMULATOR_HARDWARE_TYPE=H800_SXM \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    TASK3_SIMULATOR_PYTHON=python3 \
    TASK3_SIM_ENGINE_ROOT="${SIM_ENGINE_FIXTURE}" \
    TASK3_BUILDER="${TASK3_BUILDER_FIXTURE}" \
    TASK3_SCHEDULER="${TASK3_SCHEDULER_FIXTURE}" \
    TASK3_SIMULATOR="${TASK3_SIMULATOR_FIXTURE}" \
    "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass 'selected missing prebaked bundle fails fast'

expect_failure 'PREBAKED_ROOT is rejected' env \
    AE_OUTPUT_ROOT="${OUT}" \
    ARTIFACT_SOURCE=fresh \
    PREBAKED_ROOT="${ROOT}/unrelated" \
    SIMULATOR_HARDWARE_TYPE=H800_SXM \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    TASK3_SIMULATOR_PYTHON=python3 \
    TASK3_SIM_ENGINE_ROOT="${SIM_ENGINE_FIXTURE}" \
    TASK3_BUILDER="${TASK3_BUILDER_FIXTURE}" \
    TASK3_SCHEDULER="${TASK3_SCHEDULER_FIXTURE}" \
    TASK3_SIMULATOR="${TASK3_SIMULATOR_FIXTURE}" \
    "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass 'fresh/prebaked cross mixing is rejected'

expect_failure 'SIMULATOR_HARDWARE_TYPE' env \
    AE_OUTPUT_ROOT="${OUT}" \
    ARTIFACT_SOURCE=prebaked \
    PREBAKED_ROOT="${ROOT}/bundle" \
    TASK3_EXECUTION_MODE=synthetic \
    TASK3_META_PYTHON=python3 \
    SIMULATOR_HARDWARE_TYPE= \
    "${REPO_ROOT}/SC26-AE/task3_gpt175b.sh"
pass 'CPU simulator hardware contract is explicit'

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 9 ]] || fail "expected 9 cases, got ${PASS_COUNT}"
printf 'EVIDENCE_ROOT=%s\n' "${ROOT}"
