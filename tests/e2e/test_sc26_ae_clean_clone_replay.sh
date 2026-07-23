#!/usr/bin/env bash
# Local clean-clone-style replay for the SC26 AE public entry surface.
#
# This harness deliberately creates an isolated outer clone and standalone
# local clones of every pinned submodule.  The uncommitted AE surface is then
# copied into the isolated clone and committed there as an ephemeral synthetic
# overlay.  This avoids network access while exercising the same path-relative
# entry-point behavior that a public clean clone will use.  The resulting
# evidence is local synthetic workflow evidence, never GPU qualification or a
# release pre-dataset claim.
set -euo pipefail

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/data/ycfeng/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-clean-clone-replay.XXXXXX")
CLONE_ROOT="${TEST_ROOT}/clone"
OUTSIDE_ROOT="${TEST_ROOT}/outside-cwd"
LOG_ROOT="${TEST_ROOT}/logs"
TOOLS_ROOT="${TEST_ROOT}/tools"

mkdir -p "${OUTSIDE_ROOT}" "${LOG_ROOT}"

git clone --no-local --no-recurse-submodules --quiet "${SOURCE_ROOT}" "${CLONE_ROOT}"

ECHO_COMMIT=$(tr -d '[:space:]' <"${SOURCE_ROOT}/Echo-slowdown/.source_commit")
SIM_COMMIT=$(tr -d '[:space:]' <"${SOURCE_ROOT}/megatron-sim-engine/.source_commit")

# Copy only the evaluator-facing source surface.  In particular, do not copy
# the current worktree's ignored SC26-AE/output captures into the clean clone.
mkdir -p "${CLONE_ROOT}/SC26-AE/lib" "${CLONE_ROOT}/SC26-AE/tools"
cp -a "${SOURCE_ROOT}/SC26-AE/README.md" \
    "${SOURCE_ROOT}/SC26-AE/setup.sh" \
    "${SOURCE_ROOT}"/SC26-AE/task*.sh "${CLONE_ROOT}/SC26-AE/"
cp -a "${SOURCE_ROOT}/SC26-AE/lib/." "${CLONE_ROOT}/SC26-AE/lib/"
cp -a "${SOURCE_ROOT}/SC26-AE/tools/"*.py "${CLONE_ROOT}/SC26-AE/tools/"

# Copy the local synthetic fixture and the existing contract tests into the
# ephemeral overlay; these files are test inputs, not qualification artifacts.
mkdir -p "${CLONE_ROOT}/tests/integration/fixtures" "${CLONE_ROOT}/tests/integration" \
    "${CLONE_ROOT}/tests/e2e"
cp -a "${SOURCE_ROOT}/tests/integration/fixtures/sc26_ae_task3_fixture.py" \
    "${CLONE_ROOT}/tests/integration/fixtures/"
cp -a "${SOURCE_ROOT}/tests/integration/test_sc26_ae_task1_contracts.sh" \
    "${CLONE_ROOT}/tests/integration/"
cp -a "${SOURCE_ROOT}/tests/integration/test_sc26_ae_task2_contract.sh" \
    "${CLONE_ROOT}/tests/integration/"
cp -a "${SOURCE_ROOT}/tests/integration/test_sc26_ae_setup.sh" \
    "${CLONE_ROOT}/tests/integration/"
cp -a "${SOURCE_ROOT}/tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh" \
    "${CLONE_ROOT}/tests/e2e/"
cp -a "${SOURCE_ROOT}/tests/e2e/test_sc26_ae_fresh_chain.sh" \
    "${CLONE_ROOT}/tests/e2e/"

git -C "${CLONE_ROOT}" config user.email "sc26-ae-clean-clone@example.invalid"
git -C "${CLONE_ROOT}" config user.name "SC26 AE clean-clone fixture"
git -C "${CLONE_ROOT}" add \
    SC26-AE tests/e2e/test_sc26_ae_fresh_chain.sh \
    tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh \
    tests/integration/fixtures/sc26_ae_task3_fixture.py \
    tests/integration/test_sc26_ae_setup.sh \
    tests/integration/test_sc26_ae_task1_contracts.sh \
    tests/integration/test_sc26_ae_task2_contract.sh
if ! git -C "${CLONE_ROOT}" diff --cached --quiet; then
    git -C "${CLONE_ROOT}" commit --quiet -m "Create ephemeral synthetic AE replay surface"
fi

[[ -z "$(git -C "${CLONE_ROOT}" status --porcelain --untracked-files=all)" ]]
[[ "$(tr -d '[:space:]' <"${CLONE_ROOT}/Echo-slowdown/.source_commit")" == "${ECHO_COMMIT}" ]]
[[ "$(tr -d '[:space:]' <"${CLONE_ROOT}/megatron-sim-engine/.source_commit")" == "${SIM_COMMIT}" ]]
[[ ! -e "${CLONE_ROOT}/Echo-slowdown/.git" ]]
[[ ! -e "${CLONE_ROOT}/megatron-sim-engine/.git" ]]
[[ ! -e "${CLONE_ROOT}/megatron-sim-engine/.gitmodules" ]]
[[ ! -e "${CLONE_ROOT}/SC26-AE/output" ]]

run_case() {
    local name=$1
    shift
    printf '[RUN] %s\n' "${name}" | tee "${LOG_ROOT}/${name}.log"
    (cd -- "${OUTSIDE_ROOT}" && \
        PYTHONDONTWRITEBYTECODE=1 "$@") \
        >>"${LOG_ROOT}/${name}.log" 2>&1
    printf '[PASS] %s\n' "${name}" | tee -a "${LOG_ROOT}/${name}.log"
}

run_case setup_public_contract \
    bash "${CLONE_ROOT}/tests/integration/test_sc26_ae_setup.sh"
run_case task1_public_entries \
    bash "${CLONE_ROOT}/tests/integration/test_sc26_ae_task1_contracts.sh"
run_case task2_public_entries \
    bash "${CLONE_ROOT}/tests/integration/test_sc26_ae_task2_contract.sh"
run_case task3_prebaked_public_entries \
    bash "${CLONE_ROOT}/tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh"
run_case task1_task2_task3_fresh_chain \
    bash "${CLONE_ROOT}/tests/e2e/test_sc26_ae_fresh_chain.sh"

# The setup contract plus the three task suites cover the setup boundary and
# 3 + 3 + 3 public entries.  The fresh chain additionally proves one atomic
# producer-to-consumer path from outside the repository CWD.  Keep these
# counts explicit so a future missing entry cannot be hidden by a passing
# aggregate test.
grep -Fq 'PASS_COUNT=6' "${LOG_ROOT}/setup_public_contract.log"
grep -Fq 'PASS_COUNT=46' "${LOG_ROOT}/task1_public_entries.log"
grep -Fq 'Task2 snapshot-only execution' "${LOG_ROOT}/task2_public_entries.log"
grep -Fq 'MODEL_PASS_COUNT=3' "${LOG_ROOT}/task3_prebaked_public_entries.log"
grep -Fq 'CHAIN_PASS_COUNT=1' "${LOG_ROOT}/task1_task2_task3_fresh_chain.log"
grep -Fq 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification' \
    "${LOG_ROOT}/task3_prebaked_public_entries.log"
grep -Fq 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification' \
    "${LOG_ROOT}/task1_task2_task3_fresh_chain.log"

# Tests must not dirty the isolated source or any pinned producer.
[[ -z "$(git -C "${CLONE_ROOT}" status --porcelain --untracked-files=all)" ]]

printf 'PUBLIC_TASK1_ENTRY_COUNT=3\n'
printf 'PUBLIC_TASK2_ENTRY_COUNT=3\n'
printf 'PUBLIC_TASK3_ENTRY_COUNT=3\n'
printf 'SETUP_CONTRACT_CASE_COUNT=6\n'
printf 'FRESH_ATOMIC_CHAIN_COUNT=1\n'
printf 'OUTER_CLONE_STATUS=clean\n'
printf 'ECHO_SOURCE_STATUS=clean\n'
printf 'SIM_ENGINE_SOURCE_STATUS=clean\n'
printf 'COLLECTIVE_SIM_STATUS=optional_absent\n'
printf 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification\n'
printf 'REPLAY_ROOT=%s\n' "${TEST_ROOT}"
printf '%s\n' 'PASS: isolated clean-clone-style nine-entry synthetic replay completed.'
