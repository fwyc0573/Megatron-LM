#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

# shellcheck source=SC26-AE/lib/task3_simulation.sh
source "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"

TASK3_EXECUTION_MODE=real
TASK3_SIM_ENGINE_ROOT="${REPO_ROOT}/megatron-sim-engine"
TASK3_BUILDER="${TASK3_SIM_ENGINE_ROOT}/tools/data_prep/slowdown/build_ddp_slowdown_assets.py"
TASK3_SCHEDULER="${TASK3_SIM_ENGINE_ROOT}/src/scheduler/mg_scheduling/mg_test.py"
TASK3_SIMULATOR="${TASK3_SIM_ENGINE_ROOT}/simu_main.py"
task3_current_commits

PASS_COUNT=0

expect_mode() {
    local expected=$1
    shift
    local actual
    actual=$(task3_source_compatibility_mode "$@")
    [[ "${actual}" == "${expected}" ]] || {
        printf 'FAIL: expected mode=%s, actual=%s\n' "${expected}" "${actual}" >&2
        return 1
    }
    PASS_COUNT=$((PASS_COUNT + 1))
}

expect_failure() {
    local expected_message=$1
    shift
    local output
    if output=$(task3_source_compatibility_mode "$@" 2>&1); then
        printf 'FAIL: incompatible source commits were accepted\n%s\n' "${output}" >&2
        return 1
    fi
    grep -Fq -- "${expected_message}" <<<"${output}" || {
        printf 'FAIL: error did not identify the incompatibility\n%s\n' "${output}" >&2
        return 1
    }
    PASS_COUNT=$((PASS_COUNT + 1))
}

OLD_MAIN=df940b09c25537add927441594664c71ce01d473
OLD_SIM=2b18afc9ad3b860de2f46b9fc4b364313a21647a
NON_GITLINK_MAIN=0c498591e0f5c53c8f50c99e2fabe97b3bfa2cad
NON_GITLINK_SIM=$(git -C "${REPO_ROOT}" rev-parse "${NON_GITLINK_MAIN}:megatron-sim-engine")
DIVERGENT_SIM=ffffffffffffffffffffffffffffffffffffffff

expect_mode exact exact-current \
    "${TASK3_MAIN_COMMIT}" "${TASK3_ECHO_COMMIT}" "${TASK3_SIM_COMMIT}"
expect_mode simulator_only_reuse prior-task-artifacts \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
expect_failure 'Echo-slowdown commit differs' wrong-echo \
    "${OLD_MAIN}" ffffffffffffffffffffffffffffffffffffffff "${OLD_SIM}"
expect_failure 'outer commit is not an ancestor' non-ancestor-outer \
    ffffffffffffffffffffffffffffffffffffffff "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
expect_failure 'changes files outside megatron-sim-engine' non-gitlink-outer \
    "${NON_GITLINK_MAIN}" "${TASK3_ECHO_COMMIT}" "${NON_GITLINK_SIM}"
expect_failure 'simulator commit is not an ancestor' divergent-simulator \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${DIVERGENT_SIM}"

printf 'PASS_COUNT=%s\n' "${PASS_COUNT}"
