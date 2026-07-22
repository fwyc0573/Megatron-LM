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

expect_task2_mode() {
    local expected=$1
    shift
    local actual
    actual=$(task3_task2_source_compatibility_mode "$@")
    [[ "${actual}" == "${expected}" ]] || {
        printf 'FAIL: expected Task2 mode=%s, actual=%s\n' "${expected}" "${actual}" >&2
        return 1
    }
    PASS_COUNT=$((PASS_COUNT + 1))
}

expect_task2_failure() {
    local expected_message=$1
    shift
    local output
    if output=$(task3_task2_source_compatibility_mode "$@" 2>&1); then
        printf 'FAIL: incompatible Task2 source commits were accepted\n%s\n' "${output}" >&2
        return 1
    fi
    grep -Fq -- "${expected_message}" <<<"${output}" || {
        printf 'FAIL: Task2 error did not identify the incompatibility\n%s\n' "${output}" >&2
        return 1
    }
    PASS_COUNT=$((PASS_COUNT + 1))
}

OLD_MAIN=df940b09c25537add927441594664c71ce01d473
OLD_SIM=2b18afc9ad3b860de2f46b9fc4b364313a21647a
TASK3_ONLY_MAIN=bb326fd377e3daeef2f4e9721ffc3e81f2c6e6a9
TASK3_ONLY_SIM=$(git -C "${REPO_ROOT}" rev-parse \
    "${TASK3_ONLY_MAIN}:megatron-sim-engine")
NON_GITLINK_MAIN=d818a83bfefaac7ab816c4bc202c814fdc8fb1bb
NON_GITLINK_SIM=$(git -C "${REPO_ROOT}" rev-parse "${NON_GITLINK_MAIN}:megatron-sim-engine")
DIVERGENT_SIM=ffffffffffffffffffffffffffffffffffffffff
CHANGED_TASK2_MAIN=8d17d6355feaf7c59581eb21509fe54701597098
CHANGED_TASK2_SIM=$(git -C "${REPO_ROOT}" rev-parse \
    "${CHANGED_TASK2_MAIN}:megatron-sim-engine")
CONSUMER_ONLY_PARENT=23327306756733f14d4cbfa1599adf0c856449ba
CONSUMER_ONLY_MAIN=bb326fd377e3daeef2f4e9721ffc3e81f2c6e6a9
CONSUMER_ONLY_SIM=$(git -C "${REPO_ROOT}" rev-parse \
    "${CONSUMER_ONLY_MAIN}:megatron-sim-engine")

expect_mode exact exact-current \
    "${TASK3_MAIN_COMMIT}" "${TASK3_ECHO_COMMIT}" "${TASK3_SIM_COMMIT}"
expect_failure 'changes files outside the Task3 compatibility allowlist' old-task1-after-dense-change \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
CURRENT_MAIN=${TASK3_MAIN_COMMIT}
CURRENT_SIM=${TASK3_SIM_COMMIT}
TASK3_MAIN_COMMIT=${TASK3_ONLY_MAIN}
TASK3_SIM_COMMIT=${TASK3_ONLY_SIM}
expect_mode simulator_only_reuse prior-task1-before-dense-change \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
TASK3_MAIN_COMMIT=${CURRENT_MAIN}
TASK3_SIM_COMMIT=${CURRENT_SIM}
expect_failure 'Echo-slowdown commit differs' wrong-echo \
    "${OLD_MAIN}" ffffffffffffffffffffffffffffffffffffffff "${OLD_SIM}"
expect_failure 'outer commit is not an ancestor' non-ancestor-outer \
    ffffffffffffffffffffffffffffffffffffffff "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
expect_failure 'changes files outside the Task3 compatibility allowlist' non-gitlink-outer \
    "${NON_GITLINK_MAIN}" "${TASK3_ECHO_COMMIT}" "${NON_GITLINK_SIM}"
expect_failure 'does not contain a simulator gitlink change' control-only-outer \
    "${TASK3_MAIN_COMMIT}" "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
TASK3_MAIN_COMMIT=${TASK3_ONLY_MAIN}
TASK3_SIM_COMMIT=${TASK3_ONLY_SIM}
expect_failure 'simulator commit is not an ancestor' divergent-simulator \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${DIVERGENT_SIM}"
TASK3_MAIN_COMMIT=${CURRENT_MAIN}
TASK3_SIM_COMMIT=${CURRENT_SIM}

TASK3_MAIN_COMMIT=${CONSUMER_ONLY_MAIN}
TASK3_SIM_COMMIT=${CONSUMER_ONLY_SIM}
expect_mode task1_consumer_only_reuse prior-task1-before-consumer-only-change \
    "${CONSUMER_ONLY_PARENT}" "${TASK3_ECHO_COMMIT}" "${CONSUMER_ONLY_SIM}"
TASK3_MAIN_COMMIT=${CURRENT_MAIN}
TASK3_SIM_COMMIT=${CURRENT_SIM}

expect_task2_mode exact exact-current-task2 \
    "${TASK3_MAIN_COMMIT}" "${TASK3_ECHO_COMMIT}" "${TASK3_SIM_COMMIT}"
expect_task2_mode task2_producer_equivalent_reuse prior-task2-artifacts \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
expect_task2_failure 'Echo-slowdown commit differs' wrong-task2-echo \
    "${OLD_MAIN}" ffffffffffffffffffffffffffffffffffffffff "${OLD_SIM}"
expect_task2_failure 'outer commit is not an ancestor' non-ancestor-task2-outer \
    ffffffffffffffffffffffffffffffffffffffff "${TASK3_ECHO_COMMIT}" "${OLD_SIM}"
expect_task2_failure 'Task2 producer source changed' changed-task2-wrapper \
    "${CHANGED_TASK2_MAIN}" "${TASK3_ECHO_COMMIT}" "${CHANGED_TASK2_SIM}"
expect_task2_failure 'recorded simulator commit does not match' inconsistent-task2-simulator \
    "${OLD_MAIN}" "${TASK3_ECHO_COMMIT}" "${DIVERGENT_SIM}"

printf 'PASS_COUNT=%s\n' "${PASS_COUNT}"
