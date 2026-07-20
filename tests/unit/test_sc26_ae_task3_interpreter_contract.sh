#!/usr/bin/env bash
# Task3 real-mode interpreter binding contract.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TASK3_SH="${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"

# shellcheck disable=SC1090
source "${REPO_ROOT}/SC26-AE/lib/common.sh"
# shellcheck disable=SC1090
source "${TASK3_SH}"

expect_failure() {
    local needle=$1
    shift
    local output status
    set +e
    output=$("$@" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || {
        printf 'expected failure: %s\n' "$*" >&2
        exit 1
    }
    grep -Fq -- "${needle}" <<<"${output}" || {
        printf '%s\n' "${output}" >&2
        printf 'missing expected error: %s\n' "${needle}" >&2
        exit 1
    }
}

expect_failure 'TASK3_META_PYTHON overrides are forbidden in real mode' \
    env TASK3_EXECUTION_MODE=real TASK3_META_PYTHON=/tmp/other-python \
    TASK3_SIMULATOR_PYTHON=/opt/conda/envs/megatron_env/bin/python \
    bash -c 'source "$1"; task3_bind_interpreters' _ "${TASK3_SH}"

expect_failure 'TASK3_SIMULATOR_PYTHON overrides are forbidden in real mode' \
    env TASK3_EXECUTION_MODE=real \
    TASK3_META_PYTHON=/opt/conda/envs/megatron_env/bin/python \
    TASK3_SIMULATOR_PYTHON=/tmp/other-python \
    bash -c 'source "$1"; task3_bind_interpreters' _ "${TASK3_SH}"

TASK3_EXECUTION_MODE=synthetic
TASK3_META_PYTHON=python3
TASK3_SIMULATOR_PYTHON=python3
task3_bind_interpreters
[[ "${TASK3_META_PYTHON}" == python3 ]]
[[ "${TASK3_SIMULATOR_PYTHON}" == python3 ]]

printf 'PASS_COUNT=3\n'
