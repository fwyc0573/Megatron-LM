#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${REPO_ROOT}/SC26-AE/lib/common.sh"
source "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/sc26-ae-task3-rank-scope.XXXXXX")
FULL_TRACE_DIR="${TEST_ROOT}/full"
RANK0_TRACE_DIR="${TEST_ROOT}/rank0"
mkdir -p "${FULL_TRACE_DIR}"
printf 'rank:0:backward_step(cmd_uid=bwd-0)\n' > "${FULL_TRACE_DIR}/trace_rank0_fixture.txt"
printf 'rank:1:backward_step(cmd_uid=bwd-1)\n' > "${FULL_TRACE_DIR}/trace_rank1_fixture.txt"

task3_prepare_rank0_slowdown_trace "${FULL_TRACE_DIR}" "${RANK0_TRACE_DIR}"

[[ -f "${RANK0_TRACE_DIR}/trace_rank0_fixture.txt" ]]
[[ ! -e "${RANK0_TRACE_DIR}/trace_rank1_fixture.txt" ]]

EMPTY_TRACE_DIR="${TEST_ROOT}/empty"
mkdir -p "${EMPTY_TRACE_DIR}"
if task3_prepare_rank0_slowdown_trace "${EMPTY_TRACE_DIR}" "${TEST_ROOT}/missing-output"; then
    printf 'rank0-only preparation unexpectedly accepted an empty trace directory\n' >&2
    exit 1
fi

printf 'PASS: Task3 slowdown trace preparation keeps only global rank 0 and fails on missing rank 0\n'
