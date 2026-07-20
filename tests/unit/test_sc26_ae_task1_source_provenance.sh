#!/usr/bin/env bash
# Task1 real-source provenance contract.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task1-source-provenance.XXXXXX")
FAKE_REPO="${ROOT}/repo"
mkdir -p "${FAKE_REPO}/examples" "${FAKE_REPO}/megatron/training" "${FAKE_REPO}/megatron/profiler"
for path in \
    examples/model.sh \
    pretrain_llama.py \
    megatron/training/training.py \
    megatron/profiler/cmd.py \
    megatron/training/arguments.py; do
    printf 'stable %s\n' "${path}" >"${FAKE_REPO}/${path}"
done
git -C "${FAKE_REPO}" init -q
git -C "${FAKE_REPO}" config user.email ae-test@example.invalid
git -C "${FAKE_REPO}" config user.name sc26-ae-test
git -C "${FAKE_REPO}" add .
git -C "${FAKE_REPO}" commit -q -m source
COMMIT=$(git -C "${FAKE_REPO}" rev-parse HEAD)

# shellcheck disable=SC1090
source "${REPO_ROOT}/SC26-AE/lib/common.sh"
# shellcheck disable=SC1090
source "${REPO_ROOT}/SC26-AE/lib/task1_trace.sh"
AE_T1_SOURCE_SCRIPT="${FAKE_REPO}/examples/model.sh"

ae_task1_assert_source_provenance "${FAKE_REPO}" "${COMMIT}"

printf 'mutated runtime\n' >>"${FAKE_REPO}/megatron/profiler/cmd.py"
set +e
OUTPUT=$(ae_task1_assert_source_provenance "${FAKE_REPO}" "${COMMIT}" 2>&1)
STATUS=$?
set -e
[[ ${STATUS} -ne 0 ]] || {
    printf 'mutated Task1 source was unexpectedly accepted\n' >&2
    exit 1
}
grep -Fq 'tracked HEAD blob mismatch' <<<"${OUTPUT}" || {
    printf '%s\n' "${OUTPUT}" >&2
    exit 1
}

printf 'PASS_COUNT=2\n'
