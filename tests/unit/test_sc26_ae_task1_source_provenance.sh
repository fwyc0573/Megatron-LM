#!/usr/bin/env bash
# Task1 real-source provenance contract.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task1-source-provenance.XXXXXX")
FAKE_REPO="${ROOT}/repo"
mkdir -p \
    "${FAKE_REPO}/examples" \
    "${FAKE_REPO}/megatron/training" \
    "${FAKE_REPO}/megatron/profiler" \
    "${FAKE_REPO}/SC26-AE/lib" \
    "${FAKE_REPO}/SC26-AE/tools"
for path in \
    examples/model.sh \
    pretrain_llama.py \
    megatron/training/training.py \
    megatron/profiler/cmd.py \
    megatron/training/arguments.py \
    SC26-AE/task1_gpt175b.sh \
    SC26-AE/task1_dsv3.sh \
    SC26-AE/task1_qwen3_a30b.sh \
    SC26-AE/lib/common.sh \
    SC26-AE/lib/task1_trace.sh \
    SC26-AE/tools/artifact_manifest.py; do
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

PASS_COUNT=1
assert_mutation_rejected() {
    local relative=$1
    local output status

    printf 'mutated runtime\n' >>"${FAKE_REPO}/${relative}"
    set +e
    output=$(ae_task1_assert_source_provenance "${FAKE_REPO}" "${COMMIT}" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || {
        printf 'mutated Task1 source was unexpectedly accepted: %s\n' "${relative}" >&2
        exit 1
    }
    grep -Fq "tracked HEAD blob mismatch: ${relative}" <<<"${output}" || {
        printf '%s\n' "${output}" >&2
        exit 1
    }
    git -C "${FAKE_REPO}" show "${COMMIT}:${relative}" >"${FAKE_REPO}/${relative}"
    PASS_COUNT=$((PASS_COUNT + 1))
}

for path in \
    megatron/profiler/cmd.py \
    SC26-AE/task1_gpt175b.sh \
    SC26-AE/task1_dsv3.sh \
    SC26-AE/task1_qwen3_a30b.sh \
    SC26-AE/lib/common.sh \
    SC26-AE/lib/task1_trace.sh \
    SC26-AE/tools/artifact_manifest.py; do
    assert_mutation_rejected "${path}"
done

printf 'PASS_COUNT=%s\n' "${PASS_COUNT}"
