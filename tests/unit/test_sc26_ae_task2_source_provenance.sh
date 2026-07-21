#!/usr/bin/env bash
# Task2 real outer-producer provenance contract.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task2-source-provenance.XXXXXX")
FAKE_REPO="${ROOT}/repo"
LIBRARY="${ROOT}/task2_echo_library.sh"

SOURCE_FILES=(
    SC26-AE/task2_gpt175b.sh
    SC26-AE/task2_dsv3.sh
    SC26-AE/task2_qwen3_a30b.sh
    SC26-AE/lib/common.sh
    SC26-AE/lib/task2_echo.sh
    SC26-AE/tools/artifact_manifest.py
    SC26-AE/tools/echo_metrics.py
)

for relative in "${SOURCE_FILES[@]}"; do
    mkdir -p -- "${FAKE_REPO}/$(dirname -- "${relative}")"
    printf 'tracked source: %s\n' "${relative}" >"${FAKE_REPO}/${relative}"
done

git -C "${FAKE_REPO}" init -q
git -C "${FAKE_REPO}" config user.email ae-test@example.invalid
git -C "${FAKE_REPO}" config user.name sc26-ae-test
git -C "${FAKE_REPO}" add .
git -C "${FAKE_REPO}" commit -q -m source
COMMIT=$(git -C "${FAKE_REPO}" rev-parse HEAD)

sed '/^task2_main$/,$d' "${REPO_ROOT}/SC26-AE/lib/task2_echo.sh" >"${LIBRARY}"
# shellcheck disable=SC1090
source "${LIBRARY}"
set -euo pipefail
TASK2_REPO_ROOT=${FAKE_REPO}
TASK2_MAIN_COMMIT=${COMMIT}
TASK2_MODE=real

if grep -Fq 'rev-parse HEAD:' "${REPO_ROOT}/SC26-AE/lib/task2_echo.sh"; then
    printf 'Task2 outer gitlink lookup bypasses the recorded main commit\n' >&2
    exit 1
fi

task2_assert_outer_source_provenance "${COMMIT}"
PASS_COUNT=2

assert_mutation_rejected() {
    local relative=$1
    local output status

    printf 'mutated runtime\n' >>"${FAKE_REPO}/${relative}"
    set +e
    output=$(task2_assert_outer_source_provenance "${COMMIT}" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || {
        printf 'mutated Task2 outer source was unexpectedly accepted: %s\n' "${relative}" >&2
        exit 1
    }
    grep -Fq "Task2 tracked outer blob mismatch: ${relative}" <<<"${output}" || {
        printf '%s\n' "${output}" >&2
        exit 1
    }
    git -C "${FAKE_REPO}" show "${COMMIT}:${relative}" >"${FAKE_REPO}/${relative}"
    PASS_COUNT=$((PASS_COUNT + 1))
}

for relative in "${SOURCE_FILES[@]}"; do
    assert_mutation_rejected "${relative}"
done

printf 'runtime smoke mutation\n' >>"${FAKE_REPO}/SC26-AE/lib/task2_echo.sh"
export TASK2_SKIP_OUTER_SOURCE_PROVENANCE=1
BYPASS_OUTPUT=$(task2_maybe_assert_outer_source_provenance "${COMMIT}" 2>&1)
grep -Fq 'outer source provenance hash gate bypassed explicitly for runtime smoke' \
    <<<"${BYPASS_OUTPUT}"
git -C "${FAKE_REPO}" show "${COMMIT}:SC26-AE/lib/task2_echo.sh" \
    >"${FAKE_REPO}/SC26-AE/lib/task2_echo.sh"
PASS_COUNT=$((PASS_COUNT + 1))

TASK2_OUTPUT_ROOT=${ROOT}/output
TASK2_MODEL_KEY=gpt175b
TASK2_PREDICTOR_RUN_ID=publication-one
TASK2_META_PYTHON=python3
TASK2_ECHO_COMMIT=0000000000000000000000000000000000000000
TASK2_COMMIT_PROVENANCE=explicit_test_commit
TASK2_REBUILD=1
TASK2_SOURCE_ROOT=${ROOT}/snapshot
TASK2_RUN_COMMAND='bash run_all.sh'
TASK2_UPDATE_COMMAND='python update_configs.py'
CUDA_VISIBLE_DEVICES=0,1
PROVENANCE_PATH=${ROOT}/provenance.json
task2_write_provenance "${PROVENANCE_PATH}"
python3 - "${PROVENANCE_PATH}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["outer_source_provenance"] == {
    "bypassed": True,
    "checked": False,
    "reason": "explicit_runtime_smoke_bypass_dirty_worktree",
}
assert payload["execution_evidence"] == (
    "runtime_measurement_requires_external_two_gpu_qualification"
)
PY
PASS_COUNT=$((PASS_COUNT + 1))
export TASK2_SKIP_OUTER_SOURCE_PROVENANCE=0

RUN_ROOT="${TASK2_OUTPUT_ROOT}/_shared/task2/runs/${TASK2_PREDICTOR_RUN_ID}"
MARKER="${TASK2_OUTPUT_ROOT}/${TASK2_MODEL_KEY}/task2/predictor_marker.json"
POINTER="${TASK2_OUTPUT_ROOT}/_shared/task2/predictor_marker.json"
mkdir -p -- "${RUN_ROOT}"
cat >"${RUN_ROOT}/provenance.json" <<'JSON'
{
  "echo_commit": "0000000000000000000000000000000000000000",
  "execution_evidence": "runtime_measurement_requires_external_two_gpu_qualification"
}
JSON
printf '{}\n' >"${RUN_ROOT}/metrics.json"

assert_publication_rejected() {
    local kind=$1 output status

    printf 'mutated before publication\n' >>"${FAKE_REPO}/SC26-AE/lib/task2_echo.sh"
    set +e
    if [[ "${kind}" == marker ]]; then
        output=$(task2_write_marker "${MARKER}" "${RUN_ROOT}" "$(printf '0%.0s' {1..64})" 2>&1)
    else
        output=$(task2_write_shared_pointer "${POINTER}" "${RUN_ROOT}" "$(printf '0%.0s' {1..64})" 2>&1)
    fi
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || {
        printf 'Task2 outer mutation unexpectedly allowed %s publication\n' "${kind}" >&2
        exit 1
    }
    grep -Fq 'Task2 tracked outer blob mismatch: SC26-AE/lib/task2_echo.sh' <<<"${output}" || {
        printf '%s\n' "${output}" >&2
        exit 1
    }
    [[ ! -e "${MARKER}" && ! -e "${POINTER}" ]] || {
        printf 'Task2 %s publication occurred after outer source mutation\n' "${kind}" >&2
        exit 1
    }
    git -C "${FAKE_REPO}" show "${COMMIT}:SC26-AE/lib/task2_echo.sh" \
        >"${FAKE_REPO}/SC26-AE/lib/task2_echo.sh"
    PASS_COUNT=$((PASS_COUNT + 1))
}

assert_publication_rejected marker
assert_publication_rejected shared_pointer

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
printf 'EVIDENCE_ROOT=%s\n' "${ROOT}"
