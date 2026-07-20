#!/usr/bin/env bash
# Task2 reuse must not cross real/synthetic evidence boundaries.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="${REPO_ROOT}/SC26-AE/lib/task2_echo.sh"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
LIBRARY=$(mktemp "${TMP_PARENT%/}/sc26-ae-task2-evidence.XXXXXX")
sed '/^task2_main$/,$d' "${RUNNER}" >"${LIBRARY}"
# shellcheck disable=SC1090
source "${LIBRARY}"

set +e
OUTPUT=$(TASK2_MODE=real task2_validate_evidence_for_mode \
    local_synthetic_not_two_gpu_qualification 2>&1)
STATUS=$?
set -e
[[ ${STATUS} -ne 0 ]] || {
    printf 'real mode accepted synthetic Task2 evidence\n' >&2
    exit 1
}
grep -Fq 'real mode requires real_exact_two_h800_qualified' <<<"${OUTPUT}" || {
    printf '%s\n' "${OUTPUT}" >&2
    exit 1
}

TASK2_MODE=synthetic
task2_validate_evidence_for_mode local_synthetic_not_two_gpu_qualification
task2_validate_evidence_for_mode real_exact_two_h800_qualified

RUN_ROOT="${TMP_PARENT%/}/sc26-ae-task2-evidence-run.$$"
mkdir -p "${RUN_ROOT}"
printf '%s\n' '{"execution_evidence":"local_synthetic_not_two_gpu_qualification"}' \
    >"${RUN_ROOT}/artifact_manifest.json"
TASK2_META_PYTHON=python3
set +e
OUTPUT=$(TASK2_MODE=real task2_validate_reuse_evidence "${RUN_ROOT}" 2>&1)
STATUS=$?
set -e
[[ ${STATUS} -ne 0 ]] || {
    printf 'real reuse accepted a synthetic predictor manifest\n' >&2
    exit 1
}
grep -Fq 'real mode requires real_exact_two_h800_qualified' <<<"${OUTPUT}"

cat >"${RUN_ROOT}/artifact_manifest.json" <<'JSON'
{"execution_evidence":"local_synthetic_not_two_gpu_qualification", "execution_evidence":"real_exact_two_h800_qualified"}
JSON
set +e
OUTPUT=$(TASK2_MODE=synthetic task2_validate_reuse_evidence "${RUN_ROOT}" 2>&1)
STATUS=$?
set -e
[[ ${STATUS} -ne 0 ]] || {
    printf 'synthetic reuse accepted a duplicate execution_evidence key\n' >&2
    exit 1
}
grep -Fq 'cannot read Task2 execution evidence for reuse' <<<"${OUTPUT}"

printf 'PASS_COUNT=5\n'
