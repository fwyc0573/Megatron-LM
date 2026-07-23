#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COMMON_SH="${REPO_ROOT}/SC26-AE/lib/common.sh"
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
    local expected=$1
    shift
    local output status
    set +e
    output=$("$@" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "expected failure: $*"
    grep -Fq -- "[ERROR]" <<< "${output}" || fail "missing [ERROR] prefix: ${output}"
    grep -Fq -- "${expected}" <<< "${output}" || fail "missing '${expected}': ${output}"
}

# shellcheck source=/dev/null
source "${COMMON_SH}"

ae_require_enum QUICK 1 0 1
expect_failure "QUICK" ae_require_enum QUICK 2 0 1
pass "enum validation"

ae_require_positive_int COUNT 7
expect_failure "COUNT" ae_require_positive_int COUNT 0
expect_failure "COUNT" ae_require_positive_int COUNT -1
expect_failure "COUNT" ae_require_positive_int COUNT abc
pass "positive integer validation"

ae_require_command bash
expect_failure "missing_sc26_command" ae_require_command missing_sc26_command
pass "command validation"

TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/data/ycfeng/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEMP_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-common.XXXXXX")
mkdir -p "${TEMP_ROOT}/dir"
: > "${TEMP_ROOT}/file"
ae_require_file "${TEMP_ROOT}/file"
ae_require_dir "${TEMP_ROOT}/dir"
expect_failure "Required file" ae_require_file "${TEMP_ROOT}/missing"
expect_failure "Required directory" ae_require_dir "${TEMP_ROOT}/missing"
pass "file and directory validation"

expected_default="${REPO_ROOT}/SC26-AE/output/qwen3_a30b/task1"
actual_default=$(env -u AE_OUTPUT_ROOT bash -c "source '${COMMON_SH}'; ae_model_output_dir qwen3_a30b task1")
[[ ${actual_default} == "${expected_default}" ]] || fail "unexpected default output: ${actual_default}"
actual_custom=$(AE_OUTPUT_ROOT="${TEMP_ROOT}/output with spaces" ae_model_output_dir dsv3 task3)
[[ ${actual_custom} == "${TEMP_ROOT}/output with spaces/dsv3/task3" ]] || fail "unexpected custom output: ${actual_custom}"
expect_failure "model key" ae_model_output_dir unknown task1
expect_failure "task key" ae_model_output_dir gpt175b task4
pass "safe output path construction"

repo_from_function=$(ae_repo_root)
[[ ${repo_from_function} == "${REPO_ROOT}" ]] || fail "unexpected repo root: ${repo_from_function}"
echo_commit=$(ae_source_commit Echo-slowdown)
[[ ${echo_commit} =~ ^[0-9a-f]{40}$ ]] || fail "invalid Echo source identity: ${echo_commit}"
ae_assert_source_clean Echo-slowdown
pass "source identity and clean vendored directory validation"

FAKE_BIN="${TEMP_ROOT}/fake-bin"
mkdir -p "${FAKE_BIN}"
cat > "${FAKE_BIN}/git" <<'SH'
#!/usr/bin/env bash
if [[ "$*" == *"status --short"* ]]; then
    printf ' M dirty-file\n'
    exit 0
fi
exec /usr/bin/git "$@"
SH
chmod +x "${FAKE_BIN}/git"
expect_failure "not clean" env PATH="${FAKE_BIN}:${PATH}" bash -c "source '${COMMON_SH}'; ae_assert_source_clean Echo-slowdown"
pass "dirty vendored directory rejection"

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 7 ]] || fail "expected 7 cases, got ${PASS_COUNT}"
