#!/usr/bin/env bash
# Local synthetic integration contract for the public setup wrapper.
# This test does not install grouped-gemm or qualify the AE runtime image.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SETUP_ENTRY="${REPO_ROOT}/SC26-AE/setup.sh"
EXPECTED_INSTALLER="${REPO_ROOT}/tools/ae/setup_grouped_gemm_v1.sh"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-setup-integration.XXXXXX")
FAKE_BIN="${TEST_ROOT}/bin"
CALL_LOG="${TEST_ROOT}/installer_calls.log"
PASS_COUNT=0

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$1"
}

assert_contains() {
    local needle=$1
    local path=$2
    grep -Fq -- "${needle}" "${path}" || {
        printf 'Expected %s to contain: %s\n' "${path}" "${needle}" >&2
        cat "${path}" >&2
        exit 1
    }
}

assert_equals() {
    local expected=$1
    local actual=$2
    local context=$3
    [[ "${actual}" == "${expected}" ]] || \
        fail "${context}: expected '${expected}', got '${actual}'"
}

installer_call_count() {
    wc -l < "${CALL_LOG}"
}

run_setup_entry() {
    local setup_entry=$1
    local source_method=$2
    local installer_status=$3
    shift 3
    env \
        PATH="${FAKE_BIN}:/usr/bin:/bin" \
        GROUPED_GEMM_SOURCE="${source_method}" \
        FAKE_SETUP_CALL_LOG="${CALL_LOG}" \
        FAKE_INSTALLER_STATUS="${installer_status}" \
        /usr/bin/bash -c '
            set -euo pipefail
            source "$1"
            # Runtime behavior is covered by the dedicated verifier test.  This
            # integration lane isolates source selection and installer status
            # propagation without discovering a host interpreter.
            sc26_ae_setup_verify_runtime_contract() { :; }
            sc26_ae_setup_verify_grouped_gemm() { :; }
            sc26_ae_setup_main
        ' -- "${setup_entry}" "$@"
}

run_setup() {
    local source_method=$1
    local installer_status=$2
    shift 2
    run_setup_entry "${SETUP_ENTRY}" "${source_method}" "${installer_status}" "$@"
}

[[ -x "${SETUP_ENTRY}" ]] || fail "setup entry is not executable: ${SETUP_ENTRY}"
[[ -f "${EXPECTED_INSTALLER}" ]] || fail "expected installer is missing: ${EXPECTED_INSTALLER}"

mkdir -p "${FAKE_BIN}"
: > "${CALL_LOG}"
cat > "${FAKE_BIN}/bash" <<'SH'
#!/usr/bin/bash
set -euo pipefail

printf '%s\t%s\t%s\n' \
    "${GROUPED_GEMM_SOURCE-}" "$#" "${1-}" >> "${FAKE_SETUP_CALL_LOG}"
printf 'FAKE_GROUPED_GEMM_INSTALLER_SOURCE=%s\n' "${GROUPED_GEMM_SOURCE-}"
printf 'FAKE_GROUPED_GEMM_INSTALLER_PYTHON=%s\n' "${GROUPED_GEMM_PYTHON-}"
exit "${FAKE_INSTALLER_STATUS:-0}"
SH
chmod +x "${FAKE_BIN}/bash"

before=$(installer_call_count)
set +e
env -u GROUPED_GEMM_SOURCE \
    PATH="${FAKE_BIN}:/usr/bin:/bin" \
    FAKE_SETUP_CALL_LOG="${CALL_LOG}" \
    /usr/bin/bash "${SETUP_ENTRY}" \
    > "${TEST_ROOT}/missing.stdout" 2> "${TEST_ROOT}/missing.stderr"
missing_status=$?
set -e
[[ ${missing_status} -ne 0 ]] || fail "missing source returned success"
assert_equals "${before}" "$(installer_call_count)" "missing-source installer calls"
assert_contains "GROUPED_GEMM_SOURCE must be set to 'vcs' or 'archive'" \
    "${TEST_ROOT}/missing.stderr"
pass "missing source fails before the installer boundary"

before=$(installer_call_count)
set +e
run_setup invalid 0 \
    > "${TEST_ROOT}/invalid.stdout" 2> "${TEST_ROOT}/invalid.stderr"
invalid_status=$?
set -e
[[ ${invalid_status} -ne 0 ]] || fail "invalid source returned success"
assert_equals "${before}" "$(installer_call_count)" "invalid-source installer calls"
assert_contains "Unsupported GROUPED_GEMM_SOURCE='invalid'" \
    "${TEST_ROOT}/invalid.stderr"
pass "invalid source fails before the installer boundary"

missing_repo="${TEST_ROOT}/missing-installer-repo"
missing_entry="${missing_repo}/SC26-AE/setup.sh"
mkdir -p "${missing_repo}/SC26-AE"
cp "${SETUP_ENTRY}" "${missing_entry}"
chmod +x "${missing_entry}"
before=$(installer_call_count)
set +e
run_setup_entry "${missing_entry}" vcs 0 \
    > "${TEST_ROOT}/missing-installer.stdout" \
    2> "${TEST_ROOT}/missing-installer.stderr"
missing_installer_status=$?
set -e
[[ ${missing_installer_status} -ne 0 ]] || fail "missing installer returned success"
assert_equals "${before}" "$(installer_call_count)" "missing-installer calls"
assert_contains "Grouped-gemm installer is missing: ${missing_repo}/tools/ae/setup_grouped_gemm_v1.sh" \
    "${TEST_ROOT}/missing-installer.stderr"
pass "missing pinned installer fails before the installer boundary"

before=$(installer_call_count)
run_setup vcs 0 > "${TEST_ROOT}/vcs.stdout" 2> "${TEST_ROOT}/vcs.stderr"
assert_equals "$((before + 1))" "$(installer_call_count)" "VCS installer calls"
assert_equals $'vcs\t1\t'"${EXPECTED_INSTALLER}" "$(tail -n 1 "${CALL_LOG}")" \
    "VCS installer forwarding"
assert_contains "FAKE_GROUPED_GEMM_INSTALLER_SOURCE=vcs" "${TEST_ROOT}/vcs.stdout"
assert_contains "FAKE_GROUPED_GEMM_INSTALLER_PYTHON=/opt/conda/envs/megatron_env/bin/python" \
    "${TEST_ROOT}/vcs.stdout"
assert_contains "SC26_AE_SETUP_STATUS=verified" "${TEST_ROOT}/vcs.stdout"
pass "VCS selection reaches exactly the pinned installer entry"

before=$(installer_call_count)
run_setup archive 0 > "${TEST_ROOT}/archive.stdout" 2> "${TEST_ROOT}/archive.stderr"
assert_equals "$((before + 1))" "$(installer_call_count)" "archive installer calls"
assert_equals $'archive\t1\t'"${EXPECTED_INSTALLER}" "$(tail -n 1 "${CALL_LOG}")" \
    "archive installer forwarding"
assert_contains "FAKE_GROUPED_GEMM_INSTALLER_SOURCE=archive" \
    "${TEST_ROOT}/archive.stdout"
assert_contains "FAKE_GROUPED_GEMM_INSTALLER_PYTHON=/opt/conda/envs/megatron_env/bin/python" \
    "${TEST_ROOT}/archive.stdout"
assert_contains "SC26_AE_SETUP_STATUS=verified" "${TEST_ROOT}/archive.stdout"
pass "archive selection reaches exactly the pinned installer entry"

before=$(installer_call_count)
set +e
run_setup archive 23 \
    > "${TEST_ROOT}/downstream-failure.stdout" \
    2> "${TEST_ROOT}/downstream-failure.stderr"
downstream_status=$?
set -e
assert_equals "23" "${downstream_status}" "downstream installer status"
assert_equals "$((before + 1))" "$(installer_call_count)" \
    "downstream-failure installer calls"
assert_equals $'archive\t1\t'"${EXPECTED_INSTALLER}" "$(tail -n 1 "${CALL_LOG}")" \
    "downstream-failure installer forwarding"
pass "selected installer failure propagates without another source attempt"

printf 'EVIDENCE_CLASS=local_synthetic_setup_contract\n'
printf 'REAL_INSTALLER_EXECUTION_COUNT=0\n'
printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
printf 'EVIDENCE_ROOT=%s\n' "${TEST_ROOT}"
[[ ${PASS_COUNT} -eq 6 ]] || fail "expected 6 cases, got ${PASS_COUNT}"
