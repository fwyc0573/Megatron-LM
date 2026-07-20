#!/usr/bin/env bash

# Contract tests for the fixed-runtime SC'26 AE setup verifier.
# The test uses deterministic fake interpreters/tools and never installs a
# package or starts a GPU workload.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SETUP_ENTRY="${REPO_ROOT}/SC26-AE/setup.sh"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-setup-runtime.XXXXXX")
FAKE_ECHO_SOURCE="${TEST_ROOT}/Echo-slowdown"
FAKE_BACKEND="${TEST_ROOT}/grouped_gemm_backend.so"
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

assert_not_contains() {
    local needle=$1
    local path=$2
    if grep -Fq -- "${needle}" "${path}"; then
        printf 'Expected %s to omit: %s\n' "${path}" "${needle}" >&2
        cat "${path}" >&2
        exit 1
    fi
}

[[ -f "${SETUP_ENTRY}" ]] || fail "setup entry is missing"

# RED contract: the old wrapper only forwarded GROUPED_GEMM_SOURCE and had no
# fixed-runtime verifier or verified completion marker.  These assertions are
# intentionally checked before any fake runtime is created.
assert_contains 'SC26_AE_MEGATRON_PYTHON=' "${SETUP_ENTRY}"
assert_contains 'SC26_AE_ECHO_PYTHON=' "${SETUP_ENTRY}"
assert_contains 'SC26_AE_NSYS_BIN=' "${SETUP_ENTRY}"
assert_contains 'SC26_AE_NCU_BIN=' "${SETUP_ENTRY}"
assert_contains 'sc26_ae_setup_verify_runtime_contract' "${SETUP_ENTRY}"
assert_contains 'SC26_AE_SETUP_STATUS=verified' "${SETUP_ENTRY}"
assert_not_contains 'command -v.*python' "${SETUP_ENTRY}"
pass "fixed runtime verifier contract is present"

mkdir -p "${FAKE_ECHO_SOURCE}/training_testing"
: > "${FAKE_ECHO_SOURCE}/training_testing/prediction_api.py"
: > "${FAKE_BACKEND}"

cat > "${TEST_ROOT}/fake-python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

profile=${FAKE_RUNTIME_PROFILE:-valid}
role=${SC26_AE_SETUP_ROLE:-}
backend=${FAKE_GROUPED_GEMM_BACKEND:?FAKE_GROUPED_GEMM_BACKEND is required}

case "${role}:${profile}" in
    megatron:valid|megatron:missing_predictor)
        python_version=3.9.18; torch_version=2.1.2; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    megatron:wrong_python)
        python_version=3.10.20; torch_version=2.1.2; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    megatron:wrong_torch)
        python_version=3.9.18; torch_version=2.2.0; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    megatron:wrong_cuda)
        python_version=3.9.18; torch_version=2.1.2; torch_cuda=11.8; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    megatron:cuda_unavailable)
        python_version=3.9.18; torch_version=2.1.2; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    megatron:missing_module)
        python_version=3.9.18; torch_version=2.1.2; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    megatron:zero_nvml)
        python_version=3.9.18; torch_version=2.1.2; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=0
        ;;
    echo:valid)
        python_version=3.10.20; torch_version=2.1.2+cu121; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    echo:wrong_python)
        python_version=3.9.18; torch_version=2.1.2+cu121; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    echo:wrong_torch)
        python_version=3.10.20; torch_version=2.0.0+cu121; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=True; nvml=2
        ;;
    echo:missing_predictor)
        python_version=3.10.20; torch_version=2.1.2+cu121; torch_cuda=12.1; torchvision=0.16.2+cu121; predictor=False; nvml=2
        ;;
    *)
        printf 'unknown fake runtime profile: %s:%s\n' "${role}" "${profile}" >&2
        exit 91
        ;;
esac

cat <<EOF
python_version=${python_version}
torch_version=${torch_version}
torch_cuda_version=${torch_cuda}
cuda_available=$([[ "${profile}" == cuda_unavailable ]] && printf 'False' || printf 'True')
nvml_device_count=${nvml}
numpy_import=True
pandas_import=$([[ "${profile}" == missing_module ]] && printf 'False' || printf 'True')
openpyxl_import=True
xgboost_import=True
sklearn_import=True
torchvision_import=True
torchvision_version=${torchvision}
torchaudio_import=True
torchaudio_version=2.1.2+cu121
transformers_import=True
predictor_import=${predictor}
grouped_gemm_import=True
grouped_gemm_backend=${backend}
EOF
SH
chmod +x "${TEST_ROOT}/fake-python"

cat > "${TEST_ROOT}/fake-nsys" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    --version)
        printf 'NVIDIA Nsight Systems version %s\n' "${FAKE_NSYS_VERSION:-2024.4.2.133}"
        ;;
    profile|export)
        case "${FAKE_NSYS_CAPABILITY:-valid}:${1}" in
            valid:*) ;;
            missing_profile:profile|missing_export:export) exit 92 ;;
            *) ;;
        esac
        [[ "${2:-}" == --help ]] || { printf 'fake nsys %s\n' "$1"; }
        ;;
    *)
        printf 'profile export\n'
        ;;
esac
SH
chmod +x "${TEST_ROOT}/fake-nsys"

cat > "${TEST_ROOT}/fake-ncu" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    --version)
        printf 'NVIDIA Nsight Compute CLI %s\n' "${FAKE_NCU_VERSION:-2024.3.2.3}"
        ;;
    --help)
        if [[ "${FAKE_NCU_CAPABILITY:-valid}" == missing_csv ]]; then
            printf '%s\n' '--log-file --page --export'
        elif [[ "${FAKE_NCU_CAPABILITY:-valid}" == missing_log_file ]]; then
            printf '%s\n' '--csv --page --export'
        else
            printf '%s\n' '--csv --log-file --page --export'
        fi
        ;;
    *)
        printf '%s\n' '--csv --log-file'
        ;;
esac
SH
chmod +x "${TEST_ROOT}/fake-ncu"

cat > "${TEST_ROOT}/fake-megatron-python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
FAKE_RUNTIME_PROFILE="${FAKE_MEGATRON_PROFILE:-valid}" \
    FAKE_RUNTIME_ROLE=megatron \
    exec "$(dirname -- "$0")/fake-python" "$@"
SH
chmod +x "${TEST_ROOT}/fake-megatron-python"

cat > "${TEST_ROOT}/fake-echo-python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
FAKE_RUNTIME_PROFILE="${FAKE_ECHO_PROFILE:-valid}" \
    FAKE_RUNTIME_ROLE=echo \
    exec "$(dirname -- "$0")/fake-python" "$@"
SH
chmod +x "${TEST_ROOT}/fake-echo-python"

# The verifier accepts explicit paths only through its source-safe test seam;
# production main always supplies the fixed constants from setup.sh.
# shellcheck source=/dev/null
source "${SETUP_ENTRY}"

run_runtime_contract() {
    local profile=$1
    export FAKE_MEGATRON_PROFILE="${profile}"
    export FAKE_ECHO_PROFILE="${profile}"
    export FAKE_GROUPED_GEMM_BACKEND="${FAKE_BACKEND}"
    sc26_ae_setup_verify_runtime_contract \
        "${TEST_ROOT}/fake-megatron-python" \
        "${TEST_ROOT}/fake-echo-python" \
        "${TEST_ROOT}/fake-nsys" \
        "${TEST_ROOT}/fake-ncu" \
        "${FAKE_ECHO_SOURCE}"
}

run_runtime_contract_profiles() {
    export FAKE_MEGATRON_PROFILE=$1
    export FAKE_ECHO_PROFILE=$2
    export FAKE_GROUPED_GEMM_BACKEND="${FAKE_BACKEND}"
    sc26_ae_setup_verify_runtime_contract \
        "${TEST_ROOT}/fake-megatron-python" \
        "${TEST_ROOT}/fake-echo-python" \
        "${TEST_ROOT}/fake-nsys" \
        "${TEST_ROOT}/fake-ncu" \
        "${FAKE_ECHO_SOURCE}"
}

run_expect_failure() {
    local profile=$1
    local expected=$2
    local output status
    set +e
    output=$(run_runtime_contract "${profile}" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${profile} unexpectedly passed"
    grep -Fq -- "${expected}" <<< "${output}" || {
        printf 'Expected %s failure to contain: %s\n%s\n' "${profile}" "${expected}" "${output}" >&2
        exit 1
    }
    pass "${profile} fails closed"
}

run_runtime_contract valid
pass "valid fixed runtime contract passes"
run_expect_failure wrong_python "python_version mismatch"
run_expect_failure wrong_torch "torch_version mismatch"
run_expect_failure wrong_cuda "torch_cuda_version mismatch"
run_expect_failure cuda_unavailable "cuda_available mismatch"
run_expect_failure missing_module "pandas_import mismatch"
run_expect_failure zero_nvml "at least one NVML device"
run_expect_failure missing_predictor "predictor_import"

set +e
output=$(run_runtime_contract_profiles valid wrong_python 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "Echo wrong_python unexpectedly passed"
grep -Fq -- "echo python_version mismatch" <<< "${output}" || fail "Echo wrong_python error missing"
pass "Echo wrong Python version fails closed"

set +e
output=$(run_runtime_contract_profiles valid wrong_torch 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "Echo wrong_torch unexpectedly passed"
grep -Fq -- "echo torch_version mismatch" <<< "${output}" || fail "Echo wrong_torch error missing"
pass "Echo wrong torch version fails closed"

export FAKE_MEGATRON_PROFILE=valid FAKE_ECHO_PROFILE=valid
sc26_ae_setup_verify_grouped_gemm \
    "${TEST_ROOT}/fake-megatron-python" "${FAKE_ECHO_SOURCE}"
pass "grouped-gemm import and backend path are verified after install"

for missing_tool in fake-nsys fake-ncu; do
    local_nsys="${TEST_ROOT}/fake-nsys"
    local_ncu="${TEST_ROOT}/fake-ncu"
    if [[ "${missing_tool}" == fake-nsys ]]; then
        local_nsys="${TEST_ROOT}/missing-nsys"
    else
        local_ncu="${TEST_ROOT}/missing-ncu"
    fi
    set +e
    output=$(sc26_ae_setup_verify_tools "${local_nsys}" "${local_ncu}" 2>&1)
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${missing_tool} unexpectedly passed"
    assert_contains "binary is missing" <(printf '%s\n' "${output}")
    pass "${missing_tool} missing path fails closed"
done

set +e
export FAKE_NSYS_VERSION=2024.4.2.132
output=$(sc26_ae_setup_verify_tools \
    "${TEST_ROOT}/fake-nsys" "${TEST_ROOT}/fake-ncu" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "wrong nsys version unexpectedly passed"
grep -Fq -- "Nsight Systems version mismatch" <<< "${output}" || fail "wrong nsys version error missing"
unset FAKE_NSYS_VERSION
pass "wrong Nsight Systems version fails closed"

set +e
export FAKE_NCU_VERSION=2024.3.2.2
output=$(sc26_ae_setup_verify_tools \
    "${TEST_ROOT}/fake-nsys" "${TEST_ROOT}/fake-ncu" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "wrong ncu version unexpectedly passed"
grep -Fq -- "Nsight Compute version mismatch" <<< "${output}" || fail "wrong ncu version error missing"
unset FAKE_NCU_VERSION
pass "wrong Nsight Compute version fails closed"

set +e
export FAKE_NCU_CAPABILITY=missing_csv
output=$(sc26_ae_setup_verify_tools \
    "${TEST_ROOT}/fake-nsys" "${TEST_ROOT}/fake-ncu" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "missing ncu csv capability unexpectedly passed"
grep -Fq -- "--csv" <<< "${output}" || fail "missing ncu csv error missing"
unset FAKE_NCU_CAPABILITY
pass "missing Nsight Compute command capability fails closed"

set +e
export FAKE_NSYS_CAPABILITY=missing_profile
output=$(sc26_ae_setup_verify_tools \
    "${TEST_ROOT}/fake-nsys" "${TEST_ROOT}/fake-ncu" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "missing nsys profile capability unexpectedly passed"
grep -Fq -- "'profile' command" <<< "${output}" || fail "missing nsys profile error missing"
unset FAKE_NSYS_CAPABILITY
pass "missing Nsight Systems profile capability fails closed"

set +e
export FAKE_NSYS_CAPABILITY=missing_export
output=$(sc26_ae_setup_verify_tools \
    "${TEST_ROOT}/fake-nsys" "${TEST_ROOT}/fake-ncu" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "missing nsys export capability unexpectedly passed"
grep -Fq -- "'export' command" <<< "${output}" || fail "missing nsys export error missing"
unset FAKE_NSYS_CAPABILITY
pass "missing Nsight Systems export capability fails closed"

set +e
export FAKE_NCU_CAPABILITY=missing_log_file
output=$(sc26_ae_setup_verify_tools \
    "${TEST_ROOT}/fake-nsys" "${TEST_ROOT}/fake-ncu" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "missing ncu log-file capability unexpectedly passed"
grep -Fq -- "--log-file" <<< "${output}" || fail "missing ncu log-file error missing"
unset FAKE_NCU_CAPABILITY
pass "missing Nsight Compute log-file capability fails closed"

PATH_ROOT="${TEST_ROOT}/path-bin"
mkdir -p "${PATH_ROOT}"
ln -s "${TEST_ROOT}/fake-megatron-python" "${PATH_ROOT}/python"
ln -s "${TEST_ROOT}/fake-nsys" "${PATH_ROOT}/nsys"
ln -s "${TEST_ROOT}/fake-ncu" "${PATH_ROOT}/ncu"
set +e
output=$(env PATH="${PATH_ROOT}:/usr/bin:/bin" GROUPED_GEMM_SOURCE=vcs \
    /usr/bin/bash "${SETUP_ENTRY}" 2>&1)
status=$?
set -e
[[ ${status} -ne 0 ]] || fail "setup unexpectedly discovered PATH runtimes"
grep -Fq -- "/opt/conda/envs/megatron_env/bin/python" <<< "${output}" || \
    fail "PATH-discovery rejection did not mention the fixed Megatron path"
pass "PATH interpreter/tool discovery is rejected"

printf 'EVIDENCE_CLASS=local_synthetic_setup_runtime_contract\n'
printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 21 ]] || fail "expected 21 contract cases, got ${PASS_COUNT}"
