#!/usr/bin/env bash

# Verify the fixed SC'26 AE runtime contract, then run the explicitly selected
# grouped-gemm installer.  This entry point never discovers or switches a
# Python interpreter, Nsight binary, source method, or package version.

set -euo pipefail

readonly SC26_AE_SETUP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SC26_AE_SETUP_REPO_ROOT=$(cd -- "${SC26_AE_SETUP_DIR}/.." && pwd)
readonly SC26_AE_SETUP_INSTALLER="${SC26_AE_SETUP_REPO_ROOT}/tools/ae/setup_grouped_gemm_v1.sh"

# These paths are part of the v1.2-ae worker contract.  They are deliberately
# constants: a production invocation cannot override them through the shell
# environment or PATH lookup.
readonly SC26_AE_MEGATRON_PYTHON="/opt/conda/envs/megatron_env/bin/python"
readonly SC26_AE_ECHO_PYTHON="/opt/conda/envs/echo_slowdown/bin/python"
readonly SC26_AE_NSYS_BIN="/usr/local/bin/nsys"
readonly SC26_AE_NCU_BIN="/usr/local/cuda/bin/ncu"
readonly SC26_AE_ECHO_SOURCE="${SC26_AE_SETUP_REPO_ROOT}/Echo-slowdown"

readonly SC26_AE_MEGATRON_PYTHON_VERSION="3.9.18"
readonly SC26_AE_MEGATRON_TORCH_VERSION="2.1.2"
readonly SC26_AE_ECHO_PYTHON_VERSION="3.10.20"
readonly SC26_AE_ECHO_TORCH_VERSION="2.1.2+cu121"
readonly SC26_AE_TORCH_CUDA_VERSION="12.1"
readonly SC26_AE_ECHO_TORCHVISION_VERSION="0.16.2+cu121"
readonly SC26_AE_ECHO_TORCHAUDIO_VERSION="2.1.2+cu121"
readonly SC26_AE_NSYS_VERSION="2024.4.2.133"
readonly SC26_AE_NCU_VERSION="2024.3.2.3"

sc26_ae_setup_fail() {
    printf 'ERROR: %s\n' "$*" >&2
    return 1
}

sc26_ae_setup_validate_source() {
    local source_method=${1:-}
    case "${source_method}" in
        vcs|archive)
            ;;
        "")
            sc26_ae_setup_fail "GROUPED_GEMM_SOURCE must be set to 'vcs' or 'archive'"
            ;;
        *)
            sc26_ae_setup_fail \
                "Unsupported GROUPED_GEMM_SOURCE='${source_method}'; expected 'vcs' or 'archive'"
            ;;
    esac
}

sc26_ae_setup_field() {
    local field=$1
    local output=$2
    sed -n "s/^${field}=//p" <<< "${output}" | tail -n 1
}

sc26_ae_setup_require_field() {
    local label=$1
    local field=$2
    local expected=$3
    local output=$4
    local actual
    actual=$(sc26_ae_setup_field "${field}" "${output}")
    [[ "${actual}" == "${expected}" ]] || \
        sc26_ae_setup_fail \
            "${label} ${field} mismatch: expected '${expected}', got '${actual:-<missing>}'"
}

sc26_ae_setup_require_version_prefix() {
    local label=$1
    local field=$2
    local expected_base=$3
    local output=$4
    local actual
    actual=$(sc26_ae_setup_field "${field}" "${output}")
    case "${actual}" in
        "${expected_base}"|"${expected_base}"+*)
            ;;
        *)
            sc26_ae_setup_fail \
                "${label} ${field} mismatch: expected '${expected_base}' (optional CUDA build suffix), got '${actual:-<missing>}'"
            ;;
    esac
}

sc26_ae_setup_python_query() {
    local role=$1
    local python_bin=$2
    local echo_source=$3
    local require_grouped=$4
    local output query

    [[ -x "${python_bin}" ]] || \
        sc26_ae_setup_fail "${role} Python executable is missing or not executable: ${python_bin}" || return 1
    if [[ "${role}" == echo ]]; then
        [[ -d "${echo_source}" ]] || \
            sc26_ae_setup_fail "Pinned Echo source directory is missing: ${echo_source}" || return 1
        [[ -f "${echo_source}/training_testing/prediction_api.py" ]] || \
            sc26_ae_setup_fail \
                "Pinned Echo predictor source is missing: ${echo_source}/training_testing/prediction_api.py" || return 1
    fi

    query=$(cat <<'PY'
import importlib
import importlib.metadata
import os
import pathlib
import sys


def report(key, value):
    print(f"{key}={value}")


role = os.environ["SC26_AE_SETUP_ROLE"]
require_grouped = os.environ["SC26_AE_SETUP_REQUIRE_GROUPED"] == "1"
report("python_version", ".".join(str(part) for part in sys.version_info[:3]))

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised by the worker runtime
    raise RuntimeError(f"torch import failed: {exc}") from exc

report("torch_version", torch.__version__)
report("torch_cuda_version", torch.version.cuda or "")
report("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch reports CUDA unavailable")

try:
    import pynvml

    pynvml.nvmlInit()
    try:
        report("nvml_device_count", pynvml.nvmlDeviceGetCount())
    finally:
        pynvml.nvmlShutdown()
except Exception as exc:  # pragma: no cover - exercised by the worker runtime
    raise RuntimeError(f"NVML query failed: {exc}") from exc

for module_name in ("numpy", "pandas", "openpyxl", "xgboost", "sklearn", "torchvision"):
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised by the worker runtime
        raise RuntimeError(f"{module_name} import failed: {exc}") from exc
    report(f"{module_name}_import", True)

report("torchvision_version", importlib.metadata.version("torchvision"))

if role == "echo":
    for module_name in ("torchaudio", "transformers"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised by the worker runtime
            raise RuntimeError(f"{module_name} import failed: {exc}") from exc
        report(f"{module_name}_import", True)
    report("torchaudio_version", importlib.metadata.version("torchaudio"))
    try:
        from training_testing.prediction_api import SlowdownPredictor
    except Exception as exc:  # pragma: no cover - exercised by the worker runtime
        raise RuntimeError(f"SlowdownPredictor import failed: {exc}") from exc
    report("predictor_import", SlowdownPredictor.__name__ == "SlowdownPredictor")

if require_grouped:
    try:
        import grouped_gemm
        import grouped_gemm_backend

        backend_path = pathlib.Path(grouped_gemm_backend.__file__).resolve()
        if not backend_path.is_file():
            raise RuntimeError(f"grouped-gemm backend is not a file: {backend_path}")
        if getattr(grouped_gemm, "ops", None) is None:
            raise RuntimeError("grouped_gemm.ops is unavailable")
    except Exception as exc:  # pragma: no cover - exercised by the worker runtime
        raise RuntimeError(f"grouped-gemm import failed: {exc}") from exc
    report("grouped_gemm_import", True)
    report("grouped_gemm_backend", backend_path)
PY
)
    if [[ "${role}" == echo ]]; then
        if ! output=$( \
            SC26_AE_SETUP_ROLE="${role}" \
            SC26_AE_SETUP_REQUIRE_GROUPED="${require_grouped}" \
            PYTHONPATH="${echo_source}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${python_bin}" - <<<"${query}" 2>&1
        ); then
            printf '%s\n' "${output}" >&2
            sc26_ae_setup_fail "${role} runtime query failed for fixed interpreter: ${python_bin}" || return 1
        fi
    else
        if ! output=$( \
            SC26_AE_SETUP_ROLE="${role}" \
            SC26_AE_SETUP_REQUIRE_GROUPED="${require_grouped}" \
            "${python_bin}" - <<<"${query}" 2>&1
        ); then
            printf '%s\n' "${output}" >&2
            sc26_ae_setup_fail "${role} runtime query failed for fixed interpreter: ${python_bin}" || return 1
        fi
    fi
    printf '%s\n' "${output}"
}

sc26_ae_setup_verify_python_contract() {
    local role=$1
    local python_bin=$2
    local echo_source=$3
    local expected_python=$4
    local expected_torch=$5
    local expected_cuda=$6
    local require_grouped=$7
    local output nvml_count

    output=$(sc26_ae_setup_python_query \
        "${role}" "${python_bin}" "${echo_source}" "${require_grouped}") || return 1
    sc26_ae_setup_require_field "${role}" python_version "${expected_python}" "${output}" || return 1
    sc26_ae_setup_require_field "${role}" torch_version "${expected_torch}" "${output}" || return 1
    sc26_ae_setup_require_field "${role}" torch_cuda_version "${expected_cuda}" "${output}" || return 1
    sc26_ae_setup_require_field "${role}" cuda_available True "${output}" || return 1

    nvml_count=$(sc26_ae_setup_field nvml_device_count "${output}")
    [[ "${nvml_count}" =~ ^[1-9][0-9]*$ ]] || \
        sc26_ae_setup_fail \
            "${role} fixed runtime must expose at least one NVML device, got '${nvml_count:-<missing>}'" || return 1

    for module_name in numpy pandas openpyxl xgboost sklearn torchvision; do
        sc26_ae_setup_require_field \
            "${role}" "${module_name}_import" True "${output}" || return 1
    done

    if [[ "${role}" == echo ]]; then
        sc26_ae_setup_require_field echo predictor_import True "${output}" || return 1
        sc26_ae_setup_require_field echo torchaudio_import True "${output}" || return 1
        sc26_ae_setup_require_field echo transformers_import True "${output}" || return 1
        sc26_ae_setup_require_field \
            echo torchvision_version "${SC26_AE_ECHO_TORCHVISION_VERSION}" "${output}" || return 1
        sc26_ae_setup_require_field \
            echo torchaudio_version "${SC26_AE_ECHO_TORCHAUDIO_VERSION}" "${output}" || return 1
    else
        sc26_ae_setup_require_version_prefix \
            megatron torchvision_version "0.16.2" "${output}" || return 1
    fi

    if [[ "${require_grouped}" == 1 ]]; then
        local backend_path
        sc26_ae_setup_require_field megatron grouped_gemm_import True "${output}" || return 1
        backend_path=$(sc26_ae_setup_field grouped_gemm_backend "${output}")
        [[ -n "${backend_path}" && -f "${backend_path}" ]] || \
            sc26_ae_setup_fail \
                "grouped-gemm backend from the fixed Megatron runtime is missing: ${backend_path:-<missing>}" || return 1
    fi
}

sc26_ae_setup_verify_tools() {
    local nsys_bin=$1
    local ncu_bin=$2
    local nsys_version ncu_version ncu_help

    [[ -x "${nsys_bin}" ]] || \
        sc26_ae_setup_fail "Nsight Systems binary is missing or not executable: ${nsys_bin}" || return 1
    [[ -x "${ncu_bin}" ]] || \
        sc26_ae_setup_fail "Nsight Compute binary is missing or not executable: ${ncu_bin}" || return 1

    if ! nsys_version=$("${nsys_bin}" --version 2>&1); then
        printf '%s\n' "${nsys_version}" >&2
        sc26_ae_setup_fail "Nsight Systems version query failed: ${nsys_bin}" || return 1
    fi
    grep -Fq -- "${SC26_AE_NSYS_VERSION}" <<< "${nsys_version}" || \
        sc26_ae_setup_fail \
            "Nsight Systems version mismatch: expected ${SC26_AE_NSYS_VERSION}, got '${nsys_version}'" || return 1
    "${nsys_bin}" profile --help >/dev/null 2>&1 || \
        sc26_ae_setup_fail "Nsight Systems 'profile' command is unavailable: ${nsys_bin}" || return 1
    "${nsys_bin}" export --help >/dev/null 2>&1 || \
        sc26_ae_setup_fail "Nsight Systems 'export' command is unavailable: ${nsys_bin}" || return 1

    if ! ncu_version=$("${ncu_bin}" --version 2>&1); then
        printf '%s\n' "${ncu_version}" >&2
        sc26_ae_setup_fail "Nsight Compute version query failed: ${ncu_bin}" || return 1
    fi
    grep -Fq -- "${SC26_AE_NCU_VERSION}" <<< "${ncu_version}" || \
        sc26_ae_setup_fail \
            "Nsight Compute version mismatch: expected ${SC26_AE_NCU_VERSION}, got '${ncu_version}'" || return 1
    if ! ncu_help=$("${ncu_bin}" --help 2>&1); then
        printf '%s\n' "${ncu_help}" >&2
        sc26_ae_setup_fail "Nsight Compute help query failed: ${ncu_bin}" || return 1
    fi
    grep -Fq -- "--csv" <<< "${ncu_help}" || \
        sc26_ae_setup_fail "Nsight Compute lacks the required --csv command: ${ncu_bin}" || return 1
    grep -Fq -- "--log-file" <<< "${ncu_help}" || \
        sc26_ae_setup_fail "Nsight Compute lacks the required --log-file command: ${ncu_bin}" || return 1
}

# The arguments are explicit to keep the verifier testable with deterministic
# fixtures.  The production main function below always supplies the constants
# declared at the top of this file.
sc26_ae_setup_verify_runtime_contract() {
    [[ $# -eq 5 ]] || \
        sc26_ae_setup_fail \
            "runtime verifier expects megatron_python echo_python nsys_bin ncu_bin echo_source"
    local megatron_python=$1
    local echo_python=$2
    local nsys_bin=$3
    local ncu_bin=$4
    local echo_source=$5

    sc26_ae_setup_verify_python_contract \
        megatron "${megatron_python}" "${echo_source}" \
        "${SC26_AE_MEGATRON_PYTHON_VERSION}" \
        "${SC26_AE_MEGATRON_TORCH_VERSION}" \
        "${SC26_AE_TORCH_CUDA_VERSION}" 0 || return 1
    sc26_ae_setup_verify_python_contract \
        echo "${echo_python}" "${echo_source}" \
        "${SC26_AE_ECHO_PYTHON_VERSION}" \
        "${SC26_AE_ECHO_TORCH_VERSION}" \
        "${SC26_AE_TORCH_CUDA_VERSION}" 0 || return 1
    sc26_ae_setup_verify_tools "${nsys_bin}" "${ncu_bin}" || return 1
}

sc26_ae_setup_verify_grouped_gemm() {
    [[ $# -eq 2 ]] || \
        sc26_ae_setup_fail "grouped-gemm verifier expects megatron_python echo_source"
    local megatron_python=$1
    local echo_source=$2
    sc26_ae_setup_verify_python_contract \
        megatron "${megatron_python}" "${echo_source}" \
        "${SC26_AE_MEGATRON_PYTHON_VERSION}" \
        "${SC26_AE_MEGATRON_TORCH_VERSION}" \
        "${SC26_AE_TORCH_CUDA_VERSION}" 1 || return 1
}

sc26_ae_setup_main() {
    local source_method=${GROUPED_GEMM_SOURCE:-}
    sc26_ae_setup_validate_source "${source_method}" || return 1
    [[ -f "${SC26_AE_SETUP_INSTALLER}" ]] || \
        sc26_ae_setup_fail \
            "Grouped-gemm installer is missing: ${SC26_AE_SETUP_INSTALLER}" || return 1

    # Verify all base runtimes before any task command or package installation.
    sc26_ae_setup_verify_runtime_contract \
        "${SC26_AE_MEGATRON_PYTHON}" \
        "${SC26_AE_ECHO_PYTHON}" \
        "${SC26_AE_NSYS_BIN}" \
        "${SC26_AE_NCU_BIN}" \
        "${SC26_AE_ECHO_SOURCE}" || return 1

    export GROUPED_GEMM_SOURCE="${source_method}"
    # Pin the installer to the same fixed Megatron interpreter checked above.
    export GROUPED_GEMM_PYTHON="${SC26_AE_MEGATRON_PYTHON}"
    local installer_status
    if bash "${SC26_AE_SETUP_INSTALLER}"; then
        :
    else
        installer_status=$?
        return "${installer_status}"
    fi

    sc26_ae_setup_verify_grouped_gemm \
        "${SC26_AE_MEGATRON_PYTHON}" "${SC26_AE_ECHO_SOURCE}" || return 1
    printf 'SC26_AE_SETUP_STATUS=verified\n'
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sc26_ae_setup_main "$@"
fi
