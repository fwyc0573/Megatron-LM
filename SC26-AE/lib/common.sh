#!/usr/bin/env bash

# Shared fail-fast helpers for the SC26 AE entry scripts.

ae_die() {
    printf '[ERROR] %s\n' "$*" >&2
    return 1
}

ae_require_command() {
    local command_name=${1:-}
    [[ -n "${command_name}" ]] || ae_die "Command name must not be empty."
    command -v -- "${command_name}" >/dev/null 2>&1 || \
        ae_die "Required command is missing: ${command_name}"
}

ae_require_file() {
    local path=${1:-}
    [[ -f "${path}" ]] || ae_die "Required file is missing: ${path}"
}

ae_require_dir() {
    local path=${1:-}
    [[ -d "${path}" ]] || ae_die "Required directory is missing: ${path}"
}

ae_require_enum() {
    local variable_name=${1:-}
    local value=${2:-}
    shift 2 || return 1
    local allowed_value

    [[ -n "${variable_name}" ]] || ae_die "Variable name must not be empty."
    for allowed_value in "$@"; do
        if [[ "${value}" == "${allowed_value}" ]]; then
            return 0
        fi
    done
    ae_die "${variable_name} must be one of: $*; got '${value}'."
}

ae_require_positive_int() {
    local variable_name=${1:-}
    local value=${2:-}
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        ae_die "${variable_name} must be a positive integer; got '${value}'."
}

ae_repo_root() {
    local common_dir
    common_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 1
    cd -- "${common_dir}/../.." && pwd
}

ae_model_output_dir() {
    local model_key=${1:-}
    local task_key=${2:-}
    local output_root

    ae_require_enum "model key" "${model_key}" gpt175b qwen3_a30b dsv3 || return 1
    ae_require_enum "task key" "${task_key}" task1 task2 task3 || return 1
    output_root=${AE_OUTPUT_ROOT:-"$(ae_repo_root)/SC26-AE/output"}
    [[ "${output_root}" == /* ]] || ae_die "AE_OUTPUT_ROOT must be an absolute path: ${output_root}"
    printf '%s/%s/%s\n' "${output_root%/}" "${model_key}" "${task_key}"
}

ae_validate_repo_relative_path() {
    local variable_name=$1
    local path=$2
    [[ -n "${path}" ]] || ae_die "${variable_name} must not be empty."
    [[ "${path}" != /* ]] || ae_die "${variable_name} must be repository-relative: ${path}"
    [[ "${path}" != *'..'* ]] || ae_die "${variable_name} must not contain '..': ${path}"
    [[ "${path}" != *$'\n'* ]] || ae_die "${variable_name} must not contain a newline."
}

ae_gitlink_commit() {
    local submodule_path=${1:-}
    local repo_root stage_line mode commit

    ae_validate_repo_relative_path "submodule path" "${submodule_path}" || return 1
    repo_root=$(ae_repo_root) || return 1
    stage_line=$(git -C "${repo_root}" ls-files --stage -- "${submodule_path}") || \
        ae_die "Failed to inspect submodule gitlink: ${submodule_path}"
    [[ $(wc -l <<< "${stage_line}") -eq 1 ]] || \
        ae_die "Expected exactly one gitlink entry for: ${submodule_path}"
    read -r mode commit _ <<< "${stage_line}"
    [[ "${mode}" == "160000" && "${commit}" =~ ^[0-9a-f]{40}$ ]] || \
        ae_die "Path is not a pinned gitlink: ${submodule_path}"
    printf '%s\n' "${commit}"
}

ae_assert_submodule_clean() {
    local submodule_path=${1:-}
    local repo_root status

    ae_validate_repo_relative_path "submodule path" "${submodule_path}" || return 1
    repo_root=$(ae_repo_root) || return 1
    ae_require_dir "${repo_root}/${submodule_path}" || return 1
    status=$(git -C "${repo_root}/${submodule_path}" status --short --untracked-files=all) || \
        ae_die "Failed to inspect submodule status: ${submodule_path}"
    [[ -z "${status}" ]] || ae_die "Submodule is not clean: ${submodule_path}"
}
