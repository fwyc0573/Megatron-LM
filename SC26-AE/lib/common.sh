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

ae_source_commit() {
    local source_path=${1:-}
    local repo_root stage_line mode commit source_file

    ae_validate_repo_relative_path "source path" "${source_path}" || return 1
    repo_root=$(ae_repo_root) || return 1
    source_file="${repo_root}/${source_path}/.source_commit"
    if [[ -f "${source_file}" ]]; then
        commit=$(tr -d '[:space:]' <"${source_file}") || \
            ae_die "Failed to read source identity: ${source_file}"
        [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || {
            ae_die "Invalid source identity in ${source_file}: ${commit}"
            return 1
        }
        printf '%s\n' "${commit}"
        return 0
    fi

    stage_line=$(git -C "${repo_root}" ls-files --stage -- "${source_path}") || \
        ae_die "Failed to inspect source directory: ${source_path}"
    [[ $(wc -l <<< "${stage_line}") -eq 1 ]] || \
        ae_die "Expected one source identity or source identity entry for: ${source_path}"
    read -r mode commit _ <<< "${stage_line}"
    [[ "${mode}" == "160000" && "${commit}" =~ ^[0-9a-f]{40}$ ]] || \
        ae_die "Path is neither a source directory with .source_commit nor a pinned source identity: ${source_path}"
    printf '%s\n' "${commit}"
}

ae_assert_source_clean() {
    local source_path=${1:-}
    local repo_root status source_file source_commit

    ae_validate_repo_relative_path "source path" "${source_path}" || return 1
    repo_root=$(ae_repo_root) || return 1
    ae_require_dir "${repo_root}/${source_path}" || return 1
    source_file="${repo_root}/${source_path}/.source_commit"
    if [[ -f "${source_file}" ]]; then
        source_commit=$(tr -d '[:space:]' <"${source_file}") || \
            ae_die "Failed to read source identity: ${source_file}"
        [[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || {
            ae_die "Invalid source identity in ${source_file}: ${source_commit}"
            return 1
        }
        [[ ! -e "${repo_root}/${source_path}/.git" ]] || {
            ae_die "Vendored source directory must not contain Git metadata: ${source_path}"
            return 1
        }
        status=$(git -C "${repo_root}" status --short --untracked-files=all -- "${source_path}") || \
            ae_die "Failed to inspect vendored source status: ${source_path}"
        [[ -z "${status}" ]] || {
            ae_die "Vendored source directory is not clean: ${source_path}"
            return 1
        }
        return 0
    fi

    status=$(git -C "${repo_root}/${source_path}" status --short --untracked-files=all) || \
        ae_die "Failed to inspect source status: ${source_path}"
    [[ -z "${status}" ]] || ae_die "Source checkout is not clean: ${source_path}"
}
