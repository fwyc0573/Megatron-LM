#!/usr/bin/env bash

# Shared SC26 AE Task3 runner.
#
# The public entry scripts call ae_run_task3 with one frozen model key.  This
# runner never chooses between fresh and prebaked artifacts: ARTIFACT_SOURCE is
# mandatory, and every selected-source validation failure is terminal.

set -euo pipefail

TASK3_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK3_REPO_ROOT=$(cd -- "${TASK3_LIB_DIR}/../.." && pwd)

if ! declare -F ae_die >/dev/null 2>&1; then
    # shellcheck source=common.sh
    source "${TASK3_LIB_DIR}/common.sh"
fi

task3_error() {
    ae_die "$*"
}

task3_execution_evidence() {
    case "${TASK3_EXECUTION_MODE:-}" in
        synthetic)
            printf '%s\n' 'local_synthetic_not_gpu_qualification'
            ;;
        real)
            # A real Task3 producer run is not an external qualification
            # attestation by itself.  The sealer promotes this pending label
            # only after independently attested H800 evidence is supplied.
            printf '%s\n' 'runtime_measurement_requires_external_single_gpu_qualification'
            ;;
        *)
            task3_error "Unsupported Task3 execution mode: ${TASK3_EXECUTION_MODE:-<unset>}"
            return 1
            ;;
    esac
}

task3_validate_fixed_interpreter() {
    local requested_path=$1
    local canonical_path
    canonical_path=$(realpath -e -- "${requested_path}" 2>/dev/null) || \
        {
            task3_error \
                "Fixed real-mode Task3 interpreter must be an executable regular file: ${requested_path}"
            return 1
        }
    [[ -f "${canonical_path}" && ! -L "${canonical_path}" && -x "${canonical_path}" ]] || \
        {
            task3_error \
                "Fixed real-mode Task3 interpreter must be an executable regular file: ${requested_path}"
            return 1
        }
}

task3_bind_interpreters() {
    local fixed_python=/opt/conda/envs/megatron_env/bin/python
    local meta_overridden=0
    local simulator_overridden=0
    [[ ${TASK3_META_PYTHON+x} == x ]] && meta_overridden=1
    [[ ${TASK3_SIMULATOR_PYTHON+x} == x ]] && simulator_overridden=1

    case "${TASK3_EXECUTION_MODE:-}" in
        real)
            if ((meta_overridden)); then
                [[ "${TASK3_META_PYTHON}" == "${fixed_python}" ]] || \
                    {
                        task3_error \
                            "TASK3_META_PYTHON overrides are forbidden in real mode; use ${fixed_python}"
                        return 1
                    }
            fi
            if ((simulator_overridden)); then
                [[ "${TASK3_SIMULATOR_PYTHON}" == "${fixed_python}" ]] || \
                    {
                        task3_error \
                            "TASK3_SIMULATOR_PYTHON overrides are forbidden in real mode; use ${fixed_python}"
                        return 1
                    }
            fi
            TASK3_META_PYTHON=${fixed_python}
            TASK3_SIMULATOR_PYTHON=${fixed_python}
            task3_validate_fixed_interpreter "${fixed_python}" || return 1
            ;;
        synthetic)
            TASK3_META_PYTHON=${TASK3_META_PYTHON:-python3}
            TASK3_SIMULATOR_PYTHON=${TASK3_SIMULATOR_PYTHON:-/opt/conda/envs/megatron_env/bin/python}
            ;;
        *)
            task3_error "Unsupported Task3 execution mode: ${TASK3_EXECUTION_MODE:-<unset>}"
            return 1
            ;;
    esac
}

task3_safe_id() {
    local variable_name=$1
    local value=$2
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
        task3_error "${variable_name} must be a path-free identifier: ${value}"
}

# Prepare a builder-only trace directory containing the global rank-0 files.
# The scheduler still consumes the complete Task1 trace directory.
task3_prepare_rank0_slowdown_trace() {
    local source_dir=$1
    local destination_dir=$2

    "${TASK3_META_PYTHON:-python3}" - "${source_dir}" "${destination_dir}" <<'PY'
import pathlib
import re
import shutil
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
if source.is_symlink() or not source.is_dir():
    raise SystemExit(f"[ERROR] slowdown source trace directory is invalid: {source}")
if destination.exists() or destination.is_symlink():
    raise SystemExit(f"[ERROR] slowdown rank-0 trace destination already exists: {destination}")
destination.mkdir(parents=True, exist_ok=False)
rank_pattern = re.compile(r"(?:^|_)rank([0-9]+)(?:_|\.)")
copied = []
for path in sorted(source.iterdir()):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"[ERROR] slowdown source trace contains a non-regular file: {path}")
    if path.suffix != ".txt":
        continue
    match = rank_pattern.search(path.name)
    if match is None:
        raise SystemExit(f"[ERROR] cannot determine rank from slowdown trace: {path}")
    if int(match.group(1)) == 0:
        target = destination / path.name
        shutil.copyfile(path, target)
        copied.append(target)
if not copied:
    raise SystemExit("[ERROR] slowdown trace directory contains no global rank-0 trace")
for path in destination.iterdir():
    match = rank_pattern.search(path.name)
    if match is None or int(match.group(1)) != 0:
        raise SystemExit(f"[ERROR] rank-0 slowdown trace directory contains a non-rank-0 file: {path}")
print("SLOWDOWN_TRACE_RANK_SCOPE=global_rank_0")
print(f"SLOWDOWN_TRACE_FILE_COUNT={len(copied)}")
PY
}

task3_assert_output_subtree_safe() {
    local output_root=$1
    local model_key=$2

    "${TASK3_META_PYTHON}" - "${output_root}" "${model_key}" <<'PY'
import pathlib
import sys

root_path = pathlib.Path(sys.argv[1])
root = root_path.resolve(strict=True)
model = sys.argv[2]
for relative in (
    pathlib.PurePosixPath(model),
    pathlib.PurePosixPath(model) / "task1",
    pathlib.PurePosixPath(model) / "task1" / "runs",
    pathlib.PurePosixPath(model) / "task3",
    pathlib.PurePosixPath(model) / "task3" / "runs",
    pathlib.PurePosixPath("_work"),
):
    candidate = root_path / relative
    if candidate.is_symlink():
        raise SystemExit(f"[ERROR] Task3 output path contains a symlink: {candidate}")
    if not candidate.exists():
        continue
    if not candidate.is_dir():
        raise SystemExit(f"[ERROR] Task3 output path is not a directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"[ERROR] Task3 output path escapes AE_OUTPUT_ROOT: {candidate}"
        )
PY
}

task3_load_model_config() {
    local model_key=$1

    TASK3_LOCAL_SIZE=8
    TASK3_MICRO_BATCH_SIZE=1
    case "${model_key}" in
        gpt175b)
            TASK3_PROFILE=175
            TASK3_WORLD_SIZE=1024
            TASK3_PP=8
            TASK3_TP=8
            TASK3_DP=16
            TASK3_EXP=1
            TASK3_NUM_EXPERTS=1
            TASK3_GLOBAL_BATCH_SIZE=768
            TASK3_SEQ_LEN=2048
            TASK3_HIDDEN_SIZE=12288
            TASK3_UNTIE_EMBEDDINGS=0
            ;;
        qwen3_a30b)
            TASK3_PROFILE=full
            TASK3_WORLD_SIZE=256
            TASK3_PP=8
            TASK3_TP=8
            TASK3_DP=4
            TASK3_EXP=4
            TASK3_NUM_EXPERTS=128
            TASK3_GLOBAL_BATCH_SIZE=128
            TASK3_SEQ_LEN=256
            TASK3_HIDDEN_SIZE=2048
            TASK3_UNTIE_EMBEDDINGS=1
            ;;
        dsv3)
            TASK3_PROFILE=smoke
            TASK3_WORLD_SIZE=256
            TASK3_PP=4
            TASK3_TP=8
            TASK3_DP=8
            TASK3_EXP=8
            TASK3_NUM_EXPERTS=32
            TASK3_GLOBAL_BATCH_SIZE=128
            TASK3_SEQ_LEN=256
            TASK3_HIDDEN_SIZE=2048
            TASK3_UNTIE_EMBEDDINGS=1
            ;;
        *)
            task3_error "Unknown Task3 model key: ${model_key}"
            ;;
    esac
}

task3_current_commits() {
    TASK3_MAIN_COMMIT=$(git -C "${TASK3_REPO_ROOT}" rev-parse HEAD 2>/dev/null) || \
        task3_error "Cannot resolve the current Megatron-LM commit."
    TASK3_ECHO_COMMIT=$(ae_source_commit Echo-slowdown) || \
        task3_error "Cannot resolve the Echo-slowdown source identity."
    TASK3_SIM_COMMIT=$(ae_source_commit megatron-sim-engine) || \
        task3_error "Cannot resolve the megatron-sim-engine source identity."
    [[ "${TASK3_MAIN_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || task3_error "Invalid Megatron-LM commit."
    [[ "${TASK3_ECHO_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || task3_error "Invalid Echo-slowdown commit."
    [[ "${TASK3_SIM_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || task3_error "Invalid megatron-sim-engine commit."

    task3_assert_sim_engine_provenance
}

task3_source_compatibility_mode() {
    local label=$1
    local recorded_main_commit=$2
    local recorded_echo_commit=$3
    local recorded_sim_commit=$4
    local commit_pattern='^[0-9a-f]{40}$'

    [[ "${recorded_main_commit}" =~ ${commit_pattern} ]] || {
        task3_error "${label} Megatron-LM commit is invalid: ${recorded_main_commit}"
        return 1
    }
    [[ "${recorded_echo_commit}" =~ ${commit_pattern} ]] || {
        task3_error "${label} Echo-slowdown commit is invalid: ${recorded_echo_commit}"
        return 1
    }
    [[ "${recorded_sim_commit}" =~ ${commit_pattern} ]] || {
        task3_error "${label} megatron-sim-engine commit is invalid: ${recorded_sim_commit}"
        return 1
    }

    if [[ "${recorded_main_commit}" == "${TASK3_MAIN_COMMIT}" &&
          "${recorded_echo_commit}" == "${TASK3_ECHO_COMMIT}" &&
          "${recorded_sim_commit}" == "${TASK3_SIM_COMMIT}" ]]; then
        printf '%s\n' exact
        return 0
    fi

    [[ "${recorded_echo_commit}" == "${TASK3_ECHO_COMMIT}" ]] || {
        task3_error "${label} Echo-slowdown commit differs from the current producer."
        return 1
    }
    git -C "${TASK3_REPO_ROOT}" merge-base --is-ancestor \
        "${recorded_main_commit}" "${TASK3_MAIN_COMMIT}" >/dev/null 2>&1 || {
        task3_error "${label} outer commit is not an ancestor of the current producer."
        return 1
    }
    local changed_path
    local simulator_source_identity_changed=0
    while IFS= read -r changed_path; do
        [[ -n "${changed_path}" ]] || continue
        case "${changed_path}" in
            megatron-sim-engine)
                simulator_source_identity_changed=1
                ;;
            SC26-AE/lib/task3_simulation.sh|\
            tests/unit/test_sc26_ae_task3_source_compatibility.sh|\
            tests/unit/test_sc26_ae_task3_trace_expansion.sh)
                ;;
            *)
                task3_error \
                    "${label} producer advancement changes files outside the Task3 compatibility allowlist: ${changed_path}"
                return 1
                ;;
        esac
    done < <(
        git -C "${TASK3_REPO_ROOT}" diff --name-only --no-ext-diff \
            "${recorded_main_commit}..${TASK3_MAIN_COMMIT}"
    )
    if (( simulator_source_identity_changed )); then
        git -C "${TASK3_SIM_ENGINE_ROOT}" merge-base --is-ancestor \
            "${recorded_sim_commit}" "${TASK3_SIM_COMMIT}" >/dev/null 2>&1 || {
            task3_error "${label} simulator commit is not an ancestor of the current simulator."
            return 1
        }
        printf '%s\n' simulator_only_reuse
        return 0
    fi

    [[ "${recorded_sim_commit}" == "${TASK3_SIM_COMMIT}" ]] || {
        task3_error \
            "${label} producer advancement does not contain a simulator source identity change."
        return 1
    }
    printf '%s\n' task1_consumer_only_reuse
}

task3_task2_source_compatibility_mode() {
    local label=$1
    local recorded_main_commit=$2
    local recorded_echo_commit=$3
    local recorded_sim_commit=$4
    local commit_pattern='^[0-9a-f]{40}$'
    local recorded_echo_source_identity recorded_sim_source_identity
    local relative_path recorded_blob current_blob object_type
    local task2_source_files=(
        SC26-AE/lib/common.sh
        SC26-AE/lib/task2_echo.sh
        SC26-AE/task2_gpt175b.sh
        SC26-AE/task2_qwen3_a30b.sh
        SC26-AE/task2_dsv3.sh
        SC26-AE/tools/artifact_manifest.py
        SC26-AE/tools/echo_metrics.py
    )

    [[ "${recorded_main_commit}" =~ ${commit_pattern} ]] || {
        task3_error "${label} Megatron-LM commit is invalid: ${recorded_main_commit}"
        return 1
    }
    [[ "${recorded_echo_commit}" =~ ${commit_pattern} ]] || {
        task3_error "${label} Echo-slowdown commit is invalid: ${recorded_echo_commit}"
        return 1
    }
    [[ "${recorded_sim_commit}" =~ ${commit_pattern} ]] || {
        task3_error "${label} megatron-sim-engine commit is invalid: ${recorded_sim_commit}"
        return 1
    }

    if [[ "${recorded_main_commit}" == "${TASK3_MAIN_COMMIT}" &&
          "${recorded_echo_commit}" == "${TASK3_ECHO_COMMIT}" &&
          "${recorded_sim_commit}" == "${TASK3_SIM_COMMIT}" ]]; then
        printf '%s\n' exact
        return 0
    fi

    [[ "${recorded_echo_commit}" == "${TASK3_ECHO_COMMIT}" ]] || {
        task3_error "${label} Echo-slowdown commit differs from the current producer."
        return 1
    }
    git -C "${TASK3_REPO_ROOT}" merge-base --is-ancestor \
        "${recorded_main_commit}" "${TASK3_MAIN_COMMIT}" >/dev/null 2>&1 || {
        task3_error "${label} outer commit is not an ancestor of the current producer."
        return 1
    }

    recorded_echo_source_identity=$(git -C "${TASK3_REPO_ROOT}" rev-parse \
        "${recorded_main_commit}:Echo-slowdown" 2>/dev/null) || {
        task3_error "${label} recorded outer commit has no Echo-slowdown source identity."
        return 1
    }
    [[ "${recorded_echo_source_identity}" == "${recorded_echo_commit}" ]] || {
        task3_error "${label} recorded Echo-slowdown commit does not match its outer source identity."
        return 1
    }
    recorded_sim_source_identity=$(git -C "${TASK3_REPO_ROOT}" rev-parse \
        "${recorded_main_commit}:megatron-sim-engine" 2>/dev/null) || {
        task3_error "${label} recorded outer commit has no megatron-sim-engine source identity."
        return 1
    }
    [[ "${recorded_sim_source_identity}" == "${recorded_sim_commit}" ]] || {
        task3_error "${label} recorded simulator commit does not match its outer source identity."
        return 1
    }

    for relative_path in "${task2_source_files[@]}"; do
        recorded_blob=$(git -C "${TASK3_REPO_ROOT}" rev-parse \
            "${recorded_main_commit}:${relative_path}" 2>/dev/null) || {
            task3_error "${label} recorded Task2 producer source is missing: ${relative_path}"
            return 1
        }
        current_blob=$(git -C "${TASK3_REPO_ROOT}" rev-parse \
            "${TASK3_MAIN_COMMIT}:${relative_path}" 2>/dev/null) || {
            task3_error "${label} current Task2 producer source is missing: ${relative_path}"
            return 1
        }
        object_type=$(git -C "${TASK3_REPO_ROOT}" cat-file -t \
            "${recorded_blob}" 2>/dev/null) || {
            task3_error "${label} cannot inspect recorded Task2 producer source: ${relative_path}"
            return 1
        }
        [[ "${object_type}" == blob ]] || {
            task3_error "${label} recorded Task2 producer source is not a blob: ${relative_path}"
            return 1
        }
        object_type=$(git -C "${TASK3_REPO_ROOT}" cat-file -t \
            "${current_blob}" 2>/dev/null) || {
            task3_error "${label} cannot inspect current Task2 producer source: ${relative_path}"
            return 1
        }
        [[ "${object_type}" == blob ]] || {
            task3_error "${label} current Task2 producer source is not a blob: ${relative_path}"
            return 1
        }
        [[ "${recorded_blob}" == "${current_blob}" ]] || {
            task3_error "${label} Task2 producer source changed: ${relative_path}"
            return 1
        }
    done

    printf '%s\n' task2_producer_equivalent_reuse
}

task3_assert_real_producer_file() {
    local variable_name=$1
    local configured_path=$2
    local canonical_root=$3
    local relative_path=$4
    local expected_path="${canonical_root}/${relative_path}"
    local expected_blob actual_blob object_type

    [[ "${configured_path}" == "${expected_path}" ]] || \
        task3_error \
            "Real Task3 ${variable_name} must use the canonical tracked producer path: ${expected_path}"
    [[ -f "${configured_path}" && ! -L "${configured_path}" ]] || \
        task3_error \
            "Real Task3 ${variable_name} must be a regular canonical producer file: ${configured_path}"

    if [[ -f "${canonical_root}/.source_commit" ]]; then
        expected_blob=$(git -C "${TASK3_REPO_ROOT}" rev-parse \
            "HEAD:megatron-sim-engine/${relative_path}" 2>/dev/null) || \
            task3_error \
                "Real Task3 ${variable_name} is not tracked by the vendored producer: ${relative_path}"
        object_type=$(git -C "${TASK3_REPO_ROOT}" cat-file -t "${expected_blob}" 2>/dev/null) || \
            task3_error \
                "Cannot inspect vendored Task3 producer object for ${variable_name}: ${relative_path}"
    else
        expected_blob=$(git -C "${canonical_root}" rev-parse \
            "${TASK3_SIM_COMMIT}:${relative_path}" 2>/dev/null) || \
            task3_error \
                "Real Task3 ${variable_name} is not tracked by pinned commit ${TASK3_SIM_COMMIT}: ${relative_path}"
        object_type=$(git -C "${canonical_root}" cat-file -t "${expected_blob}" 2>/dev/null) || \
            task3_error \
                "Cannot inspect pinned Task3 producer object for ${variable_name}: ${relative_path}"
    fi
    [[ "${object_type}" == blob ]] || \
        task3_error \
            "Pinned Task3 producer path is not a file for ${variable_name}: ${relative_path}"
    actual_blob=$(git -C "${TASK3_REPO_ROOT}" hash-object --no-filters \
        "${configured_path}" 2>/dev/null) || \
        task3_error \
            "Cannot hash canonical Task3 producer file for ${variable_name}: ${configured_path}"
    [[ "${actual_blob}" == "${expected_blob}" ]] || \
        task3_error \
            "Real Task3 ${variable_name} bytes differ from pinned commit ${TASK3_SIM_COMMIT}: ${relative_path}"
}

task3_assert_sim_engine_provenance() {
    local canonical_root selected_root selected_head

    [[ -n "${TASK3_SIM_ENGINE_ROOT:-}" ]] || \
        task3_error "Task3 sim-engine producer path is not configured."
    canonical_root=$(cd -- "${TASK3_REPO_ROOT}/megatron-sim-engine" 2>/dev/null && pwd -P) || \
        task3_error "Cannot resolve the canonical megatron-sim-engine producer path."
    selected_root=$(cd -- "${TASK3_SIM_ENGINE_ROOT}" 2>/dev/null && pwd -P) || \
        task3_error "Cannot resolve the configured Task3 sim-engine producer path."

    if [[ "${selected_root}" != "${canonical_root}" ]]; then
        [[ "${TASK3_EXECUTION_MODE:-real}" == synthetic ]] || \
            task3_error "Real Task3 requires the canonical megatron-sim-engine producer path."
        return 0
    fi

    ae_assert_source_clean megatron-sim-engine || \
        task3_error "Task3 megatron-sim-engine producer is dirty; clean it before qualification."
    if [[ -f "${canonical_root}/.source_commit" ]]; then
        selected_head=$(tr -d '[:space:]' <"${canonical_root}/.source_commit") || \
            task3_error "Cannot resolve the vendored megatron-sim-engine source identity."
    else
        selected_head=$(git -C "${selected_root}" rev-parse HEAD 2>/dev/null) || \
            task3_error "Cannot resolve the checked-out megatron-sim-engine producer commit."
    fi
    [[ "${selected_head}" == "${TASK3_SIM_COMMIT}" ]] || \
        task3_error \
            "Selected megatron-sim-engine source identity ${selected_head} differs from ${TASK3_SIM_COMMIT}."

    if [[ "${TASK3_EXECUTION_MODE:-real}" == real ]]; then
        task3_assert_real_producer_file \
            TASK3_BUILDER "${TASK3_BUILDER}" "${canonical_root}" \
            tools/data_prep/slowdown/build_ddp_slowdown_assets.py
        task3_assert_real_producer_file \
            TASK3_SCHEDULER "${TASK3_SCHEDULER}" "${canonical_root}" \
            src/scheduler/mg_scheduling/mg_test.py
        task3_assert_real_producer_file \
            TASK3_SIMULATOR "${TASK3_SIMULATOR}" "${canonical_root}" simu_main.py
    fi
}

task3_json_field() {
    local json_path=$1
    local field_path=$2
    "${TASK3_META_PYTHON}" - "${json_path}" "${field_path}" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
else:
    print(value)
PY
}

task3_verify_artifact_manifest() {
    local bundle_root=$1
    local manifest_path=$2
    "${TASK3_META_PYTHON}" -B "${TASK3_ARTIFACT_TOOL}" verify \
        --root "${bundle_root}" \
        --manifest "${manifest_path}"
}

task3_assign_resolved_fields() {
    local resolved_json=$1

    TASK3_CAPTURE_ID=$(task3_json_field "${resolved_json}" capture_id)
    TASK3_PREDICTOR_RUN_ID=$(task3_json_field "${resolved_json}" predictor_run_id)
    TASK3_TRACE_DIR=$(task3_json_field "${resolved_json}" trace_dir)
    TASK3_NCU_METRICS_SOURCE=$(task3_json_field "${resolved_json}" ncu_metrics_source)
    TASK3_SLOWDOWN_TRACE_SOURCE_DIR=$(task3_json_field "${resolved_json}" slowdown_trace_source_dir)
    TASK3_SLOWDOWN_TRACE_SCOPE=$(task3_json_field "${resolved_json}" slowdown_trace_scope)
    TASK3_SLOWDOWN_TRACE_RANK_IDS=$(task3_json_field "${resolved_json}" slowdown_trace_rank_ids)
    TASK3_NSYS_SQLITE=$(task3_json_field "${resolved_json}" nsys_sqlite)
    TASK3_NCU_METRICS_CSV=$(task3_json_field "${resolved_json}" ncu_metrics_csv)
    TASK3_MODEL_PATH=$(task3_json_field "${resolved_json}" model_path)
    TASK3_SCALER_PATH=$(task3_json_field "${resolved_json}" scaler_path)
    TASK3_SOURCE_TASK1_MANIFEST=$(task3_json_field "${resolved_json}" task1_manifest)
    TASK3_SOURCE_TASK2_MANIFEST=$(task3_json_field "${resolved_json}" task2_manifest)
    TASK3_SOURCE_DISTRIBUTION_MANIFEST=$(task3_json_field "${resolved_json}" distribution_manifest)
    TASK3_SOURCE_ASSETS_DIR=$(task3_json_field "${resolved_json}" source_assets_dir)

    task3_safe_id capture_id "${TASK3_CAPTURE_ID}"
    task3_safe_id predictor_run_id "${TASK3_PREDICTOR_RUN_ID}"
    case "${TASK3_NCU_METRICS_SOURCE}" in
        task1_rank0|synthetic_fixture_compatibility)
            ;;
        *)
            task3_error "Task3 NCU metrics source is invalid: ${TASK3_NCU_METRICS_SOURCE}"
            ;;
    esac
    if [[ "${TASK3_EXECUTION_MODE:-real}" == real &&
        "${TASK3_NCU_METRICS_SOURCE}" != task1_rank0 ]]; then
        task3_error "Real Task3 requires Task1 rank-0 NCU metrics provenance."
    fi
    ae_require_dir "${TASK3_TRACE_DIR}"
    ae_require_dir "${TASK3_SLOWDOWN_TRACE_SOURCE_DIR}"
    [[ "${TASK3_SLOWDOWN_TRACE_SCOPE}" == "global_rank_0" ]] || \
        task3_error "Task3 slowdown trace scope must be global_rank_0."
    [[ "${TASK3_SLOWDOWN_TRACE_RANK_IDS}" == "[0]" ]] || \
        task3_error "Task3 slowdown trace rank ids must be [0]."
    ae_require_file "${TASK3_NSYS_SQLITE}"
    ae_require_file "${TASK3_NCU_METRICS_CSV}"
    ae_require_file "${TASK3_MODEL_PATH}"
    ae_require_file "${TASK3_SCALER_PATH}"
    ae_require_file "${TASK3_SOURCE_TASK1_MANIFEST}"
    ae_require_file "${TASK3_SOURCE_TASK2_MANIFEST}"
}

task3_verify_input_expectations() {
    local resolved_json=$1
    local consumer=$2
    local phase=$3

    "${TASK3_META_PYTHON}" - "${resolved_json}" "${consumer}" "${phase}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

resolved_path = pathlib.Path(sys.argv[1])
consumer = sys.argv[2]
phase = sys.argv[3]


def fail(message):
    raise SystemExit(
        f"[ERROR] Task3 input expectation drift ({consumer}/{phase}): {message}"
    )


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify_entry(entry, label, expected_path_text=None):
    if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
        fail(f"{label} expectation schema is invalid")
    path_text = entry.get("path")
    size_bytes = entry.get("size_bytes")
    sha256 = entry.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        fail(f"{label} expectation path is invalid")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        fail(f"{label} expectation size is invalid")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        fail(f"{label} expectation SHA256 is invalid")
    path = pathlib.Path(path_text)
    if expected_path_text is not None and path != pathlib.Path(expected_path_text):
        fail(f"{label} expectation path differs from the resolved input")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        fail(f"{label} is missing: {path}: {exc}")
    if not path.is_absolute() or path != resolved or path.is_symlink() or not path.is_file():
        fail(f"{label} is no longer a canonical regular file: {path}")
    if path.stat().st_size != size_bytes:
        fail(f"{label} size changed: {path}")
    if digest(path) != sha256:
        fail(f"{label} checksum changed: {path}")
    return path


def verify_trace_inventory(entries, trace_dir_text):
    if not isinstance(entries, list) or not entries:
        fail("trace file expectations must be a non-empty array")
    trace_dir = pathlib.Path(trace_dir_text)
    try:
        resolved_trace_dir = trace_dir.resolve(strict=True)
    except OSError as exc:
        fail(f"trace directory is missing: {trace_dir}: {exc}")
    if (
        not trace_dir.is_absolute()
        or trace_dir != resolved_trace_dir
        or trace_dir.is_symlink()
        or not trace_dir.is_dir()
    ):
        fail(f"trace directory is no longer canonical: {trace_dir}")
    expected_paths = {
        verify_entry(entry, f"trace file {index}")
        for index, entry in enumerate(entries)
    }
    observed_paths = set()
    for path in trace_dir.glob("*.txt"):
        if path.is_symlink() or not path.is_file():
            fail(f"trace directory contains a non-regular trace file: {path}")
        observed_paths.add(path.resolve(strict=True))
    if observed_paths != expected_paths:
        fail(
            "trace inventory changed: "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"unexpected={sorted(str(path) for path in observed_paths - expected_paths)}"
        )


def regular_inventory(root):
    result = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = pathlib.Path(current_root)
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                fail(f"source assets contain a non-directory or symlink: {path}")
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                fail(f"source assets contain a non-regular file or symlink: {path}")
            result.add(path.resolve(strict=True))
    return result


def verify_source_assets(entries, source_assets_dir_text):
    if not isinstance(entries, list) or not entries:
        fail("source asset expectations must be a non-empty array")
    root = pathlib.Path(source_assets_dir_text)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        fail(f"source assets directory is missing: {root}: {exc}")
    if not root.is_absolute() or root != resolved_root or root.is_symlink() or not root.is_dir():
        fail(f"source assets directory is no longer canonical: {root}")
    expected_paths = {
        verify_entry(entry, f"source asset {index}")
        for index, entry in enumerate(entries)
    }
    observed_paths = regular_inventory(root)
    if observed_paths != expected_paths:
        fail(
            "source asset inventory changed: "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"unexpected={sorted(str(path) for path in observed_paths - expected_paths)}"
        )


try:
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    fail(f"cannot load resolved input snapshot: {exc}")
if not isinstance(payload, dict) or payload.get("schema_version") != "sc26-ae-task3-resolved-inputs-v1":
    fail("resolved input snapshot schema is invalid")
source = payload.get("artifact_source")
if source not in {"fresh", "prebaked"}:
    fail("resolved artifact source is invalid")
if consumer not in {"evidence", "materialize", "simulator"}:
    fail(f"unsupported consumer: {consumer}")
if phase not in {"before", "after"}:
    fail(f"unsupported verification phase: {phase}")

expectations = payload.get("input_expectations")
required_keys = {
    "trace_files",
    "nsys_sqlite",
    "ncu_metrics_csv",
    "model_path",
    "scaler_path",
    "task1_manifest",
    "task2_manifest",
    "distribution_manifest",
    "source_assets",
}
if not isinstance(expectations, dict) or set(expectations) != required_keys:
    fail("input expectation snapshot schema is invalid")

if consumer in {"evidence", "simulator"} or (consumer == "materialize" and source == "fresh"):
    verify_trace_inventory(expectations["trace_files"], payload.get("trace_dir"))

scalar_fields = {
    "nsys_sqlite": "nsys_sqlite",
    "ncu_metrics_csv": "ncu_metrics_csv",
    "model_path": "model_path",
    "scaler_path": "scaler_path",
    "task1_manifest": "task1_manifest",
    "task2_manifest": "task2_manifest",
}
if consumer == "evidence":
    selected = tuple(scalar_fields)
elif consumer == "materialize" and source == "fresh":
    selected = ("nsys_sqlite", "ncu_metrics_csv", "model_path", "scaler_path")
elif consumer == "simulator":
    selected = ("model_path", "scaler_path")
else:
    selected = ()
for key in selected:
    verify_entry(expectations[key], key, payload.get(scalar_fields[key]))

distribution_expectation = expectations["distribution_manifest"]
if consumer == "evidence" and source == "prebaked":
    verify_entry(
        distribution_expectation,
        "distribution_manifest",
        payload.get("distribution_manifest"),
    )
elif source == "fresh" and distribution_expectation is not None:
    fail("fresh inputs must not contain a distribution manifest expectation")

source_asset_expectations = expectations["source_assets"]
if consumer == "materialize" and source == "prebaked":
    verify_source_assets(source_asset_expectations, payload.get("source_assets_dir"))
elif source == "fresh" and source_asset_expectations != []:
    fail("fresh inputs must not contain source asset expectations")
PY
}

task3_resolve_fresh() {
    local model_key=$1
    local output_root=$2
    local resolved_json=$3
    local task1_dir="${output_root}/${model_key}/task1"
    local capture_marker="${task1_dir}/capture_marker.json"
    local predictor_marker="${output_root}/_shared/task2/predictor_marker.json"

    [[ -z "${PREBAKED_ROOT:-}" ]] || \
        task3_error "PREBAKED_ROOT is rejected when ARTIFACT_SOURCE=fresh."
    ae_require_file "${capture_marker}"
    ae_require_file "${predictor_marker}"

    "${TASK3_META_PYTHON}" - \
        "${TASK3_ARTIFACT_TOOL}" "${output_root}" "${task1_dir}" \
        "${capture_marker}" "${predictor_marker}" "${model_key}" \
        "${TASK3_PROFILE}" "${TASK3_WORLD_SIZE}" "${TASK3_LOCAL_SIZE}" \
        "${TASK3_PP}" "${TASK3_TP}" "${TASK3_DP}" "${TASK3_EXP}" \
        "${TASK3_MAIN_COMMIT}" "${TASK3_ECHO_COMMIT}" "${TASK3_SIM_COMMIT}" \
        "${TASK3_EXECUTION_MODE}" \
        "${resolved_json}" <<'PY'
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

(
    artifact_tool_path,
    output_root_text,
    task1_dir_text,
    capture_marker_text,
    predictor_marker_text,
    model,
    profile,
    world_size_text,
    local_size_text,
    pp_text,
    tp_text,
    dp_text,
    exp_text,
    main_commit,
    echo_commit,
    sim_commit,
    execution_mode,
    output_text,
) = sys.argv[1:]

output_root = pathlib.Path(output_root_text).resolve(strict=True)
task1_dir = pathlib.Path(task1_dir_text).resolve(strict=True)
capture_marker_path = pathlib.Path(capture_marker_text)
predictor_marker_path = pathlib.Path(predictor_marker_text)


def fail(message):
    raise SystemExit("[ERROR] " + message)


def load_object(path, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{label} is not valid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def safe_relative(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        fail(f"{label} must be a non-empty POSIX relative path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        fail(f"{label} is unsafe: {value}")
    return path


def resolve_inside(base, relative, label, directory=False):
    path = (base / relative).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError:
        fail(f"{label} escapes its declared root: {relative}")
    if path.is_symlink() or (directory and not path.is_dir()) or (
        not directory and not path.is_file()
    ):
        fail(f"{label} has the wrong file type: {path}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_expectation(path, label):
    resolved = path.resolve(strict=True)
    if path != resolved or path.is_symlink() or not path.is_file():
        fail(f"{label} is not a canonical regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def manifest_expectation(root, relative, entries, label):
    entry = entries.get(relative)
    if entry is None:
        fail(f"{label} is absent from the verified artifact manifest: {relative}")
    path = resolve_inside(root, relative, label)
    return {
        "path": str(path),
        "size_bytes": entry["size_bytes"],
        "sha256": entry["sha256"],
    }


def verify_manifest(root, path, label):
    module_spec = importlib.util.spec_from_file_location(
        f"sc26_ae_manifest_{label}", artifact_tool_path
    )
    if module_spec is None or module_spec.loader is None:
        fail("cannot load canonical artifact manifest verifier")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    payload = load_object(path, label)
    try:
        module.verify_manifest(root, payload)
    except Exception as exc:
        fail(f"{label} verification failed: {exc}")
    return payload


expected_commits = {
    "megatron_lm": main_commit,
    "echo_slowdown": echo_commit,
    "megatron_sim_engine": sim_commit,
}
expected_topology = {
    "world_size": int(world_size_text),
    "local_size": int(local_size_text),
    "pp": int(pp_text),
    "tp": int(tp_text),
    "dp": int(dp_text),
    "exp": int(exp_text),
}


def validated_source_commits(value, label):
    if not isinstance(value, dict) or set(value) != set(expected_commits):
        fail(f"{label} source_commits schema is invalid")
    for key, commit in value.items():
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            fail(f"{label} source commit is invalid for {key}")
    return value

capture_marker = load_object(capture_marker_path, "Task1 capture marker")
if capture_marker.get("schema_version") != "sc26-ae-task1-capture-marker-v1":
    fail("Task1 capture marker schema is invalid")
if capture_marker.get("verified") is not True:
    fail("Task1 capture marker is not verified")
if capture_marker.get("model") != model:
    fail("Task1 capture marker model mismatch")
capture_id = capture_marker.get("capture_id")
if not isinstance(capture_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", capture_id):
    fail("Task1 capture_id is invalid")
capture_run_rel = safe_relative(capture_marker.get("run_path"), "Task1 run_path")
if capture_run_rel != pathlib.PurePosixPath("runs") / capture_id:
    fail("Task1 marker run_path does not match capture_id")
task1_run = resolve_inside(task1_dir, capture_run_rel, "Task1 run", directory=True)
task1_manifest_path = task1_run / "artifact_manifest.json"
task1_manifest = verify_manifest(task1_run, task1_manifest_path, "Task1 manifest")
if (
    digest(task1_manifest_path) != capture_marker.get("manifest_sha256")
    or digest(task1_manifest_path) != capture_marker.get("artifact_manifest_sha256")
):
    fail("Task1 marker manifest checksum mismatch")
if task1_manifest.get("task") != "task1" or task1_manifest.get("artifact_source") != "fresh":
    fail("Task1 manifest identity is invalid")
if task1_manifest.get("model") != model or task1_manifest.get("capture_id") != capture_id:
    fail("Task1 manifest model/capture_id mismatch")
task1_source_commits = validated_source_commits(
    task1_manifest.get("source_commits"), "Fresh Task1"
)
if task1_manifest.get("simulation_topology") != expected_topology:
    fail("Task1 simulation topology differs from the frozen Task3 topology")
if task1_manifest.get("profile") != profile:
    fail("Task1 profile differs from the frozen Task3 profile")
capture_runtime = task1_manifest.get("capture_runtime")
if not isinstance(capture_runtime, dict):
    fail("Task1 capture_runtime is missing")
if capture_runtime.get("physical_gpu_count") != 1:
    fail("Task1 physical GPU evidence must equal one")
if capture_runtime.get("scaling_min_warmup_iters") != 3 or capture_runtime.get(
    "scaling_profile_iters"
) != 1:
    fail("Task1 warmup/profile provenance mismatch")
if task1_manifest.get("precision") != "bf16" or task1_manifest.get("mock_data") is not True:
    fail("Task1 precision/mock-data provenance mismatch")
if task1_manifest.get("ddp_overlap") is not True:
    fail("Task1 DDP overlap provenance is missing")
task1_evidence = task1_manifest.get("execution_evidence")
if execution_mode == "real" and task1_evidence not in {
    "real_single_h800_qualified",
    "runtime_measurement_requires_external_single_gpu_qualification",
}:
    fail("Task1 manifest lacks valid real runtime execution evidence")
if execution_mode == "synthetic" and task1_evidence not in {
    None,
    "local_synthetic_fixture",
    "local_synthetic_not_gpu_qualification",
    "real_single_h800_qualified",
}:
    fail("Task1 execution evidence class is invalid")

task1_file_entries = {
    safe_relative(entry.get("path"), "Task1 artifact path"): entry
    for entry in task1_manifest.get("files", [])
}
task1_files = [path.as_posix() for path in task1_file_entries]
trace_rel_paths = [
    safe_relative(path, "Task1 trace path")
    for path in task1_files
    if isinstance(path, str)
    and path.startswith("runtime/profiler_log/")
    and path.endswith(".txt")
]
if not trace_rel_paths:
    fail("Task1 manifest contains no profiler trace files")
trace_paths = [resolve_inside(task1_run, path, "Task1 trace") for path in trace_rel_paths]
trace_parents = {path.parent for path in trace_paths}
if len(trace_parents) != 1:
    fail("Task1 trace files do not share one canonical trace directory")
trace_dir = next(iter(trace_parents))
if set(trace_dir.glob("*.txt")) != set(trace_paths):
    fail("Task1 trace directory contains files outside the verified manifest set")
required_operations = ("forward_step", "backward_step", "optimizer_step")
for trace_path in trace_paths:
    trace_text = trace_path.read_text(encoding="utf-8")
    for operation in required_operations:
        if not re.search(rf"^rank:[0-9]+:{operation}\(", trace_text, re.MULTILINE):
            fail(f"Task1 trace semantic validation is missing {operation}: {trace_path}")
    backward_lines = [
        line
        for line in trace_text.splitlines()
        if re.match(r"^rank:[0-9]+:backward_step\(", line)
    ]
    if not backward_lines or any(
        token not in line
        for line in backward_lines
        for token in ("cmd_uid=", "timestamp=", "duration=", "mg_state=", "stage_id=", "batch_id=")
    ):
        fail(f"Task1 backward trace metadata is incomplete: {trace_path}")
    if not re.search(
        r"^rank:[0-9]+:ddp_grad_comm\([^\n]*trigger_cmd_uid=",
        trace_text,
        re.MULTILINE,
    ):
        fail(f"Task1 trace lacks DDP-overlap trigger metadata: {trace_path}")

sqlite_rel_paths = [
    safe_relative(path, "Task1 Nsight SQLite path")
    for path in task1_files
    if isinstance(path, str) and path.endswith(".sqlite")
]
if len(sqlite_rel_paths) != 1:
    fail("Task1 manifest must contain exactly one Nsight SQLite artifact")
nsys_sqlite = resolve_inside(task1_run, sqlite_rel_paths[0], "Task1 Nsight SQLite")

task1_ncu_rel_paths = [
    safe_relative(path, "Task1 NCU feature path")
    for path in task1_files
    if isinstance(path, str)
    and path == "ncu/kernel_metric_output.csv"
]
task1_ncu_provenance = task1_manifest.get("ncu_feature_provenance")
if execution_mode == "real":
    if len(task1_ncu_rel_paths) != 1:
        fail("Real Task3 requires exactly one Task1 rank-0 NCU feature CSV")
    if not isinstance(task1_ncu_provenance, dict):
        fail("Real Task3 requires Task1 NCU feature provenance")
    if task1_ncu_provenance.get("enabled") is not True:
        fail("Task1 NCU feature provenance is not enabled")
    if task1_ncu_provenance.get("rank_scope") != "global_rank_0":
        fail("Task1 NCU feature provenance rank scope is invalid")
    if task1_ncu_provenance.get("rank_ids") != [0]:
        fail("Task1 NCU feature provenance rank ids are invalid")
    if task1_ncu_provenance.get("physical_gpu_count") != 1:
        fail("Task1 NCU feature provenance physical GPU count must equal one")

predictor_marker = load_object(predictor_marker_path, "Task2 shared predictor marker")
if predictor_marker.get("schema_version") != "sc26-ae-task2-shared-pointer-v1":
    fail("Task2 shared predictor marker schema is invalid")
if predictor_marker.get("verified") is not True:
    fail("Task2 shared predictor marker is not verified")
predictor_run_id = predictor_marker.get("predictor_run_id")
if not isinstance(predictor_run_id, str) or not re.fullmatch(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*", predictor_run_id
):
    fail("Task2 predictor_run_id is invalid")
task2_run_rel = safe_relative(predictor_marker.get("run_path"), "Task2 run_path")
expected_prefix = pathlib.PurePosixPath("_shared/task2/runs")
if task2_run_rel != expected_prefix / predictor_run_id:
    fail("Task2 marker run_path does not match predictor_run_id")
task2_run = resolve_inside(output_root, task2_run_rel, "Task2 run", directory=True)
task2_manifest_path = task2_run / "artifact_manifest.json"
task2_manifest = verify_manifest(task2_run, task2_manifest_path, "Task2 manifest")
task2_manifest_digest = digest(task2_manifest_path)
if task2_manifest_digest != predictor_marker.get("manifest_sha256") or (
    task2_manifest_digest != predictor_marker.get("artifact_manifest_sha256")
):
    fail("Task2 shared marker manifest checksum mismatch")
if task2_manifest.get("task") != "task2" or task2_manifest.get("model") != "shared_task2":
    fail("Task2 manifest identity is invalid")
if task2_manifest.get("artifact_source") != "fresh":
    fail("Task2 manifest must describe fresh predictor artifacts")
if task2_manifest.get("predictor_run_id") != predictor_run_id:
    fail("Task2 manifest predictor_run_id mismatch")
task2_source_commits = validated_source_commits(
    task2_manifest.get("source_commits"), "Fresh Task2"
)
task2_evidence = task2_manifest.get("execution_evidence")
if execution_mode == "real" and task2_evidence not in {
    "real_exact_two_h800_qualified",
    "runtime_measurement_requires_external_two_gpu_qualification",
}:
    fail("Task2 manifest lacks valid real runtime execution evidence")
if execution_mode == "synthetic" and task2_evidence not in {
    "local_synthetic_not_two_gpu_qualification",
    "runtime_measurement_requires_external_two_gpu_qualification",
    "real_exact_two_h800_qualified",
}:
    fail("Task2 execution evidence class is invalid")

required_task2_paths = {
    "ncu_metrics_csv": pathlib.PurePosixPath("merge/input/kernel_metric_output.csv"),
    "model_path": pathlib.PurePosixPath("training_testing/output/xgb_model.json"),
    "scaler_path": pathlib.PurePosixPath("training_testing/output/standard_scaler.json"),
}
task2_file_entries = {
    safe_relative(entry.get("path"), "Task2 artifact path"): entry
    for entry in task2_manifest.get("files", [])
}
listed_task2_paths = set(task2_file_entries)
for label, path in required_task2_paths.items():
    if path not in listed_task2_paths:
        fail(f"Task2 manifest is missing required {label}: {path}")

model_path = resolve_inside(
    task2_run,
    required_task2_paths["model_path"],
    "Task2 model",
)
scaler_path = resolve_inside(
    task2_run,
    required_task2_paths["scaler_path"],
    "Task2 scaler",
)
if task1_ncu_rel_paths:
    ncu_metrics_path = resolve_inside(task1_run, task1_ncu_rel_paths[0], "Task1 NCU metrics")
    ncu_metrics_relative = task1_ncu_rel_paths[0].as_posix()
    ncu_metrics_source = "task1_rank0"
else:
    if execution_mode == "real":
        fail("Task1 rank-0 NCU feature CSV is missing")
    ncu_metrics_path = resolve_inside(
        task2_run,
        pathlib.PurePosixPath("merge/input/kernel_metric_output.csv"),
        "synthetic fixture NCU metrics",
    )
    ncu_metrics_relative = "merge/input/kernel_metric_output.csv"
    ncu_metrics_source = "synthetic_fixture_compatibility"
payload = {
    "schema_version": "sc26-ae-task3-resolved-inputs-v1",
    "artifact_source": "fresh",
    "capture_id": capture_id,
    "predictor_run_id": predictor_run_id,
    "trace_dir": str(trace_dir),
    "nsys_sqlite": str(nsys_sqlite),
    "ncu_metrics_csv": str(ncu_metrics_path) if ncu_metrics_path is not None else "",
    "ncu_metrics_source": ncu_metrics_source,
    "model_path": str(model_path),
    "scaler_path": str(scaler_path),
    "task1_manifest": str(task1_manifest_path),
    "task2_manifest": str(task2_manifest_path),
    "task1_source_commits": task1_source_commits,
    "task2_source_commits": task2_source_commits,
    "distribution_manifest": "",
    "source_assets_dir": "",
    "task1_root": str(task1_run),
    "task2_root": str(task2_run),
    "trace_dir_relative": trace_dir.relative_to(task1_run).as_posix(),
    "nsys_sqlite_relative": nsys_sqlite.relative_to(task1_run).as_posix(),
    "ncu_metrics_relative": ncu_metrics_relative,
    "slowdown_trace_source_dir": str(trace_dir),
    "slowdown_trace_scope": "global_rank_0",
    "slowdown_trace_rank_ids": [0],
    "model_relative": required_task2_paths["model_path"].as_posix(),
    "scaler_relative": required_task2_paths["scaler_path"].as_posix(),
    "input_expectations": {
        "trace_files": [
            manifest_expectation(task1_run, path, task1_file_entries, "Task1 trace")
            for path in sorted(trace_rel_paths, key=lambda value: value.as_posix())
        ],
        "nsys_sqlite": manifest_expectation(
            task1_run,
            sqlite_rel_paths[0],
            task1_file_entries,
            "Task1 Nsight SQLite",
        ),
        "ncu_metrics_csv": (
            manifest_expectation(
                task1_run,
                task1_ncu_rel_paths[0],
                task1_file_entries,
                "Task1 NCU metrics",
            )
            if task1_ncu_rel_paths
            else manifest_expectation(
                task2_run,
                required_task2_paths["ncu_metrics_csv"],
                task2_file_entries,
                "synthetic fixture NCU metrics",
            )
        ),
        "model_path": manifest_expectation(
            task2_run,
            required_task2_paths["model_path"],
            task2_file_entries,
            "Task2 model",
        ),
        "scaler_path": manifest_expectation(
            task2_run,
            required_task2_paths["scaler_path"],
            task2_file_entries,
            "Task2 scaler",
        ),
        "task1_manifest": observed_expectation(task1_manifest_path, "Task1 manifest"),
        "task2_manifest": observed_expectation(task2_manifest_path, "Task2 manifest"),
        "distribution_manifest": None,
        "source_assets": [],
    },
}
pathlib.Path(output_text).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    local task1_compatibility_mode task2_compatibility_mode
    task1_compatibility_mode=$(task3_source_compatibility_mode \
        'Fresh Task1' \
        "$(task3_json_field "${resolved_json}" task1_source_commits.megatron_lm)" \
        "$(task3_json_field "${resolved_json}" task1_source_commits.echo_slowdown)" \
        "$(task3_json_field "${resolved_json}" task1_source_commits.megatron_sim_engine)")
    task2_compatibility_mode=$(task3_task2_source_compatibility_mode \
        'Fresh Task2' \
        "$(task3_json_field "${resolved_json}" task2_source_commits.megatron_lm)" \
        "$(task3_json_field "${resolved_json}" task2_source_commits.echo_slowdown)" \
        "$(task3_json_field "${resolved_json}" task2_source_commits.megatron_sim_engine)")

    "${TASK3_META_PYTHON}" - "${resolved_json}" \
        "${task1_compatibility_mode}" "${task2_compatibility_mode}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["source_compatibility"] = {
    "policy": "task_specific_source_compatibility_v2",
    "task1": sys.argv[2],
    "task2": sys.argv[3],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    task3_assign_resolved_fields "${resolved_json}"
}

task3_resolve_prebaked() {
    local model_key=$1
    local resolved_json=$2
    local prebaked_root=${PREBAKED_ROOT:-"${TASK3_REPO_ROOT}/SC26-AE/prebaked"}
    local distribution_manifest="${prebaked_root}/distribution_manifest.json"

    [[ "${prebaked_root}" == /* ]] || \
        task3_error "PREBAKED_ROOT must be an absolute path: ${prebaked_root}"
    [[ -d "${prebaked_root}" ]] || \
        task3_error "PREBAKED_ROOT directory is missing: ${prebaked_root}"
    ae_require_file "${distribution_manifest}"

    "${TASK3_META_PYTHON}" - \
        "${TASK3_ARTIFACT_TOOL}" "${prebaked_root}" "${distribution_manifest}" \
        "${model_key}" "${TASK3_PROFILE}" "${TASK3_WORLD_SIZE}" \
        "${TASK3_LOCAL_SIZE}" "${TASK3_PP}" "${TASK3_TP}" "${TASK3_DP}" \
        "${TASK3_EXP}" "${TASK3_ECHO_COMMIT}" "${TASK3_SIM_COMMIT}" \
        "${TASK3_EXECUTION_MODE}" "${TASK3_ALLOW_FUNCTIONAL_PREBAKED:-0}" \
        "${resolved_json}" <<'PY'
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import sys

(
    artifact_tool_path,
    prebaked_root_text,
    distribution_manifest_text,
    model,
    profile,
    world_size_text,
    local_size_text,
    pp_text,
    tp_text,
    dp_text,
    exp_text,
    echo_commit,
    sim_commit,
    execution_mode,
    allow_functional_prebaked_text,
    output_text,
) = sys.argv[1:]

if allow_functional_prebaked_text not in {"0", "1"}:
    raise SystemExit("[ERROR] TASK3_ALLOW_FUNCTIONAL_PREBAKED must be 0 or 1")
allow_functional_prebaked = allow_functional_prebaked_text == "1"

prebaked_root_path = pathlib.Path(prebaked_root_text)
if prebaked_root_path.is_symlink():
    raise SystemExit("[ERROR] PREBAKED_ROOT must not be a symlink")
prebaked_root = prebaked_root_path.resolve(strict=True)
distribution_manifest_path = pathlib.Path(distribution_manifest_text)


def fail(message):
    raise SystemExit("[ERROR] " + message)


def load_object(path, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{label} is not valid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def safe_relative(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        fail(f"{label} must be a non-empty POSIX relative path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        fail(f"{label} is unsafe: {value}")
    return path


def resolve_inside(base, relative, label, directory=False):
    path = (base / relative).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError:
        fail(f"{label} escapes its declared root: {relative}")
    if path.is_symlink() or (directory and not path.is_dir()) or (
        not directory and not path.is_file()
    ):
        fail(f"{label} has the wrong file type: {path}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_expectation(path, label):
    resolved = path.resolve(strict=True)
    if path != resolved or path.is_symlink() or not path.is_file():
        fail(f"{label} is not a canonical regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def manifest_expectation(root, relative, entries, label):
    entry = entries.get(relative)
    if entry is None:
        fail(f"{label} is absent from the verified artifact manifest: {relative}")
    path = resolve_inside(root, relative, label)
    return {
        "path": str(path),
        "size_bytes": entry["size_bytes"],
        "sha256": entry["sha256"],
    }


def require_execution_evidence(payload, label, required_real, allowed_synthetic):
    observed = payload.get("execution_evidence")
    if execution_mode == "real":
        if observed != required_real:
            fail(f"{label} lacks {required_real} execution evidence")
        return
    if execution_mode == "synthetic":
        allowed = set(allowed_synthetic) | {required_real}
        if allow_functional_prebaked:
            allowed.add("functional_prebaked_not_release_qualified")
        if observed not in allowed:
            fail(f"{label} execution evidence class is invalid: {observed}")
        return
    fail(f"unsupported Task3 execution mode: {execution_mode}")


module_spec = importlib.util.spec_from_file_location(
    "sc26_ae_prebaked_artifact_manifest", artifact_tool_path
)
if module_spec is None or module_spec.loader is None:
    fail("cannot load canonical artifact manifest verifier")
artifact_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(artifact_module)


def verify_manifest(root, path, label):
    payload = load_object(path, label)
    try:
        artifact_module.verify_manifest(root, payload)
    except Exception as exc:
        fail(f"{label} verification failed: {exc}")
    return payload


def regular_inventory(root):
    result = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = pathlib.Path(current_root)
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                fail(f"prebaked distribution contains a non-directory or symlink: {path}")
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                fail(f"prebaked distribution contains a non-regular file or symlink: {path}")
            result.add(path.relative_to(root).as_posix())
    return result


expected_topology = {
    "world_size": int(world_size_text),
    "local_size": int(local_size_text),
    "pp": int(pp_text),
    "tp": int(tp_text),
    "dp": int(dp_text),
    "exp": int(exp_text),
}
distribution = load_object(distribution_manifest_path, "distribution manifest")
distribution_schema = distribution.get("schema_version")
functional_distribution = distribution_schema == "sc26-ae-functional-distribution-manifest-v1"
if distribution_schema not in {
    "sc26-ae-distribution-manifest-v1",
    "sc26-ae-functional-distribution-manifest-v1",
}:
    fail("distribution manifest schema is invalid")
if distribution.get("artifact_source") != "prebaked":
    fail("distribution manifest artifact_source must be prebaked")
if functional_distribution:
    if not allow_functional_prebaked or execution_mode != "synthetic":
        fail(
            "functional prebaked input requires TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 "
            "and TASK3_EXECUTION_MODE=synthetic"
        )
require_execution_evidence(
    distribution,
    "distribution manifest",
    "real_prebaked_qualified",
    {"local_synthetic_fixture", "local_synthetic_not_gpu_qualification"},
)
distribution_id = distribution.get("distribution_id")
if not isinstance(distribution_id, str) or not re.fullmatch(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*", distribution_id
):
    fail("distribution_id is invalid")
if distribution.get("compatible_commits") != {
    "echo_slowdown": echo_commit,
    "megatron_sim_engine": sim_commit,
}:
    fail("prebaked compatible commits differ from the consumer source identities")

file_entries = distribution.get("files")
if not isinstance(file_entries, list) or not file_entries:
    fail("distribution manifest files must be a non-empty array")
listed_paths = set()
for entry in file_entries:
    if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
        fail("distribution manifest file entry schema is invalid")
    relative = safe_relative(entry["path"], "distribution file path")
    if relative.as_posix() == "distribution_manifest.json":
        fail("distribution manifest must not list itself")
    if relative.as_posix() in listed_paths:
        fail(f"duplicate distribution file path: {relative}")
    listed_paths.add(relative.as_posix())
    path = resolve_inside(prebaked_root, relative, "distribution file")
    if isinstance(entry["size_bytes"], bool) or entry["size_bytes"] != path.stat().st_size:
        fail(f"distribution file size mismatch: {relative}")
    if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        fail(f"distribution file SHA256 is invalid: {relative}")
    if entry["sha256"] != digest(path):
        fail(f"distribution file checksum mismatch: {relative}")
inventory = regular_inventory(prebaked_root)
if listed_paths != inventory - {"distribution_manifest.json"}:
    fail(
        "distribution manifest inventory mismatch: "
        f"missing={sorted((inventory - {'distribution_manifest.json'}) - listed_paths)}, "
        f"unexpected={sorted(listed_paths - inventory)}"
    )

bundles = distribution.get("bundles")
expected_bundle_keys = {
    "gpt175b",
    "qwen3_a30b",
    "shared_task2",
} if functional_distribution else {
    "gpt175b",
    "qwen3_a30b",
    "dsv3",
    "shared_task2",
}
if not isinstance(bundles, dict) or set(bundles) != expected_bundle_keys:
    fail(
        "distribution manifest bundles must contain exactly {}".format(
            sorted(expected_bundle_keys)
        )
    )


def bundle_entry(key):
    entry = bundles[key]
    if not isinstance(entry, dict):
        fail(f"distribution bundle entry must be an object: {key}")
    required = {"root", "manifest", "manifest_sha256"}
    if not required.issubset(entry):
        fail(f"distribution bundle entry is incomplete: {key}")
    root_rel = safe_relative(entry["root"], f"{key} bundle root")
    manifest_rel = safe_relative(entry["manifest"], f"{key} manifest path")
    if manifest_rel != root_rel / "artifact_manifest.json":
        fail(f"{key} nested manifest path is not canonical")
    root = resolve_inside(prebaked_root, root_rel, f"{key} bundle root", directory=True)
    manifest_path = resolve_inside(prebaked_root, manifest_rel, f"{key} manifest")
    if digest(manifest_path) != entry["manifest_sha256"]:
        fail(f"{key} nested manifest checksum mismatch")
    return entry, root, manifest_path, verify_manifest(root, manifest_path, f"{key} manifest")


model_entry, model_root, model_manifest_path, model_manifest = bundle_entry(model)
shared_entry, shared_root, shared_manifest_path, shared_manifest = bundle_entry("shared_task2")

require_execution_evidence(
    model_manifest,
    "prebaked model artifact manifest",
    "real_single_h800_qualified",
    {"local_synthetic_fixture", "local_synthetic_not_gpu_qualification"},
)
require_execution_evidence(
    shared_manifest,
    "prebaked shared Task2 manifest",
    "real_exact_two_h800_qualified",
    {
        "local_synthetic_not_two_gpu_qualification",
        "runtime_measurement_requires_external_two_gpu_qualification",
    },
)

if model_entry.get("model") != model or model_entry.get("profile") != profile:
    fail("prebaked model bundle identity/profile mismatch")
if model_entry.get("simulation_topology") != expected_topology:
    fail("prebaked distribution topology differs from the frozen Task3 topology")
if model_manifest.get("model") != model or model_manifest.get("task") != "prebaked":
    fail("prebaked model artifact manifest identity is invalid")
if model_manifest.get("artifact_source") != "prebaked":
    fail("prebaked model artifact manifest source is invalid")
if model_manifest.get("simulation_topology") != expected_topology:
    fail("prebaked model artifact topology mismatch")
if model_manifest.get("profile") != profile:
    fail("prebaked model artifact profile mismatch")
if model_manifest.get("source_commits") != model_entry.get("producer_commits"):
    fail("prebaked producer commits differ between distribution and nested manifest")

capture_id = model_manifest.get("capture_id")
predictor_run_id = model_manifest.get("predictor_run_id")
for label, value in (("capture_id", capture_id), ("predictor_run_id", predictor_run_id)):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        fail(f"prebaked {label} is invalid")
if model_entry.get("capture_id") != capture_id or model_entry.get("predictor_run_id") != predictor_run_id:
    fail("prebaked identities differ between distribution and nested model manifest")

if shared_manifest.get("model") != "shared_task2" or shared_manifest.get("task") != "task2":
    fail("prebaked shared Task2 manifest identity is invalid")
if shared_manifest.get("artifact_source") != "prebaked":
    fail("prebaked shared Task2 manifest source is invalid")
if shared_manifest.get("predictor_run_id") != predictor_run_id:
    fail("prebaked shared Task2 predictor_run_id differs from the model bundle")
if shared_entry.get("predictor_run_id") != predictor_run_id:
    fail("distribution shared predictor_run_id differs from the model bundle")
if shared_manifest.get("source_commits") != shared_entry.get("producer_commits"):
    fail("shared Task2 producer commits differ between distribution and nested manifest")

model_file_entries = {
    safe_relative(entry.get("path"), "prebaked model artifact path"): entry
    for entry in model_manifest.get("files", [])
}
model_files = list(model_file_entries)
trace_rel_paths = [
    path
    for path in model_files
    if path.as_posix().endswith(".txt")
    and (path.parts[0] == "trace" or path.as_posix().startswith("runtime/profiler_log/"))
]
if not trace_rel_paths:
    fail("prebaked model manifest contains no trace files")
trace_paths = [resolve_inside(model_root, path, "prebaked trace") for path in trace_rel_paths]
trace_parents = {path.parent for path in trace_paths}
if len(trace_parents) != 1:
    fail("prebaked trace files do not share one canonical trace directory")
trace_dir = next(iter(trace_parents))
if set(trace_dir.glob("*.txt")) != set(trace_paths):
    fail("prebaked trace directory contains files outside the nested manifest")

slowdown_trace_rel_paths = [
    path
    for path in model_files
    if path.as_posix().startswith("slowdown_trace_rank0/") and path.as_posix().endswith(".txt")
]
if execution_mode == "real" and not slowdown_trace_rel_paths:
    fail("Real prebaked Task3 requires a rank-0 slowdown trace directory")
if slowdown_trace_rel_paths:
    slowdown_trace_paths = [
        resolve_inside(model_root, path, "prebaked rank-0 slowdown trace")
        for path in slowdown_trace_rel_paths
    ]
    slowdown_trace_source_dir = slowdown_trace_paths[0].parent
    if set(slowdown_trace_source_dir.glob("*.txt")) != set(slowdown_trace_paths):
        fail("prebaked rank-0 slowdown trace directory contains unlisted files")
else:
    slowdown_trace_source_dir = trace_dir

sqlite_rel_paths = [path for path in model_files if path.as_posix().endswith(".sqlite")]
if len(sqlite_rel_paths) != 1:
    fail("prebaked model manifest must contain exactly one Nsight SQLite artifact")
nsys_sqlite = resolve_inside(model_root, sqlite_rel_paths[0], "prebaked Nsight SQLite")

model_ncu_rel_paths = [
    path for path in model_files if path.as_posix() == "ncu/kernel_metric_output.csv"
]
if (execution_mode == "real" or functional_distribution) and len(model_ncu_rel_paths) != 1:
    fail("Prebaked Task3 requires the Task1 rank-0 NCU feature CSV")

assets_rel = pathlib.PurePosixPath("slowdown_assets")
assets_root = resolve_inside(model_root, assets_rel, "prebaked slowdown assets", directory=True)
required_asset_paths = {
    assets_rel / "manifest.json",
    assets_rel / "kernel_features.json",
    assets_rel / "backward_kernel_blueprints.json",
}
if not required_asset_paths.issubset(set(model_files)):
    fail("prebaked slowdown assets are incomplete")
for path in required_asset_paths:
    resolve_inside(model_root, path, "prebaked slowdown asset")

shared_file_entries = {
    safe_relative(entry.get("path"), "prebaked shared Task2 artifact path"): entry
    for entry in shared_manifest.get("files", [])
}
shared_files = set(shared_file_entries)
required_shared_paths = {
    "ncu_metrics_csv": pathlib.PurePosixPath("merge/input/kernel_metric_output.csv"),
    "model_path": pathlib.PurePosixPath("training_testing/output/xgb_model.json"),
    "scaler_path": pathlib.PurePosixPath("training_testing/output/standard_scaler.json"),
}
for label, path in required_shared_paths.items():
    if path not in shared_files:
        fail(f"prebaked shared Task2 manifest is missing required {label}: {path}")

if model_ncu_rel_paths:
    ncu_metrics_path = resolve_inside(model_root, model_ncu_rel_paths[0], "prebaked Task1 NCU metrics")
    ncu_metrics_source = "task1_rank0"
    ncu_metrics_relative = model_ncu_rel_paths[0].as_posix()
elif functional_distribution:
    fail("functional prebaked Task3 cannot use a shared Task2 NCU fallback")
else:
    ncu_metrics_path = resolve_inside(
        shared_root,
        required_shared_paths["ncu_metrics_csv"],
        "synthetic fixture NCU metrics",
    )
    ncu_metrics_source = "synthetic_fixture_compatibility"
    ncu_metrics_relative = required_shared_paths["ncu_metrics_csv"].as_posix()
model_path = resolve_inside(
    shared_root,
    required_shared_paths["model_path"],
    "prebaked model",
)
scaler_path = resolve_inside(
    shared_root,
    required_shared_paths["scaler_path"],
    "prebaked scaler",
)
source_asset_paths = sorted(
    (
        path
        for path in model_file_entries
        if path.parts and path.parts[0] == assets_rel.as_posix()
    ),
    key=lambda value: value.as_posix(),
)
if not source_asset_paths:
    fail("prebaked slowdown asset inventory is empty")

payload = {
    "schema_version": "sc26-ae-task3-resolved-inputs-v1",
    "artifact_source": "prebaked",
    "capture_id": capture_id,
    "predictor_run_id": predictor_run_id,
    "trace_dir": str(trace_dir),
    "nsys_sqlite": str(nsys_sqlite),
    "ncu_metrics_csv": str(ncu_metrics_path),
    "ncu_metrics_source": ncu_metrics_source,
    "model_path": str(model_path),
    "scaler_path": str(scaler_path),
    "task1_manifest": str(model_manifest_path),
    "task2_manifest": str(shared_manifest_path),
    "distribution_manifest": str(distribution_manifest_path.resolve(strict=True)),
    "source_assets_dir": str(assets_root),
    "task1_root": str(model_root),
    "task2_root": str(shared_root),
    "trace_dir_relative": trace_dir.relative_to(model_root).as_posix(),
    "nsys_sqlite_relative": nsys_sqlite.relative_to(model_root).as_posix(),
    "ncu_metrics_relative": ncu_metrics_relative,
    "slowdown_trace_source_dir": str(slowdown_trace_source_dir),
    "slowdown_trace_scope": "global_rank_0",
    "slowdown_trace_rank_ids": [0],
    "model_relative": required_shared_paths["model_path"].as_posix(),
    "scaler_relative": required_shared_paths["scaler_path"].as_posix(),
    "distribution_id": distribution_id,
    "input_expectations": {
        "trace_files": [
            manifest_expectation(model_root, path, model_file_entries, "prebaked trace")
            for path in sorted(trace_rel_paths, key=lambda value: value.as_posix())
        ],
        "nsys_sqlite": manifest_expectation(
            model_root,
            sqlite_rel_paths[0],
            model_file_entries,
            "prebaked Nsight SQLite",
        ),
        "ncu_metrics_csv": manifest_expectation(
            model_root if model_ncu_rel_paths else shared_root,
            model_ncu_rel_paths[0] if model_ncu_rel_paths else required_shared_paths["ncu_metrics_csv"],
            model_file_entries if model_ncu_rel_paths else shared_file_entries,
            "prebaked Task1 NCU metrics" if model_ncu_rel_paths else "synthetic fixture NCU metrics",
        ),
        "model_path": manifest_expectation(
            shared_root,
            required_shared_paths["model_path"],
            shared_file_entries,
            "prebaked model",
        ),
        "scaler_path": manifest_expectation(
            shared_root,
            required_shared_paths["scaler_path"],
            shared_file_entries,
            "prebaked scaler",
        ),
        "task1_manifest": observed_expectation(
            model_manifest_path,
            "prebaked model manifest",
        ),
        "task2_manifest": observed_expectation(
            shared_manifest_path,
            "prebaked shared Task2 manifest",
        ),
        "distribution_manifest": observed_expectation(
            distribution_manifest_path.resolve(strict=True),
            "distribution manifest",
        ),
        "source_assets": [
            manifest_expectation(
                model_root,
                path,
                model_file_entries,
                "prebaked slowdown asset",
            )
            for path in source_asset_paths
        ],
    },
}
pathlib.Path(output_text).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    task3_assign_resolved_fields "${resolved_json}"
    ae_require_file "${TASK3_SOURCE_DISTRIBUTION_MANIFEST}"
    ae_require_dir "${TASK3_SOURCE_ASSETS_DIR}"
}

task3_prepare_run() {
    local model_key=$1
    local output_root=$2

    TASK3_TASK_DIR="${output_root}/${model_key}/task3"
    TASK3_SIMULATION_RUN_ID=${TASK3_SIMULATION_RUN_ID:-"${model_key}-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"}
    task3_safe_id simulation_run_id "${TASK3_SIMULATION_RUN_ID}"
    TASK3_RUN_ROOT="${TASK3_TASK_DIR}/runs/${TASK3_SIMULATION_RUN_ID}"
    TASK3_WORK_ROOT="${output_root}/_work/task3.${TASK3_SIMULATION_RUN_ID}"
    TASK3_SCHEDULE_DIR="${TASK3_RUN_ROOT}/schedule"
    TASK3_SLOWDOWN_TRACE_DIR="${TASK3_RUN_ROOT}/slowdown_trace_rank0"
    TASK3_SLOWDOWN_ASSETS_DIR="${TASK3_RUN_ROOT}/slowdown_assets"
    TASK3_LOG_DIR="${TASK3_RUN_ROOT}/logs"
    TASK3_RUNTIME_DIR="${TASK3_RUN_ROOT}/runtime"
    TASK3_PROVENANCE_DIR="${TASK3_RUN_ROOT}/provenance"
    TASK3_MARKER_PATH="${TASK3_TASK_DIR}/run_marker.json"

    task3_assert_output_subtree_safe "${output_root}" "${model_key}"

    [[ ! -e "${TASK3_RUN_ROOT}" && ! -L "${TASK3_RUN_ROOT}" ]] || \
        task3_error "Task3 run destination already exists: ${TASK3_RUN_ROOT}"
    [[ ! -e "${TASK3_WORK_ROOT}" && ! -L "${TASK3_WORK_ROOT}" ]] || \
        task3_error "Task3 work destination already exists: ${TASK3_WORK_ROOT}"
    [[ ! -L "${TASK3_MARKER_PATH}" ]] || \
        task3_error "Task3 run marker must not be a symlink: ${TASK3_MARKER_PATH}"

    mkdir -p \
        "${TASK3_SCHEDULE_DIR}" \
        "${TASK3_LOG_DIR}" \
        "${TASK3_RUNTIME_DIR}" \
        "${TASK3_PROVENANCE_DIR}" \
        "${TASK3_WORK_ROOT}"
}

task3_materialize_slowdown_trace() {
    local resolved_json=$1

    [[ ! -e "${TASK3_SLOWDOWN_TRACE_DIR}" && ! -L "${TASK3_SLOWDOWN_TRACE_DIR}" ]] || \
        task3_error "Task3 slowdown trace destination already exists: ${TASK3_SLOWDOWN_TRACE_DIR}"
    task3_record_command slowdown_trace \
        task3_prepare_rank0_slowdown_trace \
        "${TASK3_SLOWDOWN_TRACE_SOURCE_DIR}" "${TASK3_SLOWDOWN_TRACE_DIR}"
    task3_prepare_rank0_slowdown_trace \
        "${TASK3_SLOWDOWN_TRACE_SOURCE_DIR}" "${TASK3_SLOWDOWN_TRACE_DIR}" \
        >"${TASK3_LOG_DIR}/slowdown_trace.log"
    ae_require_dir "${TASK3_SLOWDOWN_TRACE_DIR}"
    local -a trace_files=("${TASK3_SLOWDOWN_TRACE_DIR}"/*.txt)
    [[ -f "${trace_files[0]}" ]] || \
        task3_error "Task3 rank-0 slowdown trace materialization produced no .txt files"
    task3_verify_input_expectations "${resolved_json}" materialize after
}

task3_write_input_evidence() {
    local resolved_json=$1
    local evidence_path="${TASK3_PROVENANCE_DIR}/input_evidence.json"

    task3_verify_input_expectations "${resolved_json}" evidence before
    cp -- "${resolved_json}" "${TASK3_PROVENANCE_DIR}/resolved_inputs.json"
    cp -- "${TASK3_SOURCE_TASK1_MANIFEST}" "${TASK3_PROVENANCE_DIR}/task1_manifest.json"
    cp -- "${TASK3_SOURCE_TASK2_MANIFEST}" "${TASK3_PROVENANCE_DIR}/task2_manifest.json"
    if [[ -n "${TASK3_SOURCE_DISTRIBUTION_MANIFEST}" ]]; then
        cp -- "${TASK3_SOURCE_DISTRIBUTION_MANIFEST}" \
            "${TASK3_PROVENANCE_DIR}/distribution_manifest.json"
    fi

    "${TASK3_META_PYTHON}" - \
        "${evidence_path}" "${TASK3_MODEL_KEY}" "${ARTIFACT_SOURCE}" \
        "${TASK3_SIMULATION_RUN_ID}" "${TASK3_CAPTURE_ID}" \
        "${TASK3_PREDICTOR_RUN_ID}" "${TASK3_TRACE_DIR}" \
        "${TASK3_SLOWDOWN_TRACE_DIR}" "${TASK3_SLOWDOWN_TRACE_SCOPE}" \
        "${TASK3_SLOWDOWN_TRACE_RANK_IDS}" \
        "${TASK3_NSYS_SQLITE}" "${TASK3_NCU_METRICS_CSV}" \
        "${TASK3_MODEL_PATH}" "${TASK3_SCALER_PATH}" \
        "${TASK3_SOURCE_TASK1_MANIFEST}" "${TASK3_SOURCE_TASK2_MANIFEST}" \
        "${TASK3_SOURCE_DISTRIBUTION_MANIFEST}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    output_text,
    model,
    source,
    simulation_run_id,
    capture_id,
    predictor_run_id,
    trace_dir_text,
    slowdown_trace_dir_text,
    slowdown_trace_scope,
    slowdown_trace_rank_ids_text,
    nsys_text,
    ncu_text,
    model_text,
    scaler_text,
    task1_manifest_text,
    task2_manifest_text,
    distribution_manifest_text,
) = sys.argv[1:]


def evidence(path_text):
    path = pathlib.Path(path_text).resolve(strict=True)
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


trace_dir = pathlib.Path(trace_dir_text).resolve(strict=True)
trace_files = sorted(trace_dir.glob("*.txt"))
if not trace_files:
    raise SystemExit("[ERROR] canonical Task3 trace directory contains no .txt files")
slowdown_trace_dir = pathlib.Path(slowdown_trace_dir_text).resolve(strict=True)
slowdown_trace_files = sorted(slowdown_trace_dir.glob("*.txt"))
if not slowdown_trace_files:
    raise SystemExit("[ERROR] rank-0 slowdown trace directory contains no .txt files")
if any(path.is_symlink() or not path.is_file() for path in slowdown_trace_files):
    raise SystemExit("[ERROR] rank-0 slowdown trace directory contains a non-regular file")
try:
    slowdown_rank_ids = json.loads(slowdown_trace_rank_ids_text)
except json.JSONDecodeError as exc:
    raise SystemExit(f"[ERROR] slowdown trace rank ids are invalid JSON: {exc}")
if slowdown_trace_scope != "global_rank_0" or slowdown_rank_ids != [0]:
    raise SystemExit("[ERROR] slowdown trace provenance must describe global rank 0 only")
payload = {
    "schema_version": "sc26-ae-task3-input-evidence-v1",
    "model": model,
    "artifact_source": source,
    "simulation_run_id": simulation_run_id,
    "capture_id": capture_id,
    "predictor_run_id": predictor_run_id,
    "trace_file_count": len(trace_files),
    "trace_files": [evidence(str(path)) for path in trace_files],
    "slowdown_trace_scope": slowdown_trace_scope,
    "slowdown_trace_rank_ids": slowdown_rank_ids,
    "slowdown_trace_file_count": len(slowdown_trace_files),
    "slowdown_trace_files": [evidence(str(path)) for path in slowdown_trace_files],
    "nsys_sqlite": evidence(nsys_text),
    "ncu_metrics_csv": evidence(ncu_text),
    "slowdown_model": evidence(model_text),
    "slowdown_scaler": evidence(scaler_text),
    "task1_manifest": evidence(task1_manifest_text),
    "task2_manifest": evidence(task2_manifest_text),
}
if distribution_manifest_text:
    payload["distribution_manifest"] = evidence(distribution_manifest_text)
pathlib.Path(output_text).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    task3_verify_input_expectations "${resolved_json}" evidence after
}

task3_record_command() {
    local label=$1
    shift
    {
        printf '%s=' "${label}"
        printf ' %q' "$@"
        printf '\n'
    } >>"${TASK3_LOG_DIR}/commands.log"
}

task3_validate_slowdown_assets() {
    local assets_dir=$1
    "${TASK3_META_PYTHON}" - "${assets_dir}" <<'PY'
import json
import os
import pathlib
import stat
import sys

root_path = pathlib.Path(sys.argv[1])
if root_path.is_symlink() or not root_path.is_dir():
    raise SystemExit(f"[ERROR] slowdown assets root must be a regular directory: {root_path}")
root = root_path.resolve(strict=True)
required = {
    "manifest.json",
    "kernel_features.json",
    "backward_kernel_blueprints.json",
}
for name in required:
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"[ERROR] required slowdown asset is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] slowdown asset is invalid JSON: {path}: {exc}")
    if not isinstance(value, (dict, list)):
        raise SystemExit(f"[ERROR] slowdown asset JSON must be an object or array: {path}")
for current_root, directory_names, file_names in os.walk(root, followlinks=False):
    current = pathlib.Path(current_root)
    for name in directory_names + file_names:
        mode = (current / name).lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SystemExit(f"[ERROR] slowdown assets contain a symlink: {current / name}")
PY
}


# Reuse rank-0 NCU/NSYS kernel timing for all fake-rank DDP triggers.
task3_expand_rank0_slowdown_blueprints() {
    local assets_dir="${TASK3_SLOWDOWN_ASSETS_DIR}"
    local trace_dir="${TASK3_SIMULATOR_TRACE_DIR}"
    local provenance_path="${TASK3_PROVENANCE_DIR}/slowdown_blueprint_expansion.json"
    "${TASK3_META_PYTHON}" - "${assets_dir}" "${trace_dir}" "${provenance_path}" <<'PY2'
import copy,json,pathlib,re,sys
assets=pathlib.Path(sys.argv[1]).resolve(strict=True); traces=pathlib.Path(sys.argv[2]).resolve(strict=True); out=pathlib.Path(sys.argv[3]); bp_path=assets/'backward_kernel_blueprints.json'; bp=json.loads(bp_path.read_text())
if not isinstance(bp,dict) or not bp: raise SystemExit('[ERROR] rank-0 slowdown blueprints must be non-empty')
source_uid,source_bp=next(iter(bp.items())); field=re.compile(r'(?:^|,)([A-Za-z_]+)=([^,)]*)'); back=re.compile(r'^rank:([0-9]+):backward_step\((.*)\)$'); triggers={}; context={}
for f in sorted(traces.glob('*.txt')):
 lines=f.read_text().splitlines()
 for line in lines:
  if ':ddp_grad_comm(' in line:
   d=dict(field.findall(line.split('ddp_grad_comm(',1)[1])); t=d.get('trigger_cmd_uid'); c=d.get('comm_uid')
   if t and c and c not in triggers.setdefault(t,[]): triggers[t].append(c)
  m=back.match(line)
  if m:
   d=dict(field.findall(m.group(2))); uid=d.get('cmd_uid')
   if uid: context[uid]={'rank':int(m.group(1)),'stage_id':int(d.get('stage_id',0)),'batch_id':int(d.get('batch_id',0)),'mg_state':d.get('mg_state','steady')}
if not triggers: raise SystemExit('[ERROR] no DDP slowdown triggers found in simulator trace directory')
expanded={}
for uid,comm_uids in sorted(triggers.items()):
 item=copy.deepcopy(source_bp); ctx=context.get(uid,{})
 for k in ('rank','stage_id','batch_id','mg_state'):
  if k in ctx: item[k]=ctx[k]
 markers=list(item.get('launch_markers',[]))
 if len(markers) < len(comm_uids):
  raise SystemExit(f'[ERROR] rank-0 slowdown marker count {len(markers)} is smaller than trigger {uid} marker count {len(comm_uids)}')
 markers=markers[:len(comm_uids)]
 for marker,comm in zip(markers,comm_uids): marker['comm_uid']=comm
 item['launch_markers']=markers; expanded[uid]=item
bp_path.write_text(json.dumps(expanded,indent=2,sort_keys=True)+'\n'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'schema_version':'sc26-ae-task3-rank0-slowdown-expansion-v1','scope':'global_rank_0','source_blueprint_cmd_uid':source_uid,'source_blueprint_count':len(bp),'expanded_blueprint_count':len(expanded),'expanded_trigger_cmd_uids':sorted(expanded),'trace_dir':str(traces),'mapping':'rank-0 NCU/NSYS kernel blueprint reused per trigger; DDP marker identities preserved'},indent=2,sort_keys=True)+'\n')
print(f'SLOWDOWN_BLUEPRINT_EXPANSION_COUNT={len(expanded)}')
PY2
}

task3_materialize_slowdown_assets() {
    local resolved_json=$1
    task3_verify_input_expectations "${resolved_json}" materialize before
    if [[ "${ARTIFACT_SOURCE}" == fresh ]]; then
        local -a builder_command=(
            "${TASK3_SIMULATOR_PYTHON}"
            -B
            "${TASK3_BUILDER}"
            --trace-dir "${TASK3_SLOWDOWN_TRACE_DIR}"
            --nsys-sqlite "${TASK3_NSYS_SQLITE}"
            --ncu-metrics-csv "${TASK3_NCU_METRICS_CSV}"
            --label-prefix cmd_trace
            --output-dir "${TASK3_SLOWDOWN_ASSETS_DIR}"
            --model-path "${TASK3_MODEL_PATH}"
            --scaler-path "${TASK3_SCALER_PATH}"
        )
        task3_record_command builder "${builder_command[@]}"
        "${builder_command[@]}" >"${TASK3_LOG_DIR}/builder.log" 2>&1
    else
        [[ ! -e "${TASK3_SLOWDOWN_ASSETS_DIR}" ]] || \
            task3_error "Task3 slowdown asset destination already exists: ${TASK3_SLOWDOWN_ASSETS_DIR}"
        task3_validate_slowdown_assets "${TASK3_SOURCE_ASSETS_DIR}"
        mkdir -p "${TASK3_SLOWDOWN_ASSETS_DIR}"
        cp -a -- "${TASK3_SOURCE_ASSETS_DIR}/." "${TASK3_SLOWDOWN_ASSETS_DIR}/"
        printf 'Copied verified prebaked slowdown assets from %s\n' \
            "${TASK3_SOURCE_ASSETS_DIR}" >"${TASK3_LOG_DIR}/builder.log"
    fi
    task3_validate_slowdown_assets "${TASK3_SLOWDOWN_ASSETS_DIR}"
    task3_verify_input_expectations "${resolved_json}" materialize after
}

task3_generate_schedule() {
    local -a scheduler_command=(
        "${TASK3_SIMULATOR_PYTHON}"
        -B
        "${TASK3_SCHEDULER}"
        --tensor-model-parallel-size "${TASK3_TP}"
        --pipeline-model-parallel-size "${TASK3_PP}"
        --expert-model-parallel-size "${TASK3_EXP}"
        --num-experts "${TASK3_NUM_EXPERTS}"
        --world-size "${TASK3_WORLD_SIZE}"
        --local-size "${TASK3_LOCAL_SIZE}"
        --micro-batch-size "${TASK3_MICRO_BATCH_SIZE}"
        --global-batch-size "${TASK3_GLOBAL_BATCH_SIZE}"
        --seq-length "${TASK3_SEQ_LEN}"
        --hidden-size "${TASK3_HIDDEN_SIZE}"
        --model-size "${TASK3_MODEL_KEY}"
        --bf16
        --train-iters 1
        --trace-start 0
        --output-dir "${TASK3_SCHEDULE_DIR}"
    )
    if [[ "${TASK3_UNTIE_EMBEDDINGS}" == 1 ]]; then
        scheduler_command+=(--untie-embeddings-and-output-weights)
    fi
    task3_record_command scheduler "${scheduler_command[@]}"
    (
        cd -- "${TASK3_SIM_ENGINE_ROOT}"
        "${scheduler_command[@]}"
    ) >"${TASK3_LOG_DIR}/scheduler.log" 2>&1

    "${TASK3_META_PYTHON}" - \
        "${TASK3_SCHEDULE_DIR}" \
        "${TASK3_PP}" \
        "${TASK3_GLOBAL_BATCH_SIZE}" \
        "${TASK3_MICRO_BATCH_SIZE}" \
        "${TASK3_DP}" \
        "${TASK3_SEQ_LEN}" \
        "${TASK3_HIDDEN_SIZE}" <<'PY'
import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
pp = int(sys.argv[2])
global_batch_size = int(sys.argv[3])
micro_batch_size = int(sys.argv[4])
data_parallel_size = int(sys.argv[5])
seq_length = int(sys.argv[6])
hidden_size = int(sys.argv[7])
microbatch_denominator = micro_batch_size * data_parallel_size
if microbatch_denominator <= 0 or global_batch_size <= 0:
    raise SystemExit("[ERROR] schedule batch parameters must be positive")
if global_batch_size % microbatch_denominator != 0:
    raise SystemExit(
        "[ERROR] schedule global batch size is not divisible by micro batch size times DP"
    )
expected_microbatches = global_batch_size // microbatch_denominator
if expected_microbatches <= 0:
    raise SystemExit("[ERROR] schedule must contain at least one microbatch")
expected_pp_shape = [seq_length, micro_batch_size, hidden_size]
expected_shape_text = f"input__shape={expected_pp_shape}"
expected = {f"stage{stage}_scheduling_plan.txt" for stage in range(pp)}
observed = {path.name for path in root.glob("*.txt")}
if observed != expected:
    raise SystemExit(
        f"[ERROR] schedule inventory mismatch: expected={sorted(expected)}, observed={sorted(observed)}"
    )
operation_pattern = re.compile(r"^stage:(?P<stage>[0-9]+):(?P<operation>[A-Za-z0-9_]+)\(")
pp_operations = {"recv_forward", "send_forward", "recv_backward", "send_backward"}
expected_counts = {
    "forward_step": expected_microbatches,
    "backward_step": expected_microbatches,
    "optimizer_step": 1,
}
for stage in range(pp):
    path = root / f"stage{stage}_scheduling_plan.txt"
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        match = operation_pattern.match(line)
        if match is None:
            raise SystemExit(f"[ERROR] malformed schedule record: {path}:{line_number}")
        if int(match.group("stage")) != stage:
            raise SystemExit(f"[ERROR] schedule stage identity mismatch: {path}:{line_number}")
        records.append((match.group("operation"), line, line_number))
    counts = collections.Counter(operation for operation, _, _ in records)
    for operation, expected_count in expected_counts.items():
        observed_count = counts[operation]
        if observed_count != expected_count:
            raise SystemExit(
                "[ERROR] schedule operation count mismatch: "
                f"path={path}, operation={operation}, "
                f"expected={expected_count}, observed={observed_count}"
            )
    pp_records = [record for record in records if record[0] in pp_operations]
    if pp > 1 and not pp_records:
        raise SystemExit(f"[ERROR] schedule contains no PP communication records: {path}")
    for operation, line, line_number in pp_records:
        if "group_kind=pp" not in line:
            raise SystemExit(
                f"[ERROR] PP schedule record lacks group_kind=pp: {path}:{line_number}"
            )
        if expected_shape_text not in line:
            raise SystemExit(
                "[ERROR] PP schedule tensor shape mismatch: "
                f"path={path}, line={line_number}, expected={expected_pp_shape}"
            )
        if "input__dtype=torch.bfloat16" not in line:
            raise SystemExit(
                f"[ERROR] PP schedule dtype is not torch.bfloat16: {path}:{line_number}"
            )
PY
}


# Expand representative traces to the simulator's full fake world.
task3_materialize_simulator_trace() {
    local trace_dir="${TASK3_TRACE_DIR}"
    local simulator_dir="${TASK3_RUN_ROOT}/simulator_trace"
    local expansion_metadata="${TASK3_RUN_ROOT}/provenance/simulator_trace_expansion.json"
    [[ -d "${trace_dir}" ]] || task3_error "Canonical Task3 trace directory is missing: ${trace_dir}"
    local trace_count
    trace_count=$(find "${trace_dir}" -maxdepth 1 -type f -name '*.txt' -printf '.' | wc -c)
    if (( trace_count == TASK3_WORLD_SIZE )); then TASK3_SIMULATOR_TRACE_DIR="${trace_dir}"; return 0; fi
    local representative_mode expected_representative_count
    case "${TASK3_MODEL_KEY}" in
        gpt175b)
            representative_mode=pp
            expected_representative_count=${TASK3_PP}
            ;;
        qwen3_a30b)
            representative_mode=pp_ep
            expected_representative_count=$((TASK3_PP * TASK3_EXP))
            ;;
        dsv3)
            representative_mode=pp_ep
            expected_representative_count=$((TASK3_PP * TASK3_EXP))
            ;;
        *)
            task3_error "Unsupported representative trace model: ${TASK3_MODEL_KEY}"
            ;;
    esac
    if (( trace_count != expected_representative_count )); then
        task3_error "Simulator trace inventory mismatch: observed=${trace_count}, expected world=${TASK3_WORLD_SIZE}, GPT PP=${TASK3_PP}, or Qwen PP×EP=${TASK3_PP}×${TASK3_EXP}."
    fi
    [[ ! -e "${simulator_dir}" && ! -L "${simulator_dir}" ]] || task3_error "Simulator trace destination already exists: ${simulator_dir}"
    mkdir -p "${simulator_dir}"
    "${TASK3_META_PYTHON}" - \
        "${trace_dir}" "${simulator_dir}" "${TASK3_WORLD_SIZE}" \
        "${TASK3_PP}" "${TASK3_TP}" "${TASK3_DP}" "${TASK3_EXP}" \
        "${representative_mode}" "${expansion_metadata}" <<'PY2'
import json
import pathlib
import re
import sys

source_dir = pathlib.Path(sys.argv[1]).resolve(strict=True)
destination_dir = pathlib.Path(sys.argv[2]).resolve()
world_size, pp_size, tp_size, dp_size, exp_size = map(int, sys.argv[3:8])
representative_mode = sys.argv[8]
metadata_path = pathlib.Path(sys.argv[9])
trace_files = sorted(source_dir.glob("*.txt"))
rank_pattern = re.compile(r"_rank([0-9]+)(?:_[^/]*)?\.txt$")

if representative_mode == "pp":
    expected_count = pp_size
    coordinate_names = ["pp"]
elif representative_mode == "pp_ep":
    expected_count = pp_size * exp_size
    coordinate_names = ["pp", "ep"]
else:
    raise SystemExit(f"unsupported representative trace mode: {representative_mode}")

if len(trace_files) != expected_count:
    raise SystemExit(
        "representative trace count changed during expansion: "
        f"expected={expected_count}, observed={len(trace_files)}"
    )


def coordinates(rank):
    pp_rank = rank // (tp_size * dp_size)
    if representative_mode == "pp":
        return (pp_rank,)
    ep_rank = (rank % (tp_size * dp_size)) // tp_size
    return (pp_rank, ep_rank)


def canonical_rank(key):
    if representative_mode == "pp":
        return key[0] * tp_size * dp_size
    return key[0] * tp_size * dp_size + key[1] * tp_size


representatives = {}
for trace_file in trace_files:
    match = rank_pattern.search(trace_file.name)
    if match is None:
        raise SystemExit(
            f"representative trace filename lacks rank identity: {trace_file}"
        )
    source_rank = int(match.group(1))
    if source_rank < 0 or source_rank >= world_size:
        raise SystemExit(f"representative rank is outside the fake world: {source_rank}")
    key = coordinates(source_rank)
    expected_rank = canonical_rank(key)
    if source_rank != expected_rank:
        raise SystemExit(
            "representative trace rank is not canonical for its coordinates: "
            f"rank={source_rank}, coordinates={key}, expected={expected_rank}"
        )
    if key in representatives:
        raise SystemExit(
            f"duplicate representative coordinates {key}: {trace_file}"
        )
    representatives[key] = (source_rank, trace_file)

for target_rank in range(world_size):
    key = coordinates(target_rank)
    if key not in representatives:
        raise SystemExit(
            f"missing representative for target rank {target_rank}: coordinates={key}"
        )
    source_rank, source_path = representatives[key]
    target_name = source_path.name.replace(
        f"_rank{source_rank}_", f"_rank{target_rank}_", 1
    )
    if f"_rank{target_rank}_" not in target_name:
        raise SystemExit(
            f"representative trace filename cannot encode target rank: {source_path.name}"
        )
    trace_text = source_path.read_text(encoding="utf-8")
    trace_text = re.sub(
        rf"(?m)^rank:{source_rank}:", f"rank:{target_rank}:", trace_text
    )
    trace_text = trace_text.replace(
        "group=tp,comm_func=broadcast",
        "group=tp,comm_func=load_batch_broadcast",
    )
    (destination_dir / target_name).write_text(trace_text, encoding="utf-8")

mapping = (
    "target rank maps to the captured representative with the same PP coordinate; "
    "TP and DP coordinates are expanded"
    if representative_mode == "pp"
    else "target rank maps to the captured representative with the same PP and EP "
    "coordinates; TP coordinate is expanded"
)
metadata_path.write_text(
    json.dumps(
        {
            "schema_version": "sc26-ae-task3-simulator-trace-expansion-v1",
            "source_trace_dir": str(source_dir),
            "destination_trace_dir": str(destination_dir),
            "source_trace_file_count": len(trace_files),
            "destination_trace_file_count": world_size,
            "representative_coordinates": coordinate_names,
            "topology": {
                "world_size": world_size,
                "pp_size": pp_size,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "exp_size": exp_size,
            },
            "mapping": mapping,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY2
    local expanded_count
    expanded_count=$(find "${simulator_dir}" -maxdepth 1 -type f -name '*.txt' -printf '.' | wc -c)
    (( expanded_count == TASK3_WORLD_SIZE )) || task3_error "Simulator trace expansion produced ${expanded_count} files, expected ${TASK3_WORLD_SIZE}."
    TASK3_SIMULATOR_TRACE_DIR="${simulator_dir}"
}

task3_run_simulator() {
    local resolved_json=$1
    task3_materialize_simulator_trace
    if [[ "${ARTIFACT_SOURCE}" == fresh ]]; then
        task3_expand_rank0_slowdown_blueprints
    fi
    local database_dir="${TASK3_SIMULATOR_TRACE_DIR}"
    [[ "${database_dir}" == "${TASK3_SIMULATOR_TRACE_DIR}" ]] || \
        task3_error "DATABASE_DIR must be exactly the selected simulator trace directory."

    local -a simulator_command=(
        "${TASK3_SIMULATOR_PYTHON}"
        -B
        "${TASK3_SIMULATOR}"
        --framework megatron-lm
        --mode simulate
        --trace-dir "${TASK3_SIMULATOR_TRACE_DIR}"
        --database-dir "${database_dir}"
        --schedule-dir "${TASK3_SCHEDULE_DIR}"
        --world-size "${TASK3_WORLD_SIZE}"
        --local-size "${TASK3_LOCAL_SIZE}"
        --pp-size "${TASK3_PP}"
        --tp-size "${TASK3_TP}"
        --exp-size "${TASK3_EXP}"
        --strategy 1F1B-none_interleaved
        --cc-backend analytical
        --enable-slowdown
        --overlap-mode on
        --slowdown-assets-dir "${TASK3_SLOWDOWN_ASSETS_DIR}"
        --slowdown-model-path "${TASK3_MODEL_PATH}"
        --slowdown-scaler-path "${TASK3_SCALER_PATH}"
        --no-visualize
        --report-output-dir "${TASK3_RUN_ROOT}"
        --report-model "${TASK3_MODEL_KEY}"
        --artifact-source "${ARTIFACT_SOURCE}"
    )
    task3_verify_input_expectations "${resolved_json}" simulator before
    task3_record_command simulator "${simulator_command[@]}"
    (
        cd -- "${TASK3_RUNTIME_DIR}"
        SIMULATOR_HARDWARE_TYPE="${SIMULATOR_HARDWARE_TYPE}" \
            "${simulator_command[@]}"
    ) >"${TASK3_LOG_DIR}/simulator.log" 2>&1
    task3_verify_input_expectations "${resolved_json}" simulator after
}

task3_validate_report() {
    local report_json="${TASK3_RUN_ROOT}/report.json"
    local report_markdown="${TASK3_RUN_ROOT}/report.md"
    ae_require_file "${report_json}"
    ae_require_file "${report_markdown}"

    "${TASK3_META_PYTHON}" - \
        "${report_json}" "${report_markdown}" "${TASK3_MODEL_KEY}" \
        "${ARTIFACT_SOURCE}" <<'PY'
import json
import math
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
markdown_path = pathlib.Path(sys.argv[2])
model = sys.argv[3]
source = sys.argv[4]
if json_path.is_symlink() or markdown_path.is_symlink():
    raise SystemExit("[ERROR] Task3 report files must not be symlinks")
try:
    report = json.loads(json_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"[ERROR] Task3 report.json is invalid: {exc}")
expected_fields = {
    "schema_version",
    "model",
    "artifact_source",
    "rank_id",
    "rank0_step_time_ms",
    "rank0_forward_step_duration_sum_ms",
    "rank0_backward_step_duration_sum_ms",
    "rank0_optimizer_step_duration_sum_ms",
    "rank0_comp_plus_comm_diagnostic_ms",
    "simulator_load_time_s",
    "simulator_execution_time_s",
    "simulator_wall_clock_s",
}
if not isinstance(report, dict) or set(report) != expected_fields:
    raise SystemExit(
        f"[ERROR] Task3 report fields mismatch: expected={sorted(expected_fields)}, "
        f"observed={sorted(report) if isinstance(report, dict) else type(report).__name__}"
    )
if report["schema_version"] != "sc26-ae-rank0-report-v1":
    raise SystemExit("[ERROR] Task3 report schema is invalid")
if report["model"] != model or report["artifact_source"] != source:
    raise SystemExit("[ERROR] Task3 report model/artifact_source mismatch")
rank_id = report["rank_id"]
if isinstance(rank_id, bool) or not isinstance(rank_id, int) or rank_id != 0:
    raise SystemExit("[ERROR] Task3 report rank_id must be integer zero")
numeric_fields = expected_fields - {"schema_version", "model", "artifact_source", "rank_id"}
for field in numeric_fields:
    value = report[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"[ERROR] Task3 report {field} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise SystemExit(f"[ERROR] Task3 report {field} must be finite and nonnegative")
if report["rank0_step_time_ms"] <= 0:
    raise SystemExit("[ERROR] Task3 rank0 step time must be strictly positive")
for field in (
    "rank0_forward_step_duration_sum_ms",
    "rank0_backward_step_duration_sum_ms",
    "rank0_optimizer_step_duration_sum_ms",
):
    if report[field] <= 0:
        raise SystemExit(f"[ERROR] Task3 report {field} must be strictly positive")
expected_wall = round(
    round(report["simulator_load_time_s"], 6)
    + round(report["simulator_execution_time_s"], 6),
    6,
)
if report["simulator_wall_clock_s"] != expected_wall:
    raise SystemExit("[ERROR] Task3 simulator wall-clock equality is invalid")
markdown = markdown_path.read_text(encoding="utf-8")
for key, value in report.items():
    if f"| `{key}` | `{value}` |" not in markdown:
        raise SystemExit(f"[ERROR] Task3 Markdown/JSON parity mismatch for {key}")
PY
}

task3_write_outer_manifest() {
    local metadata_path="${TASK3_WORK_ROOT}/artifact_manifest_metadata.json"
    local file_list_path="${TASK3_WORK_ROOT}/artifact_manifest_files.txt"
    local manifest_path="${TASK3_RUN_ROOT}/artifact_manifest.json"

    "${TASK3_META_PYTHON}" - \
        "${metadata_path}" "${TASK3_MODEL_KEY}" "${ARTIFACT_SOURCE}" \
        "${TASK3_CAPTURE_ID}" "${TASK3_PREDICTOR_RUN_ID}" \
        "${TASK3_SIMULATION_RUN_ID}" "${TASK3_MAIN_COMMIT}" \
        "${TASK3_ECHO_COMMIT}" "${TASK3_SIM_COMMIT}" \
        "${TASK3_WORLD_SIZE}" "${TASK3_LOCAL_SIZE}" "${TASK3_PP}" \
        "${TASK3_TP}" "${TASK3_DP}" "${TASK3_EXP}" "${TASK3_PROFILE}" \
        "${TASK3_EXECUTION_EVIDENCE}" "${TASK3_SLOWDOWN_TRACE_SCOPE}" \
        "${TASK3_SLOWDOWN_TRACE_RANK_IDS}" "${TASK3_NCU_METRICS_SOURCE}" <<'PY'
import json
import pathlib
import sys

(
    output_text,
    model,
    artifact_source,
    capture_id,
    predictor_run_id,
    simulation_run_id,
    main_commit,
    echo_commit,
    sim_commit,
    world_size,
    local_size,
    pp,
    tp,
    dp,
    exp,
    profile,
    execution_evidence,
    slowdown_trace_scope,
    slowdown_trace_rank_ids_text,
    ncu_metrics_source,
) = sys.argv[1:]
try:
    slowdown_trace_rank_ids = json.loads(slowdown_trace_rank_ids_text)
except json.JSONDecodeError as exc:
    raise SystemExit(f"[ERROR] Task3 slowdown trace rank ids are invalid JSON: {exc}")
if slowdown_trace_scope != "global_rank_0" or slowdown_trace_rank_ids != [0]:
    raise SystemExit("[ERROR] Task3 manifest slowdown trace provenance must describe global rank 0 only")
if ncu_metrics_source not in {"task1_rank0", "synthetic_fixture_compatibility"}:
    raise SystemExit(f"[ERROR] Task3 manifest NCU metrics source is invalid: {ncu_metrics_source}")
if execution_evidence != "local_synthetic_not_gpu_qualification" and ncu_metrics_source != "task1_rank0":
    raise SystemExit("[ERROR] Real Task3 manifest requires Task1 rank-0 NCU provenance")
payload = {
    "schema_version": "sc26-ae-artifact-manifest-v1",
    "model": model,
    "task": "task3",
    "artifact_source": artifact_source,
    "capture_id": capture_id,
    "predictor_run_id": predictor_run_id,
    "simulation_run_id": simulation_run_id,
    "source_commits": {
        "megatron_lm": main_commit,
        "echo_slowdown": echo_commit,
        "megatron_sim_engine": sim_commit,
    },
    "simulation_topology": {
        "world_size": int(world_size),
        "local_size": int(local_size),
        "pp": int(pp),
        "tp": int(tp),
        "dp": int(dp),
        "exp": int(exp),
    },
    "profile": profile,
    "precision": "bf16",
    "ddp_overlap": True,
    "communication_backend": "analytical",
    "overlap_mode": "on",
    "database_is_trace_dir": True,
    "execution_evidence": execution_evidence,
    "slowdown_trace_scope": slowdown_trace_scope,
    "slowdown_trace_rank_ids": slowdown_trace_rank_ids,
    "ncu_metrics_source": ncu_metrics_source,
}
pathlib.Path(output_text).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    "${TASK3_META_PYTHON}" - "${TASK3_RUN_ROOT}" "${file_list_path}" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
output = pathlib.Path(sys.argv[2])
paths = []
for current_root, directory_names, file_names in os.walk(root, followlinks=False):
    current = pathlib.Path(current_root)
    for name in directory_names:
        path = current / name
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise SystemExit(f"[ERROR] Task3 run contains a non-directory or symlink: {path}")
    for name in file_names:
        path = current / name
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise SystemExit(f"[ERROR] Task3 run contains a non-regular file or symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if relative != "artifact_manifest.json":
            paths.append(relative)
if not paths:
    raise SystemExit("[ERROR] Task3 run contains no payload files")
output.write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
PY

    "${TASK3_META_PYTHON}" -B "${TASK3_ARTIFACT_TOOL}" create \
        --root "${TASK3_RUN_ROOT}" \
        --metadata-json "${metadata_path}" \
        --file-list "${file_list_path}" \
        --output "${manifest_path}"
    task3_verify_artifact_manifest "${TASK3_RUN_ROOT}" "${manifest_path}"
    TASK3_MANIFEST_SHA256=$(sha256sum "${manifest_path}" | awk '{print $1}')
    [[ "${TASK3_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
        task3_error "Failed to calculate Task3 manifest SHA256."
}

task3_publish_marker() {
    mkdir -p "${TASK3_TASK_DIR}"
    "${TASK3_META_PYTHON}" - \
        "${TASK3_MARKER_PATH}" "${TASK3_TASK_DIR}" "${TASK3_RUN_ROOT}" \
        "${TASK3_MODEL_KEY}" "${ARTIFACT_SOURCE}" \
        "${TASK3_SIMULATION_RUN_ID}" "${TASK3_CAPTURE_ID}" \
        "${TASK3_PREDICTOR_RUN_ID}" "${TASK3_MANIFEST_SHA256}" \
        "${TASK3_EXECUTION_EVIDENCE}" "${TASK3_SLOWDOWN_TRACE_SCOPE}" \
        "${TASK3_SLOWDOWN_TRACE_RANK_IDS}" "${TASK3_NCU_METRICS_SOURCE}" <<'PY'
import json
import pathlib
import sys

(
    marker_text,
    task_dir_text,
    run_root_text,
    model,
    artifact_source,
    simulation_run_id,
    capture_id,
    predictor_run_id,
    manifest_sha256,
    execution_evidence,
    slowdown_trace_scope,
    slowdown_trace_rank_ids_text,
    ncu_metrics_source,
) = sys.argv[1:]
try:
    slowdown_trace_rank_ids = json.loads(slowdown_trace_rank_ids_text)
except json.JSONDecodeError as exc:
    raise SystemExit(f"[ERROR] Task3 marker slowdown trace rank ids are invalid JSON: {exc}")
if slowdown_trace_scope != "global_rank_0" or slowdown_trace_rank_ids != [0]:
    raise SystemExit("[ERROR] Task3 marker slowdown trace provenance must describe global rank 0 only")
if ncu_metrics_source not in {"task1_rank0", "synthetic_fixture_compatibility"}:
    raise SystemExit(f"[ERROR] Task3 marker NCU metrics source is invalid: {ncu_metrics_source}")
task_dir = pathlib.Path(task_dir_text).resolve(strict=True)
run_root = pathlib.Path(run_root_text).resolve(strict=True)
expected = task_dir / "runs" / simulation_run_id
if run_root != expected:
    raise SystemExit(
        f"[ERROR] Refusing Task3 marker path outside selected run: {run_root} != {expected}"
    )
manifest = run_root / "artifact_manifest.json"
if not manifest.is_file():
    raise SystemExit("[ERROR] Task3 marker cannot reference a missing artifact manifest")
try:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] Task3 marker cannot read its artifact manifest: {exc}")
if manifest_payload.get("execution_evidence") != execution_evidence:
    raise SystemExit(
        "[ERROR] Task3 marker evidence does not match its artifact manifest"
    )
if manifest_payload.get("slowdown_trace_scope") != slowdown_trace_scope:
    raise SystemExit("[ERROR] Task3 marker slowdown trace scope does not match its artifact manifest")
if manifest_payload.get("slowdown_trace_rank_ids") != slowdown_trace_rank_ids:
    raise SystemExit("[ERROR] Task3 marker slowdown trace ranks do not match its artifact manifest")
if manifest_payload.get("ncu_metrics_source") != ncu_metrics_source:
    raise SystemExit("[ERROR] Task3 marker NCU source does not match its artifact manifest")
payload = {
    "schema_version": "sc26-ae-task3-run-marker-v1",
    "task": "task3",
    "model": model,
    "artifact_source": artifact_source,
    "simulation_run_id": simulation_run_id,
    "capture_id": capture_id,
    "predictor_run_id": predictor_run_id,
    "run_path": f"runs/{simulation_run_id}",
    "manifest_sha256": manifest_sha256,
    "artifact_manifest_sha256": manifest_sha256,
    "execution_evidence": execution_evidence,
    "slowdown_trace_scope": slowdown_trace_scope,
    "slowdown_trace_rank_ids": slowdown_trace_rank_ids,
    "ncu_metrics_source": ncu_metrics_source,
    "verified": True,
}
pathlib.Path(marker_text).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

ae_run_task3() {
    TASK3_MODEL_KEY=${1:-}
    ae_require_enum "Task3 model" "${TASK3_MODEL_KEY}" gpt175b qwen3_a30b dsv3
    ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-}
    ae_require_enum ARTIFACT_SOURCE "${ARTIFACT_SOURCE}" fresh prebaked

    TASK3_EXECUTION_MODE=${TASK3_EXECUTION_MODE:-real}
    ae_require_enum TASK3_EXECUTION_MODE "${TASK3_EXECUTION_MODE}" real synthetic
    TASK3_ALLOW_FUNCTIONAL_PREBAKED=${TASK3_ALLOW_FUNCTIONAL_PREBAKED:-0}
    ae_require_enum TASK3_ALLOW_FUNCTIONAL_PREBAKED "${TASK3_ALLOW_FUNCTIONAL_PREBAKED}" 0 1
    if [[ "${TASK3_ALLOW_FUNCTIONAL_PREBAKED}" == "1" && "${ARTIFACT_SOURCE}" != "prebaked" ]]; then
        task3_error "TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 requires ARTIFACT_SOURCE=prebaked."
    fi
    TASK3_EXECUTION_EVIDENCE=$(task3_execution_evidence)
    [[ -n "${SIMULATOR_HARDWARE_TYPE:-}" ]] || \
        task3_error "SIMULATOR_HARDWARE_TYPE is required for Task3 CPU execution."

    task3_bind_interpreters
    TASK3_SIM_ENGINE_ROOT=${TASK3_SIM_ENGINE_ROOT:-"${TASK3_REPO_ROOT}/megatron-sim-engine"}
    TASK3_BUILDER=${TASK3_BUILDER:-"${TASK3_SIM_ENGINE_ROOT}/tools/data_prep/slowdown/build_ddp_slowdown_assets.py"}
    TASK3_SCHEDULER=${TASK3_SCHEDULER:-"${TASK3_SIM_ENGINE_ROOT}/src/scheduler/mg_scheduling/mg_test.py"}
    TASK3_SIMULATOR=${TASK3_SIMULATOR:-"${TASK3_SIM_ENGINE_ROOT}/simu_main.py"}
    TASK3_ARTIFACT_TOOL="${TASK3_REPO_ROOT}/SC26-AE/tools/artifact_manifest.py"

    ae_require_command git
    ae_require_command sha256sum
    ae_require_command "${TASK3_META_PYTHON}"
    if [[ "${TASK3_EXECUTION_MODE}" == synthetic ]]; then
        ae_require_command "${TASK3_SIMULATOR_PYTHON}"
    else
        ae_require_file "${TASK3_SIMULATOR_PYTHON}"
    fi
    ae_require_dir "${TASK3_SIM_ENGINE_ROOT}"
    ae_require_file "${TASK3_SCHEDULER}"
    ae_require_file "${TASK3_SIMULATOR}"
    ae_require_file "${TASK3_ARTIFACT_TOOL}"
    if [[ "${ARTIFACT_SOURCE}" == fresh ]]; then
        ae_require_file "${TASK3_BUILDER}"
    fi
    task3_current_commits

    local output_root=${AE_OUTPUT_ROOT:-"${TASK3_REPO_ROOT}/SC26-AE/output"}
    [[ "${output_root}" == /* ]] || \
        task3_error "AE_OUTPUT_ROOT must be an absolute path: ${output_root}"
    mkdir -p "${output_root}/_work"

    task3_load_model_config "${TASK3_MODEL_KEY}"
    task3_prepare_run "${TASK3_MODEL_KEY}" "${output_root}"

    local resolved_json="${TASK3_WORK_ROOT}/resolved_inputs.json"
    case "${ARTIFACT_SOURCE}" in
        fresh)
            task3_resolve_fresh "${TASK3_MODEL_KEY}" "${output_root}" "${resolved_json}"
            ;;
        prebaked)
            task3_resolve_prebaked "${TASK3_MODEL_KEY}" "${resolved_json}"
            ;;
        *)
            task3_error "Internal ARTIFACT_SOURCE branch failure: ${ARTIFACT_SOURCE}"
            ;;
    esac

    task3_materialize_slowdown_trace "${resolved_json}"
    task3_write_input_evidence "${resolved_json}"
    task3_materialize_slowdown_assets "${resolved_json}"
    task3_generate_schedule
    task3_run_simulator "${resolved_json}"
    task3_validate_report
    task3_write_outer_manifest
    task3_publish_marker

    printf '[PASS] Task3 model=%s artifact_source=%s simulation_run_id=%s\n' \
        "${TASK3_MODEL_KEY}" "${ARTIFACT_SOURCE}" "${TASK3_SIMULATION_RUN_ID}"
    printf 'TASK3_RUN_ROOT=%s\n' "${TASK3_RUN_ROOT}"
    printf 'TASK3_MARKER=%s\n' "${TASK3_MARKER_PATH}"
    printf 'TASK3_MANIFEST_SHA256=%s\n' "${TASK3_MANIFEST_SHA256}"
}
