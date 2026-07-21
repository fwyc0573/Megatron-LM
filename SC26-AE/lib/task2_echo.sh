#!/usr/bin/env bash
# Isolated Echo-slowdown Task2 runner.
#
# All mutable commands run from a filtered git archive under output/_work.  The
# pinned Echo checkout is read-only and is never used as a working directory.

set -uo pipefail

TASK2_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK2_REPO_ROOT=$(cd "${TASK2_SCRIPT_DIR}/../.." && pwd)
TASK2_COMMON_SH="${TASK2_SCRIPT_DIR}/common.sh"
if [[ -f "$TASK2_COMMON_SH" ]]; then
    # shellcheck disable=SC1090
    source "$TASK2_COMMON_SH"
fi
# Failure evidence is written by the explicit return-code path below.  Do not
# let a shared helper's shell option terminate the process before that record
# is emitted.
set +e

TASK2_OUTPUT_ROOT=${AE_OUTPUT_ROOT:-${TASK2_REPO_ROOT}/SC26-AE/output}
TASK2_SOURCE_REPO=${TASK2_SOURCE_REPO:-${TASK2_REPO_ROOT}/Echo-slowdown}
TASK2_MODE=${TASK2_EXECUTION_MODE:-real}
TASK2_REBUILD=${REBUILD:-0}
TASK2_MODEL_KEY=${1:-}
TASK2_PREDICTOR_RUN_ID=${PREDICTOR_RUN_ID:-${TASK2_PREDICTOR_RUN_ID:-}}
# Fixed v1.2-ae worker binding. Do not discover or switch interpreters.
TASK2_FIXED_PYTHON=/opt/conda/envs/echo_slowdown/bin/python
TASK2_META_PYTHON=${TASK2_META_PYTHON:-/opt/conda/envs/echo_slowdown/bin/python}
TASK2_PYTHON=${TASK2_PYTHON:-/opt/conda/envs/echo_slowdown/bin/python}
TASK2_ARTIFACT_TOOL=${TASK2_REPO_ROOT}/SC26-AE/tools/artifact_manifest.py
TASK2_UPDATE_COMMAND_OVERRIDDEN=0
[[ ${TASK2_UPDATE_COMMAND+x} == x ]] && TASK2_UPDATE_COMMAND_OVERRIDDEN=1
TASK2_RUN_COMMAND_OVERRIDDEN=0
[[ ${TASK2_RUN_COMMAND+x} == x ]] && TASK2_RUN_COMMAND_OVERRIDDEN=1
TASK2_UPDATE_COMMAND=${TASK2_UPDATE_COMMAND:-"${TASK2_PYTHON} update_configs.py"}
TASK2_RUN_COMMAND=${TASK2_RUN_COMMAND:-"bash run_all.sh"}
TASK2_SKIP_UPDATE_CONFIGS=${TASK2_SKIP_UPDATE_CONFIGS:-0}
TASK2_SKIP_PREDICT=${TASK2_SKIP_PREDICT:-0}
TASK2_SKIP_HARDWARE_CHECK=${TASK2_SKIP_HARDWARE_CHECK:-0}
TASK2_EXPECTED_COMMIT=${TASK2_GITLINK_COMMIT:-}

# These paths are the reviewed historical generated/runtime paths in the
# pinned upstream commit.  A new tracked path under an excluded prefix fails
# the source-inventory gate instead of being silently filtered.
TASK2_HISTORICAL_EXCLUDED_PATHS=(
    "merge/input/kernel_metric_output.csv"
    "merge/input/slowdown_stats_output_device_0.xlsx"
    "merge/output/merged_features.csv"
    "training_testing/input/test_csv/merged_features.csv"
    "training_testing/input/train_csv/merged_features.csv"
    "training_testing/output/prediction/feature_importance_merged_features.png"
    "training_testing/output/prediction/output_df_merged_features.csv"
    "training_testing/output/prediction/output_full_df_merged_features.csv"
    "training_testing/output/prediction/output_metrics.txt"
    "training_testing/output/train_dataset.csv"
    "training_testing/output/xgb_model.json"
)
TASK2_EXCLUDED_PREFIXES=(
    "merge/input/kernel_metric_output.csv"
    "merge/input/slowdown_stats_output_device_0.xlsx"
    "merge/output/"
    "training_testing/input/test_csv/"
    "training_testing/input/train_csv/"
    "training_testing/output/prediction/"
    "training_testing/output/train_dataset.csv"
    "training_testing/output/xgb_model.json"
)

TASK2_FAILURE_ROOT=""
TASK2_FAILURE_REASON=""
TASK2_WORK_ROOT=""
TASK2_RUN_ROOT=""
TASK2_SOURCE_ROOT=""
TASK2_ECHO_COMMIT=""
TASK2_MAIN_COMMIT=""
TASK2_COMMIT_PROVENANCE=""
TASK2_VERIFIED_MANIFEST_SHA256=""
TASK2_FIXED_CANONICAL_PATH=""
TASK2_FIXED_SHA256=""
TASK2_PATH_PYTHON=""
TASK2_PATH_PYTHON_CANONICAL_PATH=""
TASK2_PATH_PYTHON_SHA256=""
TASK2_INTERPRETER_BINDING_PATH=""
TASK2_INTERPRETER_BASELINE_PATH=""
TASK2_INTERPRETER_BASELINE_SET=0
TASK2_INTERPRETER_CONFIGS=(
    "kernel_metric/input/global_config.json"
    "merge/input/global_config.json"
    "slowdown_collection/input/global_config.json"
    "training_testing/input/global_config.json"
)

task2_error() {
    TASK2_FAILURE_REASON=$*
    printf '[ERROR] %s\n' "$*" >&2
    return 1
}

task2_require_command() {
    command -v "$1" >/dev/null 2>&1 || task2_error "required command is unavailable: $1"
}

task2_require_file() {
    [[ -f "$1" ]] || task2_error "required file is unavailable: $1"
}

task2_validate_interpreter_path() {
    local path=$1 label=$2
    [[ "$path" == /* ]] || {
        task2_error "$label must be an absolute path: $path"
        return 1
    }
    [[ "$path" != *[[:space:]]* ]] || {
        task2_error "$label must not contain whitespace: $path"
        return 1
    }
    [[ -e "$path" && -x "$path" ]] || {
        task2_error "$label is missing or not executable: $path"
        return 1
    }
}

task2_hash_file() {
    local path=$1 label=$2 digest
    [[ -f "$path" && ! -L "$path" ]] || {
        task2_error "$label must be a regular non-symlink file: $path"
        return 1
    }
    digest=$(sha256sum -- "$path" | awk '{print $1}') || {
        task2_error "cannot hash $label: $path"
        return 1
    }
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
        task2_error "invalid SHA256 for $label: $path"
        return 1
    }
    printf '%s\n' "$digest"
}

task2_bind_interpreter_chain() {
    [[ "$TASK2_MODE" == real ]] || return 0
    task2_require_command realpath || return 1
    task2_require_command sha256sum || return 1
    task2_validate_interpreter_path "$TASK2_FIXED_PYTHON" "fixed Echo Python" || return 1

    local canonical fixed_hash path_python path_canonical path_hash
    canonical=$(realpath -e -- "$TASK2_FIXED_PYTHON") || {
        task2_error "cannot resolve fixed Echo Python canonical path: $TASK2_FIXED_PYTHON"
        return 1
    }
    [[ -f "$canonical" && -x "$canonical" && ! -L "$canonical" ]] || {
        task2_error "fixed Echo Python canonical target is not a regular executable: $canonical"
        return 1
    }
    fixed_hash=$(task2_hash_file "$canonical" "fixed Echo Python") || return 1
    if [[ -n "$TASK2_FIXED_CANONICAL_PATH" && "$TASK2_FIXED_CANONICAL_PATH" != "$canonical" ]]; then
        task2_error "fixed Echo Python canonical path changed during execution"
        return 1
    fi
    if [[ -n "$TASK2_FIXED_SHA256" && "$TASK2_FIXED_SHA256" != "$fixed_hash" ]]; then
        task2_error "fixed Echo Python SHA256 changed during execution"
        return 1
    fi

    if [[ "$TASK2_INTERPRETER_BASELINE_SET" == 0 ]]; then
        TASK2_INTERPRETER_BASELINE_PATH=$PATH
        TASK2_INTERPRETER_BASELINE_SET=1
    fi
    export PATH="$(dirname -- "$TASK2_FIXED_PYTHON"):$TASK2_INTERPRETER_BASELINE_PATH"
    hash -r 2>/dev/null

    path_python=$(command -v python 2>/dev/null) || {
        task2_error "PATH does not resolve a python executable after fixed binding"
        return 1
    }
    [[ "$path_python" == /* ]] || {
        task2_error "PATH python resolution is not absolute: $path_python"
        return 1
    }
    path_canonical=$(realpath -e -- "$path_python") || {
        task2_error "cannot resolve PATH python canonical path: $path_python"
        return 1
    }
    path_hash=$(task2_hash_file "$path_canonical" "PATH python") || return 1
    [[ "$path_python" == "$TASK2_FIXED_PYTHON" ]] || {
        task2_error "PATH python lexical path does not match the fixed Echo Python"
        return 1
    }
    [[ "$path_canonical" == "$canonical" && "$path_hash" == "$fixed_hash" ]] || {
        task2_error "PATH python does not match the fixed Echo Python"
        return 1
    }

    local candidate candidate_canonical candidate_hash
    for candidate in "$TASK2_META_PYTHON" "$TASK2_PYTHON"; do
        task2_validate_interpreter_path "$candidate" "Task2 interpreter" || return 1
        candidate_canonical=$(realpath -e -- "$candidate") || {
            task2_error "cannot resolve Task2 interpreter canonical path: $candidate"
            return 1
        }
        candidate_hash=$(task2_hash_file "$candidate_canonical" "Task2 interpreter") || return 1
        [[ "$candidate_canonical" == "$canonical" && "$candidate_hash" == "$fixed_hash" ]] || {
            task2_error "Task2 interpreter does not match the fixed Echo Python: $candidate"
            return 1
        }
    done

    TASK2_FIXED_CANONICAL_PATH=$canonical
    TASK2_FIXED_SHA256=$fixed_hash
    TASK2_PATH_PYTHON=$path_python
    TASK2_PATH_PYTHON_CANONICAL_PATH=$path_canonical
    TASK2_PATH_PYTHON_SHA256=$path_hash
}

task2_validate_interpreter_chain() {
    [[ "$TASK2_MODE" == real ]] || return 0
    [[ -n "$TASK2_FIXED_CANONICAL_PATH" && -n "$TASK2_FIXED_SHA256" ]] || {
        task2_error "Task2 interpreter chain was not bound before validation"
        return 1
    }
    task2_bind_interpreter_chain || return 1
    [[ "$TASK2_FIXED_CANONICAL_PATH" == "$TASK2_PATH_PYTHON_CANONICAL_PATH" \
        && "$TASK2_FIXED_SHA256" == "$TASK2_PATH_PYTHON_SHA256" ]] || {
        task2_error "Task2 interpreter chain identity is inconsistent"
        return 1
    }
}

task2_validate_nested_configs() {
    [[ "$TASK2_MODE" == real ]] || return 0
    local sidecar_path="" write_sidecar=1
    [[ "$#" -ge 1 ]] && sidecar_path=$1
    [[ "$#" -ge 2 ]] && write_sidecar=$2
    [[ -n "$sidecar_path" ]] || sidecar_path="$TASK2_RUN_ROOT/interpreter_binding.json"
    [[ -n "$TASK2_RUN_ROOT" && "$sidecar_path" == "$TASK2_RUN_ROOT/interpreter_binding.json" ]] || {
        task2_error "nested interpreter sidecar must be the Task2 run-root binding artifact"
        return 1
    }
    [[ "$write_sidecar" == 0 || "$write_sidecar" == 1 ]] || {
        task2_error "nested interpreter sidecar write mode must be 0 or 1"
        return 1
    }
    [[ -n "$TASK2_SOURCE_ROOT" && -d "$TASK2_SOURCE_ROOT" ]] || {
        task2_error "Echo snapshot root is unavailable for nested interpreter validation"
        return 1
    }
    "$TASK2_META_PYTHON" - "$TASK2_SOURCE_ROOT" "$TASK2_FIXED_PYTHON" \
        "$TASK2_FIXED_CANONICAL_PATH" "$TASK2_FIXED_SHA256" "$TASK2_PATH_PYTHON" \
        "$TASK2_PATH_PYTHON_CANONICAL_PATH" "$TASK2_PATH_PYTHON_SHA256" \
        "$sidecar_path" "$write_sidecar" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
requested = sys.argv[2]
fixed_canonical = pathlib.Path(sys.argv[3]).resolve(strict=True)
fixed_sha256 = sys.argv[4]
path_python = sys.argv[5]
path_python_canonical = pathlib.Path(sys.argv[6]).resolve(strict=True)
path_python_sha256 = sys.argv[7]
sidecar = pathlib.Path(sys.argv[8])
write_sidecar = sys.argv[9] == "1"
config_paths = [
    "kernel_metric/input/global_config.json",
    "merge/input/global_config.json",
    "slowdown_collection/input/global_config.json",
    "training_testing/input/global_config.json",
]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def require_regular(path, label):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise SystemExit("{} is missing: {}".format(label, path))
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SystemExit("{} is not a regular non-symlink file: {}".format(label, path))

def safe_absolute(value, label):
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or any(ord(char) < 32 or ord(char) == 127 or char.isspace() for char in value)
    ):
        raise SystemExit("{} is unsafe: {}".format(label, value))

def reject_duplicate_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key: {}".format(key))
        payload[key] = value
    return payload

safe_absolute(requested, "fixed requested interpreter path")
if fixed_canonical != path_python_canonical:
    raise SystemExit("PATH python canonical path differs from fixed interpreter")
if digest(fixed_canonical) != fixed_sha256 or path_python_sha256 != fixed_sha256:
    raise SystemExit("PATH python SHA256 differs from fixed interpreter")
if path_python != requested:
    raise SystemExit("PATH python lexical path differs from fixed requested path")

rows = []
archive_root = sidecar.parent / "provenance" / "interpreter_configs"
for relative in config_paths:
    config = root / relative
    require_regular(config, "nested interpreter config")
    canonical_config = config.resolve(strict=True)
    try:
        canonical_config.relative_to(root)
    except ValueError:
        raise SystemExit("nested interpreter config escapes the Echo snapshot: " + relative)
    try:
        payload = json.loads(config.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
    except Exception as exc:
        raise SystemExit("nested interpreter config is not valid JSON: {} ({})".format(relative, exc))
    if not isinstance(payload, dict):
        raise SystemExit("nested interpreter config must be a JSON object: " + relative)
    value = payload.get("python_path")
    safe_absolute(value, "nested python_path")
    if value != requested:
        raise SystemExit("nested python_path differs from fixed requested path: " + relative)
    executable = pathlib.Path(value).resolve(strict=True)
    require_regular(executable, "nested Python executable")
    if not os.access(executable, os.X_OK):
        raise SystemExit("nested Python executable is not executable: " + relative)
    executable_sha256 = digest(executable)
    if executable != fixed_canonical or executable_sha256 != fixed_sha256:
        raise SystemExit("nested python_path identity differs from fixed interpreter: " + relative)

    archived_relative = "provenance/interpreter_configs/{}.global_config.json".format(
        pathlib.PurePosixPath(relative).parts[0]
    )
    archived = sidecar.parent / archived_relative
    if write_sidecar:
        if os.path.lexists(archived):
            raise SystemExit("interpreter config archive already exists: " + archived_relative)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(config.read_bytes())
    require_regular(archived, "archived interpreter config")
    config_bytes = archived.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != digest(config):
        raise SystemExit("archived interpreter config differs from live config: " + relative)
    rows.append({
        "source_relative_path": relative,
        "artifact_relative_path": archived_relative,
        "size_bytes": len(config_bytes),
        "sha256": config_sha256,
        "python_path": value,
        "python_canonical_path": executable.as_posix(),
        "python_sha256": executable_sha256,
    })

expected = {
    "schema_version": "sc26-ae-task2-interpreter-binding-v1",
    "status": "bound",
    "execution_mode": "real",
    "automatic_fallback": False,
    "fixed_requested_path": requested,
    "fixed_canonical_path": fixed_canonical.as_posix(),
    "fixed_sha256": fixed_sha256,
    "path_lookup_python": path_python,
    "path_lookup_canonical_path": path_python_canonical.as_posix(),
    "path_lookup_sha256": path_python_sha256,
    "configs": rows,
}
if write_sidecar:
    if os.path.lexists(sidecar):
        raise SystemExit("interpreter binding sidecar already exists")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
else:
    require_regular(sidecar, "interpreter binding sidecar")
    try:
        observed = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
    except Exception as exc:
        raise SystemExit("interpreter binding sidecar is invalid: {}".format(exc))
    if observed != expected:
        raise SystemExit("interpreter binding sidecar does not match current identity")
PY
    local rc=$?
    [[ "$rc" == 0 ]] || {
        task2_error "nested Echo interpreter validation failed"
        return 1
    }
    TASK2_INTERPRETER_BINDING_PATH="$sidecar_path"
}

task2_attach_interpreter_provenance() {
    [[ "$TASK2_MODE" == real ]] || return 0
    local provenance="$TASK2_RUN_ROOT/provenance.json"
    local sidecar="$TASK2_INTERPRETER_BINDING_PATH"
    [[ -n "$sidecar" && -f "$sidecar" && ! -L "$sidecar" ]] || {
        task2_error "interpreter binding sidecar is unavailable for provenance"
        return 1
    }
    "$TASK2_META_PYTHON" - "$provenance" "$sidecar" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

provenance = pathlib.Path(sys.argv[1])
sidecar = pathlib.Path(sys.argv[2])
if not provenance.is_file() or provenance.is_symlink():
    raise SystemExit("Task2 provenance is not a regular file")

def reject_duplicate_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key: {}".format(key))
        payload[key] = value
    return payload

payload = json.loads(
    provenance.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs
)
if not isinstance(payload, dict):
    raise SystemExit("Task2 provenance must be a JSON object")
binding = {
    "status": "bound",
    "artifact_path": "interpreter_binding.json",
    "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
}
if "interpreter_binding" in payload and payload["interpreter_binding"] != binding:
    raise SystemExit("Task2 provenance interpreter binding is inconsistent")
payload["interpreter_binding"] = binding
provenance.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    local rc=$?
    [[ "$rc" == 0 ]] || {
        task2_error "Task2 provenance interpreter binding update failed"
        return 1
    }
}

task2_validate_binding_sidecar() {
    local run_root=${1-}
    [[ -n "$run_root" ]] || run_root=$TASK2_RUN_ROOT
    local live_required=0
    [[ "$TASK2_MODE" == real ]] && live_required=1
    [[ -n "$run_root" && -d "$run_root" ]] || {
        task2_error "Task2 interpreter binding run root is unavailable: $run_root"
        return 1
    }
    "$TASK2_META_PYTHON" - "$run_root" "$live_required" "$TASK2_FIXED_PYTHON" \
        "$TASK2_FIXED_CANONICAL_PATH" "$TASK2_FIXED_SHA256" "$TASK2_PATH_PYTHON" \
        "$TASK2_PATH_PYTHON_CANONICAL_PATH" "$TASK2_PATH_PYTHON_SHA256" <<'PY'
import hashlib
import json
import os
import pathlib
import posixpath
import re
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
live_required = sys.argv[2] == "1"
live_requested = sys.argv[3]
live_canonical_raw = sys.argv[4]
live_sha256 = sys.argv[5]
live_path_python = sys.argv[6]
live_path_canonical_raw = sys.argv[7]
live_path_sha256 = sys.argv[8]
sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
config_paths = [
    "kernel_metric/input/global_config.json",
    "merge/input/global_config.json",
    "slowdown_collection/input/global_config.json",
    "training_testing/input/global_config.json",
]
pending_evidence = "runtime_measurement_requires_external_two_gpu_qualification"
qualified_evidence = "real_exact_two_h800_qualified"
synthetic_evidence = "local_synthetic_not_two_gpu_qualification"


def fail(message):
    raise SystemExit(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_regular(path, label):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        fail("{} is missing: {}".format(label, path))
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        fail("{} is not a regular non-symlink file: {}".format(label, path))


def require_executable(path, label):
    require_regular(path, label)
    if not os.access(path, os.X_OK):
        fail("{} is not executable: {}".format(label, path))


def safe_absolute(value, label):
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or any(ord(char) < 32 or ord(char) == 127 or char.isspace() for char in value)
    ):
        fail("{} is unsafe: {}".format(label, value))


def require_canonical_absolute(value, label):
    safe_absolute(value, label)
    if (
        value.startswith("//")
        or value != posixpath.normpath(value)
        or value != pathlib.PurePosixPath(value).as_posix()
    ):
        fail("{} is not lexically canonical: {}".format(label, value))


def require_hash(value, label):
    if not isinstance(value, str) or not sha256_pattern.fullmatch(value):
        fail("{} is not a lowercase SHA256 digest".format(label))


def reject_duplicate_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            fail("duplicate JSON key: {}".format(key))
        payload[key] = value
    return payload


def load_object(path, label):
    require_regular(path, label)
    try:
        payload = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
        )
    except Exception as exc:
        fail("{} is invalid JSON: {}".format(label, exc))
    if not isinstance(payload, dict):
        fail("{} must be a JSON object".format(label))
    return payload


manifest = load_object(root / "artifact_manifest.json", "Task2 artifact manifest")
manifest_evidence = manifest.get("execution_evidence")
if manifest_evidence == synthetic_evidence:
    if live_required:
        fail("real caller cannot consume a synthetic Task2 bundle")
    if not isinstance(manifest.get("files"), list):
        fail("synthetic Task2 artifact manifest files is not an array")
    # A local synthetic bundle has no real interpreter contract.  The generic
    # artifact verifier remains responsible for its complete file/checksum set.
    raise SystemExit(0)
if manifest_evidence not in {pending_evidence, qualified_evidence}:
    fail("Task2 artifact manifest execution evidence is not a real-chain state")

sidecar = root / "interpreter_binding.json"
sidecar_payload = load_object(sidecar, "interpreter binding sidecar")
sidecar_keys = {
    "schema_version",
    "status",
    "execution_mode",
    "automatic_fallback",
    "fixed_requested_path",
    "fixed_canonical_path",
    "fixed_sha256",
    "path_lookup_python",
    "path_lookup_canonical_path",
    "path_lookup_sha256",
    "configs",
}
if set(sidecar_payload) != sidecar_keys:
    fail("interpreter binding sidecar keys are not exact")
if sidecar_payload["schema_version"] != "sc26-ae-task2-interpreter-binding-v1":
    fail("unexpected interpreter binding sidecar schema")
if sidecar_payload["status"] != "bound" or sidecar_payload["execution_mode"] != "real":
    fail("interpreter binding sidecar is not a bound real-mode record")
if sidecar_payload["automatic_fallback"] is not False:
    fail("interpreter binding sidecar enables automatic fallback")

requested = sidecar_payload["fixed_requested_path"]
fixed_canonical = sidecar_payload["fixed_canonical_path"]
fixed_sha256 = sidecar_payload["fixed_sha256"]
path_python = sidecar_payload["path_lookup_python"]
path_canonical = sidecar_payload["path_lookup_canonical_path"]
path_sha256 = sidecar_payload["path_lookup_sha256"]
safe_absolute(requested, "sidecar fixed requested interpreter path")
require_canonical_absolute(fixed_canonical, "canonical interpreter path")
safe_absolute(path_python, "sidecar PATH python path")
require_canonical_absolute(path_canonical, "PATH canonical interpreter path")
require_hash(fixed_sha256, "sidecar fixed interpreter SHA256")
require_hash(path_sha256, "sidecar PATH interpreter SHA256")
if path_python != requested:
    fail("sidecar PATH lookup differs from fixed requested path")
if path_canonical != fixed_canonical or path_sha256 != fixed_sha256:
    fail("sidecar PATH identity differs from fixed interpreter")

# Every real-evidence bundle must remain bound to the wrapper's fixed
# requested interpreter path, even when a synthetic consumer cannot inspect
# the worker filesystem.  Internal sidecar consistency alone would otherwise
# allow a bundle to rename the entire interpreter chain coherently.
safe_absolute(live_requested, "wrapper fixed requested interpreter path")
if requested != live_requested:
    fail("sidecar fixed requested path differs from wrapper fixed interpreter")

if live_required:
    safe_absolute(live_requested, "fixed requested interpreter path")
    safe_absolute(live_canonical_raw, "fixed canonical interpreter path")
    safe_absolute(live_path_python, "PATH python path")
    safe_absolute(live_path_canonical_raw, "PATH canonical interpreter path")
    require_hash(live_sha256, "fixed interpreter SHA256")
    require_hash(live_path_sha256, "PATH interpreter SHA256")
    if live_requested != requested:
        fail("sidecar fixed requested path differs from current binding")
    if live_canonical_raw != fixed_canonical or live_sha256 != fixed_sha256:
        fail("sidecar fixed executable identity differs from current binding")
    if live_path_python != path_python or live_path_canonical_raw != path_canonical:
        fail("sidecar PATH lookup differs from current binding")
    if live_path_sha256 != path_sha256:
        fail("sidecar PATH SHA256 differs from current binding")
    live_canonical = pathlib.Path(live_canonical_raw)
    try:
        resolved_live_canonical = live_canonical.resolve(strict=True)
    except FileNotFoundError:
        fail("fixed canonical interpreter is missing")
    if resolved_live_canonical.as_posix() != live_canonical_raw:
        fail("fixed canonical interpreter is not canonical")
    require_executable(live_canonical, "fixed canonical interpreter")
    if digest(live_canonical) != fixed_sha256:
        fail("fixed interpreter SHA256 changed")
    try:
        if pathlib.Path(live_requested).resolve(strict=True) != live_canonical:
            fail("fixed requested interpreter resolves to a different executable")
        if pathlib.Path(live_path_python).resolve(strict=True) != live_canonical:
            fail("PATH python resolves to a different executable")
    except FileNotFoundError:
        fail("fixed interpreter path cannot be resolved")
    if digest(pathlib.Path(live_path_python)) != path_sha256:
        fail("PATH python SHA256 changed")

rows = sidecar_payload["configs"]
if not isinstance(rows, list) or len(rows) != len(config_paths):
    fail("interpreter binding sidecar must contain exactly four configs")
row_keys = {
    "source_relative_path",
    "artifact_relative_path",
    "size_bytes",
    "sha256",
    "python_path",
    "python_canonical_path",
    "python_sha256",
}
archive_paths = []
for row, source_relative in zip(rows, config_paths):
    if not isinstance(row, dict) or set(row) != row_keys:
        fail("interpreter binding config entry keys are not exact")
    if row["source_relative_path"] != source_relative:
        fail("interpreter binding config order or source path is invalid")
    archive_relative = "provenance/interpreter_configs/{}.global_config.json".format(
        pathlib.PurePosixPath(source_relative).parts[0]
    )
    if row["artifact_relative_path"] != archive_relative:
        fail("interpreter binding archive path is invalid: {}".format(source_relative))
    archive_paths.append(archive_relative)
    if (
        isinstance(row["size_bytes"], bool)
        or not isinstance(row["size_bytes"], int)
        or row["size_bytes"] <= 0
    ):
        fail("interpreter binding config size is invalid: {}".format(source_relative))
    require_hash(row["sha256"], "interpreter config SHA256")
    safe_absolute(row["python_path"], "nested python_path")
    if row["python_path"] != requested:
        fail("nested python_path differs from fixed requested path: {}".format(source_relative))
    require_canonical_absolute(row["python_canonical_path"], "nested canonical Python path")
    if row["python_canonical_path"] != fixed_canonical:
        fail("nested canonical Python path differs from fixed interpreter: {}".format(source_relative))
    require_hash(row["python_sha256"], "nested Python SHA256")
    if row["python_sha256"] != fixed_sha256:
        fail("nested Python SHA256 differs from fixed interpreter: {}".format(source_relative))
    archive = root / pathlib.PurePosixPath(archive_relative)
    require_regular(archive, "archived interpreter config")
    archive_bytes = archive.read_bytes()
    if len(archive_bytes) != row["size_bytes"]:
        fail("archived interpreter config size mismatch: {}".format(source_relative))
    actual_archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if actual_archive_sha256 != row["sha256"]:
        fail("archived interpreter config SHA256 mismatch: {}".format(source_relative))
    try:
        config_payload = json.loads(
            archive_bytes.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs
        )
    except Exception as exc:
        fail("archived interpreter config is invalid JSON: {} ({})".format(source_relative, exc))
    if not isinstance(config_payload, dict):
        fail("archived interpreter config must be a JSON object: {}".format(source_relative))
    if config_payload.get("python_path") != requested:
        fail("archived interpreter python_path differs from fixed path: {}".format(source_relative))

provenance = load_object(root / "provenance.json", "Task2 provenance")
if provenance.get("schema_version") != "sc26-ae-echo-provenance-v1":
    fail("unexpected Task2 provenance schema")
if provenance.get("execution_mode") != "real":
    fail("Task2 provenance execution_mode is not real")
provenance_evidence = provenance.get("execution_evidence")
if provenance_evidence not in {pending_evidence, qualified_evidence}:
    fail("Task2 provenance execution evidence is not a real-chain state")
if manifest_evidence == pending_evidence and provenance_evidence != pending_evidence:
    fail("pending Task2 manifest must retain pending provenance evidence")
if manifest_evidence == qualified_evidence and provenance_evidence not in {
    pending_evidence,
    qualified_evidence,
}:
    fail("qualified Task2 manifest has invalid provenance evidence")
if provenance.get("run_command") != "bash run_all.sh":
    fail("Task2 provenance run_command is not the pinned Echo command")
if provenance.get("update_command") != requested + " update_configs.py":
    fail("Task2 provenance update_command is not the fixed interpreter command")
if provenance.get("automatic_fallback") is not False:
    fail("Task2 provenance enables automatic fallback")
expected_binding = {
    "status": "bound",
    "artifact_path": "interpreter_binding.json",
    "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
}
if provenance.get("interpreter_binding") != expected_binding:
    fail("Task2 provenance does not reference the current interpreter sidecar")

entries = manifest.get("files")
if not isinstance(entries, list):
    fail("Task2 artifact manifest files is not an array")
entry_map = {}
for entry in entries:
    if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
        fail("Task2 artifact manifest contains an invalid file entry")
    path = entry["path"]
    if path in entry_map:
        fail("Task2 artifact manifest contains duplicate file entries")
    entry_map[path] = entry
required_paths = ["interpreter_binding.json", "provenance.json"] + archive_paths
for relative in required_paths:
    if relative not in entry_map:
        fail("Task2 artifact manifest omits interpreter binding artifact: {}".format(relative))
    artifact = root / pathlib.PurePosixPath(relative)
    require_regular(artifact, "manifest interpreter artifact")
    entry = entry_map[relative]
    if (
        isinstance(entry["size_bytes"], bool)
        or not isinstance(entry["size_bytes"], int)
        or entry["size_bytes"] < 0
    ):
        fail("Task2 artifact manifest size is invalid: {}".format(relative))
    require_hash(entry["sha256"], "manifest artifact SHA256")
    if entry["size_bytes"] != artifact.stat().st_size:
        fail("Task2 artifact manifest size mismatch: {}".format(relative))
    if entry["sha256"] != digest(artifact):
        fail("Task2 artifact manifest SHA256 mismatch: {}".format(relative))
if manifest_evidence != provenance_evidence and not (
    manifest_evidence == qualified_evidence and provenance_evidence == pending_evidence
):
    fail("Task2 manifest and provenance execution evidence differ")
PY
    local rc=$?
    [[ "$rc" == 0 ]] || {
        task2_error "Task2 interpreter binding sidecar verification failed"
        return 1
    }
}

task2_safe_id() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || task2_error "unsafe predictor_run_id: $1"
}

task2_validate_model() {
    case "$TASK2_MODEL_KEY" in
        gpt175b|qwen3_a30b|dsv3) return 0 ;;
        *) task2_error "model must be one of gpt175b|qwen3_a30b|dsv3"; return 1 ;;
    esac
}

task2_validate_execution_controls() {
    [[ "$TASK2_MODE" == real || "$TASK2_MODE" == synthetic ]] || {
        task2_error "TASK2_EXECUTION_MODE must be real or synthetic"; return 1;
    }
    [[ "$TASK2_SKIP_UPDATE_CONFIGS" == 0 || "$TASK2_SKIP_UPDATE_CONFIGS" == 1 ]] || {
        task2_error "TASK2_SKIP_UPDATE_CONFIGS must be 0 or 1"; return 1;
    }
    [[ "$TASK2_SKIP_PREDICT" == 0 || "$TASK2_SKIP_PREDICT" == 1 ]] || {
        task2_error "TASK2_SKIP_PREDICT must be 0 or 1"; return 1;
    }
    if [[ "$TASK2_MODE" == real ]]; then
        [[ "$TASK2_SKIP_UPDATE_CONFIGS" == 0 ]] || {
            task2_error "TASK2_SKIP_UPDATE_CONFIGS must be 0 in real mode"; return 1;
        }
        [[ "$TASK2_SKIP_PREDICT" == 0 ]] || {
            task2_error "TASK2_SKIP_PREDICT must be 0 in real mode"; return 1;
        }
        [[ "$TASK2_UPDATE_COMMAND_OVERRIDDEN" == 0 ]] || {
            task2_error "TASK2_UPDATE_COMMAND overrides are forbidden in real mode"; return 1;
        }
        [[ "$TASK2_RUN_COMMAND_OVERRIDDEN" == 0 ]] || {
            task2_error "TASK2_RUN_COMMAND overrides are forbidden in real mode"; return 1;
        }
        [[ "$TASK2_META_PYTHON" == "$TASK2_FIXED_PYTHON" ]] || {
            task2_error "TASK2_META_PYTHON must equal the fixed real-mode interpreter: ${TASK2_FIXED_PYTHON}"; return 1;
        }
        [[ "$TASK2_PYTHON" == "$TASK2_FIXED_PYTHON" ]] || {
            task2_error "TASK2_PYTHON must equal the fixed real-mode interpreter: ${TASK2_FIXED_PYTHON}"; return 1;
        }
    fi
}

task2_validate_evidence_for_mode() {
    local observed=${1:-}
    case "${TASK2_MODE}" in
        real)
            [[ "${observed}" == "real_exact_two_h800_qualified" ]] || {
                task2_error \
                    "real mode requires real_exact_two_h800_qualified execution evidence; observed ${observed}"
                return 1
            }
            ;;
        synthetic)
            case "${observed}" in
                local_synthetic_not_two_gpu_qualification|real_exact_two_h800_qualified)
                    ;;
                *)
                    task2_error "synthetic mode received invalid execution evidence: ${observed}"
                    return 1
                    ;;
            esac
            ;;
        *)
            task2_error "cannot validate Task2 evidence for unsupported mode: ${TASK2_MODE}"
            return 1
            ;;
    esac
}

task2_validate_reuse_evidence() {
    local run_root=$1
    local evidence
    evidence=$(
        "$TASK2_META_PYTHON" - "$run_root/artifact_manifest.json" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])

def reject_duplicate_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key: {}".format(key))
        payload[key] = value
    return payload

payload = json.loads(
    path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs
)
if not isinstance(payload, dict):
    raise ValueError("Task2 artifact manifest must be a JSON object")
print(payload.get("execution_evidence", ""))
PY
    ) || {
        task2_error "cannot read Task2 execution evidence for reuse"
        return 1
    }
    task2_validate_evidence_for_mode "$evidence"
}

task2_validate_cuda_ids() {
    local ids=${CUDA_VISIBLE_DEVICES:-} first second extra
    [[ -n "$ids" ]] || { task2_error "CUDA_VISIBLE_DEVICES must name exactly two GPUs"; return 1; }
    IFS=',' read -r first second extra <<< "$ids"
    [[ -n "$first" && -n "$second" && -z "${extra:-}" ]] || {
        task2_error "CUDA_VISIBLE_DEVICES must contain exactly two comma-separated IDs"; return 1;
    }
    [[ "$first" != "$second" ]] || { task2_error "CUDA_VISIBLE_DEVICES IDs must be distinct"; return 1; }
    [[ "$first" =~ ^[0-9]+$ && "$second" =~ ^[0-9]+$ ]] || {
        task2_error "CUDA_VISIBLE_DEVICES IDs must be decimal GPU indices"; return 1;
    }
    if [[ "$TASK2_MODE" == real && "$TASK2_SKIP_HARDWARE_CHECK" == 1 ]]; then
        task2_error "hardware checks may only be skipped in explicit synthetic mode"; return 1
    fi
    if [[ "$TASK2_MODE" != synthetic && "$TASK2_SKIP_HARDWARE_CHECK" != 1 ]]; then
        task2_require_file "$TASK2_PYTHON" || return 1
        local visible_count
        visible_count=$("$TASK2_PYTHON" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null) || {
            task2_error "fixed Echo Python cannot query CUDA device count"; return 1;
        }
        [[ "$visible_count" == 2 ]] || {
            task2_error "exactly two visible CUDA devices are required; observed ${visible_count}"; return 1;
        }
    fi
}

task2_prepare_failure_root() {
    mkdir -p "$TASK2_OUTPUT_ROOT/_work" || return 1
    TASK2_FAILURE_ROOT="${TASK2_OUTPUT_ROOT}/_work/task2-failure-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
}

task2_new_id() {
    if [[ -n "$TASK2_PREDICTOR_RUN_ID" ]]; then
        task2_safe_id "$TASK2_PREDICTOR_RUN_ID" || return 1
        return 0
    fi
    TASK2_PREDICTOR_RUN_ID="task2-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
    task2_safe_id "$TASK2_PREDICTOR_RUN_ID"
}

task2_gitlink_commit() {
    local expected source_head
    source_head=$(git -C "$TASK2_SOURCE_REPO" rev-parse HEAD 2>/dev/null) || {
        task2_error "cannot resolve Echo-slowdown source HEAD"; return 1;
    }
    if [[ "$TASK2_MODE" == synthetic && -n "$TASK2_EXPECTED_COMMIT" ]]; then
        expected=$TASK2_EXPECTED_COMMIT
        TASK2_COMMIT_PROVENANCE="explicit_test_gitlink_override"
    else
        [[ "$TASK2_MAIN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
            task2_error "main-repository commit is unavailable for Echo-slowdown gitlink validation"; return 1;
        }
        expected=$(git -C "$TASK2_REPO_ROOT" rev-parse "${TASK2_MAIN_COMMIT}:Echo-slowdown" 2>/dev/null) || {
            task2_error "cannot resolve main-repository Echo-slowdown gitlink commit"; return 1;
        }
        TASK2_COMMIT_PROVENANCE="main_repository_gitlink"
    fi
    [[ "$expected" == "$source_head" ]] || {
        task2_error "Echo source HEAD ${source_head} does not match expected gitlink ${expected}"; return 1;
    }
    TASK2_ECHO_COMMIT=$expected
}

task2_assert_source_clean() {
    local status
    status=$(git -C "$TASK2_SOURCE_REPO" status --porcelain 2>/dev/null) || {
        task2_error "cannot inspect Echo-slowdown status"; return 1;
    }
    [[ -z "$status" ]] || {
        task2_error "Echo-slowdown checkout is dirty: ${status}"; return 1;
    }
}

task2_assert_outer_source_provenance() {
    local expected_commit=$1
    local relative expected_blob actual_blob object_type working_path
    # Keep this list synchronized with tests/unit/test_sc26_ae_task2_source_provenance.sh.
    local -a source_files=(
        SC26-AE/task2_gpt175b.sh
        SC26-AE/task2_dsv3.sh
        SC26-AE/task2_qwen3_a30b.sh
        SC26-AE/lib/common.sh
        SC26-AE/lib/task2_echo.sh
        SC26-AE/tools/artifact_manifest.py
        SC26-AE/tools/echo_metrics.py
    )

    [[ -d "${TASK2_REPO_ROOT}/.git" || -f "${TASK2_REPO_ROOT}/.git" ]] || {
        task2_error "Task2 outer source provenance root is not a Git repository: ${TASK2_REPO_ROOT}"
        return 1
    }
    [[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || {
        task2_error "Task2 outer source provenance commit is invalid: ${expected_commit}"
        return 1
    }

    for relative in "${source_files[@]}"; do
        working_path="${TASK2_REPO_ROOT}/${relative}"
        [[ -f "${working_path}" && ! -L "${working_path}" ]] || {
            task2_error "Task2 load-bearing outer source is not a regular file: ${relative}"
            return 1
        }
        expected_blob=$(git -C "${TASK2_REPO_ROOT}" rev-parse \
            "${expected_commit}:${relative}" 2>/dev/null) || {
            task2_error "Task2 outer source is not tracked by pinned HEAD: ${relative}"
            return 1
        }
        object_type=$(git -C "${TASK2_REPO_ROOT}" cat-file -t "${expected_blob}" 2>/dev/null) || {
            task2_error "Cannot inspect pinned Task2 outer source object: ${relative}"
            return 1
        }
        [[ "${object_type}" == blob ]] || {
            task2_error "Pinned Task2 outer source is not a file: ${relative}"
            return 1
        }
        actual_blob=$(git -C "${TASK2_REPO_ROOT}" hash-object --no-filters \
            "${working_path}" 2>/dev/null) || {
            task2_error "Cannot hash Task2 working-tree outer source: ${relative}"
            return 1
        }
        [[ "${actual_blob}" == "${expected_blob}" ]] || {
            task2_error "Task2 tracked outer blob mismatch: ${relative}"
            return 1
        }
    done
}

task2_maybe_assert_outer_source_provenance() {
    if [[ "${TASK2_SKIP_OUTER_SOURCE_PROVENANCE:-0}" == "1" ]]; then
        printf '[WARN] Task2 outer source provenance hash gate bypassed explicitly for runtime smoke; output remains non-qualified.\n' >&2
        return 0
    fi
    task2_assert_outer_source_provenance "$1"
}

task2_is_excluded() {
    local path=$1 prefix
    for prefix in "${TASK2_EXCLUDED_PREFIXES[@]}"; do
        if [[ "$prefix" == */ ]]; then
            [[ "$path" == "${prefix}"* ]] && return 0
        else
            [[ "$path" == "$prefix" ]] && return 0
        fi
    done
    return 1
}

task2_is_known_historical() {
    local path=$1 known
    for known in "${TASK2_HISTORICAL_EXCLUDED_PATHS[@]}"; do
        [[ "$path" == "$known" ]] && return 0
    done
    return 1
}

task2_source_inventory() {
    local tracked_file=$1 included_file=$2 path
    git -C "$TASK2_SOURCE_REPO" ls-tree -r --name-only "$TASK2_ECHO_COMMIT" >"$tracked_file" 2>/dev/null || {
        task2_error "cannot enumerate pinned Echo tracked source"; return 1;
    }
    : >"$included_file" || return 1
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        if task2_is_excluded "$path"; then
            task2_is_known_historical "$path" || {
                task2_error "new tracked path under excluded output contract: ${path}"; return 1;
            }
            continue
        fi
        printf '%s\n' "$path" >>"$included_file"
    done <"$tracked_file"
}

task2_extract_snapshot() {
    local included_file=$1 path prefix excludes=()
    mkdir -p "$TASK2_SOURCE_ROOT" || { task2_error "cannot create filtered Echo snapshot"; return 1; }
    for prefix in "${TASK2_EXCLUDED_PREFIXES[@]}"; do
        excludes+=(":(exclude)${prefix}")
    done
    # Exclusions happen at archive creation time; no post-extraction deletion
    # is permitted by the Task2 source contract.
    git -C "$TASK2_SOURCE_REPO" archive --format=tar "$TASK2_ECHO_COMMIT" -- . "${excludes[@]}" \
        | tar -xf - -C "$TASK2_SOURCE_ROOT" || {
        task2_error "filtered git archive extraction failed"; return 1;
    }
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        [[ -e "$TASK2_SOURCE_ROOT/$path" ]] || {
            task2_error "tracked source missing from filtered snapshot: ${path}"; return 1;
        }
    done <"$included_file"
    for prefix in "${TASK2_EXCLUDED_PREFIXES[@]}"; do
        [[ ! -e "$TASK2_SOURCE_ROOT/$prefix" ]] || {
            task2_error "excluded generated/runtime path leaked into filtered snapshot: ${prefix}"; return 1;
        }
    done
}

task2_write_source_manifest() {
    local included_file=$1 output_file=$2
    "$TASK2_META_PYTHON" - "$TASK2_SOURCE_ROOT" "$TASK2_ECHO_COMMIT" "$TASK2_COMMIT_PROVENANCE" \
        "$included_file" "$output_file" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
included = pathlib.Path(sys.argv[4])
rows = []
for raw in included.read_text(encoding="utf-8").splitlines():
    path = raw.strip()
    if not path:
        continue
    candidate = root / path
    if not candidate.is_file():
        raise SystemExit(f"missing snapshot file: {path}")
    rows.append({"path": path, "bytes": candidate.stat().st_size, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()})
payload = {
    "schema_version": "sc26-ae-echo-source-manifest-v1",
    "echo_commit": sys.argv[2],
    "commit_provenance": sys.argv[3],
    "filtered_snapshot": True,
    "excluded_historical_paths": [
        "merge/input/kernel_metric_output.csv",
        "merge/input/slowdown_stats_output_device_0.xlsx",
        "merge/output/merged_features.csv",
        "training_testing/input/test_csv/merged_features.csv",
        "training_testing/input/train_csv/merged_features.csv",
        "training_testing/output/prediction/feature_importance_merged_features.png",
        "training_testing/output/prediction/output_df_merged_features.csv",
        "training_testing/output/prediction/output_full_df_merged_features.csv",
        "training_testing/output/prediction/output_metrics.txt",
        "training_testing/output/train_dataset.csv",
        "training_testing/output/xgb_model.json",
    ],
    "tracked_source_count": len(rows),
    "files": rows,
}
output = pathlib.Path(sys.argv[5])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

task2_write_provenance() {
    local output_file=$1
    "$TASK2_META_PYTHON" - "$output_file" "$TASK2_MODEL_KEY" "$TASK2_PREDICTOR_RUN_ID" "$TASK2_ECHO_COMMIT" \
        "$TASK2_COMMIT_PROVENANCE" "$TASK2_MODE" "${CUDA_VISIBLE_DEVICES:-}" "$TASK2_REBUILD" \
        "$TASK2_SOURCE_ROOT" "$TASK2_RUN_COMMAND" "$TASK2_UPDATE_COMMAND" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

output = pathlib.Path(sys.argv[1])
mode = sys.argv[6]
payload = {
    "schema_version": "sc26-ae-echo-provenance-v1",
    "model": sys.argv[2],
    "predictor_run_id": sys.argv[3],
    "echo_commit": sys.argv[4],
    "commit_provenance": sys.argv[5],
    "execution_mode": mode,
    "execution_evidence": (
        "local_synthetic_not_two_gpu_qualification"
        if mode == "synthetic"
        else "runtime_measurement_requires_external_two_gpu_qualification"
    ),
    "cuda_visible_devices": sys.argv[7],
    "rebuild": sys.argv[8] == "1",
    "filtered_snapshot_root": pathlib.Path(sys.argv[9]).name,
    "run_command": sys.argv[10],
    "update_command": sys.argv[11],
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "historical_outputs_excluded_at_archive_time": True,
    "automatic_fallback": False,
    "outer_source_provenance": {
        "checked": os.environ.get("TASK2_SKIP_OUTER_SOURCE_PROVENANCE", "0") == "0",
        "bypassed": os.environ.get("TASK2_SKIP_OUTER_SOURCE_PROVENANCE", "0") == "1",
        "reason": (
            "explicit_runtime_smoke_bypass_dirty_worktree"
            if os.environ.get("TASK2_SKIP_OUTER_SOURCE_PROVENANCE", "0") == "1"
            else None
        ),
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

task2_write_snapshot_precheck() {
    local output_file=$1
    "$TASK2_META_PYTHON" - "$output_file" "$TASK2_SOURCE_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[2])
excluded = [
    "merge/input/kernel_metric_output.csv",
    "merge/input/slowdown_stats_output_device_0.xlsx",
    "merge/output/",
    "training_testing/input/test_csv/",
    "training_testing/input/train_csv/",
    "training_testing/output/prediction/",
    "training_testing/output/train_dataset.csv",
    "training_testing/output/xgb_model.json",
]
present = [rel for rel in excluded if (root / rel).exists()]
if present:
    raise SystemExit(f"excluded paths were present before Echo execution: {present}")
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema_version": "sc26-ae-echo-snapshot-precheck-v1",
    "excluded_paths_present_before_run": present,
    "pre_execution_clean": not present,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

task2_copy_canonical() {
    local destination=$1 relative source_path
    local required=(
        "training_testing/output/train_dataset.csv"
        "training_testing/output/xgb_model.json"
        "training_testing/output/standard_scaler.json"
        "merge/input/kernel_metric_output.csv"
    )
    mkdir -p "$destination/logs" "$destination/training_testing/output" "$destination/merge/input" || return 1
    for relative in "${required[@]}"; do
        source_path="$TASK2_SOURCE_ROOT/$relative"
        [[ -f "$source_path" ]] || { task2_error "Echo run did not produce required artifact: ${relative}"; return 1; }
        [[ -s "$source_path" ]] || { task2_error "Echo artifact is empty: ${relative}"; return 1; }
        mkdir -p "$destination/$(dirname "$relative")" || return 1
        cp "$source_path" "$destination/$relative" || return 1
    done
    cp "$TASK2_WORK_ROOT/logs/run_all.log" "$destination/logs/run_all.log" || return 1
    cp "$TASK2_WORK_ROOT/logs/run_timing.json" "$destination/logs/run_timing.json" || return 1
}

task2_write_artifact_manifest() {
    local run_root=$1
    local metadata_file="${TASK2_WORK_ROOT}/artifact_manifest_metadata.json"
    local file_list="${TASK2_WORK_ROOT}/artifact_manifest_files.txt"
    local megatron_commit sim_engine_commit

    megatron_commit=$TASK2_MAIN_COMMIT
    [[ "$megatron_commit" =~ ^[0-9a-f]{40}$ ]] || {
        task2_error "recorded Megatron-LM source commit is invalid for artifact manifest"; return 1;
    }
    sim_engine_commit=$(git -C "$TASK2_REPO_ROOT" rev-parse \
        "${TASK2_MAIN_COMMIT}:megatron-sim-engine" 2>/dev/null) || {
        task2_error "cannot resolve megatron-sim-engine source commit for artifact manifest"; return 1;
    }
    "$TASK2_META_PYTHON" - "$metadata_file" "$megatron_commit" "$TASK2_ECHO_COMMIT" \
        "$sim_engine_commit" "$TASK2_PREDICTOR_RUN_ID" "$TASK2_MODE" "$TASK2_MODEL_KEY" <<'PY'
import json
import pathlib
import sys

metadata = {
    "schema_version": "sc26-ae-artifact-manifest-v1",
    "model": "shared_task2",
    "task": "task2",
    "artifact_source": "fresh",
    "source_commits": {
        "megatron_lm": sys.argv[2],
        "echo_slowdown": sys.argv[3],
        "megatron_sim_engine": sys.argv[4],
    },
    "predictor_run_id": sys.argv[5],
    "execution_evidence": (
        "local_synthetic_not_two_gpu_qualification"
        if sys.argv[6] == "synthetic"
        else "runtime_measurement_requires_external_two_gpu_qualification"
    ),
    "model_marker": sys.argv[7],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    [[ -s "$metadata_file" ]] || { task2_error "artifact manifest metadata was not created"; return 1; }

    find "$run_root" -type f -printf '%P\n' | LC_ALL=C sort >"$file_list" || {
        task2_error "cannot enumerate Task2 artifact files"; return 1;
    }
    sed -i '/^artifact_manifest\.json$/d' "$file_list" || {
        task2_error "cannot exclude artifact_manifest.json from file list"; return 1;
    }
    [[ -s "$file_list" ]] || { task2_error "Task2 artifact file list is empty"; return 1; }

    "$TASK2_META_PYTHON" -B "$TASK2_ARTIFACT_TOOL" create \
        --root "$run_root" \
        --metadata-json "$metadata_file" \
        --file-list "$file_list" \
        --output "$run_root/artifact_manifest.json" || {
        task2_error "canonical Task2 artifact manifest creation failed"; return 1;
    }
}

task2_verify_run() {
    local run_root=$1
    task2_validate_interpreter_chain || return 1
    [[ -d "$run_root" ]] || { task2_error "predictor run directory is missing: ${run_root}"; return 1; }
    [[ -f "$run_root/artifact_manifest.json" ]] || { task2_error "predictor artifact manifest is missing"; return 1; }
    "$TASK2_META_PYTHON" -B "$TASK2_ARTIFACT_TOOL" verify \
        --root "$run_root" \
        --manifest "$run_root/artifact_manifest.json" || {
        task2_error "canonical Task2 artifact manifest verification failed"; return 1;
    }
    task2_validate_binding_sidecar "$run_root" || return 1
    local megatron_commit sim_engine_commit
    megatron_commit=$TASK2_MAIN_COMMIT
    [[ "$megatron_commit" =~ ^[0-9a-f]{40}$ ]] || {
        task2_error "recorded Megatron-LM commit is invalid while verifying Task2 bundle"; return 1;
    }
    sim_engine_commit=$(git -C "$TASK2_REPO_ROOT" rev-parse \
        "${TASK2_MAIN_COMMIT}:megatron-sim-engine" 2>/dev/null) || {
        task2_error "cannot resolve current megatron-sim-engine commit while verifying Task2 bundle"; return 1;
    }
    "$TASK2_META_PYTHON" - "$run_root/artifact_manifest.json" "$run_root" "$TASK2_ECHO_COMMIT" \
        "$megatron_commit" "$sim_engine_commit" "$TASK2_PREDICTOR_RUN_ID" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != "sc26-ae-artifact-manifest-v1":
    raise SystemExit("unexpected canonical artifact manifest schema")
if manifest.get("model") != "shared_task2" or manifest.get("task") != "task2":
    raise SystemExit("Task2 artifact manifest model/task identity is invalid")
if manifest.get("artifact_source") != "fresh":
    raise SystemExit("Task2 artifact manifest must describe a fresh bundle")
if "capture_id" in manifest:
    raise SystemExit("Task2 artifact manifest must omit capture_id")
if manifest.get("predictor_run_id") != sys.argv[6]:
    raise SystemExit("Task2 artifact manifest predictor_run_id mismatch")
expected_commits = {
    "megatron_lm": sys.argv[4],
    "echo_slowdown": sys.argv[3],
    "megatron_sim_engine": sys.argv[5],
}
if manifest.get("source_commits") != expected_commits:
    raise SystemExit("Task2 artifact manifest source commits differ from current sources")
if manifest.get("execution_evidence") not in {
    "local_synthetic_not_two_gpu_qualification",
    "runtime_measurement_requires_external_two_gpu_qualification",
    # Canonical verification is shared by fresh production and qualified
    # reuse.  The mode-specific reuse predicate below still decides whether
    # this state may be consumed in the current execution mode.
    "real_exact_two_h800_qualified",
}:
    raise SystemExit("Task2 artifact manifest execution evidence is invalid")

root = pathlib.Path(sys.argv[2])
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
if metrics.get("predictor_run_id") != manifest["predictor_run_id"]:
    raise SystemExit("metrics predictor_run_id does not match artifact manifest")
precheck = json.loads((root / "snapshot_precheck.json").read_text(encoding="utf-8"))
if precheck.get("schema_version") != "sc26-ae-echo-snapshot-precheck-v1" or precheck.get("pre_execution_clean") is not True:
    raise SystemExit("snapshot pre-execution exclusion check did not pass")
PY
    if [[ "$?" != 0 ]]; then
        task2_error "Task2 canonical manifest metadata validation failed"; return 1
    fi
    "$TASK2_META_PYTHON" - "$run_root/provenance.json" "$TASK2_ECHO_COMMIT" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema_version") != "sc26-ae-echo-provenance-v1":
    raise SystemExit("unexpected Task2 provenance schema")
if payload.get("echo_commit") != sys.argv[2]:
    raise SystemExit("Task2 provenance Echo commit differs from current gitlink")
if payload.get("automatic_fallback") is not False:
    raise SystemExit("Task2 provenance must state automatic fallback is disabled")
PY
    if [[ "$?" != 0 ]]; then
        task2_error "Task2 provenance validation failed"; return 1
    fi
    "$TASK2_META_PYTHON" - "$run_root/metrics.json" "$TASK2_PREDICTOR_RUN_ID" <<'PY'
import json
import math
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema_version") != "sc26-ae-echo-metrics-v1":
    raise SystemExit("unexpected Task2 metrics schema")
if payload.get("predictor_run_id") != sys.argv[2]:
    raise SystemExit("Task2 metrics predictor_run_id mismatch")
elapsed = float(payload.get("task2_run_all_elapsed_seconds"))
if not math.isfinite(elapsed) or elapsed <= 0:
    raise SystemExit("Task2 elapsed metric is not positive finite")
folds = payload.get("validation_mse_by_fold")
if not isinstance(folds, list) or len(folds) != 5 or any(not math.isfinite(float(x)) or float(x) < 0 for x in folds):
    raise SystemExit("Task2 validation folds are invalid")
average = float(payload.get("average_validation_mse"))
if abs(average - sum(float(x) for x in folds) / 5) > 1e-12:
    raise SystemExit("Task2 average validation MSE is inconsistent")
test_mse = float(payload.get("test_mse"))
if not math.isfinite(test_mse) or test_mse < 0:
    raise SystemExit("Task2 test MSE is invalid")
if float(payload.get("model_reload_max_abs_prediction_delta")) > 1e-12:
    raise SystemExit("Task2 model reload delta exceeds contract")
counts = [int(payload.get(name)) for name in (
    "scaler_feature_count", "scaler_mean_count", "scaler_scale_count", "scaler_nonzero_scale_count")]
if counts[0] <= 0 or counts[:3] != [counts[0]] * 3 or counts[3] != counts[0]:
    raise SystemExit("Task2 scaler counts are inconsistent")
sample = payload.get("prediction_sample")
if not isinstance(sample, dict) or any(not math.isfinite(float(value)) for value in sample.values()):
    raise SystemExit("Task2 prediction sample is invalid")
if float(sample.get("predicted_slowdown_factor_clipped")) < 0:
    raise SystemExit("Task2 clipped slowdown is negative")
PY
    if [[ "$?" != 0 ]]; then
        task2_error "Task2 metrics validation failed"; return 1
    fi
    TASK2_VERIFIED_MANIFEST_SHA256=$(sha256sum "$run_root/artifact_manifest.json" | awk '{print $1}') || return 1
}

task2_write_marker() {
    local marker_path=$1 run_root=$2 manifest_digest=$3
    if [[ "$TASK2_MODE" == real ]]; then
        task2_maybe_assert_outer_source_provenance "$TASK2_MAIN_COMMIT" || return 1
    fi
    mkdir -p "$(dirname "$marker_path")" || return 1
    "$TASK2_META_PYTHON" - "$marker_path" "$TASK2_MODEL_KEY" "$TASK2_PREDICTOR_RUN_ID" \
        "$manifest_digest" "$TASK2_OUTPUT_ROOT" "$run_root" "$TASK2_MODE" <<'PY'
import hashlib
import json
import pathlib
import sys

marker = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[5]).resolve()
run_root = pathlib.Path(sys.argv[6]).resolve()
provenance = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
execution_evidence = provenance.get("execution_evidence")
if execution_evidence not in {
    "local_synthetic_not_two_gpu_qualification",
    "runtime_measurement_requires_external_two_gpu_qualification",
}:
    raise SystemExit("unexpected Task2 execution evidence class")
relative = run_root.relative_to(output_root).as_posix()
payload = {
    "schema_version": "sc26-ae-task2-marker-v1",
    "task": "task2",
    "model": sys.argv[2],
    "predictor_run_id": sys.argv[3],
    "run_path": relative,
    "run_relative_path": relative,
    "manifest_sha256": sys.argv[4],
    "artifact_manifest_sha256": sys.argv[4],
    "source_commit": provenance.get("echo_commit"),
    "metrics_sha256": hashlib.sha256((run_root / "metrics.json").read_bytes()).hexdigest(),
    "execution_evidence": execution_evidence,
    "verified": True,
}
marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

task2_write_shared_pointer() {
    local pointer=$1 run_root=$2 manifest_digest=$3
    if [[ "$TASK2_MODE" == real ]]; then
        task2_maybe_assert_outer_source_provenance "$TASK2_MAIN_COMMIT" || return 1
    fi
    mkdir -p "$(dirname "$pointer")" || return 1
    "$TASK2_META_PYTHON" - "$pointer" "$TASK2_OUTPUT_ROOT" "$run_root" "$TASK2_PREDICTOR_RUN_ID" "$manifest_digest" <<'PY'
import json
import pathlib
import sys

pointer = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[2]).resolve()
run_root = pathlib.Path(sys.argv[3]).resolve()
payload = {
    "schema_version": "sc26-ae-task2-shared-pointer-v1",
    "predictor_run_id": sys.argv[4],
    "run_path": run_root.relative_to(output_root).as_posix(),
    "run_relative_path": run_root.relative_to(output_root).as_posix(),
    "manifest_sha256": sys.argv[5],
    "artifact_manifest_sha256": sys.argv[5],
    "verified": True,
}
pointer.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

task2_run_reuse() {
    task2_validate_interpreter_chain || return 1
    local marker="${TASK2_OUTPUT_ROOT}/${TASK2_MODEL_KEY}/task2/predictor_marker.json"
    if [[ -f "$marker" ]]; then
        local resolved
        resolved=$("$TASK2_META_PYTHON" - "$marker" "$TASK2_OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema_version") != "sc26-ae-task2-marker-v1" or payload.get("verified") is not True:
    raise SystemExit("existing Task2 marker is not verified")
if payload.get("model") != pathlib.Path(sys.argv[1]).parts[-3]:
    raise SystemExit("existing Task2 marker model path is inconsistent")
run_path = payload.get("run_path")
run_relative_path = payload.get("run_relative_path")
if (
    not isinstance(run_path, str)
    or not run_path
    or not isinstance(run_relative_path, str)
    or not run_relative_path
    or run_path != run_relative_path
):
    raise SystemExit("existing Task2 marker path aliases are missing or differ")
predictor_run_id = payload.get("predictor_run_id")
rel = pathlib.PurePosixPath(run_path)
if rel.is_absolute() or ".." in rel.parts:
    raise SystemExit("existing Task2 marker path is unsafe")
output_root = pathlib.Path(sys.argv[2]).resolve()
run_root = (output_root / rel).resolve()
try:
    run_root.relative_to(output_root)
except ValueError:
    raise SystemExit("existing Task2 marker path escapes output root")
if predictor_run_id != run_root.name:
    raise SystemExit("existing Task2 marker predictor_run_id does not match run path")
if not isinstance(predictor_run_id, str) or not predictor_run_id or rel != (
    pathlib.PurePosixPath("_shared/task2/runs") / predictor_run_id
):
    raise SystemExit("existing Task2 marker path is not canonical for predictor_run_id")
manifest = run_root / "artifact_manifest.json"
metrics = run_root / "metrics.json"
if not manifest.is_file() or not metrics.is_file():
    raise SystemExit("existing Task2 marker points to an incomplete run")
import hashlib
manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
metrics_digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
if payload.get("manifest_sha256") != manifest_digest or payload.get("artifact_manifest_sha256") != manifest_digest:
    raise SystemExit("existing Task2 marker manifest checksum is stale")
if payload.get("metrics_sha256") != metrics_digest:
    raise SystemExit("existing Task2 marker metrics checksum is stale")
provenance = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
if payload.get("source_commit") != provenance.get("echo_commit"):
    raise SystemExit("existing Task2 marker source commit is stale")
print(run_root.resolve())
PY
) || { task2_error "existing Task2 marker is corrupt"; return 1; }
        TASK2_RUN_ROOT=$resolved
        TASK2_PREDICTOR_RUN_ID=$(basename "$TASK2_RUN_ROOT")
        task2_validate_reuse_evidence "$TASK2_RUN_ROOT" || return 1
        task2_verify_run "$TASK2_RUN_ROOT" || return 1
        printf '[INFO] Reusing verified Task2 predictor_run_id=%s\n' "$TASK2_PREDICTOR_RUN_ID"
        return 0
    fi
    local pointer="${TASK2_OUTPUT_ROOT}/_shared/task2/predictor_marker.json"
    [[ -f "$pointer" ]] || {
        task2_error "no verified shared Task2 predictor exists; build once with REBUILD=1"; return 1;
    }
    local resolved pointer_id pointer_digest pointer_artifact_digest
    resolved=$("$TASK2_META_PYTHON" - "$pointer" "$TASK2_OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema_version") != "sc26-ae-task2-shared-pointer-v1":
    raise SystemExit("unexpected shared Task2 pointer schema")
if payload.get("verified") is not True:
    raise SystemExit("shared Task2 pointer is not verified")
run_path = payload.get("run_path")
run_relative_path = payload.get("run_relative_path")
if (
    not isinstance(run_path, str)
    or not run_path
    or not isinstance(run_relative_path, str)
    or not run_relative_path
    or run_path != run_relative_path
):
    raise SystemExit("shared Task2 pointer path aliases are missing or differ")
predictor_run_id = payload.get("predictor_run_id")
rel = pathlib.PurePosixPath(run_path)
if rel.is_absolute() or ".." in rel.parts:
    raise SystemExit("shared Task2 pointer path is unsafe")
if not isinstance(predictor_run_id, str) or not predictor_run_id or rel != (
    pathlib.PurePosixPath("_shared/task2/runs") / predictor_run_id
):
    raise SystemExit("shared Task2 pointer path is not canonical for predictor_run_id")
output_root = pathlib.Path(sys.argv[2]).resolve()
run_root = (output_root / rel).resolve()
try:
    run_root.relative_to(output_root)
except ValueError:
    raise SystemExit("existing Task2 shared pointer path escapes output root")
if not run_root.is_dir():
    raise SystemExit("shared Task2 pointer run directory is missing")
print(run_root)
print(payload.get("predictor_run_id", ""))
print(payload.get("manifest_sha256", ""))
print(payload.get("artifact_manifest_sha256", ""))
PY
) || { task2_error "shared Task2 pointer is corrupt"; return 1; }
    TASK2_RUN_ROOT=$(printf '%s\n' "$resolved" | sed -n '1p')
    pointer_id=$(printf '%s\n' "$resolved" | sed -n '2p')
    pointer_digest=$(printf '%s\n' "$resolved" | sed -n '3p')
    pointer_artifact_digest=$(printf '%s\n' "$resolved" | sed -n '4p')
    TASK2_PREDICTOR_RUN_ID=$(basename "$TASK2_RUN_ROOT")
    [[ "$TASK2_PREDICTOR_RUN_ID" == "$pointer_id" ]] || {
        task2_error "shared pointer predictor_run_id does not match run path"; return 1;
    }
    task2_validate_reuse_evidence "$TASK2_RUN_ROOT" || return 1
    task2_verify_run "$TASK2_RUN_ROOT" || return 1
    [[ "$TASK2_VERIFIED_MANIFEST_SHA256" == "$pointer_digest" \
        && "$TASK2_VERIFIED_MANIFEST_SHA256" == "$pointer_artifact_digest" ]] || {
        task2_error "shared pointer manifest checksum mismatch"; return 1;
    }
    task2_write_marker "$marker" "$TASK2_RUN_ROOT" "$TASK2_VERIFIED_MANIFEST_SHA256" || return 1
    printf '[INFO] Attached model %s to verified Task2 predictor_run_id=%s\n' "$TASK2_MODEL_KEY" "$TASK2_PREDICTOR_RUN_ID"
}

task2_run_build() {
    task2_new_id || return 1
    TASK2_WORK_ROOT="${TASK2_OUTPUT_ROOT}/_work/task2.${TASK2_PREDICTOR_RUN_ID}"
    TASK2_RUN_ROOT="${TASK2_OUTPUT_ROOT}/_shared/task2/runs/${TASK2_PREDICTOR_RUN_ID}"
    TASK2_SOURCE_ROOT="${TASK2_WORK_ROOT}/source"
    [[ ! -e "$TASK2_WORK_ROOT" && ! -e "$TASK2_RUN_ROOT" ]] || {
        task2_error "Task2 run identity already exists; choose a new PREDICTOR_RUN_ID"; return 1;
    }
    mkdir -p "$TASK2_WORK_ROOT/logs" "$TASK2_RUN_ROOT" || return 1
    local tracked_file="${TASK2_WORK_ROOT}/tracked_paths.txt"
    local included_file="${TASK2_WORK_ROOT}/included_paths.txt"
    local source_manifest="${TASK2_WORK_ROOT}/source_manifest.json"
    task2_assert_source_clean || return 1
    task2_gitlink_commit || return 1
    task2_source_inventory "$tracked_file" "$included_file" || return 1
    task2_extract_snapshot "$included_file" || return 1
    task2_write_source_manifest "$included_file" "$source_manifest" || return 1
    cp "$source_manifest" "$TASK2_RUN_ROOT/source_manifest.json" || return 1
    task2_write_provenance "$TASK2_RUN_ROOT/provenance.json" || return 1
    task2_require_file "$TASK2_SOURCE_ROOT/update_configs.py" || return 1
    task2_require_file "$TASK2_SOURCE_ROOT/run_all.sh" || return 1
    task2_write_snapshot_precheck "$TASK2_RUN_ROOT/snapshot_precheck.json" || return 1
    # Git does not preserve empty directories.  The pinned Echo run_all.sh
    # copies merged datasets into these runtime input directories, so the
    # isolated snapshot must create them explicitly before execution.
    mkdir -p \
        "$TASK2_SOURCE_ROOT/training_testing/input/test_csv" \
        "$TASK2_SOURCE_ROOT/training_testing/input/train_csv" || return 1

    task2_validate_interpreter_chain || return 1
    export CUDA_VISIBLE_DEVICES
    export PYTHONPATH="${TASK2_SOURCE_ROOT}:${PYTHONPATH:-}"
    local update_rc run_rc start_ns end_ns elapsed
    if [[ "$TASK2_MODE" == synthetic && "$TASK2_SKIP_UPDATE_CONFIGS" == 1 ]]; then
        printf '[INFO] Explicit synthetic mode: update_configs.py skipped by test contract\n' >>"$TASK2_WORK_ROOT/logs/run_all.log"
    elif [[ "$TASK2_MODE" == real ]]; then
        (cd "$TASK2_SOURCE_ROOT" && "$TASK2_FIXED_PYTHON" update_configs.py) >>"$TASK2_WORK_ROOT/logs/run_all.log" 2>&1
        update_rc=$?
        [[ "$update_rc" == 0 ]] || { task2_error "Echo update_configs command failed with exit ${update_rc}"; return 1; }
    else
        (cd "$TASK2_SOURCE_ROOT" && bash -c "$TASK2_UPDATE_COMMAND") >>"$TASK2_WORK_ROOT/logs/run_all.log" 2>&1
        update_rc=$?
        [[ "$update_rc" == 0 ]] || { task2_error "Echo update_configs command failed with exit ${update_rc}"; return 1; }
    fi
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 1 || return 1
    task2_attach_interpreter_provenance || return 1
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 0 || return 1
    start_ns=$(date +%s%N)
    if [[ "$TASK2_MODE" == real ]]; then
        (cd "$TASK2_SOURCE_ROOT" && bash run_all.sh) >>"$TASK2_WORK_ROOT/logs/run_all.log" 2>&1
    else
        (cd "$TASK2_SOURCE_ROOT" && bash -c "$TASK2_RUN_COMMAND") >>"$TASK2_WORK_ROOT/logs/run_all.log" 2>&1
    fi
    run_rc=$?
    end_ns=$(date +%s%N)
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 0 || return 1
    [[ "$run_rc" == 0 ]] || { task2_error "Echo run_all command failed with exit ${run_rc}"; return 1; }
    elapsed=$("$TASK2_META_PYTHON" - "$start_ns" "$end_ns" <<'PY'
import decimal
import sys
print(decimal.Decimal(sys.argv[2]) - decimal.Decimal(sys.argv[1]))
PY
) || return 1
    "$TASK2_META_PYTHON" - "$TASK2_WORK_ROOT/logs/run_timing.json" "$elapsed" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "sc26-ae-echo-run-timing-v1",
    "elapsed_seconds": float(sys.argv[2]) / 1_000_000_000,
    "update_exit_code": 0,
    "run_all_exit_code": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    task2_copy_canonical "$TASK2_RUN_ROOT" || return 1
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 0 || return 1
    if [[ -f "$TASK2_SOURCE_ROOT/training_testing/predict.py" ]]; then
        if [[ "$TASK2_MODE" == synthetic && "$TASK2_SKIP_PREDICT" == 1 ]]; then
            printf '%s\n' 'explicit synthetic prediction skip' >"$TASK2_WORK_ROOT/logs/predict_stdout.log"
        else
            (cd "$TASK2_SOURCE_ROOT/training_testing" && "$TASK2_PYTHON" predict.py) >"$TASK2_WORK_ROOT/logs/predict_stdout.log" 2>&1
            local predict_rc=$?
            [[ "$predict_rc" == 0 ]] || { task2_error "Echo predictor sample command failed with exit ${predict_rc}"; return 1; }
        fi
    elif [[ "$TASK2_MODE" != synthetic ]]; then
        task2_error "pinned Echo snapshot is missing training_testing/predict.py"; return 1
    fi
    if [[ -f "$TASK2_WORK_ROOT/logs/predict_stdout.log" ]]; then
        cp "$TASK2_WORK_ROOT/logs/predict_stdout.log" "$TASK2_RUN_ROOT/logs/predict_stdout.log" || return 1
    fi
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 0 || return 1
    local metrics_args=(
        --dataset-path "$TASK2_RUN_ROOT/training_testing/output/train_dataset.csv"
        --model-path "$TASK2_RUN_ROOT/training_testing/output/xgb_model.json"
        --scaler-path "$TASK2_RUN_ROOT/training_testing/output/standard_scaler.json"
        --log-path "$TASK2_RUN_ROOT/logs/run_all.log"
        --timing-path "$TASK2_RUN_ROOT/logs/run_timing.json"
        --output-dir "$TASK2_RUN_ROOT"
        --predictor-run-id "$TASK2_PREDICTOR_RUN_ID"
    )
    [[ "$TASK2_MODE" == synthetic ]] && metrics_args+=(--synthetic)
    "$TASK2_PYTHON" "$TASK2_REPO_ROOT/SC26-AE/tools/echo_metrics.py" "${metrics_args[@]}" \
        >"$TASK2_WORK_ROOT/logs/metrics_stdout.log" 2>&1
    local metrics_rc=$?
    [[ "$metrics_rc" == 0 ]] || { task2_error "echo_metrics.py rejected generated Task2 artifacts (exit ${metrics_rc})"; return 1; }
    cp "$TASK2_WORK_ROOT/logs/metrics_stdout.log" "$TASK2_RUN_ROOT/logs/metrics_stdout.log" || return 1
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 0 || return 1
    task2_write_artifact_manifest "$TASK2_RUN_ROOT" || return 1
    task2_verify_run "$TASK2_RUN_ROOT" || return 1
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$TASK2_RUN_ROOT/interpreter_binding.json" 0 || return 1
    task2_write_marker "${TASK2_OUTPUT_ROOT}/${TASK2_MODEL_KEY}/task2/predictor_marker.json" \
        "$TASK2_RUN_ROOT" "$TASK2_VERIFIED_MANIFEST_SHA256" || return 1
    task2_write_shared_pointer "${TASK2_OUTPUT_ROOT}/_shared/task2/predictor_marker.json" \
        "$TASK2_RUN_ROOT" "$TASK2_VERIFIED_MANIFEST_SHA256" || return 1
    task2_assert_source_clean || return 1
    printf '[PASS] Task2 %s predictor_run_id=%s (local evidence; no GPU qualification claim)\n' \
        "$TASK2_MODEL_KEY" "$TASK2_PREDICTOR_RUN_ID"
}

task2_write_failure_evidence() {
    local rc=$1
    local failure_python=$TASK2_FIXED_PYTHON
    [[ "$TASK2_MODE" == synthetic ]] && failure_python=$TASK2_META_PYTHON
    [[ -n "$TASK2_FAILURE_ROOT" ]] || return 0
    mkdir -p "$TASK2_FAILURE_ROOT" 2>/dev/null || {
        printf '[ERROR] unable to create failure evidence root: %s\n' "$TASK2_FAILURE_ROOT" >&2
        return 1
    }
    "$failure_python" - "$TASK2_FAILURE_ROOT/failure.json" "$rc" "$TASK2_MODEL_KEY" \
        "$TASK2_PREDICTOR_RUN_ID" "$TASK2_FAILURE_REASON" "$TASK2_MODE" <<'PY' 2>/dev/null
import json
import pathlib
import sys
from datetime import datetime, timezone

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "sc26-ae-task2-failure-v1",
    "exit_code": int(sys.argv[2]),
    "model": sys.argv[3],
    "predictor_run_id": sys.argv[4],
    "reason": sys.argv[5],
    "execution_mode": sys.argv[6],
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "qualification_claim": "none",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    printf '[ERROR] Task2 evidence preserved at %s\n' "$TASK2_FAILURE_ROOT" >&2
}

task2_main() {
    task2_prepare_failure_root || return 1
    task2_validate_execution_controls || return 1
    [[ "${TASK2_SKIP_OUTER_SOURCE_PROVENANCE:-0}" == "0" || "${TASK2_SKIP_OUTER_SOURCE_PROVENANCE:-0}" == "1" ]] || {
        task2_error "TASK2_SKIP_OUTER_SOURCE_PROVENANCE must be 0 or 1"; return 1;
    }
    task2_validate_model || return 1
    [[ "$TASK2_REBUILD" == 0 || "$TASK2_REBUILD" == 1 ]] || { task2_error "REBUILD must be 0 or 1"; return 1; }
    task2_bind_interpreter_chain || return 1
    task2_validate_cuda_ids || return 1
    task2_require_command git || return 1
    task2_require_command tar || return 1
    task2_require_command sha256sum || return 1
    task2_require_command "$TASK2_META_PYTHON" || return 1
    task2_require_file "$TASK2_ARTIFACT_TOOL" || return 1
    [[ -e "$TASK2_SOURCE_REPO/.git" ]] || { task2_error "Echo-slowdown is not a Git checkout"; return 1; }
    TASK2_MAIN_COMMIT=$(git -C "$TASK2_REPO_ROOT" rev-parse HEAD 2>/dev/null) || {
        task2_error "cannot resolve main-repository source commit"; return 1;
    }
    [[ "$TASK2_MAIN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
        task2_error "main-repository source commit is invalid: ${TASK2_MAIN_COMMIT}"; return 1;
    }
    task2_assert_source_clean || return 1
    task2_gitlink_commit || return 1
    if [[ "$TASK2_MODE" == real ]]; then
        task2_maybe_assert_outer_source_provenance "$TASK2_MAIN_COMMIT" || return 1
    fi
    if [[ "$TASK2_REBUILD" == 0 ]]; then
        task2_run_reuse
    else
        task2_run_build
    fi
}

task2_main
TASK2_RC=$?
if [[ "$TASK2_RC" != 0 ]]; then
    task2_write_failure_evidence "$TASK2_RC"
    if ! printf '%s\n' "$TASK2_FAILURE_REASON" >"${TASK2_FAILURE_ROOT}/failure.log" 2>/dev/null; then
        printf '[ERROR] unable to write failure log: %s\n' "$TASK2_FAILURE_ROOT/failure.log" >&2
    fi
fi
exit "$TASK2_RC"
