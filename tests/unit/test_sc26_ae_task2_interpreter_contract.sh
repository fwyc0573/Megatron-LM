#!/usr/bin/env bash
# Verify the fixed Task2 interpreter binding used by the v1.2-ae worker image.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNNER="$REPO_ROOT/SC26-AE/lib/task2_echo.sh"
EXPECTED=/opt/conda/envs/echo_slowdown/bin/python
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "$TMP_PARENT"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task2-runtime-binding.XXXXXX")

assignment=$(grep -E '^TASK2_PYTHON=' "$RUNNER")
test "$(printf '%s\n' "$assignment" | wc -l)" -eq 1

unset TASK2_PYTHON
eval "$assignment"
if [[ "$TASK2_PYTHON" != "$EXPECTED" ]]; then
    printf 'Task2 default interpreter mismatch: expected=%s actual=%s\n' \
        "$EXPECTED" "$TASK2_PYTHON" >&2
    exit 1
fi

cat >"$ROOT/fake_echo_python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c && "${2:-}" == 'import torch; print(torch.cuda.device_count())' ]]; then
    printf '%s\n' 2
    exit 0
fi
if [[ "${TASK2_TEST_INTERCEPT_UPDATE:-0}" == 1 && "${1:-}" == update_configs.py ]]; then
    printf '%s\n' 'external TASK2_PYTHON executed' >"${TASK2_TEST_INTERPRETER_SENTINEL:?}"
    exit 91
fi
exec python3 "$@"
SH
chmod +x "$ROOT/fake_echo_python"

cat >"$ROOT/fake_meta_python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'external TASK2_META_PYTHON executed' >"${TASK2_TEST_META_SENTINEL:?}"
exit 92
SH
chmod +x "$ROOT/fake_meta_python"

PASS_COUNT=1
FAIL_COUNT=0

has_failure_log() {
    find "$1/_work" -mindepth 2 -maxdepth 2 -type f -name failure.log -size +0c \
        -print -quit 2>/dev/null | grep -q .
}

expect_real_rejection() {
    local name=$1 expected_error=$2 sentinel=$3
    shift 3
    local case_root="$ROOT/$name"
    local output_root="$case_root/output"
    local log="$case_root/run.log"
    mkdir -p "$case_root"

    set +e
    env \
        -u TASK2_UPDATE_COMMAND \
        -u TASK2_RUN_COMMAND \
        -u TASK2_SKIP_UPDATE_CONFIGS \
        -u TASK2_SKIP_PREDICT \
        -u TASK2_PYTHON \
        -u TASK2_META_PYTHON \
        CUDA_VISIBLE_DEVICES=0,1 \
        AE_OUTPUT_ROOT="$output_root" \
        TASK2_SOURCE_REPO="$REPO_ROOT/Echo-slowdown" \
        TASK2_EXECUTION_MODE=real \
        REBUILD=1 \
        PREDICTOR_RUN_ID="$name" \
        TASK2_TEST_INTERPRETER_SENTINEL="$case_root/interpreter-sentinel" \
        TASK2_TEST_META_SENTINEL="$case_root/meta-sentinel" \
        "$@" \
        "$RUNNER" gpt175b >"$log" 2>&1
    local rc=$?
    set -e

    if [[ "$rc" == 0 ]]; then
        printf 'Task2 real binding case unexpectedly passed: %s\n' "$name" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    if [[ -e "$sentinel" ]]; then
        printf 'Task2 real binding executed an external control before rejection: %s\n' "$name" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    if ! grep -Fq "$expected_error" "$log"; then
        printf 'Task2 real binding did not report the expected rejection for %s: %s\n' \
            "$name" "$expected_error" >&2
        sed -n '1,120p' "$log" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    if ! has_failure_log "$output_root"; then
        printf 'Task2 real binding did not preserve failure evidence: %s\n' "$name" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    PASS_COUNT=$((PASS_COUNT + 1))
}

expect_early_control_rejection() {
    local name=$1 mode=$2 expected_error=$3
    shift 3
    local case_root="$ROOT/$name"
    local output_root="$case_root/output"
    local log="$case_root/run.log"
    mkdir -p "$case_root"

    set +e
    env \
        -u TASK2_UPDATE_COMMAND \
        -u TASK2_RUN_COMMAND \
        -u TASK2_SKIP_UPDATE_CONFIGS \
        -u TASK2_SKIP_PREDICT \
        -u TASK2_PYTHON \
        -u TASK2_META_PYTHON \
        CUDA_VISIBLE_DEVICES=0,1 \
        AE_OUTPUT_ROOT="$output_root" \
        TASK2_SOURCE_REPO="$REPO_ROOT/Echo-slowdown" \
        TASK2_EXECUTION_MODE="$mode" \
        REBUILD=1 \
        PREDICTOR_RUN_ID="$name" \
        "$@" \
        "$RUNNER" gpt175b >"$log" 2>&1
    local rc=$?
    set -e

    if [[ "$rc" == 0 ]] || ! grep -Fq "$expected_error" "$log" || ! has_failure_log "$output_root"; then
        printf 'Task2 control validation did not fail before producer work while preserving evidence for %s\n' "$name" >&2
        sed -n '1,120p' "$log" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    PASS_COUNT=$((PASS_COUNT + 1))
}

expect_real_rejection \
    reject-update-command \
    'TASK2_UPDATE_COMMAND overrides are forbidden in real mode' \
    "$ROOT/reject-update-command/producer-sentinel" \
    "TASK2_UPDATE_COMMAND=printf external-update > '$ROOT/reject-update-command/producer-sentinel'; exit 91" \
    "TASK2_PYTHON=$ROOT/fake_echo_python"

expect_real_rejection \
    reject-run-command \
    'TASK2_RUN_COMMAND overrides are forbidden in real mode' \
    "$ROOT/reject-run-command/producer-sentinel" \
    "TASK2_RUN_COMMAND=printf external-run > '$ROOT/reject-run-command/producer-sentinel'; exit 91" \
    "TASK2_PYTHON=$ROOT/fake_echo_python"

expect_real_rejection \
    reject-skip-update \
    'TASK2_SKIP_UPDATE_CONFIGS must be 0 in real mode' \
    "$ROOT/reject-skip-update/producer-sentinel" \
    'TASK2_SKIP_UPDATE_CONFIGS=1' \
    "TASK2_UPDATE_COMMAND=printf external-update > '$ROOT/reject-skip-update/producer-sentinel'; exit 91" \
    "TASK2_PYTHON=$ROOT/fake_echo_python"

expect_real_rejection \
    reject-skip-predict \
    'TASK2_SKIP_PREDICT must be 0 in real mode' \
    "$ROOT/reject-skip-predict/producer-sentinel" \
    'TASK2_SKIP_PREDICT=1' \
    "TASK2_UPDATE_COMMAND=printf external-update > '$ROOT/reject-skip-predict/producer-sentinel'; exit 91" \
    "TASK2_PYTHON=$ROOT/fake_echo_python"

expect_real_rejection \
    reject-python \
    "TASK2_PYTHON must equal the fixed real-mode interpreter: $EXPECTED" \
    "$ROOT/reject-python/interpreter-sentinel" \
    'TASK2_TEST_INTERCEPT_UPDATE=1' \
    "TASK2_PYTHON=$ROOT/fake_echo_python"

expect_real_rejection \
    reject-meta-python \
    "TASK2_META_PYTHON must equal the fixed real-mode interpreter: $EXPECTED" \
    "$ROOT/reject-meta-python/meta-sentinel" \
    "TASK2_META_PYTHON=$ROOT/fake_meta_python" \
    "TASK2_PYTHON=$ROOT/fake_echo_python"

expect_early_control_rejection \
    reject-invalid-mode \
    invalid \
    'TASK2_EXECUTION_MODE must be real or synthetic'

expect_early_control_rejection \
    reject-invalid-skip-update \
    synthetic \
    'TASK2_SKIP_UPDATE_CONFIGS must be 0 or 1' \
    'TASK2_SKIP_UPDATE_CONFIGS=2'

expect_early_control_rejection \
    reject-invalid-skip-predict \
    synthetic \
    'TASK2_SKIP_PREDICT must be 0 or 1' \
    'TASK2_SKIP_PREDICT=2'

test_real_direct_invocation() (
    local library_copy="$ROOT/task2_echo_library.sh"
    sed '/^task2_main$/,$d' "$RUNNER" >"$library_copy"
    # shellcheck disable=SC1090
    source "$library_copy"

    local case_root="$ROOT/real-direct-invocation"
    local fixture_root="$case_root/fixture"
    mkdir -p "$fixture_root" "$case_root/bin"
    cat >"$fixture_root/update_configs.py" <<'PY'
raise SystemExit('the fixed interpreter fixture must intercept this file')
PY
    cat >"$fixture_root/run_all.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'direct run_all.sh invocation' >"${TASK2_TEST_DIRECT_RUN_SENTINEL:?}"
exit 93
SH
    chmod +x "$fixture_root/run_all.sh"
    cat >"$case_root/bin/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == update_configs.py ]]; then
    printf '%s\n' 'direct fixed Python invocation' >"${TASK2_TEST_DIRECT_UPDATE_SENTINEL:?}"
    exit 0
fi
exec python3 "$@"
SH
    chmod +x "$case_root/bin/python"

    TASK2_MODE=real
    TASK2_MODEL_KEY=gpt175b
    TASK2_PREDICTOR_RUN_ID=direct-real
    TASK2_OUTPUT_ROOT="$case_root/output"
    TASK2_FIXED_PYTHON="$case_root/bin/python"
    TASK2_PYTHON="$TASK2_FIXED_PYTHON"
    TASK2_META_PYTHON="$TASK2_FIXED_PYTHON"
    TASK2_UPDATE_COMMAND="printf external-update > '$case_root/external-update-sentinel'"
    TASK2_RUN_COMMAND="printf external-run > '$case_root/external-run-sentinel'"
    export TASK2_TEST_DIRECT_UPDATE_SENTINEL="$case_root/direct-update-sentinel"
    export TASK2_TEST_DIRECT_RUN_SENTINEL="$case_root/direct-run-sentinel"

    task2_assert_source_clean() { return 0; }
    task2_gitlink_commit() { TASK2_ECHO_COMMIT=fixture; }
    task2_source_inventory() { : >"$1"; : >"$2"; }
    task2_extract_snapshot() {
        mkdir -p "$TASK2_SOURCE_ROOT"
        cp "$fixture_root/update_configs.py" "$TASK2_SOURCE_ROOT/update_configs.py"
        cp "$fixture_root/run_all.sh" "$TASK2_SOURCE_ROOT/run_all.sh"
    }
    task2_write_source_manifest() { printf '%s\n' '{}' >"$2"; }
    task2_write_provenance() { printf '%s\n' '{}' >"$1"; }
    task2_write_snapshot_precheck() { printf '%s\n' '{}' >"$1"; }
    # This case isolates direct invocation from unrelated artifact production.
    # Keep the real interpreter identity check, but stub downstream consumers
    # that require a complete Echo output bundle.
    task2_validate_nested_configs() { return 0; }
    task2_attach_interpreter_provenance() { return 0; }
    task2_copy_canonical() { return 0; }
    task2_write_artifact_manifest() { return 0; }
    task2_verify_run() { return 0; }
    task2_write_marker() { return 0; }
    task2_write_shared_pointer() { return 0; }

    task2_bind_interpreter_chain || return 1

    set +e
    task2_run_build >"$case_root/run.log" 2>&1
    local rc=$?
    set -e
    [[ "$rc" != 0 ]] || {
        printf '%s\n' 'direct invocation fixture unexpectedly succeeded' >&2
        return 1
    }
    grep -Fq 'Echo run_all command failed with exit 93' "$case_root/run.log" || {
        printf '%s\n' 'direct invocation fixture did not preserve run_all failure' >&2
        return 1
    }
    test -s "$case_root/direct-update-sentinel" || {
        printf '%s\n' 'fixed interpreter did not receive update_configs.py directly' >&2
        return 1
    }
    test -s "$case_root/direct-run-sentinel" || {
        printf '%s\n' 'run_all.sh was not invoked directly' >&2
        return 1
    }
    test ! -e "$case_root/external-update-sentinel" || {
        printf '%s\n' 'override update command was executed in real mode' >&2
        return 1
    }
    test ! -e "$case_root/external-run-sentinel" || {
        printf '%s\n' 'override run command was executed in real mode' >&2
        return 1
    }
)

if test_real_direct_invocation; then
    PASS_COUNT=$((PASS_COUNT + 1))
else
    printf '%s\n' 'Task2 real producer did not use fixed direct invocations' >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

test_nested_interpreter_chain_binding() (
    local library_copy="$ROOT/task2_echo_nested_library.sh"
    sed '/^task2_main$/,$d' "$RUNNER" >"$library_copy"
    # shellcheck disable=SC1090
    source "$library_copy"

    set +e
    local no_arg_output
    no_arg_output=$(
        TASK2_MODE=synthetic TASK2_RUN_ROOT= TASK2_META_PYTHON=python3 \
            task2_validate_binding_sidecar 2>&1
    )
    local no_arg_status=$?
    set -e
    if [[ "$no_arg_status" == 0 ]] || ! grep -Fq \
        'Task2 interpreter binding run root is unavailable' <<<"$no_arg_output"; then
        printf '%s\n' 'Task2 sidecar validator did not reject a missing run-root argument' >&2
        printf '%s\n' "$no_arg_output" >&2
        return 1
    fi

    local case_root="$ROOT/nested-chain-binding"
    local bin_root="$case_root/bin"
    local external_root="$case_root/external"
    local source_root="$case_root/source"
    local run_root="$case_root/run"
    local run_sentinel="$case_root/nested-run-sentinel"
    local external_sentinel="$case_root/external-interpreter-sentinel"
    local identity_log="$case_root/identity.log"
    mkdir -p "$bin_root" "$external_root" "$source_root" "$run_root"

    cat >"$bin_root/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == update_configs.py ]]; then
    root=$(pwd)
    for relative in \
        kernel_metric/input/global_config.json \
        merge/input/global_config.json \
        slowdown_collection/input/global_config.json \
        training_testing/input/global_config.json; do
        mkdir -p "$root/$(dirname "$relative")"
        python3 - "$root/$relative" <<'PY'
import json
import pathlib
import subprocess
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "python_path": subprocess.check_output(["which", "python"], text=True).strip(),
}), encoding="utf-8")
PY
    done
    exit 0
fi
exec python3 "$@"
SH
    chmod +x "$bin_root/python"

    cat >"$external_root/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'external interpreter was invoked' >"${TASK2_TEST_EXTERNAL_SENTINEL:?}"
exit 97
SH
    chmod +x "$external_root/python"

    cat >"$source_root/update_configs.py" <<'PY'
# The fixture interpreter handles this entry point.
PY
    cat >"$source_root/run_all.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'bare=%s\n' "$(command -v python)" >>"${TASK2_TEST_IDENTITY_LOG:?}"
python -c 'print("bare nested invocation")' >>"${TASK2_TEST_IDENTITY_LOG:?}"
for config in \
    kernel_metric/input/global_config.json \
    merge/input/global_config.json \
    slowdown_collection/input/global_config.json \
    training_testing/input/global_config.json; do
    python_path=$(python3 - "$config" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['python_path'])
PY
)
    "$python_path" -c 'print("configured nested invocation")' >>"${TASK2_TEST_IDENTITY_LOG:?}"
done
printf '%s\n' 'nested run completed' >"${TASK2_TEST_RUN_SENTINEL:?}"
SH
    chmod +x "$source_root/run_all.sh"

    TASK2_MODE=real
    TASK2_FIXED_PYTHON="$bin_root/python"
    TASK2_META_PYTHON="$TASK2_FIXED_PYTHON"
    TASK2_PYTHON="$TASK2_FIXED_PYTHON"
    TASK2_SOURCE_ROOT="$source_root"
    TASK2_RUN_ROOT="$run_root"
    TASK2_OUTPUT_ROOT="$case_root/output"
    export TASK2_TEST_EXTERNAL_SENTINEL="$external_sentinel"
    export TASK2_TEST_IDENTITY_LOG="$identity_log"
    export TASK2_TEST_RUN_SENTINEL="$run_sentinel"
    export PATH="$external_root:$PATH"

    task2_bind_interpreter_chain || return 1
    (cd "$source_root" && "$TASK2_FIXED_PYTHON" update_configs.py) || return 1
    task2_validate_nested_configs "$run_root/interpreter_binding.json" 1 || return 1
    (cd "$source_root" && bash run_all.sh) || return 1
    task2_validate_interpreter_chain || return 1
    task2_validate_nested_configs "$run_root/interpreter_binding.json" 0 || return 1
    python3 - "$run_root" "$TASK2_FIXED_PYTHON" <<'PY' || return 1
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sidecar = root / 'interpreter_binding.json'
sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
(root / 'provenance.json').write_text(json.dumps({
    'schema_version': 'sc26-ae-echo-provenance-v1',
    'execution_mode': 'real',
    'execution_evidence': 'runtime_measurement_requires_external_two_gpu_qualification',
    'run_command': 'bash run_all.sh',
    'update_command': sys.argv[2] + ' update_configs.py',
    'automatic_fallback': False,
    'interpreter_binding': {
        'status': 'bound',
        'artifact_path': 'interpreter_binding.json',
        'sha256': sidecar_hash,
    },
}, sort_keys=True) + '\n', encoding='utf-8')
paths = ['interpreter_binding.json', 'provenance.json'] + [
    'provenance/interpreter_configs/{}.global_config.json'.format(name)
    for name in ('kernel_metric', 'merge', 'slowdown_collection', 'training_testing')
]
entries = []
for relative in sorted(paths):
    path = root / relative
    payload = path.read_bytes()
    entries.append({
        'path': relative,
        'size_bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest(),
    })
(root / 'artifact_manifest.json').write_text(json.dumps({
    'execution_evidence': 'runtime_measurement_requires_external_two_gpu_qualification',
    'files': entries,
}, sort_keys=True) + '\n', encoding='utf-8')
PY
    task2_validate_binding_sidecar "$run_root" || return 1

    test -s "$run_sentinel" || return 1
    test ! -e "$external_sentinel" || return 1
    test "$(grep -c '^bare=' "$identity_log")" -eq 1 || return 1
    test "$(grep -c 'configured nested invocation' "$identity_log")" -eq 4 || return 1
    test "$(python3 - "$run_root/interpreter_binding.json" "$TASK2_FIXED_PYTHON" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
assert payload['schema_version'] == 'sc26-ae-task2-interpreter-binding-v1'
assert payload['status'] == 'bound'
assert payload['automatic_fallback'] is False
assert payload['fixed_requested_path'] == sys.argv[2]
assert len(payload['configs']) == 4
print('1')
PY
    )" -eq 1 || return 1

    local sidecar="$run_root/interpreter_binding.json"
    local archive="$run_root/provenance/interpreter_configs/kernel_metric.global_config.json"
    local provenance="$run_root/provenance.json"
    local manifest="$run_root/artifact_manifest.json"

    # A synthetic consumer must still verify a real-evidence bundle's
    # interpreter sidecar.  Only the live executable comparison is
    # mode-specific; artifact semantics are not.
    if ! (
        TASK2_MODE=synthetic
        TASK2_META_PYTHON=python3
        TASK2_FIXED_CANONICAL_PATH=""
        TASK2_FIXED_SHA256=""
        TASK2_PATH_PYTHON=""
        TASK2_PATH_PYTHON_CANONICAL_PATH=""
        TASK2_PATH_PYTHON_SHA256=""
        task2_validate_binding_sidecar "$run_root"
    ); then
        printf '%s\n' 'synthetic caller rejected a valid real-evidence sidecar' >&2
        return 1
    fi

    # A sidecar/checksum-coherent semantic fixture that names a different
    # fixed Python environment must still be rejected by a synthetic consumer.
    # The complete generic-manifest variant is covered by the integration
    # contract; this unit fixture isolates the sidecar path-binding seam.
    local alternate_root="$case_root/alternate-fixed-path"
    mkdir -p "$alternate_root/provenance/interpreter_configs" "$alternate_root/bin"
    cp "$run_root/provenance.json" "$alternate_root/provenance.json"
    cp "$run_root/artifact_manifest.json" "$alternate_root/artifact_manifest.json"
    cp "$run_root/interpreter_binding.json" "$alternate_root/interpreter_binding.json"
    cp "$run_root/provenance/interpreter_configs/"* \
        "$alternate_root/provenance/interpreter_configs/"
    python3 - "$alternate_root" "$TASK2_FIXED_PYTHON" <<'PY' || return 1
import hashlib
import json
import os
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
fixed = pathlib.Path(sys.argv[2])
alternate = root / "bin" / "alternate-python"
shutil.copy2(fixed, alternate)
alternate.chmod(0o755)
alternate_canonical = alternate.resolve(strict=True)
alternate_sha256 = hashlib.sha256(alternate_canonical.read_bytes()).hexdigest()
old_sidecar = json.loads((root / "interpreter_binding.json").read_text(encoding="utf-8"))
old_requested = old_sidecar["fixed_requested_path"]
if old_requested == alternate.as_posix():
    raise SystemExit("alternate fixture path unexpectedly equals fixed path")

config_paths = [
    "kernel_metric",
    "merge",
    "slowdown_collection",
    "training_testing",
]
for name in config_paths:
    archive = root / "provenance" / "interpreter_configs" / (name + ".global_config.json")
    payload = json.loads(archive.read_text(encoding="utf-8"))
    payload["python_path"] = alternate.as_posix()
    archive.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

sidecar = old_sidecar
sidecar["fixed_requested_path"] = alternate.as_posix()
sidecar["fixed_canonical_path"] = alternate_canonical.as_posix()
sidecar["fixed_sha256"] = alternate_sha256
sidecar["path_lookup_python"] = alternate.as_posix()
sidecar["path_lookup_canonical_path"] = alternate_canonical.as_posix()
sidecar["path_lookup_sha256"] = alternate_sha256
for row in sidecar["configs"]:
    row["python_path"] = alternate.as_posix()
    row["python_canonical_path"] = alternate_canonical.as_posix()
    row["python_sha256"] = alternate_sha256
    archive = root / row["artifact_relative_path"]
    archive_bytes = archive.read_bytes()
    row["size_bytes"] = len(archive_bytes)
    row["sha256"] = hashlib.sha256(archive_bytes).hexdigest()
(root / "interpreter_binding.json").write_text(
    json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

provenance_path = root / "provenance.json"
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
provenance["update_command"] = alternate.as_posix() + " update_configs.py"
provenance["interpreter_binding"] = {
    "status": "bound",
    "artifact_path": "interpreter_binding.json",
    "sha256": hashlib.sha256((root / "interpreter_binding.json").read_bytes()).hexdigest(),
}
provenance_path.write_text(
    json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
)

manifest_path = root / "artifact_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for entry in manifest["files"]:
    artifact = root / entry["path"]
    data = artifact.read_bytes()
    entry["size_bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
PY
    local alternate_output
    if alternate_output=$(
        TASK2_MODE=synthetic
        TASK2_META_PYTHON=python3
        TASK2_FIXED_CANONICAL_PATH=""
        TASK2_FIXED_SHA256=""
        TASK2_PATH_PYTHON=""
        TASK2_PATH_PYTHON_CANONICAL_PATH=""
        TASK2_PATH_PYTHON_SHA256=""
        task2_validate_binding_sidecar "$alternate_root" 2>&1
    ); then
        printf '%s\n' 'synthetic caller accepted an alternate fixed interpreter path' >&2
        return 1
    fi
    grep -Fq 'sidecar fixed requested path differs from wrapper fixed interpreter' <<<"$alternate_output" || {
        printf '%s\n' 'alternate fixed path rejection did not identify the wrapper contract' >&2
        printf '%s\n' "$alternate_output" >&2
        return 1
    }
    printf '%s\n' 'ALTERNATE_FIXED_PATH_NEGATIVE=1'

    local reuse_matrix_root="$case_root/reuse-matrix-run"
    mkdir -p "$reuse_matrix_root/provenance/interpreter_configs"
    cp "$run_root/provenance.json" "$reuse_matrix_root/provenance.json"
    cp "$run_root/artifact_manifest.json" "$reuse_matrix_root/artifact_manifest.json"
    cp "$run_root/provenance/interpreter_configs/"* \
        "$reuse_matrix_root/provenance/interpreter_configs/"
    if (
        TASK2_MODE=synthetic
        TASK2_META_PYTHON=python3
        TASK2_VALIDATE_ONLY=1
        task2_validate_binding_sidecar "$reuse_matrix_root"
    ); then
        printf '%s\n' 'synthetic caller accepted a real-evidence bundle without sidecar' >&2
        return 1
    fi
    printf '%s\n' 'REUSE_MODE_MATRIX=valid-real-under-synthetic:pass,missing-sidecar:reject'

    cp "$sidecar" "$case_root/sidecar.saved" || return 1
    cp "$archive" "$case_root/archive.saved" || return 1
    cp "$provenance" "$case_root/provenance.saved" || return 1
    cp "$manifest" "$case_root/manifest.saved" || return 1

    python3 - "$sidecar" <<'PY' || return 1
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['automatic_fallback'] = True
path.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'sidecar fallback tamper was accepted' >&2
        return 1
    fi
    cp "$case_root/sidecar.saved" "$sidecar" || return 1

    printf '%s\n' 'tampered archive' >>"$archive"
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'archived config tamper was accepted' >&2
        return 1
    fi
    cp "$case_root/archive.saved" "$archive" || return 1

    python3 - "$provenance" <<'PY' || return 1
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['interpreter_binding']['sha256'] = '0' * 64
path.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'provenance tamper was accepted' >&2
        return 1
    fi
    cp "$case_root/provenance.saved" "$provenance" || return 1

    python3 - "$manifest" <<'PY' || return 1
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['files'] = [entry for entry in payload['files'] if entry['path'] != 'interpreter_binding.json']
path.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'manifest omission was accepted' >&2
        return 1
    fi
    cp "$case_root/manifest.saved" "$manifest" || return 1

    local config="$source_root/kernel_metric/input/global_config.json"
    cp "$config" "$case_root/config.saved" || return 1
    printf '%s\n' '{"python_path":"relative/python"}' >"$config"
    if task2_validate_nested_configs "$sidecar" 0; then
        printf '%s\n' 'relative nested python_path was accepted' >&2
        return 1
    fi

    for invalid_kind in malformed non_object missing non_string whitespace control del; do
        python3 - "$config" "$invalid_kind" <<'PY' || return 1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
kind = sys.argv[2]
requested = '/absolute/fixed/python'
if kind == 'malformed':
    path.write_text('{', encoding='utf-8')
elif kind == 'non_object':
    path.write_text('[]\n', encoding='utf-8')
elif kind == 'missing':
    path.write_text('{}\n', encoding='utf-8')
elif kind == 'non_string':
    path.write_text(json.dumps({'python_path': 7}) + '\n', encoding='utf-8')
elif kind == 'whitespace':
    path.write_text(json.dumps({'python_path': requested + ' '}) + '\n', encoding='utf-8')
elif kind == 'control':
    path.write_text(json.dumps({'python_path': requested + '\t'}) + '\n', encoding='utf-8')
elif kind == 'del':
    path.write_text(json.dumps({'python_path': requested + '\x7f'}) + '\n', encoding='utf-8')
else:
    raise SystemExit('unknown invalid config case: ' + kind)
PY
        if task2_validate_nested_configs "$sidecar" 0; then
            printf 'invalid nested config was accepted: %s\n' "$invalid_kind" >&2
            return 1
        fi
    done
    cp "$case_root/config.saved" "$config" || return 1

    python3 - "$sidecar" <<'PY' || return 1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['fixed_sha256'] = '0' * 64
path.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'different executable SHA256 was accepted' >&2
        return 1
    fi
    cp "$case_root/sidecar.saved" "$sidecar" || return 1

    local archive_fixture="$case_root/archive-shape"
    mkdir -p "$archive_fixture/provenance/interpreter_configs" || return 1
    cp "$run_root/interpreter_binding.json" "$archive_fixture/interpreter_binding.json" || return 1
    cp "$run_root/provenance.json" "$archive_fixture/provenance.json" || return 1
    cp "$run_root/artifact_manifest.json" "$archive_fixture/artifact_manifest.json" || return 1
    for archive_name in merge slowdown_collection training_testing; do
        cp "$run_root/provenance/interpreter_configs/${archive_name}.global_config.json" \
            "$archive_fixture/provenance/interpreter_configs/${archive_name}.global_config.json" || return 1
    done
    if task2_validate_binding_sidecar "$archive_fixture"; then
        printf '%s\n' 'missing archived interpreter config was accepted' >&2
        return 1
    fi

    ln -s "$archive" \
        "$archive_fixture/provenance/interpreter_configs/kernel_metric.global_config.json" || return 1
    if task2_validate_binding_sidecar "$archive_fixture"; then
        printf '%s\n' 'symlinked archived interpreter config was accepted' >&2
        return 1
    fi

    # Duplicate-key cases must fail closed instead of relying on Python's
    # last-value-wins JSON behavior.
    python3 - "$sidecar" <<'PY' || return 1
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding='utf-8').splitlines()
for index, line in enumerate(lines):
    if '"status": "bound"' in line:
        duplicate = line if line.rstrip().endswith(',') else line.rstrip() + ','
        lines.insert(index, duplicate)
        break
else:
    raise SystemExit('sidecar status field was not found')
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'duplicate sidecar key was accepted' >&2
        return 1
    fi
    cp "$case_root/sidecar.saved" "$sidecar" || return 1

    python3 - "$archive" "$sidecar" "$manifest" <<'PY' || return 1
import hashlib
import json
import pathlib
import sys

archive = pathlib.Path(sys.argv[1])
sidecar = pathlib.Path(sys.argv[2])
manifest = pathlib.Path(sys.argv[3])
payload = json.loads(sidecar.read_text(encoding='utf-8'))
requested = payload['fixed_requested_path']
archive.write_text(
    json.dumps({'python_path': requested}, sort_keys=True)
    .replace('}', ', "python_path": ' + json.dumps(requested) + '}')
    + '\n',
    encoding='utf-8',
)
archive_bytes = archive.read_bytes()
for row in payload['configs']:
    if row['artifact_relative_path'].endswith('/kernel_metric.global_config.json'):
        row['size_bytes'] = len(archive_bytes)
        row['sha256'] = hashlib.sha256(archive_bytes).hexdigest()
sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
manifest_payload = json.loads(manifest.read_text(encoding='utf-8'))
for entry in manifest_payload['files']:
    if entry['path'] == 'provenance/interpreter_configs/kernel_metric.global_config.json':
        entry['size_bytes'] = len(archive_bytes)
        entry['sha256'] = hashlib.sha256(archive_bytes).hexdigest()
manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'duplicate archived python_path key was accepted' >&2
        return 1
    fi
    cp "$case_root/sidecar.saved" "$sidecar" || return 1
    cp "$case_root/archive.saved" "$archive" || return 1
    cp "$case_root/manifest.saved" "$manifest" || return 1

    python3 - "$provenance" <<'PY' || return 1
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
needle = '"interpreter_binding": {'
assert text.count(needle) == 1
path.write_text(text.replace(needle, '"interpreter_binding": {} ,\n  "interpreter_binding": {', 1), encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'duplicate provenance interpreter_binding key was accepted' >&2
        return 1
    fi
    cp "$case_root/provenance.saved" "$provenance" || return 1

    python3 - "$manifest" <<'PY' || return 1
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['files'].append(dict(payload['files'][0]))
path.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
PY
    if task2_validate_binding_sidecar "$run_root"; then
        printf '%s\n' 'duplicate manifest file entry was accepted' >&2
        return 1
    fi
    cp "$case_root/manifest.saved" "$manifest" || return 1

    printf 'PARSER_NEGATIVES=%s\n' 11
    printf 'DUPLICATE_KEY_NEGATIVES=%s\n' 4
    printf 'SIDECAR_TAMPER_NEGATIVES=%s\n' 9
)

if test_nested_interpreter_chain_binding; then
    PASS_COUNT=$((PASS_COUNT + 1))
else
    printf '%s\n' 'Task2 nested interpreter chain was not bound to the fixed executable' >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [[ "$FAIL_COUNT" != 0 ]]; then
    printf 'Task2 real runtime binding failures=%s passes=%s\n' "$FAIL_COUNT" "$PASS_COUNT" >&2
    exit 1
fi

printf 'PASS_COUNT=%s\n' "$PASS_COUNT"
printf 'PASS: Task2 default interpreter=%s and real runtime controls fail fast\n' "$TASK2_PYTHON"
printf 'Evidence root: %s\n' "$ROOT"
