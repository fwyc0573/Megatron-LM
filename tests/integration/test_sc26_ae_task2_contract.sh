#!/usr/bin/env bash
# Synthetic integration contract.  It verifies snapshot-only execution and
# shared predictor identity; it is not a two-GPU Echo qualification.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task2-integration.XXXXXX")
SOURCE="$ROOT/echo"
OUT="$ROOT/output"
mkdir -p "$SOURCE/training_testing/output" "$SOURCE/merge/input"
git -C "$SOURCE" init -q
git -C "$SOURCE" config user.email ae-test@example.invalid
git -C "$SOURCE" config user.name sc26-ae-test
cat >"$SOURCE/update_configs.py" <<'PY'
from pathlib import Path
Path('update_marker').write_text('snapshot-only', encoding='utf-8')
PY
cat >"$SOURCE/run_all.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Running kernel_metric module...' 'Running slowdown_collection module...' 'Running merge module...' 'Running training_testing module...'
python3 "$TASK2_SYNTHETIC_RUNNER"
SH
cat >"$SOURCE/training_testing/predict.py" <<'PY'
print({'prediction_stdout_only': True})
PY
chmod +x "$SOURCE/run_all.sh"
git -C "$SOURCE" add .
git -C "$SOURCE" commit -q -m initial
COMMIT=$(git -C "$SOURCE" rev-parse HEAD)
cat >"$ROOT/generate.py" <<'PY'
from pathlib import Path
import json
root = Path('.')
(root / 'training_testing/output').mkdir(parents=True, exist_ok=True)
(root / 'merge/input').mkdir(parents=True, exist_ok=True)
(root / 'training_testing/output/train_dataset.csv').write_text(
    'ground_truth,Compute throughput,slowdown\n1,2,0.1\n2,3,0.2\n', encoding='utf-8')
(root / 'training_testing/output/xgb_model.json').write_text(json.dumps({
    'format': 'sc26-ae-synthetic-xgb-v1', 'weights': [0.1, 0.2], 'bias': 0.3}), encoding='utf-8')
(root / 'training_testing/output/standard_scaler.json').write_text(json.dumps({
    'feature_names': ['ground_truth', 'Compute throughput'], 'mean': [0, 0], 'scale': [1, 1]}), encoding='utf-8')
(root / 'merge/input/kernel_metric_output.csv').write_text('Kernel Name,SM\nfake,1\n', encoding='utf-8')
print('MSE for each fold (validation set): [1.0, 2.0, 3.0, 4.0, 5.0]')
print('Average MSE (validation set): 3.0')
print('Test MSE: 0.5')
PY

common_env() {
    CUDA_VISIBLE_DEVICES=0,1 \
    AE_OUTPUT_ROOT="$OUT" \
    TASK2_SOURCE_REPO="$SOURCE" \
    TASK2_GITLINK_COMMIT="$COMMIT" \
    TASK2_EXECUTION_MODE=synthetic \
    TASK2_META_PYTHON=python3 \
    TASK2_PYTHON=python3 \
    TASK2_RUN_COMMAND="TASK2_SYNTHETIC_RUNNER='$ROOT/generate.py' bash run_all.sh" \
    TASK2_SKIP_HARDWARE_CHECK=1 \
    REBUILD="$1" \
    PREDICTOR_RUN_ID="$2" \
    "$REPO_ROOT/SC26-AE/$3"
}

# Build complete synthetic runs outside OUT so the symlink cases reach the
# resolver's containment boundary with internally consistent IDs, checksums,
# metrics, and provenance.  A copied integration run would retain the original
# predictor_run_id and fail the earlier identity checks instead.
external_env() {
    local output_root=$1 predictor_id=$2 entry=$3
    CUDA_VISIBLE_DEVICES=0,1 \
    AE_OUTPUT_ROOT="$output_root" \
    TASK2_SOURCE_REPO="$SOURCE" \
    TASK2_GITLINK_COMMIT="$COMMIT" \
    TASK2_EXECUTION_MODE=synthetic \
    TASK2_META_PYTHON=python3 \
    TASK2_PYTHON=python3 \
    TASK2_UPDATE_COMMAND="python3 update_configs.py" \
    TASK2_RUN_COMMAND="TASK2_SYNTHETIC_RUNNER='$ROOT/generate.py' bash run_all.sh" \
    TASK2_SKIP_HARDWARE_CHECK=1 \
    REBUILD=1 \
    PREDICTOR_RUN_ID="$predictor_id" \
    "$REPO_ROOT/SC26-AE/$entry"
}

TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 1 integration-one task2_gpt175b.sh

GPT_MARKER="$OUT/gpt175b/task2/predictor_marker.json"
SHARED_POINTER="$OUT/_shared/task2/predictor_marker.json"

# Canonical verification must accept a real-evidence bundle when a synthetic
# caller performs artifact-only reuse validation.  Build that bundle in a
# separate fixture so the original local-synthetic producer remains untouched.
RUN="$OUT/_shared/task2/runs/integration-one"
QUALIFIED_RUN="$ROOT/qualified-real-run"
cp -a "$RUN" "$QUALIFIED_RUN"
python3 - "$QUALIFIED_RUN" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
requested_text = '/opt/conda/envs/echo_slowdown/bin/python'
canonical = pathlib.Path(sys.executable).resolve()
canonical_text = str(canonical)
digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
config_paths = [
    'kernel_metric/input/global_config.json',
    'merge/input/global_config.json',
    'slowdown_collection/input/global_config.json',
    'training_testing/input/global_config.json',
]
archive_rows = []
for source in config_paths:
    module = pathlib.PurePosixPath(source).parts[0]
    relative = 'provenance/interpreter_configs/{}.global_config.json'.format(module)
    archive = root / relative
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(json.dumps({'python_path': requested_text}) + '\n', encoding='utf-8')
    payload = archive.read_bytes()
    archive_rows.append({
        'source_relative_path': source,
        'artifact_relative_path': relative,
        'size_bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'python_path': requested_text,
        'python_canonical_path': canonical_text,
        'python_sha256': digest,
    })
sidecar = root / 'interpreter_binding.json'
sidecar.write_text(json.dumps({
    'schema_version': 'sc26-ae-task2-interpreter-binding-v1',
    'status': 'bound',
    'execution_mode': 'real',
    'automatic_fallback': False,
    'fixed_requested_path': requested_text,
    'fixed_canonical_path': canonical_text,
    'fixed_sha256': digest,
    'path_lookup_python': requested_text,
    'path_lookup_canonical_path': canonical_text,
    'path_lookup_sha256': digest,
    'configs': archive_rows,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
provenance_path = root / 'provenance.json'
provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
provenance.update({
    'schema_version': 'sc26-ae-echo-provenance-v1',
    'execution_mode': 'real',
    'execution_evidence': 'runtime_measurement_requires_external_two_gpu_qualification',
    'run_command': 'bash run_all.sh',
    'update_command': requested_text + ' update_configs.py',
    'automatic_fallback': False,
    'interpreter_binding': {
        'status': 'bound',
        'artifact_path': 'interpreter_binding.json',
        'sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    },
})
provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
manifest_path = root / 'artifact_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['execution_evidence'] = 'real_exact_two_h800_qualified'
by_path = {entry['path']: entry for entry in manifest['files']}
for relative in ['interpreter_binding.json', 'provenance.json'] + [
    row['artifact_relative_path'] for row in archive_rows
]:
    path = root / relative
    by_path[relative] = {
        'path': relative,
        'size_bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
manifest['files'] = [by_path[key] for key in sorted(by_path)]
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
VERIFY_LIBRARY="$ROOT/task2-verify-library.sh"
sed '/^task2_main$/,$d' "$REPO_ROOT/SC26-AE/lib/task2_echo.sh" >"$VERIFY_LIBRARY"
set +e
VERIFY_OUTPUT=$(
    # shellcheck disable=SC1090
    source "$VERIFY_LIBRARY"
    TASK2_REPO_ROOT="$REPO_ROOT"
    TASK2_MAIN_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
    TASK2_ARTIFACT_TOOL="$REPO_ROOT/SC26-AE/tools/artifact_manifest.py"
    TASK2_MODE=synthetic
    TASK2_META_PYTHON=python3
    TASK2_ECHO_COMMIT="$COMMIT"
    TASK2_PREDICTOR_RUN_ID=integration-one
    if task2_verify_run "$QUALIFIED_RUN"; then
        printf 'QUALIFIED_VERIFY_STATUS=accepted\n'
    else
        status=$?
        printf 'QUALIFIED_VERIFY_STATUS=rejected\n'
        exit "$status"
    fi
) 2>&1
VERIFY_STATUS=$?
set -e
if [[ "$VERIFY_STATUS" != 0 ]]; then
    printf '%s\n' "$VERIFY_OUTPUT" >&2
    echo 'canonical Task2 verification rejected an already-qualified evidence state' >&2
    exit 1
fi
grep -Fq 'QUALIFIED_VERIFY_STATUS=accepted' <<<"$VERIFY_OUTPUT"
printf '%s\n' 'PASS: canonical Task2 verification accepts qualified evidence'

# A checksum-coherent real bundle that rewrites the entire interpreter chain
# must still fail the wrapper's fixed requested-path contract.  This uses the
# complete qualified fixture (including the generic manifest schema), rather
# than the smaller sidecar-only unit fixture, so the generic verifier result
# and semantic verifier result are independently observable.
ALTERNATE_REQUESTED_RUN="$ROOT/alternate-requested-real-run"
cp -a "$QUALIFIED_RUN" "$ALTERNATE_REQUESTED_RUN"
python3 - "$ALTERNATE_REQUESTED_RUN" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
alternate = root / 'alternate-bin' / 'python'
alternate.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(pathlib.Path(sys.executable), alternate)
alternate.chmod(0o755)
alternate_canonical = alternate.resolve(strict=True)
alternate_requested = alternate.as_posix()
alternate_canonical_text = alternate_canonical.as_posix()
alternate_sha256 = hashlib.sha256(alternate_canonical.read_bytes()).hexdigest()

sidecar_path = root / 'interpreter_binding.json'
sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
sidecar['fixed_requested_path'] = alternate_requested
sidecar['fixed_canonical_path'] = alternate_canonical_text
sidecar['fixed_sha256'] = alternate_sha256
sidecar['path_lookup_python'] = alternate_requested
sidecar['path_lookup_canonical_path'] = alternate_canonical_text
sidecar['path_lookup_sha256'] = alternate_sha256
for row in sidecar['configs']:
    row['python_path'] = alternate_requested
    row['python_canonical_path'] = alternate_canonical_text
    row['python_sha256'] = alternate_sha256
    archive = root / row['artifact_relative_path']
    payload = json.loads(archive.read_text(encoding='utf-8'))
    payload['python_path'] = alternate_requested
    archive.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
    data = archive.read_bytes()
    row['size_bytes'] = len(data)
    row['sha256'] = hashlib.sha256(data).hexdigest()
sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + '\n', encoding='utf-8')

provenance_path = root / 'provenance.json'
provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
provenance['update_command'] = alternate_requested + ' update_configs.py'
provenance['interpreter_binding']['sha256'] = hashlib.sha256(
    sidecar_path.read_bytes()
).hexdigest()
provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')

manifest_path = root / 'artifact_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
alternate_data = alternate.read_bytes()
manifest['files'] = [
    entry for entry in manifest['files'] if entry['path'] != 'alternate-bin/python'
]
manifest['files'].append({
    'path': 'alternate-bin/python',
    'size_bytes': len(alternate_data),
    'sha256': hashlib.sha256(alternate_data).hexdigest(),
})
for entry in manifest['files']:
    artifact = root / entry['path']
    data = artifact.read_bytes()
    entry['size_bytes'] = len(data)
    entry['sha256'] = hashlib.sha256(data).hexdigest()
manifest['files'] = sorted(manifest['files'], key=lambda entry: entry['path'])
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
set +e
ALTERNATE_MANIFEST_OUTPUT=$(python3 "$REPO_ROOT/SC26-AE/tools/artifact_manifest.py" verify \
    --root "$ALTERNATE_REQUESTED_RUN" \
    --manifest "$ALTERNATE_REQUESTED_RUN/artifact_manifest.json" 2>&1)
ALTERNATE_MANIFEST_STATUS=$?
set -e
[[ "$ALTERNATE_MANIFEST_STATUS" == 0 ]]
grep -Fq 'MANIFEST_STATUS=verified' <<<"$ALTERNATE_MANIFEST_OUTPUT"
# The complete alternate bundle must expose its file-count evidence explicitly;
# without this marker the transcript can be confused with the smaller 13-file
# qualified fixture used by the surrounding integration checks.
grep -Fxq 'MANIFEST_FILE_COUNT=19' <<<"$ALTERNATE_MANIFEST_OUTPUT"
printf '%s\n' 'ALTERNATE_MANIFEST_STATUS=verified'
printf '%s\n' 'ALTERNATE_MANIFEST_FILE_COUNT=19'
set +e
ALTERNATE_SEMANTIC_OUTPUT=$(
    # shellcheck disable=SC1090
    source "$VERIFY_LIBRARY"
    TASK2_REPO_ROOT="$REPO_ROOT"
    TASK2_ARTIFACT_TOOL="$REPO_ROOT/SC26-AE/tools/artifact_manifest.py"
    TASK2_MODE=synthetic
    TASK2_META_PYTHON=python3
    TASK2_FIXED_CANONICAL_PATH=""
    TASK2_FIXED_SHA256=""
    TASK2_PATH_PYTHON=""
    TASK2_PATH_PYTHON_CANONICAL_PATH=""
    TASK2_PATH_PYTHON_SHA256=""
    if task2_validate_binding_sidecar "$ALTERNATE_REQUESTED_RUN" 2>&1; then
        printf 'ALTERNATE_SEMANTIC_STATUS=accepted\n'
    else
        status=$?
        printf 'ALTERNATE_SEMANTIC_STATUS=rejected\n'
        exit "$status"
    fi
)
ALTERNATE_SEMANTIC_STATUS=$?
set -e
[[ "$ALTERNATE_SEMANTIC_STATUS" != 0 ]]
grep -Fq 'sidecar fixed requested path differs from wrapper fixed interpreter' \
    <<<"$ALTERNATE_SEMANTIC_OUTPUT"
printf '%s\n' 'PASS: checksum-coherent alternate requested path is generic-verified but semantically rejected'

# A checksum-coherent tamper must still fail the interpreter semantic layer.
# Mutate the sidecar, all archived configs, and provenance together, then
# rebuild every manifest entry. The generic artifact verifier should accept
# this self-consistent byte set, while the interpreter contract rejects the
# non-canonical executable path before any marker or pointer is published.
COHERENT_TAMPER_RUN="$ROOT/coherent-tamper-real-run"
cp -a "$QUALIFIED_RUN" "$COHERENT_TAMPER_RUN"
python3 - "$COHERENT_TAMPER_RUN" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for archive in sorted((root / 'provenance/interpreter_configs').glob('*.json')):
    archive.write_bytes(archive.read_bytes() + b'\n')

sidecar_path = root / 'interpreter_binding.json'
sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
canonical = sidecar['fixed_canonical_path']
directory, name = canonical.rsplit('/', 1)
noncanonical = directory + '/./' + name
sidecar['fixed_canonical_path'] = noncanonical
sidecar['path_lookup_canonical_path'] = noncanonical
for row in sidecar['configs']:
    archive = root / row['artifact_relative_path']
    payload = archive.read_bytes()
    row['size_bytes'] = len(payload)
    row['sha256'] = hashlib.sha256(payload).hexdigest()
    row['python_canonical_path'] = noncanonical
sidecar_path.write_text(
    json.dumps(sidecar, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)

provenance_path = root / 'provenance.json'
provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
provenance['interpreter_binding']['sha256'] = hashlib.sha256(
    sidecar_path.read_bytes()
).hexdigest()
provenance_path.write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)

manifest_path = root / 'artifact_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
entries = []
for path in root.rglob('*'):
    if path.is_file() and path.name != 'artifact_manifest.json':
        payload = path.read_bytes()
        entries.append({
            'path': path.relative_to(root).as_posix(),
            'size_bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        })
manifest['files'] = sorted(entries, key=lambda entry: entry['path'])
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
PY
set +e
COHERENT_MANIFEST_OUTPUT=$(python3 "$REPO_ROOT/SC26-AE/tools/artifact_manifest.py" verify \
    --root "$COHERENT_TAMPER_RUN" \
    --manifest "$COHERENT_TAMPER_RUN/artifact_manifest.json" 2>&1)
COHERENT_MANIFEST_STATUS=$?
set -e
[[ "$COHERENT_MANIFEST_STATUS" == 0 ]]
grep -Fq 'MANIFEST_STATUS=verified' <<<"$COHERENT_MANIFEST_OUTPUT"
printf '%s\n' 'PASS: coherent tamper remains generic-manifest verified'

COHERENT_OUTPUT_ROOT="$ROOT/coherent-tamper-output"
mkdir -p "$COHERENT_OUTPUT_ROOT/_shared/task2/runs"
cp -a "$COHERENT_TAMPER_RUN" \
    "$COHERENT_OUTPUT_ROOT/_shared/task2/runs/integration-one"
COHERENT_MANIFEST_SHA256=$(sha256sum \
    "$COHERENT_OUTPUT_ROOT/_shared/task2/runs/integration-one/artifact_manifest.json" \
    | awk '{print $1}')
python3 - "$COHERENT_OUTPUT_ROOT/_shared/task2/predictor_marker.json" \
    "$COHERENT_MANIFEST_SHA256" <<'PY'
import json
import pathlib
import sys

pointer = pathlib.Path(sys.argv[1])
digest = sys.argv[2]
pointer.parent.mkdir(parents=True, exist_ok=True)
pointer.write_text(json.dumps({
    'schema_version': 'sc26-ae-task2-shared-pointer-v1',
    'predictor_run_id': 'integration-one',
    'run_path': '_shared/task2/runs/integration-one',
    'run_relative_path': '_shared/task2/runs/integration-one',
    'manifest_sha256': digest,
    'artifact_manifest_sha256': digest,
    'verified': True,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
COHERENT_POINTER_SHA256_BEFORE=$(sha256sum \
    "$COHERENT_OUTPUT_ROOT/_shared/task2/predictor_marker.json" | awk '{print $1}')
set +e
COHERENT_REUSE_OUTPUT=$(CUDA_VISIBLE_DEVICES=0,1 \
    AE_OUTPUT_ROOT="$COHERENT_OUTPUT_ROOT" \
    TASK2_SOURCE_REPO="$SOURCE" \
    TASK2_GITLINK_COMMIT="$COMMIT" \
    TASK2_EXECUTION_MODE=synthetic \
    TASK2_META_PYTHON=python3 \
    TASK2_PYTHON=python3 \
    TASK2_SKIP_HARDWARE_CHECK=1 \
    REBUILD=0 \
    PREDICTOR_RUN_ID=ignored \
    "$REPO_ROOT/SC26-AE/task2_qwen3_a30b.sh" 2>&1)
COHERENT_REUSE_STATUS=$?
set -e
[[ "$COHERENT_REUSE_STATUS" != 0 ]]
grep -Fq 'canonical interpreter path is not lexically canonical' \
    <<<"$COHERENT_REUSE_OUTPUT"
COHERENT_POINTER_SHA256_AFTER=$(sha256sum \
    "$COHERENT_OUTPUT_ROOT/_shared/task2/predictor_marker.json" | awk '{print $1}')
[[ "$COHERENT_POINTER_SHA256_AFTER" == "$COHERENT_POINTER_SHA256_BEFORE" ]]
test ! -e "$COHERENT_OUTPUT_ROOT/qwen3_a30b/task2/predictor_marker.json"
printf '%s\n' 'PASS: coherent semantic tamper fails before marker/pointer publication'

# Keep the valid producer artifacts untouched while each negative case edits
# only its selected marker/pointer payload.
cp "$GPT_MARKER" "$ROOT/gpt-marker-original.json"
cp "$SHARED_POINTER" "$ROOT/shared-pointer-original.json"

MARKER_EXTERNAL_OUT="$ROOT/external-marker-output"
POINTER_EXTERNAL_OUT="$ROOT/external-pointer-output"
external_env "$MARKER_EXTERNAL_OUT" intermediate-marker-escape task2_gpt175b.sh \
    >"$ROOT/external-marker-build.log" 2>&1
external_env "$POINTER_EXTERNAL_OUT" intermediate-pointer-escape task2_qwen3_a30b.sh \
    >"$ROOT/external-pointer-build.log" 2>&1

# A marker path may be lexically safe while an intermediate symlink resolves
# outside the canonical Task2 output root.  Reuse must reject the resolved
# escape before it can consume or publish any evidence.
ESCAPED_MARKER_RUN="$MARKER_EXTERNAL_OUT/_shared/task2/runs/intermediate-marker-escape"
ln -s "$ESCAPED_MARKER_RUN" "$OUT/_shared/task2/runs/intermediate-marker-escape"
cp "$MARKER_EXTERNAL_OUT/gpt175b/task2/predictor_marker.json" "$GPT_MARKER"
python3 - "$GPT_MARKER" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['run_path'] = '_shared/task2/runs/intermediate-marker-escape'
payload['run_relative_path'] = '_shared/task2/runs/intermediate-marker-escape'
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
if TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_gpt175b.sh \
    >"$ROOT/intermediate-marker-escape.log" 2>&1; then
    echo 'Task2 accepted a model marker whose resolved path escapes the output root' >&2
    exit 1
fi
grep -Fq 'existing Task2 marker path escapes output root' \
    "$ROOT/intermediate-marker-escape.log"
cp "$ROOT/gpt-marker-original.json" "$GPT_MARKER"
printf '%s\n' 'PASS: Task2 rejects an intermediate symlink escape in a model marker'

# The shared-pointer resolver has the same trust boundary and must fail before
# attaching a model-level marker when its selected run escapes the output root.
ESCAPED_POINTER_RUN="$POINTER_EXTERNAL_OUT/_shared/task2/runs/intermediate-pointer-escape"
ln -s "$ESCAPED_POINTER_RUN" "$OUT/_shared/task2/runs/intermediate-pointer-escape"
cp "$POINTER_EXTERNAL_OUT/_shared/task2/predictor_marker.json" "$SHARED_POINTER"
python3 - "$SHARED_POINTER" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['run_path'] = '_shared/task2/runs/intermediate-pointer-escape'
payload['run_relative_path'] = '_shared/task2/runs/intermediate-pointer-escape'
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
if TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_qwen3_a30b.sh \
    >"$ROOT/intermediate-pointer-escape.log" 2>&1; then
    echo 'Task2 accepted a shared pointer whose resolved path escapes the output root' >&2
    exit 1
fi
grep -Fq 'existing Task2 shared pointer path escapes output root' \
    "$ROOT/intermediate-pointer-escape.log"
test ! -e "$OUT/qwen3_a30b/task2/predictor_marker.json"
cp "$ROOT/shared-pointer-original.json" "$SHARED_POINTER"
printf '%s\n' 'PASS: Task2 rejects an intermediate symlink escape in a shared pointer'

# Both Task2 pointer schemas publish the same canonical run location through
# run_path and run_relative_path.  Reuse must reject a missing/divergent alias
# and any in-root path outside _shared/task2/runs/<predictor_run_id> before it
# opens the selected bundle.
set_task2_path_aliases() {
    local payload_path=$1 run_path=$2 run_relative_path=$3
    python3 - "$payload_path" "$run_path" "$run_relative_path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['run_path'] = sys.argv[2]
payload['run_relative_path'] = sys.argv[3]
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
}

I56_PATH_CONTRACT_FAILURES=0
I56_PATH_CONTRACT_PASS_COUNT=0
expect_i56_path_rejection() {
    local entry=$1 expected_error=$2 log_path=$3
    set +e
    TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored "$entry" \
        >"$log_path" 2>&1
    local status=$?
    set -e
    if [[ "$status" == 0 ]]; then
        I56_PATH_CONTRACT_FAILURES=$((I56_PATH_CONTRACT_FAILURES + 1))
        return 0
    fi
    if ! grep -Fq "$expected_error" "$log_path"; then
        cat "$log_path" >&2
        printf 'Task2 path rejection did not report the expected root cause: %s\n' \
            "$expected_error" >&2
        exit 1
    fi
    I56_PATH_CONTRACT_PASS_COUNT=$((I56_PATH_CONTRACT_PASS_COUNT + 1))
}

set_task2_path_aliases "$GPT_MARKER" \
    '_shared/task2/runs/integration-one' \
    '_shared/task2/runs/divergent-marker-alias'
expect_i56_path_rejection task2_gpt175b.sh \
    'existing Task2 marker path aliases are missing or differ' \
    "$ROOT/marker-path-alias-mismatch.log"
cp "$ROOT/gpt-marker-original.json" "$GPT_MARKER"

mkdir -p "$OUT/noncanonical-marker"
ln -s "$RUN" "$OUT/noncanonical-marker/integration-one"
set_task2_path_aliases "$GPT_MARKER" \
    'noncanonical-marker/integration-one' \
    'noncanonical-marker/integration-one'
expect_i56_path_rejection task2_gpt175b.sh \
    'existing Task2 marker path is not canonical for predictor_run_id' \
    "$ROOT/marker-noncanonical-in-root.log"
cp "$ROOT/gpt-marker-original.json" "$GPT_MARKER"

set_task2_path_aliases "$SHARED_POINTER" \
    '_shared/task2/runs/integration-one' \
    '_shared/task2/runs/divergent-pointer-alias'
expect_i56_path_rejection task2_qwen3_a30b.sh \
    'shared Task2 pointer path aliases are missing or differ' \
    "$ROOT/pointer-path-alias-mismatch.log"
cp "$ROOT/shared-pointer-original.json" "$SHARED_POINTER"

mkdir -p "$OUT/noncanonical-pointer"
ln -s "$RUN" "$OUT/noncanonical-pointer/integration-one"
set_task2_path_aliases "$SHARED_POINTER" \
    'noncanonical-pointer/integration-one' \
    'noncanonical-pointer/integration-one'
expect_i56_path_rejection task2_dsv3.sh \
    'shared Task2 pointer path is not canonical for predictor_run_id' \
    "$ROOT/pointer-noncanonical-in-root.log"
cp "$ROOT/shared-pointer-original.json" "$SHARED_POINTER"

if [[ "$I56_PATH_CONTRACT_FAILURES" != 0 ]]; then
    printf 'I56_PATH_CONTRACT_RED unexpected_acceptances=%s\n' \
        "$I56_PATH_CONTRACT_FAILURES" >&2
    exit 1
fi
[[ "$I56_PATH_CONTRACT_PASS_COUNT" == 4 ]]
printf '%s\n' 'PASS: Task2 rejects divergent path aliases and noncanonical in-root run paths in both resolvers'

# A model-level marker must bind predictor_run_id to the selected immutable
# run directory.  Changing only the marker identity must fail even when all
# artifact and metrics checksums still match.
python3 - "$GPT_MARKER" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['predictor_run_id'] = 'tampered-predictor-id'
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
if TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_gpt175b.sh \
    >"$ROOT/tampered-model-marker.log" 2>&1; then
    echo 'model marker predictor_run_id mismatch was unexpectedly accepted' >&2
    exit 1
fi
grep -Fq 'existing Task2 marker predictor_run_id does not match run path' \
    "$ROOT/tampered-model-marker.log"
python3 - "$GPT_MARKER" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['predictor_run_id'] = 'integration-one'
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
printf '%s\n' 'PASS: Task2 rejects a model marker whose predictor identity differs from its run path'

# Both checksum aliases are part of the shared-pointer contract.  Changing
# only the artifact_manifest_sha256 alias must fail before a model marker is
# published, even when the legacy manifest_sha256 alias remains correct.
python3 - "$OUT/_shared/task2/predictor_marker.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['verified'] = True
payload['artifact_manifest_sha256'] = '0' * 64
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
if TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_qwen3_a30b.sh \
    >"$ROOT/tampered-shared-pointer-alias.log" 2>&1; then
    echo 'tampered shared Task2 artifact_manifest_sha256 alias was unexpectedly accepted' >&2
    exit 1
fi
grep -Fq 'shared pointer manifest checksum mismatch' \
    "$ROOT/tampered-shared-pointer-alias.log"
test ! -e "$OUT/qwen3_a30b/task2/predictor_marker.json"
printf '%s\n' 'PASS: Task2 rejects a shared pointer with a mismatched artifact_manifest_sha256 alias'

# Restore the producer pointer before the existing unverified-pointer check.
python3 - "$OUT/_shared/task2/predictor_marker.json" \
    "$OUT/_shared/task2/runs/integration-one/artifact_manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

pointer_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
payload = json.loads(pointer_path.read_text(encoding='utf-8'))
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
payload['manifest_sha256'] = digest
payload['artifact_manifest_sha256'] = digest
pointer_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY

# The shared pointer is a producer/consumer boundary.  An unverified pointer
# must be rejected before a model-level attachment marker is published.
python3 - "$OUT/_shared/task2/predictor_marker.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['verified'] = False
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY
if TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_qwen3_a30b.sh \
    >"$ROOT/unverified-shared-pointer.log" 2>&1; then
    echo 'unverified shared Task2 pointer was unexpectedly accepted' >&2
    exit 1
fi
grep -Fq 'shared Task2 pointer is not verified' "$ROOT/unverified-shared-pointer.log"
test ! -e "$OUT/qwen3_a30b/task2/predictor_marker.json"
printf '%s\n' 'PASS: Task2 rejects an unverified shared predictor pointer before attachment'
python3 - "$OUT/_shared/task2/predictor_marker.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
payload['verified'] = True
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding='utf-8')
PY

TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_qwen3_a30b.sh
TASK2_UPDATE_COMMAND='python3 update_configs.py' common_env 0 ignored task2_dsv3.sh

for model in gpt175b qwen3_a30b dsv3; do
    marker="$OUT/$model/task2/predictor_marker.json"
    test -s "$marker"
    test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["predictor_run_id"])' "$marker")" = integration-one
    test -s "$OUT/$model/task2/../.." 2>/dev/null || true
done

RUN="$OUT/_shared/task2/runs/integration-one"
test -s "$RUN/logs/run_all.log"
test -s "$RUN/logs/predict_stdout.log"
grep -q 'prediction_stdout_only' "$RUN/logs/predict_stdout.log"
test ! -e "$RUN/training_testing/output/prediction"
test ! -e "$SOURCE/update_marker"
test -s "$RUN/metrics.json"
test -s "$RUN/artifact_manifest.json"
python3 - \
    "$RUN/metrics.json" \
    "$RUN/artifact_manifest.json" \
    "$OUT/_shared/task2/predictor_marker.json" <<'PY'
import hashlib
import json
import sys
metrics = json.load(open(sys.argv[1], encoding='utf-8'))
manifest = json.load(open(sys.argv[2], encoding='utf-8'))
shared_pointer = json.load(open(sys.argv[3], encoding='utf-8'))
assert metrics['predictor_run_id'] == 'integration-one'
assert metrics['dataset_row_count'] == 2
assert len(metrics['validation_mse_by_fold']) == 5
assert manifest['schema_version'] == 'sc26-ae-artifact-manifest-v1'
assert manifest['model'] == 'shared_task2'
assert manifest['task'] == 'task2'
assert manifest['artifact_source'] == 'fresh'
assert manifest['predictor_run_id'] == 'integration-one'
assert 'capture_id' not in manifest
assert set(manifest['source_commits']) == {
    'megatron_lm', 'echo_slowdown', 'megatron_sim_engine'
}
assert all(
    isinstance(manifest['source_commits'][key], str)
    and len(manifest['source_commits'][key]) == 40
    for key in manifest['source_commits']
)
assert manifest['execution_evidence'] == 'local_synthetic_not_two_gpu_qualification'
assert all(row['sha256'] and row['size_bytes'] > 0 for row in manifest['files'])
assert shared_pointer['schema_version'] == 'sc26-ae-task2-shared-pointer-v1'
assert shared_pointer['verified'] is True
assert shared_pointer['manifest_sha256'] == hashlib.sha256(
    open(sys.argv[2], 'rb').read()
).hexdigest()
PY
test -z "$(git -C "$SOURCE" status --porcelain)"
printf '%s\n' 'PASS: Task2 snapshot-only execution, predictor identity, and provenance contract' \
    'Evidence root:' "$ROOT"
