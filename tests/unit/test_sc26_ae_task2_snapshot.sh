#!/usr/bin/env bash
# Synthetic/local Task2 snapshot contract test.  It intentionally leaves its
# temporary evidence root available for inspection and never mutates the real
# Echo-slowdown checkout.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
RUNNER_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task2-unit.XXXXXX")
SOURCE_ROOT="$RUNNER_ROOT/fake-echo"
OUTPUT_ROOT="$RUNNER_ROOT/output"
mkdir -p "$SOURCE_ROOT/training_testing/output" "$SOURCE_ROOT/merge/output" "$SOURCE_ROOT/merge/input"
git -C "$SOURCE_ROOT" init -q
git -C "$SOURCE_ROOT" config user.email ae-test@example.invalid
git -C "$SOURCE_ROOT" config user.name sc26-ae-test
cat >"$SOURCE_ROOT/update_configs.py" <<'PY'
print('synthetic update_configs')
PY
cat >"$SOURCE_ROOT/run_all.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
python3 "$TASK2_SYNTHETIC_RUNNER"
SH
chmod +x "$SOURCE_ROOT/run_all.sh"
# This file is intentionally a known historical artifact.  Archive extraction
# must exclude it, rather than extracting and deleting it later.
printf 'stale upstream model\n' >"$SOURCE_ROOT/training_testing/output/xgb_model.json"
git -C "$SOURCE_ROOT" add .
git -C "$SOURCE_ROOT" commit -q -m initial
COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
cat >"$RUNNER_ROOT/generate.py" <<'PY'
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

run_task2() {
    local model=$1 rebuild=$2 run_id=$3
    CUDA_VISIBLE_DEVICES=0,1 \
    AE_OUTPUT_ROOT="$OUTPUT_ROOT" \
    TASK2_SOURCE_REPO="$SOURCE_ROOT" \
    TASK2_GITLINK_COMMIT="$COMMIT" \
    TASK2_EXECUTION_MODE=synthetic \
    TASK2_META_PYTHON=python3 \
    TASK2_PYTHON=python3 \
    TASK2_UPDATE_COMMAND=true \
    TASK2_RUN_COMMAND="TASK2_SYNTHETIC_RUNNER='$RUNNER_ROOT/generate.py' bash run_all.sh" \
    TASK2_SKIP_PREDICT=1 \
    REBUILD="$rebuild" \
    PREDICTOR_RUN_ID="$run_id" \
    "$REPO_ROOT/SC26-AE/lib/task2_echo.sh" "$model"
}

run_task2 gpt175b 1 snapshot-one
SNAPSHOT="$OUTPUT_ROOT/_work/task2.snapshot-one/source"
test -s "$OUTPUT_ROOT/_shared/task2/runs/snapshot-one/source_manifest.json"
test -s "$OUTPUT_ROOT/_shared/task2/runs/snapshot-one/artifact_manifest.json"
python3 - "$OUTPUT_ROOT/_shared/task2/runs/snapshot-one/snapshot_precheck.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
assert payload['pre_execution_clean'] is True
assert payload['excluded_paths_present_before_run'] == []
PY
python3 - "$OUTPUT_ROOT/_shared/task2/runs/snapshot-one/source_manifest.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
files = {row['path'] for row in payload['files']}
assert 'training_testing/output/xgb_model.json' not in files
assert 'training_testing/output/xgb_model.json' in payload['excluded_historical_paths']
PY
python3 - "$OUTPUT_ROOT/_shared/task2/runs/snapshot-one/artifact_manifest.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
assert payload['schema_version'] == 'sc26-ae-artifact-manifest-v1'
assert payload['model'] == 'shared_task2'
assert payload['task'] == 'task2'
assert payload['artifact_source'] == 'fresh'
assert payload['predictor_run_id'] == 'snapshot-one'
assert 'capture_id' not in payload
assert set(payload['source_commits']) == {
    'megatron_lm', 'echo_slowdown', 'megatron_sim_engine'
}
assert all(row['size_bytes'] > 0 and len(row['sha256']) == 64 for row in payload['files'])
PY

run_task2 qwen3_a30b 0 ignored-id
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["predictor_run_id"])' "$OUTPUT_ROOT/qwen3_a30b/task2/predictor_marker.json")" = snapshot-one

run_task2 dsv3 1 snapshot-two
test -d "$OUTPUT_ROOT/_shared/task2/runs/snapshot-one"
test -d "$OUTPUT_ROOT/_shared/task2/runs/snapshot-two"

# Dirty-before evidence is a hard failure and must leave a failure artifact.
printf 'dirty\n' >"$SOURCE_ROOT/dirty.txt"
if run_task2 gpt175b 1 dirty-run; then
    echo "dirty source unexpectedly passed" >&2
    exit 1
fi
find "$OUTPUT_ROOT/_work" -maxdepth 2 -type f -name 'failure.json' -print -quit | grep -q failure.json

# Commit the temporary dirty fixture in its isolated fake repository so the
# next assertion reaches the tracked-path inventory branch.
git -C "$SOURCE_ROOT" add dirty.txt
git -C "$SOURCE_ROOT" commit -q -m 'clean temporary fixture'

# A newly tracked path under an excluded prefix is rejected before archive
# extraction until the exclusion contract is explicitly reviewed.
printf 'new\n' >"$SOURCE_ROOT/merge/output/new.csv"
git -C "$SOURCE_ROOT" add merge/output/new.csv
git -C "$SOURCE_ROOT" commit -q -m 'new generated path'
COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
if run_task2 gpt175b 1 new-excluded-path; then
    echo "new excluded tracked path unexpectedly passed" >&2
    exit 1
fi

test -z "$(git -C "$REPO_ROOT/Echo-slowdown" status --porcelain)"
printf '%s\n' 'PASS: synthetic snapshot/reuse/rebuild/dirty/new-path contracts' \
    'Evidence root:' "$RUNNER_ROOT"
