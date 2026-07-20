#!/usr/bin/env bash
# Focused local-only Task3 portability and provenance regression coverage.
# Passing this script is not real-GPU or pre-dataset qualification evidence.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
FIXTURE_HELPER="${REPO_ROOT}/tests/integration/fixtures/sc26_ae_task3_fixture.py"
MANIFEST_TOOL="${REPO_ROOT}/SC26-AE/tools/artifact_manifest.py"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task3-portability.XXXXXX")
TOOLS_ROOT="${TEST_ROOT}/tools"
PREBAKED_SOURCE="${TEST_ROOT}/prebaked-source"
PASS_COUNT=0

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$1"
}

expect_failure() {
    local needle=$1
    local output_path=$2
    shift 2
    local status
    set +e
    "$@" >"${output_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || {
        cat "${output_path}" >&2
        fail "expected failure: $*"
    }
    grep -Fq -- "${needle}" "${output_path}" || {
        cat "${output_path}" >&2
        fail "missing failure text '${needle}'"
    }
}

task3_env() {
    local output_root=$1
    local prebaked_root=$2
    local model=$3
    local run_id=$4
    shift 4
    env \
        AE_OUTPUT_ROOT="${output_root}" \
        ARTIFACT_SOURCE=prebaked \
        PREBAKED_ROOT="${prebaked_root}" \
        SIMULATOR_HARDWARE_TYPE=H800_SXM \
        TASK3_EXECUTION_MODE=synthetic \
        TASK3_META_PYTHON=python3 \
        TASK3_SIMULATOR_PYTHON=python3 \
        TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}" \
        TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py" \
        TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py" \
        TASK3_SIMULATION_RUN_ID="${run_id}" \
        "$@" \
        bash "${REPO_ROOT}/SC26-AE/task3_${model}.sh"
}

real_task3_env() {
    local output_root=$1
    local prebaked_root=$2
    local model=$3
    local run_id=$4
    shift 4
    env \
        AE_OUTPUT_ROOT="${output_root}" \
        ARTIFACT_SOURCE=prebaked \
        PREBAKED_ROOT="${prebaked_root}" \
        SIMULATOR_HARDWARE_TYPE=H800_SXM \
        TASK3_SIM_ENGINE_ROOT="${REPO_ROOT}/megatron-sim-engine" \
        TASK3_SIMULATION_RUN_ID="${run_id}" \
        "$@" \
        bash "${REPO_ROOT}/SC26-AE/task3_${model}.sh"
}

fresh_task3_env() {
    local output_root=$1
    local model=$2
    local run_id=$3
    env \
        AE_OUTPUT_ROOT="${output_root}" \
        ARTIFACT_SOURCE=fresh \
        SIMULATOR_HARDWARE_TYPE=H800_SXM \
        TASK3_EXECUTION_MODE=synthetic \
        TASK3_META_PYTHON=python3 \
        TASK3_SIMULATOR_PYTHON=python3 \
        TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}" \
        TASK3_BUILDER="${TOOLS_ROOT}/builder.py" \
        TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py" \
        TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py" \
        TASK3_SIMULATION_RUN_ID="${run_id}" \
        bash "${REPO_ROOT}/SC26-AE/task3_${model}.sh"
}

set_json_field() {
    local path=$1
    local key=$2
    local json_value=$3
    python3 - "${path}" "${key}" "${json_value}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload[sys.argv[2]] = json.loads(sys.argv[3])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

reseal_nested_manifest() {
    local root=$1
    local bundle_key=$2
    local nested_relative=$3
    local key=$4
    local json_value=$5
    python3 - "${root}" "${bundle_key}" "${nested_relative}" "${key}" "${json_value}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
bundle_key, nested_relative, key, json_value = sys.argv[2:]
nested = root / nested_relative
payload = json.loads(nested.read_text(encoding="utf-8"))
payload[key] = json.loads(json_value)
nested.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

digest = hashlib.sha256(nested.read_bytes()).hexdigest()
distribution_path = root / "distribution_manifest.json"
distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
distribution["bundles"][bundle_key]["manifest_sha256"] = digest
for entry in distribution["files"]:
    if entry["path"] == nested_relative:
        entry["size_bytes"] = nested.stat().st_size
        entry["sha256"] = digest
        break
else:
    raise SystemExit(f"missing distribution entry: {nested_relative}")
distribution_path.write_text(
    json.dumps(distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

python3 "${FIXTURE_HELPER}" tools --output-root "${TOOLS_ROOT}" \
    >"${TEST_ROOT}/tools.json"
python3 "${FIXTURE_HELPER}" prebaked \
    --repo-root "${REPO_ROOT}" \
    --output-root "${PREBAKED_SOURCE}" \
    >"${TEST_ROOT}/prebaked.json"

HOSTILE_PRODUCER="${TEST_ROOT}/hostile_producer.py"
HOSTILE_SENTINEL="${TEST_ROOT}/hostile_producer_started"
cat >"${HOSTILE_PRODUCER}" <<'PY'
import os
import pathlib

pathlib.Path(os.environ["TASK3_HOSTILE_SENTINEL"]).write_text(
    "started\n", encoding="utf-8"
)
raise SystemExit("external Task3 producer executed")
PY
for producer_variable in TASK3_BUILDER TASK3_SCHEDULER TASK3_SIMULATOR; do
    expect_failure \
        "Fixed real-mode Task3 interpreter must be an executable regular file" \
        "${TEST_ROOT}/${producer_variable}.log" \
        real_task3_env "${TEST_ROOT}/${producer_variable}-output" \
            "${PREBAKED_SOURCE}" gpt175b "${producer_variable}-rejected" \
            TASK3_HOSTILE_SENTINEL="${HOSTILE_SENTINEL}" \
            "${producer_variable}=${HOSTILE_PRODUCER}"
    [[ ! -e "${HOSTILE_SENTINEL}" ]] || \
        fail "external ${producer_variable} executed before real-mode provenance rejection"
    [[ ! -e "${TEST_ROOT}/${producer_variable}-output" ]] || \
        fail "external ${producer_variable} created Task3 output before provenance rejection"
    pass "real Task3 rejects external ${producer_variable} before producer execution"
done

REAL_EVIDENCE_OUTPUT="${TEST_ROOT}/real-evidence-output"
expect_failure 'Fixed real-mode Task3 interpreter must be an executable regular file' \
    "${TEST_ROOT}/real-evidence.log" \
    real_task3_env "${REAL_EVIDENCE_OUTPUT}" "${PREBAKED_SOURCE}" \
        gpt175b real-evidence
[[ ! -e "${REAL_EVIDENCE_OUTPUT}/gpt175b/task3/run_marker.json" ]] || \
    fail 'synthetic prebaked evidence published a real-mode Task3 marker'
pass 'real Task3 enforces the fixed interpreter before synthetic prebaked reuse'

UNKNOWN_DISTRIBUTION_ROOT="${TEST_ROOT}/unknown-distribution-evidence-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${UNKNOWN_DISTRIBUTION_ROOT}"
set_json_field \
    "${UNKNOWN_DISTRIBUTION_ROOT}/distribution_manifest.json" \
    execution_evidence '"unsupported_synthetic_evidence"'
expect_failure 'distribution manifest execution evidence class is invalid' \
    "${TEST_ROOT}/unknown-distribution-evidence.log" \
    task3_env "${TEST_ROOT}/unknown-distribution-evidence-output" \
        "${UNKNOWN_DISTRIBUTION_ROOT}" gpt175b unknown-distribution-evidence
pass 'synthetic Task3 rejects an unknown distribution evidence class'

UNKNOWN_MODEL_ROOT="${TEST_ROOT}/unknown-model-evidence-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${UNKNOWN_MODEL_ROOT}"
reseal_nested_manifest \
    "${UNKNOWN_MODEL_ROOT}" gpt175b \
    bundles/gpt175b/artifact_manifest.json execution_evidence \
    '"unsupported_synthetic_evidence"'
expect_failure 'prebaked model artifact manifest execution evidence class is invalid' \
    "${TEST_ROOT}/unknown-model-evidence.log" \
    task3_env "${TEST_ROOT}/unknown-model-evidence-output" \
        "${UNKNOWN_MODEL_ROOT}" gpt175b unknown-model-evidence
pass 'synthetic Task3 rejects an unknown model evidence class'

UNKNOWN_SHARED_ROOT="${TEST_ROOT}/unknown-shared-evidence-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${UNKNOWN_SHARED_ROOT}"
reseal_nested_manifest \
    "${UNKNOWN_SHARED_ROOT}" shared_task2 \
    bundles/shared_task2/artifact_manifest.json execution_evidence \
    '"unsupported_synthetic_evidence"'
expect_failure 'prebaked shared Task2 manifest execution evidence class is invalid' \
    "${TEST_ROOT}/unknown-shared-evidence.log" \
    task3_env "${TEST_ROOT}/unknown-shared-evidence-output" \
        "${UNKNOWN_SHARED_ROOT}" gpt175b unknown-shared-evidence
pass 'synthetic Task3 rejects an unknown shared Task2 evidence class'

UNVERIFIED_CAPTURE_OUTPUT="${TEST_ROOT}/unverified-capture-output"
python3 "${FIXTURE_HELPER}" fresh-inputs \
    --repo-root "${REPO_ROOT}" \
    --output-root "${UNVERIFIED_CAPTURE_OUTPUT}" \
    --model gpt175b \
    >"${TEST_ROOT}/unverified-capture-inputs.json"
set_json_field \
    "${UNVERIFIED_CAPTURE_OUTPUT}/gpt175b/task1/capture_marker.json" \
    verified false
expect_failure 'Task1 capture marker is not verified' \
    "${TEST_ROOT}/unverified-capture.log" \
    fresh_task3_env "${UNVERIFIED_CAPTURE_OUTPUT}" gpt175b unverified-capture
[[ ! -e "${UNVERIFIED_CAPTURE_OUTPUT}/gpt175b/task3/run_marker.json" ]] || \
    fail 'unverified Task1 capture marker published a Task3 marker'
pass 'fresh Task3 rejects an unverified Task1 capture marker'

UNVERIFIED_PREDICTOR_OUTPUT="${TEST_ROOT}/unverified-predictor-output"
python3 "${FIXTURE_HELPER}" fresh-inputs \
    --repo-root "${REPO_ROOT}" \
    --output-root "${UNVERIFIED_PREDICTOR_OUTPUT}" \
    --model gpt175b \
    >"${TEST_ROOT}/unverified-predictor-inputs.json"
set_json_field \
    "${UNVERIFIED_PREDICTOR_OUTPUT}/_shared/task2/predictor_marker.json" \
    verified false
expect_failure 'Task2 shared predictor marker is not verified' \
    "${TEST_ROOT}/unverified-predictor.log" \
    fresh_task3_env "${UNVERIFIED_PREDICTOR_OUTPUT}" gpt175b unverified-predictor
[[ ! -e "${UNVERIFIED_PREDICTOR_OUTPUT}/gpt175b/task3/run_marker.json" ]] || \
    fail 'unverified Task2 predictor marker published a Task3 marker'
pass 'fresh Task3 rejects an unverified shared Task2 predictor marker'

TRAVERSAL_OUTPUT="${TEST_ROOT}/traversal-output"
TRAVERSAL_TARGET="${TEST_ROOT}/escaped-task3"
mkdir -p "${TRAVERSAL_OUTPUT}/gpt175b" "${TRAVERSAL_TARGET}"
ln -s "${TRAVERSAL_TARGET}" "${TRAVERSAL_OUTPUT}/gpt175b/task3"
expect_failure 'Task3 output path contains a symlink' \
    "${TEST_ROOT}/traversal.log" \
    task3_env "${TRAVERSAL_OUTPUT}" "${PREBAKED_SOURCE}" \
        gpt175b traversal-run
[[ ! -e "${TRAVERSAL_TARGET}/runs" && ! -e "${TRAVERSAL_TARGET}/run_marker.json" ]] || \
    fail 'Task3 wrote through a symlink outside the selected model output root'
pass 'Task3 rejects parent-directory symlink traversal before creating a run'

TASK1_SYMLINK_OUTPUT="${TEST_ROOT}/task1-symlink-output"
TASK1_SYMLINK_TARGET="${TEST_ROOT}/task1-symlink-target"
mkdir -p "${TASK1_SYMLINK_OUTPUT}/gpt175b" "${TASK1_SYMLINK_TARGET}"
ln -s -- "${TASK1_SYMLINK_TARGET}" "${TASK1_SYMLINK_OUTPUT}/gpt175b/task1"
python3 "${FIXTURE_HELPER}" fresh-inputs \
    --repo-root "${REPO_ROOT}" \
    --output-root "${TASK1_SYMLINK_OUTPUT}" \
    --model gpt175b \
    >"${TEST_ROOT}/task1-symlink-inputs.json"
expect_failure 'Task3 output path contains a symlink' \
    "${TEST_ROOT}/task1-symlink.log" \
    fresh_task3_env "${TASK1_SYMLINK_OUTPUT}" gpt175b task1-symlink
[[ ! -e "${TASK1_SYMLINK_OUTPUT}/gpt175b/task3/run_marker.json" ]] || \
    fail 'Task3 consumed a Task1 root through an intermediate symlink'
pass 'Task3 rejects an intermediate Task1-root symlink before resolving fresh inputs'

NON_DIRECTORY_OUTPUT="${TEST_ROOT}/non-directory-output"
mkdir -p "${NON_DIRECTORY_OUTPUT}"
printf 'not a directory\n' >"${NON_DIRECTORY_OUTPUT}/gpt175b"
expect_failure 'Task3 output path is not a directory' \
    "${TEST_ROOT}/non-directory.log" \
    task3_env "${NON_DIRECTORY_OUTPUT}" "${PREBAKED_SOURCE}" \
        gpt175b non-directory
pass 'Task3 rejects a non-directory model output component'

DANGLING_OUTPUT="${TEST_ROOT}/dangling-output"
mkdir -p "${DANGLING_OUTPUT}/gpt175b/task3/runs"
ln -s "${TEST_ROOT}/missing-run-target" \
    "${DANGLING_OUTPUT}/gpt175b/task3/runs/dangling-run"
expect_failure 'Task3 run destination already exists' \
    "${TEST_ROOT}/dangling-run.log" \
    task3_env "${DANGLING_OUTPUT}" "${PREBAKED_SOURCE}" \
        gpt175b dangling-run
pass 'Task3 rejects a dangling symlink at the selected immutable run root'

OVERLAP_ROOT="${TEST_ROOT}/overlap-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${OVERLAP_ROOT}"
reseal_nested_manifest \
    "${OVERLAP_ROOT}" gpt175b \
    bundles/gpt175b/artifact_manifest.json ddp_overlap false
expect_failure 'ddp_overlap' \
    "${TEST_ROOT}/overlap.log" \
    task3_env "${TEST_ROOT}/overlap-output" "${OVERLAP_ROOT}" \
        gpt175b overlap-false
pass 'prebaked Task3 rejects a checksum-consistent non-overlap model manifest'

SHARED_SOURCE_ROOT="${TEST_ROOT}/shared-source-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${SHARED_SOURCE_ROOT}"
reseal_nested_manifest \
    "${SHARED_SOURCE_ROOT}" shared_task2 \
    bundles/shared_task2/artifact_manifest.json artifact_source '"fresh"'
expect_failure 'prebaked shared Task2 manifest source is invalid' \
    "${TEST_ROOT}/shared-source.log" \
    task3_env "${TEST_ROOT}/shared-source-output" "${SHARED_SOURCE_ROOT}" \
        gpt175b shared-source-fresh
pass 'prebaked Task3 rejects a checksum-consistent fresh shared manifest'

BOOLEAN_RANK_OUTPUT="${TEST_ROOT}/boolean-rank-output"
expect_failure 'rank_id must be integer zero' \
    "${TEST_ROOT}/boolean-rank.log" \
    task3_env "${BOOLEAN_RANK_OUTPUT}" "${PREBAKED_SOURCE}" \
        gpt175b boolean-rank TASK3_FIXTURE_REPORT_MODE=boolean_rank
[[ ! -e "${BOOLEAN_RANK_OUTPUT}/gpt175b/task3/run_marker.json" ]] || \
    fail 'boolean report rank_id published a Task3 marker'
pass 'Task3 report schema rejects boolean rank_id'

ABSOLUTE_ROOT="${TEST_ROOT}/absolute-path-prebaked"
cp -a -- "${PREBAKED_SOURCE}" "${ABSOLUTE_ROOT}"
python3 - "${ABSOLUTE_ROOT}/distribution_manifest.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["bundles"]["gpt175b"]["root"] = "/absolute/bundle"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
expect_failure 'gpt175b bundle root' \
    "${TEST_ROOT}/absolute-path.log" \
    task3_env "${TEST_ROOT}/absolute-path-output" "${ABSOLUTE_ROOT}" \
        gpt175b absolute-path
pass 'prebaked distribution rejects absolute bundle paths'

RELOCATED_ROOT="${TEST_ROOT}/relocated/deep/tree/prebaked"
mkdir -p "$(dirname -- "${RELOCATED_ROOT}")"
cp -a -- "${PREBAKED_SOURCE}" "${RELOCATED_ROOT}"
RELOCATED_OUTPUT="${TEST_ROOT}/relocated-output"
task3_env "${RELOCATED_OUTPUT}" "${RELOCATED_ROOT}" \
    qwen3_a30b relocated-success \
    >"${TEST_ROOT}/relocated-success.log" 2>&1
RELOCATED_RUN="${RELOCATED_OUTPUT}/qwen3_a30b/task3/runs/relocated-success"
python3 -B "${MANIFEST_TOOL}" verify \
    --root "${RELOCATED_RUN}" \
    --manifest "${RELOCATED_RUN}/artifact_manifest.json" \
    >"${TEST_ROOT}/relocated-manifest.log"
python3 - "${RELOCATED_ROOT}" "${RELOCATED_RUN}" "${PREBAKED_SOURCE}" <<'PY'
import json
import pathlib
import sys

prebaked_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
run_root = pathlib.Path(sys.argv[2]).resolve(strict=True)
old_root = pathlib.Path(sys.argv[3]).resolve(strict=True)
distribution = json.loads(
    (prebaked_root / "distribution_manifest.json").read_text(encoding="utf-8")
)
for entry in distribution["files"]:
    path = pathlib.PurePosixPath(entry["path"])
    assert not path.is_absolute()
    assert all(part not in {"", ".", ".."} for part in path.parts)
for bundle in distribution["bundles"].values():
    for field in ("root", "manifest"):
        path = pathlib.PurePosixPath(bundle[field])
        assert not path.is_absolute()
        assert all(part not in {"", ".", ".."} for part in path.parts)
assert str(old_root) not in json.dumps(distribution, sort_keys=True)
resolved = json.loads(
    (run_root / "provenance/resolved_inputs.json").read_text(encoding="utf-8")
)
assert pathlib.Path(resolved["distribution_manifest"]).resolve(strict=True).parent == prebaked_root
manifest = json.loads((run_root / "artifact_manifest.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    path = pathlib.PurePosixPath(entry["path"])
    assert not path.is_absolute()
    assert all(part not in {"", ".", ".."} for part in path.parts)
PY
pass 'relocated prebaked bundle resolves only portable manifest paths and verifies output'

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 18 ]] || fail "expected 18 cases, got ${PASS_COUNT}"
printf 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification\n'
printf 'EVIDENCE_ROOT=%s\n' "${TEST_ROOT}"
