#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task1-d16.XXXXXX")
PYTHON_BIN=$(command -v python3)
PASS_COUNT=0

# shellcheck source=/dev/null
source "${REPO_ROOT}/SC26-AE/lib/common.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/SC26-AE/lib/task1_trace.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$1"
}

write_fixture() {
    local metadata_path=$1
    local summary_path=$2
    local single_rank_elapsed=${3:-1.25}
    local model=${4:-qwen3_a30b}
    "${PYTHON_BIN}" - "${metadata_path}" "${summary_path}" \
        "${single_rank_elapsed}" "${model}" <<'PY'
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
single_rank_elapsed = float(sys.argv[3])
model = sys.argv[4]
d16_gate_applicable = model in {"qwen3_a30b", "dsv3"}
estimate_rank_count = 256 if d16_gate_applicable else 8
estimated_full = single_rank_elapsed * estimate_rank_count
summary = {
    "selected_rank_count": 4 if d16_gate_applicable else 8,
    "d16_gate_applicable": d16_gate_applicable,
    "estimate_basis_rank": 0,
    "estimate_rank_count": estimate_rank_count,
    "single_rank_elapsed_seconds": single_rank_elapsed,
    "estimated_full_seconds": estimated_full,
}
if d16_gate_applicable:
    summary["fresh_capture_gate_threshold_seconds"] = 7200
    summary["fresh_capture_gate_result"] = (
        "pass" if estimated_full <= 7200 else "prebaked_required"
    )
metadata_path.write_text(
    json.dumps({"model": model, "capture_summary": summary}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)


def render_summary_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


summary_path.write_text(
    "\n".join(f"{key}={render_summary_value(value)}" for key, value in summary.items())
    + "\n",
    encoding="utf-8",
)
PY
}

assert_rejected() {
    local case_name=$1
    local expression=$2
    local expected_error=$3
    local model=${4:-qwen3_a30b}
    local case_root="${TEST_ROOT}/${case_name}"
    local metadata_path="${case_root}/metadata.json"
    local summary_path="${case_root}/summary.log"
    local log_path="${case_root}/validator.log"
    local status

    mkdir -p "${case_root}"
    write_fixture "${metadata_path}" "${summary_path}" 1.25 "${model}"
    "${PYTHON_BIN}" - "${metadata_path}" "${expression}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
summary = payload["capture_summary"]
exec(sys.argv[2], {"payload": payload, "summary": summary})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    set +e
    ae_task1_validate_d16_metadata \
        "${PYTHON_BIN}" "${metadata_path}" "${summary_path}" >"${log_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${case_name} was unexpectedly accepted"
    grep -Fq -- "${expected_error}" "${log_path}" || {
        cat "${log_path}" >&2
        fail "${case_name} did not report ${expected_error}"
    }
    pass "${case_name} fails closed"
}

VALID_ROOT="${TEST_ROOT}/valid"
mkdir -p "${VALID_ROOT}"
write_fixture "${VALID_ROOT}/metadata.json" "${VALID_ROOT}/summary.log"
ae_task1_validate_d16_metadata \
    "${PYTHON_BIN}" "${VALID_ROOT}/metadata.json" "${VALID_ROOT}/summary.log"
pass "valid D16 timing metadata"

PREFLIGHT_ROOT="${TEST_ROOT}/preflight-contract"
mkdir -p "${PREFLIGHT_ROOT}"
PREFLIGHT_INVENTORY="${PREFLIGHT_ROOT}/inventory.json"
PREFLIGHT_BATCH_LOG="${PREFLIGHT_ROOT}/global_batch_size_flags.log"
"${PYTHON_BIN}" - "${PREFLIGHT_INVENTORY}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "trace_file_count": 1,
            "memory_json_count": 1,
            "maximum_peak_allocated_mb": 90.0,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
printf '128\n' >"${PREFLIGHT_BATCH_LOG}"

write_preflight_report() {
    local report_path=$1
    local requested_scope=$2
    local gate_enforced=$3
    local gate_applied=$4
    ae_task1_write_d16_preflight_report \
        "${PYTHON_BIN}" "${report_path}" qwen3_a30b qwen3-contract capture-preflight \
        local_synthetic_not_gpu_qualification 0 "${PREFLIGHT_INVENTORY}" \
        "${PREFLIGHT_BATCH_LOG}" 1.25 "${gate_enforced}" 128 "${requested_scope}" \
        "${gate_applied}"
}

write_preflight_report "${PREFLIGHT_ROOT}/quick-valid.json" quick 0 0
ae_task1_validate_d16_preflight_report \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/quick-valid.json" qwen3_a30b qwen3-contract \
    capture-preflight 0 quick 0
pass "valid QUICK D16 preflight report"

write_preflight_report "${PREFLIGHT_ROOT}/full-valid.json" full 1 1
ae_task1_validate_d16_preflight_report \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/full-valid.json" qwen3_a30b qwen3-contract \
    capture-preflight 1 full 1
pass "valid full D16 preflight report"

assert_preflight_rejected() {
    local case_name=$1
    local mutation=$2
    local expected_error=$3
    local report_path="${PREFLIGHT_ROOT}/${case_name}.json"
    local log_path="${PREFLIGHT_ROOT}/${case_name}.log"
    local status

    write_preflight_report "${report_path}" quick 0 0
    "${PYTHON_BIN}" - "${report_path}" "${mutation}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
exec(sys.argv[2], {"payload": payload})
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    set +e
    ae_task1_validate_d16_preflight_report \
        "${PYTHON_BIN}" "${report_path}" qwen3_a30b qwen3-contract \
        capture-preflight "" "" "" >"${log_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${case_name} preflight report was unexpectedly accepted"
    grep -Fq -- "${expected_error}" "${log_path}" || {
        cat "${log_path}" >&2
        fail "${case_name} did not report ${expected_error}"
    }
    pass "${case_name} preflight report fails closed"
}

assert_preflight_rejected \
    scope-gate-mismatch \
    'payload["requested_capture_scope"] = "full"' \
    'gate enforcement must match requested capture scope'
assert_preflight_rejected \
    rank-id-float \
    'payload["selected_rank_ids"] = [0.0]' \
    'must contain exactly fake rank 0'
assert_preflight_rejected \
    selected-count-float \
    'payload["selected_rank_count"] = 1.0' \
    'must contain exactly fake rank 0'
assert_preflight_rejected \
    estimate-count-float \
    'payload["estimate_rank_count"] = 256.0' \
    'estimate basis/count is invalid'
assert_preflight_rejected \
    nonfinite-rank0 \
    'payload["rank0_elapsed_seconds"] = float("nan")' \
    'rank0_elapsed_seconds must be finite and positive'
assert_preflight_rejected \
    formula-mismatch \
    'payload["estimated_full_seconds"] = 1.0' \
    'estimated_full_seconds formula mismatch'
assert_preflight_rejected \
    wrong-gate-result \
    'payload["fresh_capture_gate_result"] = "prebaked_required"' \
    'gate result is inconsistent'
assert_preflight_rejected \
    wrong-threshold \
    'payload["fresh_capture_gate_threshold_seconds"] = 7200.0' \
    'threshold must be integer 7200'
assert_preflight_rejected \
    automatic-fallback \
    'payload["automatic_fallback"] = True' \
    'automatic_fallback must be false'

INVALID_BATCH="${PREFLIGHT_ROOT}/invalid-batch.log"
printf '128,256\n' >"${INVALID_BATCH}"
set +e
ae_task1_write_d16_preflight_report \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/invalid-batch.json" qwen3_a30b \
    qwen3-contract capture-preflight local_synthetic_not_gpu_qualification 0 \
    "${PREFLIGHT_INVENTORY}" "${INVALID_BATCH}" 1.25 0 128 quick 0 \
    >"${PREFLIGHT_ROOT}/invalid-batch.log.out" 2>&1
invalid_batch_status=$?
set -e
[[ ${invalid_batch_status} -ne 0 ]] || fail "invalid batch provenance was unexpectedly accepted"
grep -Fq -- "batch-size flags conflict" "${PREFLIGHT_ROOT}/invalid-batch.log.out" || {
    cat "${PREFLIGHT_ROOT}/invalid-batch.log.out" >&2
    fail "invalid batch provenance error was not reported"
}
pass "invalid batch provenance fails closed"

PREFLIGHT_COPY="${PREFLIGHT_ROOT}/copied/d16_preflight.json"
copy_digest=$(ae_task1_copy_preflight_report \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/quick-valid.json" "${PREFLIGHT_COPY}")
expected_copy_digest=$(sha256sum "${PREFLIGHT_COPY}" | awk '{print $1}')
[[ "${copy_digest}" == "${expected_copy_digest}" ]] || \
    fail "preflight copy digest did not match copied bytes"
binding_digest=$(ae_task1_validate_preflight_binding \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/quick-valid.json" "${PREFLIGHT_COPY}" \
    qwen3_a30b qwen3-contract capture-preflight quick 0 0 1.25)
[[ "${binding_digest}" == "${expected_copy_digest}" ]] || \
    fail "preflight binding digest did not match copied bytes"
pass "preflight provenance copy and binding hash match"

printf '\n' >>"${PREFLIGHT_COPY}"
set +e
ae_task1_validate_preflight_binding \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/quick-valid.json" "${PREFLIGHT_COPY}" \
    qwen3_a30b qwen3-contract capture-preflight quick 0 0 1.25 \
    >"${PREFLIGHT_ROOT}/copy-tamper.log" 2>&1
copy_tamper_status=$?
set -e
[[ ${copy_tamper_status} -ne 0 ]] || fail "tampered preflight copy was unexpectedly accepted"
grep -Fq -- "copy does not match source bytes" "${PREFLIGHT_ROOT}/copy-tamper.log" || {
    cat "${PREFLIGHT_ROOT}/copy-tamper.log" >&2
    fail "tampered preflight copy error was not reported"
}
pass "tampered preflight provenance copy fails closed"

mkdir -p "${PREFLIGHT_ROOT}/symlink-target"
ln -s "${PREFLIGHT_ROOT}/symlink-target" "${PREFLIGHT_ROOT}/symlink-parent"
set +e
ae_task1_copy_preflight_report \
    "${PYTHON_BIN}" "${PREFLIGHT_ROOT}/quick-valid.json" \
    "${PREFLIGHT_ROOT}/symlink-parent/nested/d16_preflight.json" \
    >"${PREFLIGHT_ROOT}/symlink-parent.log" 2>&1
symlink_copy_status=$?
set -e
[[ ${symlink_copy_status} -ne 0 ]] || fail "intermediate symlink preflight parent was accepted"
grep -Fq -- "symlink" "${PREFLIGHT_ROOT}/symlink-parent.log" || {
    cat "${PREFLIGHT_ROOT}/symlink-parent.log" >&2
    fail "intermediate symlink preflight parent error was not reported"
}
pass "intermediate symlink preflight parent fails closed"

PREFLIGHT_MANIFEST="${PREFLIGHT_ROOT}/manifest.json"
cat >"${PREFLIGHT_MANIFEST}" <<'JSON'
{"files":[{"path":"provenance/d16_preflight.json"}]}
JSON
ae_task1_validate_manifest_preflight_isolation "${PYTHON_BIN}" "${PREFLIGHT_MANIFEST}"
pass "manifest accepts the sole copied preflight artifact"

cat >"${PREFLIGHT_MANIFEST}" <<'JSON'
{"files":[{"path":"provenance/d16_preflight.json"}],"files":[{"path":"provenance/d16_preflight.json"}]}
JSON
set +e
ae_task1_validate_manifest_preflight_isolation \
    "${PYTHON_BIN}" "${PREFLIGHT_MANIFEST}" \
    >"${PREFLIGHT_ROOT}/manifest-duplicate.log" 2>&1
manifest_duplicate_status=$?
set -e
[[ ${manifest_duplicate_status} -ne 0 ]] || fail "duplicate manifest keys were unexpectedly accepted"
grep -Fq -- "duplicate JSON key" "${PREFLIGHT_ROOT}/manifest-duplicate.log" || {
    cat "${PREFLIGHT_ROOT}/manifest-duplicate.log" >&2
    fail "duplicate manifest key error was not reported"
}
pass "duplicate manifest keys fail closed"

cat >"${PREFLIGHT_MANIFEST}" <<'JSON'
{"files":[{"path":"provenance/d16_preflight.json"},{"path":"provenance/d16_preflight.json"}]}
JSON
set +e
ae_task1_validate_manifest_preflight_isolation \
    "${PYTHON_BIN}" "${PREFLIGHT_MANIFEST}" \
    >"${PREFLIGHT_ROOT}/manifest-repeat.log" 2>&1
manifest_repeat_status=$?
set -e
[[ ${manifest_repeat_status} -ne 0 ]] || fail "duplicate preflight manifest path was unexpectedly accepted"
grep -Fq -- "exactly one" "${PREFLIGHT_ROOT}/manifest-repeat.log" || {
    cat "${PREFLIGHT_ROOT}/manifest-repeat.log" >&2
    fail "duplicate preflight manifest path error was not reported"
}
pass "duplicate preflight manifest path fails closed"

assert_gate_result() {
    local case_name=$1
    local elapsed_seconds=$2
    local expected_result=$3
    local output_path="${TEST_ROOT}/gate-${case_name}.log"
    local status

    set +e
    ae_task1_d16_gate_result "${elapsed_seconds}" >"${output_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -eq 0 ]] || {
        cat "${output_path}" >&2
        fail "${case_name} gate helper failed with status ${status}"
    }
    [[ "$(cat "${output_path}")" == "${expected_result}" ]] || {
        cat "${output_path}" >&2
        fail "${case_name} gate result mismatch"
    }
    pass "${case_name} D16 gate result"
}

assert_gate_result exact-threshold 7200 pass
assert_gate_result above-threshold 7200.000001 prebaked_required

GPT_ROOT="${TEST_ROOT}/valid-gpt"
mkdir -p "${GPT_ROOT}"
write_fixture "${GPT_ROOT}/metadata.json" "${GPT_ROOT}/summary.log" 1.25 gpt175b
ae_task1_validate_d16_metadata \
    "${PYTHON_BIN}" "${GPT_ROOT}/metadata.json" "${GPT_ROOT}/summary.log"
grep -Fq 'd16_gate_applicable=false' "${GPT_ROOT}/summary.log" || \
    fail "GPT timing did not record d16_gate_applicable=false"
grep -Fq 'estimate_rank_count=8' "${GPT_ROOT}/summary.log" || \
    fail "GPT timing did not use the eight representative ranks"
grep -Fq 'estimated_full_seconds=10.0' "${GPT_ROOT}/summary.log" || \
    fail "GPT selected-capture estimate did not equal rank0 timing times eight"
if grep -Fq 'fresh_capture_gate_' "${GPT_ROOT}/summary.log"; then
    fail "GPT timing unexpectedly recorded a D16 fresh-capture gate"
fi
pass "GPT timing remains diagnostic and D16-inapplicable"

THRESHOLD_ROOT="${TEST_ROOT}/threshold"
mkdir -p "${THRESHOLD_ROOT}"
write_fixture "${THRESHOLD_ROOT}/metadata.json" "${THRESHOLD_ROOT}/summary.log" 28.125
ae_task1_validate_d16_metadata \
    "${PYTHON_BIN}" "${THRESHOLD_ROOT}/metadata.json" "${THRESHOLD_ROOT}/summary.log"
grep -Fq 'fresh_capture_gate_result=pass' "${THRESHOLD_ROOT}/summary.log" || \
    fail "exact threshold did not pass"
pass "exact 7200-second threshold remains pass"

PREBAKED_ROOT="${TEST_ROOT}/prebaked-required"
mkdir -p "${PREBAKED_ROOT}"
write_fixture "${PREBAKED_ROOT}/metadata.json" "${PREBAKED_ROOT}/summary.log" 28.125000001
ae_task1_validate_d16_metadata \
    "${PYTHON_BIN}" "${PREBAKED_ROOT}/metadata.json" "${PREBAKED_ROOT}/summary.log"
grep -Fq 'fresh_capture_gate_result=prebaked_required' \
    "${PREBAKED_ROOT}/summary.log" || fail "above-threshold estimate did not require prebaked"
pass "estimate above 7200 seconds requires prebaked"

assert_rejected \
    missing-d16-applicability \
    'summary.pop("d16_gate_applicable")' \
    'missing d16_gate_applicable'
assert_rejected \
    nonboolean-d16-applicability \
    'summary["d16_gate_applicable"] = "true"' \
    'd16_gate_applicable must be boolean'
assert_rejected \
    missing-single-rank \
    'summary.pop("single_rank_elapsed_seconds")' \
    'missing single_rank_elapsed_seconds'
assert_rejected \
    nonnumeric-single-rank \
    'summary["single_rank_elapsed_seconds"] = "1.25"' \
    'single_rank_elapsed_seconds must be numeric'
assert_rejected \
    zero-single-rank \
    'summary["single_rank_elapsed_seconds"] = 0' \
    'single_rank_elapsed_seconds must be finite and strictly positive'
assert_rejected \
    negative-single-rank \
    'summary["single_rank_elapsed_seconds"] = -1' \
    'single_rank_elapsed_seconds must be finite and strictly positive'
assert_rejected \
    mismatched-estimate \
    'summary["estimated_full_seconds"] = 319.0' \
    'estimated_full_seconds does not equal single_rank_elapsed_seconds * estimate_rank_count'
assert_rejected \
    missing-threshold \
    'summary.pop("fresh_capture_gate_threshold_seconds")' \
    'missing fresh_capture_gate_threshold_seconds'
assert_rejected \
    missing-gate-result \
    'summary.pop("fresh_capture_gate_result")' \
    'missing fresh_capture_gate_result'
assert_rejected \
    noninteger-threshold \
    'summary["fresh_capture_gate_threshold_seconds"] = 7200.0' \
    'fresh_capture_gate_threshold_seconds must be integer 7200'
assert_rejected \
    negative-threshold \
    'summary["fresh_capture_gate_threshold_seconds"] = -1' \
    'fresh_capture_gate_threshold_seconds must be integer 7200'
assert_rejected \
    wrong-gate-result \
    'summary["fresh_capture_gate_result"] = "prebaked_required"' \
    'fresh_capture_gate_result is inconsistent'
assert_rejected \
    wrong-basis-rank \
    'summary["estimate_basis_rank"] = 1' \
    'estimate_basis_rank must be integer 0'
assert_rejected \
    wrong-rank-count \
    'summary["estimate_rank_count"] = 255' \
    'estimate_rank_count must be integer 256'
assert_rejected \
    gpt-wrong-rank-count \
    'summary["estimate_rank_count"] = 256; summary["estimated_full_seconds"] = summary["single_rank_elapsed_seconds"] * 256' \
    'GPT estimate_rank_count must equal the eight selected representative ranks' \
    gpt175b
assert_rejected \
    gpt-gate-fields-present \
    'summary["fresh_capture_gate_threshold_seconds"] = 7200; summary["fresh_capture_gate_result"] = "pass"' \
    'GPT timing metadata must omit D16 gate threshold and result fields' \
    gpt175b
assert_rejected \
    moe-not-applicable \
    'summary["d16_gate_applicable"] = False' \
    'MoE Task1 timing requires d16_gate_applicable=true'
assert_rejected \
    unknown-model \
    'payload["model"] = "unknown"' \
    'Task1 D16 metadata has unsupported model'

assert_rank_timing_rejected() {
    local case_name=$1
    local selected_ranks=$2
    local timing_text=$3
    local expected_error=$4
    local case_root="${TEST_ROOT}/timing-${case_name}"
    local timing_path="${case_root}/rank_timings.log"
    local log_path="${case_root}/validator.log"
    local status

    mkdir -p "${case_root}"
    printf '%s' "${timing_text}" >"${timing_path}"
    set +e
    ae_task1_rank0_elapsed_seconds \
        "${PYTHON_BIN}" "${timing_path}" "${selected_ranks}" >"${log_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${case_name} timing log was unexpectedly accepted"
    grep -Fq -- "${expected_error}" "${log_path}" || {
        cat "${log_path}" >&2
        fail "${case_name} did not report ${expected_error}"
    }
    pass "${case_name} rank timing fails closed"
}

VALID_TIMING="${TEST_ROOT}/valid-rank-timings.log"
printf '0,100,1000000100,1000000000\n64,200,1000000200,1000000000\n' \
    >"${VALID_TIMING}"
[[ "$(ae_task1_rank0_elapsed_seconds "${PYTHON_BIN}" "${VALID_TIMING}" '0,64')" == \
    "1.000000000" ]] || fail "valid rank-0 timing value did not match 1.000000000"
pass "rank-0 elapsed time is derived from the measured interval"

assert_rank_timing_rejected \
    malformed \
    '0' \
    $'0,100,200\n' \
    'Invalid Task1 rank timing line'
assert_rank_timing_rejected \
    duplicate-rank \
    '0' \
    $'0,100,200,100\n0,300,400,100\n' \
    'Duplicate Task1 rank timing'
assert_rank_timing_rejected \
    zero-interval \
    '0' \
    $'0,100,100,0\n' \
    'Invalid Task1 rank timing interval'
assert_rank_timing_rejected \
    inconsistent-interval \
    '0' \
    $'0,100,300,100\n' \
    'Invalid Task1 rank timing interval'
assert_rank_timing_rejected \
    inventory-mismatch \
    '0,64' \
    $'0,100,200,100\n' \
    'Task1 rank timing inventory mismatch'
assert_rank_timing_rejected \
    missing-rank-zero \
    '64' \
    $'64,100,200,100\n' \
    'Task1 timing estimate requires fake rank 0 timing'

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 49 ]] || fail "expected 49 cases, got ${PASS_COUNT}"
