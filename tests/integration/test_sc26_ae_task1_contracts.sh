#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TASK1_LIB="${REPO_ROOT}/SC26-AE/lib/task1_trace.sh"
COMMON_LIB="${REPO_ROOT}/SC26-AE/lib/common.sh"
MANIFEST_TOOL="${REPO_ROOT}/SC26-AE/tools/artifact_manifest.py"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task1-contracts.XXXXXX")
FAKE_BIN="${TEST_ROOT}/bin"
TORCHRUN_LOG="${TEST_ROOT}/torchrun.log"
NSYS_LOG="${TEST_ROOT}/nsys.log"
ADAPTER_SINK_LOG="${TEST_ROOT}/adapter-sink.log"
ADAPTER_TIMING_LOG="${TEST_ROOT}/adapter-rank-timings.log"
PASS_COUNT=0

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf 'PASS: %s\n' "$1"
}

assert_contains() {
    local needle=$1
    local path=$2
    grep -Fq -- "${needle}" "${path}" || {
        printf 'Expected %s to contain: %s\n' "${path}" "${needle}" >&2
        cat "${path}" >&2
        exit 1
    }
}

assert_not_contains() {
    local needle=$1
    local path=$2
    if grep -Fq -- "${needle}" "${path}"; then
        printf 'Expected %s to omit: %s\n' "${path}" "${needle}" >&2
        cat "${path}" >&2
        exit 1
    fi
}

assert_equals() {
    local expected=$1
    local actual=$2
    local context=$3
    [[ "${actual}" == "${expected}" ]] || fail "${context}: expected '${expected}', got '${actual}'"
}

mkdir -p "${FAKE_BIN}"
: > "${TORCHRUN_LOG}"
: > "${NSYS_LOG}"
: > "${ADAPTER_SINK_LOG}"
: > "${ADAPTER_TIMING_LOG}"

cat > "${FAKE_BIN}/torchrun" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

{
    printf 'CWD=%q' "${PWD}"
    for argument in "$@"; do
        printf ' %q' "${argument}"
    done
    printf '\n'
} >> "${FAKE_TORCHRUN_LOG}"

rank_id=""
previous=""
for argument in "$@"; do
    if [[ "${previous}" == "--fake-current-rank-id" ]]; then
        rank_id=${argument}
        break
    fi
    previous=${argument}
done
[[ "${rank_id}" =~ ^[0-9]+$ ]] || {
    printf 'fake torchrun did not receive a rank id\n' >&2
    exit 91
}

if [[ "${FAKE_TORCHRUN_EXIT:-0}" != "0" ]]; then
    exit "${FAKE_TORCHRUN_EXIT}"
fi

mkdir -p "${PWD}/profiler_log/fake" "${PWD}/memory_traces_scaling"
trace_mode=${FAKE_TRACE_MODE:-valid}
case "${trace_mode}" in
    valid|missing_forward_step|missing_backward_step|missing_optimizer_step|\
        missing_backward_cmd_uid|missing_backward_timestamp|missing_backward_duration|\
        missing_backward_mg_state|missing_backward_stage_id|missing_backward_batch_id|\
        missing_ddp_trigger)
        ;;
    *)
        printf 'unsupported fake trace mode: %s\n' "${trace_mode}" >&2
        exit 93
        ;;
esac

memory_mode=${FAKE_MEMORY_MODE:-valid}
case "${memory_mode}" in
    valid|empty_payload|empty_samples|nan_peak|inf_peak|zero_peak|negative_reserved|\
        negative_allocated|all_zero_samples|missing_rank|duplicate_rank)
        ;;
    *)
        printf 'unsupported fake memory mode: %s\n' "${memory_mode}" >&2
        exit 94
        ;;
esac

trace_path="${PWD}/profiler_log/fake/trace_rank${rank_id}_20260718000000.txt"
: > "${trace_path}"
if [[ "${trace_mode}" != "missing_forward_step" ]]; then
    printf 'rank:%s:forward_step(stage_id=0,batch_id=0,mg_state=steady,duration=4.000,timestamp=100.000,cmd_uid=fwd-%s)\n' \
        "${rank_id}" "${rank_id}" >> "${trace_path}"
fi
case "${trace_mode}" in
    missing_backward_step)
        ;;
    missing_backward_cmd_uid)
        printf 'rank:%s:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=6.000,timestamp=104.000)\n' \
            "${rank_id}" >> "${trace_path}"
        ;;
    missing_backward_timestamp)
        printf 'rank:%s:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=6.000,cmd_uid=bwd-%s)\n' \
            "${rank_id}" "${rank_id}" >> "${trace_path}"
        ;;
    missing_backward_duration)
        printf 'rank:%s:backward_step(stage_id=0,batch_id=0,mg_state=steady,timestamp=104.000,cmd_uid=bwd-%s)\n' \
            "${rank_id}" "${rank_id}" >> "${trace_path}"
        ;;
    missing_backward_mg_state)
        printf 'rank:%s:backward_step(stage_id=0,batch_id=0,duration=6.000,timestamp=104.000,cmd_uid=bwd-%s)\n' \
            "${rank_id}" "${rank_id}" >> "${trace_path}"
        ;;
    missing_backward_stage_id)
        printf 'rank:%s:backward_step(batch_id=0,mg_state=steady,duration=6.000,timestamp=104.000,cmd_uid=bwd-%s)\n' \
            "${rank_id}" "${rank_id}" >> "${trace_path}"
        ;;
    missing_backward_batch_id)
        printf 'rank:%s:backward_step(stage_id=0,mg_state=steady,duration=6.000,timestamp=104.000,cmd_uid=bwd-%s)\n' \
            "${rank_id}" "${rank_id}" >> "${trace_path}"
        ;;
    *)
        printf 'rank:%s:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=6.000,timestamp=104.000,cmd_uid=bwd-%s)\n' \
            "${rank_id}" "${rank_id}" >> "${trace_path}"
        ;;
esac
if [[ "${trace_mode}" == "missing_ddp_trigger" ]]; then
    printf 'rank:%s:ddp_grad_comm(stage_id=0,batch_id=0,mg_state=steady,duration=0.500,timestamp=105.000)\n' \
        "${rank_id}" >> "${trace_path}"
else
    printf 'rank:%s:ddp_grad_comm(stage_id=0,batch_id=0,mg_state=steady,duration=0.500,timestamp=105.000,trigger_cmd_uid=bwd-%s)\n' \
        "${rank_id}" "${rank_id}" >> "${trace_path}"
fi
if [[ "${trace_mode}" != "missing_optimizer_step" ]]; then
    printf 'rank:%s:optimizer_step(stage_id=0,batch_id=0,mg_state=finalize,duration=1.500,timestamp=110.000,cmd_uid=opt-%s)\n' \
        "${rank_id}" "${rank_id}" >> "${trace_path}"
fi
if [[ "${memory_mode}" == "missing_rank" && "${rank_id}" == "0" ]]; then
    exit 0
fi
memory_rank=${rank_id}
if [[ "${memory_mode}" == "duplicate_rank" ]]; then
    memory_rank=64
fi
memory_path="${PWD}/memory_traces_scaling/memory_trace_rank${memory_rank}_fixture.json"
case "${memory_mode}" in
    valid)
        cat > "${memory_path}" <<JSON
{
  "0": {
    "samples": [
      {"timestamp_s": 0.1, "reserved_memory_MB": 100.0, "allocated_memory_MB": 80.0}
    ],
    "peak_allocated_MB": 90.0,
    "theoretical_memory_MB": 120.0
  }
}
JSON
        ;;
    empty_payload)
        printf '{}\n' > "${memory_path}"
        ;;
    empty_samples)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[],"peak_allocated_MB":90.0}}
JSON
        ;;
    nan_peak)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[{"reserved_memory_MB":100.0,"allocated_memory_MB":80.0}],"peak_allocated_MB":NaN}}
JSON
        ;;
    inf_peak)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[{"reserved_memory_MB":100.0,"allocated_memory_MB":80.0}],"peak_allocated_MB":Infinity}}
JSON
        ;;
    zero_peak)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[{"reserved_memory_MB":100.0,"allocated_memory_MB":80.0}],"peak_allocated_MB":0.0}}
JSON
        ;;
    negative_reserved)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[{"reserved_memory_MB":-1.0,"allocated_memory_MB":80.0}],"peak_allocated_MB":90.0}}
JSON
        ;;
    negative_allocated)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[{"reserved_memory_MB":100.0,"allocated_memory_MB":-1.0}],"peak_allocated_MB":90.0}}
JSON
        ;;
    all_zero_samples)
        cat > "${memory_path}" <<JSON
{"0":{"samples":[{"reserved_memory_MB":0.0,"allocated_memory_MB":0.0}],"peak_allocated_MB":90.0}}
JSON
        ;;
    missing_rank|duplicate_rank)
        cat > "${memory_path}" <<JSON
{
  "0": {
    "samples": [
      {"timestamp_s": 0.1, "reserved_memory_MB": 100.0, "allocated_memory_MB": 80.0}
    ],
    "peak_allocated_MB": 90.0,
    "theoretical_memory_MB": 120.0
  }
}
JSON
        ;;
esac
SH
chmod +x "${FAKE_BIN}/torchrun"

cat > "${FAKE_BIN}/torchrun-sink" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_ADAPTER_SINK_LOG}"
printf '\n' >> "${FAKE_ADAPTER_SINK_LOG}"
SH
chmod +x "${FAKE_BIN}/torchrun-sink"

cat > "${FAKE_BIN}/nsys" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_NSYS_LOG}"
printf '\n' >> "${FAKE_NSYS_LOG}"

subcommand=$1
shift
case "${subcommand}" in
    profile)
        output_base=""
        while (($#)); do
            case "$1" in
                --output)
                    output_base=$2
                    shift 2
                    ;;
                bash)
                    bash "$2"
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done
        [[ -n "${output_base}" ]]
        printf 'fake nsys report\n' > "${output_base}.nsys-rep"
        ;;
    export)
        output_path=""
        input_path=""
        while (($#)); do
            case "$1" in
                -o)
                    output_path=$2
                    shift 2
                    ;;
                -t|--force-overwrite)
                    shift 2
                    ;;
                *)
                    input_path=$1
                    shift
                    ;;
            esac
        done
        [[ -s "${input_path}" && -n "${output_path}" ]]
        printf 'fake sqlite\n' > "${output_path}"
        ;;
    *)
        printf 'unsupported fake nsys subcommand: %s\n' "${subcommand}" >&2
        exit 92
        ;;
esac
SH
chmod +x "${FAKE_BIN}/nsys"

cat > "${FAKE_BIN}/date" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "+%s%N" && "${FAKE_D16_ABOVE_THRESHOLD:-0}" == "1" &&
    "${PWD}" == *"/task1-preflight."* ]]; then
    state_file=${FAKE_D16_DATE_STATE:?FAKE_D16_DATE_STATE is required}
    if [[ ! -e "${state_file}" ]]; then
        printf '1000000000\n'
        : > "${state_file}"
    else
        # Thirty thousand seconds exceeds the fixed 7200-second gate while
        # remaining within signed 64-bit shell arithmetic.
        printf '30001000000000\n'
    fi
    exit 0
fi
exec /bin/date "$@"
SH
chmod +x "${FAKE_BIN}/date"

# shellcheck source=/dev/null
source "${COMMON_LIB}"
# shellcheck source=/dev/null
source "${TASK1_LIB}"

ADAPTER_ROOT="${TEST_ROOT}/adapter-contract"
ADAPTER_PATH="${ADAPTER_ROOT}/torchrun"
ADAPTER_BATCH_LOG="${ADAPTER_ROOT}/global_batch_size_flags.log"
mkdir -p "${ADAPTER_ROOT}"
ae_task1_write_torchrun_adapter "${ADAPTER_PATH}"

TOOLCHAIN_BIN="${TEST_ROOT}/megatron-env/bin"
RANK_LOOP_PATH="${TEST_ROOT}/rank-loop-path-contract.sh"
mkdir -p "${TOOLCHAIN_BIN}"
: > "${TOOLCHAIN_BIN}/torchrun"
chmod +x "${TOOLCHAIN_BIN}/torchrun"
AE_TASK1_REAL_TORCHRUN="${TOOLCHAIN_BIN}/torchrun"
AE_T1_PROFILE=full
AE_T1_WORLD_SIZE=256
AE_T1_PP=8
AE_T1_TP=8
AE_T1_DP=4
AE_T1_EXP=4
AE_T1_MICRO_BATCH_SIZE=1
AE_T1_NUM_MICROBATCHES=32
AE_T1_GLOBAL_BATCH_SIZE=128
AE_T1_SEQ_LEN=256
AE_T1_TRANSFORMER_IMPL=transformer_engine
ae_task1_write_rank_loop \
    "${RANK_LOOP_PATH}" "${TEST_ROOT}/runtime" "${ADAPTER_ROOT}" \
    "${REPO_ROOT}/examples/pretrain_qwen3_30b_a3b_moe.sh" \
    qwen3_a30b "0" 0 qwen3_a30b-path-contract \
    "${ADAPTER_BATCH_LOG}" "${ADAPTER_TIMING_LOG}"
assert_contains "${ADAPTER_ROOT}:${TOOLCHAIN_BIN}:" "${RANK_LOOP_PATH}"
pass "Task1 rank loop binds helper builds to the real torchrun environment"

assert_adapter_batch_failure() {
    local case_name=$1
    local expected_error=$2
    shift 2
    local before_calls after_calls status
    before_calls=$(wc -l < "${ADAPTER_SINK_LOG}")
    set +e
    AE_TASK1_REAL_TORCHRUN="${FAKE_BIN}/torchrun-sink" \
    AE_TASK1_GLOBAL_BATCH_LOG="${ADAPTER_BATCH_LOG}" \
    AE_TASK1_RANK_TIMING_LOG="${ADAPTER_TIMING_LOG}" \
    FAKE_ADAPTER_SINK_LOG="${ADAPTER_SINK_LOG}" \
        "${ADAPTER_PATH}" "$@" \
        > "${ADAPTER_ROOT}/${case_name}.stdout" \
        2> "${ADAPTER_ROOT}/${case_name}.stderr"
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${case_name} batch contract returned success"
    after_calls=$(wc -l < "${ADAPTER_SINK_LOG}")
    assert_equals "${before_calls}" "${after_calls}" "${case_name} real torchrun calls"
    assert_contains "${expected_error}" "${ADAPTER_ROOT}/${case_name}.stderr"
    [[ ! -s "${ADAPTER_BATCH_LOG}" ]] || fail "${case_name} wrote batch provenance after rejection"
    pass "${case_name} batch contract fails before real torchrun"
}

assert_adapter_batch_failure \
    missing \
    "Missing --global-batch-size" \
    pretrain_llama.py
assert_adapter_batch_failure \
    malformed \
    "--global-batch-size must be a positive integer" \
    pretrain_llama.py --global-batch-size invalid
assert_adapter_batch_failure \
    conflicting \
    "Conflicting --global-batch-size values" \
    pretrain_llama.py --global-batch-size 128 --global-batch-size 256

assert_equals "0,128,256,384,512,640,768,896" "$(ae_task1_selected_ranks gpt175b 0)" "GPT full ranks"
assert_equals "0,128,256,384,512,640,768,896" "$(ae_task1_selected_ranks gpt175b 1)" "GPT QUICK ranks"
assert_equals "0,8,16,24" "$(ae_task1_selected_ranks qwen3_a30b 1)" "Qwen QUICK ranks"
assert_equals "0,64,128,192" "$(ae_task1_selected_ranks dsv3 1)" "DSV3 QUICK ranks"
qwen_full=$(ae_task1_selected_ranks qwen3_a30b 0)
assert_equals "32" "$(awk -F, '{print NF}' <<< "${qwen_full}")" "Qwen representative rank count"
assert_equals "0" "${qwen_full%%,*}" "Qwen first full rank"
assert_equals "248" "${qwen_full##*,}" "Qwen last representative rank"
assert_equals "representative" "$(ae_task1_capture_scope gpt175b 0)" "GPT representative capture scope"
assert_equals "representative" "$(ae_task1_capture_scope gpt175b 1)" "GPT QUICK capture scope"
assert_equals "quick" "$(ae_task1_capture_scope qwen3_a30b 1)" "Qwen QUICK capture scope"
assert_equals "representative_ep" "$(ae_task1_capture_scope qwen3_a30b 0)" "Qwen representative capture scope"
assert_equals "quick" "$(ae_task1_capture_scope dsv3 1)" "DSV3 QUICK capture scope"
assert_equals "full" "$(ae_task1_capture_scope dsv3 0)" "DSV3 full capture scope"
pass "frozen full/QUICK rank scopes and capture scopes"

assert_equals "local_synthetic_not_gpu_qualification" "$(ae_task1_execution_evidence 1)" \
    "Task1 test-mode evidence class"
assert_equals \
    "runtime_measurement_requires_external_single_gpu_qualification" \
    "$(ae_task1_execution_evidence 0)" \
    "Task1 real-mode evidence remains pending until external sealing"
pass "Task1 evidence class is explicit and never self-promotes"

assert_invalid_gpu_selector() {
    local case_name=$1
    local scale_gpu=$2
    local visible_gpu=$3
    local expected_error=$4
    local output_path="${TEST_ROOT}/${case_name}.log"
    local status
    set +e
    SCALE_GPU="${scale_gpu}" CUDA_VISIBLE_DEVICES="${visible_gpu}" \
        ae_task1_resolve_gpu >"${output_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${case_name} GPU selector was unexpectedly accepted"
    assert_contains "${expected_error}" "${output_path}"
}

assert_invalid_gpu_selector comma-scale-gpu "0,1" "" "select exactly one GPU"
assert_invalid_gpu_selector nondecimal-scale-gpu foo "" "decimal GPU index"
assert_invalid_gpu_selector nondecimal-visible-gpu "" foo "decimal GPU index"
assert_equals "7" "$(SCALE_GPU= CUDA_VISIBLE_DEVICES=7 ae_task1_resolve_gpu)" \
    "single decimal CUDA_VISIBLE_DEVICES selector"
pass "Task1 rejects non-decimal GPU selectors before workload execution"

run_entry() {
    local model=$1
    local capture_nsys=$2
    local output_root="${TEST_ROOT}/output-${model}"
    local capture_id="${model}-20260718T00000${PASS_COUNT}Z"
    local preflight_id="${capture_id}-preflight"
    local entry="${REPO_ROOT}/SC26-AE/task1_${model}.sh"

    PATH="${FAKE_BIN}:${PATH}" \
    AE_OUTPUT_ROOT="${output_root}" \
    AE_TASK1_TEST_MODE=1 \
    AE_TASK1_TEST_CAPTURE_ID="${capture_id}" \
    AE_TASK1_TEST_PREFLIGHT_CAPTURE_ID="${preflight_id}" \
    AE_TASK1_TORCHRUN="${FAKE_BIN}/torchrun" \
    AE_MEGATRON_PYTHON="$(command -v python3)" \
    AE_NSYS_BIN="${FAKE_BIN}/nsys" \
    FAKE_TORCHRUN_LOG="${TORCHRUN_LOG}" \
    FAKE_NSYS_LOG="${NSYS_LOG}" \
    QUICK=1 \
    SCALE_GPU=0 \
    CAPTURE_NSYS="${capture_nsys}" \
    bash "${entry}" > "${TEST_ROOT}/${model}.log" 2>&1

    local task_dir="${output_root}/${model}/task1"
    local run_root="${task_dir}/runs/${capture_id}"
    local preflight_root="${output_root}/_work/task1-preflight.${preflight_id}"
    local preflight_report="${preflight_root}/preflight_result.json"
    local marker="${task_dir}/capture_marker.json"
    local manifest="${run_root}/artifact_manifest.json"
    [[ -s "${marker}" ]] || fail "missing marker for ${model}"
    [[ -s "${manifest}" ]] || fail "missing manifest for ${model}"
    if [[ "${model}" == "gpt175b" ]]; then
        [[ ! -e "${preflight_root}" ]] || fail "GPT unexpectedly ran a D16 preflight"
    else
        [[ -s "${preflight_report}" ]] || fail "missing independent preflight report for ${model}"
        [[ -s "${run_root}/provenance/d16_preflight.json" ]] || \
            fail "full bundle did not preserve the preflight provenance artifact for ${model}"
        preflight_calls=$(grep -F "/output-${model}/_work/task1-preflight.${preflight_id}/" "${TORCHRUN_LOG}" | wc -l)
        assert_equals "1" "${preflight_calls}" "${model} independent rank-0 preflight invocation count"
        preflight_rank=$(grep -F "/output-${model}/_work/task1-preflight.${preflight_id}/" "${TORCHRUN_LOG}" | \
            grep -o -- '--fake-current-rank-id [0-9][0-9]*' | head -1 | awk '{print $2}')
        assert_equals "0" "${preflight_rank}" "${model} preflight rank id"
    fi
    PYTHONDONTWRITEBYTECODE=1 "$(command -v python3)" -B "${MANIFEST_TOOL}" verify \
        --root "${run_root}" --manifest "${manifest}" \
        > "${TEST_ROOT}/${model}-verify.log"

    "$(command -v python3)" - "${model}" "${capture_id}" "${task_dir}" "${preflight_report}" <<'PY'
import hashlib
import json
import pathlib
import sys

model, capture_id, task_dir_text, preflight_report_text = sys.argv[1:]
task_dir = pathlib.Path(task_dir_text)
preflight_report_path = pathlib.Path(preflight_report_text)
marker = json.loads((task_dir / "capture_marker.json").read_text(encoding="utf-8"))
run_root = task_dir / "runs" / capture_id
manifest_path = run_root / "artifact_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_files = {entry["path"] for entry in manifest["files"]}
expected_counts = {"gpt175b": 8, "qwen3_a30b": 4, "dsv3": 4}
expected_profiles = {"gpt175b": "175", "qwen3_a30b": "full", "dsv3": "smoke"}
expected_world_sizes = {"gpt175b": 1024, "qwen3_a30b": 256, "dsv3": 256}
expected_fake_node_sizes = {"gpt175b": 8, "qwen3_a30b": 256, "dsv3": 256}
expected_capture_scopes = {"gpt175b": "representative", "qwen3_a30b": "quick", "dsv3": "quick"}
expected_global_batch_flags = {
    "gpt175b": [768],
    "qwen3_a30b": [128, 128],
    "dsv3": [128, 128],
}
expected_timing_multipliers = {"gpt175b": 8, "qwen3_a30b": 256, "dsv3": 256}
expected_d16_applicability = {"gpt175b": False, "qwen3_a30b": True, "dsv3": True}

assert marker["schema_version"] == "sc26-ae-task1-capture-marker-v1"
assert marker["model"] == model
assert marker["capture_id"] == capture_id
assert marker["verified"] is True
assert marker["run_path"] == f"runs/{capture_id}"
assert marker["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert marker["artifact_manifest_sha256"] == marker["manifest_sha256"]
assert manifest["model"] == model
assert manifest["task"] == "task1"
assert manifest["artifact_source"] == "fresh"
assert manifest["capture_id"] == capture_id
assert manifest["profile"] == expected_profiles[model]
assert manifest["precision"] == "bf16"
assert manifest["mock_data"] is True
assert manifest["ddp_overlap"] is True
assert manifest["execution_evidence"] == "local_synthetic_not_gpu_qualification"
assert manifest["simulation_topology"]["world_size"] == expected_world_sizes[model]
assert manifest["simulation_topology"]["local_size"] == 8
assert manifest["capture_runtime"]["physical_gpu_count"] == 1
assert manifest["capture_runtime"]["fake_gpus_per_node"] == expected_fake_node_sizes[model]
assert manifest["capture_runtime"]["scaling_min_warmup_iters"] == 3
assert manifest["capture_runtime"]["scaling_profile_iters"] == 1
summary = manifest["capture_summary"]
assert summary["capture_scope"] == expected_capture_scopes[model]
assert summary["selected_rank_count"] == expected_counts[model]
assert summary["trace_file_count"] == expected_counts[model]
assert summary["memory_json_count"] == expected_counts[model]
assert summary["maximum_peak_allocated_mb"] == 90.0
assert summary["effective_global_batch_size"] == expected_global_batch_flags[model][0]
assert summary["global_batch_size_flag_values"] == expected_global_batch_flags[model], (
    f"{model} global_batch_size_flag_values: "
    f"expected={expected_global_batch_flags[model]}, "
    f"observed={summary['global_batch_size_flag_values']}"
)
single_rank = summary["single_rank_elapsed_seconds"]
estimated_full = summary["estimated_full_seconds"]
assert isinstance(single_rank, (int, float)) and not isinstance(single_rank, bool)
assert single_rank > 0
assert estimated_full == single_rank * expected_timing_multipliers[model]
assert summary["estimate_basis_rank"] == 0
assert summary["estimate_rank_count"] == expected_timing_multipliers[model]
assert summary["d16_gate_applicable"] is expected_d16_applicability[model]
if expected_d16_applicability[model]:
    assert summary["fresh_capture_gate_threshold_seconds"] == 7200
    expected_gate = "pass" if estimated_full <= 7200 else "prebaked_required"
    assert summary["fresh_capture_gate_result"] == expected_gate
    assert preflight_report_path.is_file()
    preflight = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    assert preflight["schema_version"] == "sc26-ae-task1-d16-preflight-v1"
    assert preflight["model"] == model
    assert preflight["preflight_capture_id"] == f"{capture_id}-preflight"
    assert preflight["rank_id"] == 0
    assert preflight["selected_rank_ids"] == [0]
    assert preflight["selected_rank_count"] == 1
    assert preflight["estimate_rank_count"] == 256
    assert preflight["estimated_full_seconds"] == preflight["rank0_elapsed_seconds"] * 256
    assert preflight["fresh_capture_gate_result"] == "pass"
    assert summary["single_rank_elapsed_seconds"] == preflight["rank0_elapsed_seconds"]
    assert summary["selected_capture_rank0_elapsed_seconds"] > 0
    assert summary["d16_preflight_capture_id"] == f"{capture_id}-preflight"
    assert "provenance/d16_preflight.json" in manifest_files
else:
    assert "fresh_capture_gate_threshold_seconds" not in summary
    assert "fresh_capture_gate_result" not in summary
    assert "selected_capture_rank0_elapsed_seconds" not in summary
assert all("_work/task1-preflight" not in path for path in manifest_files)
assert len(manifest_files) >= expected_counts[model] * 2 + 2
PY

    assert_contains "scaling_min_warmup_iters=3" "${run_root}/logs/summary.log"
    assert_contains "scaling_profile_iters=1" "${run_root}/logs/summary.log"
    assert_contains "capture_id=${capture_id}" "${run_root}/logs/summary.log"
    assert_contains "execution_evidence=local_synthetic_not_gpu_qualification" \
        "${run_root}/logs/summary.log"
    case "${model}" in
        gpt175b)
            assert_contains "capture_scope=representative" "${run_root}/logs/summary.log"
            assert_contains "d16_gate_applicable=false" "${run_root}/logs/summary.log"
            assert_contains "estimate_rank_count=8" "${run_root}/logs/summary.log"
            assert_not_contains "fresh_capture_gate_threshold_seconds=" \
                "${run_root}/logs/summary.log"
            assert_not_contains "fresh_capture_gate_result=" "${run_root}/logs/summary.log"
            ;;
        qwen3_a30b|dsv3)
            assert_contains "capture_scope=quick" "${run_root}/logs/summary.log"
            assert_contains "d16_gate_applicable=true" "${run_root}/logs/summary.log"
            assert_contains "estimate_rank_count=256" "${run_root}/logs/summary.log"
            assert_contains "fresh_capture_gate_threshold_seconds=7200" \
                "${run_root}/logs/summary.log"
            ;;
    esac
    assert_contains "MANIFEST_STATUS=verified" "${TEST_ROOT}/${model}-verify.log"
    case "${model}" in
        gpt175b)
            assert_contains "global_batch_size_flag_values=768" "${run_root}/logs/summary.log"
            ;;
        qwen3_a30b|dsv3)
            assert_contains "global_batch_size_flag_values=128,128" "${run_root}/logs/summary.log"
            ;;
    esac
}

run_d16_behavior_case() {
    local case_name=$1
    local quick=$2
    local force_above_threshold=$3
    local expected_status=$4
    local output_root="${TEST_ROOT}/d16-${case_name}"
    local capture_id="dsv3-${case_name}"
    local preflight_id="${capture_id}-preflight"
    local entry="${REPO_ROOT}/SC26-AE/task1_dsv3.sh"
    local log_path="${TEST_ROOT}/d16-${case_name}.log"
    local state_path="${TEST_ROOT}/d16-${case_name}.date-state"
    local before_calls after_calls status

    before_calls=$(wc -l < "${TORCHRUN_LOG}")
    set +e
    PATH="${FAKE_BIN}:${PATH}" \
    AE_OUTPUT_ROOT="${output_root}" \
    AE_TASK1_TEST_MODE=1 \
    AE_TASK1_TEST_CAPTURE_ID="${capture_id}" \
    AE_TASK1_TEST_PREFLIGHT_CAPTURE_ID="${preflight_id}" \
    AE_TASK1_TORCHRUN="${FAKE_BIN}/torchrun" \
    AE_MEGATRON_PYTHON="$(command -v python3)" \
    FAKE_TORCHRUN_LOG="${TORCHRUN_LOG}" \
    FAKE_TRACE_MODE=valid \
    FAKE_MEMORY_MODE=valid \
    FAKE_D16_ABOVE_THRESHOLD="${force_above_threshold}" \
    FAKE_D16_DATE_STATE="${state_path}" \
    QUICK="${quick}" \
    SCALE_GPU=0 \
    CAPTURE_NSYS=0 \
        bash "${entry}" >"${log_path}" 2>&1
    status=$?
    set -e
    assert_equals "${expected_status}" "${status}" "${case_name} exit status"
    after_calls=$(wc -l < "${TORCHRUN_LOG}")
    local invocation_delta=$((after_calls - before_calls))
    local task_dir="${output_root}/dsv3/task1"
    local run_root="${task_dir}/runs/${capture_id}"
    local preflight_root="${output_root}/_work/task1-preflight.${preflight_id}"
    local report_path="${preflight_root}/preflight_result.json"
    [[ -s "${report_path}" ]] || fail "${case_name} missing preflight report"

    "$(command -v python3)" - "${report_path}" "${quick}" "${force_above_threshold}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
quick = sys.argv[2] == "1"
above = sys.argv[3] == "1"
assert payload["requested_capture_scope"] == ("quick" if quick else "full")
assert payload["d16_gate_enforced"] is (not quick)
assert payload["gate_decision_applied"] is (not quick)
assert payload["fresh_capture_gate_result"] == ("prebaked_required" if above else "pass")
assert payload["rank_id"] == 0
assert payload["selected_rank_ids"] == [0]
assert payload["estimate_rank_count"] == 256
if above:
    assert payload["estimated_full_seconds"] > 7200
else:
    assert payload["estimated_full_seconds"] <= 7200
PY

    case "${case_name}" in
        quick-above-threshold)
            [[ -s "${task_dir}/capture_marker.json" ]] || fail "${case_name} missing QUICK marker"
            [[ -s "${run_root}/artifact_manifest.json" ]] || fail "${case_name} missing QUICK manifest"
            assert_equals "5" "${invocation_delta}" "${case_name} preflight plus four-rank smoke calls"
            assert_contains "TASK1_STATUS=verified" "${log_path}"
            assert_contains "TASK1_RUN_ROOT=${run_root}" "${log_path}"
            ;;
        full-pass)
            [[ -s "${task_dir}/capture_marker.json" ]] || fail "${case_name} missing full marker"
            [[ -s "${run_root}/artifact_manifest.json" ]] || fail "${case_name} missing full manifest"
            [[ -s "${run_root}/provenance/d16_preflight.json" ]] || \
                fail "${case_name} missing copied preflight report"
            assert_equals "257" "${invocation_delta}" "${case_name} preflight plus 256-rank calls"
            assert_contains "TASK1_STATUS=verified" "${log_path}"
            ;;
        full-above-threshold)
            [[ ! -e "${run_root}" ]] || fail "${case_name} created a full run root"
            [[ ! -e "${task_dir}/capture_marker.json" ]] || fail "${case_name} published a marker"
            [[ ! -e "${task_dir}/runs/${capture_id}/artifact_manifest.json" ]] || \
                fail "${case_name} created a full manifest"
            assert_equals "1" "${invocation_delta}" "${case_name} only ran the independent preflight"
            assert_contains "TASK1_STATUS=prebaked_required" "${log_path}"
            assert_not_contains "TASK1_RUN_ROOT=" "${log_path}"
            ;;
        *)
            fail "unknown D16 behavior case: ${case_name}"
            ;;
    esac
    pass "${case_name} D16 behavior matrix"
}

run_d16_behavior_case quick-above-threshold 1 1 0
run_d16_behavior_case full-pass 0 0 0
run_d16_behavior_case full-above-threshold 0 1 2

metadata_scope_probe_root="${TEST_ROOT}/metadata-scope-probe"
mkdir -p "${metadata_scope_probe_root}"
metadata_scope_inventory="${metadata_scope_probe_root}/inventory.json"
metadata_scope_batch_log="${metadata_scope_probe_root}/global_batch_size_flags.log"
metadata_scope_metadata="${metadata_scope_probe_root}/metadata.json"
metadata_scope_summary="${metadata_scope_probe_root}/summary.log"
"$(command -v python3)" - "${metadata_scope_inventory}" "${metadata_scope_batch_log}" <<'PY'
import json
import pathlib
import sys

inventory_path = pathlib.Path(sys.argv[1])
batch_path = pathlib.Path(sys.argv[2])
selected = [pp * 8 * 4 + exp * 8 for pp in range(8) for exp in range(4)]
json.dump(
    {
        "trace_file_count": len(selected),
        "memory_json_count": len(selected),
        "per_rank_peak_allocated_mb": {str(rank): 90.0 for rank in selected},
        "maximum_peak_allocated_mb": 90.0,
        "nsys_rep_path": None,
        "sqlite_path": None,
    },
    inventory_path.open("w", encoding="utf-8"),
    indent=2,
    sort_keys=True,
)
inventory_path.write_text(inventory_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
batch_path.write_text("\n".join(["128"] * len(selected)) + "\n", encoding="utf-8")
PY
ae_task1_load_config qwen3_a30b "${REPO_ROOT}"
metadata_scope_selected=$(ae_task1_selected_ranks qwen3_a30b 0)
    ae_task1_write_metadata_and_summary \
        "$(command -v python3)" "${metadata_scope_metadata}" "${metadata_scope_summary}" \
        "${metadata_scope_inventory}" qwen3_a30b qwen3-full-probe "${metadata_scope_selected}" 0 \
        "$(git rev-parse HEAD)" "$(ae_source_commit Echo-slowdown)" \
        "$(ae_source_commit megatron-sim-engine)" 0 "${metadata_scope_batch_log}" \
        runtime_measurement_requires_external_single_gpu_qualification representative_ep 1.25
"$(command -v python3)" - "${metadata_scope_metadata}" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = metadata["capture_summary"]
assert summary["capture_scope"] == "representative_ep"
assert summary["selected_rank_ids"] == [pp * 8 * 4 + exp * 8 for pp in range(8) for exp in range(4)]
assert summary["selected_rank_count"] == 32
assert summary["trace_file_count"] == 32
assert summary["memory_json_count"] == 32
assert summary["single_rank_elapsed_seconds"] == 1.25
assert summary["estimated_full_seconds"] == 320.0
assert summary["estimate_basis_rank"] == 0
assert summary["estimate_rank_count"] == 256
assert summary["d16_gate_applicable"] is True
assert summary["fresh_capture_gate_threshold_seconds"] == 7200
assert summary["fresh_capture_gate_result"] == "pass"
PY
assert_contains "capture_scope=representative_ep" "${metadata_scope_summary}"

metadata_timing_rejected() {
    local case_name=$1
    local expected_error=$2
    shift 2
    local case_root="${metadata_scope_probe_root}/${case_name}"
    local metadata_path="${case_root}/metadata.json"
    local summary_path="${case_root}/summary.log"
    local log_path="${case_root}/writer.log"
    local status

    mkdir -p "${case_root}"
    set +e
    ae_task1_write_metadata_and_summary \
        "$(command -v python3)" "${metadata_path}" "${summary_path}" \
        "${metadata_scope_inventory}" qwen3_a30b "qwen3-${case_name}" \
        "${metadata_scope_selected}" 0 "$(git rev-parse HEAD)" \
        "$(ae_source_commit Echo-slowdown)" "$(ae_source_commit megatron-sim-engine)" \
        0 "${metadata_scope_batch_log}" \
        runtime_measurement_requires_external_single_gpu_qualification full "$@" \
        >"${log_path}" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "${case_name} D16 timing input was unexpectedly accepted"
    assert_contains "${expected_error}" "${log_path}"
    pass "${case_name} D16 timing input fails closed"
}

metadata_timing_rejected missing-timing "single_rank_elapsed_seconds must be numeric"
metadata_timing_rejected nonnumeric-timing "single_rank_elapsed_seconds must be numeric" invalid
metadata_timing_rejected zero-timing "finite and strictly positive" 0
metadata_timing_rejected negative-timing "finite and strictly positive" -1

run_entry gpt175b 0
run_entry dsv3 0
run_entry qwen3_a30b 1
dsv3_batch_flag_counts=$("$(command -v python3)" - "${TORCHRUN_LOG}" <<'PY'
import pathlib
import shlex
import sys

lines = [
    line
    for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if "/output-dsv3/" in line
]
print(",".join(str(shlex.split(line).count("--global-batch-size")) for line in lines))
PY
)
assert_equals "2,2,2,2,2" "${dsv3_batch_flag_counts}" \
    "DSV3 duplicate global-batch-size flags remain visible (preflight plus QUICK)"
pass "three Task1 entries produce isolated verified mock bundles"

torchrun_count=$(wc -l < "${TORCHRUN_LOG}")
assert_equals "281" "${torchrun_count}" "fake torchrun invocation count including D16 behavior matrix"
assert_contains "--bf16" "${TORCHRUN_LOG}"
assert_contains "--mock-data" "${TORCHRUN_LOG}"
assert_contains "--overlap-grad-reduce" "${TORCHRUN_LOG}"
assert_contains "--trace-memory" "${TORCHRUN_LOG}"
assert_contains "--trace-kernel-ground-truth" "${TORCHRUN_LOG}"
assert_contains "--trace-kernel-ground-truth-phase" "${TORCHRUN_LOG}"
assert_contains "--scaling-min-warmup-iters 3" "${TORCHRUN_LOG}"
assert_contains "--scaling-profile-iters 1" "${TORCHRUN_LOG}"
assert_not_contains "--fp16" "${TORCHRUN_LOG}"
warmup_count=$(grep -o -- '--scaling-min-warmup-iters' "${TORCHRUN_LOG}" | wc -l)
profile_count=$(grep -o -- '--scaling-profile-iters' "${TORCHRUN_LOG}" | wc -l)
assert_equals "281" "${warmup_count}" "one warmup flag per invocation"
assert_equals "281" "${profile_count}" "one profile flag per invocation"
assert_contains "CWD=${TEST_ROOT}/output-gpt175b/gpt175b/task1/runs/gpt175b-" "${TORCHRUN_LOG}"
gpt_transformer_impl_count=$(grep -F "/output-gpt175b/" "${TORCHRUN_LOG}" | grep -F -- "--transformer-impl local" | wc -l || true)
assert_equals "8" "${gpt_transformer_impl_count}" "GPT Task1 fixes local transformer implementation"
pass "adapter enforces bf16, mock data, overlap, trace memory, and warmup/profile"

assert_contains "profile" "${NSYS_LOG}"
assert_contains '--trace=cuda\,nvtx\,osrt' "${NSYS_LOG}"
assert_contains "--trace-fork-before-exec=true" "${NSYS_LOG}"
assert_contains "export" "${NSYS_LOG}"
qwen_run=$("$(command -v python3)" - "${TEST_ROOT}/output-qwen3_a30b/qwen3_a30b/task1/capture_marker.json" <<'PY'
import json
import pathlib
import sys
marker = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(pathlib.Path(sys.argv[1]).parent / marker["run_path"])
PY
)
[[ -s "${qwen_run}/nsys/qwen3_a30b.nsys-rep" ]] || fail "missing qwen nsys report"
[[ -s "${qwen_run}/nsys/qwen3_a30b.sqlite" ]] || fail "missing qwen sqlite"
pass "one Nsight invocation wraps the complete selected-rank source loop"

FAIL_OUTPUT="${TEST_ROOT}/failure-output"
set +e
PATH="${FAKE_BIN}:${PATH}" \
AE_OUTPUT_ROOT="${FAIL_OUTPUT}" \
AE_TASK1_TEST_MODE=1 \
AE_TASK1_TEST_CAPTURE_ID="gpt175b-20260718T000009Z" \
AE_TASK1_TORCHRUN="${FAKE_BIN}/torchrun" \
AE_MEGATRON_PYTHON="$(command -v python3)" \
FAKE_TORCHRUN_LOG="${TORCHRUN_LOG}" \
FAKE_TORCHRUN_EXIT=17 \
QUICK=1 SCALE_GPU=0 CAPTURE_NSYS=0 \
bash "${REPO_ROOT}/SC26-AE/task1_gpt175b.sh" > "${TEST_ROOT}/failure.log" 2>&1
failure_status=$?
set -e
[[ ${failure_status} -ne 0 ]] || fail "expected source failure to propagate"
[[ ! -e "${FAIL_OUTPUT}/gpt175b/task1/capture_marker.json" ]] || fail "marker published after source failure"
assert_contains "[ERROR]" "${TEST_ROOT}/failure.log"
pass "source failure leaves the partial run unverified and publishes no marker"

assert_trace_semantic_failure() {
    local case_name=$1
    local trace_mode=$2
    local expected_error=$3
    local output_root="${TEST_ROOT}/semantic-failure-${case_name}"
    local capture_id="dsv3-semantic-${case_name}"
    local log_path="${TEST_ROOT}/semantic-failure-${case_name}.log"
    local status

    set +e
    PATH="${FAKE_BIN}:${PATH}" \
    AE_OUTPUT_ROOT="${output_root}" \
    AE_TASK1_TEST_MODE=1 \
    AE_TASK1_TEST_CAPTURE_ID="${capture_id}" \
    AE_TASK1_TORCHRUN="${FAKE_BIN}/torchrun" \
    AE_MEGATRON_PYTHON="$(command -v python3)" \
    FAKE_TORCHRUN_LOG="${TORCHRUN_LOG}" \
    FAKE_TRACE_MODE="${trace_mode}" \
    QUICK=1 SCALE_GPU=0 CAPTURE_NSYS=0 \
        bash "${REPO_ROOT}/SC26-AE/task1_dsv3.sh" > "${log_path}" 2>&1
    status=$?
    set -e

    [[ ${status} -ne 0 ]] || fail "${case_name} trace semantics were unexpectedly accepted"
    [[ ! -e "${output_root}/dsv3/task1/capture_marker.json" ]] || \
        fail "${case_name} published a marker after semantic rejection"
    assert_contains "${expected_error}" "${log_path}"
    pass "${case_name} trace semantics fail before marker publication"
}

assert_trace_semantic_failure \
    missing-forward-step \
    missing_forward_step \
    "Task1 trace semantic validation is missing forward_step"
assert_trace_semantic_failure \
    missing-backward-step \
    missing_backward_step \
    "Task1 trace semantic validation is missing backward_step"
assert_trace_semantic_failure \
    missing-optimizer-step \
    missing_optimizer_step \
    "Task1 trace semantic validation is missing optimizer_step"
assert_trace_semantic_failure \
    missing-backward-cmd-uid \
    missing_backward_cmd_uid \
    "Task1 backward trace metadata is incomplete"
assert_trace_semantic_failure \
    missing-backward-timestamp \
    missing_backward_timestamp \
    "Task1 backward trace metadata is incomplete"
assert_trace_semantic_failure \
    missing-backward-duration \
    missing_backward_duration \
    "Task1 backward trace metadata is incomplete"
assert_trace_semantic_failure \
    missing-backward-mg-state \
    missing_backward_mg_state \
    "Task1 backward trace metadata is incomplete"
assert_trace_semantic_failure \
    missing-backward-stage-id \
    missing_backward_stage_id \
    "Task1 backward trace metadata is incomplete"
assert_trace_semantic_failure \
    missing-backward-batch-id \
    missing_backward_batch_id \
    "Task1 backward trace metadata is incomplete"
assert_trace_semantic_failure \
    missing-ddp-trigger \
    missing_ddp_trigger \
    "Task1 trace lacks DDP-overlap trigger metadata"

assert_memory_semantic_failure() {
    local case_name=$1
    local memory_mode=$2
    local expected_error=$3
    local output_root="${TEST_ROOT}/memory-failure-${case_name}"
    local capture_id="dsv3-memory-${case_name}"
    local log_path="${TEST_ROOT}/memory-failure-${case_name}.log"
    local status

    set +e
    PATH="${FAKE_BIN}:${PATH}" \
    AE_OUTPUT_ROOT="${output_root}" \
    AE_TASK1_TEST_MODE=1 \
    AE_TASK1_TEST_CAPTURE_ID="${capture_id}" \
    AE_TASK1_TORCHRUN="${FAKE_BIN}/torchrun" \
    AE_MEGATRON_PYTHON="$(command -v python3)" \
    FAKE_TORCHRUN_LOG="${TORCHRUN_LOG}" \
    FAKE_MEMORY_MODE="${memory_mode}" \
    QUICK=1 SCALE_GPU=0 CAPTURE_NSYS=0 \
        bash "${REPO_ROOT}/SC26-AE/task1_dsv3.sh" > "${log_path}" 2>&1
    status=$?
    set -e

    [[ ${status} -ne 0 ]] || fail "${case_name} memory semantics were unexpectedly accepted"
    [[ ! -e "${output_root}/dsv3/task1/capture_marker.json" ]] || \
        fail "${case_name} published a marker after memory rejection"
    assert_contains "${expected_error}" "${log_path}"
    pass "${case_name} memory semantics fail before marker publication"
}

assert_memory_semantic_failure \
    empty-payload \
    empty_payload \
    "Empty memory payload"
assert_memory_semantic_failure \
    empty-samples \
    empty_samples \
    "Memory samples are empty"
assert_memory_semantic_failure \
    nan-peak \
    nan_peak \
    "Non-positive peak_allocated_MB"
assert_memory_semantic_failure \
    inf-peak \
    inf_peak \
    "Non-positive peak_allocated_MB"
assert_memory_semantic_failure \
    zero-peak \
    zero_peak \
    "Non-positive peak_allocated_MB"
assert_memory_semantic_failure \
    negative-reserved \
    negative_reserved \
    "Invalid reserved_memory_MB"
assert_memory_semantic_failure \
    negative-allocated \
    negative_allocated \
    "Invalid allocated_memory_MB"
assert_memory_semantic_failure \
    all-zero-samples \
    all_zero_samples \
    "Memory samples never report positive usage"
assert_memory_semantic_failure \
    missing-rank \
    missing_rank \
    "Memory rank inventory mismatch"
assert_memory_semantic_failure \
    duplicate-rank \
    duplicate_rank \
    "Memory rank inventory mismatch"

assert_contains 'FAKE_RANK_ORDER' "${REPO_ROOT}/examples/update_pretrain_gpt.sh"
pass "GPT source accepts an explicit fake-rank order for rank-scoped captures"
assert_contains 'FAKE_RANK_ORDER=0' "${REPO_ROOT}/SC26-AE/lib/task1_trace.sh"
pass "Task1 NCU capture scopes the slowdown workload to global rank 0"

# Exercise the GPT source parser without launching Megatron: a fake torchrun
# records invocations, while invalid explicit rank lists must fail before the
# first workload process is started.
GPT_SOURCE_ROOT="${TEST_ROOT}/gpt-rank-order"
GPT_SOURCE_LOG="${GPT_SOURCE_ROOT}/torchrun.log"
GPT_SOURCE_OUTPUT="${GPT_SOURCE_ROOT}/output"
mkdir -p "${GPT_SOURCE_ROOT}/bin"
cat > "${GPT_SOURCE_ROOT}/bin/torchrun" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${GPT_SOURCE_LOG}"
printf '\n' >> "${GPT_SOURCE_LOG}"
SH
chmod +x "${GPT_SOURCE_ROOT}/bin/torchrun"
: > "${GPT_SOURCE_LOG}"

env \
    PATH="${GPT_SOURCE_ROOT}/bin:${PATH}" \
    GPT_SOURCE_LOG="${GPT_SOURCE_LOG}" \
    BASE_PATH="${REPO_ROOT}" \
    LOG_ROOT="${GPT_SOURCE_OUTPUT}" \
    MODEL_SIZE=tiny \
    TRANSFORMER_IMPL=local \
    MOCK_DATA=1 \
    FAKE_WORLD_SIZE=8 \
    FAKE_PP=2 \
    FAKE_TP=2 \
    FAKE_RANK_ORDER=3 \
    bash "${REPO_ROOT}/examples/update_pretrain_gpt.sh" \
    > "${GPT_SOURCE_ROOT}/valid.log" 2>&1
assert_equals "1" "$(wc -l < "${GPT_SOURCE_LOG}")" "GPT explicit rank invocation count"
assert_contains '--fake-current-rank-id 3' "${GPT_SOURCE_LOG}"
pass "GPT source executes exactly the requested explicit fake rank"

assert_gpt_rank_order_failure() {
    local rank_order=$1
    local case_name=$2
    local before_calls after_calls status
    before_calls=$(wc -l < "${GPT_SOURCE_LOG}")
    set +e
    env \
        PATH="${GPT_SOURCE_ROOT}/bin:${PATH}" \
        GPT_SOURCE_LOG="${GPT_SOURCE_LOG}" \
        BASE_PATH="${REPO_ROOT}" \
        LOG_ROOT="${GPT_SOURCE_OUTPUT}/${case_name}" \
        MODEL_SIZE=tiny \
        TRANSFORMER_IMPL=local \
        MOCK_DATA=1 \
        FAKE_WORLD_SIZE=8 \
        FAKE_PP=2 \
        FAKE_TP=2 \
        FAKE_RANK_ORDER="${rank_order}" \
        bash "${REPO_ROOT}/examples/update_pretrain_gpt.sh" \
        > "${GPT_SOURCE_ROOT}/${case_name}.log" 2>&1
    status=$?
    set -e
    [[ ${status} -ne 0 ]] || fail "GPT rank order ${rank_order} unexpectedly succeeded"
    after_calls=$(wc -l < "${GPT_SOURCE_LOG}")
    assert_equals "${before_calls}" "${after_calls}" "GPT rank order ${rank_order} torchrun calls"
    pass "GPT rank order ${rank_order} fails before workload execution"
}

assert_gpt_rank_order_failure "8" out_of_range
assert_gpt_rank_order_failure "1,foo" non_decimal
assert_gpt_rank_order_failure "1,1" duplicate
assert_gpt_rank_order_failure "," empty_item

EXISTING_OUTPUT="${TEST_ROOT}/existing-output"
COMMON_ENV=(
    PATH="${FAKE_BIN}:${PATH}"
    AE_OUTPUT_ROOT="${EXISTING_OUTPUT}"
    AE_TASK1_TEST_MODE=1
    AE_TASK1_TEST_CAPTURE_ID="dsv3-20260718T000010Z"
    AE_TASK1_TORCHRUN="${FAKE_BIN}/torchrun"
    AE_MEGATRON_PYTHON="$(command -v python3)"
    FAKE_TORCHRUN_LOG="${TORCHRUN_LOG}"
    QUICK=1
    SCALE_GPU=0
    CAPTURE_NSYS=0
)
env "${COMMON_ENV[@]}" bash "${REPO_ROOT}/SC26-AE/task1_dsv3.sh" > "${TEST_ROOT}/existing-first.log" 2>&1
set +e
env "${COMMON_ENV[@]}" bash "${REPO_ROOT}/SC26-AE/task1_dsv3.sh" > "${TEST_ROOT}/existing-second.log" 2>&1
existing_status=$?
set -e
[[ ${existing_status} -ne 0 ]] || fail "expected existing run root rejection"
assert_contains "already exists" "${TEST_ROOT}/existing-second.log"
pass "existing run destinations are never reused"

printf 'PASS_COUNT=%d\n' "${PASS_COUNT}"
[[ ${PASS_COUNT} -eq 46 ]] || fail "expected 46 cases, got ${PASS_COUNT}"
