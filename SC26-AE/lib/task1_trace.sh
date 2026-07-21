#!/usr/bin/env bash

# Shared Task1 workload-trace runner. Public entry scripts call ae_run_task1.

if ! declare -F ae_die >/dev/null 2>&1; then
    _AE_TASK1_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    # shellcheck source=common.sh
    source "${_AE_TASK1_LIB_DIR}/common.sh"
fi

ae_task1_selected_ranks() {
    local model_key=${1:-}
    local quick=${2:-0}
    local -a ranks=()
    local rank

    ae_require_enum QUICK "${quick}" 0 1 || return 1
    case "${model_key}" in
        gpt175b)
            printf '%s\n' '0,128,256,384,512,640,768,896'
            ;;
        qwen3_a30b)
            if [[ "${quick}" == "1" ]]; then
                printf '%s\n' '0,8,16,24'
                return 0
            fi
            local pp_stage exp_rank
            for ((pp_stage = 0; pp_stage < 8; pp_stage++)); do
                for ((exp_rank = 0; exp_rank < 4; exp_rank++)); do
                    ranks+=("$((pp_stage * 8 * 4 + exp_rank * 8))")
                done
            done
            local joined
            IFS=,
            joined="${ranks[*]}"
            unset IFS
            printf '%s\n' "${joined}"
            ;;
        dsv3)
            if [[ "${quick}" == "1" ]]; then
                printf '%s\n' '0,64,128,192'
                return 0
            fi
            for ((rank = 0; rank < 256; rank++)); do
                ranks+=("${rank}")
            done
            local joined
            IFS=,
            joined="${ranks[*]}"
            unset IFS
            printf '%s\n' "${joined}"
            ;;
        *)
            ae_die "Unknown Task1 model key: ${model_key}"
            ;;
    esac
}

ae_task1_capture_scope() {
    local model_key=${1:-}
    local quick=${2:-0}

    ae_require_enum QUICK "${quick}" 0 1 || return 1
    case "${model_key}" in
        gpt175b)
            # GPT-175B intentionally uses the paper's representative rank
            # set rather than a 1024-rank capture.
            printf '%s\n' 'representative'
            ;;
        qwen3_a30b)
            if [[ "${quick}" == "1" ]]; then
                printf '%s\n' 'quick'
            else
                printf '%s\n' 'representative_ep'
            fi
            ;;
        dsv3)
            if [[ "${quick}" == "1" ]]; then
                printf '%s\n' 'quick'
            else
                printf '%s\n' 'full'
            fi
            ;;
        *)
            ae_die "Unknown Task1 model key: ${model_key}"
            return 1
            ;;
    esac
}

ae_task1_load_config() {
    local model_key=$1
    local repo_root=$2

    case "${model_key}" in
        gpt175b)
            AE_T1_SOURCE_SCRIPT="${repo_root}/examples/update_pretrain_gpt.sh"
            AE_T1_PROFILE=175
            AE_T1_WORLD_SIZE=1024
            AE_T1_LOCAL_SIZE=8
            AE_T1_PP=8
            AE_T1_TP=8
            AE_T1_DP=16
            AE_T1_EXP=1
            AE_T1_NUM_EXPERTS=1
            AE_T1_LAYERS=96
            AE_T1_HIDDEN_SIZE=12288
            AE_T1_SEQ_LEN=2048
            AE_T1_MICRO_BATCH_SIZE=1
            AE_T1_NUM_MICROBATCHES=48
            AE_T1_GLOBAL_BATCH_SIZE=768
            AE_T1_FAKE_GPUS_PER_NODE=8
            AE_T1_TRANSFORMER_IMPL=local
            ;;
        qwen3_a30b)
            AE_T1_SOURCE_SCRIPT="${repo_root}/examples/pretrain_qwen3_30b_a3b_moe.sh"
            AE_T1_PROFILE=full
            AE_T1_WORLD_SIZE=256
            AE_T1_LOCAL_SIZE=8
            AE_T1_PP=8
            AE_T1_TP=8
            AE_T1_DP=4
            AE_T1_EXP=4
            AE_T1_NUM_EXPERTS=128
            AE_T1_LAYERS=48
            AE_T1_HIDDEN_SIZE=2048
            AE_T1_SEQ_LEN=256
            AE_T1_MICRO_BATCH_SIZE=1
            # Preserve global batch size 128 after changing DP from 8 to 4:
            # 32 microbatches * micro batch 1 * DP 4 = 128.
            AE_T1_NUM_MICROBATCHES=32
            AE_T1_GLOBAL_BATCH_SIZE=128
            AE_T1_FAKE_GPUS_PER_NODE=256
            AE_T1_TRANSFORMER_IMPL=transformer_engine
            ;;
        dsv3)
            AE_T1_SOURCE_SCRIPT="${repo_root}/examples/pretrain_deepseek_v3_moe.sh"
            AE_T1_PROFILE=smoke
            AE_T1_WORLD_SIZE=256
            AE_T1_LOCAL_SIZE=8
            AE_T1_PP=4
            AE_T1_TP=8
            AE_T1_DP=8
            AE_T1_EXP=8
            AE_T1_NUM_EXPERTS=32
            AE_T1_LAYERS=32
            AE_T1_HIDDEN_SIZE=2048
            AE_T1_SEQ_LEN=256
            AE_T1_MICRO_BATCH_SIZE=1
            AE_T1_NUM_MICROBATCHES=16
            AE_T1_GLOBAL_BATCH_SIZE=128
            AE_T1_FAKE_GPUS_PER_NODE=256
            AE_T1_TRANSFORMER_IMPL=local
            ;;
        *)
            ae_die "Unknown Task1 model key: ${model_key}"
            return 1
            ;;
    esac
    ae_require_file "${AE_T1_SOURCE_SCRIPT}"
}

ae_task1_assert_source_provenance() {
    local repo_root=$1
    local expected_commit=$2
    local source_relative relative expected_blob actual_blob object_type
    local -a source_files=()

    [[ -d "${repo_root}/.git" || -f "${repo_root}/.git" ]] || {
        ae_die "Task1 source provenance root is not a Git repository: ${repo_root}"
        return 1
    }
    [[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || {
        ae_die "Task1 source provenance commit is invalid: ${expected_commit}"
        return 1
    }
    [[ "${AE_T1_SOURCE_SCRIPT}" == "${repo_root}/"* ]] || {
        ae_die "Task1 source script is outside the repository root: ${AE_T1_SOURCE_SCRIPT}"
        return 1
    }
    source_relative=${AE_T1_SOURCE_SCRIPT#"${repo_root}/"}
    # Keep this list synchronized with every load-bearing Task1 producer entry and helper.
    source_files+=(
        "${source_relative}"
        pretrain_llama.py
        megatron/training/training.py
        megatron/profiler/cmd.py
        megatron/training/arguments.py
        SC26-AE/task1_gpt175b.sh
        SC26-AE/task1_dsv3.sh
        SC26-AE/task1_qwen3_a30b.sh
        SC26-AE/lib/common.sh
        SC26-AE/lib/task1_trace.sh
        SC26-AE/tools/artifact_manifest.py
    )

    for relative in "${source_files[@]}"; do
        local working_path="${repo_root}/${relative}"
        [[ -f "${working_path}" && ! -L "${working_path}" ]] || {
            ae_die "Task1 load-bearing source is not a regular file: ${relative}"
            return 1
        }
        expected_blob=$(git -C "${repo_root}" rev-parse \
            "${expected_commit}:${relative}" 2>/dev/null) || {
            ae_die "Task1 source is not tracked by pinned HEAD: ${relative}"
            return 1
        }
        object_type=$(git -C "${repo_root}" cat-file -t "${expected_blob}" 2>/dev/null) || {
            ae_die "Cannot inspect pinned Task1 source object: ${relative}"
            return 1
        }
        [[ "${object_type}" == blob ]] || {
            ae_die "Pinned Task1 source is not a file: ${relative}"
            return 1
        }
        actual_blob=$(git -C "${repo_root}" hash-object --no-filters \
            "${working_path}" 2>/dev/null) || {
            ae_die "Cannot hash Task1 working-tree source: ${relative}"
            return 1
        }
        [[ "${actual_blob}" == "${expected_blob}" ]] || {
            ae_die "Task1 tracked HEAD blob mismatch: ${relative}"
            return 1
        }
    done
}

ae_task1_resolve_gpu() {
    local scale_gpu=${SCALE_GPU:-}
    local visible=${CUDA_VISIBLE_DEVICES:-}

    if [[ -n "${scale_gpu}" ]]; then
        [[ "${scale_gpu}" != *,* ]] || {
            ae_die "SCALE_GPU must select exactly one GPU."
            return 1
        }
        [[ "${scale_gpu}" =~ ^[0-9]+$ ]] || {
            ae_die "SCALE_GPU must be a decimal GPU index."
            return 1
        }
        printf '%s\n' "${scale_gpu}"
        return 0
    fi
    if [[ -n "${visible}" && "${visible}" != *,* ]]; then
        [[ "${visible}" =~ ^[0-9]+$ ]] || {
            ae_die "CUDA_VISIBLE_DEVICES must select exactly one decimal GPU index."
            return 1
        }
        printf '%s\n' "${visible}"
        return 0
    fi
    ae_die "SCALE_GPU is required unless CUDA_VISIBLE_DEVICES selects exactly one GPU."
}

ae_task1_execution_evidence() {
    local test_mode=${1:-}
    case "${test_mode}" in
        1)
            printf '%s\n' 'local_synthetic_not_gpu_qualification'
            ;;
        0)
            # A real producer run is not a qualification attestation by
            # itself.  External H800 evidence must seal this pending label.
            printf '%s\n' 'runtime_measurement_requires_external_single_gpu_qualification'
            ;;
        *)
            ae_die "Task1 test mode must be 0 or 1; got '${test_mode}'."
            return 1
            ;;
    esac
}

ae_task1_write_torchrun_adapter() {
    local adapter_path=$1
    cat > "${adapter_path}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

real_torchrun=${AE_TASK1_REAL_TORCHRUN:?AE_TASK1_REAL_TORCHRUN is required}
[[ -x "${real_torchrun}" ]] || {
    printf '[ERROR] Task1 torchrun is missing or not executable: %s\n' "${real_torchrun}" >&2
    exit 1
}

batch_flag_log=${AE_TASK1_GLOBAL_BATCH_LOG:-}
[[ -n "${batch_flag_log}" ]] || {
    printf '[ERROR] AE_TASK1_GLOBAL_BATCH_LOG is required.\n' >&2
    exit 1
}
rank_timing_log=${AE_TASK1_RANK_TIMING_LOG:-}
[[ -n "${rank_timing_log}" ]] || {
    printf '[ERROR] AE_TASK1_RANK_TIMING_LOG is required.\n' >&2
    exit 1
}

arguments=("$@")
batch_values=()
rank_values=()
argument_index=0
while ((argument_index < ${#arguments[@]})); do
    argument=${arguments[argument_index]}
    case "${argument}" in
        --global-batch-size)
            value_index=$((argument_index + 1))
            ((value_index < ${#arguments[@]})) || {
                printf '[ERROR] --global-batch-size requires a value.\n' >&2
                exit 1
            }
            batch_value=${arguments[value_index]}
            [[ "${batch_value}" =~ ^[1-9][0-9]*$ ]] || {
                printf '[ERROR] --global-batch-size must be a positive integer, got %q.\n' \
                    "${batch_value}" >&2
                exit 1
            }
            batch_values+=("${batch_value}")
            argument_index=$((argument_index + 2))
            continue
            ;;
        --global-batch-size=*)
            batch_value=${argument#*=}
            [[ "${batch_value}" =~ ^[1-9][0-9]*$ ]] || {
                printf '[ERROR] --global-batch-size must be a positive integer, got %q.\n' \
                    "${batch_value}" >&2
                exit 1
            }
            batch_values+=("${batch_value}")
            ;;
        --fake-current-rank-id)
            value_index=$((argument_index + 1))
            ((value_index < ${#arguments[@]})) || {
                printf '[ERROR] --fake-current-rank-id requires a value.\n' >&2
                exit 1
            }
            rank_values+=("${arguments[value_index]}")
            argument_index=$((argument_index + 2))
            continue
            ;;
        --fake-current-rank-id=*)
            rank_values+=("${argument#*=}")
            ;;
    esac
    argument_index=$((argument_index + 1))
done

((${#batch_values[@]} > 0)) || {
    printf '[ERROR] Missing --global-batch-size in Task1 source torchrun arguments.\n' >&2
    exit 1
}
batch_value_csv=$(IFS=,; printf '%s' "${batch_values[*]}")
for batch_value in "${batch_values[@]}"; do
    [[ "${batch_value}" == "${batch_values[0]}" ]] || {
        printf '[ERROR] Conflicting --global-batch-size values: %s.\n' \
            "${batch_value_csv}" >&2
        exit 1
    }
done

((${#rank_values[@]} == 1)) || {
    printf '[ERROR] Task1 requires exactly one --fake-current-rank-id value; observed %s.\n' \
        "${#rank_values[@]}" >&2
    exit 1
}
rank_id=${rank_values[0]}
[[ "${rank_id}" =~ ^[0-9]+$ ]] || {
    printf '[ERROR] --fake-current-rank-id must be a non-negative integer, got %q.\n' \
        "${rank_id}" >&2
    exit 1
}

filtered=()
skip_next=0
for argument in "$@"; do
    if ((skip_next)); then
        skip_next=0
        continue
    fi
    case "${argument}" in
        --fp16|--bf16|--mock-data|--overlap-grad-reduce|--trace-memory|\
        --trace-kernel-ground-truth|--trace-kernel-ground-truth-phase)
            ;;
        --trace-memory-interval|--trace-kernel-ground-truth-prefix|\
        --trace-kernel-boundary-sync-mode|--scaling-min-warmup-iters|\
        --scaling-profile-iters|--train-iters|--lr-decay-iters|\
        --trace-start|--nsight-start)
            skip_next=1
            ;;
        *)
            filtered+=("${argument}")
            ;;
    esac
done
((skip_next == 0)) || {
    printf '[ERROR] Task1 source ended with an option that requires a value.\n' >&2
    exit 1
}

batch_flag_dir=$(dirname -- "${batch_flag_log}")
[[ -d "${batch_flag_dir}" && ! -L "${batch_flag_log}" ]] || {
    printf '[ERROR] Task1 global-batch-size log path is invalid: %s\n' \
        "${batch_flag_log}" >&2
    exit 1
}
rank_timing_dir=$(dirname -- "${rank_timing_log}")
[[ -d "${rank_timing_dir}" && ! -L "${rank_timing_log}" ]] || {
    printf '[ERROR] Task1 rank timing log path is invalid: %s\n' \
        "${rank_timing_log}" >&2
    exit 1
}
if ! printf '%s\n' "${batch_value_csv}" >> "${batch_flag_log}"; then
    printf '[ERROR] Failed to persist Task1 global-batch-size values: %s\n' \
        "${batch_flag_log}" >&2
    exit 1
fi

filtered+=(
    --train-iters 1
    --lr-decay-iters 1
    --trace-start 1
    --bf16
    --mock-data
    --overlap-grad-reduce
    --trace-memory
    --trace-memory-interval 0.01
    --trace-kernel-ground-truth
    --trace-kernel-ground-truth-prefix cmd_trace
    --trace-kernel-ground-truth-phase
    --trace-kernel-boundary-sync-mode event
    --scaling-min-warmup-iters 3
    --scaling-profile-iters 1
)
start_ns=$(date +%s%N)
[[ "${start_ns}" =~ ^[0-9]+$ ]] || {
    printf '[ERROR] Failed to read a nanosecond Task1 start timestamp: %q.\n' \
        "${start_ns}" >&2
    exit 1
}
set +e
"${real_torchrun}" "${filtered[@]}"
torchrun_status=$?
set -e
end_ns=$(date +%s%N)
[[ "${end_ns}" =~ ^[0-9]+$ ]] || {
    printf '[ERROR] Failed to read a nanosecond Task1 end timestamp: %q.\n' \
        "${end_ns}" >&2
    exit 1
}
((end_ns > start_ns)) || {
    printf '[ERROR] Task1 rank timing is not strictly positive: start=%s end=%s.\n' \
        "${start_ns}" "${end_ns}" >&2
    exit 1
}
elapsed_ns=$((end_ns - start_ns))
printf '%s,%s,%s,%s\n' "${rank_id}" "${start_ns}" "${end_ns}" "${elapsed_ns}" \
    >> "${rank_timing_log}" || {
        printf '[ERROR] Failed to persist Task1 rank timing: %s\n' \
            "${rank_timing_log}" >&2
        exit 1
    }
exit "${torchrun_status}"
SH
    chmod +x "${adapter_path}"
}

ae_task1_write_rank_loop() {
    local loop_path=$1
    local runtime_dir=$2
    local adapter_bin_dir=$3
    local source_script=$4
    local model_key=$5
    local selected_ranks=$6
    local scale_gpu=$7
    local capture_id=$8
    local batch_flag_log=$9
    local rank_timing_log=${10}
    local real_torchrun_dir

    real_torchrun_dir=$(dirname -- "${AE_TASK1_REAL_TORCHRUN}")

    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'export PATH=%q\n' "${adapter_bin_dir}:${real_torchrun_dir}:${PATH}"
        printf 'export AE_TASK1_REAL_TORCHRUN=%q\n' "${AE_TASK1_REAL_TORCHRUN}"
        printf 'export MODE=scaling\n'
        printf 'export SCALE_GPU=%q\n' "${scale_gpu}"
        printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${scale_gpu}"
        printf 'export MODEL_PROFILE=%q\n' "${AE_T1_PROFILE}"
        printf 'export MODEL_SIZE=%q\n' "${AE_T1_PROFILE}"
        printf 'export MOCK_DATA=1\n'
        printf 'export USE_BF16=1\n'
        printf 'export TRAIN_ITERS=1\n'
        printf 'export TRACE_START=1\n'
        printf 'export DO_TRACE=True\n'
        printf 'export TRACE_MEMORY=1\n'
        printf 'export TRACE_MEMORY_INTERVAL=0.01\n'
        printf 'export TRACE_KERNEL_GROUND_TRUTH=1\n'
        printf 'export TRACE_KERNEL_GROUND_TRUTH_PHASE=1\n'
        printf 'export TRACE_KERNEL_BOUNDARY_SYNC_MODE=event\n'
        printf 'export OVERLAP_GRAD_REDUCE=1\n'
        printf 'export SCALING_MIN_WARMUP_ITERS=3\n'
        printf 'export SCALING_PROFILE_ITERS=1\n'
        printf 'export SCALING_REPLAY_CACHE_TAG=%q\n' "${capture_id}"
        printf 'export AE_TASK1_GLOBAL_BATCH_LOG=%q\n' "${batch_flag_log}"
        printf 'export AE_TASK1_RANK_TIMING_LOG=%q\n' "${rank_timing_log}"
        printf 'export FAKE_WORLD_SIZE=%q\n' "${AE_T1_WORLD_SIZE}"
        printf 'export FAKE_PP=%q\n' "${AE_T1_PP}"
        printf 'export FAKE_TP=%q\n' "${AE_T1_TP}"
        printf 'export FAKE_DP=%q\n' "${AE_T1_DP}"
        printf 'export FAKE_EXP=%q\n' "${AE_T1_EXP}"
        printf 'export PP=%q\n' "${AE_T1_PP}"
        printf 'export TP=%q\n' "${AE_T1_TP}"
        printf 'export EP=%q\n' "${AE_T1_EXP}"
        printf 'export MICRO_BATCH_SIZE=%q\n' "${AE_T1_MICRO_BATCH_SIZE}"
        printf 'export NUM_MICBATCH=%q\n' "${AE_T1_NUM_MICROBATCHES}"
        printf 'export GLOBAL_BATCH_SIZE=%q\n' "${AE_T1_GLOBAL_BATCH_SIZE}"
        if [[ "${model_key}" == "qwen3_a30b" ]]; then
            # Scaling Task1 profiles one iteration, so a one-step warmup would
            # equal the decay horizon and violate the scheduler contract.
            printf 'export LR_WARMUP_ITERS=0\n'
        fi
        printf 'export SEQ_LEN=%q\n' "${AE_T1_SEQ_LEN}"
        printf 'export TRANSFORMER_IMPL=%q\n' "${AE_T1_TRANSFORMER_IMPL}"
        printf 'export BASE_PATH=%q\n' "$(ae_repo_root)"
        printf 'export LOG_ROOT=%q\n' "${runtime_dir}/source_logs"
        case "${model_key}" in
            gpt175b|qwen3_a30b)
                # Dense GPT and Qwen source scripts consume FAKE_RANK_ORDER.
                # The NCU caller passes the literal singleton FAKE_RANK_ORDER=0;
                # normal Task1 tracing passes the selected representative list.
                printf 'export FAKE_RANK_ORDER=%q\n' "${selected_ranks}"
                ;;
            dsv3)
                printf 'export SCALING_FAKE_RANK_ORDER=%q\n' "${selected_ranks}"
                ;;
        esac
        printf 'cd -- %q\n' "${runtime_dir}"
        printf 'exec bash %q\n' "${source_script}"
    } > "${loop_path}"
    chmod +x "${loop_path}"
}

ae_task1_validate_outputs() {
    local python_bin=$1
    local run_root=$2
    local selected_ranks=$3
    local capture_nsys=$4
    local model_key=$5
    local output_json=$6

    "${python_bin}" - "${run_root}" "${selected_ranks}" "${capture_nsys}" "${model_key}" "${output_json}" <<'PY'
import json
import math
import pathlib
import re
import sys

run_root = pathlib.Path(sys.argv[1]).resolve()
selected = [int(value) for value in sys.argv[2].split(",")]
capture_nsys = sys.argv[3] == "1"
model = sys.argv[4]
output_path = pathlib.Path(sys.argv[5])
expected = set(selected)

trace_paths = sorted((run_root / "runtime" / "profiler_log").rglob("*.txt"))
trace_ranks = []
for path in trace_paths:
    match = re.search(r"(?:^|_)rank([0-9]+)(?:_|\.)", path.name)
    if match is None:
        raise SystemExit(f"[ERROR] Cannot determine rank from trace file: {path}")
    trace_ranks.append(int(match.group(1)))
if len(trace_paths) != len(selected) or set(trace_ranks) != expected or len(set(trace_ranks)) != len(trace_ranks):
    raise SystemExit(
        f"[ERROR] Trace rank inventory mismatch: expected={selected}, observed={trace_ranks}"
    )

required_operations = ("forward_step", "backward_step", "optimizer_step")
for path in trace_paths:
    trace_text = path.read_text(encoding="utf-8")
    for operation in required_operations:
        if not re.search(rf"^rank:[0-9]+:{operation}\(", trace_text, re.MULTILINE):
            raise SystemExit(
                f"[ERROR] Task1 trace semantic validation is missing {operation}: {path}"
            )
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
        raise SystemExit(f"[ERROR] Task1 backward trace metadata is incomplete: {path}")
    if not re.search(
        r"^rank:[0-9]+:ddp_grad_comm\([^\n]*trigger_cmd_uid=",
        trace_text,
        re.MULTILINE,
    ):
        raise SystemExit(f"[ERROR] Task1 trace lacks DDP-overlap trigger metadata: {path}")

memory_paths = sorted((run_root / "runtime" / "memory_traces_scaling").glob("*.json"))
memory_ranks = []
per_rank_peak = {}
for path in memory_paths:
    match = re.search(r"memory_trace_rank([0-9]+)(?:_|\.)", path.name)
    if match is None:
        raise SystemExit(f"[ERROR] Cannot determine rank from memory file: {path}")
    rank = int(match.group(1))
    memory_ranks.append(rank)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"[ERROR] Empty memory payload: {path}")
    peaks = []
    positive_sample = False
    for iteration in payload.values():
        if not isinstance(iteration, dict):
            raise SystemExit(f"[ERROR] Invalid memory iteration payload: {path}")
        peak = iteration.get("peak_allocated_MB")
        if not isinstance(peak, (int, float)) or isinstance(peak, bool) or not math.isfinite(peak) or peak <= 0:
            raise SystemExit(f"[ERROR] Non-positive peak_allocated_MB in {path}: {peak}")
        peaks.append(float(peak))
        samples = iteration.get("samples")
        if not isinstance(samples, list) or not samples:
            raise SystemExit(f"[ERROR] Memory samples are empty: {path}")
        for sample in samples:
            for field in ("reserved_memory_MB", "allocated_memory_MB"):
                value = sample.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise SystemExit(f"[ERROR] Invalid {field} in {path}: {value}")
                positive_sample = positive_sample or value > 0
    if not positive_sample:
        raise SystemExit(f"[ERROR] Memory samples never report positive usage: {path}")
    per_rank_peak[str(rank)] = max(peaks)

if len(memory_paths) != len(selected) or set(memory_ranks) != expected or len(set(memory_ranks)) != len(memory_ranks):
    raise SystemExit(
        f"[ERROR] Memory rank inventory mismatch: expected={selected}, observed={memory_ranks}"
    )

nsys_rep = run_root / "nsys" / f"{model}.nsys-rep"
sqlite = run_root / "nsys" / f"{model}.sqlite"
if capture_nsys:
    for path in (nsys_rep, sqlite):
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"[ERROR] Required Nsight artifact is missing or empty: {path}")

result = {
    "trace_file_count": len(trace_paths),
    "memory_json_count": len(memory_paths),
    "per_rank_peak_allocated_mb": per_rank_peak,
    "maximum_peak_allocated_mb": max(per_rank_peak.values()),
    "nsys_rep_path": "nsys/{}.nsys-rep".format(model) if capture_nsys else None,
    "sqlite_path": "nsys/{}.sqlite".format(model) if capture_nsys else None,
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

ae_task1_rank0_elapsed_seconds() {
    local python_bin=$1
    local timing_path=$2
    local selected_ranks=$3

    "${python_bin}" - "${timing_path}" "${selected_ranks}" <<'PY'
import pathlib
import sys

timing_path = pathlib.Path(sys.argv[1])
selected = [int(value) for value in sys.argv[2].split(",")]
if not timing_path.is_file() or timing_path.is_symlink():
    raise SystemExit(f"[ERROR] Task1 rank timing log is missing or invalid: {timing_path}")

observed = {}
for line_number, line in enumerate(
    timing_path.read_text(encoding="utf-8").splitlines(), start=1
):
    fields = line.split(",")
    if len(fields) != 4 or any(not field.isdigit() for field in fields):
        raise SystemExit(
            f"[ERROR] Invalid Task1 rank timing line {line_number}: {line!r}"
        )
    rank_id, start_ns, end_ns, elapsed_ns = (int(field) for field in fields)
    if rank_id in observed:
        raise SystemExit(f"[ERROR] Duplicate Task1 rank timing for rank {rank_id}")
    if end_ns <= start_ns or elapsed_ns <= 0 or elapsed_ns != end_ns - start_ns:
        raise SystemExit(
            "[ERROR] Invalid Task1 rank timing interval: "
            f"rank={rank_id}, start={start_ns}, end={end_ns}, elapsed={elapsed_ns}"
        )
    observed[rank_id] = elapsed_ns

if set(observed) != set(selected) or len(observed) != len(selected):
    raise SystemExit(
        "[ERROR] Task1 rank timing inventory mismatch: "
        f"expected={selected}, observed={sorted(observed)}"
    )
if 0 not in observed:
    raise SystemExit("[ERROR] Task1 timing estimate requires fake rank 0 timing")

print(f"{observed[0] / 1_000_000_000:.9f}")
PY
}

ae_task1_d16_gate_result() {
    local estimated_full_seconds=${1:-}
    local python_bin=${2:-}

    [[ -n "${estimated_full_seconds}" ]] || {
        ae_die "Task1 D16 gate requires an estimated full-capture time."
        return 1
    }
    if [[ -z "${python_bin}" ]]; then
        python_bin=$(command -v python3) || {
            ae_die "Task1 D16 gate requires python3."
            return 1
        }
    fi
    [[ -x "${python_bin}" ]] || {
        ae_die "Task1 D16 gate Python is not executable: ${python_bin}"
        return 1
    }

    "${python_bin}" - "${estimated_full_seconds}" <<'PY'
import math
import sys

try:
    estimated_full = float(sys.argv[1])
except (IndexError, ValueError) as exc:
    raise SystemExit("[ERROR] Task1 D16 estimated full time must be numeric") from exc
if not math.isfinite(estimated_full) or estimated_full <= 0:
    raise SystemExit(
        "[ERROR] Task1 D16 estimated full time must be finite and strictly positive"
    )
print("pass" if estimated_full <= 7200 else "prebaked_required")
PY
}

ae_task1_write_d16_preflight_report() {
    local python_bin=$1
    local report_path=$2
    local model_key=$3
    local capture_id=$4
    local preflight_capture_id=$5
    local execution_evidence=$6
    local capture_nsys=$7
    local inventory_path=$8
    local batch_flag_log=$9
    local rank0_elapsed_seconds=${10}
    local gate_enforced=${11}
    local expected_global_batch_size=${12}
    local requested_capture_scope=${13:-}
    local gate_decision_applied=${14:-}

    "${python_bin}" - \
        "${report_path}" "${model_key}" "${capture_id}" "${preflight_capture_id}" \
        "${execution_evidence}" "${capture_nsys}" "${inventory_path}" "${batch_flag_log}" \
        "${rank0_elapsed_seconds}" "${gate_enforced}" "${expected_global_batch_size}" \
        "${requested_capture_scope}" "${gate_decision_applied}" <<'PY'
import json
import math
import pathlib
import sys

(
    report_text,
    model,
    capture_id,
    preflight_capture_id,
    execution_evidence,
    capture_nsys_text,
    inventory_text,
    batch_text,
    rank0_elapsed_text,
    gate_enforced_text,
    expected_global_batch_text,
    requested_capture_scope,
    gate_decision_applied_text,
) = sys.argv[1:]

if model not in {"qwen3_a30b", "dsv3"}:
    raise SystemExit(f"[ERROR] D16 preflight does not support model: {model}")
if capture_nsys_text not in {"0", "1"}:
    raise SystemExit("[ERROR] D16 preflight capture_nsys must be 0 or 1")
if gate_enforced_text not in {"0", "1"}:
    raise SystemExit("[ERROR] D16 preflight gate_enforced must be 0 or 1")
if requested_capture_scope not in {"quick", "full", "representative_ep"}:
    raise SystemExit("[ERROR] D16 preflight requested_capture_scope is invalid")
if gate_decision_applied_text not in {"0", "1"}:
    raise SystemExit("[ERROR] D16 preflight gate_decision_applied must be 0 or 1")
if gate_decision_applied_text != gate_enforced_text:
    raise SystemExit(
        "[ERROR] D16 preflight gate_decision_applied must match d16_gate_enforced"
    )
expected_gate_enforced = "1" if requested_capture_scope == "full" else "0"
if gate_enforced_text != expected_gate_enforced:
    raise SystemExit(
        "[ERROR] D16 preflight gate enforcement must match requested capture scope"
    )
try:
    rank0_elapsed = float(rank0_elapsed_text)
    expected_global_batch = int(expected_global_batch_text)
except ValueError as exc:
    raise SystemExit("[ERROR] Invalid D16 preflight numeric input") from exc
if not math.isfinite(rank0_elapsed) or rank0_elapsed <= 0:
    raise SystemExit("[ERROR] D16 preflight rank-0 elapsed time must be finite and positive")
if expected_global_batch <= 0:
    raise SystemExit("[ERROR] D16 preflight global batch size must be positive")

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


try:
    inventory = json.loads(
        pathlib.Path(inventory_text).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] Cannot read D16 preflight inventory: {exc}") from exc
if not isinstance(inventory, dict):
    raise SystemExit("[ERROR] D16 preflight inventory must be an object")
for field in ("trace_file_count", "memory_json_count", "maximum_peak_allocated_mb"):
    if field not in inventory:
        raise SystemExit(f"[ERROR] D16 preflight inventory is missing {field}")
for field in ("trace_file_count", "memory_json_count"):
    if isinstance(inventory[field], bool) or not isinstance(inventory[field], int):
        raise SystemExit(f"[ERROR] D16 preflight inventory {field} must be integer 1")
if inventory["trace_file_count"] != 1 or inventory["memory_json_count"] != 1:
    raise SystemExit("[ERROR] D16 preflight must contain exactly one trace and one memory file")
peak = inventory["maximum_peak_allocated_mb"]
if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
    raise SystemExit("[ERROR] D16 preflight maximum peak memory must be finite and positive")

batch_path = pathlib.Path(batch_text)
if not batch_path.is_file() or batch_path.is_symlink():
    raise SystemExit(f"[ERROR] D16 preflight batch log is missing or invalid: {batch_path}")
batch_lines = batch_path.read_text(encoding="utf-8").splitlines()
if len(batch_lines) != 1:
    raise SystemExit(
        f"[ERROR] D16 preflight requires one batch-size invocation, observed {len(batch_lines)}"
    )
batch_values = batch_lines[0].split(",")
if not batch_values or any(not value.isdigit() or int(value) <= 0 for value in batch_values):
    raise SystemExit("[ERROR] D16 preflight batch-size provenance is invalid")
batch_values_int = [int(value) for value in batch_values]
if any(value != batch_values_int[0] for value in batch_values_int):
    raise SystemExit("[ERROR] D16 preflight batch-size flags conflict")
if batch_values_int[0] != expected_global_batch:
    raise SystemExit(
        "[ERROR] D16 preflight batch-size differs from the frozen configuration: "
        f"expected={expected_global_batch}, observed={batch_values_int[0]}"
    )

estimated_full = rank0_elapsed * 256
gate_result = "pass" if estimated_full <= 7200 else "prebaked_required"
payload = {
    "schema_version": "sc26-ae-task1-d16-preflight-v1",
    "model": model,
    "capture_id": capture_id,
    "preflight_capture_id": preflight_capture_id,
    "execution_evidence": execution_evidence,
    "capture_scope": "d16_rank0_only",
    "rank_id": 0,
    "selected_rank_ids": [0],
    "selected_rank_count": 1,
    "requested_capture_scope": requested_capture_scope,
    "d16_gate_applicable": True,
    "d16_gate_enforced": gate_enforced_text == "1",
    "gate_decision_applied": gate_decision_applied_text == "1",
    "d16_timing_source": "independent_rank0_preflight",
    "estimate_basis_rank": 0,
    "rank0_elapsed_seconds": rank0_elapsed,
    "estimate_rank_count": 256,
    "estimated_full_seconds": estimated_full,
    "fresh_capture_gate_threshold_seconds": 7200,
    "fresh_capture_gate_result": gate_result,
    "automatic_fallback": False,
    "capture_nsys": capture_nsys_text == "1",
    "trace_file_count": 1,
    "memory_json_count": 1,
    "maximum_peak_allocated_mb": float(peak),
    "effective_global_batch_size": batch_values_int[0],
    "global_batch_size_flag_values": batch_values_int,
}
report_path = pathlib.Path(report_text)
if report_path.exists() or report_path.is_symlink():
    raise SystemExit(f"[ERROR] D16 preflight report destination already exists: {report_path}")
if report_path.parent.exists() and report_path.parent.is_symlink():
    raise SystemExit(f"[ERROR] D16 preflight report parent is a symlink: {report_path.parent}")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

ae_task1_validate_d16_preflight_report() {
    local python_bin=$1
    local report_path=$2
    local expected_model=${3:-}
    local expected_capture_id=${4:-}
    local expected_preflight_capture_id=${5:-}
    local expected_gate_enforced=${6:-}
    local expected_requested_capture_scope=${7:-}
    local expected_gate_decision_applied=${8:-}

    "${python_bin}" - \
        "${report_path}" "${expected_model}" "${expected_capture_id}" \
        "${expected_preflight_capture_id}" "${expected_gate_enforced}" \
        "${expected_requested_capture_scope}" "${expected_gate_decision_applied}" <<'PY'
import json
import math
import pathlib
import sys


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


report_path = pathlib.Path(sys.argv[1])
(
    expected_model,
    expected_capture_id,
    expected_preflight_id,
    expected_gate_text,
    expected_scope,
    expected_decision_text,
) = sys.argv[2:]
if not report_path.is_file() or report_path.is_symlink():
    raise SystemExit(f"[ERROR] D16 preflight report is missing or invalid: {report_path}")
try:
    payload = json.loads(
        report_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] Invalid D16 preflight report JSON: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit("[ERROR] D16 preflight report must be a JSON object")
required = {
    "schema_version", "model", "capture_id", "preflight_capture_id", "execution_evidence",
    "capture_scope", "rank_id", "selected_rank_ids", "selected_rank_count",
    "requested_capture_scope", "d16_gate_applicable", "d16_gate_enforced",
    "gate_decision_applied", "d16_timing_source", "estimate_basis_rank",
    "rank0_elapsed_seconds", "estimate_rank_count", "estimated_full_seconds",
    "fresh_capture_gate_threshold_seconds", "fresh_capture_gate_result", "automatic_fallback",
    "capture_nsys", "trace_file_count", "memory_json_count", "maximum_peak_allocated_mb",
    "effective_global_batch_size", "global_batch_size_flag_values",
}
if set(payload) != required:
    raise SystemExit(
        "[ERROR] D16 preflight report keys are not exact: "
        f"expected={sorted(required)}, observed={sorted(payload)}"
    )
if payload["schema_version"] != "sc26-ae-task1-d16-preflight-v1":
    raise SystemExit("[ERROR] Invalid D16 preflight report schema_version")
if payload["model"] not in {"qwen3_a30b", "dsv3"}:
    raise SystemExit("[ERROR] D16 preflight report has unsupported model")
for field in ("capture_id", "preflight_capture_id", "execution_evidence"):
    if not isinstance(payload[field], str) or not payload[field] or "/" in payload[field] or "\\" in payload[field]:
        raise SystemExit(f"[ERROR] D16 preflight {field} is invalid")
if payload["capture_id"] == payload["preflight_capture_id"]:
    raise SystemExit("[ERROR] D16 preflight capture ids must differ")
if expected_model and payload["model"] != expected_model:
    raise SystemExit("[ERROR] D16 preflight report model mismatch")
if expected_capture_id and payload["capture_id"] != expected_capture_id:
    raise SystemExit("[ERROR] D16 preflight report capture_id mismatch")
if expected_preflight_id and payload["preflight_capture_id"] != expected_preflight_id:
    raise SystemExit("[ERROR] D16 preflight report preflight_capture_id mismatch")
if expected_gate_text:
    if expected_gate_text not in {"0", "1"}:
        raise SystemExit("[ERROR] Invalid expected D16 preflight gate state")
    if payload["d16_gate_enforced"] is not (expected_gate_text == "1"):
        raise SystemExit("[ERROR] D16 preflight gate-enforced state mismatch")
if expected_scope and payload["requested_capture_scope"] != expected_scope:
    raise SystemExit("[ERROR] D16 preflight requested capture scope mismatch")
if expected_decision_text:
    if expected_decision_text not in {"0", "1"}:
        raise SystemExit("[ERROR] Invalid expected D16 gate decision state")
    if payload["gate_decision_applied"] is not (expected_decision_text == "1"):
        raise SystemExit("[ERROR] D16 preflight gate decision state mismatch")
if payload["execution_evidence"] not in {
    "local_synthetic_not_gpu_qualification",
    "runtime_measurement_requires_external_single_gpu_qualification",
}:
    raise SystemExit("[ERROR] D16 preflight execution evidence is not allowed")
if payload["capture_scope"] != "d16_rank0_only":
    raise SystemExit("[ERROR] D16 preflight capture_scope must be d16_rank0_only")
if (
    isinstance(payload["rank_id"], bool)
    or not isinstance(payload["rank_id"], int)
    or payload["rank_id"] != 0
    or not isinstance(payload["selected_rank_ids"], list)
    or len(payload["selected_rank_ids"]) != 1
    or isinstance(payload["selected_rank_ids"][0], bool)
    or not isinstance(payload["selected_rank_ids"][0], int)
    or payload["selected_rank_ids"][0] != 0
    or isinstance(payload["selected_rank_count"], bool)
    or not isinstance(payload["selected_rank_count"], int)
    or payload["selected_rank_count"] != 1
):
    raise SystemExit("[ERROR] D16 preflight must contain exactly fake rank 0")
if payload["d16_gate_applicable"] is not True:
    raise SystemExit("[ERROR] D16 preflight requires d16_gate_applicable=true")
if payload["requested_capture_scope"] not in {"quick", "full", "representative_ep"}:
    raise SystemExit("[ERROR] D16 preflight requested_capture_scope is invalid")
if not isinstance(payload["d16_gate_enforced"], bool):
    raise SystemExit("[ERROR] D16 preflight d16_gate_enforced must be boolean")
if not isinstance(payload["gate_decision_applied"], bool):
    raise SystemExit("[ERROR] D16 preflight gate_decision_applied must be boolean")
if payload["gate_decision_applied"] is not payload["d16_gate_enforced"]:
    raise SystemExit(
        "[ERROR] D16 preflight gate decision must match d16_gate_enforced"
    )
expected_gate_enforced = payload["requested_capture_scope"] == "full"
if payload["d16_gate_enforced"] is not expected_gate_enforced:
    raise SystemExit(
        "[ERROR] D16 preflight gate enforcement must match requested capture scope"
    )
if payload["d16_timing_source"] != "independent_rank0_preflight":
    raise SystemExit("[ERROR] D16 preflight timing source is not independent")
if (
    isinstance(payload["estimate_basis_rank"], bool)
    or not isinstance(payload["estimate_basis_rank"], int)
    or payload["estimate_basis_rank"] != 0
    or isinstance(payload["estimate_rank_count"], bool)
    or not isinstance(payload["estimate_rank_count"], int)
    or payload["estimate_rank_count"] != 256
):
    raise SystemExit("[ERROR] D16 preflight estimate basis/count is invalid")
if (
    isinstance(payload["fresh_capture_gate_threshold_seconds"], bool)
    or not isinstance(payload["fresh_capture_gate_threshold_seconds"], int)
    or payload["fresh_capture_gate_threshold_seconds"] != 7200
):
    raise SystemExit("[ERROR] D16 preflight threshold must be integer 7200")
if payload["automatic_fallback"] is not False:
    raise SystemExit("[ERROR] D16 preflight automatic_fallback must be false")
if not isinstance(payload["capture_nsys"], bool):
    raise SystemExit("[ERROR] D16 preflight capture_nsys must be boolean")
for field in ("trace_file_count", "memory_json_count"):
    if isinstance(payload[field], bool) or not isinstance(payload[field], int) or payload[field] != 1:
        raise SystemExit(f"[ERROR] D16 preflight {field} must equal 1")
for field in ("rank0_elapsed_seconds", "estimated_full_seconds", "maximum_peak_allocated_mb"):
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"[ERROR] D16 preflight {field} must be finite and positive")
expected_estimate = payload["rank0_elapsed_seconds"] * 256
if not math.isclose(payload["estimated_full_seconds"], expected_estimate, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("[ERROR] D16 preflight estimated_full_seconds formula mismatch")
expected_result = "pass" if payload["estimated_full_seconds"] <= 7200 else "prebaked_required"
if payload["fresh_capture_gate_result"] not in {"pass", "prebaked_required"}:
    raise SystemExit("[ERROR] D16 preflight gate result is invalid")
if payload["fresh_capture_gate_result"] != expected_result:
    raise SystemExit("[ERROR] D16 preflight gate result is inconsistent")
if (
    isinstance(payload["effective_global_batch_size"], bool)
    or not isinstance(payload["effective_global_batch_size"], int)
    or payload["effective_global_batch_size"] <= 0
):
    raise SystemExit("[ERROR] D16 preflight effective global batch size is invalid")
values = payload["global_batch_size_flag_values"]
if not isinstance(values, list) or not values or any(
    isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
):
    raise SystemExit("[ERROR] D16 preflight global batch flag values are invalid")
if any(value != values[0] for value in values) or values[0] != payload["effective_global_batch_size"]:
    raise SystemExit("[ERROR] D16 preflight global batch flag values conflict")
PY
}

ae_task1_write_metadata_and_summary() {
    local python_bin=$1
    local metadata_path=$2
    local summary_path=$3
    local inventory_path=$4
    local model_key=$5
    local capture_id=$6
    local selected_ranks=$7
    local elapsed_seconds=$8
    local main_commit=$9
    local echo_commit=${10}
    local sim_commit=${11}
    local capture_nsys=${12}
    local batch_flag_log=${13}
    local execution_evidence=${14}
    local capture_scope=${15}
    local single_rank_elapsed_seconds=${16-}
    local selected_capture_rank0_elapsed_seconds=${17-}
    local d16_preflight_capture_id=${18-}
    local d16_timing_source=${19-}
    local d16_gate_enforced=${20-}
    local d16_preflight_report_sha256=${21-}
    local d16_gate_decision_applied=${22-}

    "${python_bin}" - \
        "${metadata_path}" "${summary_path}" "${inventory_path}" \
        "${model_key}" "${capture_id}" "${selected_ranks}" "${elapsed_seconds}" \
        "${main_commit}" "${echo_commit}" "${sim_commit}" "${capture_nsys}" \
        "${batch_flag_log}" \
        "${execution_evidence}" "${capture_scope}" \
        "${single_rank_elapsed_seconds}" \
        "${selected_capture_rank0_elapsed_seconds}" "${d16_preflight_capture_id}" \
        "${d16_timing_source}" "${d16_gate_enforced}" "${d16_preflight_report_sha256}" \
        "${d16_gate_decision_applied}" \
        "${AE_T1_PROFILE}" "${AE_T1_WORLD_SIZE}" "${AE_T1_LOCAL_SIZE}" \
        "${AE_T1_PP}" "${AE_T1_TP}" "${AE_T1_DP}" "${AE_T1_EXP}" \
        "${AE_T1_FAKE_GPUS_PER_NODE}" "${AE_T1_GLOBAL_BATCH_SIZE}" \
        "${AE_TASK1_SKIP_SOURCE_PROVENANCE:-0}" <<'PY'
import json
import math
import pathlib
import sys

(
    metadata_text,
    summary_text,
    inventory_text,
    model,
    capture_id,
    selected_csv,
    elapsed_text,
    main_commit,
    echo_commit,
    sim_commit,
    capture_nsys_text,
    batch_flag_text,
    execution_evidence,
    capture_scope,
    single_rank_elapsed_text,
    selected_capture_rank0_elapsed_text,
    d16_preflight_capture_id,
    d16_timing_source,
    d16_gate_enforced_text,
    d16_preflight_report_sha256,
    d16_gate_decision_applied_text,
    profile,
    world_size_text,
    local_size_text,
    pp_text,
    tp_text,
    dp_text,
    exp_text,
    fake_node_text,
    global_batch_text,
    source_provenance_bypass_text,
) = sys.argv[1:]
inventory = json.loads(pathlib.Path(inventory_text).read_text(encoding="utf-8"))
selected = [int(value) for value in selected_csv.split(",")]
capture_nsys = capture_nsys_text == "1"
expected_global_batch_size = int(global_batch_text)
if source_provenance_bypass_text not in {"0", "1"}:
    raise SystemExit("[ERROR] Task1 source-provenance bypass must be 0 or 1")
try:
    single_rank_elapsed_seconds = float(single_rank_elapsed_text)
except ValueError as exc:
    raise SystemExit(
        "[ERROR] Task1 single_rank_elapsed_seconds must be numeric: "
        f"{single_rank_elapsed_text!r}"
    ) from exc
if not math.isfinite(single_rank_elapsed_seconds) or single_rank_elapsed_seconds <= 0:
    raise SystemExit(
        "[ERROR] Task1 single_rank_elapsed_seconds must be finite and strictly positive: "
        f"{single_rank_elapsed_text!r}"
    )
selected_capture_rank0_elapsed_seconds = None
if selected_capture_rank0_elapsed_text:
    try:
        selected_capture_rank0_elapsed_seconds = float(selected_capture_rank0_elapsed_text)
    except ValueError as exc:
        raise SystemExit(
            "[ERROR] Task1 selected_capture_rank0_elapsed_seconds must be numeric: "
            f"{selected_capture_rank0_elapsed_text!r}"
        ) from exc
    if not math.isfinite(selected_capture_rank0_elapsed_seconds) or selected_capture_rank0_elapsed_seconds <= 0:
        raise SystemExit(
            "[ERROR] Task1 selected_capture_rank0_elapsed_seconds must be finite and strictly positive"
        )
if model == "gpt175b":
    d16_gate_applicable = False
    estimate_rank_count = 8
    if len(selected) != estimate_rank_count or capture_scope != "representative":
        raise SystemExit(
            "[ERROR] GPT Task1 timing estimate requires exactly eight selected "
            "representative ranks"
        )
elif model in {"qwen3_a30b", "dsv3"}:
    d16_gate_applicable = True
    estimate_rank_count = 256
    if int(world_size_text) != estimate_rank_count:
        raise SystemExit(
            "[ERROR] MoE Task1 D16 timing requires simulation world size 256"
        )
else:
    raise SystemExit(f"[ERROR] Unsupported Task1 model for timing metadata: {model}")
if not d16_gate_applicable and any(
    value
    for value in (
        selected_capture_rank0_elapsed_text,
        d16_preflight_capture_id,
        d16_timing_source,
        d16_gate_enforced_text,
        d16_preflight_report_sha256,
        d16_gate_decision_applied_text,
    )
):
    raise SystemExit("[ERROR] GPT timing metadata must omit D16 preflight provenance fields")
if d16_gate_applicable and any(
    value
    for value in (
        d16_preflight_capture_id,
        d16_timing_source,
        d16_gate_enforced_text,
        d16_preflight_report_sha256,
        d16_gate_decision_applied_text,
    )
):
    if not selected_capture_rank0_elapsed_text:
        raise SystemExit(
            "[ERROR] Task1 D16 provenance requires selected-capture rank-0 timing"
        )
    if not d16_preflight_capture_id:
        raise SystemExit("[ERROR] Task1 D16 provenance is missing preflight capture id")
    if d16_timing_source != "independent_rank0_preflight":
        raise SystemExit("[ERROR] Task1 D16 timing source must be independent_rank0_preflight")
    if d16_gate_enforced_text not in {"0", "1"}:
        raise SystemExit("[ERROR] Task1 D16 gate-enforced value must be 0 or 1")
    if d16_gate_decision_applied_text not in {"0", "1"}:
        raise SystemExit("[ERROR] Task1 D16 gate-decision-applied value must be 0 or 1")
    if d16_gate_decision_applied_text != d16_gate_enforced_text:
        raise SystemExit(
            "[ERROR] Task1 D16 gate-decision-applied must match d16_gate_enforced"
        )
    if not __import__("re").fullmatch(r"[0-9a-f]{64}", d16_preflight_report_sha256):
        raise SystemExit("[ERROR] Task1 D16 preflight report SHA256 is invalid")
elif d16_gate_applicable and selected_capture_rank0_elapsed_text:
    raise SystemExit(
        "[ERROR] Task1 selected-capture timing requires D16 preflight provenance"
    )
estimate_basis_rank = 0
estimated_full_seconds = single_rank_elapsed_seconds * estimate_rank_count
if d16_gate_applicable:
    fresh_capture_gate_threshold_seconds = 7200
    fresh_capture_gate_result = (
        "pass"
        if estimated_full_seconds <= fresh_capture_gate_threshold_seconds
        else "prebaked_required"
    )

batch_flag_path = pathlib.Path(batch_flag_text)
if not batch_flag_path.is_file():
    raise SystemExit(f"[ERROR] Task1 global-batch-size log is missing: {batch_flag_path}")
batch_flag_lines = batch_flag_path.read_text(encoding="utf-8").splitlines()
if len(batch_flag_lines) != len(selected):
    raise SystemExit(
        "[ERROR] Task1 global-batch-size invocation count mismatch: "
        f"expected={len(selected)}, observed={len(batch_flag_lines)}"
    )
batch_flag_vectors = []
for line_number, line in enumerate(batch_flag_lines, start=1):
    fields = line.split(",")
    if not fields or any(not field.isdigit() or int(field) <= 0 for field in fields):
        raise SystemExit(
            "[ERROR] Invalid Task1 global-batch-size log line "
            f"{line_number}: {line!r}"
        )
    values = [int(field) for field in fields]
    if any(value != values[0] for value in values[1:]):
        raise SystemExit(
            "[ERROR] Conflicting Task1 global-batch-size values escaped the adapter: "
            f"line={line_number}, values={values}"
        )
    batch_flag_vectors.append(values)

global_batch_size_flag_values = batch_flag_vectors[0]
for invocation_index, values in enumerate(batch_flag_vectors[1:], start=2):
    if values != global_batch_size_flag_values:
        raise SystemExit(
            "[ERROR] Task1 global-batch-size flag vectors differ across invocations: "
            f"first={global_batch_size_flag_values}, "
            f"invocation_{invocation_index}={values}"
        )
effective_global_batch_size = global_batch_size_flag_values[0]
if effective_global_batch_size != expected_global_batch_size:
    raise SystemExit(
        "[ERROR] Task1 global-batch-size differs from the frozen configuration: "
        f"expected={expected_global_batch_size}, observed={effective_global_batch_size}"
    )

capture_summary = {
    "capture_scope": capture_scope,
    "selected_rank_ids": selected,
    "selected_rank_count": len(selected),
    "trace_file_count": inventory["trace_file_count"],
    "memory_json_count": inventory["memory_json_count"],
    "effective_global_batch_size": effective_global_batch_size,
    "global_batch_size_flag_values": global_batch_size_flag_values,
    "per_rank_peak_allocated_mb": inventory["per_rank_peak_allocated_mb"],
    "maximum_peak_allocated_mb": inventory["maximum_peak_allocated_mb"],
    "capture_elapsed_seconds": int(elapsed_text),
    "d16_gate_applicable": d16_gate_applicable,
    "estimate_basis_rank": estimate_basis_rank,
    "estimate_rank_count": estimate_rank_count,
    "single_rank_elapsed_seconds": single_rank_elapsed_seconds,
    "estimated_full_seconds": estimated_full_seconds,
    "capture_nsys": capture_nsys,
}
if d16_gate_applicable:
    capture_summary.update(
        {
            "fresh_capture_gate_threshold_seconds": fresh_capture_gate_threshold_seconds,
            "fresh_capture_gate_result": fresh_capture_gate_result,
        }
    )
    if d16_preflight_capture_id:
        capture_summary.update(
            {
                "selected_capture_rank0_elapsed_seconds": selected_capture_rank0_elapsed_seconds,
                "d16_preflight_capture_id": d16_preflight_capture_id,
                "d16_timing_source": d16_timing_source,
                "d16_gate_enforced": d16_gate_enforced_text == "1",
                "d16_gate_decision_applied": d16_gate_decision_applied_text == "1",
                "d16_preflight_report_sha256": d16_preflight_report_sha256,
            }
        )
metadata = {
    "schema_version": "sc26-ae-artifact-manifest-v1",
    "model": model,
    "task": "task1",
    "artifact_source": "fresh",
    "capture_id": capture_id,
    "source_commits": {
        "megatron_lm": main_commit,
        "echo_slowdown": echo_commit,
        "megatron_sim_engine": sim_commit,
    },
    "simulation_topology": {
        "world_size": int(world_size_text),
        "local_size": int(local_size_text),
        "pp": int(pp_text),
        "tp": int(tp_text),
        "dp": int(dp_text),
        "exp": int(exp_text),
    },
    "capture_runtime": {
        "physical_gpu_count": 1,
        "fake_gpus_per_node": int(fake_node_text),
        "scaling_min_warmup_iters": 3,
        "scaling_profile_iters": 1,
    },
    "profile": profile,
    "precision": "bf16",
    "mock_data": True,
    "ddp_overlap": True,
    "execution_evidence": execution_evidence,
    "source_provenance": {
        "checked": source_provenance_bypass_text == "0",
        "bypassed": source_provenance_bypass_text == "1",
        "reason": (
            "explicit_runtime_smoke_bypass_dirty_worktree"
            if source_provenance_bypass_text == "1"
            else None
        ),
    },
    "capture_summary": capture_summary,
}
pathlib.Path(metadata_text).write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

summary_lines = [
    f"model={model}",
    f"selected_rank_ids={selected_csv}",
    f"selected_rank_count={len(selected)}",
    f"trace_file_count={inventory['trace_file_count']}",
    f"memory_json_count={inventory['memory_json_count']}",
    f"effective_global_batch_size={effective_global_batch_size}",
    "global_batch_size_flag_values="
    + ",".join(str(value) for value in global_batch_size_flag_values),
    "scaling_min_warmup_iters=3",
    "scaling_profile_iters=1",
    f"capture_runtime_fake_gpus_per_node={int(fake_node_text)}",
    f"simulation_topology_local_size={int(local_size_text)}",
    "per_rank_peak_allocated_mb=" + json.dumps(inventory["per_rank_peak_allocated_mb"], sort_keys=True),
    f"maximum_peak_allocated_mb={inventory['maximum_peak_allocated_mb']}",
    f"capture_id={capture_id}",
    f"capture_elapsed_seconds={int(elapsed_text)}",
    f"d16_gate_applicable={str(d16_gate_applicable).lower()}",
    f"estimate_basis_rank={estimate_basis_rank}",
    f"estimate_rank_count={estimate_rank_count}",
    f"single_rank_elapsed_seconds={single_rank_elapsed_seconds:.9f}",
    f"estimated_full_seconds={estimated_full_seconds:.9f}",
    f"execution_evidence={execution_evidence}",
    f"source_provenance_checked={'false' if source_provenance_bypass_text == '1' else 'true'}",
    f"source_provenance_bypassed={'true' if source_provenance_bypass_text == '1' else 'false'}",
    f"capture_scope={capture_scope}",
]
if d16_gate_applicable:
    summary_lines.extend(
        [
            f"fresh_capture_gate_threshold_seconds={fresh_capture_gate_threshold_seconds}",
            f"fresh_capture_gate_result={fresh_capture_gate_result}",
        ]
    )
    if d16_preflight_capture_id:
        summary_lines.extend(
            [
                f"selected_capture_rank0_elapsed_seconds={selected_capture_rank0_elapsed_seconds:.9f}",
                f"d16_preflight_capture_id={d16_preflight_capture_id}",
                f"d16_timing_source={d16_timing_source}",
                f"d16_gate_enforced={'true' if d16_gate_enforced_text == '1' else 'false'}",
                f"d16_gate_decision_applied={'true' if d16_gate_decision_applied_text == '1' else 'false'}",
                f"d16_preflight_report_sha256={d16_preflight_report_sha256}",
            ]
        )
if capture_nsys:
    summary_lines.extend(
        [
            f"nsys_rep_path={inventory['nsys_rep_path']}",
            f"sqlite_path={inventory['sqlite_path']}",
        ]
    )
pathlib.Path(summary_text).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
PY
}

ae_task1_validate_d16_metadata() {
    local python_bin=$1
    local metadata_path=$2
    local summary_path=$3

    "${python_bin}" - "${metadata_path}" "${summary_path}" <<'PY'
import json
import math
import pathlib
import sys


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


metadata_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
if not metadata_path.is_file() or metadata_path.is_symlink():
    raise SystemExit(f"[ERROR] Task1 metadata is missing or invalid: {metadata_path}")
if not summary_path.is_file() or summary_path.is_symlink():
    raise SystemExit(f"[ERROR] Task1 summary is missing or invalid: {summary_path}")

try:
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] Invalid Task1 metadata JSON: {exc}") from exc
if not isinstance(metadata, dict):
    raise SystemExit("[ERROR] Task1 metadata must be a JSON object")
capture_summary = metadata.get("capture_summary")
if not isinstance(capture_summary, dict):
    raise SystemExit("[ERROR] Task1 metadata capture_summary must be an object")

model = metadata.get("model")
model_contracts = {
    "gpt175b": {"d16_gate_applicable": False, "estimate_rank_count": 8},
    "qwen3_a30b": {"d16_gate_applicable": True, "estimate_rank_count": 256},
    "dsv3": {"d16_gate_applicable": True, "estimate_rank_count": 256},
}
if model not in model_contracts:
    raise SystemExit(f"[ERROR] Task1 D16 metadata has unsupported model: {model!r}")
contract = model_contracts[model]

base_required = (
    "d16_gate_applicable",
    "estimate_basis_rank",
    "estimate_rank_count",
    "single_rank_elapsed_seconds",
    "estimated_full_seconds",
)
for field in base_required:
    if field not in capture_summary:
        raise SystemExit(f"[ERROR] Task1 D16 metadata is missing {field}")

d16_gate_applicable = capture_summary["d16_gate_applicable"]
if not isinstance(d16_gate_applicable, bool):
    raise SystemExit("[ERROR] Task1 d16_gate_applicable must be boolean")
if d16_gate_applicable is not contract["d16_gate_applicable"]:
    if model == "gpt175b":
        raise SystemExit("[ERROR] GPT Task1 timing requires d16_gate_applicable=false")
    raise SystemExit("[ERROR] MoE Task1 timing requires d16_gate_applicable=true")

estimate_basis_rank = capture_summary["estimate_basis_rank"]
if isinstance(estimate_basis_rank, bool) or not isinstance(estimate_basis_rank, int) or estimate_basis_rank != 0:
    raise SystemExit("[ERROR] Task1 estimate_basis_rank must be integer 0")
estimate_rank_count = capture_summary["estimate_rank_count"]
expected_rank_count = contract["estimate_rank_count"]
if isinstance(estimate_rank_count, bool) or not isinstance(estimate_rank_count, int):
    raise SystemExit("[ERROR] Task1 estimate_rank_count must be an integer")
if estimate_rank_count != expected_rank_count:
    if model == "gpt175b":
        raise SystemExit(
            "[ERROR] GPT estimate_rank_count must equal the eight selected representative ranks"
        )
    raise SystemExit("[ERROR] Task1 estimate_rank_count must be integer 256")
if model == "gpt175b":
    selected_rank_count = capture_summary.get("selected_rank_count")
    if (
        isinstance(selected_rank_count, bool)
        or not isinstance(selected_rank_count, int)
        or selected_rank_count != expected_rank_count
    ):
        raise SystemExit(
            "[ERROR] GPT timing metadata requires selected_rank_count=8"
        )

single_rank_elapsed_seconds = capture_summary["single_rank_elapsed_seconds"]
if isinstance(single_rank_elapsed_seconds, bool) or not isinstance(
    single_rank_elapsed_seconds, (int, float)
):
    raise SystemExit("[ERROR] Task1 single_rank_elapsed_seconds must be numeric")
if not math.isfinite(single_rank_elapsed_seconds) or single_rank_elapsed_seconds <= 0:
    raise SystemExit(
        "[ERROR] Task1 single_rank_elapsed_seconds must be finite and strictly positive"
    )

estimated_full_seconds = capture_summary["estimated_full_seconds"]
if isinstance(estimated_full_seconds, bool) or not isinstance(
    estimated_full_seconds, (int, float)
):
    raise SystemExit("[ERROR] Task1 estimated_full_seconds must be numeric")
expected_estimate = single_rank_elapsed_seconds * estimate_rank_count
if not math.isfinite(estimated_full_seconds) or not math.isclose(
    estimated_full_seconds, expected_estimate, rel_tol=0.0, abs_tol=1e-12
):
    raise SystemExit(
        "[ERROR] Task1 estimated_full_seconds does not equal "
        "single_rank_elapsed_seconds * estimate_rank_count"
    )

gate_fields = (
    "fresh_capture_gate_threshold_seconds",
    "fresh_capture_gate_result",
)
threshold = None
result = None
if d16_gate_applicable:
    provenance_fields = (
        "selected_capture_rank0_elapsed_seconds",
        "d16_preflight_capture_id",
        "d16_timing_source",
        "d16_gate_enforced",
        "d16_gate_decision_applied",
        "d16_preflight_report_sha256",
    )
    present_provenance = [field for field in provenance_fields if field in capture_summary]
    if present_provenance and set(present_provenance) != set(provenance_fields):
        raise SystemExit(
            "[ERROR] Task1 D16 provenance fields must be present as an exact set"
        )
    if present_provenance:
        selected_capture_elapsed = capture_summary["selected_capture_rank0_elapsed_seconds"]
        if (
            isinstance(selected_capture_elapsed, bool)
            or not isinstance(selected_capture_elapsed, (int, float))
            or not math.isfinite(selected_capture_elapsed)
            or selected_capture_elapsed <= 0
        ):
            raise SystemExit(
                "[ERROR] Task1 selected_capture_rank0_elapsed_seconds must be finite and positive"
            )
        if not isinstance(capture_summary["d16_preflight_capture_id"], str) or not capture_summary[
            "d16_preflight_capture_id"
        ]:
            raise SystemExit("[ERROR] Task1 D16 preflight capture id is invalid")
        if capture_summary["d16_timing_source"] != "independent_rank0_preflight":
            raise SystemExit("[ERROR] Task1 D16 timing source is not independent")
        for field in ("d16_gate_enforced", "d16_gate_decision_applied"):
            if not isinstance(capture_summary[field], bool):
                raise SystemExit(f"[ERROR] Task1 {field} must be boolean")
        if capture_summary["d16_gate_decision_applied"] is not capture_summary["d16_gate_enforced"]:
            raise SystemExit(
                "[ERROR] Task1 D16 gate decision must match d16_gate_enforced"
            )
        if not __import__("re").fullmatch(
            r"[0-9a-f]{64}", capture_summary["d16_preflight_report_sha256"]
        ):
            raise SystemExit("[ERROR] Task1 D16 preflight report SHA256 is invalid")
    for field in gate_fields:
        if field not in capture_summary:
            raise SystemExit(f"[ERROR] Task1 D16 metadata is missing {field}")
    threshold = capture_summary["fresh_capture_gate_threshold_seconds"]
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold != 7200:
        raise SystemExit(
            "[ERROR] Task1 fresh_capture_gate_threshold_seconds must be integer 7200"
        )
    result = capture_summary["fresh_capture_gate_result"]
    expected_result = (
        "pass" if estimated_full_seconds <= threshold else "prebaked_required"
    )
    if result not in ("pass", "prebaked_required") or result != expected_result:
        raise SystemExit("[ERROR] Task1 fresh_capture_gate_result is inconsistent")
elif any(field in capture_summary for field in gate_fields):
    raise SystemExit(
        "[ERROR] GPT timing metadata must omit D16 gate threshold and result fields"
    )

summary_values = {}
for line_number, line in enumerate(
    summary_path.read_text(encoding="utf-8").splitlines(), start=1
):
    if "=" not in line:
        raise SystemExit(f"[ERROR] Invalid Task1 summary line {line_number}: {line!r}")
    key, value = line.split("=", 1)
    if key in summary_values:
        raise SystemExit(f"[ERROR] Duplicate Task1 summary field: {key}")
    summary_values[key] = value
for field in base_required:
    if field not in summary_values:
        raise SystemExit(f"[ERROR] Task1 summary is missing {field}")
expected_applicable_text = "true" if d16_gate_applicable else "false"
if summary_values["d16_gate_applicable"] != expected_applicable_text:
    raise SystemExit("[ERROR] Task1 summary d16_gate_applicable does not match metadata")
if d16_gate_applicable:
    if present_provenance:
        for field in (
            "selected_capture_rank0_elapsed_seconds",
            "d16_preflight_capture_id",
            "d16_timing_source",
            "d16_gate_enforced",
            "d16_gate_decision_applied",
            "d16_preflight_report_sha256",
        ):
            if field not in summary_values:
                raise SystemExit(f"[ERROR] Task1 summary is missing {field}")
        if summary_values["d16_timing_source"] != "independent_rank0_preflight":
            raise SystemExit("[ERROR] Task1 summary D16 timing source is not independent")
        expected_enforced = "true" if capture_summary["d16_gate_enforced"] else "false"
        expected_applied = "true" if capture_summary["d16_gate_decision_applied"] else "false"
        if summary_values["d16_gate_enforced"] != expected_enforced:
            raise SystemExit("[ERROR] Task1 summary d16_gate_enforced does not match metadata")
        if summary_values["d16_gate_decision_applied"] != expected_applied:
            raise SystemExit(
                "[ERROR] Task1 summary d16_gate_decision_applied does not match metadata"
            )
        if summary_values["d16_preflight_report_sha256"] != capture_summary[
            "d16_preflight_report_sha256"
        ]:
            raise SystemExit("[ERROR] Task1 summary preflight report SHA256 does not match metadata")
    for field in gate_fields:
        if field not in summary_values:
            raise SystemExit(f"[ERROR] Task1 summary is missing {field}")
elif any(field in summary_values for field in gate_fields):
    raise SystemExit("[ERROR] GPT summary must omit D16 gate threshold and result fields")

integer_fields = {
    "estimate_basis_rank": estimate_basis_rank,
    "estimate_rank_count": estimate_rank_count,
}
if d16_gate_applicable:
    integer_fields["fresh_capture_gate_threshold_seconds"] = threshold
for field, expected in integer_fields.items():
    if summary_values[field] != str(expected):
        raise SystemExit(f"[ERROR] Task1 summary {field} does not match metadata")
float_fields = {
    "single_rank_elapsed_seconds": float(single_rank_elapsed_seconds),
    "estimated_full_seconds": float(estimated_full_seconds),
}
for field, expected in float_fields.items():
    try:
        observed = float(summary_values[field])
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Task1 summary {field} must be numeric") from exc
    if not math.isfinite(observed) or not math.isclose(
        observed, expected, rel_tol=0.0, abs_tol=5e-10
    ):
        raise SystemExit(f"[ERROR] Task1 summary {field} does not match metadata")
if d16_gate_applicable and summary_values["fresh_capture_gate_result"] != result:
    raise SystemExit(
        "[ERROR] Task1 summary fresh_capture_gate_result does not match metadata"
    )
PY
}

ae_task1_write_file_list() {
    local python_bin=$1
    local run_root=$2
    local file_list_path=$3
    "${python_bin}" - "${run_root}" "${file_list_path}" <<'PY'
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
paths = []
for path in sorted(root.rglob("*")):
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit(f"[ERROR] Task1 run contains a symlink: {path}")
    if stat.S_ISREG(mode):
        relative = path.relative_to(root).as_posix()
        if relative != "artifact_manifest.json":
            paths.append(relative)
    elif not stat.S_ISDIR(mode):
        raise SystemExit(f"[ERROR] Task1 run contains a special file: {path}")
if not paths:
    raise SystemExit("[ERROR] Task1 run has no payload files")
output.write_text("\n".join(paths) + "\n", encoding="utf-8")
PY
}

ae_task1_publish_marker() {
    local python_bin=$1
    local task_dir=$2
    local model_key=$3
    local capture_id=$4
    local manifest_path=$5
    local marker_path=$6
    local ncu_feature_csv=${7:-}

    "${python_bin}" - "${task_dir}" "${model_key}" "${capture_id}" "${manifest_path}" "${marker_path}" "${ncu_feature_csv}" <<'PY'
import hashlib
import json
import pathlib
import sys

task_dir = pathlib.Path(sys.argv[1]).resolve()
model = sys.argv[2]
capture_id = sys.argv[3]
manifest_path = pathlib.Path(sys.argv[4]).resolve(strict=True)
marker_path = pathlib.Path(sys.argv[5])
ncu_feature_csv_text = sys.argv[6]
run_root = manifest_path.parent
expected_run_root = task_dir / "runs" / capture_id
if run_root != expected_run_root:
    raise SystemExit(
        f"[ERROR] Refusing marker path outside the selected run: {run_root} != {expected_run_root}"
    )
payload = {
    "schema_version": "sc26-ae-task1-capture-marker-v1",
    "model": model,
    "capture_id": capture_id,
    "run_path": f"runs/{capture_id}",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "artifact_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "verified": True,
}
if ncu_feature_csv_text:
    ncu_feature_csv = pathlib.Path(ncu_feature_csv_text).resolve(strict=True)
    if ncu_feature_csv.is_symlink() or not ncu_feature_csv.is_file():
        raise SystemExit("[ERROR] Task1 marker NCU feature CSV is invalid")
    try:
        ncu_feature_csv.relative_to(run_root)
    except ValueError as exc:
        raise SystemExit("[ERROR] Task1 marker NCU feature CSV escapes the run") from exc
    payload.update(
        {
            "ncu_feature_scope": "global_rank_0",
            "ncu_feature_rank_ids": [0],
            "ncu_feature_csv_path": ncu_feature_csv.relative_to(run_root).as_posix(),
            "ncu_feature_csv_sha256": hashlib.sha256(ncu_feature_csv.read_bytes()).hexdigest(),
        }
    )
marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

# Execute one isolated scaling capture. The caller decides whether this is
# the independent rank-0 preflight or the selected-rank capture; this helper
# never publishes metadata, manifests, or markers.
ae_task1_execute_capture() {
    local python_bin=$1
    local real_torchrun=$2
    local nsys_bin=${3:-}
    local model_key=$4
    local capture_id=$5
    local selected_ranks=$6
    local scale_gpu=$7
    local capture_nsys=$8
    local capture_root=$9
    local work_root=${10}
    local runtime_dir logs_dir nsys_dir adapter_bin_dir adapter_path loop_path
    local source_log batch_flag_log rank_timing_log inventory_path
    local start_seconds end_seconds source_status tee_status
    local -a pipeline_status=()

    [[ ! -e "${capture_root}" ]] || {
        ae_die "Task1 capture root already exists: ${capture_root}"
        return 1
    }
    [[ ! -e "${work_root}" ]] || {
        ae_die "Task1 capture work root already exists: ${work_root}"
        return 1
    }
    [[ "${capture_nsys}" == "0" || "${capture_nsys}" == "1" ]] || {
        ae_die "Task1 capture_nsys must be 0 or 1."
        return 1
    }
    if [[ "${capture_nsys}" == "1" ]]; then
        [[ -n "${nsys_bin}" && -x "${nsys_bin}" ]] || {
            ae_die "Task1 Nsight Systems executable is required for this capture."
            return 1
        }
    fi

    runtime_dir="${capture_root}/runtime"
    logs_dir="${capture_root}/logs"
    nsys_dir="${capture_root}/nsys"
    adapter_bin_dir="${work_root}/bin"
    adapter_path="${adapter_bin_dir}/torchrun"
    loop_path="${work_root}/selected_rank_loop.sh"
    source_log="${logs_dir}/source.log"
    batch_flag_log="${logs_dir}/global_batch_size_flags.log"
    rank_timing_log="${logs_dir}/rank_timings.log"
    inventory_path="${work_root}/inventory.json"

    mkdir -p "${runtime_dir}" "${logs_dir}" "${nsys_dir}" "${adapter_bin_dir}"
    AE_TASK1_REAL_TORCHRUN=${real_torchrun}
    export AE_TASK1_REAL_TORCHRUN
    ae_task1_write_torchrun_adapter "${adapter_path}"
    ae_task1_write_rank_loop \
        "${loop_path}" "${runtime_dir}" "${adapter_bin_dir}" "${AE_T1_SOURCE_SCRIPT}" \
        "${model_key}" "${selected_ranks}" "${scale_gpu}" "${capture_id}" \
        "${batch_flag_log}" "${rank_timing_log}"

    start_seconds=$(date +%s)
    set +e
    if [[ "${capture_nsys}" == "1" ]]; then
        "${nsys_bin}" profile \
            --trace=cuda,nvtx,osrt \
            --sample=none \
            --wait=all \
            --trace-fork-before-exec=true \
            --force-overwrite true \
            --output "${nsys_dir}/${model_key}" \
            bash "${loop_path}" 2>&1 | tee "${source_log}"
    else
        bash "${loop_path}" 2>&1 | tee "${source_log}"
    fi
    pipeline_status=("${PIPESTATUS[@]}")
    source_status=${pipeline_status[0]}
    tee_status=${pipeline_status[1]}
    set -e
    ((tee_status == 0)) || {
        ae_die "Failed to persist Task1 source log: ${source_log}"
        return 1
    }
    ((source_status == 0)) || {
        ae_die "Task1 source execution failed with status ${source_status}."
        return 1
    }

    if [[ "${capture_nsys}" == "1" ]]; then
        "${nsys_bin}" export \
            -t sqlite \
            --force-overwrite true \
            -o "${nsys_dir}/${model_key}.sqlite" \
            "${nsys_dir}/${model_key}.nsys-rep" \
            >> "${source_log}" 2>&1 || {
                ae_die "Nsight Systems export failed for ${model_key}."
                return 1
            }
    fi
    end_seconds=$(date +%s)

    ae_task1_validate_outputs \
        "${python_bin}" "${capture_root}" "${selected_ranks}" "${capture_nsys}" \
        "${model_key}" "${inventory_path}" || return 1

    AE_T1_CAPTURE_ROOT=${capture_root}
    AE_T1_CAPTURE_WORK_ROOT=${work_root}
    AE_T1_CAPTURE_RUNTIME_DIR=${runtime_dir}
    AE_T1_CAPTURE_LOGS_DIR=${logs_dir}
    AE_T1_CAPTURE_NSYS_DIR=${nsys_dir}
    AE_T1_CAPTURE_SOURCE_LOG=${source_log}
    AE_T1_CAPTURE_BATCH_FLAG_LOG=${batch_flag_log}
    AE_T1_CAPTURE_RANK_TIMING_LOG=${rank_timing_log}
    AE_T1_CAPTURE_INVENTORY_PATH=${inventory_path}
    AE_T1_CAPTURE_ELAPSED_SECONDS=$((end_seconds - start_seconds))
    AE_T1_CAPTURE_RANK0_ELAPSED_SECONDS=$(ae_task1_rank0_elapsed_seconds \
        "${python_bin}" "${rank_timing_log}" "${selected_ranks}") || return 1
    export AE_T1_CAPTURE_ROOT AE_T1_CAPTURE_WORK_ROOT AE_T1_CAPTURE_RUNTIME_DIR \
        AE_T1_CAPTURE_LOGS_DIR AE_T1_CAPTURE_NSYS_DIR AE_T1_CAPTURE_SOURCE_LOG \
        AE_T1_CAPTURE_BATCH_FLAG_LOG AE_T1_CAPTURE_RANK_TIMING_LOG \
        AE_T1_CAPTURE_INVENTORY_PATH AE_T1_CAPTURE_ELAPSED_SECONDS \
        AE_T1_CAPTURE_RANK0_ELAPSED_SECONDS
}

ae_task1_copy_preflight_report() {
    local python_bin=$1
    local source_path=$2
    local destination_path=$3

    "${python_bin}" - "${source_path}" "${destination_path}" <<'PY'
import hashlib
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
if source.is_symlink() or not source.is_file():
    raise SystemExit(f"[ERROR] Preflight report source is not a regular file: {source}")
if not destination.is_absolute() or ".." in destination.parts:
    raise SystemExit(f"[ERROR] Preflight provenance destination must be an absolute safe path: {destination}")
if destination.exists() or destination.is_symlink():
    raise SystemExit(f"[ERROR] Preflight provenance destination already exists: {destination}")


def assert_safe_parent_chain(path):
    current = pathlib.Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise SystemExit(f"[ERROR] Preflight provenance parent is a symlink: {current}")
        if current.exists() and not current.is_dir():
            raise SystemExit(f"[ERROR] Preflight provenance parent is not a directory: {current}")


assert_safe_parent_chain(destination.parent)
destination.parent.mkdir(parents=True, exist_ok=True)
assert_safe_parent_chain(destination.parent)
if destination.parent.is_symlink() or not destination.parent.is_dir():
    raise SystemExit(f"[ERROR] Preflight provenance parent is invalid: {destination.parent}")
shutil.copyfile(source, destination)
if destination.is_symlink() or not destination.is_file():
    raise SystemExit(f"[ERROR] Preflight provenance copy is invalid: {destination}")
source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
destination_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
if source_digest != destination_digest:
    raise SystemExit("[ERROR] Preflight provenance copy changed bytes")
print(destination_digest)
PY
}

ae_task1_validate_preflight_binding() {
    local python_bin=$1
    local report_path=$2
    local copied_path=$3
    local expected_model=$4
    local expected_capture_id=$5
    local expected_preflight_id=$6
    local expected_scope=$7
    local expected_gate_enforced=$8
    local expected_gate_applied=$9
    local expected_rank0_elapsed=${10}

    "${python_bin}" - "${report_path}" "${copied_path}" "${expected_model}" \
        "${expected_capture_id}" "${expected_preflight_id}" "${expected_scope}" \
        "${expected_gate_enforced}" "${expected_gate_applied}" \
        "${expected_rank0_elapsed}" <<'PY'
import hashlib
import json
import math
import pathlib
import sys


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


source = pathlib.Path(sys.argv[1])
copied = pathlib.Path(sys.argv[2])
expected_model, expected_capture, expected_preflight, expected_scope = sys.argv[3:7]
expected_gate, expected_applied = sys.argv[7:9]
expected_elapsed = float(sys.argv[9])
if source.is_symlink() or not source.is_file() or copied.is_symlink() or not copied.is_file():
    raise SystemExit("[ERROR] D16 preflight provenance files must be regular files")
source_bytes = source.read_bytes()
copied_bytes = copied.read_bytes()
if source_bytes != copied_bytes:
    raise SystemExit("[ERROR] D16 preflight provenance copy does not match source bytes")
digest = hashlib.sha256(copied_bytes).hexdigest()
try:
    payload = json.loads(copied_bytes.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit(f"[ERROR] Invalid copied D16 preflight report: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit("[ERROR] Copied D16 preflight report must be an object")
if payload.get("model") != expected_model or payload.get("capture_id") != expected_capture:
    raise SystemExit("[ERROR] Copied D16 preflight report identity does not match capture")
if payload.get("preflight_capture_id") != expected_preflight:
    raise SystemExit("[ERROR] Copied D16 preflight report id does not match capture")
if payload.get("requested_capture_scope") != expected_scope:
    raise SystemExit("[ERROR] Copied D16 preflight requested scope does not match capture")
if payload.get("d16_gate_enforced") is not (expected_gate == "1"):
    raise SystemExit("[ERROR] Copied D16 preflight gate enforcement does not match capture")
if payload.get("gate_decision_applied") is not (expected_applied == "1"):
    raise SystemExit("[ERROR] Copied D16 preflight gate decision does not match capture")
if not math.isclose(float(payload["rank0_elapsed_seconds"]), expected_elapsed, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("[ERROR] Copied D16 preflight timing does not match measured timing")
print(digest)
PY
}

ae_task1_validate_manifest_preflight_isolation() {
    local python_bin=$1
    local manifest_path=$2
    "${python_bin}" - "${manifest_path}" <<'PY'
import json
import pathlib
import sys

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


try:
    manifest = json.loads(
        pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] Invalid Task1 artifact manifest JSON: {exc}") from exc
if not isinstance(manifest, dict):
    raise SystemExit("[ERROR] Task1 artifact manifest must be an object")
entries = manifest.get("files")
if not isinstance(entries, list):
    raise SystemExit("[ERROR] Task1 artifact manifest files must be a list")
paths = []
for entry in entries:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise SystemExit("[ERROR] Task1 artifact manifest file entries must contain a path")
    paths.append(entry["path"])
allowed = "provenance/d16_preflight.json"
if paths.count(allowed) != 1:
    raise SystemExit(
        "[ERROR] Task1 manifest must contain exactly one copied D16 preflight provenance"
    )
if len(paths) != len(set(paths)):
    raise SystemExit("[ERROR] Task1 manifest contains duplicate artifact paths")
if allowed not in paths:
    raise SystemExit("[ERROR] Task1 manifest is missing copied D16 preflight provenance")
for path in paths:
    if "preflight" in path and path != allowed:
        raise SystemExit(f"[ERROR] Task1 manifest contains raw preflight artifact: {path}")
PY
}

ae_task1_preflight_gate_result() {
    local python_bin=$1
    local report_path=$2
    "${python_bin}" - "${report_path}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload.get("fresh_capture_gate_result")
if result not in {"pass", "prebaked_required"}:
    raise SystemExit("[ERROR] D16 preflight report has no valid gate result")
print(result)
PY
}

ae_run_task1() {
    local model_key=${1:-}
    local repo_root quick capture_nsys capture_ncu selected_capture_nsys test_mode scale_gpu selected_ranks capture_scope
    local enforce_d16
    local python_bin real_torchrun nsys_bin ncu_bin capture_id task_dir run_root work_root
    local ncu_capture_root ncu_work_root
    local execution_evidence main_commit echo_commit sim_commit output_root
    local skip_source_provenance
    local preflight_capture_id preflight_root preflight_work_root preflight_report
    local preflight_gate preflight_gate_enforced preflight_gate_applied
    local preflight_report_sha256 preflight_rank0_elapsed_seconds
    local metadata_path file_list_path manifest_path marker_path summary_path

    repo_root=$(ae_repo_root) || return 1
    quick=${QUICK:-0}
    capture_nsys=${CAPTURE_NSYS:-0}
    capture_ncu=${CAPTURE_NCU:-0}
    selected_capture_nsys=${capture_nsys}
    test_mode=${AE_TASK1_TEST_MODE:-0}
    # Fake-level AE runs must complete the requested rank loop.  The 7200 s
    # D16 estimate is a release-qualification check and is therefore opt-in
    # for real runs; test mode retains the historical matrix coverage.
    enforce_d16=${AE_TASK1_ENFORCE_D16:-0}
    skip_source_provenance=${AE_TASK1_SKIP_SOURCE_PROVENANCE:-0}
    ae_require_enum QUICK "${quick}" 0 1 || return 1
    ae_require_enum CAPTURE_NSYS "${capture_nsys}" 0 1 || return 1
    ae_require_enum CAPTURE_NCU "${capture_ncu}" 0 1 || return 1
    ae_require_enum AE_TASK1_TEST_MODE "${test_mode}" 0 1 || return 1
    ae_require_enum AE_TASK1_ENFORCE_D16 "${enforce_d16}" 0 1 || return 1
    ae_require_enum AE_TASK1_SKIP_SOURCE_PROVENANCE "${skip_source_provenance}" 0 1 || return 1
    if [[ "${test_mode}" == "1" && "${skip_source_provenance}" == "1" ]]; then
        ae_die "AE_TASK1_SKIP_SOURCE_PROVENANCE is only valid for real runtime smoke." || return 1
    fi
    if [[ "${test_mode}" == "0" && "${capture_ncu}" != "1" ]]; then
        ae_die "CAPTURE_NCU=1 is required for real Task1 runs so Task3 has workload-aligned kernel features." || return 1
    fi
    export AE_TASK1_SKIP_SOURCE_PROVENANCE=${skip_source_provenance}
    execution_evidence=$(ae_task1_execution_evidence "${test_mode}") || return 1
    ae_task1_load_config "${model_key}" "${repo_root}" || return 1
    scale_gpu=$(ae_task1_resolve_gpu) || return 1
    selected_ranks=$(ae_task1_selected_ranks "${model_key}" "${quick}") || return 1
    capture_scope=$(ae_task1_capture_scope "${model_key}" "${quick}") || return 1

    if [[ "${test_mode}" == "1" ]]; then
        python_bin=${AE_MEGATRON_PYTHON:?AE_MEGATRON_PYTHON is required in test mode}
        real_torchrun=${AE_TASK1_TORCHRUN:?AE_TASK1_TORCHRUN is required in test mode}
        capture_id=${AE_TASK1_TEST_CAPTURE_ID:?AE_TASK1_TEST_CAPTURE_ID is required in test mode}
        preflight_capture_id=${AE_TASK1_TEST_PREFLIGHT_CAPTURE_ID:-${capture_id}-preflight}
    else
        [[ -z "${AE_TASK1_TEST_CAPTURE_ID:-}" ]] || \
            ae_die "AE_TASK1_TEST_CAPTURE_ID is forbidden outside test mode." || return 1
        [[ -z "${AE_TASK1_TEST_PREFLIGHT_CAPTURE_ID:-}" ]] || \
            ae_die "AE_TASK1_TEST_PREFLIGHT_CAPTURE_ID is forbidden outside test mode." || return 1
        python_bin=/opt/conda/envs/megatron_env/bin/python
        real_torchrun=/opt/conda/envs/megatron_env/bin/torchrun
        capture_id="${model_key}-$(date -u +%Y%m%dT%H%M%SZ)"
        preflight_capture_id="${capture_id}-preflight"
    fi
    [[ "${capture_id}" =~ ^${model_key}-[A-Za-z0-9._-]+$ ]] || \
        ae_die "Invalid capture_id: ${capture_id}" || return 1
    [[ "${preflight_capture_id}" =~ ^${model_key}-[A-Za-z0-9._-]+$ ]] || \
        ae_die "Invalid preflight_capture_id: ${preflight_capture_id}" || return 1
    [[ "${preflight_capture_id}" != "${capture_id}" ]] || \
        ae_die "preflight_capture_id must differ from capture_id." || return 1

    if [[ "${test_mode}" == "0" ]]; then
        main_commit=$(git -C "${repo_root}" rev-parse HEAD) || return 1
        if [[ "${skip_source_provenance}" == "0" ]]; then
            ae_task1_assert_source_provenance "${repo_root}" "${main_commit}" || return 1
        else
            printf '[WARN] Task1 source provenance hash gate bypassed explicitly for runtime smoke; output remains non-qualified.\n' >&2
        fi
    fi
    ae_require_file "${python_bin}" || return 1
    [[ -x "${python_bin}" ]] || ae_die "Megatron Python is not executable: ${python_bin}" || return 1
    ae_require_file "${real_torchrun}" || return 1
    [[ -x "${real_torchrun}" ]] || ae_die "Task1 torchrun is not executable: ${real_torchrun}" || return 1

    if [[ "${capture_nsys}" == "1" ]]; then
        if [[ "${test_mode}" == "1" ]]; then
            nsys_bin=${AE_NSYS_BIN:?AE_NSYS_BIN is required for test-mode Nsight capture}
        else
            ae_require_command nsys || return 1
            nsys_bin=$(command -v nsys)
        fi
        ae_require_file "${nsys_bin}" || return 1
        [[ -x "${nsys_bin}" ]] || ae_die "Nsight Systems is not executable: ${nsys_bin}" || return 1
    fi
    if [[ "${capture_ncu}" == "1" ]]; then
        if [[ "${test_mode}" == "1" ]]; then
            ncu_bin=${AE_NCU_BIN:?AE_NCU_BIN is required for test-mode Nsight Compute capture}
        else
            ae_require_command ncu || return 1
            ncu_bin=$(command -v ncu)
        fi
        ae_require_file "${ncu_bin}" || return 1
        [[ -x "${ncu_bin}" ]] || ae_die "Nsight Compute is not executable: ${ncu_bin}" || return 1
    fi

    task_dir=$(ae_model_output_dir "${model_key}" task1) || return 1
    output_root=${AE_OUTPUT_ROOT:-${repo_root}/SC26-AE/output}
    run_root="${task_dir}/runs/${capture_id}"
    work_root="${output_root}/_work/task1.${capture_id}"
    marker_path="${task_dir}/capture_marker.json"
    [[ ! -e "${run_root}" ]] || {
        ae_die "Task1 run directory already exists: ${run_root}"
        return 1
    }
    [[ ! -e "${work_root}" ]] || {
        ae_die "Task1 work directory already exists: ${work_root}"
        return 1
    }

    if [[ "${model_key}" == "qwen3_a30b" || "${model_key}" == "dsv3" ]]; then
        # Every MoE attempt gets a fresh, isolated rank-0 measurement. QUICK
        # records the gate result but deliberately does not enforce it.
        preflight_root="${output_root}/_work/task1-preflight.${preflight_capture_id}"
        preflight_work_root="${preflight_root}/work"
        preflight_report="${preflight_root}/preflight_result.json"
        preflight_gate_enforced=0
        if [[ "${quick}" == "0" ]]; then
            if [[ "${test_mode}" == "1" || "${enforce_d16}" == "1" ]]; then
                preflight_gate_enforced=1
            fi
        fi
        preflight_gate_applied=${preflight_gate_enforced}

        ae_task1_execute_capture \
            "${python_bin}" "${real_torchrun}" "${nsys_bin:-}" "${model_key}" \
            "${preflight_capture_id}" "0" "${scale_gpu}" "${capture_nsys}" \
            "${preflight_root}" "${preflight_work_root}" || return 1
        ae_task1_write_d16_preflight_report \
            "${python_bin}" "${preflight_report}" "${model_key}" "${capture_id}" \
            "${preflight_capture_id}" "${execution_evidence}" "${capture_nsys}" \
            "${AE_T1_CAPTURE_INVENTORY_PATH}" "${AE_T1_CAPTURE_BATCH_FLAG_LOG}" \
            "${AE_T1_CAPTURE_RANK0_ELAPSED_SECONDS}" "${preflight_gate_enforced}" \
            "${AE_T1_GLOBAL_BATCH_SIZE}" "${capture_scope}" "${preflight_gate_applied}"
        ae_task1_validate_d16_preflight_report \
            "${python_bin}" "${preflight_report}" "${model_key}" "${capture_id}" \
            "${preflight_capture_id}" "${preflight_gate_enforced}" "${capture_scope}" \
            "${preflight_gate_applied}"
        preflight_rank0_elapsed_seconds=${AE_T1_CAPTURE_RANK0_ELAPSED_SECONDS}
        preflight_gate=$(ae_task1_preflight_gate_result "${python_bin}" "${preflight_report}") || return 1
        if [[ "${quick}" == "0" && "${preflight_gate}" != "pass" ]]; then
            printf 'TASK1_STATUS=prebaked_required\n'
            printf 'TASK1_MODEL=%s\n' "${model_key}"
            printf 'TASK1_CAPTURE_ID=%s\n' "${capture_id}"
            printf 'TASK1_PREFLIGHT_REPORT=%s\n' "${preflight_report}"
            return 2
        fi

        # Nsight Systems can become unstable when it follows hundreds of
        # sequential torchrun children in one capture.  Keep the complete
        # fake-rank trace/memory loop intact, but use the independent rank-0
        # preflight Nsight artifact as the workload provenance for the full
        # capture when explicitly requested by the worker harness.
        if [[ "${model_key}" == "dsv3" && "${capture_nsys}" == "1" \
            && "${AE_TASK1_NSYS_RANK0_ONLY:-0}" == "1" ]]; then
            selected_capture_nsys=0
        fi
    fi

    # Full/selected capture roots are created only after the MoE preflight
    # has passed (or after a QUICK observation has been validated).
    ae_task1_execute_capture \
        "${python_bin}" "${real_torchrun}" "${nsys_bin:-}" "${model_key}" \
        "${capture_id}" "${selected_ranks}" "${scale_gpu}" "${selected_capture_nsys}" \
        "${run_root}" "${work_root}" || return 1

    if [[ "${selected_capture_nsys}" == "0" && "${capture_nsys}" == "1" ]]; then
        [[ -f "${preflight_root}/nsys/${model_key}.nsys-rep" \
            && -f "${preflight_root}/nsys/${model_key}.sqlite" ]] || {
            ae_die "Rank-0 preflight Nsight artifacts are required for the selected capture."
            return 1
        }
        mkdir -p "${run_root}/nsys"
        cp -- "${preflight_root}/nsys/${model_key}.nsys-rep" \
            "${run_root}/nsys/${model_key}.nsys-rep"
        cp -- "${preflight_root}/nsys/${model_key}.sqlite" \
            "${run_root}/nsys/${model_key}.sqlite"
        ae_task1_validate_outputs \
            "${python_bin}" "${run_root}" "${selected_ranks}" "1" \
            "${model_key}" "${work_root}/inventory.json" || return 1
    fi

    if [[ "${capture_ncu}" == "1" ]]; then
        ncu_work_root="${output_root}/_work/task1-ncu.${capture_id}"
        ncu_capture_root="${ncu_work_root}/capture"
        ae_task1_execute_ncu_capture \
            "${python_bin}" "${ncu_bin}" "${model_key}" "${capture_id}-ncu" \
            "${scale_gpu}" "${ncu_capture_root}" "${ncu_work_root}/work" || return 1
        mkdir -p "${run_root}/ncu"
        cp -- "${AE_T1_NCU_REPORT}" "${run_root}/ncu/rank0.ncu-rep"
        cp -- "${AE_T1_NCU_DETAILS_CSV}" "${run_root}/ncu/rank0_details.csv"
        cp -- "${AE_T1_NCU_RAW_CSV}" "${run_root}/ncu/rank0_raw.csv"
        cp -- "${AE_T1_NCU_FEATURE_CSV}" "${run_root}/ncu/kernel_metric_output.csv"
        AE_T1_NCU_REPORT="${run_root}/ncu/rank0.ncu-rep"
        AE_T1_NCU_DETAILS_CSV="${run_root}/ncu/rank0_details.csv"
        AE_T1_NCU_RAW_CSV="${run_root}/ncu/rank0_raw.csv"
        AE_T1_NCU_FEATURE_CSV="${run_root}/ncu/kernel_metric_output.csv"
    fi

    metadata_path="${AE_T1_CAPTURE_WORK_ROOT}/metadata.json"
    file_list_path="${AE_T1_CAPTURE_WORK_ROOT}/files.txt"
    manifest_path="${run_root}/artifact_manifest.json"
    summary_path="${AE_T1_CAPTURE_LOGS_DIR}/summary.log"

    if [[ "${model_key}" == "qwen3_a30b" || "${model_key}" == "dsv3" ]]; then
        ae_task1_copy_preflight_report \
            "${python_bin}" "${preflight_report}" \
            "${run_root}/provenance/d16_preflight.json" \
            >"${AE_T1_CAPTURE_WORK_ROOT}/preflight_copy_sha256.txt"
        preflight_report_sha256=$(cat "${AE_T1_CAPTURE_WORK_ROOT}/preflight_copy_sha256.txt")
        ae_task1_validate_preflight_binding \
            "${python_bin}" "${preflight_report}" \
            "${run_root}/provenance/d16_preflight.json" "${model_key}" "${capture_id}" \
            "${preflight_capture_id}" "${capture_scope}" "${preflight_gate_enforced}" \
            "${preflight_gate_applied}" "${preflight_rank0_elapsed_seconds}" >/dev/null
    fi

    if [[ "${test_mode}" == "1" ]]; then
        main_commit=$(git -C "${repo_root}" rev-parse HEAD) || return 1
    else
        if [[ "${skip_source_provenance}" == "0" ]]; then
            ae_task1_assert_source_provenance "${repo_root}" "${main_commit}" || return 1
        fi
    fi
    echo_commit=$(ae_gitlink_commit Echo-slowdown) || return 1
    sim_commit=$(ae_gitlink_commit megatron-sim-engine) || return 1

    if [[ "${model_key}" == "qwen3_a30b" || "${model_key}" == "dsv3" ]]; then
        ae_task1_write_metadata_and_summary \
            "${python_bin}" "${metadata_path}" "${summary_path}" "${AE_T1_CAPTURE_INVENTORY_PATH}" \
            "${model_key}" "${capture_id}" "${selected_ranks}" "${AE_T1_CAPTURE_ELAPSED_SECONDS}" \
            "${main_commit}" "${echo_commit}" "${sim_commit}" "${capture_nsys}" \
            "${AE_T1_CAPTURE_BATCH_FLAG_LOG}" "${execution_evidence}" "${capture_scope}" \
            "${preflight_rank0_elapsed_seconds}" "${AE_T1_CAPTURE_RANK0_ELAPSED_SECONDS}" \
            "${preflight_capture_id}" "independent_rank0_preflight" \
            "${preflight_gate_enforced}" "${preflight_report_sha256}" \
            "${preflight_gate_applied}"
    else
        ae_task1_write_metadata_and_summary \
            "${python_bin}" "${metadata_path}" "${summary_path}" "${AE_T1_CAPTURE_INVENTORY_PATH}" \
            "${model_key}" "${capture_id}" "${selected_ranks}" "${AE_T1_CAPTURE_ELAPSED_SECONDS}" \
            "${main_commit}" "${echo_commit}" "${sim_commit}" "${capture_nsys}" \
            "${AE_T1_CAPTURE_BATCH_FLAG_LOG}" "${execution_evidence}" "${capture_scope}" \
            "${AE_T1_CAPTURE_RANK0_ELAPSED_SECONDS}"
    fi
    if [[ "${capture_ncu}" == "1" ]]; then
        ae_task1_attach_ncu_metadata \
            "${python_bin}" "${metadata_path}" "${run_root}" \
            "${AE_T1_NCU_FEATURE_CSV}" "${AE_T1_NCU_DETAILS_CSV}" "${AE_T1_NCU_RAW_CSV}"
    fi
    ae_task1_validate_d16_metadata \
        "${python_bin}" "${metadata_path}" "${summary_path}" || return 1
    ae_task1_write_file_list "${python_bin}" "${run_root}" "${file_list_path}"
    "${python_bin}" -B "${repo_root}/SC26-AE/tools/artifact_manifest.py" create \
        --root "${run_root}" \
        --metadata-json "${metadata_path}" \
        --file-list "${file_list_path}" \
        --output "${manifest_path}"
    "${python_bin}" -B "${repo_root}/SC26-AE/tools/artifact_manifest.py" verify \
        --root "${run_root}" \
        --manifest "${manifest_path}"
    if [[ "${model_key}" == "qwen3_a30b" || "${model_key}" == "dsv3" ]]; then
        ae_task1_validate_manifest_preflight_isolation "${python_bin}" "${manifest_path}"
    fi
    ae_task1_publish_marker \
        "${python_bin}" "${task_dir}" "${model_key}" "${capture_id}" \
        "${manifest_path}" "${marker_path}" \
        "${AE_T1_NCU_FEATURE_CSV:-}"

    printf 'TASK1_STATUS=verified\n'
    printf 'TASK1_MODEL=%s\n' "${model_key}"
    printf 'TASK1_CAPTURE_ID=%s\n' "${capture_id}"
    printf 'TASK1_RUN_ROOT=%s\n' "${run_root}"
    printf 'TASK1_MARKER=%s\n' "${marker_path}"
}

# Capture Nsight Compute features for only global fake rank 0.  This pass is
# separate from Nsight Systems so the full Task1 trace inventory is preserved.
ae_task1_execute_ncu_capture() {
    local python_bin=$1
    local ncu_bin=$2
    local model_key=$3
    local capture_id=$4
    local scale_gpu=$5
    local capture_root=$6
    local work_root=$7
    local runtime_dir logs_dir ncu_dir adapter_bin_dir adapter_path loop_path
    local source_log batch_flag_log rank_timing_log report_base source_status tee_status
    local -a pipeline_status=()

    [[ ! -e "${capture_root}" ]] || { ae_die "Task1 NCU capture root already exists: ${capture_root}"; return 1; }
    [[ ! -e "${work_root}" ]] || { ae_die "Task1 NCU capture work root already exists: ${work_root}"; return 1; }
    [[ -x "${ncu_bin}" ]] || { ae_die "Task1 Nsight Compute executable is required: ${ncu_bin}"; return 1; }

    runtime_dir="${work_root}/runtime"
    logs_dir="${work_root}/logs"
    ncu_dir="${capture_root}/ncu"
    adapter_bin_dir="${work_root}/bin"
    adapter_path="${adapter_bin_dir}/torchrun"
    loop_path="${work_root}/rank0_loop.sh"
    source_log="${logs_dir}/source.log"
    batch_flag_log="${logs_dir}/global_batch_size_flags.log"
    rank_timing_log="${logs_dir}/rank_timings.log"
    report_base="${ncu_dir}/rank0"
    mkdir -p "${runtime_dir}" "${logs_dir}" "${ncu_dir}" "${adapter_bin_dir}"

    export AE_TASK1_REAL_TORCHRUN
    ae_task1_write_torchrun_adapter "${adapter_path}"
    ae_task1_write_rank_loop \
        "${loop_path}" "${runtime_dir}" "${adapter_bin_dir}" "${AE_T1_SOURCE_SCRIPT}" \
        "${model_key}" "0" "${scale_gpu}" "${capture_id}" \
        "${batch_flag_log}" "${rank_timing_log}"

    set +e
    "${ncu_bin}" -o "${report_base}" -f \
        --replay-mode application --app-replay-mode relaxed \
        --target-processes all --device 0 --kernel-name-base function \
        --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
        bash "${loop_path}" 2>&1 | tee "${source_log}"
    pipeline_status=("${PIPESTATUS[@]}")
    source_status=${pipeline_status[0]}
    tee_status=${pipeline_status[1]}
    set -e
    ((tee_status == 0)) || { ae_die "Failed to persist Task1 NCU source log: ${source_log}"; return 1; }
    ((source_status == 0)) || { ae_die "Task1 NCU rank-0 execution failed with status ${source_status}."; return 1; }
    [[ -f "${report_base}.ncu-rep" ]] || { ae_die "Task1 NCU report was not produced: ${report_base}.ncu-rep"; return 1; }
    "${ncu_bin}" -i "${report_base}.ncu-rep" --page details --csv \
        --log-file "${report_base}_details.csv" >>"${source_log}" 2>&1
    "${ncu_bin}" -i "${report_base}.ncu-rep" --print-kernel-base function --csv \
        >"${report_base}_raw.csv" 2>>"${source_log}"
    ae_require_file "${report_base}_details.csv"
    ae_require_file "${report_base}_raw.csv"
    "${python_bin}" -B "$(ae_repo_root)/SC26-AE/tools/normalize_ncu_metrics.py" \
        --details-csv "${report_base}_details.csv" \
        --kernel-names-csv "${report_base}_raw.csv" \
        --output-csv "${ncu_dir}/kernel_metric_output.csv" \
        >>"${source_log}" 2>&1 || { ae_die "Task1 NCU CSV normalization failed for ${model_key}."; return 1; }

    AE_T1_NCU_ROOT=${capture_root}
    AE_T1_NCU_DIR=${ncu_dir}
    AE_T1_NCU_REPORT=${report_base}.ncu-rep
    AE_T1_NCU_DETAILS_CSV=${report_base}_details.csv
    AE_T1_NCU_RAW_CSV=${report_base}_raw.csv
    AE_T1_NCU_FEATURE_CSV=${ncu_dir}/kernel_metric_output.csv
    AE_T1_NCU_SOURCE_LOG=${source_log}
    export AE_T1_NCU_ROOT AE_T1_NCU_DIR AE_T1_NCU_REPORT AE_T1_NCU_DETAILS_CSV \
        AE_T1_NCU_RAW_CSV AE_T1_NCU_FEATURE_CSV AE_T1_NCU_SOURCE_LOG
}

ae_task1_attach_ncu_metadata() {
    local python_bin=$1 metadata_path=$2 run_root=$3 feature_csv=$4 details_csv=$5 raw_csv=$6
    "${python_bin}" - "${metadata_path}" "${run_root}" "${feature_csv}" "${details_csv}" "${raw_csv}" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

metadata_path, run_root_text, feature_text, details_text, raw_text = sys.argv[1:]
run_root = pathlib.Path(run_root_text).resolve(strict=True)
feature = pathlib.Path(feature_text).resolve(strict=True)
details = pathlib.Path(details_text).resolve(strict=True)
raw = pathlib.Path(raw_text).resolve(strict=True)
required = {
    "Kernel Name", "Compute throughput", "Memory throughput", "DRAM throughput",
    "Achieved occupancy", "Maximum occupancy", "L1 hit rate", "L2 hit rate",
}
with feature.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise SystemExit("[ERROR] Task1 NCU feature CSV is missing required columns")
    rows = list(reader)
if not rows:
    raise SystemExit("[ERROR] Task1 NCU feature CSV contains no rows")
kernels = {str(row.get("Kernel Name", "")).strip() for row in rows}
if "" in kernels or "nan" in kernels:
    raise SystemExit("[ERROR] Task1 NCU feature CSV contains an empty kernel name")
for path in (feature, details, raw):
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"[ERROR] Task1 NCU artifact is missing or empty: {path}")
    try:
        relative = path.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Task1 NCU artifact escapes the Task1 run: {path}") from exc
    if not relative.startswith("ncu/"):
        raise SystemExit(f"[ERROR] Task1 NCU artifact is outside ncu/: {path}")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

metadata = json.loads(pathlib.Path(metadata_path).read_text(encoding="utf-8"))
metadata["ncu_feature_provenance"] = {
    "enabled": True,
    "rank_scope": "global_rank_0",
    "rank_ids": [0],
    "physical_gpu_count": 1,
    "feature_csv_relative": feature.relative_to(run_root).as_posix(),
    "details_csv_relative": details.relative_to(run_root).as_posix(),
    "raw_csv_relative": raw.relative_to(run_root).as_posix(),
    "feature_csv_sha256": digest(feature),
    "details_csv_sha256": digest(details),
    "raw_csv_sha256": digest(raw),
    "required_kernel_count": len(kernels),
    "missing_kernel_count": 0,
    "required_columns": sorted(required),
}
pathlib.Path(metadata_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
