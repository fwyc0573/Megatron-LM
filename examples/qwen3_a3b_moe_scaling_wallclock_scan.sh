#!/bin/bash

set -euo pipefail

# Runtime environment defaults.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

SCALE_GPU=${SCALE_GPU:-}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-6400}
OUTPUT_CSV=${OUTPUT_CSV:-"${PROJECT_ROOT}/docs/data/qwen3_a3b_moe_scaling_wallclock_timing.csv"}
LOG_ROOT=${LOG_ROOT:-"${PROJECT_ROOT}/log/qwen3_a3b_moe_scaling_wallclock"}
DRY_RUN=${DRY_RUN:-0}
APPEND_CSV=${APPEND_CSV:-0}

NNODES=1
GPUS_PER_NODE=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

MICRO_BATCH_SIZE=1
SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=40960
VOCAB_SIZE=151936

NUM_LAYERS=48
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=6144
NUM_HEADS=32
NUM_QUERY_GROUPS=8
NUM_EXPERTS=128
MOE_FFN_HIDDEN_SIZE=768
MOE_ROUTER_TOPK=8

TRAIN_ITERS=1
TRACE_START=1
TRACE_SUBOP_SYNC_MODE=global
SCALING_MIN_WARMUP_ITERS=0
SCALING_PROFILE_ITERS=1
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}

# Fixed configurations: world_size pp_size tp_size ep_size dp_size
CONFIGS=(
  "256 8 8 4 4"
  "1024 8 8 16 16"
  "4096 16 8 32 32"
  "8192 16 8 64 64"
)

CONFIG_START_INDEX=${CONFIG_START_INDEX:-0}
CONFIG_END_INDEX=${CONFIG_END_INDEX:-$(( ${#CONFIGS[@]} - 1 ))}

pick_idle_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Please set SCALE_GPU explicitly." >&2
    return 1
  fi

  local candidate
  candidate=$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t',' -k2,2n \
      | head -n1 \
      | cut -d',' -f1 \
      | tr -d ' '
  )

  if [[ -z "${candidate}" ]]; then
    echo "[ERROR] Failed to auto-select idle GPU. Please set SCALE_GPU explicitly." >&2
    return 1
  fi

  echo "${candidate}"
}

validate_moe_config() {
  local world_size=$1
  local pp_size=$2
  local tp_size=$3
  local ep_size=$4
  local dp_size=$5

  if (( world_size <= 0 || pp_size <= 0 || tp_size <= 0 || ep_size <= 0 || dp_size <= 0 )); then
    echo "[ERROR] Invalid non-positive parallel size: ws=${world_size} pp=${pp_size} tp=${tp_size} ep=${ep_size} dp=${dp_size}" >&2
    return 1
  fi

  if (( world_size != pp_size * tp_size * dp_size )); then
    echo "[ERROR] Invalid config: ws=${world_size}, pp=${pp_size}, tp=${tp_size}, dp=${dp_size}. Require ws == pp*tp*dp." >&2
    return 1
  fi

  if (( dp_size % ep_size != 0 )); then
    echo "[ERROR] Invalid config: dp=${dp_size}, ep=${ep_size}. Require dp % ep == 0 for MoE." >&2
    return 1
  fi

  if (( ep_size > dp_size )); then
    echo "[ERROR] Invalid config: ep=${ep_size}, dp=${dp_size}. Require ep <= dp." >&2
    return 1
  fi

  return 0
}

build_selected_ranks() {
  local pp_size=$1
  local tp_size=$2
  local ep_size=$3
  local dp_size=$4
  local ranks=()

  # Representative rank mapping for MoE rank skipping:
  # rank = pp_stage * tp_size * dp_size + exp_rank * tp_size
  for ((pp_stage=0; pp_stage<pp_size; pp_stage++)); do
    for ((exp_rank=0; exp_rank<ep_size; exp_rank++)); do
      ranks+=("$((pp_stage * tp_size * dp_size + exp_rank * tp_size))")
    done
  done

  echo "${ranks[*]}"
}

mkdir -p "$(dirname "${OUTPUT_CSV}")" "${LOG_ROOT}"

if [[ -z "${SCALE_GPU}" ]]; then
  SCALE_GPU=$(pick_idle_gpu)
  echo "[INFO] Auto-selected SCALE_GPU=${SCALE_GPU}"
fi

echo "[INFO] Output CSV: ${OUTPUT_CSV}"
echo "[INFO] Log root: ${LOG_ROOT}"
echo "[INFO] DRY_RUN=${DRY_RUN}"
echo "[INFO] APPEND_CSV=${APPEND_CSV}"
echo "[INFO] CONFIG_START_INDEX=${CONFIG_START_INDEX}"
echo "[INFO] CONFIG_END_INDEX=${CONFIG_END_INDEX}"

if [[ "${APPEND_CSV}" != "0" && "${APPEND_CSV}" != "1" ]]; then
  echo "[ERROR] APPEND_CSV must be 0 or 1, got ${APPEND_CSV}" >&2
  exit 1
fi

if ! [[ "${CONFIG_START_INDEX}" =~ ^[0-9]+$ && "${CONFIG_END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] CONFIG_START_INDEX and CONFIG_END_INDEX must be non-negative integers." >&2
  exit 1
fi

max_config_index=$(( ${#CONFIGS[@]} - 1 ))
if (( CONFIG_START_INDEX > CONFIG_END_INDEX )); then
  echo "[ERROR] CONFIG_START_INDEX (${CONFIG_START_INDEX}) must be <= CONFIG_END_INDEX (${CONFIG_END_INDEX})." >&2
  exit 1
fi
if (( CONFIG_END_INDEX > max_config_index )); then
  echo "[ERROR] CONFIG_END_INDEX (${CONFIG_END_INDEX}) out of range [0, ${max_config_index}]." >&2
  exit 1
fi

csv_header="world_size,pp_size,tp_size,ep_size,dp_size,measured_ranks_count,single_iter_wallclock_seconds,estimated_5_iters_seconds"
if [[ "${APPEND_CSV}" == "1" ]]; then
  if [[ ! -f "${OUTPUT_CSV}" ]]; then
    echo "[ERROR] APPEND_CSV=1 but OUTPUT_CSV does not exist: ${OUTPUT_CSV}" >&2
    exit 1
  fi
else
  echo "${csv_header}" > "${OUTPUT_CSV}"
fi

for config_idx in "${!CONFIGS[@]}"; do
  if (( config_idx < CONFIG_START_INDEX || config_idx > CONFIG_END_INDEX )); then
    continue
  fi

  read -r world_size pp_size tp_size ep_size dp_size <<< "${CONFIGS[config_idx]}"

  validate_moe_config "${world_size}" "${pp_size}" "${tp_size}" "${ep_size}" "${dp_size}"

  global_batch_size=$((MICRO_BATCH_SIZE * dp_size))

  selected_ranks_raw=$(build_selected_ranks "${pp_size}" "${tp_size}" "${ep_size}" "${dp_size}")
  IFS=' ' read -r -a selected_ranks <<< "${selected_ranks_raw}"
  measured_ranks_count=${#selected_ranks[@]}

  config_name="ws${world_size}_pp${pp_size}_tp${tp_size}_ep${ep_size}_dp${dp_size}"
  config_log_dir="${LOG_ROOT}/${config_name}"
  mkdir -p "${config_log_dir}"

  echo "============================================================"
  echo "[INFO] Config $((config_idx + 1))/${#CONFIGS[@]}: ${config_name}"
  echo "[INFO] Selected ranks (${measured_ranks_count}): ${selected_ranks[*]}"
  echo "[INFO] global_batch_size=${global_batch_size}, micro_batch_size=${MICRO_BATCH_SIZE}"
  echo "============================================================"

  config_start_ns=$(date +%s%N)

  for rank_pos in "${!selected_ranks[@]}"; do
    fake_current_rank_id=${selected_ranks[rank_pos]}
    rank_log_path="${config_log_dir}/rank_${fake_current_rank_id}.log"

    echo "[INFO] Running rank ${fake_current_rank_id} (${rank_pos}/${measured_ranks_count})"

    if [[ "${DRY_RUN}" == "1" ]]; then
      {
        echo "[DRY_RUN] Skip torchrun for rank ${fake_current_rank_id}"
      } | tee "${rank_log_path}"
      continue
    fi

    run_port=$((MASTER_PORT_BASE + config_idx * 1000 + rank_pos))

    CUDA_VISIBLE_DEVICES="${SCALE_GPU}" torchrun \
      --nproc_per_node="${GPUS_PER_NODE}" \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${run_port}" \
      "${PROJECT_ROOT}/pretrain_llama.py" \
      --kv-channels 128 \
      --qk-layernorm \
      --use-mcore-models \
      --transformer-impl "${TRANSFORMER_IMPL}" \
      --mock-data \
      --dataloader-type cyclic \
      --tokenizer-type NullTokenizer \
      --vocab-size "${VOCAB_SIZE}" \
      --make-vocab-size-divisible-by 128 \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1 \
      --expert-model-parallel-size 1 \
      --sequence-parallel \
      --num-layers "${NUM_LAYERS}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
      --num-attention-heads "${NUM_HEADS}" \
      --group-query-attention \
      --num-query-groups "${NUM_QUERY_GROUPS}" \
      --position-embedding-type rope \
      --rotary-percent 1.0 \
      --rotary-base 1000000 \
      --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
      --normalization RMSNorm \
      --norm-epsilon 1e-6 \
      --swiglu \
      --untie-embeddings-and-output-weights \
      --disable-bias-linear \
      --num-experts "${NUM_EXPERTS}" \
      --moe-layer-freq 1 \
      --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
      --moe-router-load-balancing-type aux_loss \
      --moe-router-topk "${MOE_ROUTER_TOPK}" \
      --moe-grouped-gemm \
      --moe-aux-loss-coeff 1e-3 \
      --moe-token-dispatcher-type alltoall \
      --seq-length "${SEQ_LEN}" \
      --micro-batch-size "${MICRO_BATCH_SIZE}" \
      --global-batch-size "${global_batch_size}" \
      --train-iters "${TRAIN_ITERS}" \
      --lr 1.2e-4 \
      --lr-decay-iters 1 \
      --lr-decay-style cosine \
      --min-lr 1.2e-5 \
      --weight-decay 1e-1 \
      --lr-warmup-fraction .01 \
      --clip-grad 1.0 \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \
      --bf16 \
      --do-trace True \
      --trace-start "${TRACE_START}" \
      --trace-subop-sync-mode "${TRACE_SUBOP_SYNC_MODE}" \
      --is-scaling-mode \
      --fake-world-size "${world_size}" \
      --fake-wrank 0 \
      --fake-gpus-per-node "${world_size}" \
      --fake-local-rank 0 \
      --fake-pp "${pp_size}" \
      --fake-dp "${dp_size}" \
      --fake-tp "${tp_size}" \
      --fake-exp "${ep_size}" \
      --fake-num-experts "${NUM_EXPERTS}" \
      --fake-current-rank-id "${fake_current_rank_id}" \
      --scaling-min-warmup-iters "${SCALING_MIN_WARMUP_ITERS}" \
      --scaling-profile-iters "${SCALING_PROFILE_ITERS}" \
      --distributed-backend nccl \
      --seed 42 2>&1 | tee "${rank_log_path}"

    rank_exit=${PIPESTATUS[0]}
    if [[ "${rank_exit}" -ne 0 ]]; then
      echo "[ERROR] Rank ${fake_current_rank_id} failed with exit code ${rank_exit}. Log: ${rank_log_path}" >&2
      exit "${rank_exit}"
    fi
  done

  config_end_ns=$(date +%s%N)
  single_iter_wallclock_seconds=$(awk "BEGIN {printf \"%.6f\", (${config_end_ns}-${config_start_ns})/1000000000}")
  estimated_5_iters_seconds=$(awk "BEGIN {printf \"%.6f\", ${single_iter_wallclock_seconds}*5}")

  csv_row="${world_size},${pp_size},${tp_size},${ep_size},${dp_size},${measured_ranks_count},${single_iter_wallclock_seconds},${estimated_5_iters_seconds}"
  echo "${csv_row}" >> "${OUTPUT_CSV}"

  echo "[INFO] Config ${config_name} done: single_iter_wallclock_seconds=${single_iter_wallclock_seconds}, estimated_5_iters_seconds=${estimated_5_iters_seconds}"
done

echo "[INFO] Completed all configurations. CSV saved to ${OUTPUT_CSV}"
