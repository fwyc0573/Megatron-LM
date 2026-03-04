#!/bin/bash

set -euo pipefail

# Runtime environment defaults.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# Keep fake node topology aligned with 8-GPU-per-node H800 clusters.
FAKE_GPUS_PER_NODE=${FAKE_GPUS_PER_NODE:-8}
SCALE_GPU=${SCALE_GPU:-${CUDA_VISIBLE_DEVICES:-0}}

NNODES=1
GPUS_PER_NODE=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-6300}

# Target distributed topology from the real run.
FAKE_WORLD_SIZE=256
FAKE_PP=16
FAKE_TP=8
FAKE_DP=2
FAKE_EXP=1

if (( FAKE_WORLD_SIZE != FAKE_PP * FAKE_TP * FAKE_DP )); then
  echo "[ERROR] Invalid topology: require world_size == pp * tp * dp." >&2
  exit 1
fi

# Model/runtime arguments copied from the provided real command.
NUM_LAYERS=96
HIDDEN_SIZE=12288
FFN_HIDDEN_SIZE=49152
NUM_HEADS=96
SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2048
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
TRAIN_ITERS=10
TRANSFORMER_IMPL=transformer_engine

TRACE_ITER_NUM=1
TRACE_START=$((TRAIN_ITERS - TRACE_ITER_NUM + 1))
TRACE_SUBOP_SYNC_MODE=${TRACE_SUBOP_SYNC_MODE:-global}

if (( TRACE_ITER_NUM > TRAIN_ITERS - 1 )); then
  echo "[ERROR] TRACE_ITER_NUM must satisfy TRACE_ITER_NUM <= TRAIN_ITERS - 1." >&2
  exit 1
fi

LOG_ROOT=${LOG_ROOT:-"${PROJECT_ROOT}/logs/scaling_profile_dense_ws256_pp16_tp8_dp2"}
mkdir -p "${LOG_ROOT}"

build_selected_ranks() {
  local pp_size=$1
  local tp_size=$2
  local dp_size=$3
  local ranks=()

  # Dense-model optimization:
  # only profile ranks with tp_rank=0 and dp_rank=0 for every PP stage.
  for ((pp_stage = 0; pp_stage < pp_size; pp_stage++)); do
    ranks+=("$((pp_stage * tp_size * dp_size))")
  done

  echo "${ranks[*]}"
}

selected_ranks_raw=$(build_selected_ranks "${FAKE_PP}" "${FAKE_TP}" "${FAKE_DP}")
IFS=' ' read -r -a selected_ranks <<< "${selected_ranks_raw}"

echo "============================================================"
echo "[INFO] Dense scaling profiling starts"
echo "[INFO] Fake topology: WS=${FAKE_WORLD_SIZE}, PP=${FAKE_PP}, TP=${FAKE_TP}, DP=${FAKE_DP}"
echo "[INFO] Selected ranks (tp=0, dp=0 across all PP stages): ${selected_ranks[*]}"
echo "[INFO] Selected rank count: ${#selected_ranks[@]} / ${FAKE_WORLD_SIZE}"
echo "[INFO] Log root: ${LOG_ROOT}"
echo "============================================================"

for rank_idx in "${!selected_ranks[@]}"; do
  fake_current_rank_id=${selected_ranks[rank_idx]}
  pp_stage=$((fake_current_rank_id / (FAKE_TP * FAKE_DP)))
  run_port=$((MASTER_PORT_BASE + rank_idx))
  rank_log_path="${LOG_ROOT}/rank_${fake_current_rank_id}.log"

  echo "[INFO] Profiling fake rank ${fake_current_rank_id} (PP stage ${pp_stage})"

  CUDA_VISIBLE_DEVICES="${SCALE_GPU}" torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${run_port}" \
    "${PROJECT_ROOT}/pretrain_llama.py" \
    --use-mcore-models \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 51200 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --num-layers "${NUM_LAYERS}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_HEADS}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --transformer-impl "${TRANSFORMER_IMPL}" \
    --train-iters "${TRAIN_ITERS}" \
    --lr 0.00015 \
    --lr-decay-iters "${TRAIN_ITERS}" \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --do-trace True \
    --trace-start "${TRACE_START}" \
    --trace-subop-sync-mode "${TRACE_SUBOP_SYNC_MODE}" \
    --is-scaling-mode \
    --fake-world-size "${FAKE_WORLD_SIZE}" \
    --fake-wrank 0 \
    --fake-gpus-per-node "${FAKE_GPUS_PER_NODE}" \
    --fake-local-rank 0 \
    --fake-pp "${FAKE_PP}" \
    --fake-dp "${FAKE_DP}" \
    --fake-tp "${FAKE_TP}" \
    --fake-exp "${FAKE_EXP}" \
    --fake-current-rank-id "${fake_current_rank_id}" \
    --distributed-backend nccl \
    --seed 42 2>&1 | tee "${rank_log_path}"

  run_status=${PIPESTATUS[0]}
  if [[ "${run_status}" -ne 0 ]]; then
    echo "[ERROR] Fake rank ${fake_current_rank_id} failed with exit code ${run_status}. Log: ${rank_log_path}" >&2
    exit "${run_status}"
  fi
done

echo "[INFO] Scaling profiling completed. Logs: ${LOG_ROOT}"
