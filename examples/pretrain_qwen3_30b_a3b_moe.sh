#!/bin/bash

# cmd：
# MODE=distributed \
# TRAIN_ITERS=10 \
# TRACE_START=8 \
# SEQ_LEN=2048 \
# MICRO_BATCH_SIZE=8 \
# TRACE_SUBOP_SYNC_MODE=event \
# bash examples/pretrain_qwen3_30b_a3b_moe.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

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

MODE=${MODE:-distributed} # distributed | scaling
MODEL_PROFILE=${MODEL_PROFILE:-smoke} # smoke | full
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
TRACE_START=${TRACE_START:-1}
TRAIN_ITERS=${TRAIN_ITERS:-10}
TRACE_SUBOP_SYNC_MODE=${TRACE_SUBOP_SYNC_MODE:-event}
TRACE_KERNEL_GROUND_TRUTH=${TRACE_KERNEL_GROUND_TRUTH:-0}
TRACE_KERNEL_GROUND_TRUTH_PREFIX=${TRACE_KERNEL_GROUND_TRUTH_PREFIX:-cmd_trace}
DO_TRACE=${DO_TRACE:-True}
LR=${LR:-1.2e-4}
MIN_LR=${MIN_LR:-1.2e-5}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6100}

TP=${TP:-1}
PP=${PP:-4}
EP=${EP:-2}
CP=${CP:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-256}

if [[ "${MODEL_PROFILE}" == "full" ]]; then
  NUM_LAYERS=48
  HIDDEN_SIZE=2048
  NUM_HEADS=32
  NUM_QUERY_GROUPS=4
  FFN_HIDDEN_SIZE=6144
  NUM_EXPERTS=128
  MOE_FFN_HIDDEN_SIZE=768
  MOE_ROUTER_TOPK=8
  MAX_POSITION_EMBEDDINGS=40960
  VOCAB_SIZE=151936
else
  # Smoke profile for fast local tracing verification.
  NUM_LAYERS=12
  HIDDEN_SIZE=1024
  NUM_HEADS=16
  NUM_QUERY_GROUPS=4
  FFN_HIDDEN_SIZE=3072
  NUM_EXPERTS=32
  MOE_FFN_HIDDEN_SIZE=512
  MOE_ROUTER_TOPK=4
  MAX_POSITION_EMBEDDINGS=8192
  VOCAB_SIZE=49152
fi

FAKE_WORLD_SIZE=${FAKE_WORLD_SIZE:-8}
FAKE_PP=${FAKE_PP:-${PP}}
FAKE_TP=${FAKE_TP:-${TP}}
FAKE_EXP=${FAKE_EXP:-${EP}}
FAKE_DP=${FAKE_DP:-$((FAKE_WORLD_SIZE / FAKE_PP / FAKE_TP))}

if (( FAKE_DP * FAKE_PP * FAKE_TP != FAKE_WORLD_SIZE )); then
  echo "[ERROR] Invalid fake parallel setup: fake_dp * fake_pp * fake_tp != fake_world_size"
  exit 1
fi

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * (GPUS_PER_NODE / TP / PP)))}

TRACE_ARGS=(
  --do-trace "${DO_TRACE}"
  --trace-start "${TRACE_START}"
  --trace-subop-sync-mode "${TRACE_SUBOP_SYNC_MODE}"
)
if [[ "${TRACE_KERNEL_GROUND_TRUTH}" == "1" ]]; then
  TRACE_ARGS+=(--trace-kernel-ground-truth)
  TRACE_ARGS+=(--trace-kernel-ground-truth-prefix "${TRACE_KERNEL_GROUND_TRUTH_PREFIX}")
fi

COMMON_ARGS=(
  --use-mcore-models
  --transformer-impl "${TRANSFORMER_IMPL}"
  --mock-data
  --dataloader-type cyclic
  --tokenizer-type NullTokenizer
  --vocab-size "${VOCAB_SIZE}"
  --make-vocab-size-divisible-by 128
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --context-parallel-size "${CP}"
  --expert-model-parallel-size "${EP}"
  --sequence-parallel
  --num-layers "${NUM_LAYERS}"
  --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
  --num-attention-heads "${NUM_HEADS}"
  --group-query-attention
  --num-query-groups "${NUM_QUERY_GROUPS}"
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --swiglu
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --num-experts "${NUM_EXPERTS}"
  --moe-layer-freq 1
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk "${MOE_ROUTER_TOPK}"
  --moe-grouped-gemm
  --moe-aux-loss-coeff 1e-3
  --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}"
  --seq-length "${SEQ_LEN}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-decay-style cosine
  --lr-decay-iters "${TRAIN_ITERS}"
  --lr-warmup-iters 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --weight-decay 0.1
  --clip-grad 1.0
  --log-interval 1
  --eval-interval 10000
  --bf16
)

if [[ "${MODE}" == "distributed" ]]; then
  torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/pretrain_llama.py" \
    "${COMMON_ARGS[@]}" \
    "${TRACE_ARGS[@]}"
elif [[ "${MODE}" == "scaling" ]]; then
  SCALE_GPU=${SCALE_GPU:-}
  if [[ -z "${SCALE_GPU}" ]]; then
    SCALE_GPU=$(pick_idle_gpu)
    echo "[Scaling Mode] auto-selected SCALE_GPU=${SCALE_GPU}"
  fi
  FAKE_RANK_ORDER=${FAKE_RANK_ORDER:-}
  SCALING_OVERRIDE_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --global-batch-size "$((MICRO_BATCH_SIZE * FAKE_DP))"
  )
  RANK_IDS=()
  if [[ -n "${FAKE_RANK_ORDER}" ]]; then
    IFS=',' read -r -a RANK_IDS <<< "${FAKE_RANK_ORDER}"
  else
    for ((rank_id=0; rank_id<FAKE_WORLD_SIZE; rank_id++)); do
      RANK_IDS+=("${rank_id}")
    done
  fi

  for FAKE_CURRENT_RANK_ID in "${RANK_IDS[@]}"; do
    if (( FAKE_CURRENT_RANK_ID < 0 || FAKE_CURRENT_RANK_ID >= FAKE_WORLD_SIZE )); then
      echo "[ERROR] Invalid rank ${FAKE_CURRENT_RANK_ID} in FAKE_RANK_ORDER for fake_world_size=${FAKE_WORLD_SIZE}"
      exit 1
    fi
    echo "[Scaling Mode] fake_current_rank_id=${FAKE_CURRENT_RANK_ID}/${FAKE_WORLD_SIZE}"
    CUDA_VISIBLE_DEVICES="${SCALE_GPU}" torchrun \
      --nproc_per_node=1 \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr="${MASTER_ADDR}" \
      --master_port="$((MASTER_PORT + FAKE_CURRENT_RANK_ID))" \
      "${PROJECT_ROOT}/pretrain_llama.py" \
      "${COMMON_ARGS[@]}" \
      "${TRACE_ARGS[@]}" \
      "${SCALING_OVERRIDE_ARGS[@]}" \
      --is-scaling-mode \
      --fake-world-size "${FAKE_WORLD_SIZE}" \
      --fake-wrank 0 \
      --fake-gpus-per-node "${FAKE_WORLD_SIZE}" \
      --fake-local-rank 0 \
      --fake-pp "${FAKE_PP}" \
      --fake-dp "${FAKE_DP}" \
      --fake-tp "${FAKE_TP}" \
      --fake-exp "${FAKE_EXP}" \
      --fake-num-experts "${NUM_EXPERTS}" \
      --fake-current-rank-id "${FAKE_CURRENT_RANK_ID}"
  done
else
  echo "[ERROR] Unsupported MODE=${MODE}. Use distributed or scaling."
  exit 1
fi
