#!/bin/bash

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
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-local}
TRACE_START=${TRACE_START:-1}
TRAIN_ITERS=${TRAIN_ITERS:-3}
TRACE_SUBOP_SYNC_MODE=${TRACE_SUBOP_SYNC_MODE:-global}
TRACE_CMD_SYNC_MODE=${TRACE_CMD_SYNC_MODE:-global}
TRACE_OPTIMIZER_MICROPHASES=${TRACE_OPTIMIZER_MICROPHASES:-0}
DO_TRACE=${DO_TRACE:-True}
SCALING_MIN_WARMUP_ITERS=${SCALING_MIN_WARMUP_ITERS:-0}
SCALING_PROFILE_ITERS=${SCALING_PROFILE_ITERS:-3}
SCALING_REPLAY_CACHE_TAG=${SCALING_REPLAY_CACHE_TAG:-}
if [[ -z "${SCALING_REPLAY_CACHE_TAG}" ]]; then
  SCALING_REPLAY_CACHE_TAG=$(date +%Y%m%d%H%M%S)
fi
SCALING_FAKE_RANK_ORDER=${SCALING_FAKE_RANK_ORDER:-}

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6300}

TP=${TP:-1}
PP=${PP:-2}
EP=${EP:-2}
CP=${CP:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-256}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.006}

if [[ "${MODEL_PROFILE}" == "full" ]]; then
  NUM_LAYERS=61
  HIDDEN_SIZE=7168
  NUM_HEADS=128
  FFN_HIDDEN_SIZE=18432
  NUM_EXPERTS=256
  MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-2048}
  MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-2048}
  MOE_ROUTER_TOPK=8
  MOE_ROUTER_NUM_GROUPS=8
  MOE_ROUTER_GROUP_TOPK=4
  MOE_ROUTER_TOPK_SCALING_FACTOR=${MOE_ROUTER_TOPK_SCALING_FACTOR:-2.5}
  MAX_POSITION_EMBEDDINGS=16384
  ORIGINAL_MAX_POSITION_EMBEDDINGS=4096
  VOCAB_SIZE=129280
  Q_LORA_RANK=1536
  KV_LORA_RANK=512
  QK_HEAD_DIM=128
  QK_POS_EMB_HEAD_DIM=64
  V_HEAD_DIM=128
  ROTARY_SCALING_FACTOR=40
  MOE_LAYER_FREQ='([0]*3+[1]*58)'
else
  NUM_LAYERS=8
  HIDDEN_SIZE=1024
  NUM_HEADS=8
  FFN_HIDDEN_SIZE=4096
  NUM_EXPERTS=16
  MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-512}
  MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-512}
  MOE_ROUTER_TOPK=2
  MOE_ROUTER_NUM_GROUPS=4
  MOE_ROUTER_GROUP_TOPK=2
  MOE_ROUTER_TOPK_SCALING_FACTOR=${MOE_ROUTER_TOPK_SCALING_FACTOR:-1.0}
  MAX_POSITION_EMBEDDINGS=8192
  ORIGINAL_MAX_POSITION_EMBEDDINGS=4096
  VOCAB_SIZE=32768
  Q_LORA_RANK=256
  KV_LORA_RANK=128
  QK_HEAD_DIM=64
  QK_POS_EMB_HEAD_DIM=32
  V_HEAD_DIM=64
  ROTARY_SCALING_FACTOR=4
  MOE_LAYER_FREQ='([0]*2+[1]*6)'
fi

MOE_SHARED_EXPERT_GATE=${MOE_SHARED_EXPERT_GATE:-1}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-1}
USE_BF16=${USE_BF16:-1}
MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-1}
if (( TRAIN_ITERS <= 0 )); then
  echo "[ERROR] TRAIN_ITERS must be > 0, got ${TRAIN_ITERS}."
  exit 1
fi
if (( LR_WARMUP_ITERS >= TRAIN_ITERS )); then
  LR_WARMUP_ITERS=$((TRAIN_ITERS - 1))
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
  --trace-cmd-sync-mode "${TRACE_CMD_SYNC_MODE}"
)
if [[ "${TRACE_OPTIMIZER_MICROPHASES}" != "0" && "${TRACE_OPTIMIZER_MICROPHASES}" != "1" ]]; then
  echo "[ERROR] TRACE_OPTIMIZER_MICROPHASES must be 0 or 1, got ${TRACE_OPTIMIZER_MICROPHASES}."
  exit 1
fi
if [[ "${TRACE_OPTIMIZER_MICROPHASES}" == "1" ]]; then
  TRACE_ARGS+=(--trace-optimizer-microphases)
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
  --init-method-std "${INIT_METHOD_STD}"
  --position-embedding-type rope
  --rope-type yarn
  --rotary-percent 1.0
  --rotary-base 10000
  --rotary-scaling-factor "${ROTARY_SCALING_FACTOR}"
  --original-max-position-embeddings "${ORIGINAL_MAX_POSITION_EMBEDDINGS}"
  --beta-fast 32.0
  --beta-slow 1.0
  --mscale 1.0
  --mscale-all-dim 1.0
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --qk-layernorm
  --swiglu
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --multi-latent-attention
  --q-lora-rank "${Q_LORA_RANK}"
  --kv-lora-rank "${KV_LORA_RANK}"
  --qk-head-dim "${QK_HEAD_DIM}"
  --qk-pos-emb-head-dim "${QK_POS_EMB_HEAD_DIM}"
  --v-head-dim "${V_HEAD_DIM}"
  --num-experts "${NUM_EXPERTS}"
  --moe-layer-freq "${MOE_LAYER_FREQ}"
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
  --moe-shared-expert-intermediate-size "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE}"
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk "${MOE_ROUTER_TOPK}"
  --moe-router-num-groups "${MOE_ROUTER_NUM_GROUPS}"
  --moe-router-group-topk "${MOE_ROUTER_GROUP_TOPK}"
  --moe-router-score-function sigmoid
  --moe-router-topk-scaling-factor "${MOE_ROUTER_TOPK_SCALING_FACTOR}"
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 1e-3
  --moe-router-dtype fp32
  --moe-shared-expert-gate
  --moe-grouped-gemm
  --moe-aux-loss-coeff 1e-4
  --moe-token-dispatcher-type alltoall
  --seq-length "${SEQ_LEN}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --lr 3.9e-6
  --min-lr 3.9e-7
  --lr-decay-style cosine
  --lr-decay-iters "${TRAIN_ITERS}"
  --lr-warmup-iters "${LR_WARMUP_ITERS}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --weight-decay 0.1
  --clip-grad 1.0
  --log-interval 1
  --eval-interval 10000
  --bf16
)

if (( USE_BF16 == 0 )); then
  FILTERED_COMMON_ARGS=()
  for arg in "${COMMON_ARGS[@]}"; do
    if [[ "${arg}" != "--bf16" ]]; then
      FILTERED_COMMON_ARGS+=("${arg}")
    fi
  done
  COMMON_ARGS=("${FILTERED_COMMON_ARGS[@]}")
fi

if (( MOE_GROUPED_GEMM == 0 )); then
  FILTERED_COMMON_ARGS=()
  for arg in "${COMMON_ARGS[@]}"; do
    if [[ "${arg}" != "--moe-grouped-gemm" ]]; then
      FILTERED_COMMON_ARGS+=("${arg}")
    fi
  done
  COMMON_ARGS=("${FILTERED_COMMON_ARGS[@]}")
fi

if (( MOE_SHARED_EXPERT_GATE == 0 )); then
  FILTERED_COMMON_ARGS=()
  for arg in "${COMMON_ARGS[@]}"; do
    if [[ "${arg}" != "--moe-shared-expert-gate" ]]; then
      FILTERED_COMMON_ARGS+=("${arg}")
    fi
  done
  COMMON_ARGS=("${FILTERED_COMMON_ARGS[@]}")
fi

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

  if [[ -n "${SCALING_FAKE_RANK_ORDER}" ]]; then
    IFS=',' read -r -a FAKE_RANK_IDS <<< "${SCALING_FAKE_RANK_ORDER}"
  else
    FAKE_RANK_IDS=()
    for ((fake_rank=0; fake_rank<FAKE_WORLD_SIZE; fake_rank++)); do
      FAKE_RANK_IDS+=("${fake_rank}")
    done
  fi

  if (( ${#FAKE_RANK_IDS[@]} == 0 )); then
    echo "[ERROR] SCALING_FAKE_RANK_ORDER resolved to empty rank list."
    exit 1
  fi

  SCALING_OVERRIDE_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --global-batch-size "$((MICRO_BATCH_SIZE * FAKE_DP))"
  )

  for FAKE_CURRENT_RANK_ID in "${FAKE_RANK_IDS[@]}"; do
    if ! [[ "${FAKE_CURRENT_RANK_ID}" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] Invalid fake rank id '${FAKE_CURRENT_RANK_ID}' in SCALING_FAKE_RANK_ORDER."
      exit 1
    fi
    if (( FAKE_CURRENT_RANK_ID < 0 || FAKE_CURRENT_RANK_ID >= FAKE_WORLD_SIZE )); then
      echo "[ERROR] fake rank id ${FAKE_CURRENT_RANK_ID} out of range [0, $((FAKE_WORLD_SIZE - 1))]."
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
      --fake-current-rank-id "${FAKE_CURRENT_RANK_ID}" \
      --scaling-min-warmup-iters "${SCALING_MIN_WARMUP_ITERS}" \
      --scaling-profile-iters "${SCALING_PROFILE_ITERS}" \
      --scaling-replay-cache-tag "${SCALING_REPLAY_CACHE_TAG}"
  done
else
  echo "[ERROR] Unsupported MODE=${MODE}. Use distributed or scaling."
  exit 1
fi
