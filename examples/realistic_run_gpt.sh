#! /bin/bash
# ./realistic_run_gpt.sh 1 0 6 6000 localhost 3 2 "tiny"
# NNODES, NODE_RANK, GPUS_PER_NODE, MASTER_PORT, MASTER_ADDR, PP, TP, MODEL_SIZE

set -o pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG="${NCCL_DEBUG:-TRACE}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens81f0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"



NNODES=$1
NODE_RANK=$2
GPUS_PER_NODE=$3
MASTER_PORT=$4
MASTER_ADDR=$5
PP=$6
TP=$7
MODEL_SIZE=$8


MAX_TOPOLOGY_VALUE=2147483647

require_positive_integer() {
    local name=$1
    local value=$2

    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer, got ${value}" >&2
        return 1
    fi

    if (( ${#value} > ${#MAX_TOPOLOGY_VALUE} )) || (( 10#${value} > MAX_TOPOLOGY_VALUE )); then
        echo "ERROR: ${name} must be between 1 and ${MAX_TOPOLOGY_VALUE}, got ${value}" >&2
        return 1
    fi
}

require_positive_integer "NNODES" "${NNODES}" || exit 1
require_positive_integer "GPUS_PER_NODE" "${GPUS_PER_NODE}" || exit 1
require_positive_integer "PP" "${PP}" || exit 1
require_positive_integer "TP" "${TP}" || exit 1

# Calculate some derived values
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
# NODE_RANK=0

# Derived variables
PARALLEL_SIZE=$((TP * PP))
if (( GPU_NUM % PARALLEL_SIZE != 0 )); then
    echo "ERROR: Invalid topology: GPU_NUM=${GPU_NUM} must be divisible by PP * TP = ${PARALLEL_SIZE}" >&2
    exit 1
fi
DP=$((${GPU_NUM}/${TP}/${PP}))

NUM_MICBATCH=1
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * DP))

# Model size settings based on input
if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=80; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == "tiny_h64" ]]; then HIDDEN_SIZE=256; NUM_HEAD=4; NUM_LAYERS=4;
else echo "Invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

# Trace and batch settings
DO_TRACE=True
TRAIN_ITERS=10
TRACE_ITER_NUM=1
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1))
NSIGHT_START=$(($TRAIN_ITERS))

MAX_SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2048

# Check if trace_iter_num is within valid range
if [ $TRACE_ITER_NUM -gt $((TRAIN_ITERS - 1)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi

TRACE_ARGS=" \
    --do-trace $DO_TRACE \
    --trace-start $TRACE_START \
    --nsight-start $NSIGHT_START \
    "

FAKE_WORLD_SIZE=8
FAKE_WRANK=0
FAKE_GPUS_PER_NODE=8
FAKE_LOCAL_RANK=0
FAKE_PP=8
FAKE_TP=1
FAKE_DP=$((FAKE_WORLD_SIZE / FAKE_PP / FAKE_TP))

if [ "$((FAKE_DP * FAKE_PP * FAKE_TP))" -ne "$FAKE_WORLD_SIZE" ]; then
    echo "Error: FAKE_DP must be an integer."
    exit 1
fi

    # --is-scaling-mode \
SIM_ARGS=" \
    --fake-world-size $FAKE_WORLD_SIZE \
    --fake-wrank $FAKE_WRANK \
    --fake-gpus-per-node $FAKE_GPUS_PER_NODE \
    --fake-local-rank $FAKE_LOCAL_RANK \
    --fake-pp $FAKE_PP \
    --fake-dp $FAKE_DP \
    --fake-tp $FAKE_TP \
    "

# Adjust settings if using scaling mode
if echo "$SIM_ARGS" | grep -q -- "--is-scaling-mode"; then
    GPUS_PER_NODE=1
    GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
    WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
    TP=1
    PP=1
    DP=$((${GPU_NUM}/${TP}/${PP}))
fi


GPT_ARGS=" \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEAD \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.00015 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
"

# Data and output settings
BASE_PATH="${BASE_PATH:-/research/d1/gds/ytyang/yichengfeng/Megatron-LM}"
if [[ "${BASE_PATH}" =~ [[:space:]] ]]; then
    echo "ERROR: BASE_PATH must not contain whitespace: ${BASE_PATH}" >&2
    exit 1
fi
VOCAB_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-vocab.json
MERGE_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-merges.txt
DATA_PATH=${BASE_PATH}/data/output_prefix_gpt2/my-gpt2_text_document

MOCK_DATA="${MOCK_DATA:-0}"
if [[ "${MOCK_DATA}" == "1" ]]; then
    DATA_ARGS=(
        --mock-data
        --main-tokenizer-type NullTokenizer
        --tokenizer-type NullTokenizer
        --vocab-size 51200
    )
elif [[ "${MOCK_DATA}" == "0" ]]; then
    DATA_ARGS=(
        --data-path "${DATA_PATH}"
        --vocab-file "${VOCAB_FILE}"
        --merge-file "${MERGE_FILE}"
        --split 949,50,1
        --main-tokenizer-type GPT2BPETokenizer
        --vocab-size 3200
    )
else
    echo "ERROR: MOCK_DATA must be 0 or 1, got ${MOCK_DATA}" >&2
    exit 1
fi

OUTPUT_ARGS=" \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1 \
"

LOG_DIR="${LOG_DIR:-run_log}"
if [ ! -d "$LOG_DIR" ]; then
  if ! mkdir -p -- "$LOG_DIR"; then
    echo "ERROR: Failed to create log directory: ${LOG_DIR}" >&2
    exit 1
  fi
fi

# Log file based on parameters
LOG_FILE="${LOG_DIR}/realistic_log_${NNODES}nodes_${GPUS_PER_NODE}gpus_${PP}pp_${TP}tp_${MODEL_SIZE}model.log"

# Run the training process and log output to file
torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} "${BASE_PATH}/pretrain_llama.py" \
    $GPT_ARGS \
    "${DATA_ARGS[@]}" \
    $OUTPUT_ARGS \
    $SIM_ARGS \
    $TRACE_ARGS \
    --distributed-backend nccl \
    --use-mcore-models \
    2>&1 | tee -- "${LOG_FILE}"
