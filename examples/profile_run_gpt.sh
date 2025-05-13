#! /bin/bash
# FAKE_WORLD_SIZE, FAKE_PP, FAKE_TP, and MODEL_SIZE

# ./profile_run_gpt.sh 96 16 6 175
# ./profile_run_gpt.sh 96 12 8 175


# ./profile_run_gpt.sh 96 6 8 70
# ./profile_run_gpt.sh 96 12 4 70


# ./profile_run_gpt.sh 64 4 8 70
# ./profile_run_gpt.sh 64 8 8 70




# Read the parameters from command line arguments
export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=ens81f0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

FAKE_WORLD_SIZE=$1
FAKE_PP=$2
FAKE_TP=$3
MODEL_SIZE=$4
FAKE_GPUS_PER_NODE=8



# Calculate DP based on FAKE_WORLD_SIZE, FAKE_PP, FAKE_TP
FAKE_DP=$((FAKE_WORLD_SIZE / FAKE_PP / FAKE_TP))
if [ "$((FAKE_DP * FAKE_PP * FAKE_TP))" -ne "$FAKE_WORLD_SIZE" ]; then
    echo "Error: FAKE_DP must be an integer."
    exit 1
fi

# Model size settings based on input
if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
else echo "Invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

# TRACE settings
DO_TRACE=True
TRAIN_ITERS=10
TRACE_ITER_NUM=1
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1))
NSIGHT_START=$(($TRAIN_ITERS))

MAX_SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2048


NUM_MICBATCH=1
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * FAKE_DP))


TRACE_ARGS=" \
    --do-trace $DO_TRACE \
    --trace-start $TRACE_START \
    --nsight-start $NSIGHT_START \
    "

# SIMULATION arguments based on input parameters
SIM_ARGS=" \
    --is-scaling-mode \
    --fake-world-size $FAKE_WORLD_SIZE \
    --fake-wrank 0 \
    --fake-gpus-per-node $FAKE_GPUS_PER_NODE \
    --fake-local-rank 0 \
    --fake-pp $FAKE_PP \
    --fake-dp $FAKE_DP \
    --fake-tp $FAKE_TP \
    "

# 当采用is-scaling-mode时,采用单个rank进行PROFILE
if echo "$SIM_ARGS" | grep -q -- "--is-scaling-mode"; then
    GPUS_PER_NODE=1
    NNODES=1
    GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
    WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
    TP=1
    PP=1
    DP=$((${GPU_NUM}/${TP}/${PP}))
fi



GPT_ARGS=" \
    --main-tokenizer-type GPT2BPETokenizer \
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
    # --fp16 \



BASE_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM
VOCAB_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-vocab.json
MERGE_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-merges.txt
DATA_PATH=${BASE_PATH}/data/output_prefix_gpt2/my-gpt2_text_document


DATA_ARGS=" \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --vocab-size 3200 \
"

OUTPUT_ARGS=" \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1 \
"

# Create run_log directory if it doesn't exist
LOG_DIR="run_log"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Log file based on parameters and store in run_log folder
LOG_FILE="${LOG_DIR}/sim_log_${FAKE_WORLD_SIZE}world_${FAKE_GPUS_PER_NODE}gpus_${FAKE_PP}pp_${FAKE_TP}tp_${MODEL_SIZE}model.log"


# Run the training process and log output to file
torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank 0 --master-addr "localhost" --master-port 6000 ${BASE_PATH}/pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $SIM_ARGS \
    $TRACE_ARGS \
    --distributed-backend nccl \
    --use-mcore-models \
    2>&1 | tee ${LOG_FILE}
