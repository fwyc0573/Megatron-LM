#! /bin/bash

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=DEBUG # WARN INFO
export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6000
MASTER_ADDR="localhost" #"localhost"

# Parallelism variables
PP=2
TP=2
DP=$((${GPU_NUM}/${TP}/${PP}))


NUM_MICBATCH=2
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SZIE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * DP))


# Network size variables
MODEL_SIZE=13 # "tiny"

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_QUERY_GROUP=8; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

# TRACE控制参数
TRAIN_ITERS=3
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters)
DO_TRACE=True

# 检查trace_iter_num是否在合理的范围内
if [ $TRACE_ITER_NUM -gt $((TRAIN_ITERS - 1)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=2048 # 4096 2048
MAX_POSITION_EMBEDDINGS=2048 # 4096 2048

# Paths
BASE_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM
SRC_PATH=${BASE_PATH}/pretrain_llama.py

LOG_NAME=llama2_${MODEL_SIZE}_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

DATA_PATH=${BASE_PATH}/data/output_prefix_llama/output_prefix_llama_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

# SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
# LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
# mkdir -p ${BASE_PATH}/log/${LOG_NAME}

TOKENIZER_PATH=${BASE_PATH}/megatron/training/tokenizer


# ------------------------------额外补充的arguments--------------------------
FAKE_WORLD_SIZE=32
FAKE_WRANK=0
FAKE_GPUS_PER_NODE=8
FAKE_LOCAL_RANK=0
# IS_SCALING_MODE=False
FAKE_PP=4
FAKE_TP=4
FAKE_DP=$((FAKE_WORLD_SIZE / FAKE_PP / FAKE_TP))
if [ "$((FAKE_DP * FAKE_PP * FAKE_TP))" -ne "$FAKE_WORLD_SIZE" ]; then
    echo "Error: FAKE_DP must be an integer."
    exit 1
fi

#        --is-scaling-mode \
SIM_ARGS=" \
       --is-scaling-mode \
       --fake-world-size $FAKE_WORLD_SIZE \
       --fake-wrank $FAKE_WRANK \
       --fake-gpus-per-node $FAKE_GPUS_PER_NODE \
       --fake-local-rank $FAKE_LOCAL_RANK \
       --fake-pp $FAKE_PP \
       --fake-dp $FAKE_DP \
       --fake-tp $FAKE_TP \
       "

TRACE_ARGS=" \
       --do-trace $DO_TRACE \
       --trace-start $TRACE_START \
       --nsight-start $NSIGHT_START \
       "

# 当采用is-scaling-mode时,采用单个rank进行PROFILE
if echo "$SIM_ARGS" | grep -q -- "--is-scaling-mode"; then
    GPUS_PER_NODE=1
    GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
    WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
    TP=1
    PP=1
    DP=$((${GPU_NUM}/${TP}/${PP}))
fi

# ------------------------------额外补充的arguments--------------------------


# Set training command
# torchrun \
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node-rank ${NODE_RANK} \
       --master-addr ${MASTER_ADDR} \
       --master-port ${MASTER_PORT} \
       "
# LAUNCHER=" \
#        deepspeed \
#        --hostfile ${BASE_PATH}/hostfile \
#        --master_addr 192.168.50.186 \
#        --master_port 25555 \
#        "

       # --use-distributed-optimizer \
       # --sequence-parallel \
# OVERLAP_REDUCE="--overlap-grad-reduce"
# BUCKET_SIZE="--ddp-bucket-size 10000"
# DIST_OPTIM="--use-distributed-optimizer"

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       ${DIST_OPTIM} \
       ${BUCKET_SIZE} \
       ${OVERLAP_REDUCE} \
       "
       # --use-flash-attn \
       # --untie-embeddings-and-output-weights \
NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --make-vocab-size-divisible-by 1 \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \

       "

       # --log-timers-to-tensorboard \
       # --log-validation-ppl-to-tensorboard \
       # --log-memory-to-tensorboard \
LOGGING_ARGS=" \
       "

       # --adam-beta1 0.9 \
       # --adam-beta2 0.95 \
       # --adam-eps 1e-8 \
       # --attention-dropout ${DROP_OUT} \
       # --hidden-dropout ${DROP_OUT} \
REGULATIZATION_ARGS=" \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       "

       # --disable-bias-linear \
       # --no-bias-gelu-fusion \
       # --recompute-activations \
       # --recompute-granularity selective \
MCORE_MODELS="--use-mcore-models"

# num_microbatches = global_batch_size / micro_batch_size / dp
TRAINING_ARGS=" \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SZIE \
       --train-iters $TRAIN_ITERS \
       --log-interval 100 \
       --optimizer adam \
       --exit-interval 100 \
       ${MCORE_MODELS} \
       "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "


LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${MIN_LR} \
       "

       # --finetune \
       # --no-load-optim \
       # --no-load-rng \
       # --save ${SAVE_PATH} \
CHECKPOINTING_ARGS=" \
       --save-interval 1000 \
       "

       # --fp16 \
       # --no-query-key-layer-scaling \
MIXED_PRECISION_ARGS=" \
       --fp16 \
       --attention-softmax-in-fp32 \
       "

VALIDATION_ARGS=" \
       --eval-interval 1000 \
       --eval-iters 0 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --data-cache-path ${DATA_CACHE_PATH} \
       --vocab-size 3200 \
       "

NYSY=''
# NYSY="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o ./trace_3d/_${MCORE_MODELS:+_mcore}_pptest_llama_GPUs${GPUS_PER_NODE}_Model${MODEL_SIZE}_PP${PP}_TP${TP}_DP${DP}_"
# NYSY="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o ./trace_3d/_${MCORE_MODELS:+_mcore}_pptest_llama_GPUs${GPUS_PER_NODE}_Model${MODEL_SIZE}_PP${PP}_TP${TP}_DP${DP}_Layers${NUM_LAYERS}${OVERLAP_REDUCE:+_overlap}${BUCKET_SIZE:+_resetbucketsize}${DIST_OPTIM:+_distoptim}"
CMD="${NYSY} \
       ${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       ${TRACE_ARGS} \
       ${SIM_ARGS}
       "
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}