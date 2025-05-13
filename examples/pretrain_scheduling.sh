#! /bin/bash

# Setting the environment variables
# export OMP_NUM_THREADS=1
# export PYTHONPATH="${PYTHONPATH}:/research/d1/gds/ytyang/yichengfeng/Megatron-LM/megatron"
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
MASTER_ADDR=192.168.50.186 #"localhost"

# Parallelism variables
TP=2
PP=2
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables
MODEL_SIZE="tiny"

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=3; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=512 # 4096
MAX_POSITION_EMBEDDINGS=512 # 4096

# Paths
BASE_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM
SRC_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM/pretrain_llama_scheduling.py

LOG_NAME=llama2_${MODEL_SIZE}_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

DATA_PATH=${BASE_PATH}/data/output_prefix_llama/output_prefix_llama_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

# SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
# LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
# mkdir -p ${BASE_PATH}/log/${LOG_NAME}

TOKENIZER_PATH=${BASE_PATH}/tokenizers/Llama2Tokenizer/tokenizer.model

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
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
DIST_OPTIM="--use-distributed-optimizer"

DISTRIBUTED_ARGS=" \
       --world-size 8
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
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --train-iters 1 \
       --log-interval 1 \
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
       "
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}