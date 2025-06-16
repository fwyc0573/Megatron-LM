#! /bin/bash

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN # WARN INFO
# export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"

export CUDA_VISIBLE_DEVICES=4,5,6,7 #0,1,2,3

# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=4
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6000
MASTER_ADDR="localhost" #"localhost"


# Parallelism variables 
PP=2
TP=2
DP=$((${GPU_NUM}/${TP}/${PP}))


NUM_MICBATCH=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SZIE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * DP))

# size variables
MODEL_SIZE=6.7 # "tiny" 6.7

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 6.7 ]];  then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


BASE_PATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM
LOG_NAME=GPT_${MODEL_SIZE}_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log

# 确保日志目录存在
mkdir -p $(dirname ${LOG_PATH})


DO_TRACE=True
# TRACE控制参数
TRAIN_ITERS=10
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters)


MAX_SEQ_LEN=512 # 4096 2048 1024
MAX_POSITION_EMBEDDINGS=512 # 4096 2048 1024

# 检查trace_iter_num是否在合理的范围内
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
# IS_SCALING_MODE=Falsef
FAKE_PP=2
FAKE_TP=4
FAKE_DP=$((FAKE_WORLD_SIZE / FAKE_PP / FAKE_TP))
if [ "$((FAKE_DP * FAKE_PP * FAKE_TP))" -ne "$FAKE_WORLD_SIZE" ]; then
    echo "Error: FAKE_DP must be an integer."
    exit 1
fi

#        --is-scaling-mode \
SIM_ARGS=" \
       --fake-world-size $FAKE_WORLD_SIZE \
       --fake-wrank $FAKE_WRANK \
       --fake-gpus-per-node $FAKE_GPUS_PER_NODE \
       --fake-local-rank $FAKE_LOCAL_RANK \
       --fake-pp $FAKE_PP \
       --fake-dp $FAKE_DP \
       --fake-tp $FAKE_TP \
       "
    #    --trace-memory \
    #    --trace-memory-interval 0.005 \

# 当采用is-scaling-mode时,采用单个rank进行PROFILE
if echo "$SIM_ARGS" | grep -q -- "--is-scaling-mode"; then
    GPUS_PER_NODE=1
    GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
    WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
    TP=1
    PP=1
    DP=$((${GPU_NUM}/${TP}/${PP}))
fi



# EP=2  # 专家并行度
# NUM_EXPERTS=6  # 专家总数（必须是EP的倍数）

# # 确保专家数量能被EP整除
# if [ "$((NUM_EXPERTS % EP))" -ne "0" ]; then
#     echo "Error: NUM_EXPERTS must be divisible by EP"
#     exit 1
# fi

# MOE_ARGS="
#     --num-experts $NUM_EXPERTS \
#     --expert-model-parallel-size $EP \
#     --moe-router-load-balancing-type aux_loss \
#     --moe-router-topk 2 \
#     --moe-aux-loss-coeff 1e-2 \
#     --moe-token-dispatcher-type alltoall \
#     --disable-bias-linear
#     --moe-grouped-gemm \
#     --bf16 
# "
    # 以下参数需要捆绑使用只使用, --moe-token-dispatcher-type默认是allgather，且可正常使用
    # --moe-token-dispatcher-type alltoall \
    # --disable-bias-linear

    # --moe-grouped-gemm \
    # --bf16 \

    # 不开启moe_input_jitter_eps


GPT_ARGS="
    --main-tokenizer-type GPT2BPETokenizer \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEAD \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SZIE \
    --lr 0.00015 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
"
    # --fp16
    # --mock-data

# CHECKPOINT_PATH=<Specify path>
# BASE_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM
VOCAB_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-vocab.json
MERGE_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-merges.txt
DATA_PATH=${BASE_PATH}/data/output_prefix_gpt2/my-gpt2_text_document
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --vocab-size 1000 
"
# --vocab-size 3200

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1
"


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6
# mgpuretest_distopt_tpcomm_sequen_2tp2pp2dp_2connection
# export HF_HOME="/research/d1/gds/ytyang/yichengfeng/.hf_saved_menu"
# export PYTHONPATH="${PYTHONPATH}:/data/ytyang/yichengfeng/DeepSpeed/Megatron-DeepSpeed/megatron"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o optimzer_find_test—_tp2pp4 \
torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${BASE_PATH}/pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $SIM_ARGS \
    $TRACE_ARGS \
    --distributed-backend nccl \
    --use-mcore-models 2>&1 | tee ${LOG_PATH}


    # --overlap-grad-reduce \
    # --overlap-param-gather \
    # --tp-comm-overlap \
    # --sequence-parallel
    # --use-distributed-optimizer \

    #  --sequence-parallel \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
    # --overlap-param-gather \
