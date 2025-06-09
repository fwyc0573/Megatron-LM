#! /bin/bash

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN # WARN INFO
# export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"

export CUDA_VISIBLE_DEVICES=7 #0,1,2,3

# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=1
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6000
MASTER_ADDR="localhost"


# Parallelism variables 
PP=1
TP=1
EP=1
NUM_EXPERTS=1
DP=$((${GPU_NUM}/${TP}/${PP}))

BASE_PATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM

# 模拟的并行度设置
FAKE_PP=3
FAKE_TP=1 # when TP>1, SP should be supported
FAKE_EXP=2
FAKE_NUM_EXPERTS=6
FAKE_WORLD_SIZE=6
FAKE_DP=$((${FAKE_WORLD_SIZE}/${FAKE_PP}/${FAKE_TP}))
if [ "$((FAKE_DP * FAKE_PP * FAKE_TP))" -ne "$FAKE_WORLD_SIZE" ]; then
    echo "Error: FAKE_DP must be an integer."
    exit 1
fi

# 创建统一的日志目录
MODEL_SIZE="tiny"  # "tiny" # 使用原脚本中的模型大小
LOG_NAME=SIM_GPT_${MODEL_SIZE}_FakeWS${FAKE_WORLD_SIZE}_TP${FAKE_TP}_PP${FAKE_PP}_EXP${FAKE_EXP}_EX${FAKE_NUM_EXPERTS}
LOG_DIR=${BASE_PATH}/log/${LOG_NAME}

# 确保日志目录存在
mkdir -p ${LOG_DIR}


NUM_MICBATCH=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SZIE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * FAKE_DP)) # 使用FAKE_DP计算全局批次大小

if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
elif [[ ${MODEL_SIZE} == 6.7 ]];  then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


DO_TRACE=True
# TRACE控制参数
TRAIN_ITERS=10
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters)


MAX_SEQ_LEN=2048 # 4096 2048 1024
MAX_POSITION_EMBEDDINGS=2048 # 4096 2048 1024

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

FAKE_GPUS_PER_NODE=8
FAKE_WRANK=0
FAKE_LOCAL_RANK=0
# IS_SCALING_MODE=Falsef

SIM_ARGS=" \
       --is-scaling-mode \
       --fake-world-size $FAKE_WORLD_SIZE \
       --fake-wrank $FAKE_WRANK \
       --fake-gpus-per-node $FAKE_GPUS_PER_NODE \
       --fake-local-rank $FAKE_LOCAL_RANK \
       --fake-pp $FAKE_PP \
       --fake-dp $FAKE_DP \
       --fake-tp $FAKE_TP \
       --fake-exp $FAKE_EXP \
       --fake-num-experts $FAKE_NUM_EXPERTS \
       "


MOE_ARGS="
    --num-experts $NUM_EXPERTS \
    --expert-model-parallel-size $EP \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-2 \
    --moe-token-dispatcher-type alltoall \
    --disable-bias-linear \
    --moe-grouped-gemm \
    --bf16 \
"
    # plz use these paras together, --moe-token-dispatcher-type (default: allgather)
    # --moe-token-dispatcher-type alltoall \
    # --disable-bias-linear

    # --moe-grouped-gemm \
    # --bf16 \

    # close moe_input_jitter_eps

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
    --vocab-size 3200 \
"
# --vocab-size 3200

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1 \
"

# 添加迭代执行逻辑
echo "============================================================"
echo "Starting Megatron-LM Single GPU Simulation for ${FAKE_WORLD_SIZE} ranks"
echo "Model: GPT-${MODEL_SIZE}, PP=${FAKE_PP}, TP=${FAKE_TP}, DP=${FAKE_DP}, EP=${FAKE_EXP}, NUM_EXPERTS=${FAKE_NUM_EXPERTS}"
echo "============================================================"

# 迭代所有模拟的rank
for current_fake_rank_id in $(seq 0 $((${FAKE_WORLD_SIZE} - 1)))
do
    echo "------------------------------------------------------------"
    echo ">>> Simulating FAKE RANK ID: ${current_fake_rank_id} / $((${FAKE_WORLD_SIZE} - 1)) <<<"
    echo "------------------------------------------------------------"

    # 为当前rank创建日志文件
    RANK_LOG_PATH="${LOG_DIR}/fake_rank_${current_fake_rank_id}.log"
    
    echo "Running simulation for rank ${current_fake_rank_id}, log: ${RANK_LOG_PATH}"

    # 使用torchrun执行Python脚本，添加fake-current-rank-id参数
    torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${BASE_PATH}/pretrain_llama.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $MOE_ARGS \
        $SIM_ARGS \
        $TRACE_ARGS \
        --fake-current-rank-id ${current_fake_rank_id} \
        --distributed-backend nccl \
        --use-mcore-models \
        --seed 42 2>&1 | tee ${RANK_LOG_PATH}

    # 检查执行状态
    exit_status=${PIPESTATUS[0]}
    if [ ${exit_status} -ne 0 ]; then
        echo "ERROR: Python script failed for rank ${current_fake_rank_id} with status ${exit_status}"
        echo "Check log: ${RANK_LOG_PATH}"
        exit 1
    fi

    echo ">>> Finished simulation for FAKE RANK ID: ${current_fake_rank_id} <<<"
done

echo "============================================================"
echo "All fake rank simulations completed successfully"
echo "Log directory: ${LOG_DIR}"
echo "Profiler logs in: ${BASE_PATH}/profiler_log/"
echo "============================================================"