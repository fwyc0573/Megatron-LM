#! /bin/bash

# ============================================================================
# MoE模型集中式批量模拟脚本 - 自动迭代pretrain_gpt_moe-copy3.sh的所有参数组合
# 
# 该脚本从pretrain_gpt_moe-copy3.sh中提取了所有的参数组合配置，
# 可以在单GPU上模拟分布式训练的每个rank的负载。
# 
# 脚本会自动迭代运行所有9个配置组合，每个配置都会模拟所有16个ranks。
#
# 使用方法：
# 1. 直接运行脚本: ./update_pretrain_gpt_moe-copy2.sh
# 2. 脚本会自动依次运行所有配置组合
# 3. 每个配置的日志会保存在独立的目录中
#
# 配置组合（来自pretrain_gpt_moe-copy3.sh）:
# - 配置1-9: 不同的PP, EP, NUM_EXPERTS, MOE_EXP_SIGNLE_SIZE, MAX_SEQ_LEN组合
# - 所有配置都使用FAKE_WORLD_SIZE=16 (对应2nodes×8gpus_per_node)
# - 总共会运行 9个配置 × 16个ranks = 144个模拟实例
# ============================================================================

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_DEBUG=WARN # WARN INFO
# export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"

export CUDA_VISIBLE_DEVICES=1 #0,1,2,3

# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=1
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6002
MASTER_ADDR="localhost"


# Fixed Parallelism variables 
PP=1
TP=1
EP=1
NUM_EXPERTS=1
DP=$((${GPU_NUM}/${TP}/${PP}))

BASE_PATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM

# 模拟的并行度设置 - 对应 pretrain_gpt_moe-copy3.sh 的所有参数组合
# 从 pretrain_gpt_moe-copy3.sh 提取的配置组合 (NNODES=2, GPUS_PER_NODE=8, WORLD_SIZE=16)
FAKE_WORLD_SIZE=16
FAKE_TP=1 # when TP>1, SP should be supported

# ==================== 配置组合定义 ====================
# 定义所有配置组合的数组 (PP EP NUM_EXPERTS MOE_EXP_SIGNLE_SIZE MAX_SEQ_LEN)
declare -a CONFIGS=(
    "2 2 8 1.75 1024"    # 配置1
    "2 4 8 1.75 1024"    # 配置2
    "2 8 8 1.75 1024"    # 配置3
    "2 4 16 1.75 1024"   # 配置4
    "2 8 16 1.75 1024"   # 配置5
    "4 2 8 1.75 1024"    # 配置6
    "4 4 8 1.75 1024"    # 配置7
    "4 4 8 1.75 1024"    # 配置8
    "4 4 16 1.75 1024"   # 配置9
)

# ==================== 配置组合定义结束 ====================

# 函数：设置当前配置的参数
set_config_params() {
    local config="$1"
    read FAKE_PP FAKE_EXP FAKE_NUM_EXPERTS MOE_EXP_SIGNLE_SIZE MAX_SEQ_LEN <<< "$config"
    
    # 计算派生参数
    FAKE_DP=$((${FAKE_WORLD_SIZE}/${FAKE_PP}/${FAKE_TP}))
    if [ "$((FAKE_DP * FAKE_PP * FAKE_TP))" -ne "$FAKE_WORLD_SIZE" ]; then
        echo "Error: FAKE_DP must be an integer for config: $config"
        return 1
    fi
    
    NUM_MICBATCH=4*${FAKE_PP}
    MICRO_BATCH_SIZE=1
    GLOBAL_BATCH_SZIE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * FAKE_DP))
    
    MODEL_SIZE="Mixtral_${FAKE_NUM_EXPERTS}x${MOE_EXP_SIGNLE_SIZE}B"
    
    # 设置模型参数
    if   [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=32; NUM_LAYERS=40;
    elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_LAYERS=80;
    elif [[ ${MODEL_SIZE} == 175 ]];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_LAYERS=96;
    elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=8; NUM_LAYERS=4;
    elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=7680;  NUM_HEAD=48; NUM_LAYERS=40;
    elif [[ ${MODEL_SIZE} == 40 ]];   then HIDDEN_SIZE=9216;  NUM_HEAD=72; NUM_LAYERS=40;
    elif [[ ${MODEL_SIZE} == 6.7 ]];  then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32;
    elif [[ ${MODEL_SIZE} == "Mixtral_${FAKE_NUM_EXPERTS}x1.75B" ]]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=8 ; FFN_HIDDEN_SIZE=14336;
    elif [[ ${MODEL_SIZE} == "Mixtral_${FAKE_NUM_EXPERTS}x7B" ]]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=14336;
    # elif [[ ${MODEL_SIZE} == "Mixtral_${FAKE_NUM_EXPERTS}x22B" ]]; then HIDDEN_SIZE=6144;  NUM_HEAD=48; NUM_LAYERS=56; FFN_HIDDEN_SIZE=16384;
    else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; return 1
    fi
    
    return 0
}

# 函数：生成GPT参数
generate_gpt_args() {
    GPT_ARGS="
        --tokenizer-type GPT2BPETokenizer \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
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
}

# 函数：生成MoE参数
generate_moe_args() {
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
        --no-position-embedding \
        --position-embedding-type rope \
        --normalization RMSNorm \
        --swiglu \
        --group-query-attention \
        --num-query-groups 8 \
        --untie-embeddings-and-output-weights \
    "
}

# 函数：生成SIM参数
generate_sim_args() {
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
           --trace-memory \
           --trace-memory-interval 0.001 \
           "
}

DO_TRACE=True
# TRACE控制参数
TRAIN_ITERS=5
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters)


MAX_POSITION_EMBEDDINGS=32768 # 4096 2048

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



# GPT_ARGS="
#     --main-tokenizer-type GPT2BPETokenizer \
#     --tensor-model-parallel-size $TP \
#     --pipeline-model-parallel-size $PP \
#     --num-layers $NUM_LAYERS \
#     --hidden-size $HIDDEN_SIZE \
#     --num-attention-heads $NUM_HEAD \
#     --seq-length $MAX_SEQ_LEN \
#     --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
#     --micro-batch-size $MICRO_BATCH_SIZE \
#     --global-batch-size $GLOBAL_BATCH_SZIE \
#     --lr 0.00015 \
#     --train-iters $TRAIN_ITERS \
#     --lr-decay-iters 320000 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
# "
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

# ==================== 主要执行逻辑 ====================
echo "============================================================================"
echo "Starting Megatron-LM MoE Configurations Batch Simulation"
echo "Total configurations to run: ${#CONFIGS[@]}"
echo "FAKE_WORLD_SIZE: ${FAKE_WORLD_SIZE}"
echo "============================================================================"

# 迭代所有配置组合
for config_idx in "${!CONFIGS[@]}"; do
    current_config="${CONFIGS[$config_idx]}"
    config_num=$((config_idx + 1))
    
    echo ""
    echo "========================================================================"
    echo ">>> Running Configuration ${config_num}/${#CONFIGS[@]}: ${current_config} <<<"
    echo "========================================================================"
    
    # 设置当前配置的参数
    if ! set_config_params "$current_config"; then
        echo "ERROR: Failed to set parameters for config: $current_config"
        continue
    fi
    
    # 生成所有参数（必须在set_config_params之后调用）
    generate_gpt_args
    generate_moe_args
    generate_sim_args
    
    # 创建配置特定的日志目录
    LOG_NAME=SIM_moe_${MODEL_SIZE}_FakeWS${FAKE_WORLD_SIZE}_TP${FAKE_TP}_PP${FAKE_PP}_EP${FAKE_EXP}_NUM_EXPERTS${FAKE_NUM_EXPERTS}_seq${MAX_SEQ_LEN}_MICRO_BATCH_SIZE${MICRO_BATCH_SIZE}
    LOG_DIR=${BASE_PATH}/log/${LOG_NAME}
    mkdir -p ${LOG_DIR}
    
    echo "Configuration Details:"
    echo "  - Model: ${MODEL_SIZE}"
    echo "  - PP=${FAKE_PP}, TP=${FAKE_TP}, DP=${FAKE_DP}, EP=${FAKE_EXP}"
    echo "  - NUM_EXPERTS=${FAKE_NUM_EXPERTS}, SEQ_LEN=${MAX_SEQ_LEN}"
    echo "  - Log Directory: ${LOG_DIR}"
    echo ""
    
    # 迭代所有模拟的rank
    for current_fake_rank_id in $(seq 0 $((${FAKE_WORLD_SIZE} - 1))); do
        echo "------------------------------------------------------------"
        echo ">>> Config ${config_num}/${#CONFIGS[@]} - Simulating FAKE RANK ID: ${current_fake_rank_id} / $((${FAKE_WORLD_SIZE} - 1)) <<<"
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
            echo "ERROR: Python script failed for config ${config_num}, rank ${current_fake_rank_id} with status ${exit_status}"
            echo "Check log: ${RANK_LOG_PATH}"
            echo "Continuing with next rank..."
            continue
        fi

        echo ">>> Finished simulation for Config ${config_num} - FAKE RANK ID: ${current_fake_rank_id} <<<"
    done
    
    echo ""
    echo ">>> Configuration ${config_num}/${#CONFIGS[@]} completed successfully <<<"
    echo ">>> Log directory: ${LOG_DIR} <<<"
    echo ""
done

echo "============================================================================"
echo "All configurations batch simulation completed!"
echo "Total configurations processed: ${#CONFIGS[@]}"
echo "Base log directory: ${BASE_PATH}/log/"
echo "Profiler logs in: ${BASE_PATH}/profiler_log/"
echo "============================================================================"