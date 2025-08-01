#! /bin/bash

# =============================================================================
# Mixtral 8×22B MoE Model Training Script for Megatron-LM
# =============================================================================
#
# 此脚本配置了 Mixtral 8×22B 模型的训练参数，基于官方架构规格：
# - 模型参数：141B (8 experts × 22B each)
# - 隐藏层维度：6144
# - 注意力头数：48 (包含 8 个 key-value heads for GQA)
# - 层数：56
# - FFN 隐藏层维度：16384
# - 词汇表大小：32768
# - 专家数量：8
# - 支持 64K context length
#
# 关键特性：
# - Group Query Attention (GQA)
# - RoPE 位置编码
# - SwiGLU 激活函数
# - RMSNorm 归一化
# - MoE with Top-2 routing
#
# 并行策略针对大模型优化：
# - Pipeline Parallelism: 4 (处理 56 层)
# - Expert Parallelism: 2 (8 experts / 2 = 4 experts per rank)
# - Data Parallelism: 2 (8 GPUs / 4 PP = 2)
# - Tensor Parallelism: 1 (MoE 最佳实践)
#
# =============================================================================

# Setting the environment variables
# export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN # WARN INFO
# export NCCL_ALGO=RING #Ring
# export GLOO_SOCKET_IFNAME="bond4"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6000
MASTER_ADDR="localhost" #"localhost"


BASE_PATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM #/data/ytyang/yichengfeng/Megatron-LM

# =============================================================================
# Model Type Selection
# =============================================================================
# 选择模型规格：Mixtral_8x7B 或 Mixtral_8x22B
MODEL_TYPE="Mixtral_8x22B"  # 可选: "Mixtral_8x7B" 或 "Mixtral_8x22B"

# =============================================================================
# Parallelism Strategy Configuration
# =============================================================================
# 针对不同 Mixtral 模型的并行策略优化

# 基础并行配置
if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    # Mixtral 8×7B (46.7B 参数) - 相对较小的模型
    PP=2  # Pipeline Parallelism
    TP=1  # Tensor Parallelism (MoE 模型通常 TP=1)
    EP=2  # Expert Parallelism (8 experts / 2 = 4 experts per rank)
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    # Mixtral 8×22B (141B 参数) - 大型模型，需要更多并行
    PP=4  # 增加 Pipeline Parallelism 以处理更多层
    TP=1  # 保持 TP=1 (MoE 最佳实践)
    EP=2  # Expert Parallelism (8 experts / 2 = 4 experts per rank)
fi

DP=$((${GPU_NUM}/${TP}/${PP}))

echo "=== Parallelism Configuration ==="
echo "Model Type: ${MODEL_TYPE}"
echo "Pipeline Parallel (PP): ${PP}"
echo "Tensor Parallel (TP): ${TP}"
echo "Expert Parallel (EP): ${EP}"
echo "Data Parallel (DP): ${DP}"
echo "Total GPUs: ${GPU_NUM}"
echo "================================="

# =============================================================================
# Batch Size Configuration
# =============================================================================
# 根据模型大小和 GPU 内存调整 batch size
if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    MICRO_BATCH_SIZE=2  # 可以使用较大的 micro batch size
    NUM_MICBATCH=1
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    MICRO_BATCH_SIZE=1  # 大模型使用较小的 micro batch size
    NUM_MICBATCH=1
fi

GLOBAL_BATCH_SIZE=$((NUM_MICBATCH * MICRO_BATCH_SIZE * DP))

echo "=== Batch Configuration ==="
echo "Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "Num Micro Batches: ${NUM_MICBATCH}"
echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "==========================="

# =============================================================================
# Mixtral Model Architecture Configuration
# =============================================================================
NUM_EXPERTS=8 # 专家总数（必须是EP的倍数）

# 选择模型规格：Mixtral_8x7B 或 Mixtral_8x22B
MODEL_TYPE="Mixtral_8x22B"  # 可选: "Mixtral_8x7B" 或 "Mixtral_8x22B"

# 根据官方 Mixtral 架构参数配置模型
if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    # Mixtral 8×7B 配置
    HIDDEN_SIZE=4096
    NUM_HEAD=32
    NUM_LAYERS=32
    FFN_HIDDEN_SIZE=14336
    NUM_KEY_VALUE_HEADS=8  # Group Query Attention
    VOCAB_SIZE=32000
    MODEL_SIZE="Mixtral_8x7B"
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    # Mixtral 8×22B 配置 (基于提供的参数表)
    HIDDEN_SIZE=6144
    NUM_HEAD=48
    NUM_LAYERS=56
    FFN_HIDDEN_SIZE=16384
    NUM_KEY_VALUE_HEADS=8  # Group Query Attention
    VOCAB_SIZE=32768  # 更大的词汇表
    MODEL_SIZE="Mixtral_8x22B"
else
    echo "Error: Invalid MODEL_TYPE: ${MODEL_TYPE}. Must be 'Mixtral_8x7B' or 'Mixtral_8x22B'"
    exit 1
fi

echo "=== Mixtral Model Configuration ==="
echo "Model Type: ${MODEL_TYPE}"
echo "Hidden Size: ${HIDDEN_SIZE}"
echo "Attention Heads: ${NUM_HEAD}"
echo "Key-Value Heads: ${NUM_KEY_VALUE_HEADS}"
echo "Layers: ${NUM_LAYERS}"
echo "FFN Hidden Size: ${FFN_HIDDEN_SIZE}"
echo "Experts: ${NUM_EXPERTS}"
echo "Vocab Size: ${VOCAB_SIZE}"
echo "=================================="

DO_TRACE=True
# TRACE控制参数
TRAIN_ITERS=10
TRACE_ITER_NUM=1 # trace_iter_num的范围<=train_iters-1（除去第一次）
TRACE_START=$(($TRAIN_ITERS-$TRACE_ITER_NUM+1)) # [start, train_iters]
NSIGHT_START=$(($TRAIN_ITERS)) # [start, train_iters)


# =============================================================================
# Sequence and Position Configuration
# =============================================================================
# 根据 Mixtral 架构调整序列长度和位置编码
if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    MAX_SEQ_LEN=4096  # Mixtral 8x7B 支持 32K context，但训练时使用较小值
    MAX_POSITION_EMBEDDINGS=32768
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    MAX_SEQ_LEN=4096  # Mixtral 8x22B 支持 64K context，但训练时使用较小值
    MAX_POSITION_EMBEDDINGS=65536
fi

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
# IS_SCALING_MODE=False
FAKE_PP=1
FAKE_TP=1
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
       --trace-memory \
       --trace-memory-interval 0.005 \
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


# =============================================================================
# Expert Parallelism Validation
# =============================================================================
# EP 已在上面的并行策略中定义，这里进行验证

echo "=== Expert Parallelism Validation ==="
echo "Number of Experts: ${NUM_EXPERTS}"
echo "Expert Parallel Size: ${EP}"
echo "Experts per EP rank: $((NUM_EXPERTS / EP))"

# 验证 EP 配置的合法性
if [ "$((NUM_EXPERTS % EP))" -ne "0" ]; then
    echo "Error: NUM_EXPERTS (${NUM_EXPERTS}) must be divisible by EP (${EP})"
    exit 1
fi

if [ "$EP" -gt "1" ]; then
    if [ "$TP" -ne "1" ]; then
        echo "Error: TP must be 1 for MoE models (current TP=${TP})"
        exit 1
    fi
    if [ "$((DP % EP))" -ne "0" ]; then
        echo "Error: DP (${DP}) must be divisible by EP (${EP})"
        exit 1
    fi
fi

echo "MoE Data Parallelism Size: $((DP / EP))"
echo "Expert Parallelism validation passed!"
echo "====================================="


LOG_NAME=moe_${MODEL_SIZE}_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}_EP${EP}_NUM_EXPERTS${NUM_EXPERTS}_seq${MAX_SEQ_LEN}_MICRO_BATCH_SIZE${MICRO_BATCH_SIZE}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log

# 确保日志目录存在
mkdir -p $(dirname ${LOG_PATH})



# =============================================================================
# Mixtral Model Training Arguments
# =============================================================================
GPT_ARGS="
    --tokenizer-type SentencePieceTokenizer \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEAD \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --vocab-size $VOCAB_SIZE \
    --make-vocab-size-divisible-by 128 \
    --lr 3e-4 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 3e-5 \
    --weight-decay 0.1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
"



# =============================================================================
# Mixtral MoE Specific Arguments
# =============================================================================
MOE_ARGS="
    --num-experts $NUM_EXPERTS \
    --expert-model-parallel-size $EP \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 0.01 \
    --moe-z-loss-coeff 0.001 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --disable-bias-linear \
"

# =============================================================================
# Mixtral Architecture Specific Arguments
# =============================================================================
MIXTRAL_ARGS="
    --bf16 \
    --no-position-embedding \
    --position-embedding-type rope \
    --rotary-interleaved \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --group-query-attention \
    --num-query-groups $NUM_KEY_VALUE_HEADS \
    --untie-embeddings-and-output-weights \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
"

echo "=== MoE Configuration ==="
echo "Router TopK: 2"
echo "Aux Loss Coeff: 0.01"
echo "Z Loss Coeff: 0.001"
echo "Token Dispatcher: alltoall"
echo "Grouped GEMM: enabled"
echo "========================="


# CHECKPOINT_PATH=<Specify path>
# BASE_PATH=/research/d1/gds/ytyang/yichengfeng/Megatron-LM
VOCAB_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-vocab.json
MERGE_FILE=${BASE_PATH}/data/output_prefix_gpt2/gpt2-merges.txt
DATA_PATH=${BASE_PATH}/data/output_prefix_gpt2/my-gpt2_text_document
# =============================================================================
# Data Configuration for Mixtral
# =============================================================================
# 注意：实际训练时应使用 SentencePiece tokenizer 和相应的数据
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
"
# 注意：vocab-size 已在 GPT_ARGS 中设置

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1 \
"


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6
# mgpuretest_distopt_tpcomm_sequen_2tp2pp2dp_2connection
# export HF_HOME="/research/d1/gds/ytyang/yichengfeng/.hf_saved_menu"
# export PYTHONPATH="${PYTHONPATH}:/data/ytyang/yichengfeng/DeepSpeed/Megatron-DeepSpeed/megatron"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NSYS_OUT_NAME="${MODEL_SIZE}_pp${PP}_dp${DP}_tp${TP}_ep${EP}_numexp${NUM_EXPERTS}"

# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --force-overwrite=true  -x true -o ${NSYS_OUT_NAME} \
# =============================================================================
# Launch Training
# =============================================================================
echo "=== Starting Mixtral ${MODEL_TYPE} Training ==="
echo "Log file: ${LOG_PATH}"
echo "=============================================="

torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NNODES} --node-rank ${NODE_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} ${BASE_PATH}/pretrain_llama.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $MIXTRAL_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $SIM_ARGS \
    $TRACE_ARGS \
    --distributed-backend nccl \
    --seed 42 \
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
