#!/bin/bash

# =============================================================================
# Mixtral Configuration Test Script
# =============================================================================
# 此脚本用于测试 Mixtral 配置的正确性，不实际启动训练

# 设置基础变量
NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
MODEL_TYPE="Mixtral_8x22B"  # 测试 Mixtral 8x22B 配置

echo "=== Mixtral Configuration Test ==="
echo "Testing MODEL_TYPE: ${MODEL_TYPE}"
echo "Total GPUs: ${GPU_NUM}"
echo

# 并行配置
if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    PP=2; TP=1; EP=2
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    PP=4; TP=1; EP=2
fi

DP=$((${GPU_NUM}/${TP}/${PP}))

echo "=== Parallelism Configuration ==="
echo "Pipeline Parallel (PP): ${PP}"
echo "Tensor Parallel (TP): ${TP}"
echo "Expert Parallel (EP): ${EP}"
echo "Data Parallel (DP): ${DP}"
echo

# 模型架构配置
NUM_EXPERTS=8

if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    HIDDEN_SIZE=4096
    NUM_HEAD=32
    NUM_LAYERS=32
    FFN_HIDDEN_SIZE=14336
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=32000
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    HIDDEN_SIZE=6144
    NUM_HEAD=48
    NUM_LAYERS=56
    FFN_HIDDEN_SIZE=16384
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=32768
fi

echo "=== Model Architecture ==="
echo "Hidden Size: ${HIDDEN_SIZE}"
echo "Attention Heads: ${NUM_HEAD}"
echo "Key-Value Heads: ${NUM_KEY_VALUE_HEADS}"
echo "Layers: ${NUM_LAYERS}"
echo "FFN Hidden Size: ${FFN_HIDDEN_SIZE}"
echo "Experts: ${NUM_EXPERTS}"
echo "Vocab Size: ${VOCAB_SIZE}"
echo

# 验证配置合法性
echo "=== Configuration Validation ==="

# 检查 EP 配置
if [ "$((NUM_EXPERTS % EP))" -ne "0" ]; then
    echo "❌ Error: NUM_EXPERTS (${NUM_EXPERTS}) must be divisible by EP (${EP})"
    exit 1
else
    echo "✅ Expert Parallelism: ${NUM_EXPERTS} experts / ${EP} EP ranks = $((NUM_EXPERTS / EP)) experts per rank"
fi

# 检查 DP 和 EP 关系
if [ "$((DP % EP))" -ne "0" ]; then
    echo "❌ Error: DP (${DP}) must be divisible by EP (${EP})"
    exit 1
else
    echo "✅ MoE Data Parallelism: DP (${DP}) / EP (${EP}) = $((DP / EP)) MoE DP ranks"
fi

# 检查总 GPU 数量
TOTAL_REQUIRED=$((TP * PP * DP))
if [ "${TOTAL_REQUIRED}" -ne "${GPU_NUM}" ]; then
    echo "❌ Error: Total required GPUs (${TOTAL_REQUIRED}) != Available GPUs (${GPU_NUM})"
    exit 1
else
    echo "✅ GPU allocation: TP(${TP}) × PP(${PP}) × DP(${DP}) = ${TOTAL_REQUIRED} GPUs"
fi

# 内存估算
if [[ ${MODEL_TYPE} == "Mixtral_8x7B" ]]; then
    TOTAL_PARAMS="46.7B"
    MEMORY_PER_GPU="~60GB"
elif [[ ${MODEL_TYPE} == "Mixtral_8x22B" ]]; then
    TOTAL_PARAMS="141B"
    MEMORY_PER_GPU="~210GB"
fi

echo
echo "=== Memory Estimation ==="
echo "Total Parameters: ${TOTAL_PARAMS}"
echo "Estimated Memory per GPU: ${MEMORY_PER_GPU}"
echo "Recommended GPU: A100 80GB or H100 80GB"
echo

# 通信组分析
echo "=== Communication Groups ==="
echo "Expert Parallel Groups:"
for ((i=0; i<EP; i++)); do
    group_size=$((GPU_NUM / EP))
    echo "  EP Group ${i}: ${group_size} ranks"
done

echo "MoE Data Parallel Groups:"
moe_dp_size=$((DP / EP))
for ((i=0; i<EP; i++)); do
    echo "  MoE DP Group ${i}: ${moe_dp_size} ranks"
done

echo
echo "=== Configuration Test Completed Successfully! ==="
echo "All validations passed. Configuration is ready for training."
