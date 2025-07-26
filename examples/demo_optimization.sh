#!/bin/bash

# 演示脚本：展示优化前后的差异
# 模拟执行时间和rank选择

echo "Megatron-LM Workload Tracer Optimization Demo"
echo "=============================================="

# 配置参数
FAKE_PP=2
FAKE_TP=2
FAKE_DP=1
FAKE_WORLD_SIZE=4

echo ""
echo "Configuration:"
echo "- Pipeline Parallel (PP): ${FAKE_PP}"
echo "- Tensor Parallel (TP): ${FAKE_TP}"
echo "- Data Parallel (DP): ${FAKE_DP}"
echo "- Total World Size: ${FAKE_WORLD_SIZE}"

echo ""
echo "=============================================="
echo "BEFORE OPTIMIZATION:"
echo "=============================================="

echo "Original approach - simulating ALL ranks:"
original_ranks=($(seq 0 $((${FAKE_WORLD_SIZE} - 1))))
echo "Ranks to simulate: ${original_ranks[@]}"
echo "Total ranks: ${#original_ranks[@]}"

# 模拟执行时间
echo ""
echo "Simulated execution:"
for rank in "${original_ranks[@]}"; do
    echo "  [$(date '+%H:%M:%S')] Simulating rank ${rank}... (estimated 5 minutes)"
    sleep 0.5  # 模拟短暂延迟
done
echo "  Total estimated time: $((${#original_ranks[@]} * 5)) minutes"

echo ""
echo "=============================================="
echo "AFTER OPTIMIZATION:"
echo "=============================================="

echo "Optimized approach - simulating SELECTED ranks only:"

# 计算选定的ranks (与修改后的脚本逻辑相同)
SELECTED_RANKS=()
for pp_stage in $(seq 0 $((${FAKE_PP} - 1))); do
    world_rank=$((0 + 0 * ${FAKE_TP} + ${pp_stage} * ${FAKE_TP} * ${FAKE_DP}))
    SELECTED_RANKS+=(${world_rank})
done

echo "Ranks to simulate: ${SELECTED_RANKS[@]}"
echo "Total ranks: ${#SELECTED_RANKS[@]}"

echo ""
echo "Rank mapping details:"
for i in "${!SELECTED_RANKS[@]}"; do
    rank=${SELECTED_RANKS[$i]}
    echo "  PP stage ${i} -> world rank ${rank} (TP=0, DP=0)"
done

# 模拟执行时间
echo ""
echo "Simulated execution:"
for rank in "${SELECTED_RANKS[@]}"; do
    pp_stage=$((${rank} / (${FAKE_TP} * ${FAKE_DP})))
    echo "  [$(date '+%H:%M:%S')] Simulating rank ${rank} (PP stage ${pp_stage})... (estimated 5 minutes)"
    sleep 0.5  # 模拟短暂延迟
done
echo "  Total estimated time: $((${#SELECTED_RANKS[@]} * 5)) minutes"

echo ""
echo "=============================================="
echo "OPTIMIZATION SUMMARY:"
echo "=============================================="

time_reduction=$((100 - 100 * ${#SELECTED_RANKS[@]} / ${FAKE_WORLD_SIZE}))
time_saved=$(((${#original_ranks[@]} - ${#SELECTED_RANKS[@]}) * 5))

echo "Original ranks:     ${#original_ranks[@]}"
echo "Optimized ranks:    ${#SELECTED_RANKS[@]}"
echo "Time reduction:     ${time_reduction}%"
echo "Time saved:         ${time_saved} minutes"
echo ""
echo "✓ All PP stages covered with minimal rank simulation"
echo "✓ Complete workload pattern captured"
echo "✓ Significant time savings achieved"

echo ""
echo "=============================================="
echo "Demo completed!"
echo "=============================================="
