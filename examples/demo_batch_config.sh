#!/bin/bash

# 演示批量配置功能的脚本
# 展示优化前后的差异和批量处理的优势

echo "Megatron-LM Workload Tracer - Batch Configuration Demo"
echo "======================================================"

# 示例配置数组
DEMO_CONFIGS=(
    "8192 16 8"   # world_size=8192, pp=16, tp=8
    "8192 32 8"   # world_size=8192, pp=32, tp=8  
    "8192 32 4"   # world_size=8192, pp=32, tp=4
    "4096 16 4"   # world_size=4096, pp=16, tp=4
)

echo ""
echo "Demo Configuration Set:"
echo "======================="
for i in "${!DEMO_CONFIGS[@]}"; do
    config="${DEMO_CONFIGS[$i]}"
    read -r world_size pp_size tp_size <<< "$config"
    dp_size=$((world_size / (pp_size * tp_size)))
    echo "  Config $((i+1)): WORLD_SIZE=${world_size}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
done

echo ""
echo "======================================================"
echo "BEFORE OPTIMIZATION (Traditional Approach)"
echo "======================================================"

total_original_time=0
echo "If running ALL ranks for each configuration:"
echo ""

for i in "${!DEMO_CONFIGS[@]}"; do
    config="${DEMO_CONFIGS[$i]}"
    read -r world_size pp_size tp_size <<< "$config"
    config_index=$((i+1))
    
    # 假设每个rank需要5分钟
    config_time=$((world_size * 5))
    total_original_time=$((total_original_time + config_time))
    
    echo "Config ${config_index}: ${world_size} ranks × 5 min = ${config_time} minutes"
done

echo ""
echo "Total time for all configurations: ${total_original_time} minutes"
echo "                                  = $((total_original_time / 60)) hours"
echo "                                  = $((total_original_time / 60 / 24)) days"

echo ""
echo "======================================================"
echo "AFTER OPTIMIZATION (Batch Configuration + Rank Selection)"
echo "======================================================"

total_optimized_time=0
total_original_ranks=0
total_optimized_ranks=0

echo "With rank optimization (only PP stages with TP=0, DP=0):"
echo ""

for i in "${!DEMO_CONFIGS[@]}"; do
    config="${DEMO_CONFIGS[$i]}"
    read -r world_size pp_size tp_size <<< "$config"
    dp_size=$((world_size / (pp_size * tp_size)))
    config_index=$((i+1))
    
    # 只需要运行PP_SIZE个ranks
    optimized_ranks=$pp_size
    config_time=$((optimized_ranks * 5))
    total_optimized_time=$((total_optimized_time + config_time))
    
    total_original_ranks=$((total_original_ranks + world_size))
    total_optimized_ranks=$((total_optimized_ranks + optimized_ranks))
    
    time_reduction=$((100 - 100 * optimized_ranks / world_size))
    
    echo "Config ${config_index}: ${optimized_ranks} ranks × 5 min = ${config_time} minutes (${time_reduction}% reduction)"
    echo "  Selected ranks: PP stages 0-$((pp_size-1)) with TP=0, DP=0"
    
    # 显示具体的rank映射
    echo "  Rank mapping: "
    for pp_stage in $(seq 0 $((pp_size > 4 ? 3 : pp_size-1))); do
        rank=$((pp_stage * tp_size * dp_size))
        echo "    PP stage ${pp_stage} -> world rank ${rank}"
    done
    if [ "$pp_size" -gt 4 ]; then
        echo "    ... (total ${pp_size} PP stages)"
    fi
    echo ""
done

echo "Total time for all configurations: ${total_optimized_time} minutes"
echo "                                  = $((total_optimized_time / 60)) hours"

echo ""
echo "======================================================"
echo "OPTIMIZATION SUMMARY"
echo "======================================================"

time_saved=$((total_original_time - total_optimized_time))
overall_reduction=$((100 - 100 * total_optimized_time / total_original_time))

echo "Original approach:"
echo "  Total ranks to simulate: ${total_original_ranks}"
echo "  Total time required: ${total_original_time} minutes ($((total_original_time / 60)) hours)"
echo ""
echo "Optimized batch approach:"
echo "  Total ranks to simulate: ${total_optimized_ranks}"
echo "  Total time required: ${total_optimized_time} minutes ($((total_optimized_time / 60)) hours)"
echo ""
echo "Optimization results:"
echo "  Time saved: ${time_saved} minutes ($((time_saved / 60)) hours)"
echo "  Overall reduction: ${overall_reduction}%"
echo "  Efficiency gain: $((total_original_time / total_optimized_time))x faster"

echo ""
echo "======================================================"
echo "BATCH CONFIGURATION ADVANTAGES"
echo "======================================================"

echo "✓ One-time setup: Define all configurations in a single array"
echo "✓ Automated execution: No manual parameter changes between runs"
echo "✓ Independent logging: Each configuration gets its own log directory"
echo "✓ Comprehensive statistics: Detailed optimization reports for each config"
echo "✓ Error isolation: Failed configurations don't affect others"
echo "✓ Easy comparison: Unified output format for performance analysis"

echo ""
echo "Example log directory structure:"
echo "logs/"
echo "├── SIM_GPT_485_Config1_WS8192_PP16_TP8_DP64_..."
echo "├── SIM_GPT_485_Config2_WS8192_PP32_TP8_DP32_..."
echo "├── SIM_GPT_485_Config3_WS8192_PP32_TP4_DP64_..."
echo "└── SIM_GPT_485_Config4_WS4096_PP16_TP4_DP64_..."

echo ""
echo "======================================================"
echo "USAGE EXAMPLE"
echo "======================================================"

echo "To use batch configuration in update_pretrain_gpt.sh:"
echo ""
echo "1. Edit the BATCH_CONFIGS array:"
echo "   BATCH_CONFIGS=("
echo "       \"8192 16 8\"   # world_size=8192, pp=16, tp=8"
echo "       \"8192 32 8\"   # world_size=8192, pp=32, tp=8"
echo "       \"8192 32 4\"   # world_size=8192, pp=32, tp=4"
echo "   )"
echo ""
echo "2. Run the script:"
echo "   bash examples/update_pretrain_gpt.sh"
echo ""
echo "3. Check results in individual log directories"

echo ""
echo "======================================================"
echo "Demo completed!"
echo "======================================================"
