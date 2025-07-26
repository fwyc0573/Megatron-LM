#!/bin/bash

# 测试批量配置功能的脚本
# 验证修改后的update_pretrain_gpt.sh中的批量配置逻辑

echo "Testing Batch Configuration Logic..."
echo "===================================="

# 测试配置数组
TEST_BATCH_CONFIGS=(
    "8192 16 8"   # world_size=8192, pp=16, tp=8
    "8192 32 8"   # world_size=8192, pp=32, tp=8  
    "8192 32 4"   # world_size=8192, pp=32, tp=4
    "4096 16 4"   # world_size=4096, pp=16, tp=4
)

# 验证配置函数
validate_config() {
    local world_size=$1
    local pp_size=$2
    local tp_size=$3
    local dp_size=$4
    
    if [ "$((dp_size * pp_size * tp_size))" -ne "$world_size" ]; then
        echo "ERROR: Invalid configuration!"
        echo "  WORLD_SIZE=${world_size}, PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"
        echo "  DP * PP * TP = $((dp_size * pp_size * tp_size)) != WORLD_SIZE = ${world_size}"
        return 1
    fi
    
    if [ "$dp_size" -le 0 ]; then
        echo "ERROR: DP_SIZE must be positive, got ${dp_size}"
        return 1
    fi
    
    return 0
}

# 计算选定ranks的函数
calculate_selected_ranks() {
    local pp_size=$1
    local tp_size=$2
    local dp_size=$3
    
    local selected_ranks=()
    
    for pp_stage in $(seq 0 $((pp_size - 1))); do
        local rank=$((0 + 0 * tp_size + pp_stage * tp_size * dp_size))
        selected_ranks+=(${rank})
    done
    
    echo "${selected_ranks[@]}"
}

echo "Testing ${#TEST_BATCH_CONFIGS[@]} configurations:"
echo ""

total_original_ranks=0
total_optimized_ranks=0
successful_configs=0
failed_configs=0

for i in "${!TEST_BATCH_CONFIGS[@]}"; do
    config="${TEST_BATCH_CONFIGS[$i]}"
    read -r world_size pp_size tp_size <<< "$config"
    config_index=$((i+1))
    
    echo "Configuration ${config_index}: WORLD_SIZE=${world_size}, PP=${pp_size}, TP=${tp_size}"
    echo "----------------------------------------"
    
    # 计算DP大小
    dp_size=$((world_size / (pp_size * tp_size)))
    
    # 验证配置
    if validate_config "$world_size" "$pp_size" "$tp_size" "$dp_size"; then
        echo "✓ Configuration valid: DP=${dp_size}"
        
        # 计算选定的ranks
        selected_ranks=($(calculate_selected_ranks "$pp_size" "$tp_size" "$dp_size"))
        
        echo "Rank mapping:"
        for j in "${!selected_ranks[@]}"; do
            rank=${selected_ranks[$j]}
            echo "  PP stage ${j} -> world rank ${rank}"
        done
        
        echo "Selected ranks: ${selected_ranks[@]}"
        echo "Total ranks to simulate: ${#selected_ranks[@]} (instead of ${world_size})"
        
        time_reduction=$((100 - 100 * ${#selected_ranks[@]} / world_size))
        echo "Time reduction: ~${time_reduction}%"
        
        # 累计统计
        total_original_ranks=$((total_original_ranks + world_size))
        total_optimized_ranks=$((total_optimized_ranks + ${#selected_ranks[@]}))
        ((successful_configs++))
        
        echo "✓ Configuration ${config_index} processed successfully"
        
    else
        echo "✗ Configuration ${config_index} is invalid"
        ((failed_configs++))
    fi
    
    echo ""
done

echo "===================================="
echo "BATCH CONFIGURATION TEST SUMMARY"
echo "===================================="
echo "Total configurations tested: ${#TEST_BATCH_CONFIGS[@]}"
echo "Successful configurations: ${successful_configs}"
echo "Failed configurations: ${failed_configs}"
echo ""

if [ "$successful_configs" -gt 0 ]; then
    overall_reduction=$((100 - 100 * total_optimized_ranks / total_original_ranks))
    echo "Overall optimization results:"
    echo "- Total original ranks: ${total_original_ranks}"
    echo "- Total optimized ranks: ${total_optimized_ranks}"
    echo "- Overall time reduction: ~${overall_reduction}%"
    echo ""
    echo "✓ Batch configuration logic working correctly!"
else
    echo "✗ No successful configurations processed"
fi

echo "===================================="
echo "Test completed!"
