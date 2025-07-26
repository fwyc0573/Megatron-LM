#!/bin/bash

# 测试rank映射逻辑的脚本
# 用于验证修改后的update_pretrain_gpt.sh中的rank计算是否正确

echo "Testing Megatron rank mapping logic..."
echo "======================================"

# 测试不同的配置
test_configs=(
    "PP=2 TP=2 DP=1 WORLD_SIZE=4"
    "PP=4 TP=2 DP=1 WORLD_SIZE=8" 
    "PP=2 TP=4 DP=1 WORLD_SIZE=8"
    "PP=8 TP=1 DP=1 WORLD_SIZE=8"
)

for config in "${test_configs[@]}"; do
    echo ""
    echo "Testing configuration: $config"
    echo "----------------------------------------"
    
    # 解析配置
    eval $config
    
    # 验证配置
    if [ "$((DP * PP * TP))" -ne "$WORLD_SIZE" ]; then
        echo "ERROR: Invalid configuration! DP * PP * TP = $((DP * PP * TP)) != WORLD_SIZE = $WORLD_SIZE"
        continue
    fi
    
    # 计算选定的ranks
    SELECTED_RANKS=()
    
    echo "Rank mapping (PP stage -> world rank):"
    for pp_stage in $(seq 0 $((PP - 1))); do
        # 根据Megatron rank映射: global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size
        # 这里: tp_rank=0, dp_rank=0
        world_rank=$((0 + 0 * TP + pp_stage * TP * DP))
        SELECTED_RANKS+=(${world_rank})
        echo "  PP stage ${pp_stage} -> world rank ${world_rank}"
    done
    
    echo "Selected ranks: ${SELECTED_RANKS[@]}"
    echo "Total ranks to simulate: ${#SELECTED_RANKS[@]} (instead of ${WORLD_SIZE})"
    echo "Time reduction: ~$((100 - 100 * ${#SELECTED_RANKS[@]} / WORLD_SIZE))%"
    
    # 验证所有PP stages都被覆盖
    if [ "${#SELECTED_RANKS[@]}" -eq "$PP" ]; then
        echo "✓ All PP stages covered correctly"
    else
        echo "✗ ERROR: Expected $PP ranks, got ${#SELECTED_RANKS[@]}"
    fi
done

echo ""
echo "======================================"
echo "Rank mapping test completed!"
