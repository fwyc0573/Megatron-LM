#!/usr/bin/env python3
"""
测试MoE模型通信组构成的脚本
验证与Megatron-LM官方实现的一致性
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.static_graphs.parallel_group_manager import ParallelGroupManager

def test_moe_communication_groups():
    """测试当前配置下的MoE通信组构成"""
    
    # 当前配置：dp_size=4, tp_size=1, pp_size=2, exp_size=2, 总8个GPU
    world_size = 8
    pp_size = 2
    tp_size = 1
    dp_size = 4
    exp_size = 2
    cp_size = 1
    local_size = 8  # 假设单节点
    
    print("=== MoE通信组构成测试 ===")
    print(f"配置: world_size={world_size}, pp_size={pp_size}, tp_size={tp_size}")
    print(f"      dp_size={dp_size}, exp_size={exp_size}, cp_size={cp_size}")
    print()
    
    # 创建ParallelGroupManager
    manager = ParallelGroupManager(
        local_size=local_size,
        world_size=world_size,
        pp_size=pp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        exp_size=exp_size,
        cp_size=cp_size
    )
    
    # 获取所有通信组（通过get_all_groups方法）
    groups_info = manager.get_all_groups()
    
    print("=== 基础并行组 ===")
    print(f"Pipeline Parallel Groups: {groups_info['pp_groups']}")
    print(f"Tensor Parallel Groups: {groups_info['tp_groups']}")
    print(f"Data Parallel Groups: {groups_info['dp_groups']}")
    print()

    print("=== MoE相关通信组 ===")
    print(f"Expert Model Parallel Groups: {groups_info['exp_groups']}")
    print(f"Data Modulo Expert Parallel Groups: {groups_info['dp_modulo_exp_groups']}")
    print()

    print("=== 复合通信组 ===")
    print(f"Tensor and Expert Parallel Groups: {groups_info['tp_exp_groups']}")
    print(f"Tensor and Data Parallel Groups: {groups_info['tp_dp_groups']}")
    print()

    # 分析通信操作映射
    print("=== 通信操作映射分析 ===")

    # 1. exp_allgather 和 exp_all_to_all 使用 expert_model_parallel_group
    exp_groups = groups_info['exp_groups']
    print(f"exp_allgather/exp_all_to_all 使用的通信组:")
    for i, group in enumerate(exp_groups):
        print(f"  Expert Group {i}: {group}")

    # 2. exp_dp_allreduce 使用 data_modulo_expert_parallel_group
    exp_dp_groups = groups_info['dp_modulo_exp_groups']
    print(f"exp_dp_allreduce 使用的通信组:")
    for i, group in enumerate(exp_dp_groups):
        print(f"  Expert DP Group {i}: {group}")
    
    print()
    
    # 验证通信组完整性
    print("=== 通信组完整性验证 ===")
    
    # 检查每个rank所属的通信组
    for rank_id in range(world_size):
        print(f"Rank {rank_id}:")
        
        # 找到所属的expert parallel group
        exp_group = None
        for i, group in enumerate(exp_groups):
            if rank_id in group:
                exp_group = (i, group)
                break
        
        # 找到所属的expert data parallel group
        exp_dp_group = None
        for i, group in enumerate(exp_dp_groups):
            if rank_id in group:
                exp_dp_group = (i, group)
                break
        
        print(f"  Expert Parallel Group: {exp_group}")
        print(f"  Expert DP Group: {exp_dp_group}")
        print()
    
    # 分析selected_ranks策略的影响
    print("=== Selected Ranks策略影响分析 ===")
    
    # 模拟当前的selected_ranks策略
    selected_ranks = set()
    
    # 基础选择：每个PP stage的代表rank
    for pp_rank in range(pp_size):
        representative_rank = pp_rank * tp_size * dp_size
        selected_ranks.add(representative_rank)
    
    # MoE额外选择：所有DP ranks (for pp_local_rank=0)
    for dp_rank in range(dp_size):
        moe_rank = dp_rank * tp_size
        selected_ranks.add(moe_rank)
    
    print(f"当前selected_ranks策略选择的ranks: {sorted(selected_ranks)}")
    print(f"缺失的ranks: {set(range(world_size)) - selected_ranks}")
    print()
    
    # 检查通信组完整性
    print("=== 通信组完整性检查 ===")
    
    for i, group in enumerate(exp_groups):
        missing_ranks = set(group) - selected_ranks
        if missing_ranks:
            print(f"❌ Expert Group {i} {group} 缺失ranks: {missing_ranks}")
        else:
            print(f"✅ Expert Group {i} {group} 完整")
    
    for i, group in enumerate(exp_dp_groups):
        missing_ranks = set(group) - selected_ranks
        if missing_ranks:
            print(f"❌ Expert DP Group {i} {group} 缺失ranks: {missing_ranks}")
        else:
            print(f"✅ Expert DP Group {i} {group} 完整")
    
    return groups_info, selected_ranks

if __name__ == "__main__":
    groups_info, selected_ranks = test_moe_communication_groups()
