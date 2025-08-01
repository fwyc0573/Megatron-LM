#!/usr/bin/env python3
"""
简化版本的 MoE rank 映射分析
基于 Megatron-LM 源码逻辑手动计算
"""

def analyze_moe_mapping():
    """基于源码逻辑手动分析 MoE 映射关系"""
    
    # 配置参数
    world_size = 8
    tp = 1  # Tensor Parallelism
    pp = 2  # Pipeline Parallelism  
    ep = 2  # Expert Parallelism
    cp = 1  # Context Parallelism
    dp = world_size // (tp * pp * cp)  # Data Parallelism = 4
    num_experts = 8
    
    print("=== 配置参数 ===")
    print(f"World Size: {world_size}")
    print(f"TP: {tp}, PP: {pp}, DP: {dp}, EP: {ep}, CP: {cp}")
    print(f"Total Experts: {num_experts}")
    print(f"Experts per EP rank: {num_experts // ep}")
    print()
    
    # 根据 Megatron-LM 的 order="tp-cp-ep-dp-pp" 和源码逻辑
    # 当 independent_ep=True 时，DP 被分解为 DP//EP
    
    print("=== 基于源码逻辑的 Rank 分布 ===")
    
    # 根据 order="tp-cp-ep-dp-pp" 和 independent_ep=True
    # ordered_size_w_ep = [tp, cp, ep, dp//ep, pp] = [1, 1, 2, 2, 2]
    
    # 对于 EP groups (mask=[False, False, True, False, False])
    # 每个 EP group 包含 world_size // ep = 4 个 ranks
    print("1. Expert Parallel Groups:")
    ep_groups = []
    for ep_rank in range(ep):
        group = []
        for i in range(world_size // ep):
            # 根据 order 计算 global rank
            # rank = tp_rank + cp_rank*tp + ep_rank*tp*cp + dp_rank*tp*cp*ep + pp_rank*tp*cp*ep*(dp//ep)
            tp_rank = 0  # tp=1, 所以 tp_rank=0
            cp_rank = 0  # cp=1, 所以 cp_rank=0
            dp_rank = i % (dp // ep)  # dp//ep = 2
            pp_rank = i // (dp // ep)  # pp = 2
            
            global_rank = tp_rank + cp_rank*tp + ep_rank*tp*cp + dp_rank*tp*cp*ep + pp_rank*tp*cp*ep*(dp//ep)
            group.append(global_rank)
        ep_groups.append(group)
        print(f"   EP Group {ep_rank}: {group}")
    
    print()
    
    # 对于 DP groups (modulo EP) (mask=[False, False, False, True, False])
    print("2. Data Parallel Groups (modulo EP):")
    dp_groups = []
    for dp_rank in range(dp // ep):
        group = []
        for i in range(world_size // (dp // ep)):
            # 计算其他维度的组合
            tp_rank = 0
            cp_rank = 0
            ep_rank = i % ep
            pp_rank = i // ep
            
            global_rank = tp_rank + cp_rank*tp + ep_rank*tp*cp + dp_rank*tp*cp*ep + pp_rank*tp*cp*ep*(dp//ep)
            group.append(global_rank)
        dp_groups.append(group)
        print(f"   DP Group {dp_rank}: {group}")
    
    print()
    
    # 对于 PP groups (mask=[False, False, False, False, True])
    print("3. Pipeline Parallel Groups:")
    pp_groups = []
    for pp_rank in range(pp):
        group = []
        for i in range(world_size // pp):
            # 计算其他维度的组合
            tp_rank = 0
            cp_rank = 0
            ep_rank = i % ep
            dp_rank = i // ep
            
            global_rank = tp_rank + cp_rank*tp + ep_rank*tp*cp + dp_rank*tp*cp*ep + pp_rank*tp*cp*ep*(dp//ep)
            group.append(global_rank)
        pp_groups.append(group)
        print(f"   PP Group {pp_rank}: {group}")
    
    print()
    print("=== 每个 Rank 的详细信息 ===")
    
    for rank in range(world_size):
        # 根据 rank 反推各个并行维度的 rank
        # rank = tp_rank + cp_rank*tp + ep_rank*tp*cp + dp_rank*tp*cp*ep + pp_rank*tp*cp*ep*(dp//ep)
        
        remaining = rank
        tp_rank = remaining % tp
        remaining //= tp
        
        cp_rank = remaining % cp
        remaining //= cp
        
        ep_rank = remaining % ep
        remaining //= ep
        
        dp_rank = remaining % (dp // ep)
        remaining //= (dp // ep)
        
        pp_rank = remaining
        
        # 计算该 rank 负责的 experts
        num_local_experts = num_experts // ep
        local_expert_indices_offset = ep_rank * num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]
        
        print(f"Rank {rank}: TP={tp_rank}, CP={cp_rank}, EP={ep_rank}, DP={dp_rank}, PP={pp_rank}, "
              f"Local Experts: {local_expert_indices}")
    
    print()
    print("=== EP RANK 与 DP RANK 映射关系 ===")
    
    # 分析 EP RANK 与 DP RANK 的映射关系
    ep_to_dp_mapping = {}
    dp_to_ep_mapping = {}
    
    for rank in range(world_size):
        remaining = rank
        remaining //= tp  # skip tp
        remaining //= cp  # skip cp
        ep_rank = remaining % ep
        remaining //= ep
        dp_rank = remaining % (dp // ep)
        
        if ep_rank not in ep_to_dp_mapping:
            ep_to_dp_mapping[ep_rank] = []
        ep_to_dp_mapping[ep_rank].append((rank, dp_rank))
        
        if dp_rank not in dp_to_ep_mapping:
            dp_to_ep_mapping[dp_rank] = []
        dp_to_ep_mapping[dp_rank].append((rank, ep_rank))
    
    print("EP RANK -> (Global Rank, DP RANK) 映射:")
    for ep_rank in sorted(ep_to_dp_mapping.keys()):
        print(f"   EP RANK {ep_rank}: {ep_to_dp_mapping[ep_rank]}")
    
    print("\nDP RANK -> (Global Rank, EP RANK) 映射:")
    for dp_rank in sorted(dp_to_ep_mapping.keys()):
        print(f"   DP RANK {dp_rank}: {dp_to_ep_mapping[dp_rank]}")
    
    print()
    print("=== 隐式 MoE Data Parallelism 分析 ===")
    
    moe_data_parallel_size = dp // ep
    print(f"隐式 MoE Data Parallelism Size = DP_SIZE / EP_SIZE = {dp} / {ep} = {moe_data_parallel_size}")
    
    # 验证：每个 EP rank 应该有 moe_data_parallel_size 个对应的 DP ranks
    for ep_rank in sorted(ep_to_dp_mapping.keys()):
        dp_ranks = [dp_rank for _, dp_rank in ep_to_dp_mapping[ep_rank]]
        print(f"EP RANK {ep_rank} 对应的 DP RANKs: {dp_ranks} (数量: {len(dp_ranks)})")
    
    print()
    print("=== Expert 在物理 GPU 上的分布 ===")
    
    # 分析 experts 在物理 GPU 上的分布
    gpu_to_experts = {}
    for rank in range(world_size):
        remaining = rank
        remaining //= tp
        remaining //= cp
        ep_rank = remaining % ep
        
        num_local_experts = num_experts // ep
        local_expert_indices_offset = ep_rank * num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]
        
        gpu_to_experts[rank] = local_expert_indices
    
    print("GPU -> Local Experts 映射:")
    for gpu in sorted(gpu_to_experts.keys()):
        print(f"   GPU {gpu}: Experts {gpu_to_experts[gpu]}")

if __name__ == "__main__":
    analyze_moe_mapping()
