#!/usr/bin/env python3
"""
测试脚本：分析 Megatron-LM MoE 中 Expert Parallelism 的 rank 映射关系
基于当前配置：8 GPU, PP=2, TP=1, DP=4, EP=2, NUM_EXPERTS=8
"""

# 简化版本的 RankGenerator 实现，基于 Megatron-LM 源码逻辑
def inner_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def decompose(index, shape):
    """将一维索引分解为多维索引"""
    result = []
    for s in reversed(shape):
        result.append(index % s)
        index //= s
    return list(reversed(result))

def generate_masked_orthogonal_rank_groups(world_size, parallel_size, mask):
    """生成正交的 rank 组"""
    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    if not masked_shape:
        return [[i for i in range(world_size)]]
    if not unmasked_shape:
        return [[i] for i in range(world_size)]

    def get_stride(shape):
        stride = [1]
        for s in reversed(shape[1:]):
            stride.append(stride[-1] * s)
        return list(reversed(stride))

    masked_stride = get_stride(masked_shape)
    unmasked_stride = get_stride(unmasked_shape)

    group_size = inner_product(masked_shape, [1] * len(masked_shape))
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks

class RankGenerator:
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        if 'ep' in order:
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)
        return ranks

def analyze_moe_rank_mapping():
    """分析 MoE rank 映射关系"""
    
    # 当前配置参数
    world_size = 8
    tp = 1  # Tensor Parallelism
    pp = 2  # Pipeline Parallelism  
    ep = 2  # Expert Parallelism
    cp = 1  # Context Parallelism
    dp = world_size // (tp * pp * cp)  # Data Parallelism = 8 // (1 * 2 * 1) = 4
    num_experts = 8
    
    print("=== MoE 配置分析 ===")
    print(f"World Size: {world_size}")
    print(f"TP: {tp}, PP: {pp}, DP: {dp}, EP: {ep}, CP: {cp}")
    print(f"Total Experts: {num_experts}")
    print(f"Experts per EP rank: {num_experts // ep}")
    print()
    
    # 创建 RankGenerator
    rank_generator = RankGenerator(
        tp=tp,
        ep=ep, 
        dp=dp,
        pp=pp,
        cp=cp,
        order="tp-cp-ep-dp-pp"
    )
    
    print("=== Rank Groups 分析 ===")
    
    # 1. Expert Parallel Groups
    print("1. Expert Parallel Groups (EP):")
    ep_groups = list(rank_generator.get_ranks('ep', independent_ep=True))
    for i, group in enumerate(ep_groups):
        print(f"   EP Group {i}: {group}")
    print()
    
    # 2. Data Parallel Groups (modulo EP)
    print("2. Data Parallel Groups (modulo EP):")
    dp_groups = list(rank_generator.get_ranks('dp', independent_ep=True))
    for i, group in enumerate(dp_groups):
        print(f"   DP Group {i}: {group}")
    print()
    
    # 3. Full Data Parallel Groups (without EP consideration)
    print("3. Full Data Parallel Groups (without EP):")
    full_dp_groups = list(rank_generator.get_ranks('dp', independent_ep=False))
    for i, group in enumerate(full_dp_groups):
        print(f"   Full DP Group {i}: {group}")
    print()
    
    # 4. Pipeline Parallel Groups
    print("4. Pipeline Parallel Groups (PP):")
    pp_groups = list(rank_generator.get_ranks('pp', independent_ep=True))
    for i, group in enumerate(pp_groups):
        print(f"   PP Group {i}: {group}")
    print()
    
    print("=== Expert 分布分析 ===")
    
    # 分析每个 rank 负责的 experts
    for rank in range(world_size):
        # 计算该 rank 的 EP rank
        # 根据源码：get_expert_model_parallel_rank() 的实现
        # EP rank = tensor_and_expert_parallel_rank // tensor_model_parallel_size

        # 找到该 rank 属于哪个 EP group
        ep_rank = None
        ep_group_id = None
        for i, group in enumerate(ep_groups):
            if rank in group:
                ep_rank = group.index(rank)
                ep_group_id = i
                break

        # 找到该 rank 属于哪个 DP group (modulo EP)
        dp_rank = None
        dp_group_id = None
        for i, group in enumerate(dp_groups):
            if rank in group:
                dp_rank = group.index(rank)
                dp_group_id = i
                break

        # 找到该 rank 属于哪个 PP group
        pp_rank = None
        pp_group_id = None
        for i, group in enumerate(pp_groups):
            if rank in group:
                pp_rank = group.index(rank)
                pp_group_id = i
                break

        # 计算该 rank 负责的 experts
        if ep_rank is not None:
            num_local_experts = num_experts // ep
            local_expert_indices_offset = ep_rank * num_local_experts
            local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]
        else:
            local_expert_indices = []

        print(f"Rank {rank}: EP_rank={ep_rank} (Group {ep_group_id}), "
              f"DP_rank={dp_rank} (Group {dp_group_id}), "
              f"PP_rank={pp_rank} (Group {pp_group_id}), "
              f"Local Experts: {local_expert_indices}")
    
    print()
    print("=== EP RANK 与 DP RANK 映射关系 ===")
    
    # 分析 EP RANK 与 DP RANK 的映射关系
    ep_to_dp_mapping = {}
    dp_to_ep_mapping = {}
    
    for rank in range(world_size):
        # 找到 EP rank
        ep_rank = None
        for i, group in enumerate(ep_groups):
            if rank in group:
                ep_rank = group.index(rank)
                break
        
        # 找到 DP rank (modulo EP)
        dp_rank = None
        for i, group in enumerate(dp_groups):
            if rank in group:
                dp_rank = group.index(rank)
                break
        
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
    
    # 计算隐式 MoE Data Parallelism
    moe_data_parallel_size = dp // ep
    print(f"隐式 MoE Data Parallelism Size = DP_SIZE / EP_SIZE = {dp} / {ep} = {moe_data_parallel_size}")
    
    # 验证：每个 EP rank 应该有 moe_data_parallel_size 个对应的 DP ranks
    for ep_rank in sorted(ep_to_dp_mapping.keys()):
        dp_ranks = [dp_rank for _, dp_rank in ep_to_dp_mapping[ep_rank]]
        print(f"EP RANK {ep_rank} 对应的 DP RANKs: {dp_ranks} (数量: {len(dp_ranks)})")

if __name__ == "__main__":
    analyze_moe_rank_mapping()
