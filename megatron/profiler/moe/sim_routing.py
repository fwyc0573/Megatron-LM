# Copyright (c) 2024, Yicheng Feng. All rights reserved.

from typing import Callable, Optional, Tuple
import torch

from megatron.profiler.moe.router_wrapper import TopKRouterWrapper
from megatron.core.transformer.transformer_config import TransformerConfig

# --- Example Standalone Custom Routing Functions ---
# These functions demonstrate the API for custom load balancing.
# They can be passed to the TopKRouterWrapper during its initialization.

def naive_topk_routing(logits: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A naive top-k routing without any load balancing.
    This is a standalone function that can be used as a custom routing API.
    """
    top_logits, indices = torch.topk(logits, k=topk, dim=1)
    scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
    return scores, indices

def fixed_round_robin_routing(
    logits: torch.Tensor, topk: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A fixed round-robin assignment for profiling purposes.
    It ignores the logits and assigns tokens to experts sequentially.
    This is a standalone function that can be used as a custom routing API.
    """
    num_tokens = logits.size(0)

    # Round-robin assignment for the first expert choice.
    base_indices = torch.arange(num_tokens, device=logits.device) % num_experts
    
    # Create assignments for top-k experts.
    indices_list = []
    for k_i in range(topk):
        indices_list.append(((base_indices + k_i) % num_experts).unsqueeze(1))
    indices = torch.cat(indices_list, dim=1)

    # Assign uniform scores to the selected experts.
    scores = torch.full_like(indices, 1.0 / topk, dtype=logits.dtype)
    return scores, indices


def sim_routing(config: TransformerConfig, hidden_states_shape: Tuple[int, int, int]):
    """
    Simulate the routing process.
    """
    from megatron.training import get_args
    args = get_args()
    torch.manual_seed(args.seed)

    router_wrapper = TopKRouterWrapper(config=config)
    router_wrapper.to(device='cuda', dtype=config.params_dtype)
    # Dictionary to store routing results for each EP rank
    routing_results = {}

    # for each rank, we randomly init the hidden_states to simulate the routing process to get scores and indices
    with torch.no_grad():
        for ep_rank in range(config.expert_model_parallel_size):
            torch.manual_seed(args.seed + ep_rank) 
            
            hidden_states_shape = tuple(map(int, hidden_states_shape))
            hidden_states = torch.rand(hidden_states_shape, dtype=config.params_dtype, device='cuda')
            # TODO-YC: add customized load-balance func API cotrol
            scores, indices = router_wrapper(hidden_states)
            routing_results[ep_rank] = {
                'hidden_states': hidden_states.cpu(),
                'scores': scores.cpu(),
                'indices': indices.cpu()
            }
        
    return routing_results



