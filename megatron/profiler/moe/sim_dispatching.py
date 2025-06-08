import torch
from megatron.core.transformer.transformer_config import TransformerConfig

def _sim_gather_along_first_dim_expert_parallel(all_num_local_tokens_per_expert: list[torch.Tensor]):
    """
    Simulates the all-gather operation for num_local_tokens_per_expert across all expert parallel ranks.

    In real distributed training, each rank would call an all_gather collective
    and then reshape the output. This function simulates that by stacking the tensors
    from all ranks, which are provided as a list.

    Args:
        all_num_local_tokens_per_expert: A list of 1D tensors, where each tensor
            represents num_local_tokens_per_expert for a given EP rank.

    Returns:
        A 2D tensor of shape (ep_size, num_experts) representing the
        global token counts per expert.
    """
    # In the real implementation, _gather_along_first_dim_expert_parallel
    # performs an all-gather, resulting in a flat tensor which is then
    # likely reshaped to (ep_size, num_experts). torch.stack is a
    # direct way to achieve this in a simulation.
    num_global_tokens_per_expert = torch.stack(all_num_local_tokens_per_expert, dim=0)
    return num_global_tokens_per_expert


def sim_dispatching(config: TransformerConfig):
    ep_size = config.expert_model_parallel_size
    num_experts = config.num_moe_experts

    all_num_local_tokens_per_expert = []
    dispatching_results_per_rank = {}
    for ep_rank in range(ep_size):
        indices = config.pre_fixed_routing_results[ep_rank]['indices']

        num_local_tokens_per_expert = torch.histc(
            indices, bins=num_experts, min=0, max=num_experts
        )
        dispatching_results_per_rank[ep_rank] = {
            'num_local_tokens_per_expert': num_local_tokens_per_expert
        }
        all_num_local_tokens_per_expert.append(num_local_tokens_per_expert)

    num_global_tokens_per_expert = _sim_gather_along_first_dim_expert_parallel(
        all_num_local_tokens_per_expert
    )

    # In the real code, each rank gets the same global tensor.
    # We add it to the final results dictionary.
    dispatching_results = {
        'per_rank_results': dispatching_results_per_rank,
        'num_global_tokens_per_expert': num_global_tokens_per_expert
    }

    return dispatching_results
