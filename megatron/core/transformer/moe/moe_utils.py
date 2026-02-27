# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state


def _normalize_scores(selected_scores: torch.Tensor) -> torch.Tensor:
    """Normalize selected routing scores safely in float32 to avoid 0/0 on bf16."""
    denom = selected_scores.float().sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=torch.finfo(torch.float32).tiny)
    return (selected_scores.float() / denom).type_as(selected_scores)


def switch_load_balancing_loss_func(gates, mask, moe_aux_loss_coeff):
    """Calculate the auxiliary loss for better load balacing. 
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        gates (torch.Tensor): The gates tensor representing the routing probabilities for each expert.
        mask (torch.Tensor): The 2D mask tensor indicating which experts are selected.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_experts = mask.size(-1)
    gates_mean = gates.mean(dim=0)
    top_k = mask[0].count_nonzero()
    selection_mean = mask.float().mean(dim=0) / top_k
    aux_loss = torch.sum(gates_mean * selection_mean) * num_experts
    aux_loss *= moe_aux_loss_coeff
    return aux_loss


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.
    
    Args:
        logits (torch.Tensor): The logits of the router.
    
    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
    """Sinkhorn based MoE routing function"""
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    """Select top-k experts from a limited subset of groups for each token."""
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    return torch.topk(masked_scores, k=topk, dim=-1)


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    num_groups: int = None,
    group_topk: int = None,
    scaling_factor: float = None,
    score_function: str = "softmax",
    expert_bias: torch.Tensor = None,
):
    """Compute top-k routing probs and indices for softmax/sigmoid routing."""
    if logits.dim() != 2:
        raise ValueError(f"Expected logits to be 2D [num_tokens, num_experts], got {logits.shape}")
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores: torch.Tensor):
        if group_topk is not None:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        return torch.topk(scores, k=topk, dim=-1)

    if score_function == "softmax":
        top_scores, top_indices = _compute_topk(logits)
        probs = torch.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            _, top_indices = _compute_topk(scores + expert_bias)
            top_scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            top_scores, top_indices = _compute_topk(scores)
        probs = top_scores if topk == 1 else _normalize_scores(top_scores)
    else:
        raise ValueError(f'Unsupported score_function "{score_function}".')

    if scaling_factor is not None:
        probs = probs * scaling_factor

    return probs, top_indices


def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor, topk: int, score_function: str
):
    """Build routing_map and normalized full-expert scores used by aux losses."""
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = _normalize_scores(scores).float()
    else:
        raise ValueError(f'Unsupported score_function "{score_function}".')
    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


def get_updated_expert_bias(
    tokens_per_expert: torch.Tensor,
    expert_bias: torch.Tensor,
    expert_bias_update_rate: float,
    is_scaling_mode: bool = False,
):
    """Update expert bias according to token imbalance among experts."""
    with torch.no_grad():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if is_scaling_mode:
                reduce_group = None
            else:
                reduce_group = parallel_state.get_tensor_and_data_parallel_group(
                    with_context_parallel=True
                )
            torch.distributed.all_reduce(tokens_per_expert, group=reduce_group)
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        return expert_bias + torch.sign(offset) * expert_bias_update_rate


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss.

    """

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.
        
        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.
        
        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def permute(tokens, indices, topk: int = 1):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens, topk].
        topk (int, optional): The topk value. Defaults to 1.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    if topk > 1:
        assert indices.size(1) == topk
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, topk: int = 1):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
        topk (int, optional): The number of top tokens to consider for merging with probabilities. Defaults to 1.
    """
    if topk > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // topk
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // topk
        assert probs.size(1) == topk, f"probs size {probs.size()} merge_factor {topk}"

    unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def save_to_aux_losses_tracker(name: str, loss: torch.Tensor, layer_number: int, num_layers: int):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    if name not in parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER:
        parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER[name] = torch.zeros(
            num_layers, device=loss.device
        )
    parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER[name][layer_number - 1] += loss.detach()


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    for name in parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER:
        parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER[name].zero_()


def get_aux_losses_tracker():
    """Return the auxiliary losses."""
    return parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER


def aggregate_aux_losses_tracker_across_pipeline_parallel():
    """Sum aux losses across PP."""
    for name in parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER:
        loss = parallel_state._MOE_AUX_LOSSES_LOGGING_TRACKER[name]
        torch.distributed.all_reduce(loss, group=parallel_state.get_pipeline_model_parallel_group())


def track_moe_metrics(
    loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None, per_layer_logging=False
):
    # Aux loss logging
    aggregate_aux_losses_tracker_across_pipeline_parallel()
    if writer is not None:
        aux_losses = {k: v.float() * loss_scale for k, v in get_aux_losses_tracker().items()}
        for name, loss_list in aux_losses.items():
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.mean()
                else:
                    total_loss_dict[name] += loss_list.mean()

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            writer.add_scalar(name, loss_list.mean(), iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                wandb_writer.log({f"{name}": loss_list.mean()}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {
                            f"moe/{name}_layer_{i}": loss
                            for i, loss in enumerate(loss_list.tolist())
                        },
                        iteration,
                    )

    clear_aux_losses_tracker()
