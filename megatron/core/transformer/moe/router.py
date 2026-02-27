# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
    _normalize_scores,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.layer_number = None

        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size))
        )
        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            config.init_method(self.weight)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of router gating linear."""
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        return torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        assert config.moe_token_dropping is False
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32, device=device),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to logits."""

        def _sinkhorn_activation(scores):
            if self.topk == 1:
                return torch.sigmoid(scores)
            return torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(logits.to(dtype=torch.float32))
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            probs = _sinkhorn_activation(logits)
            scores = torch.gather(probs, 1, indices)
        else:
            probs = _sinkhorn_activation(logits)
            scores, indices = torch.topk(probs, k=self.topk, dim=1)
        return scores, indices

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss_name: str,
    ):
        """Attach auxiliary load-balancing loss to routing activation."""
        if aux_loss_coeff == 0:
            return activation
        mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)
        aux_loss = switch_load_balancing_loss_func(probs, mask, aux_loss_coeff)
        save_to_aux_losses_tracker(
            aux_loss_name,
            aux_loss / aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
        )
        return MoEAuxLossAutoScaler.apply(activation, aux_loss)

    def apply_seq_load_balancing_loss(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
        activation: torch.Tensor,
        seq_length: int,
        batch_size: int,
    ):
        """Apply sequence-level auxiliary load-balancing loss."""
        coeff = self.config.moe_aux_loss_coeff
        if coeff == 0:
            return activation
        probs = probs.view(seq_length, batch_size, self.num_experts)
        mask = (
            torch.nn.functional.one_hot(indices, num_classes=self.num_experts)
            .sum(dim=1)
            .view(seq_length, batch_size, self.num_experts)
            .float()
        )
        aux_loss = 0.0
        for batch_id in range(batch_size):
            aux_loss = aux_loss + switch_load_balancing_loss_func(
                probs[:, batch_id, :], mask[:, batch_id, :], coeff
            )
        aux_loss = aux_loss / batch_size
        save_to_aux_losses_tracker(
            'seq_load_balancing_loss',
            aux_loss / coeff,
            self.layer_number,
            self.config.num_layers,
        )
        return MoEAuxLossAutoScaler.apply(activation, aux_loss)

    def apply_z_loss(self, logits: torch.Tensor):
        """Apply z-loss to stabilize router logits."""
        if self.config.moe_z_loss_coeff is not None:
            z_loss = z_loss_func(logits, self.config.moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                'z_loss',
                z_loss / self.config.moe_z_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add multiplicative jitter to router input."""
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        return input

    def _compute_scores_for_aux_loss(self, logits: torch.Tensor):
        _, scores_for_aux = compute_routing_scores_for_aux_loss(
            logits=logits,
            topk=self.topk,
            score_function=self.score_function,
        )
        return scores_for_aux

    def _apply_expert_bias(self, routing_map: torch.Tensor):
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

    def compute_scores_from_logits_and_indices(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_length: int = None,
        batch_size: int = None,
    ):
        """Compute routing scores for fixed indices while preserving router gradients."""
        if self.score_function == 'softmax':
            top_logits = torch.gather(logits, 1, indices)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        elif self.score_function == 'sigmoid':
            sigmoid_scores = torch.sigmoid(logits.float()).type_as(logits)
            selected_scores = torch.gather(sigmoid_scores, 1, indices)
            scores = selected_scores if self.topk == 1 else _normalize_scores(selected_scores)
        else:
            raise ValueError(f'Unsupported score_function "{self.score_function}".')

        if self.config.moe_router_topk_scaling_factor is not None:
            scores = scores * self.config.moe_router_topk_scaling_factor

        if self.training and torch.is_grad_enabled():
            scores_for_aux = self._compute_scores_for_aux_loss(logits)
            if self.routing_type == 'aux_loss':
                scores = self.apply_load_balancing_loss(
                    scores_for_aux,
                    indices,
                    scores,
                    self.config.moe_aux_loss_coeff,
                    'load_balancing_loss',
                )
            elif self.routing_type == 'seq_aux_loss':
                if seq_length is None or batch_size is None:
                    raise ValueError(
                        'seq_aux_loss requires seq_length and batch_size for fixed routing.'
                    )
                scores = self.apply_seq_load_balancing_loss(
                    scores_for_aux,
                    indices,
                    scores,
                    seq_length,
                    batch_size,
                )
        routing_map = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)
        self._apply_expert_bias(routing_map)
        return scores

    def routing(self, logits: torch.Tensor):
        """Top-k routing function returning scores and indices."""
        if logits.dim() == 3:
            seq_length, batch_size = logits.shape[:2]
            logits = logits.view(-1, self.config.num_moe_experts)
        elif logits.dim() == 2:
            seq_length, batch_size = None, None
        else:
            raise ValueError(f'Unsupported logits shape: {logits.shape}')

        logits = self.apply_z_loss(logits)
        tp_size = self.config.fake_tp if self.config.is_scaling_mode else self.config.tensor_model_parallel_size
        if tp_size > 1 and self.config.moe_token_dispatcher_type == 'alltoall':
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == 'sinkhorn':
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type in ('aux_loss', 'seq_aux_loss', 'none'):
            scores, indices = topk_routing_with_score_function(
                logits=logits,
                topk=self.topk,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
            if self.training and torch.is_grad_enabled() and self.routing_type != 'none':
                scores_for_aux = self._compute_scores_for_aux_loss(logits)
                if self.routing_type == 'aux_loss':
                    scores = self.apply_load_balancing_loss(
                        scores_for_aux,
                        indices,
                        scores,
                        self.config.moe_aux_loss_coeff,
                        'load_balancing_loss',
                    )
                elif self.routing_type == 'seq_aux_loss':
                    if seq_length is None or batch_size is None:
                        raise ValueError('seq_aux_loss requires [seq, batch, expert] router logits.')
                    scores = self.apply_seq_load_balancing_loss(
                        scores_for_aux,
                        indices,
                        scores,
                        seq_length,
                        batch_size,
                    )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        routing_map = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)
        self._apply_expert_bias(routing_map)
        return scores, indices

    def forward(self, input: torch.Tensor):
        """Forward pass of router."""
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        return self.routing(logits)
