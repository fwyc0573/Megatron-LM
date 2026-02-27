# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import copy

import torch

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


class SharedExpertMLP(torch.nn.Module):
    """Shared expert MLP applied to all tokens and added to routed expert output."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.moe_shared_expert_intermediate_size is None:
            raise ValueError("moe_shared_expert_intermediate_size must be set for SharedExpertMLP.")

        shared_config = copy(config)
        shared_config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        submodules = MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
        self.mlp = MLP(config=shared_config, submodules=submodules, is_expert=False)
        self.use_shared_expert_gate = config.moe_shared_expert_gate
        if self.use_shared_expert_gate:
            self.gate_weight = torch.nn.Parameter(
                torch.empty((1, config.hidden_size), dtype=config.params_dtype)
            )
            if config.perform_initialization:
                config.init_method(self.gate_weight)
            setattr(self.gate_weight, 'sequence_parallel', config.sequence_parallel)
        else:
            self.gate_weight = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, bias = self.mlp(hidden_states)
        if bias is not None:
            output = output + bias
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output
