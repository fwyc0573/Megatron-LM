# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config

        # if config.is_scaling_mode:
        #     self.expert_parallel_size = config.expert_model_parallel_size
        # else:
        #     self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        self.expert_parallel_size = config.expert_model_parallel_size
        if not config.is_scaling_mode:
            assert self.expert_parallel_size == parallel_state.get_expert_model_parallel_world_size(), \
                f"expert_parallel_size {self.expert_parallel_size} is not equal to expert_model_parallel_world_size {parallel_state.get_expert_model_parallel_world_size()}"

        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size

        if config.is_scaling_mode:
            exp_rank = config.exp_rank
        else:
            exp_rank = parallel_state.get_expert_model_parallel_rank()

        local_expert_indices_offset = (
            exp_rank * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        
        # 添加详细的并行信息日志
        if not config.is_scaling_mode:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # 获取各种并行维度的rank和size
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            
            dp_rank = parallel_state.get_data_parallel_rank()
            dp_size = parallel_state.get_data_parallel_world_size()
            
            # exp_rank = parallel_state.get_expert_model_parallel_rank()
            exp_size = self.expert_parallel_size
        else:
            global_rank = config.current_fake_rank_id
            world_size = config.fake_world_size

            tp_rank = config.tp_rank
            tp_size = config.fake_tp

            pp_rank = config.pp_rank
            pp_size = config.fake_pp

            dp_rank = config.dp_rank
            dp_size = config.fake_dp

            # exp_rank = config.exp_rank
            exp_size = config.fake_exp

        # 获取当前层的编号
        layer_str = f"Layer {layer_number}" if layer_number is not None else "Layer Unknown"
        
        # 计算每个EP rank负责的专家范围
        experts_start = local_expert_indices_offset
        experts_end = experts_start + self.num_local_experts - 1
        
        # 打印详细的并行配置和专家分配信息
        print(f"[{layer_str}] Rank {global_rank}/{world_size} (TP={tp_rank}/{tp_size}, "
              f"PP={pp_rank}/{pp_size}, DP={dp_rank}/{dp_size}, EP={exp_rank}/{exp_size}): "
              f"Responsible for {self.num_local_experts} local experts [ID {experts_start}-{experts_end}] "
              f"out of {self.config.num_moe_experts} total experts")
        
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            if config.is_scaling_mode:
                raise NotImplementedError("SequentialMLP is not supported in scaling mode")
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

    def forward(self, hidden_states: torch.Tensor):
        # process MoE
        
        if self.config.is_scaling_mode:
            exp_rank = self.config.exp_rank
        else:
            exp_rank = parallel_state.get_expert_model_parallel_rank()

        # hidden_states shape = [seq_len, micro_batch_size, hidden_size]
        # TODO-YC:
        # hidden_states = self.config.hidden_states[exp_rank]

        scores, indices = self.router(hidden_states)

        # print(f"[DEBUG] Before fixed routing - scores shape: {scores.shape}, dtype: {scores.dtype}")
        # print(f"[DEBUG] Before fixed routing - indices shape: {indices.shape}, dtype: {indices.dtype}")
        # print(f"[DEBUG] Before fixed routing - hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

        #################### replaced by fixed routing results ##############
        if self.config.pre_fixed_routing_results:

            scores = self.config.pre_fixed_routing_results[exp_rank]['scores']
            indices = self.config.pre_fixed_routing_results[exp_rank]['indices']
            hidden_states = self.config.pre_fixed_routing_results[exp_rank]['hidden_states']
            
            # Move tensors to GPU if they're not already there
            if not scores.is_cuda:
                scores = scores.cuda()
            if not indices.is_cuda:
                indices = indices.cuda()
            if not hidden_states.is_cuda:
                hidden_states = hidden_states.cuda()

            # print(f"[DEBUG] After fixed routing - scores shape: {scores.shape}, dtype: {scores.dtype}")
            # print(f"[DEBUG] After fixed routing - indices shape: {indices.shape}, dtype: {indices.dtype}")
            # print(f"[DEBUG] After fixed routing - hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")


        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, scores, indices
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        return output, mlp_bias
