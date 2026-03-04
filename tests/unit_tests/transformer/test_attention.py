# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

class TestParallelAttention:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        self.parallel_attention = SelfAttention(self.transformer_config,
                                                get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
                                                layer_number=1)


    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.parallel_attention, SelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])
        assert num_weights == 648

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        output, bias = self.parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_fused_rope_gpu_forward(self):
        self.parallel_attention.config.apply_rope_fusion = True
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        rotary_pos_emb = torch.ones(sequence_length, 1, 1, self.parallel_attention.config.kv_channels).cuda()
        output, bias = self.parallel_attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False


    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity='selective'
        checkpointed_parallel_attention = SelfAttention(transformer_config,
                                                        get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
                                                        layer_number=1)
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_attention_backward_segment_hooks_disabled_by_default(self):
        assert (
            len(self.parallel_attention._attention_backward_segment_hook_handles) == 0
        )
        assert len(self.parallel_attention._attention_backward_segment_hook_names) == 0

    def test_attention_backward_segment_hooks_enabled(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            trace_attention_backward_segments=True,
        )
        local_submodules = (
            get_gpt_layer_local_spec()
            .submodules.self_attention.submodules
        )
        attention = SelfAttention(
            config,
            local_submodules,
            layer_number=1,
        )
        assert len(attention._attention_backward_segment_hook_handles) == 10
        assert len(attention._attention_backward_segment_hook_names) == 5
        assert "attn_qkv_bwd" in attention._attention_backward_segment_hook_names
        assert "attn_core_bwd" in attention._attention_backward_segment_hook_names
        assert "attn_proj_bwd" in attention._attention_backward_segment_hook_names
        assert (
            attention._attention_backward_segment_hook_names.count("attn_qk_layernorm_bwd")
            == 2
        )
