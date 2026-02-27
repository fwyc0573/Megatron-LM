# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.transformer.moe.moe_utils import (
    compute_routing_scores_for_aux_loss,
    topk_routing_with_score_function,
)
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


class TestTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        self.router = self.sequential_mlp.router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward(self):
        with torch.no_grad():
            self.router = self.router.cuda()
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda()
            scores, indices = self.router(hidden_states)
            print(scores.shape, indices.shape)
            assert scores.shape == (64, 2)
            assert indices.shape == (64, 2)
            print(
                (indices == 0).sum(), (indices == 1).sum(), (indices == 2).sum(), (indices == 3).sum()
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()
        
        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0
        
        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

        # With Z loss
        self.transformer_config.moe_aux_loss_coeff = 0
        self.transformer_config.moe_z_loss_coeff = 1
        self.sequential_mlp.router.weight.grad.fill_(0)
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_router_sigmoid_group_limited():
    Utils.initialize_model_parallel(1, 1)
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        cfg = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=8,
            num_moe_experts=16,
            use_cpu_initialization=True,
            is_scaling_mode=True,
            fake_tp=1,
            moe_router_topk=4,
            moe_router_num_groups=4,
            moe_router_group_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_load_balancing_type="none",
        )
        router = TopKRouter(cfg).cuda()
        hidden_states = torch.randn(8, 2, cfg.hidden_size, device="cuda")
        scores, indices = router(hidden_states)
        assert scores.shape == (16, 4)
        assert indices.shape == (16, 4)
    finally:
        Utils.destroy_model_parallel()


def test_sigmoid_topk_zero_scores_no_nan():
    logits = torch.full((4, 8), -1.0e4, dtype=torch.bfloat16)
    scores, indices = topk_routing_with_score_function(
        logits=logits,
        topk=2,
        score_function="sigmoid",
    )
    assert scores.shape == (4, 2)
    assert indices.shape == (4, 2)
    assert torch.isfinite(scores).all()
    assert torch.all(scores == 0)


def test_sigmoid_aux_scores_zero_logits_no_nan():
    logits = torch.full((4, 8), -1.0e4, dtype=torch.bfloat16)
    routing_map, scores = compute_routing_scores_for_aux_loss(
        logits=logits,
        topk=2,
        score_function="sigmoid",
    )
    assert routing_map.shape == (4, 8)
    assert scores.shape == (4, 8)
    assert torch.isfinite(scores).all()
    assert torch.all(scores == 0)
