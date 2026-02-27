import torch
import pytest

from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_experts_forward_scaling_mode():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_shared_expert_intermediate_size=64,
            is_scaling_mode=True,
            fake_tp=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        module = SharedExpertMLP(config).cuda()
        hidden_states = torch.randn(4, 2, config.hidden_size, device="cuda", dtype=torch.float32)
        output = module(hidden_states)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_experts_gate_forward_scaling_mode():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_shared_expert_intermediate_size=64,
            moe_shared_expert_gate=True,
            is_scaling_mode=True,
            fake_tp=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        module = SharedExpertMLP(config).cuda()
        hidden_states = torch.randn(4, 2, config.hidden_size, device="cuda", dtype=torch.float32)
        output = module(hidden_states)
        assert module.gate_weight is not None
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
    finally:
        Utils.destroy_model_parallel()
