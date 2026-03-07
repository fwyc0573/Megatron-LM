import pytest
import torch

from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tenorm_rmsnorm_casts_input_dtype():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=8,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
    )
    norm = TENorm(config=config, hidden_size=64, eps=config.layernorm_epsilon).cuda()
    hidden_states = torch.randn(4, 2, 64, device="cuda", dtype=torch.float32)
    output = norm(hidden_states)

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tenorm_rmsnorm_handles_non_contiguous_input():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=8,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
    )
    norm = TENorm(config=config, hidden_size=64, eps=config.layernorm_epsilon).cuda()
    hidden_states = torch.randn(4, 2, 64, device="cuda", dtype=torch.float32)
    non_contiguous_states = hidden_states.transpose(0, 1)
    assert not non_contiguous_states.is_contiguous()

    output = norm(non_contiguous_states)

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
