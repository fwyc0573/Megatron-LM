import torch
from unittest import mock

from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.transformer.transformer_config import TransformerConfig


class _DummyRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_bias = torch.zeros(4, dtype=torch.float32)
        self.local_tokens_per_expert = torch.tensor([10.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.training = True


class _DummyModelChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _DummyRouter()


def test_update_router_expert_bias_without_distributed_init():
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=8,
        use_cpu_initialization=True,
        moe_router_bias_update_rate=0.1,
        is_scaling_mode=True,
        fake_tp=1,
    )
    model = [_DummyModelChunk()]
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        _update_router_expert_bias(model, config)

    updated = model[0].router.expert_bias
    expected = torch.tensor([-0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
    assert torch.allclose(updated, expected)
    assert torch.equal(
        model[0].router.local_tokens_per_expert, torch.zeros_like(model[0].router.local_tokens_per_expert)
    )
