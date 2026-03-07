"""Tests that attention q/k layernorm sees contiguous tensors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.attention import SelfAttention


class _CaptureNorm:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(
            {
                "is_contiguous": tensor.is_contiguous(),
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
            }
        )
        return tensor


class _LinearQKVStub:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def __call__(self, hidden_states: torch.Tensor):
        return self.output, None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_self_attention_qk_layernorm_inputs_are_contiguous() -> None:
    seq_len = 2048
    batch_size = 1
    num_attention_heads_per_partition = 4
    num_query_groups_per_partition = 1
    hidden_size_per_attention_head = 64
    hidden_states = torch.randn(seq_len, batch_size, 2048, device="cuda", dtype=torch.float32)
    mixed_qkv = torch.randn(
        seq_len,
        batch_size,
        (num_attention_heads_per_partition + 2) * hidden_size_per_attention_head,
        device="cuda",
        dtype=torch.float32,
    )

    q_capture = _CaptureNorm()
    k_capture = _CaptureNorm()
    fake_self_attention = SimpleNamespace(
        linear_qkv=_LinearQKVStub(mixed_qkv),
        num_query_groups_per_partition=num_query_groups_per_partition,
        num_attention_heads_per_partition=num_attention_heads_per_partition,
        hidden_size_per_attention_head=hidden_size_per_attention_head,
        q_layernorm=q_capture,
        k_layernorm=k_capture,
        config=SimpleNamespace(test_mode=False),
    )

    SelfAttention.get_query_key_value_tensors(fake_self_attention, hidden_states)

    assert q_capture.calls == [
        {
            "is_contiguous": True,
            "shape": (seq_len, batch_size, num_attention_heads_per_partition, hidden_size_per_attention_head),
            "stride": (batch_size * num_attention_heads_per_partition * hidden_size_per_attention_head,
                       num_attention_heads_per_partition * hidden_size_per_attention_head,
                       hidden_size_per_attention_head,
                       1),
        }
    ]
    assert k_capture.calls == [
        {
            "is_contiguous": True,
            "shape": (seq_len, batch_size, num_query_groups_per_partition, hidden_size_per_attention_head),
            "stride": (batch_size * num_query_groups_per_partition * hidden_size_per_attention_head,
                       num_query_groups_per_partition * hidden_size_per_attention_head,
                       hidden_size_per_attention_head,
                       1),
        }
    ]
