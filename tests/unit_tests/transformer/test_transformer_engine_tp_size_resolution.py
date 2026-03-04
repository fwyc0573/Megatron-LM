import pytest
import torch

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

te = pytest.importorskip("transformer_engine")
from megatron.core.transformer.custom_layers.transformer_engine import (  # noqa: E402
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
)


def _build_config(tp_size: int, fake_tp: int, is_scaling_mode: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=32,
        num_query_groups=32,
        kv_channels=8,
        ffn_hidden_size=1024,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        is_scaling_mode=is_scaling_mode,
        fake_tp=fake_tp,
    )


def _build_fc1_layer(config: TransformerConfig):
    return TELayerNormColumnParallelLinear(
        input_size=256,
        output_size=1024,
        config=config,
        init_method=lambda w: None,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        skip_weight_param_allocation=False,
        tp_comm_buffer_name=None,
    )


def test_scaling_mode_without_tp_override_keeps_full_weight_partition() -> None:
    config = _build_config(tp_size=1, fake_tp=16, is_scaling_mode=True)
    layer = _build_fc1_layer(config)

    assert tuple(layer.weight.shape) == (1024, 256)


def test_scaling_mode_with_tp_override_partitions_weight_by_tp_size() -> None:
    # This mirrors pretrain_llama.py scaling-mode behavior:
    # config.tensor_model_parallel_size = args.fake_tp
    config = _build_config(tp_size=16, fake_tp=16, is_scaling_mode=True)
    layer = _build_fc1_layer(config)

    assert tuple(layer.weight.shape) == (64, 256)


def test_scaling_mode_attention_uses_overridden_tp_size() -> None:
    config = _build_config(tp_size=16, fake_tp=16, is_scaling_mode=True)
    attn = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
    )

    assert getattr(attn, "tp_size", None) == 16
