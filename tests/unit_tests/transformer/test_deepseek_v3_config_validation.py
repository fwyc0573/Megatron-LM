import pytest

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.transformer_config import TransformerConfig


def _build_config(**overrides):
    kwargs = {
        "num_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "use_cpu_initialization": True,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_valid_deepseek_v3_config_path():
    cfg = _build_config(
        multi_latent_attention=True,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type="yarn",
        rotary_scaling_factor=8.0,
        original_max_position_embeddings=4096,
        num_moe_experts=16,
        moe_router_topk=4,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_score_function="sigmoid",
        moe_router_num_groups=4,
        moe_router_group_topk=2,
        moe_shared_expert_intermediate_size=128,
    )
    assert cfg.multi_latent_attention is True
    assert cfg.rope_type == "yarn"
    assert cfg.moe_router_score_function == "sigmoid"


def test_mla_missing_fields_fail_fast():
    with pytest.raises(ValueError, match="requires MLA fields"):
        _build_config(
            multi_latent_attention=True,
            q_lora_rank=32,
            kv_lora_rank=16,
        )


def test_group_limited_topk_validation():
    with pytest.raises(ValueError, match="divisible by moe_router_group_topk"):
        _build_config(
            num_moe_experts=16,
            moe_router_topk=3,
            moe_router_num_groups=4,
            moe_router_group_topk=2,
        )


def test_shared_expert_gate_requires_shared_size():
    with pytest.raises(ValueError, match="requires moe_shared_expert_intermediate_size"):
        _build_config(
            num_moe_experts=16,
            moe_shared_expert_gate=True,
        )


def test_local_gpt_spec_uses_te_norm_for_rmsnorm():
    cfg = _build_config(
        normalization="RMSNorm",
        multi_latent_attention=True,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
    )
    layer_specs = get_gpt_decoder_layer_specs(cfg, use_transformer_engine=False)
    assert layer_specs[0].submodules.input_layernorm is TENorm
    assert layer_specs[0].submodules.pre_mlp_layernorm is TENorm
