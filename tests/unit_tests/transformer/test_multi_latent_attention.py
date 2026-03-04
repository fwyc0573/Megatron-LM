import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_multi_latent_attention_backward_segment_hooks_default_disabled():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            multi_latent_attention=True,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            rope_type="rope",
            is_scaling_mode=True,
            fake_tp=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            attention_dropout=0.0,
        )
        module = MLASelfAttention(
            config=config,
            submodules=MLASelfAttentionSubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        assert len(module._attention_backward_segment_hook_handles) == 0
        assert len(module._attention_backward_segment_hook_names) == 0
    finally:
        Utils.destroy_model_parallel()


def test_multi_latent_attention_backward_segment_hooks_enabled():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            multi_latent_attention=True,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            rope_type="rope",
            is_scaling_mode=True,
            fake_tp=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            attention_dropout=0.0,
            trace_attention_backward_segments=True,
        )
        module = MLASelfAttention(
            config=config,
            submodules=MLASelfAttentionSubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        assert len(module._attention_backward_segment_hook_handles) == 28
        assert len(module._attention_backward_segment_hook_names) == 14
        assert module._attention_backward_segment_hook_names.count("attn_qkv_bwd") == 4
        assert module._attention_backward_segment_hook_names.count("attn_qk_layernorm_bwd") == 2
        assert "attn_core_bwd" in module._attention_backward_segment_hook_names
        assert "attn_core_precast_bwd" in module._attention_backward_segment_hook_names
        assert "attn_core_sdpa_bwd" in module._attention_backward_segment_hook_names
        assert "attn_core_sdpa_prefmha_bwd" in module._attention_backward_segment_hook_names
        assert "attn_core_sdpa_fmha_bwd" in module._attention_backward_segment_hook_names
        assert "attn_core_sdpa_postfmha_bwd" in module._attention_backward_segment_hook_names
        assert "attn_core_postcast_bwd" in module._attention_backward_segment_hook_names
        assert "attn_proj_bwd" in module._attention_backward_segment_hook_names
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_latent_attention_forward_shape():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=8,
            multi_latent_attention=True,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type="yarn",
            rotary_scaling_factor=8.0,
            original_max_position_embeddings=4096,
            max_position_embeddings=4096,
            is_scaling_mode=True,
            fake_tp=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            attention_dropout=0.0,
        )
        module = MLASelfAttention(
            config=config,
            submodules=MLASelfAttentionSubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        seq_len = 8
        batch = 2
        hidden_states = torch.randn(
            seq_len, batch, config.hidden_size, device="cuda", dtype=torch.float32
        )
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        output, bias = module(hidden_states, attention_mask=attention_mask)
        assert output.shape == (seq_len, batch, config.hidden_size)
        assert torch.isfinite(output).all()
        if bias is not None:
            assert bias.shape[-1] == config.hidden_size
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported on this GPU")
def test_multi_latent_attention_fp32_softmax_path(monkeypatch):
    Utils.initialize_model_parallel(1, 1)
    try:
        captured = {}
        original_sdpa = F.scaled_dot_product_attention

        def _capture_sdpa(query, key, value, *args, **kwargs):
            captured["query_dtype"] = query.dtype
            captured["key_dtype"] = key.dtype
            captured["value_dtype"] = value.dtype
            return original_sdpa(query, key, value, *args, **kwargs)

        monkeypatch.setattr(F, "scaled_dot_product_attention", _capture_sdpa)

        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=8,
            multi_latent_attention=True,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type="yarn",
            rotary_scaling_factor=8.0,
            original_max_position_embeddings=4096,
            max_position_embeddings=4096,
            is_scaling_mode=True,
            fake_tp=1,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            attention_softmax_in_fp32=True,
        )
        module = MLASelfAttention(
            config=config,
            submodules=MLASelfAttentionSubmodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        seq_len = 8
        batch = 2
        hidden_states = torch.randn(
            seq_len, batch, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        output, _ = module(hidden_states, attention_mask=attention_mask)
        assert captured["query_dtype"] == torch.float32
        assert captured["key_dtype"] == torch.float32
        assert captured["value_dtype"] == torch.float32
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
    finally:
        Utils.destroy_model_parallel()
