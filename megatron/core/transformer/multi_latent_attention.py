# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import YarnRotaryEmbedding, _yarn_get_mscale
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MLASelfAttentionSubmodules:
    linear_q_up_proj: Union[ModuleSpec, type] = ColumnParallelLinear
    linear_kv_up_proj: Union[ModuleSpec, type] = ColumnParallelLinear
    linear_proj: Union[ModuleSpec, type] = RowParallelLinear


class MLASelfAttention(MegatronModule):
    """PyTorch SDPA implementation of DeepSeek-style MLA."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        if not self.config.multi_latent_attention:
            raise ValueError("MLASelfAttention requires config.multi_latent_attention=True.")

        if self.config.q_lora_rank is None or self.config.kv_lora_rank is None:
            raise ValueError("q_lora_rank and kv_lora_rank must be set for MLA.")

        if not config.is_scaling_mode:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            tp_size = config.fake_tp
        if tp_size <= 0:
            raise ValueError(f"Invalid tensor-parallel size for MLA: {tp_size}")

        self.num_attention_heads_per_partition = config.num_attention_heads // tp_size
        self.q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim

        if config.rope_type == "yarn":
            mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
            self.softmax_scale = (mscale * mscale) / math.sqrt(self.q_head_dim)
        else:
            self.softmax_scale = 1.0 / math.sqrt(self.q_head_dim)

        # Dense low-rank compression paths.
        self.linear_q_down_proj = torch.nn.Linear(
            config.hidden_size,
            config.q_lora_rank,
            bias=config.add_bias_linear,
            dtype=config.params_dtype,
        )
        self.linear_kv_down_proj = torch.nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
            bias=config.add_bias_linear,
            dtype=config.params_dtype,
        )

        if config.perform_initialization:
            config.init_method(self.linear_q_down_proj.weight)
            config.init_method(self.linear_kv_down_proj.weight)
            if self.linear_q_down_proj.bias is not None:
                torch.nn.init.zeros_(self.linear_q_down_proj.bias)
            if self.linear_kv_down_proj.bias is not None:
                torch.nn.init.zeros_(self.linear_kv_down_proj.bias)

        self.q_layernorm = (
            torch.nn.LayerNorm(config.q_lora_rank, eps=config.layernorm_epsilon, dtype=config.params_dtype)
            if config.qk_layernorm
            else torch.nn.Identity()
        )
        self.kv_layernorm = (
            torch.nn.LayerNorm(config.kv_lora_rank, eps=config.layernorm_epsilon, dtype=config.params_dtype)
            if config.qk_layernorm
            else torch.nn.Identity()
        )

        # Tensor-parallel up projections.
        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            config.q_lora_rank,
            config.num_attention_heads * self.q_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='mla_q_up',
        )
        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_head_dim + config.v_head_dim),
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='mla_kv_up',
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='mla_proj',
        )

        if config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_percent=1.0,
                rotary_interleaved=config.rotary_interleaved,
                rotary_base=float(config.rotary_base),
                scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.original_max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                mscale_all_dim=config.mscale_all_dim,
            )
        elif config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_percent=1.0,
                rotary_interleaved=config.rotary_interleaved,
                rotary_base=config.rotary_base,
            )
        else:
            raise ValueError(f'Unsupported rope_type for MLA: {config.rope_type}')

    def _build_sdpa_mask(self, attention_mask: torch.Tensor):
        if attention_mask is None:
            return None
        # Megatron mask uses True=masked, SDPA bool mask uses True=allowed.
        if attention_mask.dtype == torch.bool:
            return ~attention_mask
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        if key_value_states is not None:
            raise NotImplementedError("MLASelfAttention does not support cross attention.")
        if inference_params is not None:
            raise NotImplementedError("MLASelfAttention does not support inference kv-cache yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("MLASelfAttention does not support packed sequence yet.")
        if rotary_pos_emb is not None:
            raise ValueError("MLASelfAttention manages RoPE internally; external rotary_pos_emb is unsupported.")

        seq_len, batch_size, _ = hidden_states.shape

        q_compressed = self.q_layernorm(self.linear_q_down_proj(hidden_states))
        q_up, q_up_bias = self.linear_q_up_proj(q_compressed)
        if q_up_bias is not None:
            q_up = q_up + q_up_bias
        q_up = q_up.view(seq_len, batch_size, self.num_attention_heads_per_partition, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q_up, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )

        kv_down = self.linear_kv_down_proj(hidden_states)
        kv_compressed, k_pe = torch.split(
            kv_down, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        kv_compressed = self.kv_layernorm(kv_compressed)
        kv_up, kv_up_bias = self.linear_kv_up_proj(kv_compressed)
        if kv_up_bias is not None:
            kv_up = kv_up + kv_up_bias
        kv_up = kv_up.view(
            seq_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )
        k_nope, value = torch.split(kv_up, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.num_attention_heads_per_partition, -1)
        rope_concentration = 1.0
        if self.config.rope_type == "yarn":
            rope_freqs, rope_concentration = self.rotary_pos_emb.get_emb(seq_len)
        else:
            rope_freqs = self.rotary_pos_emb(seq_len)

        q_pe = apply_rotary_pos_emb(q_pe, rope_freqs, config=self.config)
        k_pe = apply_rotary_pos_emb(k_pe, rope_freqs, config=self.config)
        if rope_concentration != 1.0:
            q_pe = q_pe * rope_concentration
            k_pe = k_pe * rope_concentration

        query = torch.cat((q_nope, q_pe), dim=-1).permute(1, 2, 0, 3)
        key = torch.cat((k_nope, k_pe), dim=-1).permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)

        sdpa_mask = self._build_sdpa_mask(attention_mask)
        sdpa_query = query
        sdpa_key = key
        sdpa_value = value
        original_attn_dtype = query.dtype
        if self.config.attention_softmax_in_fp32 and query.dtype != torch.float32:
            # Keep softmax numerics in fp32 for bf16/fp16 training stability.
            sdpa_query = query.float()
            sdpa_key = key.float()
            sdpa_value = value.float()

        attn_dropout = self.config.attention_dropout if self.training else 0.0
        context = F.scaled_dot_product_attention(
            sdpa_query,
            sdpa_key,
            sdpa_value,
            attn_mask=sdpa_mask,
            dropout_p=attn_dropout,
            is_causal=False,
            scale=self.softmax_scale,
        )
        if context.dtype != original_attn_dtype:
            context = context.to(dtype=original_attn_dtype)

        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_len, batch_size, -1)
        output, bias = self.linear_proj(context)
        return output, bias
