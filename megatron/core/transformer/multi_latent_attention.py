# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn.functional as F

from megatron.profiler.cmd import CMD
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


class _MLASDPAMaskBuilder(torch.nn.Module):
    def forward(self, attention_mask: torch.Tensor):
        if attention_mask is None:
            return None
        # Megatron mask uses True=masked, SDPA bool mask uses True=allowed.
        if attention_mask.dtype == torch.bool:
            return ~attention_mask
        return attention_mask


class _MLASDPAPreCast(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.attention_softmax_in_fp32 and query.dtype != torch.float32:
            # Keep softmax numerics in fp32 for bf16/fp16 training stability.
            return query.float(), key.float(), value.float()
        return query, key, value


class _MLASDPABackend(torch.nn.Module):
    class _PreFMHARuntimeContext(torch.nn.Module):
        def forward(
            self,
            sdpa_query: torch.Tensor,
            sdpa_key: torch.Tensor,
            sdpa_value: torch.Tensor,
            sdpa_mask: torch.Tensor,
            dropout_p: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
            return sdpa_query, sdpa_key, sdpa_value, sdpa_mask, dropout_p

    class _FMHACall(torch.nn.Module):
        def __init__(self, softmax_scale: float):
            super().__init__()
            self.softmax_scale = softmax_scale

        def forward(
            self,
            sdpa_query: torch.Tensor,
            sdpa_key: torch.Tensor,
            sdpa_value: torch.Tensor,
            sdpa_mask: torch.Tensor,
            dropout_p: float,
        ) -> torch.Tensor:
            return F.scaled_dot_product_attention(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.softmax_scale,
            )

    class _PostFMHARuntimeContext(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context

    def __init__(self, softmax_scale: float, trace_attention_backward_segments: bool):
        super().__init__()
        self.softmax_scale = softmax_scale
        # Keep fine-grained SDPA runtime-context segmentation debug-only.
        self.trace_attention_backward_segments = trace_attention_backward_segments
        self.pre_fmha_context = self._PreFMHARuntimeContext()
        self.fmha_call = self._FMHACall(softmax_scale=softmax_scale)
        self.post_fmha_context = self._PostFMHARuntimeContext()

    def forward(
        self,
        sdpa_query: torch.Tensor,
        sdpa_key: torch.Tensor,
        sdpa_value: torch.Tensor,
        sdpa_mask: torch.Tensor,
        dropout_p: float,
    ) -> torch.Tensor:
        if not self.trace_attention_backward_segments:
            return F.scaled_dot_product_attention(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self.softmax_scale,
            )
        sdpa_query, sdpa_key, sdpa_value, sdpa_mask, dropout_p = self.pre_fmha_context(
            sdpa_query,
            sdpa_key,
            sdpa_value,
            sdpa_mask,
            dropout_p,
        )
        context = self.fmha_call(
            sdpa_query,
            sdpa_key,
            sdpa_value,
            sdpa_mask,
            dropout_p,
        )
        return self.post_fmha_context(context)


class _MLASDPAPostCast(torch.nn.Module):
    def forward(self, context: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        if context.dtype != output_dtype:
            return context.to(dtype=output_dtype)
        return context


class _MLASDPACoreAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig, softmax_scale: float):
        super().__init__()
        self.config = config
        self.mask_builder = _MLASDPAMaskBuilder()
        self.pre_cast = _MLASDPAPreCast(config=config)
        self.sdpa_backend = _MLASDPABackend(
            softmax_scale=softmax_scale,
            trace_attention_backward_segments=getattr(
                config, "trace_attention_backward_segments", False
            ),
        )
        self.post_cast = _MLASDPAPostCast()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sdpa_mask = self.mask_builder(attention_mask)
        original_attn_dtype = query.dtype
        sdpa_query, sdpa_key, sdpa_value = self.pre_cast(query, key, value)
        attn_dropout = self.config.attention_dropout if self.training else 0.0
        context = self.sdpa_backend(
            sdpa_query,
            sdpa_key,
            sdpa_value,
            sdpa_mask,
            attn_dropout,
        )
        return self.post_cast(context, original_attn_dtype)


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
        self.core_attention = _MLASDPACoreAttention(
            config=self.config,
            softmax_scale=self.softmax_scale,
        )

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
        self._attention_backward_segment_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._attention_backward_segment_hook_names: List[str] = []
        self._register_attention_backward_segment_hook(self.linear_q_down_proj, "attn_qkv_bwd")
        self._register_attention_backward_segment_hook(self.linear_kv_down_proj, "attn_qkv_bwd")
        self._register_attention_backward_segment_hook(self.q_layernorm, "attn_qk_layernorm_bwd")
        self._register_attention_backward_segment_hook(self.kv_layernorm, "attn_qk_layernorm_bwd")
        self._register_attention_backward_segment_hook(self.linear_q_up_proj, "attn_qkv_bwd")
        self._register_attention_backward_segment_hook(self.linear_kv_up_proj, "attn_qkv_bwd")
        self._register_attention_backward_segment_hook(self.core_attention, "attn_core_bwd")
        self._register_attention_backward_segment_hook(
            self.core_attention.pre_cast, "attn_core_precast_bwd"
        )
        self._register_attention_backward_segment_hook(
            self.core_attention.sdpa_backend, "attn_core_sdpa_bwd"
        )
        self._register_attention_backward_segment_hook(
            self.core_attention.sdpa_backend.pre_fmha_context,
            "attn_core_sdpa_prefmha_bwd",
        )
        self._register_attention_backward_segment_hook(
            self.core_attention.sdpa_backend.fmha_call,
            "attn_core_sdpa_fmha_bwd",
        )
        self._register_attention_backward_segment_hook(
            self.core_attention.sdpa_backend.post_fmha_context,
            "attn_core_sdpa_postfmha_bwd",
        )
        self._register_attention_backward_segment_hook(
            self.core_attention.post_cast, "attn_core_postcast_bwd"
        )
        self._register_attention_backward_segment_hook(self.linear_proj, "attn_proj_bwd")

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

    def _register_attention_backward_segment_hook(self, module: torch.nn.Module, segment_name: str):
        if not getattr(self.config, "trace_attention_backward_segments", False):
            return
        if module is None:
            return

        stack_attr = "_cmd_attn_segment_cmd_stack"

        def _pre_hook(hook_module, grad_output):
            current_cmd = CMD.get_current_cmd()
            if current_cmd is None:
                return
            pushed = current_cmd._push_kernel_phase_nvtx(
                "compute",
                extra_tags={"attn_bwd_segment": segment_name},
            )
            if not pushed:
                return
            cmd_stack = getattr(hook_module, stack_attr, None)
            if cmd_stack is None:
                cmd_stack = []
                setattr(hook_module, stack_attr, cmd_stack)
            cmd_stack.append(current_cmd)

        def _post_hook(hook_module, grad_input, grad_output):
            cmd_stack = getattr(hook_module, stack_attr, None)
            if not cmd_stack:
                return
            cmd = cmd_stack.pop()
            cmd._pop_kernel_phase_nvtx()

        self._attention_backward_segment_hook_handles.append(
            module.register_full_backward_pre_hook(_pre_hook)
        )
        self._attention_backward_segment_hook_handles.append(
            module.register_full_backward_hook(_post_hook)
        )
        self._attention_backward_segment_hook_names.append(segment_name)

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

        context = self.core_attention(
            query,
            key,
            value,
            attention_mask,
        )

        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(seq_len, batch_size, -1)
        output, bias = self.linear_proj(context)
        return output, bias
