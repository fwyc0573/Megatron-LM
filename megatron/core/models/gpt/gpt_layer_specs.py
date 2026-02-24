# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def _get_moe_layer_pattern(config: TransformerConfig):
    """Build dense/MoE layer pattern from config.moe_layer_freq."""
    if config.num_moe_experts is None:
        return [0 for _ in range(config.num_layers)]

    if isinstance(config.moe_layer_freq, int):
        if config.moe_layer_freq <= 0:
            raise ValueError(f"moe_layer_freq must be positive, got {config.moe_layer_freq}")
        return [1 if (layer_id % config.moe_layer_freq == 0) else 0 for layer_id in range(config.num_layers)]

    if isinstance(config.moe_layer_freq, list):
        if len(config.moe_layer_freq) != config.num_layers:
            raise ValueError(
                f"Invalid moe_layer_freq length {len(config.moe_layer_freq)}; expected {config.num_layers}"
            )
        invalid_items = [item for item in config.moe_layer_freq if item not in (0, 1)]
        if invalid_items:
            raise ValueError(
                f"moe_layer_freq list only supports 0/1 entries, got {invalid_items}"
            )
        return config.moe_layer_freq

    raise ValueError(
        f"Invalid moe_layer_freq type {type(config.moe_layer_freq)}, value {config.moe_layer_freq}"
    )


def get_gpt_decoder_layer_specs(
    config: TransformerConfig, use_transformer_engine: bool
):
    """Build full decoder layer specs with dense/MoE mix."""
    if use_transformer_engine:
        dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
        )
        moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
        )
    else:
        dense_layer_spec = get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
        )
        moe_layer_spec = get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
        )

    layer_pattern = _get_moe_layer_pattern(config)
    layer_specs = []
    for layer_type in layer_pattern:
        if layer_type == 1:
            layer_specs.append(moe_layer_spec)
        elif layer_type == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer type in moe_layer_freq pattern: {layer_type}")
    return layer_specs


def get_gpt_decoder_block_spec(
    config: TransformerConfig, use_transformer_engine: bool
) -> TransformerBlockSubmodules:
    """Build local block spec for current pipeline rank."""
    layer_specs = get_gpt_decoder_layer_specs(config, use_transformer_engine)
    num_layers_to_build = get_num_layers_to_build(config)
    if config.is_scaling_mode:
        pp_rank = config.pp_rank
    else:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    offset = pp_rank * num_layers_to_build
    local_layer_specs = layer_specs[offset : offset + num_layers_to_build]
    if len(local_layer_specs) != num_layers_to_build:
        raise ValueError(
            f"Invalid local layer specs slice: pp_rank={pp_rank}, offset={offset}, "
            f"num_layers_to_build={num_layers_to_build}, total_layers={len(layer_specs)}"
        )
    return TransformerBlockSubmodules(layer_specs=local_layer_specs)


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )
