import os
import sys
from unittest import mock

import pytest

from megatron.training.arguments import parse_args, validate_args


def _base_cli():
    return [
        "test_deepseek_v3_args.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
        "--max-position-embeddings",
        "4096",
        "--seq-length",
        "128",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--train-iters",
        "1",
    ]


def _parse_and_validate(argv):
    with mock.patch.object(sys, "argv", argv):
        args = parse_args(ignore_unknown_args=True)
    with mock.patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}, clear=False):
        return validate_args(args)


def test_parse_deepseek_v3_stage2_args():
    test_argv = _base_cli() + [
        "--position-embedding-type",
        "rope",
        "--multi-latent-attention",
        "--q-lora-rank",
        "32",
        "--kv-lora-rank",
        "16",
        "--qk-head-dim",
        "64",
        "--qk-pos-emb-head-dim",
        "32",
        "--v-head-dim",
        "64",
        "--rope-type",
        "yarn",
        "--rotary-scaling-factor",
        "8",
        "--original-max-position-embeddings",
        "4096",
        "--num-experts",
        "16",
        "--moe-router-load-balancing-type",
        "seq_aux_loss",
        "--moe-router-score-function",
        "sigmoid",
        "--moe-router-num-groups",
        "4",
        "--moe-router-group-topk",
        "2",
        "--moe-router-topk",
        "4",
        "--moe-router-topk-scaling-factor",
        "2.5",
        "--moe-router-enable-expert-bias",
        "--moe-router-bias-update-rate",
        "0.001",
        "--moe-router-dtype",
        "fp32",
        "--moe-shared-expert-intermediate-size",
        "256",
        "--moe-shared-expert-gate",
    ]
    args = _parse_and_validate(test_argv)

    assert args.multi_latent_attention is True
    assert args.rope_type == "yarn"
    assert args.moe_router_load_balancing_type == "seq_aux_loss"
    assert args.moe_router_score_function == "sigmoid"
    assert args.moe_router_enable_expert_bias is True
    assert args.moe_shared_expert_intermediate_size == 256
    assert args.moe_shared_expert_gate is True


def test_group_limited_router_requires_paired_args():
    test_argv = _base_cli() + [
        "--num-experts",
        "16",
        "--moe-router-num-groups",
        "4",
    ]
    with pytest.raises(RuntimeError, match="must be set together"):
        _parse_and_validate(test_argv)


def test_mla_requires_required_dims():
    test_argv = _base_cli() + [
        "--position-embedding-type",
        "rope",
        "--multi-latent-attention",
        "--q-lora-rank",
        "32",
    ]
    with pytest.raises(RuntimeError, match="requires MLA args"):
        _parse_and_validate(test_argv)


def test_shared_expert_gate_requires_intermediate_size():
    test_argv = _base_cli() + [
        "--num-experts",
        "16",
        "--moe-shared-expert-gate",
    ]
    with pytest.raises(RuntimeError, match="requires --moe-shared-expert-intermediate-size"):
        _parse_and_validate(test_argv)
