import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.training.global_vars import set_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args
from megatron.training.training import build_train_valid_test_data_iterators
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return 1, 2, 3


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False

    return args


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )

        assert (train_iter, valid_iter, test_iter) == (1, 2, 3)

    def test_parse_new_moe_cli_args(self):
        test_argv = [
            "test_training.py",
            "--num-layers", "14",
            "--hidden-size", "1024",
            "--num-attention-heads", "16",
            "--num-experts", "32",
            "--moe-layer-freq", "([0]*3+[1]*11)",
            "--moe-ffn-hidden-size", "512",
            "--rotary-base", "1000000",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            args = parse_args(ignore_unknown_args=True)

        assert args.moe_layer_freq == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert args.moe_ffn_hidden_size == 512
        assert args.rotary_base == 1000000

    def test_trace_subop_sync_mode_default_global(self):
        test_argv = [
            "test_training.py",
            "--num-layers", "2",
            "--hidden-size", "128",
            "--num-attention-heads", "8",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            args = parse_args(ignore_unknown_args=True)
        assert args.trace_subop_sync_mode == "global"

    def test_trace_subop_sync_mode_event(self):
        test_argv = [
            "test_training.py",
            "--num-layers", "2",
            "--hidden-size", "128",
            "--num-attention-heads", "8",
            "--trace-subop-sync-mode", "event",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            args = parse_args(ignore_unknown_args=True)
        assert args.trace_subop_sync_mode == "event"

    def test_trace_subop_sync_mode_invalid_value(self):
        test_argv = [
            "test_training.py",
            "--num-layers", "2",
            "--hidden-size", "128",
            "--num-attention-heads", "8",
            "--trace-subop-sync-mode", "bad-value",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            with pytest.raises(SystemExit):
                parse_args(ignore_unknown_args=False)

    def test_trace_kernel_ground_truth_defaults(self):
        test_argv = [
            "test_training.py",
            "--num-layers", "2",
            "--hidden-size", "128",
            "--num-attention-heads", "8",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            args = parse_args(ignore_unknown_args=True)
        assert args.trace_kernel_ground_truth is False
        assert args.trace_kernel_ground_truth_prefix == "cmd_trace"

    def test_trace_kernel_ground_truth_args(self):
        test_argv = [
            "test_training.py",
            "--num-layers", "2",
            "--hidden-size", "128",
            "--num-attention-heads", "8",
            "--trace-kernel-ground-truth",
            "--trace-kernel-ground-truth-prefix", "cmd_gt",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            args = parse_args(ignore_unknown_args=True)
        assert args.trace_kernel_ground_truth is True
        assert args.trace_kernel_ground_truth_prefix == "cmd_gt"

    def test_core_transformer_config_injects_new_fields(self):
        args = SimpleNamespace(
            # Dataclass fields used in this test.
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=8,
            ffn_hidden_size=512,
            num_moe_experts=16,
            moe_layer_freq=[0, 1, 0, 1],
            moe_ffn_hidden_size=192,
            rotary_base=1000000,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            normalization="LayerNorm",
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            sequence_parallel=False,
            params_dtype=torch.float32,
            fp16=False,
            bf16=False,
            # Extra arguments consumed by core_transformer_config_from_args.
            no_persist_layer_norm=False,
            apply_layernorm_1p=False,
            norm_epsilon=1e-5,
            overlap_p2p_comm=False,
            num_experts=16,
            rotary_interleaved=False,
            swiglu=False,
            bias_swiglu_fusion=False,
            bias_gelu_fusion=False,
            squared_relu=False,
            init_method_xavier_uniform=False,
            group_query_attention=False,
        )
        config = core_transformer_config_from_args(args)
        assert config.moe_layer_freq == [0, 1, 0, 1]
        assert config.moe_ffn_hidden_size == 192
        assert config.rotary_base == 1000000

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
