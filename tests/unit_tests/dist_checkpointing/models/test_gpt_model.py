# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
from torch.distributed._tensor import DeviceMesh

from megatron.core.dist_checkpointing import save, load, load_plain_tensors
from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import gpt_layer_with_transformer_engine_spec


def initialize_gpt_model(**config_kwargs):
    default_config_kwargs=dict(num_layers=8, hidden_size=16, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    model_parallel_cuda_manual_seed(123)
    transformer_config = TransformerConfig(**default_config_kwargs)
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()
    model = GPTModel(config=transformer_config, transformer_layer_spec=gpt_layer_with_transformer_engine_spec, vocab_size=128, max_sequence_length=4,
                     pre_process=pre_process, post_process=post_process)

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


class TestGPTModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(2,4)
        self.gpt_model = initialize_gpt_model()


    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _save_sharded_state_dict(self, ckpt_dir, strategy=None):
        sharded_state_dict = self.gpt_model.sharded_state_dict()
        save(sharded_state_dict, ckpt_dir, strategy)

    def _load_sharded_state_dict(self, ckpt_dir):
        sharded_state_dict = self.gpt_model.sharded_state_dict()
        state_dict = load(sharded_state_dict, ckpt_dir)
        self.gpt_model.load_state_dict(state_dict)

    def test_sharded_state_dict_save_load(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model') as ckpt_dir:
            self._save_sharded_state_dict(ckpt_dir)
            self._load_sharded_state_dict(ckpt_dir)


class TestGPTModelReconfiguration:
    @pytest.mark.parametrize("src_tp_pp,dest_tp_pp", [
        ((2, 4), (4, 2)),
        ((1, 8), (8, 1)),
        ((2, 1), (1, 8)),
        ((1, 1), (2, 2)),
    ])
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        """ Test model saving and loading with different TP/PP """
        with (TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_A') as ckpt_dir_A,
              TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_B') as ckpt_dir_B):
            # Save checkpoint A
            Utils.initialize_model_parallel(*src_tp_pp)
            gpt_model_A = initialize_gpt_model()
            save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            gpt_model_B = initialize_gpt_model()
            state_dict = load(gpt_model_B.sharded_state_dict(), ckpt_dir_A)
            gpt_model_B.load_state_dict(state_dict)
            save(gpt_model_B.sharded_state_dict(), ckpt_dir_B)
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(state_dict_A, state_dict_B)
            assert not any(map(bool, diffs)), diffs

    def test_state_dict_comparison(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)
        with (TempNamedDir(tmp_path_dist_ckpt / 'test_state_dict_comparison_A') as ckpt_dir_A,
              TempNamedDir(tmp_path_dist_ckpt / 'test_state_dict_comparison_B') as ckpt_dir_B):
            gpt_model_A = initialize_gpt_model()
            save(gpt_model_A.sharded_state_dict(), ckpt_dir_A)
            gpt_model_B = initialize_gpt_model()
            save(gpt_model_B.sharded_state_dict(), ckpt_dir_B)

            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_A_dup = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)

            # Test that A matches A
            diffs = diff(state_dict_A, state_dict_A_dup)
            assert not any(map(bool, diffs)), diffs

            # Test that A *keys* match B *keys*, but the tensors content is different
            only_left, only_right, mismatch = diff(state_dict_A, state_dict_B)
            assert (not only_left and not only_right), (only_left, only_right)
            assert len(mismatch) == len(state_dict_A), (len(mismatch), (len(state_dict_A)))
