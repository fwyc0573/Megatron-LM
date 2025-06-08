# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from typing import Union, Dict
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

from megatron.profiler.cmd import CMD, current_cmd_var
import torch.cuda.nvtx as nvtx

stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    print(f"args.is_scaling_mode:{args.is_scaling_mode}")
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    def add_extra_config_kwargs():
        # fake args for profile
        config.fake_pp = args.fake_pp
        config.fake_dp = args.fake_dp
        config.fake_tp = args.fake_tp
        config.fake_exp = args.fake_exp
        config.fake_world_size = args.fake_world_size
        config.fake_num_experts = args.fake_num_experts

        # TODO-YC: all fake args? what about in real runing mode? I think they should be same, because ranks in both 2 mode are presented the same meaning.
        config.is_pre_process = args.is_pre_process
        config.is_post_process = args.is_post_process
        config.pp_rank = args.pp_rank
        config.dp_rank = args.dp_rank
        config.tp_rank = args.tp_rank
        config.exp_rank = args.exp_rank
        config.is_rank_in_embedding_group = args.is_rank_in_embedding_group

        config.is_scaling_mode = args.is_scaling_mode

        # only scaling mode
        if config.is_scaling_mode:
            config.pipeline_model_parallel_size = args.fake_pp
            config.tensor_model_parallel_size = args.fake_tp
            config.expert_model_parallel_size = args.fake_exp
            config.num_moe_experts = args.fake_num_experts
            config.current_fake_rank_id = args.fake_current_rank_id

        if config.expert_model_parallel_size > 1:
            # get global routing table
            config.routing_hidden_states_shape = (args.seq_length/args.fake_tp, args.micro_batch_size, args.hidden_size)

            # get each ep rank's routing results (scores and indices)
            # we transfer scores and indices tensor from GPU to CPU
            # for real running mode, we use sim_routing to control its token distribution;
            # for scaling mode, we further use sim_routing to sim comm. process to share token info among all ep ranks.
            from megatron.profiler.moe.sim_pre_moe import set_pre_distribution_moe
            set_pre_distribution_moe(config=config)


    add_extra_config_kwargs()

    if args.is_scaling_mode:
        num_experts = args.fake_num_experts
    else:
        num_experts = args.num_experts

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(num_experts, args.moe_grouped_gemm)

        # Note: pre_process和post_process在传入参数时已经根据MODE修正
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
        print_rank_0(f"use mcore models, use_te = {use_te}")
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
        print_rank_0(f"use legacy models, use_te = {use_te}")

    return model



def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None
    
    # Note: tp==0时，只有memcp，否则有braodcast；使用cuda event进行trace吗？
    args = get_args()

    if not hasattr(args, 'simu_state'):
        setattr(args, 'simu_state', None)
        
    cmd = CMD(
        rank_id=args.simu_rank,
        mg_state=args.simu_state,
        name_cmd="get_batch",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.simu_stage_id,
        simu_start=args.simu_start,
        trace_start=args.trace_start,
        current_iter=args.current_iter,
        args=args
    )
    CMD.set_current_cmd(cmd)
    # token = current_cmd_var.set(cmd)
    # print(f"args.simu_micro_batch_ids:{args.simu_micro_batch_ids}")
    with cmd:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)
    # current_cmd_var.reset(token)
    # print(f"rank:{args.simu_rank}, getbatch_subop num: {len(cmd.sub_operations)}, getbatch_subop: {cmd.sub_operations}")

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    nvtx.range_push("loss_func")

    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    back_reuslt = loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}

    nvtx.range_pop()
    return back_reuslt


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start() 
    # global stimer
    nvtx.range_push("tp_get_batch")
    # with stimer(bdata=True):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    nvtx.range_pop()
    timers('batch-generator').stop()

    cmd = CMD(
        rank_id=args.simu_rank,
        mg_state=args.simu_state,
        name_cmd="forward_step",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.simu_stage_id,
        simu_start=args.simu_start,
        trace_start=args.trace_start,
        current_iter=args.current_iter,
        args=args
    )
    # token = current_cmd_var.set(cmd)
    CMD.set_current_cmd(cmd)
    with cmd:
        nvtx.range_push("model_fwd_step")
        # with stimer:
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels)
        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"rank_id: {args.simu_rank}, model_fwd time: {elapsed_time_ms}")
        nvtx.range_pop()
    # current_cmd_var.reset(token)
    # print(f"rank:{args.simu_rank}, fwd_sub_op num: {len(cmd.sub_operations)}, fwd_sub_op: {cmd.sub_operations}")
    # raise 0

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    # TP rank为0，且是第一或者最后一个stage
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'LLaMaSentencePieceTokenizer'}) # LLaMaSentencePieceTokenizer