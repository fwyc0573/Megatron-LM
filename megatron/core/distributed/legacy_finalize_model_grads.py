# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .. import parallel_state
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config

import torch.cuda.nvtx as nvtx
# from megatron.training import get_args
from megatron.profiler.cmd import CMD


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            # print(f"_allreduce_word_embedding_grads | grad.shape: {grad.shape}, grad.d: {grad.dtype}")
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            cmd = CMD.get_current_cmd()
            cmd.set_tensor_shape_and_dtype(grad.shape, grad.dtype)


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())
        cmd = CMD.get_current_cmd()
        cmd.set_tensor_shape_and_dtype(grad.shape, grad.dtype)

def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config) # for T5


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if (
                    getattr(param, 'sequence_parallel', False)
                    or 'q_layernorm' in name
                    or 'k_layernorm' in name
                ):
                    grad = param.main_grad
                    grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


def finalize_model_grads(model: List[torch.nn.Module], args):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied).
    """

    config = get_model_config(model[0])
    # args = get_args()

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    cmd = CMD(
        rank_id=args.simu_rank,
        mg_state=args.simu_state,
        name_cmd="dp_allreduce",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.simu_stage_id,
        simu_start=args.simu_start,
        description="model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas",
        group_kind="dp",
        trace_start=args.trace_start,
        current_iter=args.current_iter,
        args=args
    )
    CMD.set_current_cmd(cmd)
    with cmd:
        nvtx.range_push(f"allreduce_grads_sync_model_chunk")
        # YC: check here, does it include ep optimizer's allreduce?
        for model_chunk in model:
            model_chunk.finish_grad_sync()
        nvtx.range_pop()

    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    # sequence_parallel
    _allreduce_layernorm_grads(model, config) 
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    # 拆分了_allreduce_embedding_grads()函数分别对应到了_allreduce_word_embedding_grads()和_allreduce_position_embedding_grads()
    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        # 检查是否需要进行实际的 embedding allreduce
        # 只有当 share_embeddings_and_output_weights=True 时才需要创建 ep_allreduce CMD
        need_embedding_allreduce = False
        for model_module in model:
            if isinstance(model_module, list):
                model_module = model_module[0]
            model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
            if hasattr(model_module, 'share_embeddings_and_output_weights') and model_module.share_embeddings_and_output_weights:
                need_embedding_allreduce = True
                break

        if need_embedding_allreduce:
            cmd = CMD(
                rank_id=args.simu_rank,
                mg_state=args.simu_state,
                name_cmd="ep_allreduce",
                use_cuda=True,
                stage_operations_trace_dict=args.stage_operations_trace,
                micro_batch_ids_dict=args.simu_micro_batch_ids,
                stage_id=args.simu_stage_id,
                simu_start=args.simu_start,
                description="_allreduce_word_embedding_grads",
                group_kind="ep",
                trace_start=args.trace_start,
                current_iter=args.current_iter,
                args=args
            )
            CMD.set_current_cmd(cmd)
            with cmd:
                nvtx.range_push(f"allreduce_word_embedding_grads")
                # _allreduce_embedding_grads(model, config)
                _allreduce_word_embedding_grads(model, config)
                nvtx.range_pop()
        else:
            # 对于 untied embeddings 的情况，直接调用函数但不创建 CMD
            # 这样可以保持训练逻辑的完整性，但不会在 trace 中记录无效的 ep_allreduce
            nvtx.range_push(f"allreduce_word_embedding_grads")
            _allreduce_word_embedding_grads(model, config)
            nvtx.range_pop()
            print(f"Skipped ep_allreduce CMD creation for untied embeddings (share_embeddings_and_output_weights=False)")

    if (
        parallel_state.is_rank_in_position_embedding_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        cmd = CMD(
            rank_id=args.simu_rank,
            mg_state=args.simu_state,
            name_cmd="pep_allreduce",
            use_cuda=True,
            stage_operations_trace_dict=args.stage_operations_trace,
            micro_batch_ids_dict=args.simu_micro_batch_ids,
            stage_id=args.simu_stage_id,
            simu_start=args.simu_start,
            description="_allreduce_position_embedding_grads",
            group_kind="pep",
            trace_start=args.trace_start,
            current_iter=args.current_iter,
            args=args
        )
        with cmd: 
            nvtx.range_push(f"allreduce_position_embedding_grads")
            _allreduce_position_embedding_grads(model, config)
            nvtx.range_pop()

    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

