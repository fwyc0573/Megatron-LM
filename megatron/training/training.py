# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import dataclasses
from datetime import datetime
import gc
import logging
import math
import os
import sys
from .log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
import random
import numpy as np


from megatron.core import mpu, tensor_parallel
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config, StragglerDetector
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.pipeline_parallel import get_forward_backward_func

from megatron.core import parallel_state
from megatron.profiler.cmd import CMD, write_list_to_file
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
from megatron.profiler import profiler, comm_hook, initialize_profiling, install_hooks, uninstall_hooks

from .utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model)
from .global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)


stimer = StragglerDetector()

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
        12
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    + (args.seq_length / args.hidden_size)
                ) * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )


def append_to_progress_log(string):
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                    f"# GPUs: {num_gpus}\t{string}\n")


def get_start_time_from_progress_log():
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    args = get_args()
    assert args.save is not None
    progress_log_filename = os.path.join(args.save, "progress.txt")

    # start_time is time when job with same world size started.
    # start_num_floating_point_operations is the number of floating-point operations
    # completed when this job started.
    # latest_num_floating_point_operations is the number of floating-point operations
    # completed in most recent saved checkpoint.
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(': ')[1])

    with open(progress_log_filename, 'r') as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split('\t')
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = \
                    _get_field(line_tokens[7], float)
            if world_size_in_line != args.world_size:
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = \
                        latest_num_floating_point_operations
    assert start_time is not None and start_num_floating_point_operations is not None, \
        "Should have seen at least one 'Starting job' entry with same world_size"
    return datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), \
        start_num_floating_point_operations

def set_seed(seed):
    # TODO-YC: will it affect the performance?
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    
def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """
    # args = get_args()
    # args_defaults = {'tokenizer_type':args.tokenizer_type}
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    # YC: seed set
    set_seed(args.seed)


    # 设定模拟开始为false
    args.simu_start = False
    # args.simu_micro_batch_ids = {"recv_forward":-1, "forward_step":-1, "send_forward":-1, "recv_backward":-1,
    #                               "backward_step":-1, "send_backward":-1, "tp_load_batch_broadcast":-1, "dp_allreduce":-1, "tp_allreduce":-1}

    # todo-yc: maybe the value of vars here are equal to '0'? check it.
    args.simu_stage_id = parallel_state.get_pipeline_model_parallel_rank()
    args.simu_rank = str(torch.distributed.get_rank())
    args.global_model_params_dict = {}
    
    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    # args = get_args()
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })


    # TODO-YC: abvoed operations are all called in the first process which mean only call one time in sim. scaling mode (rank0, i.e., the only one gpu device); But... is it a correct method?


    """ iter-rank start, profile submodel of each rank. 
    for wrank in range(args.fake_world_size):
        补充所有参数**scaling_kwargs: first_process, last_process, tp_rank, dp_rank, pp_rank,local_rank

        FWD和BWD涉及4个关键参数: input_tensor, input_tensor_grad, output_tensor, output_tensor_grad, 对应fwd, bwd, sned/recv grad/activation
    """
    # from megatron.profiler.easy_timer import CUDATimer
    from megatron.profiler.utils import (
        sim_backward_step,
        deallocate_output_tensor,
        sim_forward_step,
        get_attr_wrapped_model
        )
        # first_or_last_stage_fake_get_batch,
        # sim_loss_func,
        # get_input_tensor_shape,
        # get_input_tensor,
    def add_extra_args_kwargs(rank_instance=None):
        args.is_pre_process = rank_instance.is_pre_process() if rank_instance is not None else None
        args.is_post_process = rank_instance.is_post_process() if rank_instance is not None else None
        args.pp_rank = rank_instance._get_pp_local_rank() if rank_instance is not None else None
        args.dp_rank = rank_instance._get_dp_local_rank() if rank_instance is not None else None
        args.tp_rank = rank_instance._get_tp_local_rank() if rank_instance is not None else None
        args.mp_rank = rank_instance._get_mp_local_rank() if rank_instance is not None else None
        args.exp_rank = rank_instance._get_exp_local_rank() if rank_instance is not None else None
        args.dp_groups= rank_instance._get_dp_groups() if rank_instance is not None else None
        args.is_rank_in_embedding_group = rank_instance.is_rank_in_embedding_group() if rank_instance is not None else None
        args.simu_start = False
        args.stage_operations_trace = {}
        args.simu_micro_batch_ids = {
            "recv_forward": -1, "forward_step": -1, "send_forward": -1, "recv_backward": -1,
            "backward_step": -1, "send_backward": -1, "tp_load_batch_broadcast": -1, "dp_allreduce": -1,
            "tp_allreduce": -1, "optimizer_step": -1, 'get_batch': -1, 'loss_func': -1, 'ep_allreduce': -1
        }

    if args.is_scaling_mode:
        if not hasattr(args, 'fake_current_rank_id') or args.fake_current_rank_id is None:
            print_rank_0("ERROR: In scaling mode, --fake-current-rank-id must be provided via command line.")
            sys.exit(1)
        if args.fake_current_rank_id < 0 or args.fake_current_rank_id >= args.fake_world_size:
            print_rank_0(f"ERROR: fake_current_rank_id {args.fake_current_rank_id} is out of range for fake_world_size {args.fake_world_size}.")
            sys.exit(1)

        from megatron.profiler.rank_manager import RankManager
        from megatron.profiler.parallel_group_manager import MPUInfo, ParallelGroupManager
        manager = ParallelGroupManager(local_size=args.fake_local_rank, world_size=args.fake_world_size, pp_size=args.fake_pp, tp_size=args.fake_tp)
        mpu_info: MPUInfo = manager.get_mpu_info()
        all_groups = manager.get_all_groups()
        rank_manager = RankManager(args, all_groups)
        rank_instances: dict = rank_manager.get_rank_zoos()
        print(f"mpu_info:{mpu_info}")
        print(f"all_groups:{all_groups}")
        # raise 0 
    
        current_fake_rank_id_int = args.fake_current_rank_id
        if current_fake_rank_id_int not in rank_instances:
            print_rank_0(f"ERROR: Rank ID {current_fake_rank_id_int} not found in rank_manager. Available: {list(rank_instances.keys())}")
            sys.exit(1)
        rank_instance = rank_instances[current_fake_rank_id_int]
        rank_id = current_fake_rank_id_int
        print_rank_0(f"===> Megatron-LM Single GPU Simulation: Processing Fake Rank {rank_id} <===")


        # profile_timer = CUDATimer()
        warm_up_iter = 10 # args.train_iters
        args.iteration=0
        # YC: this sign control async profile mode. however, do we need it this?
        CMD.set_current_profile_sign(True)

        # for rank_id, rank_instance in rank_instances.items():
        add_extra_args_kwargs(rank_instance)

        # 这里占用了大量的内存，但实际上只有tp_rank=0的才涉及next(data_iterator)?
        args.iteration=0
        train_data_iterator, _, _= build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    
        # get model and optimizer
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type, args=args)
        config = get_model_config(model[0])

        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # forward_data_store = []

        # warm up
        # simu_start为False时，上下文管理器不trace；current_cmd为None时，装饰器不trace；
        if args.simu_start == False:
            for _ in range(warm_up_iter):
                output_tensor, input_tensor = sim_forward_step(rank_id, model, model_type, args, parallel_state, config, train_data_iterator)
                output_tensor = output_tensor.contiguous()
                output_tensor_grad = [torch.randn_like(output_tensor)]
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                # TODO:确认这儿，装饰器已写了，上下文管理器没有，能否正常追踪？
                input_tensor_grad = sim_backward_step(rank_id, input_tensor, [output_tensor], output_tensor_grad, model_type, config)
                optimizer.step()
                for model_chunk in model:
                    model_chunk.zero_grad_buffer()
                optimizer.zero_grad()
                args.iteration += 1

            # profile_timer只有在warm_up_sign为False才进行测量和记录
            # profile_timer.warm_up_sign = False
            args.simu_start = True
            del output_tensor,input_tensor,output_tensor_grad, input_tensor_grad
            # forward_data_store.clear()
            print(f"rank_id = {rank_id}, finish warm up ...")
            args.iteration = 0
        

        # TODO: if we finish warmup, why we not clean the param in optimizer for each iter ?(follow training step func()?) 

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        nvtx.range_push(f"rank:{rank_id}, complete iteration")
        nvtx.range_push(f"rank:{rank_id}, model_fwd_step")
    
        # get_batch / FWD (loss_func:dp_allreudce)
        output_tensor, input_tensor = sim_forward_step(rank_id, model, model_type, args, parallel_state, config, train_data_iterator)
        # TODO: 为解决"counter-productive to free a view of another tensor."，是否有其他计算影响？
        output_tensor = output_tensor.contiguous()
        print(f"rank_id = {rank_id}, finish FWD profile ...")
        
        nvtx.range_pop()

        # BWD
        # 生成模拟的 output_tensor_grad(recv grad)
        output_tensor_grad = [torch.randn_like(output_tensor)]
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        # with profile_timer(rank_id, "BWD"):

        cmd = CMD(
        rank_id=rank_id,
        mg_state=None,
        name_cmd="backward_step",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.pp_rank,
        simu_start=args.simu_start,
        description=None, 
        trace_start=args.trace_start,
        current_iter=args.trace_start,
        args=args
        )
        CMD.set_current_cmd(cmd)
        with cmd:
            nvtx.range_push(f"rank:{rank_id}, model_bwd_step")
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            input_tensor_grad = sim_backward_step(rank_id, input_tensor, [output_tensor], output_tensor_grad, model_type, config)
            stop_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(stop_event)
            nvtx.range_pop()
            if args.simu_start == True:
                print(f"rank:{rank_id},bwd time: {duration}")
                print(f"rank:{rank_id}, bwd_subop num: {len(cmd.sub_operations)}, bwd_subop: {cmd.sub_operations}")
                print(f"rank:{rank_id}, finish BWD profile ...")

        # dp_allreduce
        pos_p_t = (args.pp_rank,args.tp_rank)
        used_dtype = torch.float16 if args.fp16 or args.bf16 else torch.float32
        cmd = CMD(
        rank_id=rank_id,
        mg_state=None,
        name_cmd="dp_allreduce",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.pp_rank,
        simu_start=args.simu_start,
        description=None, 
        group_kind="dp",
        trace_start=args.trace_start,
        current_iter=args.trace_start,
        args=args,
        input__shape=[args.global_model_params_dict[pos_p_t]["elem_sum"]], 
        input__dtype=used_dtype,
        )
        cmd.no_trace_update(0,0)


        # ep_allreduce
        if (args.is_rank_in_embedding_group and args.fake_pp > 1):
            ep_input__shape = None
            ep_input__dtype = None
            if args.is_pre_process:
                model_module = model[0]
            elif args.is_post_process:
                model_module = model[-1]
            else:  # We do not support the interleaved schedule for T5 yet.
                model_module = model[0]
            model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
            if model_module.share_embeddings_and_output_weights:
                weight = model_module.shared_embedding_or_output_weight()
                grad = weight.main_grad
                ep_input__shape = grad.shape
                ep_input__dtype = grad.dtype

            cmd = CMD(
            rank_id=rank_id,
            mg_state=None,
            name_cmd="ep_allreduce",
            use_cuda=True,
            stage_operations_trace_dict=args.stage_operations_trace,
            micro_batch_ids_dict=args.simu_micro_batch_ids,
            stage_id=args.pp_rank,
            simu_start=args.simu_start,
            description=None,
            group_kind="ep",
            trace_start=args.trace_start,
            current_iter=args.trace_start,
            args=args,
            input__shape=ep_input__shape, 
            input__dtype=ep_input__dtype,
            )
            cmd.no_trace_update(0,0)


        # optimizer.step()
        # with profile_timer(rank_id, "OPTIM_STEP"):
        cmd = CMD(
        rank_id=rank_id,
        mg_state=None,
        name_cmd="optimizer_step",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.pp_rank,
        simu_start=args.simu_start,
        description=None, 
        trace_start=args.trace_start,
        current_iter=args.trace_start,
        args=args
        )
        with cmd:
            nvtx.range_push(f"rank:{rank_id}, optimizer_step")
            params = optimizer.get_parameters()  # 获取所有参数
            total_param_count = sum(param.numel() for param in params)  # 计算参数总量
            grads_for_norm = optimizer.get_main_grads_for_grad_norm()  # 获取所有梯度
            total_grad_count = sum(grad.numel() for grad in grads_for_norm)  # 计算梯度总量

            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)

            print(f"Finish warmup and Optimizer structure is {optimizer}")
            # torch.cuda.synchronize()
            # optimizer.zero_grad()
            start_event.record()

            optimizer.step()

            stop_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(stop_event)
            nvtx.range_pop()
            if args.simu_start == True:
                print(f"rank:{rank_id},optimizer_step time: {duration}, total_param_count: {total_param_count}, total_grad_count: {total_grad_count}")
        print(f"rank:{rank_id}, finish optimizer.step profile ...")
        del params,total_param_count,grads_for_norm,total_grad_count

        nvtx.range_pop()
        # profile_timer.warm_up_sign = True

        # release GPU memory
        allocated_memory_before = torch.cuda.memory_allocated()
        reserved_memory_before = torch.cuda.memory_reserved()
        print(f"rank:{rank_id}, Before memory release - Allocated: {allocated_memory_before}, Reserved: {reserved_memory_before}")
        
        # name_args = f"wd{args.world_size}_tp{args.tensor_model_parallel_size}_pp{args.pipeline_model_parallel_size}_numl{args.num_layers}_\
        #     bs{args.micro_batch_size}_{args.main_tokenizer_type}"
        
        name_args = f"wd{args.fake_world_size}_tp{args.fake_tp}_pp{args.fake_pp}_numl{args.num_layers}_bs{args.micro_batch_size}_{args.main_tokenizer_type}"
        write_list_to_file(rank_id, args.stage_operations_trace[rank_id], file_path="profiler_log", name_args=name_args)
        print(f"rank:{rank_id}, trace log has been written to txt...")

        # del input_tensor
        # del output_tensor
        # del output_tensor_grad
        # del input_tensor_grad
        # # del forward_data_store
        # del config #, reserved_memory_before, allocated_memory_before
        # # del tokens, labels, loss_mask, attention_mask, position_ids
        # del model, optimizer, opt_param_scheduler, train_data_iterator
        # torch.cuda.empty_cache()
        # gc.collect()
        print(f"rank:{rank_id}, finish release GPU memory ...")

        # 输出释放内存后的GPU内存使用情况
        allocated_memory_after = torch.cuda.memory_allocated()
        reserved_memory_after = torch.cuda.memory_reserved()
        print(f"Rank:{rank_id}, Memory Allocated: {allocated_memory_after}, Reserved: {reserved_memory_after}")
        # del allocated_memory_after, reserved_memory_after

        return
    
    else:
        # 用于初始化参数
        # todo-yc: a little bit of confusing here. real running mode also call this function? why? what's the difference? rank instance seems the key point here
        # for real running mode ranks here, it finish the init definition? but why we need None value?
        # but for sim-scaling mode, it seems that sequentially get the parallel rank index in loop
        add_extra_args_kwargs()

    # ------------- up to here, the process get ranks' parallel index = None? (e.g., pp/tp/dp local rank index) ---------------
    # following steps used for real running mode

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, args=args)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)


    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:

            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config)
        print_datetime('after training is done')

    else:
        print_rank_0('skipping training (--skip-train is on) ...')
        iteration = args.iteration



def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        # 需要指定pre_process和post_process，并且要及时释放GPU内存
        if not args.is_scaling_mode:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
        else:
            pre_process = args.is_pre_process
            post_process = args.is_post_process
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    # TODO-YC: we should compare the runing mode with sim scaling mode here.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if not args.is_scaling_mode:
        if mpu.get_data_parallel_rank() == 0:
            # print(' > number of parameters on (tensor, pipeline) '
            #     'model parallel rank ({}, {}): {}'.format(
            #     mpu.get_tensor_model_parallel_rank(),
            #     mpu.get_pipeline_model_parallel_rank(),
            #     sum([sum([p.nelement() for p in model_module.parameters()])
            #         for model_module in model])), flush=True)
            total_params = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])

            print(' > number of parameters on (tensor, pipeline) '
                'model parallel rank ({}, {}): {}'.format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                total_params), flush=True)
            
            pos_p_t = (mpu.get_pipeline_model_parallel_rank(),mpu.get_tensor_model_parallel_rank())
            if pos_p_t not in args.global_model_params_dict:
                args.global_model_params_dict[pos_p_t] = {}
            args.global_model_params_dict[pos_p_t] = {"elem_sum": total_params,"dtype_list": None}
    else:
        if args.dp_rank == 0:
            total_params = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
            # TODO-YC: we need compare here.
            print(' > number of parameters on (tensor, pipeline) '
                'model parallel rank ({}, {}): {}'.format(
                args.tp_rank,
                args.pp_rank,
                total_params), flush=True)
            
            pos_p_t = (args.pp_rank,args.tp_rank)
            if pos_p_t not in args.global_model_params_dict:
                args.global_model_params_dict[pos_p_t] = {}
            args.global_model_params_dict[pos_p_t] = {"elem_sum": total_params,"dtype_list": None}

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    # if not args.is_scaling_mode:
    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size)
        model = [DDP(config,
                    ddp_config,
                    model_chunk,
                    data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                    expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                    # Turn off bucketing for model_chunk 2 onwards, since communication for these
                    # model chunks is overlapped with compute anyway.
                    disable_bucketing=(model_chunk_idx > 0))
                for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            # YC: check it here
            raise 0
            for model_module in model:
                model_module.broadcast_params()

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              args=None):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult, args)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None or args.pretrained_checkpoint is not None:
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler



def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    # print(f"args.pipeline_model_parallel_size = {args.pipeline_model_parallel_size}")
    forward_backward_func = get_forward_backward_func()

    # temp add
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    nvtx.range_push("fwd_bwd_func")
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)
    nvtx.range_pop()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_fb_time = start_event.elapsed_time(end_event)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.current_iter)

    # Update parameters.
    cmd = CMD(
        rank_id=args.simu_rank,
        mg_state=args.simu_state,
        name_cmd="optimizer_step",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.simu_stage_id,
        simu_start=args.simu_start,
        trace_start=args.trace_start,
        current_iter=args.current_iter,
        args=args
    )
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)

    # params = optimizer.get_parameters()  # 获取所有参数
    # total_param_count = sum(param.numel() for param in params)  # 计算参数总量
    # grads_for_norm = optimizer.get_main_grads_for_grad_norm()  # 获取所有梯度
    # total_grad_count = sum(grad.numel() for grad in grads_for_norm)  # 计算梯度总量

    with cmd:
        nvtx.range_push("param optim")
        print(f"Optimizer type: {type(optimizer).__name__}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_optim_time_ms = start_event.elapsed_time(end_event)
        nvtx.range_pop()
        print(f"rank_id: {args.simu_rank}, optim_step time: {elapsed_optim_time_ms}")
        # print(f"rank_id: {args.simu_rank}, total_parm_count: {total_param_count}, total_grad_count: {total_grad_count}")
    timers('optimizer').stop()
    # args.simu_micro_batch_ids["optimizer_step"] += 1
    # cmd = CMD(args.simu_rank, "finalize", "optimizer_step", args.simu_micro_batch_ids["optimizer_step"])
    # args.stage_operations_trace[args.simu_rank].append(str(cmd))
    elapsed_sum = elapsed_optim_time_ms + elapsed_fb_time
    print(f"rank_id: {args.simu_rank}, sum(1f1b+optim) time: {elapsed_sum}")

    nvtx.range_push("afet parm optim")
    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.current_iter)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()
    nvtx.range_pop()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # cmd = CMD(
        #     rank_id=args.simu_rank,
        #     mg_state=args.simu_state,
        #     name_cmd="calcu_loss",
        #     use_cuda=True,
        #     stage_operations_trace_dict=args.stage_operations_trace,
        #     micro_batch_ids_dict=args.simu_micro_batch_ids,
        #     stage_id=args.simu_stage_id,
        #     simu_start=args.simu_start,
        #     description="Average loss across microbatches(in last stage)",
        #     trace_start=args.trace_start,
        #     current_iter=args.current_iter,
        #     args=args
        # )
        # with cmd:
        #     # Average loss across microbatches.
        #     loss_reduced = {}
        #     for key in losses_reduced[0]:
        #         losses_reduced_for_key = [x[key] for x in losses_reduced]
        #         loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        # return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return None, None, None, None
    
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    if one_logger:
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)
        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def compute_throughputs_and_append_to_progress_log(iteration,
                                                   num_floating_point_operations_so_far):
    args = get_args()
    if args.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    global _TRAIN_START_TIME
    job_throughput = \
        (num_floating_point_operations_so_far -
         args.num_floating_point_operations_so_far) / (
            (time.time() - _TRAIN_START_TIME) * 10**12 * args.world_size)

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = \
        (num_floating_point_operations_so_far -
         start_num_floating_point_operations) / (
            elapsed_time * 10**12 * args.world_size)

    tokens_so_far = args.consumed_train_samples * args.seq_length

    append_to_progress_log(f"Saved checkpoint\tIteration: {iteration}\t"
                           f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
                           f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
                           f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
                           f"Tokens (in billions): {tokens_so_far / 10**9:.2f}")


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler,
                             num_floating_point_operations_so_far):
    args = get_args()
    timers = get_timers()
    # Extra barrier is added to make sure all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])

    if args.log_progress:
        compute_throughputs_and_append_to_progress_log(iteration,
                                                       num_floating_point_operations_so_far)


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    # write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    one_logger = get_one_logger()
    if one_logger:
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    exit = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton Initialization
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    total_flops = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0
    def track_e2e_metrics():
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:
            train_duration = timers('interval-time').active_time()  # overall_elapsed
            train_samples = args.consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (train_duration * 1000.0) / train_iterations
            if eval_iterations:
                validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            one_logger.log_metrics({
                'train_iterations_end': iteration,
                'train_samples_end': args.consumed_train_samples,
                'train_iterations': train_iterations,
                'train_samples': train_samples,
                'train_iterations_time_msecs_avg': train_iterations_time_msecs_avg,
                'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg
            })

    args.simu_start = True
    args.stage_operations_trace = {}
    while iteration < args.train_iters:
        # if args.profile and \
        #    iteration == args.profile_step_start and \
        #    torch.distributed.get_rank() in args.profile_ranks:
        #     torch.cuda.cudart().cudaProfilerStart()
        #     torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        # if iteration == args.train_iters-1:
        #     break

        torch.cuda.synchronize()
        dist.barrier()

        # 每个iter RESET batch id记录和非trace iter的cmd dict
        args.simu_micro_batch_ids = {
            "recv_forward": -1, "forward_step": -1, "send_forward": -1, "recv_backward": -1,
            "backward_step": -1, "send_backward": -1, "tp_load_batch_broadcast": -1, "dp_allreduce": -1,
            "tp_allreduce": -1, "optimizer_step": -1, "loss_func": -1, 'get_batch': -1, "ep_allreduce": -1
        }
        if iteration < args.trace_start:
            args.stage_operations_trace = {}

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far)
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        args.current_iter = iteration
        nvtx.range_push("iteration:"+str(iteration))
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        nvtx.range_pop()

        nvtx.range_push("after iteration")
        iteration += 1
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
        num_fp_ops = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_fp_ops
        total_flops += num_fp_ops

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()
        nvtx.range_pop()

    if args.do_trace:
        # name_args = f"wd{args.fake_world_size}_tp{args.fake_tp}_pp{args.fake_pp}_{args.main_tokenizer_type}"
        # args.micro_batch_size args.main_tokenizer_type
        name_args = f"wd{args.world_size}_tp{args.tensor_model_parallel_size}_pp{args.pipeline_model_parallel_size}_numl{args.num_layers}_bs{args.micro_batch_size}_{args.main_tokenizer_type}"
        write_list_to_file(args.simu_rank, args.stage_operations_trace[args.simu_rank], name_args=name_args)
        print(f"rank:{args.simu_rank}, trace log has been written to txt...")
    # track_e2e_metrics()

    # # Flush TensorBoard and WandB writers.
    # writer = get_tensorboard_writer()
    # if writer:
    #     writer.flush()
    # wandb_writer = get_wandb_writer()
    # if wandb_writer:
    #     wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:
        optimizer.disable_pre_hook()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:
        sys.exit()

    return iteration, num_floating_point_operations_so_far


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

    timers('evaluate').stop()
    timers.log(['evaluate'])

    return total_loss_dict, collected_non_loss_data, False

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    # Timelimit hit during evaluation
    if timelimit:
        return
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)
            if wandb_writer and is_last_rank():
                wandb_writer.log({
                    '{} validation'.format(key): total_loss_dict[key].item()},
                    iteration)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    if not args.is_scaling_mode:
        tp_rank = mpu.get_tensor_model_parallel_rank()
    else:
        tp_rank = args.tp_rank
    if is_distributed or tp_rank == 0:

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    if not args.is_scaling_mode:
        torch.distributed.broadcast(flags, 0)

        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()
    else:
        args.do_train = True
        args.do_valid = False
        args.do_test = False

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'external']

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if not args.is_scaling_mode:
        if valid_dataloader is not None:
            valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
        else:
            valid_data_iterator = None

        if test_dataloader is not None:
            test_data_iterator = _get_iterator(dl_type, test_dataloader)
        else:
            test_data_iterator = None

        return train_data_iterator, valid_data_iterator, test_data_iterator
    
    else:
        return train_data_iterator, None, None