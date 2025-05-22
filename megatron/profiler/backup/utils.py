import torch
from megatron.core.enums import ModelType
from torch.autograd.variable import Variable
from megatron.training.utils import unwrap_model
from megatron.core.utils import get_attr_wrapped_model
# from easy_timer import CUDATimer  
from megatron.profiler.cmd import CMD, current_cmd_var


def first_or_last_stage_fake_get_batch(is_pre_process:bool, pp_size:int, args):
    if pp_size == 1:
        tokens = torch.randint(1, 1001, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.randint(1, 1001, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.ones((args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device())
        attention_mask = torch.randint(0, 2, (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())
        position_ids = torch.randint(0, args.seq_length, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
    else:
        if is_pre_process:
            labels = None
            loss_mask = None
            tokens = torch.randint(1, 1001, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
            attention_mask = torch.randint(0, 2, (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())
            position_ids = torch.randint(0, args.seq_length, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        else:
            tokens = None
            position_ids = None
            labels = torch.randint(1, 1001, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
            loss_mask = torch.ones((args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device())
            attention_mask = torch.randint(0, 2, (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())
    
    batch = {
        'tokens': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'attention_mask': attention_mask,
        'position_ids': position_ids
        }
    
    return batch.values()
    

def get_input_tensor_shape(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    parallel_state
    ):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    # TODO:
    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))

    return tensor_shapes


def get_input_tensor(input_tensor_shapes:list, config):
    input_tensors = []
    for shape in input_tensor_shapes:
        tensor = (torch.rand(shape, device=torch.cuda.current_device())*0.2-0.1
                  ).to(dtype=config.pipeline_dtype, non_blocking=True)
        tensor.requires_grad_(True)
        input_tensors.append(tensor)
    return input_tensors


@CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype'], 'func': ['name']}, group_type='dp', comm_func='allreduce')
def allreduce(
    input_: torch.Tensor,
    group = None,
    async_op: bool = False,
    func: str = None
):

    # 假设进行了allreduce
    # torch.distributed.all_reduce(input_, group=tp_group, async_op=async_op)

    return input_



def average_losses_across_data_parallel_group(losses, args):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    # torch.distributed.all_reduce(averaged_losses,
    #                              group=mpu.get_data_parallel_group())

    _ = allreduce(input_=averaged_losses, group=None, func='loss_func')
    averaged_losses = averaged_losses / args.fake_dp

    return averaged_losses


def sim_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, args):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        # torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    # if args.check_for_nan_in_loss_and_grad:
    #     global_rank = torch.distributed.get_rank()
    #     assert not loss.isnan(), (
    #         f'Rank {global_rank}: found NaN in local forward loss calculation. '
    #         f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
    #     )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss], args)

    back_reuslt = loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}

    return back_reuslt

def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)


def sim_backward_step(rank_id, input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # with profile_timer(rank_id, "model_bwd"):
    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    # if (
    #     parallel_state.get_pipeline_model_parallel_world_size() > 1
    #     and parallel_state.is_pipeline_stage_after_split()
    #     and model_type == ModelType.encoder_and_decoder
    # ):
    #     if output_tensor_grad[1] is not None:
    #         input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad



def get_batch_on_this_tp_rank(data_iterator, args):

    @CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype'], 'func': ['name']}, group_type='tp', comm_func='broadcast')
    def _broadcast(input_, func=None):
       if input_ is not None:
            # torch.distributed.broadcast(input_, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
            # print("_broadcast | sim -> torch.distributed.broadcast")
            pass

    if args.tp_rank  == 0:
       if data_iterator is not None:
           data = next(data_iterator)
       else:
           data = None

       batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
       }

       if args.fake_pp == 1:
            _broadcast(batch['tokens'],func="tokens")
            _broadcast(batch['labels'],func="labels")
            _broadcast(batch['loss_mask'],func="loss_mask")
            _broadcast(batch['attention_mask'],func="attention_mask")
            _broadcast(batch['position_ids'],func="position_ids")
       elif args.is_pre_process:
            _broadcast(batch['tokens'],func="tokens")
            _broadcast(batch['attention_mask'],func="attention_mask")
            _broadcast(batch['position_ids'],func="position_ids")

       elif args.is_post_process:
            _broadcast(batch['labels'],func="labels")
            _broadcast(batch['loss_mask'],func="loss_mask")
            _broadcast(batch['attention_mask'],func="attention_mask")

    else:
       # load_batch_memcpy, 考虑到没有真实执行broadcast，因此将empty替换为randint
       tokens = torch.randint(1, 1001, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
       labels = torch.randint(1, 1001, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
       loss_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
            attention_mask = torch.randint(0, 2, (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())
       else:
           attention_mask=None
       position_ids = torch.randint(0, args.seq_length, (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())

       if args.fake_pp == 1:
            _broadcast(tokens,func="tokens")
            _broadcast(labels,func="labels")
            _broadcast(loss_mask,func="loss_mask")
            _broadcast(attention_mask,func="attention_mask")
            _broadcast(position_ids,func="position_ids")
            
       elif args.is_pre_process:
            labels = None
            loss_mask = None
            
            _broadcast(tokens,func="tokens")
            _broadcast(attention_mask,func="attention_mask")
            _broadcast(position_ids,func="position_ids")
       elif args.is_post_process:
            tokens = None
            position_ids = None
            
            _broadcast(labels,func="labels")
            _broadcast(loss_mask,func="loss_mask")
            _broadcast(attention_mask,func="attention_mask")

       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids
       }

    return batch



def sim_get_batch(args, data_iterator):
    batch = get_batch_on_this_tp_rank(data_iterator=data_iterator, args=args)

    return batch.values()



def sim_forward_step(rank_id, model, model_type, args, parallel_state, config, train_data_iterator):
    # 1. set_input_tensor(tensor recv)
    input_tensor_shapes = []
    if not args.is_pre_process:
        # simulate recv tensor
        input_tensor_shapes = get_input_tensor_shape(rank=rank_id, model_type=model_type, seq_length=args.seq_length,
                                                     micro_batch_size=args.micro_batch_size, decoder_seq_length=args.decoder_seq_length,
                                                     config=config, parallel_state=parallel_state)
        input_tensor = get_input_tensor(input_tensor_shapes=input_tensor_shapes, config=config)
        # set attr
        set_input_tensor = get_attr_wrapped_model(model[0], "set_input_tensor")
        set_input_tensor(input_tensor)
    else:
        input_tensor = [None]
    if args.simu_start == True:
        print(f"rank_id = {rank_id}, input_tensor_shapes: {input_tensor_shapes}")

    # 2. forward_step_func()
    #   2.1 get_batch()
    if args.is_pre_process or args.is_post_process:
        cmd = CMD(
            rank_id=rank_id,
            mg_state=None,
            name_cmd="get_batch",
            use_cuda=True,
            stage_operations_trace_dict=args.stage_operations_trace,
            micro_batch_ids_dict=args.simu_micro_batch_ids,
            stage_id=args.pp_rank,
            simu_start=args.simu_start,
            description=None, 
            group_kind="tp",
            trace_start=args.trace_start,
            current_iter=args.trace_start,
            args=args
        )
        CMD.set_current_cmd(cmd)
        with cmd:
            tokens, labels, loss_mask, attention_mask, position_ids = sim_get_batch(data_iterator=train_data_iterator, args=args)
            # if args.simu_start == True:
            #     print(f"rank:{rank_id}, getbatch_subop num: {len(cmd.sub_operations)}, getbatch_subop: {cmd.sub_operations}")
    else:
        tokens, labels, loss_mask, attention_mask, position_ids = None, None, None, None, None

    # # 覆盖上面的tokens, labels, loss_mask, attention_mask, position_ids
    # if args.is_pre_process or args.is_post_process:
    #     cmd = CMD(
    #         rank_id=rank_id,
    #         mg_state=None,
    #         name_cmd="get_batch",
    #         use_cuda=True,
    #         stage_operations_trace_dict=args.stage_operations_trace,
    #         micro_batch_ids_dict=args.simu_micro_batch_ids,
    #         stage_id=args.pp_rank,
    #         simu_start=args.simu_start,
    #         description=None, 
    #         group_kind="tp",
    #         trace_start=args.trace_start,
    #         current_iter=args.trace_start,
    #         args=args
    #     )
    #     CMD.set_current_cmd(cmd)
    #     with cmd:
    #         tokens, labels, loss_mask, attention_mask, position_ids = first_or_last_stage_fake_get_batch(
    #             is_pre_process=args.is_pre_process, pp_size=args.fake_pp, args=args)
    # else:
    #     tokens, labels, loss_mask, attention_mask, position_ids = None, None, None, None, None


    #   2.2 output_tensor = model()
    unwrapped_model = unwrap_model(model)
    unwrapped_model = unwrapped_model[0]
    from megatron.core.models.gpt import GPTModel
    assert isinstance(unwrapped_model, GPTModel)
    # with profile_timer(rank_id, "model_fwd"):
    cmd = CMD(
    rank_id=rank_id,
    mg_state=None,
    name_cmd="forward_step",
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
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output_tensor = unwrapped_model(tokens, position_ids, attention_mask, labels=labels)
        output_tensor = output_tensor.contiguous()
        stop_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(stop_event)
        if args.simu_start == True:
            print(f"rank:{rank_id},cuda fwd time: {duration}")
            print(f"rank:{rank_id}, fwd_subop num: {len(cmd.sub_operations)}, fwd_subop: {cmd.sub_operations}")
            
    # 3. is_pipeline_last_stage -> loss_func()
    if args.is_post_process:
        # with profile_timer(rank_id, "loss_func"):
        cmd = CMD(
        rank_id=rank_id,
        mg_state=None,
        name_cmd="loss_func",
        use_cuda=True,
        stage_operations_trace_dict=args.stage_operations_trace,
        micro_batch_ids_dict=args.simu_micro_batch_ids,
        stage_id=args.pp_rank,
        simu_start=args.simu_start,
        description="loss_func, calculate and DP allreduce for the last stage", 
        trace_start=args.trace_start,
        current_iter=args.trace_start,
        args=args
        )
        CMD.set_current_cmd(cmd)
        with cmd:
            output_tensor = sim_loss_func(loss_mask=loss_mask, output_tensor=output_tensor, args=args)
            loss, loss_reduced = output_tensor
            output_tensor = loss / 2
            # forward_data_store.append(loss_reduced)

    del tokens, labels, loss_mask, attention_mask, position_ids

    # 4. return tensor
    return output_tensor, input_tensor