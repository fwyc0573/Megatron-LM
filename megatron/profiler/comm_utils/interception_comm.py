import torch
from typing import Optional, Tuple
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.distributed import get_distributed_world_size
from megatron.profiler.cmd import CMD

@CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype'], 'func': ['name'], 'overlap_op': ['name']}, group_type='tp', comm_func='allreduce')
def allreduce_wrapper(
    input_: torch.Tensor,
    tp_group: Optional[dist_group_type] = None,
    async_op: bool = False,
    func: str = None,
    overlap_op: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    # if get_distributed_world_size(tp_group) == 1:
    #     return input_, None

    # All-reduce.
    handle = torch.distributed.all_reduce(input_, group=tp_group, async_op=async_op)

    return input_, handle

        
@CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype'], 'func': ['name']}, group_type='tp', comm_func='broadcast')
def broadcast_wrapper(
    input_: torch.Tensor,
    func: str = None,
    tp_group: Optional[dist_group_type] = None,
    tp_src_rank: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast the input tensor across model parallel group."""

    if tp_group is not None and tp_src_rank is not None:
        torch.distributed.broadcast(input_, tp_src_rank, group=tp_group)


@CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype'], 'func': ['name']}, group_type='tp', comm_func='allreduce')
def reduce_wrapper(input_, func=None, op=torch.distributed.ReduceOp.SUM, async_op=False, tp_group=None):
    """All-reduce the input tensor across model parallel group."""

    # All-reduce.
    if async_op is True:
        handle = torch.distributed.all_reduce(input_, op=op, group=tp_group, async_op=True)
        return handle
    else:
        torch.distributed.all_reduce(input_, op=op, group=tp_group)

    return input_