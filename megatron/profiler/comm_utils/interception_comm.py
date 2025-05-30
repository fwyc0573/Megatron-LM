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
    func = func
    overlap_op = overlap_op

    # Bypass the function if we are using only 1 GPU.
    if get_distributed_world_size(tp_group) == 1:
        return input_, None

    # All-reduce.
    handle = torch.distributed.all_reduce(input_, group=tp_group, async_op=async_op)

    return input_, handle


