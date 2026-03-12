from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import Bucket
from megatron.profiler.cmd import CMD


def _build_args():
    return SimpleNamespace(
        trace_subop_sync_mode="global",
        trace_cmd_sync_mode="global",
        trace_kernel_ground_truth=False,
        trace_kernel_ground_truth_phase=False,
        trace_ddp_grad_overlap=True,
        is_scaling_mode=True,
        scaling_trace_metadata_comm_duration=False,
    )


def _build_cmd(name_cmd, stage_operations_trace, micro_batch_ids, args):
    return CMD(
        rank_id=0,
        mg_state="cooldown",
        name_cmd=name_cmd,
        use_cuda=False,
        stage_operations_trace_dict=stage_operations_trace,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=0,
        simu_start=True,
        trace_start=0,
        current_iter=0,
        args=args,
    )


def test_scaling_overlap_trace_emits_launch_only_without_collective():
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
    )
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    with mock.patch.object(torch.distributed, "get_rank", return_value=0):
        bucket = Bucket(
            ddp_config=ddp_config,
            params=[param],
            param_data=None,
            grad_data=torch.ones(4, dtype=torch.float32),
            offset=0,
            numel_unpadded=4,
            data_parallel_group=object(),
            data_parallel_world_size=2,
            gradient_scaling_factor=1.0,
            buffer_id=9,
            bucket_id=1,
        )

    stage_operations_trace = {}
    micro_batch_ids = {"backward_step": -1, "dp_allreduce": -1}
    args = _build_args()

    def _unexpected_collective(*args, **kwargs):
        raise AssertionError("scaling mode must not call real all_reduce")

    with mock.patch.object(torch.distributed, "all_reduce", side_effect=_unexpected_collective):
        backward_cmd = _build_cmd("backward_step", stage_operations_trace, micro_batch_ids, args)
        CMD.set_current_cmd(backward_cmd)
        with backward_cmd:
            bucket.register_grad_ready(param)

        dp_cmd = _build_cmd("dp_allreduce", stage_operations_trace, micro_batch_ids, args)
        CMD.set_current_cmd(dp_cmd)
        with dp_cmd:
            bucket.finish_grad_sync()

    event_lines = [line for line in stage_operations_trace[0] if ":ddp_grad_comm(" in line]
    assert len(event_lines) == 1
    line = event_lines[0]
    assert "timing_domain=metadata_only" in line
    assert "metadata_only=True" in line
    assert "status=launch_only" in line
    assert "completion_observed_timestamp_ms=None" in line
    assert "wait_cmd_uid=None" in line


def test_scaling_overlap_trace_skips_warmup_cmds():
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
    )
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    with mock.patch.object(torch.distributed, "get_rank", return_value=0):
        bucket = Bucket(
            ddp_config=ddp_config,
            params=[param],
            param_data=None,
            grad_data=torch.ones(4, dtype=torch.float32),
            offset=0,
            numel_unpadded=4,
            data_parallel_group=object(),
            data_parallel_world_size=2,
            gradient_scaling_factor=1.0,
            buffer_id=9,
            bucket_id=1,
        )

    stage_operations_trace = {}
    micro_batch_ids = {"loss_func": -1}
    args = _build_args()
    warmup_cmd = CMD(
        rank_id=0,
        mg_state="warmup",
        name_cmd="loss_func",
        use_cuda=False,
        stage_operations_trace_dict=stage_operations_trace,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=0,
        simu_start=False,
        trace_start=2,
        current_iter=1,
        args=args,
    )

    CMD.set_current_cmd(warmup_cmd)
    with warmup_cmd:
        bucket.register_grad_ready(param)

    assert stage_operations_trace == {}
