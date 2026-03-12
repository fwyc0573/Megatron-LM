from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import Bucket
from megatron.profiler.cmd import CMD


class _DummyFuture:
    def then(self, callback):
        callback(self)
        return self


class _DummyWork:
    def __init__(self):
        self.wait_called = False

    def get_future(self):
        return _DummyFuture()

    def wait(self):
        self.wait_called = True
        return True


class _NoFutureWork:
    def wait(self):
        return True


def _build_args(is_scaling_mode=False):
    return SimpleNamespace(
        trace_subop_sync_mode="global",
        trace_cmd_sync_mode="global",
        trace_kernel_ground_truth=False,
        trace_kernel_ground_truth_phase=False,
        trace_ddp_grad_overlap=True,
        is_scaling_mode=is_scaling_mode,
        scaling_trace_metadata_comm_duration=False,
    )


def _build_cmd(name_cmd, stage_operations_trace, micro_batch_ids, args, op_semantics=None):
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
        op_semantics=op_semantics,
    )


def _build_bucket():
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
            buffer_id=7,
            bucket_id=3,
        )
    return bucket, param


def test_bucket_overlap_trace_records_launch_completion_and_wait():
    stage_operations_trace = {}
    micro_batch_ids = {"backward_step": -1, "dp_allreduce": -1}
    args = _build_args(is_scaling_mode=False)
    bucket, param = _build_bucket()

    with mock.patch.object(torch.distributed, "all_reduce", return_value=_DummyWork()), mock.patch.object(
        torch.cuda, "synchronize"
    ):
        backward_cmd = _build_cmd("backward_step", stage_operations_trace, micro_batch_ids, args)
        CMD.set_current_cmd(backward_cmd)
        with backward_cmd:
            bucket.register_grad_ready(param)

        dp_cmd = _build_cmd(
            "dp_allreduce",
            stage_operations_trace,
            micro_batch_ids,
            args,
            op_semantics="wait_flush_only",
        )
        CMD.set_current_cmd(dp_cmd)
        with dp_cmd:
            bucket.finish_grad_sync()

    event_lines = [line for line in stage_operations_trace[0] if ":ddp_grad_comm(" in line]
    assert len(event_lines) == 1
    line = event_lines[0]
    assert "buffer_id=7" in line
    assert "bucket_id=3" in line
    assert "launch_source=param_hook" in line
    assert "trigger_cmd_uid=" in line
    assert "wait_cmd_uid=" in line
    assert "completion_source=future_callback" in line
    assert "status=completed" in line
    assert "metadata_only=False" in line


def test_bucket_overlap_trace_requires_future_support():
    stage_operations_trace = {}
    micro_batch_ids = {"backward_step": -1}
    args = _build_args(is_scaling_mode=False)
    bucket, param = _build_bucket()

    with mock.patch.object(torch.distributed, "all_reduce", return_value=_NoFutureWork()), mock.patch.object(
        torch.cuda, "synchronize"
    ):
        backward_cmd = _build_cmd("backward_step", stage_operations_trace, micro_batch_ids, args)
        CMD.set_current_cmd(backward_cmd)
        with pytest.raises(RuntimeError, match="Work.get_future"):
            with backward_cmd:
                bucket.register_grad_ready(param)
    assert stage_operations_trace[0][0].startswith("rank:0:backward_step(")
