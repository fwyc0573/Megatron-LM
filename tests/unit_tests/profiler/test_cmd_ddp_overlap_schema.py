import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_cmd_module():
    cmd_path = (
        Path(__file__).resolve().parents[3] / "megatron" / "profiler" / "cmd.py"
    )
    spec = importlib.util.spec_from_file_location("test_cmd_module_ddp_overlap", cmd_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {cmd_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmd_module = _load_cmd_module()
CMD = cmd_module.CMD


class _DummyEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 0.25


def _build_args():
    return SimpleNamespace(
        trace_subop_sync_mode="global",
        trace_cmd_sync_mode="global",
        trace_kernel_ground_truth=False,
        trace_kernel_ground_truth_phase=False,
        is_scaling_mode=False,
        trace_ddp_grad_overlap=True,
    )


def test_cmd_serializes_cmd_uid_and_op_semantics():
    stage_operations_trace = {}
    micro_batch_ids = {"dp_allreduce": -1}
    cmd = CMD(
        rank_id=0,
        mg_state="finalize",
        name_cmd="dp_allreduce",
        use_cuda=True,
        stage_operations_trace_dict=stage_operations_trace,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=0,
        simu_start=True,
        trace_start=0,
        current_iter=0,
        description="test",
        args=_build_args(),
        op_semantics="wait_flush_only",
    )
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ):
        with cmd:
            pass

    record = stage_operations_trace[0][0]
    assert "cmd_uid=" in record
    assert "op_semantics=wait_flush_only" in record
    assert record.startswith("rank:0:dp_allreduce(")
    assert record.endswith("sub_operations=[])")


def test_emit_trace_event_writes_same_file_shape():
    trace = {}
    CMD.emit_trace_event(
        rank_id=0,
        stage_operations_trace_dict=trace,
        event_name="ddp_grad_comm",
        fields={
            "comm_uid": "comm-1",
            "iter": 0,
            "stage_id": 0,
            "mg_state": "cooldown",
            "group_kind": "dp",
            "comm_func": "allreduce",
            "duration": 1.5,
            "timestamp": 3.5,
        },
    )

    line = trace[0][0]
    assert line.startswith("rank:0:ddp_grad_comm(")
    assert "comm_uid=comm-1" in line
    assert "duration=1.5" in line
    assert line.endswith("sub_operations=[])")


def test_cmd_serializes_finalize_base_duration_ms():
    stage_operations_trace = {}
    micro_batch_ids = {"dp_allreduce": -1}
    cmd = CMD(
        rank_id=0,
        mg_state="finalize",
        name_cmd="dp_allreduce",
        use_cuda=True,
        stage_operations_trace_dict=stage_operations_trace,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=0,
        simu_start=True,
        trace_start=0,
        current_iter=0,
        description="test",
        args=_build_args(),
        op_semantics="metadata_placeholder",
        finalize_base_duration_ms=15.25,
    )
    cmd.no_trace_update(15.25, 445.0)

    record = stage_operations_trace[0][0]
    assert "duration=15.25" in record
    assert "timestamp=445.0" in record
    assert "finalize_base_duration_ms=15.25" in record
    assert record.endswith("sub_operations=[])")
