import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.training import training as training_module



def _load_cmd_module():
    cmd_path = (
        Path(__file__).resolve().parents[2] / "megatron" / "profiler" / "cmd.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_cmd_module_scaling_finalize_base", cmd_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {cmd_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmd_module = _load_cmd_module()
CMD = cmd_module.CMD



def _build_args():
    return SimpleNamespace(trace_ddp_grad_overlap=True)



def _build_cmd(stage_operations_trace, micro_batch_ids):
    return CMD(
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
        description="simulation",
        group_kind="dp",
        args=_build_args(),
        op_semantics="metadata_placeholder",
        input__shape=[1024],
        input__dtype="torch.float32",
    )



def test_emit_scaling_dp_allreduce_placeholder_inserts_explicit_finalize_base():
    stage_operations_trace = {
        0: [
            "rank:0:backward_step(stage_id=0,batch_id=0,mg_state=steady,duration=40.0,description=simulation,group_kind=None,cmd_uid=cmd-bwd,op_semantics=None,input__shape=None,input__dtype=None,timestamp=430.0,sub_operations=[])",
            "rank:0:ep_allreduce(stage_id=0,batch_id=0,mg_state=finalize,duration=0.0,description=simulation,group_kind=ep,cmd_uid=cmd-ep,op_semantics=None,input__shape=None,input__dtype=None,timestamp=0.0,sub_operations=[])",
        ]
    }
    micro_batch_ids = {"dp_allreduce": -1}
    cmd = _build_cmd(stage_operations_trace, micro_batch_ids)

    duration_ms = training_module._emit_scaling_dp_allreduce_placeholder(
        cmd=cmd,
        insert_index=1,
        start_timestamp_ms=430.0,
        end_timestamp_ms=445.5,
    )

    assert duration_ms == pytest.approx(15.5)
    assert len(stage_operations_trace[0]) == 3
    record = stage_operations_trace[0][1]
    assert record.startswith("rank:0:dp_allreduce(")
    assert "duration=15.5" in record
    assert "timestamp=445.5" in record
    assert "finalize_base_duration_ms=15.5" in record
    assert "op_semantics=metadata_placeholder" in record



def test_emit_scaling_dp_allreduce_placeholder_rejects_negative_duration():
    stage_operations_trace = {0: []}
    micro_batch_ids = {"dp_allreduce": -1}
    cmd = _build_cmd(stage_operations_trace, micro_batch_ids)

    with pytest.raises(RuntimeError, match="negative finalize base duration"):
        training_module._emit_scaling_dp_allreduce_placeholder(
            cmd=cmd,
            insert_index=0,
            start_timestamp_ms=445.5,
            end_timestamp_ms=445.0,
        )


def test_emit_scaling_dp_allreduce_placeholder_requires_trace_dict():
    micro_batch_ids = {"dp_allreduce": -1}
    cmd = CMD(
        rank_id=0,
        mg_state="finalize",
        name_cmd="dp_allreduce",
        use_cuda=True,
        stage_operations_trace_dict=None,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=0,
        simu_start=True,
        trace_start=0,
        current_iter=0,
        description="simulation",
        group_kind="dp",
        args=_build_args(),
        op_semantics="metadata_placeholder",
        input__shape=[1024],
        input__dtype="torch.float32",
    )

    with pytest.raises(RuntimeError, match="stage_operations_trace_dict"):
        training_module._emit_scaling_dp_allreduce_placeholder(
            cmd=cmd,
            insert_index=0,
            start_timestamp_ms=430.0,
            end_timestamp_ms=445.0,
        )



def test_emit_scaling_dp_allreduce_placeholder_requires_dp_allreduce_cmd():
    stage_operations_trace = {0: []}
    micro_batch_ids = {"optimizer_step": -1}
    cmd = CMD(
        rank_id=0,
        mg_state="finalize",
        name_cmd="optimizer_step",
        use_cuda=True,
        stage_operations_trace_dict=stage_operations_trace,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=0,
        simu_start=True,
        trace_start=0,
        current_iter=0,
        description="simulation",
        group_kind=None,
        args=_build_args(),
        op_semantics=None,
    )

    with pytest.raises(RuntimeError, match='name_cmd=dp_allreduce'):
        training_module._emit_scaling_dp_allreduce_placeholder(
            cmd=cmd,
            insert_index=0,
            start_timestamp_ms=430.0,
            end_timestamp_ms=445.0,
        )



def test_emit_scaling_dp_allreduce_placeholder_requires_metadata_placeholder_semantics():
    stage_operations_trace = {0: []}
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
        description="simulation",
        group_kind="dp",
        args=_build_args(),
        op_semantics="wait_flush_only",
        input__shape=[1024],
        input__dtype="torch.float32",
    )

    with pytest.raises(RuntimeError, match='op_semantics=metadata_placeholder'):
        training_module._emit_scaling_dp_allreduce_placeholder(
            cmd=cmd,
            insert_index=0,
            start_timestamp_ms=430.0,
            end_timestamp_ms=445.0,
        )
