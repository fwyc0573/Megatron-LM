"""Unit tests for trace-shaped PP schedule generation."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_prep.schedule.build_trace_shaped_pp_schedule import (  # noqa: E402
    build_trace_shaped_schedule,
    main as build_trace_shaped_schedule_main,
)
from tools.data_prep.common.megatron_trace_utils import parse_trace_line  # noqa: E402


def _write_trace_file(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage0_lines(rank_id: int) -> list[str]:
    return [
        f"rank:{rank_id}:get_batch(stage_id=0,batch_id=0,mg_state=warmup,duration=0.57,description=None,group_kind=None,cmd_uid=cmd-get-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=10.0,sub_operations=[])",
        f"rank:{rank_id}:forward_step(stage_id=0,batch_id=0,mg_state=warmup,duration=26.34,description=None,group_kind=None,cmd_uid=cmd-fwd-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=36.34,sub_operations=[])",
        f"rank:{rank_id}:backward_step(stage_id=0,batch_id=0,mg_state=cooldown,duration=42.44,description=None,group_kind=None,cmd_uid=cmd-bwd-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=78.78,sub_operations=[])",
        f"rank:{rank_id}:ddp_grad_comm(comm_uid=ddp-{rank_id},iter=2,stage_id=0,mg_state=cooldown,group_kind=dp,comm_func=allreduce,buffer_id=1,bucket_id=1,bucket_offset=0,bucket_numel=4,bucket_numel_unpadded=4,param_count=1,trigger_cmd_uid=cmd-bwd-{rank_id},trigger_op=backward_step,trigger_batch_id=0,trigger_timestamp_ms=36.34,launch_timestamp_ms=40.0,launch_source=param_hook,timing_domain=metadata_only,metadata_only=True,status=launch_only,completion_observed_timestamp_ms=None,completion_source=None,wait_cmd_uid=cmd-wait-{rank_id},wait_start_timestamp_ms=None,wait_end_timestamp_ms=None,logical_stream_role=dp_comm,grad_dtype=torch.float16,data_parallel_world_size=2,duration=0.0,timestamp=40.0,sub_operations=[])",
        f"rank:{rank_id}:dp_allreduce(stage_id=0,batch_id=0,mg_state=finalize,duration=0.09,description=model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas,group_kind=dp,cmd_uid=cmd-wait-{rank_id},op_semantics=metadata_placeholder,finalize_base_duration_ms=0.09,input__shape=None,input__dtype=None,timestamp=79.0,sub_operations=[])",
        f"rank:{rank_id}:ep_allreduce(stage_id=0,batch_id=0,mg_state=finalize,duration=0.0,description=_allreduce_word_embedding_grads,group_kind=ep,cmd_uid=cmd-ep-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=79.1,sub_operations=[])",
        f"rank:{rank_id}:optimizer_step(stage_id=0,batch_id=0,mg_state=finalize,duration=1.23,description=None,group_kind=None,cmd_uid=cmd-opt-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=80.33,sub_operations=[])",
    ]


def _stage1_lines(rank_id: int) -> list[str]:
    return [
        f"rank:{rank_id}:get_batch(stage_id=1,batch_id=0,mg_state=steady,duration=0.68,description=None,group_kind=None,cmd_uid=cmd-get-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=11.0,sub_operations=[])",
        f"rank:{rank_id}:forward_step(stage_id=1,batch_id=0,mg_state=steady,duration=30.22,description=None,group_kind=None,cmd_uid=cmd-fwd-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=41.22,sub_operations=[])",
        f"rank:{rank_id}:loss_func(stage_id=1,batch_id=0,mg_state=steady,duration=0.01,description=None,group_kind=None,cmd_uid=cmd-loss-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=41.23,sub_operations=[])",
        f"rank:{rank_id}:backward_step(stage_id=1,batch_id=0,mg_state=steady,duration=44.05,description=None,group_kind=None,cmd_uid=cmd-bwd-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=85.28,sub_operations=[])",
        f"rank:{rank_id}:ddp_grad_comm(comm_uid=ddp-{rank_id},iter=2,stage_id=1,mg_state=steady,group_kind=dp,comm_func=allreduce,buffer_id=1,bucket_id=1,bucket_offset=0,bucket_numel=4,bucket_numel_unpadded=4,param_count=1,trigger_cmd_uid=cmd-bwd-{rank_id},trigger_op=backward_step,trigger_batch_id=0,trigger_timestamp_ms=41.23,launch_timestamp_ms=45.0,launch_source=param_hook,timing_domain=metadata_only,metadata_only=True,status=launch_only,completion_observed_timestamp_ms=None,completion_source=None,wait_cmd_uid=cmd-wait-{rank_id},wait_start_timestamp_ms=None,wait_end_timestamp_ms=None,logical_stream_role=dp_comm,grad_dtype=torch.float16,data_parallel_world_size=2,duration=0.0,timestamp=45.0,sub_operations=[])",
        f"rank:{rank_id}:dp_allreduce(stage_id=1,batch_id=0,mg_state=finalize,duration=0.11,description=model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas,group_kind=dp,cmd_uid=cmd-wait-{rank_id},op_semantics=metadata_placeholder,finalize_base_duration_ms=0.11,input__shape=None,input__dtype=None,timestamp=85.39,sub_operations=[])",
        f"rank:{rank_id}:ep_allreduce(stage_id=1,batch_id=0,mg_state=finalize,duration=0.0,description=_allreduce_word_embedding_grads,group_kind=ep,cmd_uid=cmd-ep-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=85.4,sub_operations=[])",
        f"rank:{rank_id}:optimizer_step(stage_id=1,batch_id=0,mg_state=finalize,duration=1.11,description=None,group_kind=None,cmd_uid=cmd-opt-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=86.5,sub_operations=[])",
    ]


def test_build_trace_shaped_schedule_generates_pp2_manual_schedule(tmp_path: pathlib.Path) -> None:
    trace_dir = tmp_path / "trace"
    output_dir = tmp_path / "schedule"
    trace_dir.mkdir()
    _write_trace_file(trace_dir / "rank0.txt", _stage0_lines(0))
    _write_trace_file(trace_dir / "rank1.txt", _stage0_lines(1))
    _write_trace_file(trace_dir / "rank2.txt", _stage1_lines(2))
    _write_trace_file(trace_dir / "rank3.txt", _stage1_lines(3))

    written_files = build_trace_shaped_schedule(
        trace_dir=trace_dir,
        output_dir=output_dir,
        pp_size=2,
        seq_length=256,
        micro_batch_size=1,
        hidden_size=4096,
        pipeline_dtype="torch.float16",
    )

    assert [path.name for path in written_files] == [
        "stage0_trace_shaped_scheduling_plan.txt",
        "stage1_trace_shaped_scheduling_plan.txt",
    ]

    stage0_lines = output_dir.joinpath("stage0_trace_shaped_scheduling_plan.txt").read_text(encoding="utf-8").splitlines()
    stage1_lines = output_dir.joinpath("stage1_trace_shaped_scheduling_plan.txt").read_text(encoding="utf-8").splitlines()

    assert [parse_trace_line(line.replace("stage:", "rank:", 1))[1] for line in stage0_lines] == [
        "get_batch",
        "forward_step",
        "send_forward",
        "recv_backward",
        "backward_step",
        "dp_allreduce",
        "ep_allreduce",
        "optimizer_step",
    ]
    assert [parse_trace_line(line.replace("stage:", "rank:", 1))[1] for line in stage1_lines] == [
        "recv_forward",
        "get_batch",
        "forward_step",
        "loss_func",
        "backward_step",
        "send_backward",
        "dp_allreduce",
        "ep_allreduce",
        "optimizer_step",
    ]

    _, _, stage0_send_forward = parse_trace_line(stage0_lines[2].replace("stage:", "rank:", 1))
    _, _, stage1_recv_forward = parse_trace_line(stage1_lines[0].replace("stage:", "rank:", 1))
    assert stage0_send_forward["group_kind"] == "pp"
    assert stage0_send_forward["input__shape"] == [256, 1, 4096]
    assert stage0_send_forward["input__dtype"] == "torch.float16"
    assert stage1_recv_forward["mg_state"] == "help"

    _, _, stage0_wait = parse_trace_line(stage0_lines[5].replace("stage:", "rank:", 1))
    assert stage0_wait["description"] == "model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas"
    assert stage0_wait["op_semantics"] == "metadata_placeholder"
    assert stage0_wait["finalize_base_duration_ms"] == pytest.approx(0.09)


def test_build_trace_shaped_schedule_fails_fast_when_stage_trace_is_incomplete(tmp_path: pathlib.Path) -> None:
    trace_dir = tmp_path / "trace"
    output_dir = tmp_path / "schedule"
    trace_dir.mkdir()
    incomplete_stage0 = _stage0_lines(0)[:-3]
    _write_trace_file(trace_dir / "rank0.txt", incomplete_stage0)
    _write_trace_file(trace_dir / "rank2.txt", _stage1_lines(2))

    with pytest.raises(ValueError, match="missing required top-level op"):
        build_trace_shaped_schedule(
            trace_dir=trace_dir,
            output_dir=output_dir,
            pp_size=2,
            seq_length=256,
            micro_batch_size=1,
            hidden_size=4096,
            pipeline_dtype="torch.float16",
        )


def test_build_trace_shaped_schedule_cli_writes_schedule_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_dir = tmp_path / "trace"
    output_dir = tmp_path / "schedule"
    trace_dir.mkdir()
    _write_trace_file(trace_dir / "rank0.txt", _stage0_lines(0))
    _write_trace_file(trace_dir / "rank1.txt", _stage0_lines(1))
    _write_trace_file(trace_dir / "rank2.txt", _stage1_lines(2))
    _write_trace_file(trace_dir / "rank3.txt", _stage1_lines(3))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_trace_shaped_pp_schedule.py",
            "--trace-dir",
            str(trace_dir),
            "--output-dir",
            str(output_dir),
            "--pp-size",
            "2",
            "--seq-length",
            "256",
            "--micro-batch-size",
            "1",
            "--hidden-size",
            "4096",
            "--pipeline-dtype",
            "torch.float16",
        ],
    )

    assert build_trace_shaped_schedule_main() == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [pathlib.Path(line).name for line in out_lines] == [
        "stage0_trace_shaped_scheduling_plan.txt",
        "stage1_trace_shaped_scheduling_plan.txt",
    ]
