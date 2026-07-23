"""Unit tests for trace-shaped pp2 schedule generation."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.mg_scheduling.build_trace_shaped_pp2_schedule import (  # noqa: E402
    build_trace_shaped_pp2_schedule,
)


def _write_trace(path: pathlib.Path, rank: int, stage_id: int, ops: list[tuple[str, str]]) -> None:
    lines = []
    for index, (op_name, mg_state) in enumerate(ops):
        lines.append(
            f"rank:{rank}:{op_name}(stage_id={stage_id},batch_id=0,mg_state={mg_state},duration={10 + index}.0,description=simulation,group_kind=None,cmd_uid=cmd-{rank}-{index},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp={1000 + index}.0,sub_operations=[])"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_trace_shaped_pp2_schedule_injects_required_pp_ops(tmp_path: pathlib.Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank0.txt",
        rank=0,
        stage_id=0,
        ops=[
            ("get_batch", "warmup"),
            ("forward_step", "warmup"),
            ("backward_step", "cooldown"),
            ("dp_allreduce", "finalize"),
            ("ep_allreduce", "finalize"),
            ("optimizer_step", "finalize"),
        ],
    )
    _write_trace(
        trace_dir / "rank2.txt",
        rank=2,
        stage_id=1,
        ops=[
            ("get_batch", "steady"),
            ("forward_step", "steady"),
            ("loss_func", "steady"),
            ("backward_step", "steady"),
            ("dp_allreduce", "finalize"),
            ("ep_allreduce", "finalize"),
            ("optimizer_step", "finalize"),
        ],
    )

    output_dir = tmp_path / "schedule"
    build_trace_shaped_pp2_schedule(
        trace_dir=trace_dir,
        output_dir=output_dir,
        pp_size=2,
        seq_length=256,
        micro_batch_size=1,
        hidden_size=4096,
        pipeline_dtype="torch.float16",
    )

    stage0_lines = (output_dir / "stage0_trace_shaped_scheduling_plan.txt").read_text(encoding="utf-8").splitlines()
    stage1_lines = (output_dir / "stage1_trace_shaped_scheduling_plan.txt").read_text(encoding="utf-8").splitlines()

    assert [line.split(":", 2)[2].split("(", 1)[0] for line in stage0_lines] == [
        "get_batch",
        "forward_step",
        "send_forward",
        "recv_backward",
        "backward_step",
        "dp_allreduce",
        "ep_allreduce",
        "optimizer_step",
    ]
    assert [line.split(":", 2)[2].split("(", 1)[0] for line in stage1_lines] == [
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
    assert "mg_state=help" in stage1_lines[0]
    assert "input__shape=[256, 1, 4096]" in stage0_lines[2]
    assert "input__dtype=torch.float16" in stage0_lines[2]


def test_build_trace_shaped_pp2_schedule_rejects_inconsistent_stage_replicas(tmp_path: pathlib.Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank0.txt",
        rank=0,
        stage_id=0,
        ops=[
            ("get_batch", "warmup"),
            ("forward_step", "warmup"),
            ("backward_step", "cooldown"),
            ("dp_allreduce", "finalize"),
            ("optimizer_step", "finalize"),
        ],
    )
    _write_trace(
        trace_dir / "rank1.txt",
        rank=1,
        stage_id=0,
        ops=[
            ("get_batch", "warmup"),
            ("forward_step", "warmup"),
            ("loss_func", "warmup"),
            ("backward_step", "cooldown"),
            ("dp_allreduce", "finalize"),
            ("optimizer_step", "finalize"),
        ],
    )
    _write_trace(
        trace_dir / "rank2.txt",
        rank=2,
        stage_id=1,
        ops=[
            ("get_batch", "steady"),
            ("forward_step", "steady"),
            ("loss_func", "steady"),
            ("backward_step", "steady"),
            ("dp_allreduce", "finalize"),
            ("optimizer_step", "finalize"),
        ],
    )

    with pytest.raises(ValueError, match="Inconsistent top-level op sequence"):
        build_trace_shaped_pp2_schedule(
            trace_dir=trace_dir,
            output_dir=tmp_path / "schedule",
            pp_size=2,
            seq_length=256,
            micro_batch_size=1,
            hidden_size=4096,
            pipeline_dtype="torch.float16",
        )
