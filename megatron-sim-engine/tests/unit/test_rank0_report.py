from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Keep the test import contract independent of the caller's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simu_main


def _op(name: str, join_time: float, finish_time: float) -> SimpleNamespace:
    return SimpleNamespace(name=name, join_time=join_time, finish_time=finish_time)


def _engine(
    *,
    comp: list[SimpleNamespace] | None = None,
    comm: list[SimpleNamespace] | None = None,
    final: list[SimpleNamespace] | None = None,
    include_rank0: bool = True,
) -> SimpleNamespace:
    default_comp = [
        _op("forward_step", 10.0, 12.0),
        _op("forward_step_extra", 12.0, 19.0),
        _op("backward_step", 14.0, 17.0),
        _op("optimizer_step", 19.0, 20.0),
    ]
    default_comm = [_op("dp_allreduce", 12.0, 16.0)]
    timeline = SimpleNamespace(
        comp_timeline=default_comp if comp is None else comp,
        comm_timeline=default_comm if comm is None else comm,
        final_merge_timeline=(
            default_comp + default_comm if final is None else final
        ),
    )
    timelines = {0: timeline} if include_rank0 else {1: timeline}
    return SimpleNamespace(
        timeline_manager=SimpleNamespace(stages_timeline_process_dict=timelines)
    )


def test_build_rank0_report_uses_timeline_span_and_exact_operation_names() -> None:
    report = simu_main.build_rank0_report(
        _engine(),
        model="qwen3_a30b",
        artifact_source="fresh",
        load_time_s=0.123456789,
        execution_time_s=1.00000049,
    )

    assert report == {
        "schema_version": "sc26-ae-rank0-report-v1",
        "model": "qwen3_a30b",
        "artifact_source": "fresh",
        "rank_id": 0,
        "rank0_step_time_ms": 10.0,
        "rank0_forward_step_duration_sum_ms": 2.0,
        "rank0_backward_step_duration_sum_ms": 3.0,
        "rank0_optimizer_step_duration_sum_ms": 1.0,
        "rank0_comp_plus_comm_diagnostic_ms": 17.0,
        "simulator_load_time_s": 0.123457,
        "simulator_execution_time_s": 1.0,
        "simulator_wall_clock_s": 1.123457,
    }


@pytest.mark.parametrize("missing_name", ["forward_step", "backward_step", "optimizer_step"])
def test_build_rank0_report_rejects_each_missing_target_operation(
    missing_name: str,
) -> None:
    comp = [
        op
        for op in _engine().timeline_manager.stages_timeline_process_dict[0].comp_timeline
        if op.name != missing_name
    ]
    with pytest.raises(ValueError, match=missing_name):
        simu_main.build_rank0_report(_engine(comp=comp), "gpt175b", "prebaked", 1.0, 1.0)


def test_build_rank0_report_rejects_missing_rank0_and_empty_timeline() -> None:
    with pytest.raises(ValueError, match="rank0"):
        simu_main.build_rank0_report(_engine(include_rank0=False), "dsv3", "fresh", 1.0, 1.0)

    with pytest.raises(ValueError, match="empty"):
        simu_main.build_rank0_report(
            _engine(comp=[], comm=[], final=[]), "dsv3", "fresh", 1.0, 1.0
        )


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (SimpleNamespace(name="forward_step", join_time=None, finish_time=1.0), "timestamp"),
        (_op("forward_step", math.nan, 1.0), "finite"),
        (_op("forward_step", 2.0, 1.0), "finish_time"),
    ],
)
def test_build_rank0_report_rejects_invalid_operation_times(
    op: SimpleNamespace,
    message: str,
) -> None:
    base = _engine().timeline_manager.stages_timeline_process_dict[0]
    comp = [op, *base.comp_timeline[1:]]
    final = [op, *base.final_merge_timeline[1:]]
    with pytest.raises(ValueError, match=message):
        simu_main.build_rank0_report(
            _engine(comp=comp, final=final), "qwen3_a30b", "fresh", 1.0, 1.0
        )


def test_build_rank0_report_rejects_nonpositive_span() -> None:
    comp = [
        _op("forward_step", 1.0, 1.0),
        _op("backward_step", 1.0, 1.0),
        _op("optimizer_step", 1.0, 1.0),
    ]
    with pytest.raises(ValueError, match="positive"):
        simu_main.build_rank0_report(
            _engine(comp=comp, comm=[], final=comp), "gpt175b", "fresh", 1.0, 1.0
        )


@pytest.mark.parametrize(
    ("model", "source", "load_time", "execution_time", "message"),
    [
        ("unknown", "fresh", 1.0, 1.0, "model"),
        ("gpt175b", "auto", 1.0, 1.0, "artifact_source"),
        ("gpt175b", "fresh", -1.0, 1.0, "load_time"),
        ("gpt175b", "fresh", 1.0, math.inf, "execution_time"),
    ],
)
def test_build_rank0_report_rejects_invalid_metadata(
    model: str,
    source: str,
    load_time: float,
    execution_time: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        simu_main.build_rank0_report(
            _engine(), model, source, load_time, execution_time
        )


def test_write_rank0_report_keeps_json_and_markdown_values_in_sync(tmp_path: Path) -> None:
    report = simu_main.build_rank0_report(
        _engine(), "qwen3_a30b", "prebaked", 0.25, 0.75
    )
    simu_main.write_rank0_report(report, tmp_path)

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# SC26 AE Rank0 Simulation Report\n")
    for key, value in report.items():
        assert f"| `{key}` | `{value}` |" in markdown


def test_write_rank0_report_rejects_existing_file_as_output_dir(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    report = simu_main.build_rank0_report(_engine(), "dsv3", "fresh", 1.0, 1.0)

    with pytest.raises((FileExistsError, NotADirectoryError)):
        simu_main.write_rank0_report(report, output_file)
