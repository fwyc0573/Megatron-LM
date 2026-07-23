"""End-to-end CLI coverage for the SC26 AE rank0 report contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = PROJECT_ROOT / "src" / "scheduler" / "mg_scheduling" / "mg_test.py"
SIMULATOR = PROJECT_ROOT / "simu_main.py"
DATABASE_DIR = (
    PROJECT_ROOT
    / "simulation_inputs"
    / "megatron_operation_log"
    / "2pp_2_2_tiny_llama"
    / "database_profile"
)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SIMULATOR_HARDWARE_TYPE"] = "H800_SXM"
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _generate_schedule(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            sys.executable,
            "-B",
            str(SCHEDULER),
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "2",
            "--expert-model-parallel-size",
            "1",
            "--num-experts",
            "1",
            "--local-size",
            "8",
            "--world-size",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "8",
            "--seq-length",
            "512",
            "--hidden-size",
            "128",
            "--model-size",
            "qwen3_a30b",
            "--bf16",
            "--output-dir",
            str(output_dir),
            "--train-iters",
            "1",
            "--trace-start",
            "0",
        ]
    )


def _simulator_command(schedule_dir: Path, report_output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(SIMULATOR),
        "--framework",
        "megatron-lm",
        "--mode",
        "simulate",
        "--database-dir",
        str(DATABASE_DIR),
        "--schedule-dir",
        str(schedule_dir),
        "--world-size",
        "8",
        "--local-size",
        "8",
        "--pp-size",
        "2",
        "--tp-size",
        "1",
        "--exp-size",
        "1",
        "--strategy",
        "1F1B-none_interleaved",
        "--cc-backend",
        "analytical",
        "--no-visualize",
        "--report-output-dir",
        str(report_output_dir),
        "--report-model",
        "qwen3_a30b",
        "--artifact-source",
        "prebaked",
    ]


def _assert_generated_schedule(schedule_dir: Path) -> None:
    assert {path.name for path in schedule_dir.glob("*.txt")} == {
        "stage0_scheduling_plan.txt",
        "stage1_scheduling_plan.txt",
    }
    for path in schedule_dir.glob("*.txt"):
        schedule = path.read_text(encoding="utf-8")
        assert "torch.bfloat16" in schedule
        assert schedule.count(":forward_step(") == 2
        assert schedule.count(":backward_step(") == 2
        assert schedule.count(":optimizer_step(") == 1


def test_generated_pp2_schedule_produces_rank0_json_and_markdown(
    tmp_path: Path,
) -> None:
    schedule_dir = tmp_path / "schedule"
    schedule_result = _generate_schedule(schedule_dir)
    assert schedule_result.returncode == 0, schedule_result.stderr
    _assert_generated_schedule(schedule_dir)

    report_dir = tmp_path / "report"
    result = _run(_simulator_command(schedule_dir, report_dir))
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert set(report) == {
        "schema_version",
        "model",
        "artifact_source",
        "rank_id",
        "rank0_step_time_ms",
        "rank0_forward_step_duration_sum_ms",
        "rank0_backward_step_duration_sum_ms",
        "rank0_optimizer_step_duration_sum_ms",
        "rank0_comp_plus_comm_diagnostic_ms",
        "simulator_load_time_s",
        "simulator_execution_time_s",
        "simulator_wall_clock_s",
    }
    assert report["schema_version"] == "sc26-ae-rank0-report-v1"
    assert report["model"] == "qwen3_a30b"
    assert report["artifact_source"] == "prebaked"
    assert report["rank_id"] == 0
    for field in (
        "rank0_step_time_ms",
        "rank0_forward_step_duration_sum_ms",
        "rank0_backward_step_duration_sum_ms",
        "rank0_optimizer_step_duration_sum_ms",
        "rank0_comp_plus_comm_diagnostic_ms",
        "simulator_wall_clock_s",
    ):
        assert report[field] > 0
    assert report["simulator_wall_clock_s"] == round(
        report["simulator_load_time_s"] + report["simulator_execution_time_s"],
        6,
    )

    markdown = (report_dir / "report.md").read_text(encoding="utf-8")
    for key, value in report.items():
        assert f"| `{key}` | `{value}` |" in markdown


@pytest.mark.parametrize(
    "report_flags",
    [
        ["--report-output-dir", "report-only"],
        ["--report-model", "qwen3_a30b"],
        ["--artifact-source", "prebaked"],
    ],
)
def test_cli_rejects_incomplete_report_flag_sets(
    tmp_path: Path,
    report_flags: list[str],
) -> None:
    schedule_dir = tmp_path / "schedule"
    command = _simulator_command(schedule_dir, tmp_path / "unused")
    command = command[: command.index("--report-output-dir")] + report_flags
    result = _run(command)
    assert result.returncode == 1
    assert "must be provided together" in result.stdout


def test_cli_rejects_non_eight_analytical_local_size_before_engine(
    tmp_path: Path,
) -> None:
    command = _simulator_command(tmp_path / "schedule", tmp_path / "report")
    local_size_index = command.index("--local-size") + 1
    command[local_size_index] = "4"
    result = _run(command)
    assert result.returncode == 1
    assert "Analytical backend local-size mismatch" in result.stdout
    assert not (tmp_path / "report").exists()


def test_cli_rejects_missing_optimizer_in_generated_schedule(tmp_path: Path) -> None:
    base_schedule_dir = tmp_path / "base-schedule"
    schedule_result = _generate_schedule(base_schedule_dir)
    assert schedule_result.returncode == 0, schedule_result.stderr

    missing_optimizer_dir = tmp_path / "missing-optimizer"
    missing_optimizer_dir.mkdir()
    for source in base_schedule_dir.glob("*.txt"):
        retained_lines = [
            line
            for line in source.read_text(encoding="utf-8").splitlines()
            if ":optimizer_step(" not in line
        ]
        (missing_optimizer_dir / source.name).write_text(
            "\n".join(retained_lines) + "\n",
            encoding="utf-8",
        )

    result = _run(
        _simulator_command(missing_optimizer_dir, tmp_path / "missing-report")
    )
    assert result.returncode == 1
    assert "missing exact operation optimizer_step" in result.stdout
    assert not (tmp_path / "missing-report" / "report.json").exists()


def test_cli_rejects_file_as_report_output_directory(tmp_path: Path) -> None:
    schedule_dir = tmp_path / "schedule"
    schedule_result = _generate_schedule(schedule_dir)
    assert schedule_result.returncode == 0, schedule_result.stderr

    occupied_path = tmp_path / "occupied"
    occupied_path.write_text("not a directory\n", encoding="utf-8")
    result = _run(_simulator_command(schedule_dir, occupied_path))
    assert result.returncode == 1
    assert "File exists" in result.stdout or "Not a directory" in result.stdout
    assert occupied_path.read_text(encoding="utf-8") == "not a directory\n"
