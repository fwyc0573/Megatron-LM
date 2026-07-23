"""Unit tests for explicit EP/EXP communication semantics in simulator engine."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import (
    MODE_SIMULATE,
    Operation,
    SimulatorEngine,
    TimelinesManager,
    process_mg_files,
)


def test_comm_semantics_ep_allreduce_maps_to_ep() -> None:
    manager = TimelinesManager.__new__(TimelinesManager)
    operation = Operation(name="ep_allreduce", op_kind="comm", group_kind=None, mg_state="steady")

    comm_kind, parallel_kind = manager._get_comm_operation_kind_and_parallel_dimension(operation)

    assert comm_kind == "allreduce"
    assert parallel_kind == "ep"
    assert operation.group_kind == "ep"


def test_comm_semantics_exp_collectives_map_to_exp_domain() -> None:
    manager = TimelinesManager.__new__(TimelinesManager)

    all_to_all = Operation(name="exp_all_to_all", op_kind="comm", group_kind=None, mg_state="steady")
    allgather = Operation(name="exp_allgather", op_kind="comm", group_kind=None, mg_state="steady")
    exp_dp = Operation(name="exp_dp_allreduce", op_kind="comm", group_kind=None, mg_state="steady")
    ep_dp_alias = Operation(name="ep_dp_allreduce", op_kind="comm", group_kind=None, mg_state="steady")

    assert manager._get_comm_operation_kind_and_parallel_dimension(all_to_all) == ("all_to_all", "exp")
    assert manager._get_comm_operation_kind_and_parallel_dimension(allgather) == ("allgather", "exp")
    assert manager._get_comm_operation_kind_and_parallel_dimension(exp_dp) == ("allreduce", "exp_dp")
    assert manager._get_comm_operation_kind_and_parallel_dimension(ep_dp_alias) == ("allreduce", "exp_dp")


def test_comm_semantics_raise_on_group_kind_mismatch() -> None:
    manager = TimelinesManager.__new__(TimelinesManager)
    operation = Operation(name="ep_allreduce", op_kind="comm", group_kind="exp", mg_state="steady")

    with pytest.raises(ValueError, match="expects group_kind `ep`"):
        manager._get_comm_operation_kind_and_parallel_dimension(operation)


def test_parse_and_validate_profile_filename_parallel_tokens() -> None:
    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.mpu = SimpleNamespace(exp_size=4, ep_size=2)

    parsed = engine.parse_profile_filename_parallel_tokens("wd8_tp1_pp2_exp4_ep2_rank0_20250101.txt")
    assert parsed == {"exp": 4, "ep": 2}

    engine.validate_profile_filename_parallel_tokens("wd8_tp1_pp2_exp4_ep2_rank0_20250101.txt", parsed)

    with pytest.raises(ValueError, match="expects exp_size=4"):
        engine.validate_profile_filename_parallel_tokens(
            "wd8_tp1_pp2_exp2_ep2_rank0_20250101.txt",
            {"exp": 2, "ep": 2},
        )


def test_validate_database_profile_filename_semantics_fail_fast_on_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    database_dir = tmp_path / "database_profile"
    database_dir.mkdir()
    (database_dir / "wd8_tp1_pp2_exp2_rank0_20250101.txt").write_text("dummy\\n", encoding="utf-8")

    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.torchgraph_filepath = str(database_dir)
    engine.mpu = SimpleNamespace(exp_size=4, ep_size=2)

    with pytest.raises(ValueError, match="semantic mismatch"):
        engine._validate_database_profile_filename_semantics()


@pytest.mark.parametrize(
    "operation_line, mpu_info, expected_error",
    [
        (
            "stage:0:ep_allreduce(batch_id=0, mg_state=steady, input__shape=[1024], input__dtype=torch.float16)",
            SimpleNamespace(
                world_size=8,
                pp_size=2,
                dp_size=4,
                tp_size=1,
                ep_size=0,
                exp_size=2,
                dp_modulo_exp_groups=[[0, 2]],
            ),
            "Invalid comm_group_size=0 inferred for operation ep_allreduce",
        ),
        (
            "stage:0:exp_dp_allreduce(batch_id=0, mg_state=steady, input__shape=[1024], input__dtype=torch.float16)",
            SimpleNamespace(
                world_size=8,
                pp_size=2,
                dp_size=4,
                tp_size=1,
                ep_size=2,
                exp_size=2,
                dp_modulo_exp_groups=None,
            ),
            "dp_modulo_exp_groups is required",
        ),
    ],
)
def test_process_mg_files_schedule_fail_fast_for_invalid_group_size(
    tmp_path: pathlib.Path,
    operation_line: str,
    mpu_info: SimpleNamespace,
    expected_error: str,
) -> None:
    schedule_dir = tmp_path / "schedule"
    schedule_dir.mkdir()
    (schedule_dir / "stage0.txt").write_text(operation_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        process_mg_files(
            my_filepath=str(schedule_dir),
            rank_instances_dict={0: object()},
            running_mode=MODE_SIMULATE,
            torch_graph_stage_op_dict={},
            is_trace=False,
            mpu=mpu_info,
            num_files=1,
        )
