"""Unit tests for legacy-free communication metadata parsing in process_mg_profile_files."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.core.simu_engine as simu_engine


def _write_profile_file(tmp_path: pathlib.Path, filename: str, lines: list[str]) -> pathlib.Path:
    profile_dir = tmp_path / "database_profile"
    profile_dir.mkdir()
    (profile_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile_dir


def _minimal_mpu():
    return SimpleNamespace(tp_size=1, dp_size=2, exp_size=1)


def test_process_mg_profile_files_does_not_call_legacy_allreduce_estimator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = _write_profile_file(
        tmp_path=tmp_path,
        filename="stage0_rank0.txt",
        lines=[
            "stage:0:dp_allreduce(batch_id=0,mg_state=steady,duration=7.5,input__shape=[4,8],input__dtype=torch.float16,group_kind=dp)"
        ],
    )

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("legacy get_comm_op_exc_time should not be called in parser")

    monkeypatch.setattr(simu_engine, "get_comm_op_exc_time", _raise_if_called)

    parsed = simu_engine.process_mg_profile_files(
        my_filepath=str(profile_dir),
        mpu=_minimal_mpu(),
        rank_instances_dict={0: object()},
    )

    assert parsed[0]["dp_allreduce"]["duration"] == 7.5
    assert parsed[0]["dp_allreduce"]["tensor_shape"] == [4, 8]
    assert parsed[0]["dp_allreduce"]["tensor_dtype"] == "torch.float16"


def test_process_mg_profile_files_keeps_missing_ep_allreduce_as_zero_duration(
    tmp_path: pathlib.Path,
) -> None:
    profile_dir = _write_profile_file(
        tmp_path=tmp_path,
        filename="stage0_rank0.txt",
        lines=[
            "stage:0:ep_allreduce(batch_id=0,mg_state=steady,duration=9.0,input__shape=None,input__dtype=None,group_kind=ep)"
        ],
    )

    parsed = simu_engine.process_mg_profile_files(
        my_filepath=str(profile_dir),
        mpu=_minimal_mpu(),
        rank_instances_dict={0: object()},
    )

    assert parsed[0]["ep_allreduce"]["duration"] == 0.0
    assert parsed[0]["ep_allreduce"]["tensor_shape"] is None
    assert parsed[0]["ep_allreduce"]["tensor_dtype"] is None
