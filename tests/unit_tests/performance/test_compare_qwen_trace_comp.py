import importlib.util
from pathlib import Path

import pytest


def _load_compare_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "performance"
        / "compare_qwen_trace_comp.py"
    )
    spec = importlib.util.spec_from_file_location("compare_qwen_trace_comp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_module = _load_compare_module()


def _write_trace(path: Path):
    sub_ops = [
        "trace_src_func=tp_allreduce,duration=1.5,timestamp=1.0,comm_func=allreduce",
        "trace_src_func=non_comm_probe,duration=2.0,timestamp=2.0",
    ]
    line = (
        "rank:0:forward_step("
        "stage_id=0,batch_id=0,mg_state=None,duration=10.0,description=None,"
        "group_kind=None,input__shape=None,input__dtype=None,timestamp=1.0,"
        f"sub_operations={sub_ops})"
    )
    path.write_text(line + "\n")


def test_parse_trace_file_subtracts_only_comm_subops(tmp_path: Path):
    trace_path = tmp_path / "trace_rank0_20260225000000.txt"
    _write_trace(trace_path)

    parsed = compare_module.parse_trace_file(trace_path, subtract_comm=True)
    stats = parsed["forward_step"][0]

    assert stats.total_ms == 10.0
    assert stats.comm_ms == 1.5
    assert stats.comp_ms == 8.5


def test_parse_trace_file_scaling_keeps_total_comp(tmp_path: Path):
    trace_path = tmp_path / "trace_rank0_20260225000000.txt"
    _write_trace(trace_path)

    parsed = compare_module.parse_trace_file(trace_path, subtract_comm=False)
    stats = parsed["forward_step"][0]

    assert stats.total_ms == 10.0
    assert stats.comm_ms == 1.5
    assert stats.comp_ms == 10.0


def test_parse_csv_ints_empty_fails_fast():
    with pytest.raises(ValueError, match="Empty rank list"):
        compare_module.parse_csv_ints("")

