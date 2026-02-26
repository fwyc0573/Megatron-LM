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


def test_parse_trace_file_applies_op_specific_comm_scale(tmp_path: Path):
    sub_ops = [
        "trace_src_func=tp_allreduce,duration=2.0,timestamp=1.0,comm_func=allreduce",
    ]
    line = (
        "rank:0:forward_step("
        "stage_id=0,batch_id=0,mg_state=None,duration=12.0,description=None,"
        "group_kind=None,input__shape=None,input__dtype=None,timestamp=1.0,"
        f"sub_operations={sub_ops})"
    )
    trace_path = tmp_path / "trace_rank0_20260225000000.txt"
    trace_path.write_text(line + "\n")

    parsed = compare_module.parse_trace_file(
        trace_path,
        subtract_comm=True,
        comm_scale=1.0,
        comm_scale_map={"forward_step": 0.5},
    )
    stats = parsed["forward_step"][0]

    assert stats.total_ms == 12.0
    assert stats.comm_ms == 2.0
    assert stats.effective_comm_ms == 1.0
    assert stats.comp_ms == 11.0


def test_parse_csv_ints_empty_fails_fast():
    with pytest.raises(ValueError, match="Empty rank list"):
        compare_module.parse_csv_ints("")


def test_parse_op_float_map_invalid_entry_fails_fast():
    with pytest.raises(ValueError, match="expected format"):
        compare_module.parse_op_float_map("forward_step", "--distributed-comm-scale-map")


def test_compute_trimmed_mean_reduces_outlier_impact():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]

    trimmed = compare_module._compute_trimmed_mean(values, trim_ratio=0.2)

    assert trimmed == pytest.approx(3.0)


def test_build_repeat_summary_supports_trimmed_rows_key():
    records = [
        {
            "rows": [
                {
                    "rank": 0,
                    "op": "forward_step",
                    "mg_state": "ALL",
                    "diff_pct": 50.0,
                }
            ],
            "trimmed_rows": [
                {
                    "rank": 0,
                    "op": "forward_step",
                    "mg_state": "ALL",
                    "diff_pct": 4.0,
                }
            ],
        },
        {
            "rows": [
                {
                    "rank": 0,
                    "op": "forward_step",
                    "mg_state": "ALL",
                    "diff_pct": 60.0,
                }
            ],
            "trimmed_rows": [
                {
                    "rank": 0,
                    "op": "forward_step",
                    "mg_state": "ALL",
                    "diff_pct": 3.0,
                }
            ],
        },
    ]

    lines, failed_checks = compare_module.build_repeat_summary(
        records, threshold_pct=5.0, row_key="trimmed_rows"
    )

    assert any("median_diff_pct" in line for line in lines)
    assert any("| 0 | forward_step | ALL | 2 | 3.50 | PASS |" in line for line in lines)
    assert failed_checks == 0


def test_build_op_median_summary_reports_per_op_median():
    rows = [
        {"op": "forward_step", "diff_pct": 2.0},
        {"op": "forward_step", "diff_pct": 8.0},
        {"op": "backward_step", "diff_pct": 3.0},
        {"op": "backward_step", "diff_pct": 4.0},
    ]

    lines, failed = compare_module.build_op_median_summary(rows, threshold_pct=5.0)

    assert any("| backward_step | 2 | 3.50 | 3.00 | PASS |" in line for line in lines)
    assert any("| forward_step | 2 | 5.00 | 2.00 | PASS |" in line for line in lines)
    assert failed == 0


def test_build_repeat_op_median_summary_aggregates_runs():
    records = [
        {
            "rows": [
                {"op": "forward_step", "diff_pct": 4.0},
                {"op": "forward_step", "diff_pct": 6.0},
                {"op": "backward_step", "diff_pct": 2.0},
                {"op": "backward_step", "diff_pct": 3.0},
            ]
        },
        {
            "rows": [
                {"op": "forward_step", "diff_pct": 3.0},
                {"op": "forward_step", "diff_pct": 5.0},
                {"op": "backward_step", "diff_pct": 1.0},
                {"op": "backward_step", "diff_pct": 2.0},
            ]
        },
    ]

    lines, failed = compare_module.build_repeat_op_median_summary(
        records, threshold_pct=5.0, row_key="rows"
    )

    assert any("| backward_step | 2 | 2.00 | PASS |" in line for line in lines)
    assert any("| forward_step | 2 | 4.50 | PASS |" in line for line in lines)
    assert failed == 0
