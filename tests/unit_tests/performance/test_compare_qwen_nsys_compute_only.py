import importlib.util
from pathlib import Path

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "performance"
        / "compare_qwen_nsys_compute_only.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_qwen_nsys_compute_only", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_compute_trimmed_mean_reduces_outlier_impact():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    trimmed = module.compute_trimmed_mean(values, trim_ratio=0.2)
    assert trimmed == pytest.approx(3.0)


def test_reduce_values_supports_all_reducers():
    values = [1.0, 3.0, 100.0]
    assert module.reduce_values(values, method="mean", trim_ratio=0.2) == pytest.approx(
        34.6666666667
    )
    assert module.reduce_values(values, method="median", trim_ratio=0.2) == pytest.approx(
        3.0
    )
    assert module.reduce_values(
        values, method="trimmed_mean", trim_ratio=0.2
    ) == pytest.approx(3.0)


def test_parse_csv_ints_empty_fails_fast():
    with pytest.raises(ValueError, match="Empty rank list"):
        module.parse_csv_ints("")


def test_build_repeat_summary_uses_median_across_runs():
    records = [
        {
            "rows": [
                {"rank": 0, "op": "forward_step", "mg_state": "ALL", "diff_pct": 7.0},
                {"rank": 1, "op": "forward_step", "mg_state": "ALL", "diff_pct": 3.0},
            ]
        },
        {
            "rows": [
                {"rank": 0, "op": "forward_step", "mg_state": "ALL", "diff_pct": 5.0},
                {"rank": 1, "op": "forward_step", "mg_state": "ALL", "diff_pct": 4.0},
            ]
        },
    ]
    lines, failed = module.build_repeat_summary(records, threshold_pct=5.0)
    assert any("| 0 | forward_step | ALL | 2 | 6.00 | FAIL |" in line for line in lines)
    assert any("| 1 | forward_step | ALL | 2 | 3.50 | PASS |" in line for line in lines)
    assert failed == 1


def test_build_repeat_op_rank_median_summary():
    records = [
        {
            "rows": [
                {"op": "forward_step", "diff_pct": 8.0},
                {"op": "forward_step", "diff_pct": 2.0},
                {"op": "backward_step", "diff_pct": 3.0},
                {"op": "backward_step", "diff_pct": 1.0},
            ]
        },
        {
            "rows": [
                {"op": "forward_step", "diff_pct": 6.0},
                {"op": "forward_step", "diff_pct": 4.0},
                {"op": "backward_step", "diff_pct": 5.0},
                {"op": "backward_step", "diff_pct": 3.0},
            ]
        },
    ]
    lines, failed = module.build_repeat_op_rank_median_summary(
        records, threshold_pct=5.0
    )
    assert any("| backward_step | 2 | 3.00 | PASS |" in line for line in lines)
    assert any("| forward_step | 2 | 5.00 | PASS |" in line for line in lines)
    assert failed == 0
