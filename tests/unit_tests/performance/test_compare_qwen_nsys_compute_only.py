import importlib.util
import json
from pathlib import Path
import sys

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


def test_build_rank_total_summary_aggregates_selected_rows():
    rows = [
        {"rank": 0, "dist_compute_ms": 10.0, "scale_compute_ms": 9.0},
        {"rank": 0, "dist_compute_ms": 5.0, "scale_compute_ms": 6.0},
        {"rank": 1, "dist_compute_ms": 20.0, "scale_compute_ms": 21.0},
    ]
    lines, failed = module.build_rank_total_summary(rows, threshold_pct=10.0)
    assert any("| 0 | 2 | 15.0000 | 15.0000 | 0.00 | PASS |" in line for line in lines)
    assert any("| 1 | 1 | 20.0000 | 21.0000 | 5.00 | PASS |" in line for line in lines)
    assert failed == 0


def test_shared_kernel_names_intersection_works():
    dist_samples = [
        {"primary_stream_compute_kernel_name_overlap_ms": {"a": 1.0, "b": 2.0}},
        {"primary_stream_compute_kernel_name_overlap_ms": {"a": 0.5, "c": 0.5}},
    ]
    scale_samples = [
        {"primary_stream_compute_kernel_name_overlap_ms": {"a": 1.3, "d": 0.2}},
    ]
    shared = module._shared_kernel_names(
        dist_samples, scale_samples, shared_kernel_source="primary_stream"
    )
    assert shared == {"a"}


def test_shared_overlap_metric_uses_only_shared_names():
    row = {"primary_stream_compute_kernel_name_overlap_ms": {"a": 1.0, "b": 2.0}}
    metric = module._shared_overlap_metric(
        row=row, shared_names={"b"}, shared_kernel_source="primary_stream"
    )
    assert metric == pytest.approx(2.0)


def test_metric_from_row_supports_pure_primary_union():
    row = {
        "compute_pure_primary_union_ms": 5.0,
        "compute_primary_stream_union_ms": 8.0,
    }
    assert module._metric_from_row(row, "pure_primary_union") == pytest.approx(5.0)


def test_main_fails_when_contamination_exceeds_threshold(tmp_path, monkeypatch):
    dist_json = tmp_path / "dist.json"
    scale_json = tmp_path / "scale.json"
    payload = {
        "event_rows": [
            {
                "rank": 0,
                "op": "backward_step",
                "mg_state": "steady",
                "compute_kernel_ms": 10.0,
                "comm_kernel_ms": 0.0,
                "compute_pure_primary_union_ms": 10.0,
                "compute_primary_stream_union_ms": 10.0,
                "compute_kernel_name_overlap_ms": {"k": 10.0},
                "primary_stream_compute_kernel_name_overlap_ms": {"k": 10.0},
                "compute_pure_kernel_name_overlap_ms": {"k": 10.0},
                "primary_stream_compute_pure_kernel_name_overlap_ms": {"k": 10.0},
                "contamination_ms": 2.0,
            }
        ]
    }
    dist_json.write_text(json.dumps(payload))
    scale_json.write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_qwen_nsys_compute_only.py",
            "--distributed-json",
            str(dist_json),
            "--scaling-json",
            str(scale_json),
            "--ranks",
            "0",
            "--ops",
            "backward_step",
            "--compute-metric",
            "pure_primary_union",
            "--require-low-contamination-pct",
            "10",
        ],
    )
    rc = module.main()
    assert rc == 1
