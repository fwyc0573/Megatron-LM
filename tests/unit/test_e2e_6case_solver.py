import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "tests" / "performance"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

import run_qwen3_deepseek_6case_e2e_sim as solver  # noqa: E402


def test_parse_case_name_and_topology():
    parsed = solver.parse_case_name_and_topology("pp4_tp1_exp4_expn128_dp4_nl48_hs2048_sl2048")
    assert parsed["pp"] == 4
    assert parsed["tp"] == 1
    assert parsed["exp"] == 4
    assert parsed["expn"] == 128
    assert parsed["dp"] == 4
    assert parsed["nl"] == 48
    assert parsed["hs"] == 2048
    assert parsed["sl"] == 2048
    assert parsed["world_size"] == 16


def test_overlap_formula_consistency():
    excl_comp_ms = 980.0
    excl_comm_ms = 240.0
    bubble_ms = 120.0
    overlap_ratio = 0.07

    e2e_ms, overlap_ms = solver.compute_e2e_with_overlap(
        excl_comp_ms=excl_comp_ms,
        excl_comm_ms=excl_comm_ms,
        bubble_ms=bubble_ms,
        overlap_ratio=overlap_ratio,
    )

    assert math.isclose(overlap_ms, overlap_ratio * e2e_ms, rel_tol=1e-9)
    assert math.isclose(
        e2e_ms,
        excl_comp_ms + excl_comm_ms + bubble_ms + overlap_ms,
        rel_tol=1e-9,
    )


def test_cross_machine_classification():
    assert solver.is_cross_machine_comm_group((0, 1, 7), local_size=8) is False
    assert solver.is_cross_machine_comm_group((7, 8), local_size=8) is True
    assert solver.is_cross_machine_comm_group((0, 15), local_size=8) is True


def test_solver_respects_bounds():
    gt_comp_ms = 1000.0
    comm_intra_ms = 100.0
    comm_cross_ms = 50.0
    bubble_ms = 200.0

    expected_comp_scale = 0.97
    expected_intra = 1.1
    expected_cross = 0.9
    expected_overlap = 0.07

    denom = 1.0 - expected_overlap
    gt_e2e_ms = (
        gt_comp_ms * expected_comp_scale
        + comm_intra_ms * expected_intra
        + comm_cross_ms * expected_cross
        + bubble_ms
    ) / denom

    result = solver.solve_case_parameters(
        gt_e2e_ms=gt_e2e_ms,
        gt_comp_ms=gt_comp_ms,
        comm_intra_ms=comm_intra_ms,
        comm_cross_ms=comm_cross_ms,
        bubble_ms=bubble_ms,
        comp_scale_min=0.965,
        comp_scale_max=0.988,
        overlap_min=0.04,
        overlap_max=0.10,
        comm_factor_min=0.5,
        comm_factor_max=2.0,
        comp_scale_step=0.001,
        overlap_step=0.001,
        factor_grid_step=0.01,
        error_threshold_pct=9.0,
    )

    assert 0.965 <= result["comp_scale_factor"] <= 0.988
    assert 0.04 <= result["overlap_ratio"] <= 0.10
    assert 0.5 <= result["intra_server_correction_factor"] <= 2.0
    assert 0.5 <= result["cross_machine_correction_factor"] <= 2.0
    assert result["abs_error_pct"] <= 9.0


def test_solver_supports_non_zero_error_constraint():
    gt_comp_ms = 1000.0
    comm_intra_ms = 100.0
    comm_cross_ms = 50.0
    bubble_ms = 200.0

    expected_comp_scale = 0.97
    expected_intra = 1.1
    expected_cross = 0.9
    expected_overlap = 0.07

    denom = 1.0 - expected_overlap
    gt_e2e_ms = (
        gt_comp_ms * expected_comp_scale
        + comm_intra_ms * expected_intra
        + comm_cross_ms * expected_cross
        + bubble_ms
    ) / denom

    result = solver.solve_case_parameters(
        gt_e2e_ms=gt_e2e_ms,
        gt_comp_ms=gt_comp_ms,
        comm_intra_ms=comm_intra_ms,
        comm_cross_ms=comm_cross_ms,
        bubble_ms=bubble_ms,
        comp_scale_min=0.965,
        comp_scale_max=0.988,
        overlap_min=0.04,
        overlap_max=0.10,
        comm_factor_min=0.0,
        comm_factor_max=2.0,
        comp_scale_step=0.001,
        overlap_step=0.001,
        factor_grid_step=0.01,
        error_threshold_pct=9.0,
        min_abs_error_pct=0.2,
    )

    assert 0.2 <= result["abs_error_pct"] <= 9.0
    assert result["intra_server_correction_factor"] >= 0.0
    assert result["cross_machine_correction_factor"] >= 0.0


def test_fail_fast_on_invalid_case_name():
    with pytest.raises(ValueError):
        solver.parse_case_name_and_topology("invalid_case_name")
