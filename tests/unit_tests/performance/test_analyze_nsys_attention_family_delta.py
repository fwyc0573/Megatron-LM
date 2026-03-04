import importlib.util
from pathlib import Path
import sys

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "performance"
        / "analyze_nsys_attention_family_delta.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_nsys_attention_family_delta", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_parse_cmd_nvtx_label_with_phase():
    label = (
        "cmd_trace|rank=4|op=backward_step|state=steady|stage=1|batch=2|iter=5|phase=compute"
    )
    parsed = module.parse_cmd_nvtx_label(label, prefix="cmd_trace")
    assert parsed is not None
    assert parsed["rank"] == "4"
    assert parsed["op"] == "backward_step"
    assert parsed["phase"] == "compute"


def test_parse_csv_strs_fails_fast_on_empty():
    with pytest.raises(ValueError, match="Empty value list"):
        module.parse_csv_strs("")


def test_interquartile_range_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    # q1=1.75, q3=3.25 => iqr=1.5 with linear interpolation.
    assert module.interquartile_range(values) == pytest.approx(1.5)


def test_pair_windows_orders_by_numeric_batch_id():
    dist_rows = [
        {"rank": 4, "iter_id": "3", "batch_id": "10", "marker": "dist10"},
        {"rank": 4, "iter_id": "3", "batch_id": "2", "marker": "dist2"},
    ]
    scale_rows = [
        {"rank": 4, "iter_id": "3", "batch_id": "2", "marker": "scale2"},
        {"rank": 4, "iter_id": "3", "batch_id": "10", "marker": "scale10"},
    ]

    pairs, missing = module.pair_windows(dist_rows, scale_rows)
    assert missing == 0
    assert len(pairs) == 2
    assert pairs[0][0]["marker"] == "dist2"
    assert pairs[0][1]["marker"] == "scale2"
    assert pairs[1][0]["marker"] == "dist10"
    assert pairs[1][1]["marker"] == "scale10"


def test_summarize_window_uses_overlap_duration_for_fmha():
    window = module.PhaseWindow(
        start_ns=100,
        end_ns=200,
        global_pid=7,
        rank=4,
        op="backward_step",
        mg_state="steady",
        stage_id="1",
        batch_id="1",
        iter_id="3",
        label="cmd_trace|rank=4|op=backward_step|state=steady|stage=1|batch=1|iter=3|phase=compute",
    )
    kernels_by_pid = {
        7: [
            module.KernelRecord(
                start_ns=150,
                end_ns=250,
                stream_id=1,
                name="fmha_cutlassB_kernel",
                grid_x=64,
                grid_y=8,
                grid_z=1,
                block_x=512,
                block_y=1,
                block_z=1,
                registers_per_thread=128,
                static_shared_memory=0,
                dynamic_shared_memory=0,
            ),
            module.KernelRecord(
                start_ns=110,
                end_ns=180,
                stream_id=9,
                name="ncclKernel_AllToAll",
                grid_x=1,
                grid_y=1,
                grid_z=1,
                block_x=1,
                block_y=1,
                block_z=1,
                registers_per_thread=1,
                static_shared_memory=0,
                dynamic_shared_memory=0,
            ),
        ]
    }

    row = module.summarize_window(
        window,
        kernels_by_pid,
        small_kernel_threshold_us=60.0,
        adjacency_window_us=200.0,
    )

    # Overlap only covers [150, 200] => 50ns => 0.05us.
    assert row["fmha_kernel_durations_us"] == pytest.approx([0.05])
    assert row["primary_union_ms"] == pytest.approx(50 / 1_000_000.0)
    assert row["primary_name_ms"]["fmha_cutlassB_kernel"] == pytest.approx(
        50 / 1_000_000.0
    )
    assert row["compute_stream_count"] == 1
    assert row["primary_small_kernel_count"] == 1
    assert row["small_kernel_adjacent_pre_count"] == 0
    assert row["small_kernel_adjacent_post_count"] == 0
    assert row["small_kernel_adjacent_pre_name_count"] == {}
    assert row["small_kernel_adjacent_post_name_count"] == {}
    assert row["small_kernel_adjacent_pre_name_ms"] == {}
    assert row["small_kernel_adjacent_post_name_ms"] == {}
    assert row["small_kernel_immediate_pre_count"] == 0
    assert row["small_kernel_immediate_post_count"] == 0
    assert row["small_kernel_immediate_pre_name_count"] == {}
    assert row["small_kernel_immediate_post_name_count"] == {}
    assert row["small_kernel_immediate_pre_name_ms"] == {}
    assert row["small_kernel_immediate_post_name_ms"] == {}


def test_summarize_window_tracks_immediate_same_stream_neighbors():
    window = module.PhaseWindow(
        start_ns=0,
        end_ns=200_000,
        global_pid=9,
        rank=4,
        op="backward_step",
        mg_state="steady",
        stage_id="1",
        batch_id="1",
        iter_id="3",
        label="cmd_trace|rank=4|op=backward_step|state=steady|stage=1|batch=1|iter=3|phase=compute",
    )
    kernels_by_pid = {
        9: [
            module.KernelRecord(
                start_ns=10_000,
                end_ns=40_000,
                stream_id=3,
                name="vectorized_elementwise_kernel<FillFunctor<unsigned char>>",
                grid_x=1,
                grid_y=1,
                grid_z=1,
                block_x=1,
                block_y=1,
                block_z=1,
                registers_per_thread=1,
                static_shared_memory=0,
                dynamic_shared_memory=0,
            ),
            module.KernelRecord(
                start_ns=40_500,
                end_ns=140_500,
                stream_id=3,
                name="fmha_cutlassB_kernel",
                grid_x=64,
                grid_y=8,
                grid_z=1,
                block_x=512,
                block_y=1,
                block_z=1,
                registers_per_thread=128,
                static_shared_memory=0,
                dynamic_shared_memory=0,
            ),
            module.KernelRecord(
                start_ns=141_000,
                end_ns=160_000,
                stream_id=3,
                name="CUDAFunctor_add<float>",
                grid_x=1,
                grid_y=1,
                grid_z=1,
                block_x=1,
                block_y=1,
                block_z=1,
                registers_per_thread=1,
                static_shared_memory=0,
                dynamic_shared_memory=0,
            ),
        ]
    }

    row = module.summarize_window(
        window,
        kernels_by_pid,
        small_kernel_threshold_us=60.0,
        adjacency_window_us=200.0,
    )

    assert row["small_kernel_immediate_pre_count"] == 1
    assert row["small_kernel_immediate_post_count"] == 1
    assert row["small_kernel_immediate_pre_name_count"] == {
        "vectorized_elementwise_kernel<FillFunctor<unsigned char>>": 1
    }
    assert row["small_kernel_immediate_post_name_count"] == {"CUDAFunctor_add<float>": 1}
    assert row["small_kernel_immediate_pre_name_ms"][
        "vectorized_elementwise_kernel<FillFunctor<unsigned char>>"
    ] == pytest.approx(0.03)
    assert row["small_kernel_immediate_post_name_ms"]["CUDAFunctor_add<float>"] == pytest.approx(
        0.019
    )


def test_build_report_outputs_expected_payload_fields():
    dist_row = {
        "rank": 4,
        "iter_id": "3",
        "batch_id": "1",
        "primary_union_ms": 10.0,
        "primary_stream_id": 1,
        "compute_stream_count": 1,
        "primary_name_ms": {"fmha_cutlassB": 6.0, "k1": 4.0},
        "primary_small_kernel_count": 2,
        "primary_small_kernel_ms": 0.05,
        "small_kernel_adjacent_pre_count": 3,
        "small_kernel_adjacent_post_count": 1,
        "small_kernel_adjacent_pre_ms": 0.03,
        "small_kernel_adjacent_post_ms": 0.02,
        "small_kernel_adjacent_pre_name_count": {"CUDAFunctor_add<float>": 3},
        "small_kernel_adjacent_post_name_count": {"rmsnorm_bwd": 1},
        "small_kernel_adjacent_pre_name_ms": {"CUDAFunctor_add<float>": 0.03},
        "small_kernel_adjacent_post_name_ms": {"rmsnorm_bwd": 0.02},
        "small_kernel_immediate_pre_count": 1,
        "small_kernel_immediate_post_count": 0,
        "small_kernel_immediate_pre_ms": 0.01,
        "small_kernel_immediate_post_ms": 0.0,
        "small_kernel_immediate_pre_name_count": {"FillFunctor<unsigned char>": 1},
        "small_kernel_immediate_post_name_count": {},
        "small_kernel_immediate_pre_name_ms": {"FillFunctor<unsigned char>": 0.01},
        "small_kernel_immediate_post_name_ms": {},
        "fmha_stream_ids": [9],
        "fmha_kernel_durations_us": [100.0, 120.0, 140.0],
        "fmha_launch_configs": [(64, 8, 1, 512, 1, 1, 128, 0, 0)],
    }
    scale_row = {
        "rank": 4,
        "iter_id": "3",
        "batch_id": "1",
        "primary_union_ms": 14.0,
        "primary_stream_id": 2,
        "compute_stream_count": 2,
        "primary_name_ms": {"fmha_cutlassB": 9.0, "k1": 5.0},
        "primary_small_kernel_count": 4,
        "primary_small_kernel_ms": 0.09,
        "small_kernel_adjacent_pre_count": 5,
        "small_kernel_adjacent_post_count": 2,
        "small_kernel_adjacent_pre_ms": 0.05,
        "small_kernel_adjacent_post_ms": 0.03,
        "small_kernel_adjacent_pre_name_count": {"CUDAFunctor_add<float>": 5},
        "small_kernel_adjacent_post_name_count": {"rmsnorm_bwd": 2},
        "small_kernel_adjacent_pre_name_ms": {"CUDAFunctor_add<float>": 0.05},
        "small_kernel_adjacent_post_name_ms": {"rmsnorm_bwd": 0.03},
        "small_kernel_immediate_pre_count": 1,
        "small_kernel_immediate_post_count": 0,
        "small_kernel_immediate_pre_ms": 0.02,
        "small_kernel_immediate_post_ms": 0.0,
        "small_kernel_immediate_pre_name_count": {"FillFunctor<unsigned char>": 1},
        "small_kernel_immediate_post_name_count": {},
        "small_kernel_immediate_pre_name_ms": {"FillFunctor<unsigned char>": 0.02},
        "small_kernel_immediate_post_name_ms": {},
        "fmha_stream_ids": [11],
        "fmha_kernel_durations_us": [110.0, 130.0, 150.0],
        "fmha_launch_configs": [(64, 8, 1, 512, 1, 1, 128, 0, 0)],
    }

    report, payload = module.build_report([(dist_row, scale_row)], missing=0)

    assert "fmha launch-config parity" in report
    assert "fmha duration IQR by rank" in report
    assert payload["gap_ms"] == pytest.approx(4.0)
    assert payload["fmha_gap_ms"] == pytest.approx(3.0)
    assert payload["fmha_gap_share_pct"] == pytest.approx(75.0)
    assert payload["fmha_duration_stats_by_rank"][4]["dist_iqr_us"] == pytest.approx(20.0)
    assert payload["fmha_cfg_set_dist"] == payload["fmha_cfg_set_scale"]
    assert payload["primary_stream_id_mismatch_pairs"] == 1
    assert payload["fmha_stream_set_mismatch_pairs"] == 1
    assert payload["dist_primary_small_kernel_count_total"] == 2
    assert payload["scale_primary_small_kernel_count_total"] == 4
    assert payload["dist_small_adjacent_pre_top_names"][0]["name"] == "CUDAFunctor_add<float>"
    assert payload["dist_small_adjacent_pre_top_names"][0]["count"] == 3
    assert payload["scale_small_adjacent_pre_top_names"][0]["count"] == 5
    assert payload["dist_small_immediate_pre_top_names"][0]["name"] == "FillFunctor<unsigned char>"
    assert payload["dist_small_immediate_pre_top_names"][0]["count"] == 1
    assert payload["scale_small_immediate_pre_top_names"][0]["ms"] == pytest.approx(0.02)
