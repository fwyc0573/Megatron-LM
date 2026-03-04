import importlib.util
from pathlib import Path
import sqlite3
import sys

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "performance"
        / "analyze_nsys_cmd_kernel_breakdown.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_nsys_cmd_kernel_breakdown", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_parse_cmd_nvtx_label():
    label = "cmd_trace|rank=7|op=forward_step|state=steady|stage=3|batch=2|iter=4"
    parsed = module.parse_cmd_nvtx_label(label, prefix="cmd_trace")
    assert parsed is not None
    assert parsed["rank"] == "7"
    assert parsed["op"] == "forward_step"
    assert parsed["state"] == "steady"


def test_parse_cmd_nvtx_label_with_phase():
    label = (
        "cmd_trace|rank=7|op=backward_step|state=steady|stage=3|batch=2|iter=4|phase=compute"
    )
    parsed = module.parse_cmd_nvtx_label(label, prefix="cmd_trace")
    assert parsed is not None
    assert parsed["phase"] == "compute"


def test_parse_cmd_nvtx_label_invalid_prefix_returns_none():
    label = "other_prefix|rank=0|op=forward_step|state=steady|stage=0|batch=1|iter=1"
    parsed = module.parse_cmd_nvtx_label(label, prefix="cmd_trace")
    assert parsed is None


def test_derive_global_pid_masks_local_tid_bits():
    global_tid = (12345 << 24) + 999
    global_pid = module.derive_global_pid(global_tid)
    assert global_pid == (12345 << 24)


def test_load_nvtx_cmd_ranges_skips_phase_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)"
    )
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (100, 200, ?, ?)",
        ("cmd_trace|rank=0|op=backward_step|state=steady|stage=0|batch=1|iter=2", 1 << 24),
    )
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (110, 190, ?, ?)",
        (
            "cmd_trace|rank=0|op=backward_step|state=steady|stage=0|batch=1|iter=2|phase=compute",
            1 << 24,
        ),
    )
    conn.commit()
    try:
        rows = module.load_nvtx_cmd_ranges(
            conn=conn,
            label_prefix="cmd_trace",
            allowed_ops=["backward_step"],
            rank_filter=[0],
        )
    finally:
        conn.close()
    assert len(rows) == 1


def test_summarize_nvtx_ranges_splits_compute_and_comm_overlap():
    cmd_range = module.NvtxCmdRange(
        start_ns=100,
        end_ns=300,
        global_pid=1,
        rank=0,
        op="forward_step",
        mg_state="steady",
        stage_id="0",
        batch_id="1",
        iter_id="4",
        label="cmd_trace|rank=0|op=forward_step|state=steady|stage=0|batch=1|iter=4",
    )
    kernels_by_pid = {
        1: [
            module.KernelRecord(
                start_ns=120,
                end_ns=180,
                stream_id=7,
                name="gemm_kernel",
                is_comm=False,
            ),
            module.KernelRecord(
                start_ns=150,
                end_ns=260,
                stream_id=41,
                name="ncclKernel_AllToAll",
                is_comm=True,
            ),
        ]
    }

    rows = module.summarize_nvtx_ranges([cmd_range], kernels_by_pid)
    assert len(rows) == 1
    row = rows[0]
    assert row["compute_kernel_ms"] == pytest.approx((180 - 120) / 1_000_000.0)
    assert row["comm_kernel_ms"] == pytest.approx((260 - 150) / 1_000_000.0)
    assert row["compute_kernel_union_ms"] == pytest.approx(row["compute_kernel_ms"])
    assert row["comm_kernel_union_ms"] == pytest.approx(row["comm_kernel_ms"])
    assert row["compute_primary_stream_union_ms"] == pytest.approx(
        row["compute_kernel_ms"]
    )
    assert row["primary_compute_stream_id"] == 7
    assert row["compute_stream_count"] == 1
    assert "gemm_kernel" in row["compute_kernel_name_overlap_ms"]
    assert "gemm_kernel" in row["primary_stream_compute_kernel_name_overlap_ms"]
    assert row["kernel_count"] == 2


def test_summarize_nvtx_ranges_phase_pure_metrics_and_contamination():
    cmd_range = module.NvtxCmdRange(
        start_ns=100,
        end_ns=300,
        global_pid=1,
        rank=0,
        op="backward_step",
        mg_state="steady",
        stage_id="0",
        batch_id="1",
        iter_id="4",
        label="cmd_trace|rank=0|op=backward_step|state=steady|stage=0|batch=1|iter=4",
    )
    kernels_by_pid = {
        1: [
            module.KernelRecord(
                start_ns=120,
                end_ns=180,
                stream_id=7,
                name="gemm_kernel",
                is_comm=False,
            ),
            module.KernelRecord(
                start_ns=150,
                end_ns=260,
                stream_id=41,
                name="ncclKernel_AllToAll",
                is_comm=True,
            ),
            module.KernelRecord(
                start_ns=240,
                end_ns=260,
                stream_id=7,
                name="copy_kernel",
                is_comm=False,
            ),
        ]
    }
    phase_windows = {
        module.range_identity(cmd_range): {
            "compute": [(110, 220)],
            "comm": [(220, 280)],
        }
    }

    rows = module.summarize_nvtx_ranges(
        [cmd_range], kernels_by_pid, phase_windows_by_identity=phase_windows
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["compute_kernel_ms"] == pytest.approx((60 + 20) / 1_000_000.0)
    assert row["compute_pure_ms"] == pytest.approx(60 / 1_000_000.0)
    assert row["compute_pure_union_ms"] == pytest.approx(60 / 1_000_000.0)
    assert row["compute_pure_primary_union_ms"] == pytest.approx(60 / 1_000_000.0)
    assert row["contamination_ms"] == pytest.approx(20 / 1_000_000.0)
    assert row["phase_compute_window_count"] == 1
    assert row["phase_comm_window_count"] == 1
