import importlib.util
from pathlib import Path
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


def test_parse_cmd_nvtx_label_invalid_prefix_returns_none():
    label = "other_prefix|rank=0|op=forward_step|state=steady|stage=0|batch=1|iter=1"
    parsed = module.parse_cmd_nvtx_label(label, prefix="cmd_trace")
    assert parsed is None


def test_derive_global_pid_masks_local_tid_bits():
    global_tid = (12345 << 24) + 999
    global_pid = module.derive_global_pid(global_tid)
    assert global_pid == (12345 << 24)


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
