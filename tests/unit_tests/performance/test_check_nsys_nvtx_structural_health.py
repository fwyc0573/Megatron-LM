import importlib.util
from pathlib import Path
import sqlite3
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "performance"
        / "check_nsys_nvtx_structural_health.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_nsys_nvtx_structural_health", module_path
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
        "cmd_trace|rank=3|op=backward_step|state=steady|"
        "stage=1|batch=2|iter=4|phase=compute"
    )
    parsed = module.parse_cmd_nvtx_label(label, "cmd_trace")
    assert parsed is not None
    assert parsed["rank"] == "3"
    assert parsed["op"] == "backward_step"
    assert parsed["phase"] == "compute"


def test_load_parent_cmd_ranges_skips_phase_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)"
    )
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (100, 300, ?, ?)",
        ("cmd_trace|rank=0|op=forward_step|state=steady|stage=0|batch=0|iter=0", 1 << 24),
    )
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (120, 280, ?, ?)",
        (
            "cmd_trace|rank=0|op=forward_step|state=steady|stage=0|batch=0|iter=0|phase=compute",
            1 << 24,
        ),
    )
    conn.commit()
    try:
        rows = module.load_parent_cmd_ranges(
            conn=conn,
            label_prefix="cmd_trace",
            allowed_ops=["forward_step", "backward_step"],
            rank_filter=None,
        )
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0].op == "forward_step"


def test_open_count_and_overlap_stats():
    max_end = 1_000
    ranges = [
        module.NvtxCmdRange(
            start_ns=100,
            end_ns=max_end,
            global_pid=0,
            rank=0,
            op="forward_step",
            mg_state="steady",
            stage_id="0",
            batch_id="0",
            iter_id="0",
            label="fwd",
        ),
        module.NvtxCmdRange(
            start_ns=300,
            end_ns=700,
            global_pid=0,
            rank=0,
            op="backward_step",
            mg_state="steady",
            stage_id="0",
            batch_id="0",
            iter_id="0",
            label="bwd",
        ),
        module.NvtxCmdRange(
            start_ns=200,
            end_ns=250,
            global_pid=0,
            rank=1,
            op="forward_step",
            mg_state="steady",
            stage_id="1",
            batch_id="0",
            iter_id="0",
            label="rank1_fwd",
        ),
    ]

    open_counts = module.count_open_cmd_ranges(ranges, global_max_end_ns=max_end)
    assert open_counts["forward_step"] == 1
    assert open_counts.get("backward_step", 0) == 0

    overlap_count, overlap_ms, overlap_count_by_rank, overlap_ms_by_rank = (
        module.compute_forward_backward_overlap_stats(ranges)
    )
    assert overlap_count == 1
    assert overlap_ms == (700 - 300) / 1_000_000.0
    assert overlap_count_by_rank == {0: 1}
    assert overlap_ms_by_rank == {0: (700 - 300) / 1_000_000.0}


def test_evaluate_gate_fail_and_pass():
    passed, failures = module.evaluate_gate(
        open_counts={"forward_step": 2, "backward_step": 1},
        overlap_count=3,
        max_open_forward=0,
        max_open_backward=0,
        max_overlap_count=0,
    )
    assert passed is False
    assert len(failures) == 3

    passed, failures = module.evaluate_gate(
        open_counts={"forward_step": 0, "backward_step": 0},
        overlap_count=0,
        max_open_forward=0,
        max_open_backward=0,
        max_overlap_count=0,
    )
    assert passed is True
    assert failures == []
