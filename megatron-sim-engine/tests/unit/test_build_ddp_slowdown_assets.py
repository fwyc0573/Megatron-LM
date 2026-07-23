"""Unit tests for offline DDP slowdown asset building."""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys

import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_prep.slowdown.build_ddp_slowdown_assets import (  # noqa: E402
    build_assets,
    collect_required_kernel_names,
    parse_trace_line,
)
from src.core.simu_engine import TimelinesManager  # noqa: E402


def _write_trace(trace_dir: pathlib.Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "rank0_trace.txt").write_text(
        "\n".join(
            [
                "rank:0:backward_step(stage_id=0,batch_id=1,mg_state=steady,duration=6.0,description=simulation,group_kind=None,cmd_uid=cmd-bwd-1,op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=106.0,sub_operations=[])",
                "rank:0:ddp_grad_comm(comm_uid=comm-1,iter=7,stage_id=0,mg_state=steady,group_kind=dp,comm_func=allreduce,buffer_id=2,bucket_id=3,bucket_offset=0,bucket_numel=4,bucket_numel_unpadded=4,param_count=1,trigger_cmd_uid=cmd-bwd-1,trigger_op=backward_step,trigger_batch_id=1,trigger_timestamp_ms=102.0,launch_timestamp_ms=103.0,launch_source=param_hook,timing_domain=metadata_only,metadata_only=True,status=launch_only,completion_observed_timestamp_ms=None,completion_source=None,wait_cmd_uid=None,wait_start_timestamp_ms=None,wait_end_timestamp_ms=None,logical_stream_role=dp_comm,grad_dtype=torch.float32,data_parallel_world_size=2,duration=0.0,timestamp=103.0,sub_operations=[])",
            ]
        )
    )


def _write_sqlite(
    sqlite_path: pathlib.Path,
    include_compute_phase: bool = True,
    kernel_intervals=None,
) -> None:
    pid = 1 << 24
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, streamId INTEGER, globalPid INTEGER, shortName INTEGER, demangledName INTEGER)"
        )
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [(1, "kernel_a"), (2, "kernel_b"), (3, "ncclKernel"), (11, "void kernel_a<float>()"), (12, "void kernel_b<float>()"), (13, "void ncclKernel<float>()")],
        )
        nvtx_rows = [
            (
                100_000_000,
                106_000_000,
                "cmd_trace|rank=0|op=backward_step|state=steady|stage=0|batch=1|iter=7|cmd_uid=cmd-bwd-1",
                pid,
            ),
        ]
        if include_compute_phase:
            nvtx_rows.append(
                (
                    100_000_000,
                    106_000_000,
                    "cmd_trace|rank=0|op=backward_step|state=steady|stage=0|batch=1|iter=7|cmd_uid=cmd-bwd-1|phase=compute",
                    pid,
                )
            )
        conn.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)", nvtx_rows)
        if kernel_intervals is None:
            kernel_intervals = [
                (101_000_000, 103_000_000, 7, 1, 11),
                (103_500_000, 105_500_000, 7, 2, 12),
                (102_000_000, 105_000_000, 9, 3, 13),
            ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            [
                (start_ns, end_ns, stream_id, pid, short_name_id, demangled_name_id)
                for start_ns, end_ns, stream_id, short_name_id, demangled_name_id in kernel_intervals
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _write_ncu_csv(csv_path: pathlib.Path) -> None:
    pd.DataFrame(
        [
            {
                "Kernel Name": "kernel_a",
                "Compute throughput": 10.0,
                "Memory throughput": 20.0,
                "DRAM throughput": 30.0,
                "Achieved occupancy": 40.0,
                "Maximum occupancy": 50.0,
                "L1 hit rate": 60.0,
                "L2 hit rate": 70.0,
            },
            {
                "Kernel Name": "kernel_a",
                "Compute throughput": 14.0,
                "Memory throughput": 24.0,
                "DRAM throughput": 34.0,
                "Achieved occupancy": 44.0,
                "Maximum occupancy": 54.0,
                "L1 hit rate": 64.0,
                "L2 hit rate": 74.0,
            },
            {
                "Kernel Name": "kernel_b",
                "Compute throughput": 15.0,
                "Memory throughput": 25.0,
                "DRAM throughput": 35.0,
                "Achieved occupancy": 45.0,
                "Maximum occupancy": 55.0,
                "L1 hit rate": 65.0,
                "L2 hit rate": 75.0,
            },
        ]
    ).to_csv(csv_path, index=False)


def _build_args(
    tmp_path: pathlib.Path,
    *,
    include_compute_phase: bool = True,
    kernel_intervals=None,
) -> argparse.Namespace:
    trace_dir = tmp_path / "trace"
    sqlite_path = tmp_path / "baseline.sqlite"
    ncu_csv = tmp_path / "kernel_metrics.csv"
    output_dir = tmp_path / "slowdown_assets"
    model_path = tmp_path / "xgb_model.json"
    scaler_path = tmp_path / "standard_scaler.json"
    model_path.write_text("{}")
    scaler_path.write_text('{"feature_names": ["ground_truth", "Compute throughput", "Memory throughput", "DRAM throughput", "Achieved occupancy", "Maximum occupancy", "L1 hit rate", "L2 hit rate"], "mean": [1, 2, 3, 4, 5, 6, 7, 8], "scale": [1, 1, 1, 1, 1, 1, 1, 1]}')
    _write_trace(trace_dir)
    _write_sqlite(
        sqlite_path,
        include_compute_phase=include_compute_phase,
        kernel_intervals=kernel_intervals,
    )
    _write_ncu_csv(ncu_csv)
    return argparse.Namespace(
        trace_dir=str(trace_dir),
        nsys_sqlite=str(sqlite_path),
        ncu_metrics_csv=str(ncu_csv),
        output_dir=str(output_dir),
        label_prefix="cmd_trace",
        model_path=str(model_path),
        scaler_path=str(scaler_path),
    )


def test_collect_required_kernel_names_returns_non_comm_kernel_short_names(tmp_path: pathlib.Path) -> None:
    args = _build_args(tmp_path)

    required = collect_required_kernel_names(
        trace_dir=pathlib.Path(args.trace_dir),
        nsys_sqlite=pathlib.Path(args.nsys_sqlite),
        label_prefix=args.label_prefix,
    )

    assert required == ["kernel_a", "kernel_b"]


def test_build_assets_outputs_expected_manifest_and_blueprint(tmp_path: pathlib.Path) -> None:
    args = _build_args(tmp_path)

    build_assets(args)

    manifest = json.loads((pathlib.Path(args.output_dir) / "manifest.json").read_text())
    kernel_features = json.loads((pathlib.Path(args.output_dir) / "kernel_features.json").read_text())
    blueprints = json.loads((pathlib.Path(args.output_dir) / "backward_kernel_blueprints.json").read_text())

    assert manifest["scope"] == "ddp_backward_only"
    assert manifest["clip_negative_slowdown"] is True
    assert manifest["generator_version"] == "v3"
    assert manifest["kernel_timeline_model"] == "serial_interval_union_projection_v1"
    assert manifest["model_path"] == args.model_path
    assert manifest["scaler_path"] == args.scaler_path
    assert sorted(kernel_features) == ["kernel_a", "kernel_b"]
    assert kernel_features["kernel_a"]["Compute throughput"] == pytest.approx(12.0)
    assert kernel_features["kernel_a"]["L2 hit rate"] == pytest.approx(72.0)

    blueprint = blueprints["cmd-bwd-1"]
    assert blueprint["rank"] == 0
    assert blueprint["stage_id"] == 0
    assert blueprint["batch_id"] == 1
    assert blueprint["iter_id"] == 7
    assert blueprint["baseline_duration_ms"] == pytest.approx(6.0)
    assert [kernel["kernel_name"] for kernel in blueprint["kernels"]] == ["kernel_a", "kernel_b"]
    assert [kernel["start_offset_ms"] for kernel in blueprint["kernels"]] == pytest.approx([1.0, 3.5])
    assert [kernel["baseline_duration_ms"] for kernel in blueprint["kernels"]] == pytest.approx([2.0, 2.0])
    assert [kernel["stream_id"] for kernel in blueprint["kernels"]] == [7, 7]
    assert [kernel["serial_projection_trimmed_ms"] for kernel in blueprint["kernels"]] == pytest.approx(
        [0.0, 0.0]
    )
    assert blueprint["launch_markers"] == [
        {"comm_uid": "comm-1", "baseline_offset_ms": 3.0, "bucket_id": 3, "buffer_id": 2}
    ]


def test_build_assets_matches_kernel_features_by_short_name(tmp_path: pathlib.Path) -> None:
    args = _build_args(tmp_path)

    build_assets(args)

    blueprints = json.loads((pathlib.Path(args.output_dir) / "backward_kernel_blueprints.json").read_text())
    assert blueprints["cmd-bwd-1"]["kernels"][0]["kernel_name"] == "kernel_a"
    assert blueprints["cmd-bwd-1"]["kernels"][1]["kernel_name"] == "kernel_b"


def test_build_assets_fails_fast_without_compute_phase_window(tmp_path: pathlib.Path) -> None:
    args = _build_args(tmp_path, include_compute_phase=False)

    with pytest.raises(ValueError, match="phase=compute"):
        build_assets(args)


class _NoSlowdownPredictor:
    @staticmethod
    def predict_slowdown_factor(kernel_name: str, ground_truth_ms: float, feature_row: dict) -> float:
        del kernel_name, ground_truth_ms, feature_row
        return 0.0


def test_build_assets_projects_cross_stream_overlap_to_serial_interval_union(
    tmp_path: pathlib.Path,
) -> None:
    args = _build_args(
        tmp_path,
        kernel_intervals=[
            (101_000_000, 103_000_000, 146, 1, 11),
            (102_999_936, 105_000_000, 144, 2, 12),
        ],
    )

    build_assets(args)

    output_dir = pathlib.Path(args.output_dir)
    blueprints = json.loads((output_dir / "backward_kernel_blueprints.json").read_text())
    kernel_features = json.loads((output_dir / "kernel_features.json").read_text())
    blueprint = blueprints["cmd-bwd-1"]
    kernels = blueprint["kernels"]

    assert [kernel["kernel_name"] for kernel in kernels] == ["kernel_a", "kernel_b"]
    assert [kernel["stream_id"] for kernel in kernels] == [146, 144]
    assert [kernel["source_start_offset_ms"] for kernel in kernels] == pytest.approx(
        [1.0, 2.999936]
    )
    assert [kernel["source_baseline_duration_ms"] for kernel in kernels] == pytest.approx(
        [2.0, 2.000064]
    )
    assert [kernel["start_offset_ms"] for kernel in kernels] == pytest.approx([1.0, 3.0])
    assert [kernel["baseline_duration_ms"] for kernel in kernels] == pytest.approx([2.0, 2.0])
    assert [kernel["serial_projection_trimmed_ms"] for kernel in kernels] == pytest.approx(
        [0.0, 0.000064]
    )
    assert sum(kernel["baseline_duration_ms"] for kernel in kernels) == pytest.approx(4.0)

    replay = TimelinesManager._simulate_backward_slowdown_schedule(
        backward_start_time_ms=100.0,
        blueprint=blueprint,
        kernel_features=kernel_features,
        predictor=_NoSlowdownPredictor(),
        comm_duration_by_uid={"comm-1": 0.0},
        max_iters=20,
        tol_ms=1e-6,
    )

    assert replay["backward_duration_ms"] == pytest.approx(6.0)
    assert replay["residual_duration_ms"] == pytest.approx(1.0)


def test_build_assets_rejects_fully_covered_cross_stream_kernel(tmp_path: pathlib.Path) -> None:
    args = _build_args(
        tmp_path,
        kernel_intervals=[
            (101_000_000, 105_000_000, 146, 1, 11),
            (102_000_000, 103_000_000, 144, 2, 12),
        ],
    )

    with pytest.raises(ValueError, match="fully covered by earlier compute intervals"):
        build_assets(args)



def test_parse_trace_line_tolerates_unquoted_description_commas() -> None:
    _, event_name, fields = parse_trace_line(
        "rank:0:dp_allreduce(stage_id=0,batch_id=0,mg_state=finalize,duration=0.09,description=model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas,group_kind=dp,cmd_uid=cmd-final,op_semantics=metadata_placeholder,finalize_base_duration_ms=0.09,input__shape=[10624512],input__dtype=torch.float16,timestamp=1.0,sub_operations=[])"
    )

    assert event_name == 'dp_allreduce'
    assert fields['description'] == 'model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas'
    assert fields['group_kind'] == 'dp'
    assert fields['cmd_uid'] == 'cmd-final'
