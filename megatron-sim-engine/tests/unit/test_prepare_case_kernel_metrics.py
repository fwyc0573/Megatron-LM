"""Unit tests for case-local kernel metrics preparation."""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_prep.slowdown.prepare_case_kernel_metrics import (  # noqa: E402
    prepare_case_kernel_metrics,
    main as prepare_case_kernel_metrics_main,
)


PID0 = 1 << 24
PID2 = 2 << 24


def _write_trace(trace_dir: pathlib.Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "rank0.txt").write_text(
        "\n".join(
            [
                "rank:0:backward_step(stage_id=0,batch_id=0,mg_state=cooldown,duration=6.0,description=simulation,group_kind=None,cmd_uid=cmd-r0,op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=106.0,sub_operations=[])",
                "rank:0:ddp_grad_comm(comm_uid=comm-r0,iter=1,stage_id=0,mg_state=cooldown,group_kind=dp,comm_func=allreduce,buffer_id=0,bucket_id=1,bucket_offset=0,bucket_numel=1,bucket_numel_unpadded=1,param_count=1,trigger_cmd_uid=cmd-r0,trigger_op=backward_step,trigger_batch_id=0,trigger_timestamp_ms=101.0,launch_timestamp_ms=103.0,launch_source=param_hook,timing_domain=metadata_only,metadata_only=True,status=launch_only,completion_observed_timestamp_ms=None,completion_source=None,wait_cmd_uid=None,wait_start_timestamp_ms=None,wait_end_timestamp_ms=None,logical_stream_role=dp_comm,grad_dtype=torch.float16,data_parallel_world_size=2,duration=0.0,timestamp=103.0,sub_operations=[])",
            ]
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank2.txt").write_text(
        "\n".join(
            [
                "rank:2:backward_step(stage_id=1,batch_id=0,mg_state=steady,duration=7.0,description=simulation,group_kind=None,cmd_uid=cmd-r2,op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=207.0,sub_operations=[])",
                "rank:2:ddp_grad_comm(comm_uid=comm-r2,iter=1,stage_id=1,mg_state=steady,group_kind=dp,comm_func=allreduce,buffer_id=0,bucket_id=2,bucket_offset=0,bucket_numel=1,bucket_numel_unpadded=1,param_count=1,trigger_cmd_uid=cmd-r2,trigger_op=backward_step,trigger_batch_id=0,trigger_timestamp_ms=201.0,launch_timestamp_ms=203.0,launch_source=param_hook,timing_domain=metadata_only,metadata_only=True,status=launch_only,completion_observed_timestamp_ms=None,completion_source=None,wait_cmd_uid=None,wait_start_timestamp_ms=None,wait_end_timestamp_ms=None,logical_stream_role=dp_comm,grad_dtype=torch.float16,data_parallel_world_size=2,duration=0.0,timestamp=203.0,sub_operations=[])",
            ]
        ),
        encoding="utf-8",
    )


def _write_sqlite(sqlite_path: pathlib.Path) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, streamId INTEGER, globalPid INTEGER, shortName INTEGER, demangledName INTEGER)")
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [
                (1, "kernel_a"),
                (2, "ln_bwd_tuned_kernel"),
                (11, "void kernel_a<float>()"),
                (12, "void ln_bwd_tuned_kernel<float>()"),
            ],
        )
        conn.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (100_000_000, 106_000_000, "cmd_trace|rank=0|op=backward_step|state=cooldown|stage=0|batch=0|iter=1|cmd_uid=cmd-r0", PID0),
                (100_000_000, 106_000_000, "cmd_trace|rank=0|op=backward_step|state=cooldown|stage=0|batch=0|iter=1|cmd_uid=cmd-r0|phase=compute", PID0),
                (200_000_000, 207_000_000, "cmd_trace|rank=2|op=backward_step|state=steady|stage=1|batch=0|iter=1|cmd_uid=cmd-r2", PID2),
                (200_000_000, 207_000_000, "cmd_trace|rank=2|op=backward_step|state=steady|stage=1|batch=0|iter=1|cmd_uid=cmd-r2|phase=compute", PID2),
            ],
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            [
                (101_000_000, 104_000_000, 7, PID0, 1, 11),
                (201_000_000, 205_000_000, 7, PID2, 2, 12),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_prepare_case_kernel_metrics_applies_alias_and_reports_missing_by_rank(tmp_path: pathlib.Path) -> None:
    trace_dir = tmp_path / "trace"
    sqlite_path = tmp_path / "baseline.sqlite"
    _write_trace(trace_dir)
    _write_sqlite(sqlite_path)

    broad_csv = tmp_path / "broad.csv"
    pd.DataFrame(
        [
            {
                "Kernel Name": "kernel_a",
                "Compute throughput": 1.0,
                "Memory throughput": 2.0,
                "DRAM throughput": 3.0,
                "Achieved occupancy": 4.0,
                "Maximum occupancy": 5.0,
                "L1 hit rate": 6.0,
                "L2 hit rate": 7.0,
            },
            {
                "Kernel Name": "ln_bwd_general_kernel",
                "Compute throughput": 10.0,
                "Memory throughput": 20.0,
                "DRAM throughput": 30.0,
                "Achieved occupancy": 40.0,
                "Maximum occupancy": 50.0,
                "L1 hit rate": 60.0,
                "L2 hit rate": 70.0,
            },
        ]
    ).to_csv(broad_csv, index=False)

    output_csv = tmp_path / "prepared.csv"
    report_json = tmp_path / "report.json"
    report = prepare_case_kernel_metrics(
        trace_dir=trace_dir,
        nsys_sqlite=sqlite_path,
        label_prefix="cmd_trace",
        candidate_csv_paths=[broad_csv],
        output_csv=output_csv,
        report_json=report_json,
        aliases={"ln_bwd_general_kernel": "ln_bwd_tuned_kernel"},
    )

    prepared = pd.read_csv(output_csv)
    assert set(prepared["Kernel Name"]) == {"kernel_a", "ln_bwd_tuned_kernel"}
    assert report["missing_kernels"] == []
    assert report["missing_kernels_by_rank"] == {"0": [], "2": []}

    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_report["required_kernels_by_rank"]["2"] == ["ln_bwd_tuned_kernel"]


def test_prepare_case_kernel_metrics_cli_writes_prepared_csv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_dir = tmp_path / "trace"
    sqlite_path = tmp_path / "baseline.sqlite"
    _write_trace(trace_dir)
    _write_sqlite(sqlite_path)

    broad_csv = tmp_path / "broad.csv"
    pd.DataFrame(
        [
            {
                "Kernel Name": "kernel_a",
                "Compute throughput": 1.0,
                "Memory throughput": 2.0,
                "DRAM throughput": 3.0,
                "Achieved occupancy": 4.0,
                "Maximum occupancy": 5.0,
                "L1 hit rate": 6.0,
                "L2 hit rate": 7.0,
            },
            {
                "Kernel Name": "ln_bwd_general_kernel",
                "Compute throughput": 10.0,
                "Memory throughput": 20.0,
                "DRAM throughput": 30.0,
                "Achieved occupancy": 40.0,
                "Maximum occupancy": 50.0,
                "L1 hit rate": 60.0,
                "L2 hit rate": 70.0,
            },
        ]
    ).to_csv(broad_csv, index=False)

    output_csv = tmp_path / "prepared.csv"
    report_json = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_case_kernel_metrics.py",
            "--trace-dir",
            str(trace_dir),
            "--nsys-sqlite",
            str(sqlite_path),
            "--label-prefix",
            "cmd_trace",
            "--candidate-csv",
            str(broad_csv),
            "--output-csv",
            str(output_csv),
            "--report-json",
            str(report_json),
            "--alias",
            "ln_bwd_general_kernel=ln_bwd_tuned_kernel",
        ],
    )

    assert prepare_case_kernel_metrics_main() == 0
    stdout_report = json.loads(capsys.readouterr().out)
    prepared = pd.read_csv(output_csv)
    assert set(prepared["Kernel Name"]) == {"kernel_a", "ln_bwd_tuned_kernel"}
    assert stdout_report["missing_kernels"] == []
    assert json.loads(report_json.read_text(encoding="utf-8"))["missing_kernels_by_rank"] == {"0": [], "2": []}
