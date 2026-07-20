#!/usr/bin/env bash
# CPU-only public-entry e2e using a strict synthetic prebaked distribution.
# This validates workflow wiring only; it is not real GPU or dataset qualification.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
FIXTURE_HELPER="${REPO_ROOT}/tests/integration/fixtures/sc26_ae_task3_fixture.py"
MANIFEST_TOOL="${REPO_ROOT}/SC26-AE/tools/artifact_manifest.py"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
TEST_ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task3-prebaked-e2e.XXXXXX")
TOOLS_ROOT="${TEST_ROOT}/tools"
PREBAKED_ROOT="${TEST_ROOT}/prebaked"
OUTPUT_ROOT="${TEST_ROOT}/output"
SUMMARY_JSON="${TEST_ROOT}/summary.json"

python3 "${FIXTURE_HELPER}" tools --output-root "${TOOLS_ROOT}" \
    >"${TEST_ROOT}/tools.json"
python3 "${FIXTURE_HELPER}" prebaked \
    --repo-root "${REPO_ROOT}" \
    --output-root "${PREBAKED_ROOT}" \
    >"${TEST_ROOT}/prebaked.json"

for model in gpt175b qwen3_a30b dsv3; do
    run_id="prebaked-cpu-${model}"
    measurement="${TEST_ROOT}/${model}-measurement.json"
    env \
        AE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
        ARTIFACT_SOURCE=prebaked \
        PREBAKED_ROOT="${PREBAKED_ROOT}" \
        SIMULATOR_HARDWARE_TYPE=H800_SXM \
        TASK3_EXECUTION_MODE=synthetic \
        TASK3_META_PYTHON=python3 \
        TASK3_SIMULATOR_PYTHON=python3 \
        TASK3_SIM_ENGINE_ROOT="${TOOLS_ROOT}" \
        TASK3_SCHEDULER="${TOOLS_ROOT}/scheduler.py" \
        TASK3_SIMULATOR="${TOOLS_ROOT}/simulator.py" \
        TASK3_SIMULATION_RUN_ID="${run_id}" \
        python3 "${FIXTURE_HELPER}" measure \
            --output "${measurement}" \
            --allocation-mib 32 \
            -- bash "${REPO_ROOT}/SC26-AE/task3_${model}.sh" \
            >"${TEST_ROOT}/${model}.log" 2>&1

    run_root="${OUTPUT_ROOT}/${model}/task3/runs/${run_id}"
    python3 -B "${MANIFEST_TOOL}" verify \
        --root "${run_root}" \
        --manifest "${run_root}/artifact_manifest.json" \
        >"${TEST_ROOT}/${model}-manifest.log"
done

python3 - \
    "${OUTPUT_ROOT}" "${TEST_ROOT}" "${SUMMARY_JSON}" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

output_root = pathlib.Path(sys.argv[1])
test_root = pathlib.Path(sys.argv[2])
summary_path = pathlib.Path(sys.argv[3])
expected = {
    "gpt175b": {
        "step": 18.5, "forward": 5.0, "backward": 9.0, "optimizer": 2.0,
        "pp": 8, "microbatches": 48, "pp_shape": [2048, 1, 12288],
    },
    "qwen3_a30b": {
        "step": 22.5, "forward": 6.0, "backward": 11.0, "optimizer": 2.5,
        "pp": 4, "microbatches": 16, "pp_shape": [256, 1, 2048],
    },
    "dsv3": {
        "step": 24.5, "forward": 6.5, "backward": 12.0, "optimizer": 3.0,
        "pp": 4, "microbatches": 16, "pp_shape": [256, 1, 2048],
    },
}
rows = []
for model, values in expected.items():
    run_id = f"prebaked-cpu-{model}"
    task_dir = output_root / model / "task3"
    run_root = task_dir / "runs" / run_id
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    measurement = json.loads((test_root / f"{model}-measurement.json").read_text(encoding="utf-8"))
    marker = json.loads((task_dir / "run_marker.json").read_text(encoding="utf-8"))
    manifest_path = run_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report["rank0_step_time_ms"] == values["step"]
    assert report["rank0_forward_step_duration_sum_ms"] == values["forward"]
    assert report["rank0_backward_step_duration_sum_ms"] == values["backward"]
    assert report["rank0_optimizer_step_duration_sum_ms"] == values["optimizer"]
    assert report["simulator_load_time_s"] == 0.125
    assert report["simulator_execution_time_s"] == 0.375
    assert report["simulator_wall_clock_s"] == 0.5
    for field, value in report.items():
        if field in {"schema_version", "model", "artifact_source"}:
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value)) and value >= 0
    assert report["rank0_step_time_ms"] > 0
    assert report["rank0_forward_step_duration_sum_ms"] > 0
    assert report["rank0_backward_step_duration_sum_ms"] > 0
    assert report["rank0_optimizer_step_duration_sum_ms"] > 0
    markdown = (run_root / "report.md").read_text(encoding="utf-8")
    for field, value in report.items():
        assert f"| `{field}` | `{value}` |" in markdown

    assert measurement["exit_code"] == 0
    assert measurement["wall_clock_s"] > 0
    assert measurement["peak_rss_kib"] > 0
    assert measurement["peak_rss_gib"] > 0
    assert measurement["tested_host_allocation_bytes"] == 32 * 1024 * 1024
    assert measurement["tested_host_allocation_mib"] == 32
    assert measurement["execution_evidence"] == "local_synthetic_fixture"

    assert marker["verified"] is True
    assert marker["artifact_source"] == "prebaked"
    assert marker["simulation_run_id"] == run_id
    assert marker["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert marker["artifact_manifest_sha256"] == marker["manifest_sha256"]
    assert manifest["communication_backend"] == "analytical"
    assert manifest["overlap_mode"] == "on"
    assert manifest["database_is_trace_dir"] is True
    assert manifest["simulation_topology"]["local_size"] == 8
    schedule_paths = sorted((run_root / "schedule").glob("stage*_scheduling_plan.txt"))
    assert len(schedule_paths) == values["pp"]
    for schedule_path in schedule_paths:
        lines = schedule_path.read_text(encoding="utf-8").splitlines()
        assert sum(":forward_step(" in line for line in lines) == values["microbatches"]
        assert sum(":backward_step(" in line for line in lines) == values["microbatches"]
        assert sum(":optimizer_step(" in line for line in lines) == 1
        pp_lines = [
            line
            for line in lines
            if any(f":{operation}(" in line for operation in (
                "recv_forward", "send_forward", "recv_backward", "send_backward"
            ))
        ]
        assert pp_lines
        assert all("group_kind=pp" in line for line in pp_lines)
        assert all(f"input__shape={values['pp_shape']}" in line for line in pp_lines)
        assert all("input__dtype=torch.bfloat16" in line for line in pp_lines)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert all(entry["size_bytes"] >= 0 and len(entry["sha256"]) == 64 for entry in entries.values())
    required_nonempty = {
        "logs/commands.log",
        "report.json",
        "report.md",
        "provenance/resolved_inputs.json",
        "provenance/input_evidence.json",
        "provenance/task1_manifest.json",
        "provenance/task2_manifest.json",
        "provenance/distribution_manifest.json",
        "slowdown_assets/manifest.json",
        "slowdown_assets/kernel_features.json",
        "slowdown_assets/backward_kernel_blueprints.json",
    }
    required_nonempty.update(
        f"schedule/stage{stage}_scheduling_plan.txt" for stage in range(values["pp"])
    )
    assert required_nonempty.issubset(entries)
    assert all(entries[path]["size_bytes"] > 0 for path in required_nonempty)
    assert entries["logs/scheduler.log"]["size_bytes"] == 0
    assert entries["logs/simulator.log"]["size_bytes"] == 0

    rows.append(
        {
            "model": model,
            "rank0_step_time_ms": report["rank0_step_time_ms"],
            "forward_ms": report["rank0_forward_step_duration_sum_ms"],
            "backward_ms": report["rank0_backward_step_duration_sum_ms"],
            "optimizer_ms": report["rank0_optimizer_step_duration_sum_ms"],
            "simulator_load_time_s": report["simulator_load_time_s"],
            "simulator_execution_time_s": report["simulator_execution_time_s"],
            "simulator_wall_clock_s": report["simulator_wall_clock_s"],
            "process_wall_clock_s": measurement["wall_clock_s"],
            "peak_rss_kib": measurement["peak_rss_kib"],
            "peak_rss_gib": measurement["peak_rss_gib"],
            "tested_host_allocation_mib": measurement["tested_host_allocation_mib"],
            "manifest_file_count": len(manifest["files"]),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "schedule_stage_count": len(schedule_paths),
            "schedule_microbatches_per_stage": values["microbatches"],
        }
    )

summary = {
    "schema_version": "sc26-ae-task3-prebaked-synthetic-summary-v1",
    "execution_evidence": "local_synthetic_not_gpu_qualification",
    "model_count": len(rows),
    "rows": rows,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for row in rows:
    print(
        "MODEL={model} RANK0_STEP_MS={rank0_step_time_ms} FORWARD_MS={forward_ms} "
        "BACKWARD_MS={backward_ms} OPTIMIZER_MS={optimizer_ms} LOAD_S={simulator_load_time_s} "
        "EXECUTION_S={simulator_execution_time_s} SIM_WALL_S={simulator_wall_clock_s} "
        "PROCESS_WALL_S={process_wall_clock_s} PEAK_RSS_KIB={peak_rss_kib} "
        "PEAK_RSS_GIB={peak_rss_gib} TESTED_HOST_ALLOCATION_MIB={tested_host_allocation_mib} "
        "SCHEDULE_STAGES={schedule_stage_count} "
        "SCHEDULE_MICROBATCHES_PER_STAGE={schedule_microbatches_per_stage} "
        "MANIFEST_FILE_COUNT={manifest_file_count} MANIFEST_SHA256={manifest_sha256}".format(**row)
    )
PY

printf 'PASS: all three Task3 public entries completed the strict prebaked CPU synthetic workflow.\n'
printf 'MODEL_PASS_COUNT=3\n'
printf 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification\n'
printf 'SUMMARY_JSON=%s\n' "${SUMMARY_JSON}"
printf 'EVIDENCE_ROOT=%s\n' "${TEST_ROOT}"
