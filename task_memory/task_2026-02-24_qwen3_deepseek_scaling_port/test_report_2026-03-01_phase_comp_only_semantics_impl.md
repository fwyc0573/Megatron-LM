## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added implementation validation report for phase-level kernel ground-truth instrumentation, NSYS pure-compute metrics, contamination gating, and script argument wiring |

## Test Report: Phase-level Comp-only Semantics Implementation

**Date**: 2026-03-01  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18), CUDA 12.1, Nsight Systems 2023.1.2.43  
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

---

### 1) Test Script Information

- Updated code paths:
  - `megatron/training/arguments.py`
  - `megatron/profiler/cmd.py`
  - `megatron/core/pipeline_parallel/schedules.py`
  - `megatron/training/training.py`
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/performance/compare_qwen_nsys_compute_only.py`
  - `examples/pretrain_deepseek_v3_moe.sh`
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
- Added/updated tests:
  - `tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py`
  - `tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py`
  - `tests/unit_tests/test_training.py`

---

### 2) Validation Criteria

1. Parser accepts new trace arguments and rejects invalid boundary sync values.
2. CMD emits phase-level NVTX labels (`phase=compute|comm`) under kernel-ground-truth mode.
3. NSYS analyzer outputs new pure-compute and contamination fields.
4. NSYS compare supports `--compute-metric pure_primary_union` and contamination gating.
5. Existing unit suites for affected modules remain green.
6. Updated scripts pass shell syntax checks.
7. Analyzer/compare can run on existing round68 sqlite artifacts in backward-compatible mode.

---

### 3) RED → GREEN (TDD evidence)

#### RED (expected fail before implementation)

```bash
pytest -q \
  tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py::test_cmd_kernel_ground_truth_phase_context_emits_phase_range \
  tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py::test_cmd_kernel_ground_truth_phase_disabled_skips_phase_range \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py::test_parse_cmd_nvtx_label_with_phase \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py::test_summarize_nvtx_ranges_phase_pure_metrics_and_contamination \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py::test_metric_from_row_supports_pure_primary_union \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py::test_main_fails_when_contamination_exceeds_threshold
```

- Result: **FAIL** (`5 failed, 1 passed`)
- Representative failures:
  - `AttributeError: 'CMD' object has no attribute 'phase_range'`
  - `AttributeError: module ... has no attribute 'range_identity'`
  - `ValueError: Unsupported compute metric: pure_primary_union`

#### GREEN (after implementation)

```bash
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py \
  tests/unit_tests/test_training.py::TestTraining::test_trace_kernel_ground_truth_defaults \
  tests/unit_tests/test_training.py::TestTraining::test_trace_kernel_ground_truth_args \
  tests/unit_tests/test_training.py::TestTraining::test_trace_kernel_ground_truth_phase_defaults \
  tests/unit_tests/test_training.py::TestTraining::test_trace_kernel_ground_truth_phase_args \
  tests/unit_tests/test_training.py::TestTraining::test_trace_kernel_boundary_sync_mode_invalid_value \
  tests/unit_tests/test_training_optimizer_microphase.py
```

- Result: **PASS** (`42 passed`)

---

### 4) Static/Syntax Validation

```bash
python -m py_compile \
  megatron/profiler/cmd.py \
  tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  tests/performance/compare_qwen_nsys_compute_only.py \
  megatron/training/arguments.py \
  megatron/training/training.py \
  megatron/core/pipeline_parallel/schedules.py

bash -n examples/pretrain_deepseek_v3_moe.sh
bash -n examples/pretrain_qwen3_30b_a3b_moe.sh
```

- Result: **PASS** (exit code 0)

---

### 5) Integration Replay Validation (existing round68 sqlite)

> Note: these are compatibility replays on historical traces (no new phase labels in source traces).

#### Analyzer replay

```bash
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_dist_sqlite \
  --label-prefix cmd_trace \
  --ops forward_step,backward_step,optimizer_step \
  --ranks 0,1,2,3,4,5,6,7 \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_dist_kernel_breakdown_phasecheck.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_dist_kernel_breakdown_phasecheck.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_scaling_sqlite \
  --label-prefix cmd_trace \
  --ops forward_step,backward_step,optimizer_step \
  --ranks 0,1,2,3,4,5,6,7 \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_scaling_kernel_breakdown_phasecheck.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_scaling_kernel_breakdown_phasecheck.md
```

- Result: **PASS**
- Key evidence:
  - analyzer reports `phase_window_parents=0` on historical traces (expected compatibility behavior)
  - new pure fields are still emitted with backward-compatible fallback

#### Compare replay (new metric + contamination gate)

```bash
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_dist_kernel_breakdown_phasecheck.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run5_scaling_kernel_breakdown_phasecheck.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 100 \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 10 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_round68_nsys_run5_compute_only_pure_phasecheck.log
```

- Result: **PASS** (`[RESULT] PASS (all checks within threshold).`)
- Key evidence:
  - output includes `require_low_contamination_pct=10.00`
  - per-row contamination columns are present (`dist_contam_pct`, `scale_contam_pct`)

---

### 6) Summary

| Suite | Result | Notes |
|------|--------|-------|
| New unit tests (phase/pure/contamination) | PASS | RED→GREEN completed |
| Existing affected unit suites | PASS | no regression in covered scope |
| Static/syntax checks | PASS | python + bash scripts |
| Integration replay on round68 sqlite | PASS | compatibility path confirmed |

### 7) Known Limitation (post-implementation)

- Historical round68 traces used in replay do not contain phase labels, so they cannot validate phase purification quality by themselves.
- Fresh NSYS captures with `TRACE_KERNEL_GROUND_TRUTH=1` and `TRACE_KERNEL_GROUND_TRUTH_PHASE=1` are still required for official backward gate freeze.
