## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Validated NVTX attribution artifact hypothesis on seq8192 phase-repeat traces, identified unbalanced NVTX push/pop root cause in TP mapping path, implemented minimal fix, and completed RED→GREEN verification |

## Test Report: NVTX Attribution Artifact Validation and Root-Cause Fix

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)

### 1) Test Scope

1. Validate whether backward residual can be explained by NVTX attribution artifact (forward/backward CMD overlap and unclosed ranges).
2. Locate concrete root cause in code path.
3. Implement minimal invasive fix and verify with unit/regression tests.

### 2) Reproducible Commands

#### 2.1 Existing NSYS sqlite diagnosis (no new capture)

```bash
# Forward/backward overlap + open CMD statistics across repeat-x5 (dist/scaling_on/scaling_off)
python - <<PY > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nvtx_overlap_diagnosis_20260301.log
# (inline script used in this round; see log file for full output)
PY

# Unclosed label fingerprint (row_g_fwd / model_fwd_step / cmd forward)
python - <<PY > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nvtx_open_label_diagnosis_20260301.log
# (inline script used in this round; see log file for full output)
PY

# Concrete example: rank4 stage1 steady forward/backward nesting (run1 dist)
python - <<PY > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nvtx_rank4_stage1_nested_example_run1.log
# (inline script used in this round; see log file for full output)
PY
```

#### 2.2 RED test (before fix)

```bash
python -m pytest tests/unit_tests/tensor_parallel/test_mappings_moe_api.py -q
```

#### 2.3 GREEN + regression tests (after fix)

```bash
python -m pytest tests/unit_tests/tensor_parallel/test_mappings_moe_api.py -q
python -m pytest tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py -q
python -m py_compile megatron/core/tensor_parallel/mappings.py \
  tests/unit_tests/tensor_parallel/test_mappings_moe_api.py
```

### 3) Validation Criteria

1. If attribution artifact exists, repeated traces should show:
   - systematic forward/backward CMD overlaps;
   - multiple forward CMD ranges ending at session max timestamp (unclosed symptom).
2. Root cause must be code-level deterministic bug, not statistical coincidence.
3. Fix must be minimal and verified by explicit failing->passing tests.

### 4) Results and Evidence

#### 4.1 Artifact is real and systematic (repeat-x5 all branches)

From `logs/nvtx_overlap_diagnosis_20260301.log`:

- Every run/branch shows identical structural anomaly:
  - `open_forward=24`, `open_backward=0`
  - `overlap_cnt=48`
  - `overlap_ms` is large (e.g., run01 dist `25927.7 ms`)

Interpretation:
- backward CMD ranges are repeatedly nested inside forward CMD ranges;
- this is incompatible with expected per-op closed windows and confirms attribution pollution risk.

#### 4.2 Unclosed-label fingerprint points to TP mapping NVTX stack leak

From `logs/nvtx_open_label_diagnosis_20260301.log`:

- `row_g_fwd_open` is consistently non-zero in every run/branch:
  - dist: `48`
  - scaling_on/off: `96`
- `cmd_forward_open=24` in every run/branch.

From `logs/nvtx_rank4_stage1_nested_example_run1.log`:

- rank4 stage1 steady forward CMD windows share the same final end timestamp;
- each backward window is fully overlapped by one or more forward windows.

Interpretation:
- forward CMD pop sequence is corrupted by leaked NVTX pushes;
- leaked `row_g_fwd` is a stable fingerprint across all traces.

#### 4.3 Root cause located (code-level)

Root cause in `megatron/core/tensor_parallel/mappings.py:238`:

- `_ReduceFromModelParallelRegion.forward` did `nvtx.range_push("row_g_fwd")`;
- when `get_tensor_model_parallel_world_size() == 1`, function returned early without `nvtx.range_pop()`;
- this leaks NVTX stack entries and shifts subsequent pop targets, causing CMD-level attribution corruption.

#### 4.4 Fix implemented (minimal, fail-fast)

Code fix in `megatron/core/tensor_parallel/mappings.py:238`:

- wrapped `row_g_fwd` region with `try/finally`;
- guaranteed `nvtx.range_pop()` on all return paths (including world_size==1 fast path).

New unit coverage in `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`:

- `test_reduce_from_model_parallel_region_nvtx_balanced_world_size_one`
- `test_reduce_from_model_parallel_region_nvtx_balanced_world_size_gt_one`

Both validate push/pop balance explicitly.

#### 4.5 RED -> GREEN evidence

- RED (before fix):
  - `python -m pytest tests/unit_tests/tensor_parallel/test_mappings_moe_api.py -q`
  - result: `1 failed, 8 passed`
  - failing test: `test_reduce_from_model_parallel_region_nvtx_balanced_world_size_one`
  - failure detail: missing `("pop", None)` event.

- GREEN (after fix):
  - same command result: `9 passed`

- Additional regression:
  - profiler/performance unit suites: `22 passed`
  - `py_compile`: PASS

### 5) Conclusion

1. User hypothesis is **substantiated**: there is a real NVTX attribution artifact caused by instrumentation stack corruption, not just random variance.
2. The direct root cause is confirmed as **unbalanced NVTX push/pop** in TP mapping world_size==1 fast path.
3. This artifact can pollute per-op kernel ownership (especially forward/backward boundary semantics), so previous op-window conclusions based on affected traces are not reliable enough for freeze.
4. Minimal fix is landed and unit/regression checks are green.

### 6) Next Actions (for semantic freeze)

1. Re-capture one fresh seq8192 phase-pure distributed/scaling sanity run with patched code, then verify:
   - `open_forward==0`
   - no systematic forward/backward CMD overlap.
2. If sanity passes, rerun formal repeat-x5 freeze protocol on patched branch.
3. Re-evaluate backward residual on clean-attribution traces before deciding whether to continue attention-family deep diagnostics.
