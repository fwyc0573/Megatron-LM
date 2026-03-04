## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added attention-family residual diagnosis on patched seq8192 phase-pure x1 (`stage1 backward rank4..7`) with dedicated analyzer script, unit tests, and on/off evidence |

## Test Report: Patched Seq8192 x1 Attention-family Diagnosis

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1)

### 1) Test Scope

1. Complete and harden `tests/performance/analyze_nsys_attention_family_delta.py` for attention-path diagnosis.
2. Validate the analyzer with dedicated unit tests (parser / pairing / report / IQR).
3. Run diagnosis on patched clean-x1 sqlite captures:
   - distributed vs scaling_on
   - distributed vs scaling_off
4. Confirm whether backward residual is still dominated by attention-family (`fmha_cutlassB`) after NVTX structural cleanup.

### 2) Reproducible Commands

```bash
BASE=task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_patched_x1

# 1) Unit tests for new analyzer
python -m pytest tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py -q

# 2) Regression tests for performance analyzers/compare
python -m pytest \
  tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py \
  tests/unit_tests/performance/test_check_nsys_nvtx_structural_health.py -q

# 3) Static syntax check
python -m py_compile \
  tests/performance/analyze_nsys_attention_family_delta.py \
  tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py

# 4) Attention diagnosis (distributed vs scaling_on)
python tests/performance/analyze_nsys_attention_family_delta.py \
  --dist-sqlite "$BASE/deepseek_phase_sl8192_patched_x1_dist" \
  --scale-sqlite "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on" \
  --label-prefix cmd_trace \
  --ranks 4,5,6,7 \
  --op backward_step \
  --mg-state steady \
  --stage-id 1 \
  --phase compute \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_on.md" \
  --json-path "$BASE/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_on.json"

# 5) Attention diagnosis (distributed vs scaling_off)
python tests/performance/analyze_nsys_attention_family_delta.py \
  --dist-sqlite "$BASE/deepseek_phase_sl8192_patched_x1_dist" \
  --scale-sqlite "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off" \
  --label-prefix cmd_trace \
  --ranks 4,5,6,7 \
  --op backward_step \
  --mg-state steady \
  --stage-id 1 \
  --phase compute \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_off.md" \
  --json-path "$BASE/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_off.json"
```

### 3) Validation Criteria

1. New analyzer unit tests pass (focus on correctness of pairing/order/stat aggregation).
2. Regression suites for performance analyzer path remain green.
3. Analyzer outputs must include:
   - paired-window totals and gap
   - fmha launch-config parity
   - per-rank fmha duration mean/p50/IQR
   - top kernel deltas
4. Evidence must be sufficient to decide whether attention-family remains dominant.

### 4) Results and Evidence

#### 4.1 Test results

| Suite | Result | Evidence |
|------|--------|---------|
| `test_analyze_nsys_attention_family_delta.py` | PASS | `5 passed in 0.04s` |
| Performance analyzer regression set | PASS | `26 passed in 0.05s` |
| `py_compile` | PASS | exit code `0` |

#### 4.2 Diagnosis output (stage1 backward, steady, rank4..7, phase=compute)

- scaling_on report: `logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_on.md`
- scaling_off report: `logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_off.md`

Key metrics:

| Comparison | paired_windows | gap_ms (scale-dist) | fmha_gap_ms | fmha_gap_share_pct |
|------------|----------------|---------------------|-------------|--------------------|
| dist vs scaling_on | 12 | +42.122 | +25.442 | 60.40% |
| dist vs scaling_off | 12 | +28.344 | +20.650 | 72.86% |

Launch parity (both comparisons):
- `dist_unique_cfg=1`, `scale_unique_cfg=1`, `cfg_sets_equal=True`
- Conclusion: launch config mismatch is not the source of fmha residual.

Per-rank fmha mean/p50 trend (us):
- rank5 is closest to parity (off mean ratio `~0.998x`), but rank4/rank6/rank7 remain `>1.03x` to `>1.15x`.
- rank6/7 show the most obvious p50 shift (`scale_p50` around `~10.1ms` vs dist `~8.2-8.3ms`).

Top kernel deltas:
- both on/off: top1 absolute delta is `fmha_cutlassB...`
- DDP-off only partially reduces fmha contribution (`25.442 -> 20.650 ms`), but fmha remains dominant.

### 5) Interpretation

1. Attention-family remains the dominant residual component after NVTX structural cleanup and phase-pure contamination control.
2. Residual is not explained by kernel launch shape mismatch (`cfg_sets_equal=True`).
3. DDP-off reduces total gap and fmha gap, but does not change dominance ordering or bring backward to gate threshold.
4. Next work should prioritize attention-path micro-segmentation / debug-only tags to locate where fmha slowdown accumulates (e.g., pre-attention normalization, qkv projection adjacency, stream scheduling around attention backward).

### 6) Artifacts

- Script:
  - `tests/performance/analyze_nsys_attention_family_delta.py`
- Unit test:
  - `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py`
- Output reports:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_on.md`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_off.md`
- Output json:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_on.json`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_off.json`
