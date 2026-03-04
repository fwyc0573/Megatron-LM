## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added immediate same-stream adjacency diagnostics for `attn_core_sdpa_bwd`, reran x1 on existing sqlite with threshold sweep (`60us`/`80us`), and verified threshold sensitivity vs residual stability |

## Test Report: SDPA Immediate-Neighbor Diagnosis (x1 Re-analysis)

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)

### 1) Scope

1. Extend attention-family analyzer with **immediate same-stream neighbor** diagnostics around `fmha_cutlassB`.
2. Keep training/capture untouched (debug analysis only), reuse existing x1 sqlite artifacts:
   - `dist`
   - `scaling_on`
   - `scaling_off`
3. Validate whether pre-fmha small-kernel difference is real insertion asymmetry or threshold artifact.

### 2) Code Changes

- `tests/performance/analyze_nsys_attention_family_delta.py`
  - Added immediate same-stream metrics around fmha:
    - `small_kernel_immediate_pre_count/ms`
    - `small_kernel_immediate_post_count/ms`
    - immediate pre/post top names (`name/ms/count`)
  - Added report + JSON payload sections for these fields.
- `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py`
  - Added immediate-neighbor assertions.
  - Added synthetic test for immediate pre/post detection on same stream.

### 3) Reproducible Commands

```bash
# 1) Static + unit
python -m py_compile \
  tests/performance/analyze_nsys_attention_family_delta.py \
  tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py -q

# 2) Re-analyze existing x1 sqlite (threshold=60us, existing default)
BASE=task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1
DIST="$BASE/deepseek_phase_sl8192_attncore_x1_dist"
for MODE in on off; do
  SCALE="$BASE/deepseek_phase_sl8192_attncore_x1_scaling_${MODE}"
  python tests/performance/analyze_nsys_attention_family_delta.py \
    --dist-sqlite "$DIST" \
    --scale-sqlite "$SCALE" \
    --label-prefix cmd_trace \
    --ranks 4,5,6,7 \
    --op backward_step --mg-state steady --stage-id 1 --phase compute \
    --segment-key attn_bwd_segment --segment-values attn_core_sdpa_bwd \
    --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_${MODE}_attn_core_sdpa_bwd_diag_immediate.md" \
    --json-path "$BASE/deepseek_phase_sl8192_attncore_x1_${MODE}_attn_core_sdpa_bwd_diag_immediate.json"
done

# 3) Threshold sweep (80us)
for MODE in on off; do
  SCALE="$BASE/deepseek_phase_sl8192_attncore_x1_scaling_${MODE}"
  python tests/performance/analyze_nsys_attention_family_delta.py \
    --dist-sqlite "$DIST" \
    --scale-sqlite "$SCALE" \
    --label-prefix cmd_trace \
    --ranks 4,5,6,7 \
    --op backward_step --mg-state steady --stage-id 1 --phase compute \
    --segment-key attn_bwd_segment --segment-values attn_core_sdpa_bwd \
    --small-kernel-threshold-us 80 \
    --adjacency-window-us 200 \
    --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_${MODE}_attn_core_sdpa_bwd_diag_immediate_th80.md" \
    --json-path "$BASE/deepseek_phase_sl8192_attncore_x1_${MODE}_attn_core_sdpa_bwd_diag_immediate_th80.json"
done
```

### 4) Validation Criteria

1. Analyzer unit tests pass after payload/schema change.
2. `attn_core_sdpa_bwd` reports include immediate same-stream fields.
3. Determine whether distributed/scaling pre-fmha difference is insertion asymmetry or threshold-sensitive classification.

### 5) Results

#### 5.1 Unit/static checks

- `py_compile`: PASS
- `test_analyze_nsys_attention_family_delta.py`: `7 passed`

#### 5.2 Immediate-neighbor evidence (`stage1/backward/steady/rank4..7`)

For both scaling branches, immediate pre-fmha top kernel remains:
- `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`

Threshold = `60us` (default):
- scaling_on:
  - `gap_ms=23.115` (unchanged)
  - immediate pre count: `dist=35`, `scale=27`
  - immediate pre ms: `dist=1.751`, `scale=1.393`
- scaling_off:
  - `gap_ms=33.740` (unchanged)
  - immediate pre count: `dist=35`, `scale=18`
  - immediate pre ms: `dist=1.751`, `scale=0.900`

Threshold = `80us`:
- scaling_on:
  - `gap_ms=23.115` (unchanged)
  - immediate pre count: `dist=44`, `scale=44`
  - immediate pre ms: `dist=2.311`, `scale=2.451`
- scaling_off:
  - `gap_ms=33.740` (unchanged)
  - immediate pre count: `dist=44`, `scale=44`
  - immediate pre ms: `dist=2.311`, `scale=2.518`

Immediate post-fmha small-kernel counts remain `0` for all compared branches.

### 6) Interpretation

1. Pre-fmha neighboring kernel family is **the same** across distributed/scaling and sits on the same stream as fmha.
2. Observed count asymmetry under `60us` is largely a **threshold sensitivity artifact** (classification cutoff), not a missing-path insertion asymmetry.
3. Backward residual (`gap_ms`) is unaffected by threshold sweep, so dominant mismatch remains inside `attn_core_sdpa_bwd` fmha runtime context.

### 7) Artifacts

- Analyzer outputs (default threshold):
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_on_attn_core_sdpa_bwd_diag_immediate.md`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_off_attn_core_sdpa_bwd_diag_immediate.md`
- Analyzer outputs (`80us` sweep):
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_on_attn_core_sdpa_bwd_diag_immediate_th80.md`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_off_attn_core_sdpa_bwd_diag_immediate_th80.md`
