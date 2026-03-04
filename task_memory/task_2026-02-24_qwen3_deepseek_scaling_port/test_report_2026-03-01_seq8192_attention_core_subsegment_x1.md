## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added MLA attention-core micro-segment debug-only hooks (`attn_core_precast_bwd` / `attn_core_sdpa_bwd` / `attn_core_postcast_bwd`), upgraded adjacency top-k diagnostics, and completed seq8192 clean-x1 (dist/scaling on/off) localization run |

## Test Report: Seq8192 Attention-Core Micro-Segment x1

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Scope

1. Keep training semantics unchanged; only add debug-only attention-core backward segmentation inside MLA SDPA core.
2. Reuse phase-pure pipeline and run clean x1 (`distributed + scaling_on + scaling_off`) on `SEQ_LEN=8192`.
3. Verify NVTX structure gate and phase contamination gate are both clean.
4. Identify whether residual is in `attn_core_sdpa_bwd` or in non-SDPA subsegments.
5. Add pre-fmha adjacent small-kernel top-name evidence to the analyzer.

### 2) Code Changes

- `megatron/core/transformer/multi_latent_attention.py`
  - Split `_MLASDPACoreAttention` into internal modules:
    - `_MLASDPAPreCast`
    - `_MLASDPABackend`
    - `_MLASDPAPostCast`
  - Register debug-only backward segment hooks:
    - `attn_core_precast_bwd`
    - `attn_core_sdpa_bwd`
    - `attn_core_postcast_bwd`
- `tests/performance/analyze_nsys_attention_family_delta.py`
  - Add per-name adjacency aggregation:
    - `small_kernel_adjacent_pre_name_count/ms`
    - `small_kernel_adjacent_post_name_count/ms`
  - Add report/json outputs for top-k adjacent kernel names (pre/post fmha).
- `tests/unit_tests/transformer/test_multi_latent_attention.py`
  - Update hook count and segment-name assertions for new core subsegments.
- `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py`
  - Add assertions for new adjacency-name fields and payload top-name entries.

### 3) Reproducible Commands

```bash
# 0) Static + unit
python -m py_compile \
  megatron/core/transformer/multi_latent_attention.py \
  tests/performance/analyze_nsys_attention_family_delta.py \
  tests/unit_tests/transformer/test_multi_latent_attention.py \
  tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py -q

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest tests/unit_tests/transformer/test_multi_latent_attention.py -q

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest tests/unit_tests/transformer/test_attention.py -k "attention_backward_segment_hooks" -q

# 1) x1 capture
BASE=task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1
mkdir -p "$BASE"

nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$BASE/deepseek_phase_sl8192_attncore_x1_dist" \
  bash -lc "MODE=distributed MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global TRACE_ATTENTION_BACKWARD_SEGMENTS=1 MASTER_PORT=8711 bash examples/pretrain_deepseek_v3_moe.sh"

nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on" \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global TRACE_ATTENTION_BACKWARD_SEGMENTS=1 SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=0 SCALE_GPU=7 MASTER_PORT=8721 bash examples/pretrain_deepseek_v3_moe.sh"

nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off" \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global TRACE_ATTENTION_BACKWARD_SEGMENTS=1 SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=1 SCALE_GPU=7 MASTER_PORT=8731 bash examples/pretrain_deepseek_v3_moe.sh"

nsys export --type sqlite --force-overwrite=true --output "$BASE/deepseek_phase_sl8192_attncore_x1_dist" "$BASE/deepseek_phase_sl8192_attncore_x1_dist.nsys-rep"
nsys export --type sqlite --force-overwrite=true --output "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on" "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on.nsys-rep"
nsys export --type sqlite --force-overwrite=true --output "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off" "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off.nsys-rep"

# 2) NVTX structure gate
python tests/performance/check_nsys_nvtx_structural_health.py --sqlite "$BASE/deepseek_phase_sl8192_attncore_x1_dist" --label-prefix cmd_trace --ops forward_step,backward_step --max-open-forward 0 --max-open-backward 0 --max-overlap-count 0 --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_dist.nvtx_gate.log"
python tests/performance/check_nsys_nvtx_structural_health.py --sqlite "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on" --label-prefix cmd_trace --ops forward_step,backward_step --max-open-forward 0 --max-open-backward 0 --max-overlap-count 0 --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on.nvtx_gate.log"
python tests/performance/check_nsys_nvtx_structural_health.py --sqlite "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off" --label-prefix cmd_trace --ops forward_step,backward_step --max-open-forward 0 --max-open-backward 0 --max-overlap-count 0 --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off.nvtx_gate.log"

# 3) phase-pure analyze + compare
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py --sqlite "$BASE/deepseek_phase_sl8192_attncore_x1_dist" --ops forward_step,backward_step,optimizer_step --json-path "$BASE/deepseek_phase_sl8192_attncore_x1_dist_breakdown.json" --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_dist_breakdown.md"
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py --sqlite "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on" --ops forward_step,backward_step,optimizer_step --json-path "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on_breakdown.json" --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on_breakdown.md"
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py --sqlite "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off" --ops forward_step,backward_step,optimizer_step --json-path "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off_breakdown.json" --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off_breakdown.md"

python tests/performance/compare_qwen_nsys_compute_only.py --distributed-json "$BASE/deepseek_phase_sl8192_attncore_x1_dist_breakdown.json" --scaling-json "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_on_breakdown.json" --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step --compute-metric pure_primary_union --kernel-scope shared --shared-kernel-source primary_stream --require-low-contamination-pct 1 --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_compare_on_pure_primary_union_shared.log"
python tests/performance/compare_qwen_nsys_compute_only.py --distributed-json "$BASE/deepseek_phase_sl8192_attncore_x1_dist_breakdown.json" --scaling-json "$BASE/deepseek_phase_sl8192_attncore_x1_scaling_off_breakdown.json" --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step --compute-metric pure_primary_union --kernel-scope shared --shared-kernel-source primary_stream --require-low-contamination-pct 1 --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_compare_off_pure_primary_union_shared.log"

# 4) segment-level diagnosis
for MODE in on off; do
  SCALE="$BASE/deepseek_phase_sl8192_attncore_x1_scaling_${MODE}"
  DIST="$BASE/deepseek_phase_sl8192_attncore_x1_dist"
  for SEG in attn_core_bwd attn_core_precast_bwd attn_core_sdpa_bwd attn_core_postcast_bwd; do
    python tests/performance/analyze_nsys_attention_family_delta.py \
      --dist-sqlite "$DIST" \
      --scale-sqlite "$SCALE" \
      --label-prefix cmd_trace \
      --ranks 4,5,6,7 \
      --op backward_step --mg-state steady --stage-id 1 --phase compute \
      --segment-key attn_bwd_segment --segment-values "$SEG" \
      --report-path "$BASE/deepseek_phase_sl8192_attncore_x1_${MODE}_${SEG}_diag.md" \
      --json-path "$BASE/deepseek_phase_sl8192_attncore_x1_${MODE}_${SEG}_diag.json"
  done
done
```

### 4) Validation Criteria

1. Unit/regression tests pass for changed logic.
2. NVTX structural gate must pass on dist/scaling_on/scaling_off.
3. Segment labels for new core subsegments must exist in all three captures.
4. Segment-level diagnosis should identify which subsegment dominates `fmha_cutlassB` residual.
5. Analyzer should emit adjacent small-kernel top names for pre-fmha neighborhood.

### 5) Results

#### 5.1 Unit and static checks

- `py_compile`: PASS
- `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py`: `6 passed`
- `tests/unit_tests/transformer/test_multi_latent_attention.py`: `4 passed`
- `tests/unit_tests/transformer/test_attention.py -k attention_backward_segment_hooks`: `2 passed, 5 deselected`

#### 5.2 NVTX/phase cleanliness

- Dist/scaling_on/scaling_off structural gate: all PASS
  - `open_forward_step=0`
  - `open_backward_step=0`
  - `forward_backward_overlap_count=0`
- Phase analyzer contamination: all rows remain `0.00%` (compare contamination gate passes).

#### 5.3 Segment-label sanity

In all three branches (`dist/scaling_on/scaling_off`), label counts are consistent:

- `attn_core_bwd`: 96
- `attn_core_precast_bwd`: 96
- `attn_core_sdpa_bwd`: 96
- `attn_core_postcast_bwd`: 96

#### 5.4 Core subsegment localization (`stage1/steady/rank4..7`)

From `deepseek_phase_sl8192_attncore_x1_summary.md/json`:

- `dist vs scaling_on`
  - `attn_core_bwd`: `gap_ms=23.419`, `fmha_gap_share=96.74%`
  - `attn_core_sdpa_bwd`: `gap_ms=23.115`, `fmha_gap_share=98.02%`
  - `attn_core_precast_bwd`: `gap_ms=0.000`
  - `attn_core_postcast_bwd`: `gap_ms=0.000`
- `dist vs scaling_off`
  - `attn_core_bwd`: `gap_ms=34.155`, `fmha_gap_share=96.92%`
  - `attn_core_sdpa_bwd`: `gap_ms=33.740`, `fmha_gap_share=98.11%`
  - `attn_core_precast_bwd`: `gap_ms=0.000`
  - `attn_core_postcast_bwd`: `gap_ms=0.000`

Interpretation: residual is effectively fully localized in `attn_core_sdpa_bwd`; pre/post cast segments are non-material.

#### 5.5 Pre-fmha adjacency top-k (new evidence)

`attn_core_sdpa_bwd` pre-fmha top adjacent small kernel (both on/off) is stable:

- kernel name:
  - `void at::native::vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`
- `dist`: `1.751 ms`, `35 kernels`
- `scaling_on`: `1.393 ms`, `27 kernels`
- `scaling_off`: `0.900 ms`, `18 kernels`

Interpretation: distributed-side pre-fmha neighbor load remains consistently higher, and the dominant adjacent kernel family is now concretely identified.

#### 5.6 Op-level compare snapshot (x1)

- scaling_on backward rank-median: `8.23%` (FAIL)
- scaling_off backward rank-median: `15.39%` (FAIL)

This report focuses on segment localization; official freeze still requires protocol-level repeat-x5 decision.

### 6) Verdict

1. Debug-only micro-seg instrumentation works and does not alter training semantics.
2. Localization converges: dominant residual is `attn_core_sdpa_bwd` (not pre/post cast subsegments).
3. New adjacency top-k evidence confirms stable pre-fmha `FillFunctor<unsigned char>` inflation on distributed side.
4. Next step should target `attn_core_sdpa_bwd` neighborhood-specific diagnostics (or direct repeat-x5 freeze if user prefers to lock current evidence first).

### 7) Key Artifacts

- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_summary.md`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_summary.json`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_on_attn_core_sdpa_bwd_diag.md`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_off_attn_core_sdpa_bwd_diag.md`
