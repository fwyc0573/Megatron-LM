## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Executed patched seq8192 phase-pure x1 validation (dist/scaling_on/scaling_off) with newly added NVTX structural-health gate, completed analyze/compare, and produced post-fix decision evidence |

## Test Report: Patched Seq8192 Phase-pure x1 + NVTX Structural Gate

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Scope

1. Validate patched code under `seq8192 phase-pure x1` protocol:
   - distributed
   - scaling DDP-on
   - scaling DDP-off
2. Enforce a new structural gate before compare:
   - `open_forward_step == 0`
   - `open_backward_step == 0`
   - `forward_backward_overlap_count == 0`
3. Reuse existing `analyze_nsys_cmd_kernel_breakdown.py` + `compare_qwen_nsys_compute_only.py` workflow.
4. Decide next priority by user rule:
   - if clean x1 backward still >5%, shift to attention-family diagnosis.

### 2) Reproducible Commands

```bash
BASE=task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_patched_x1
mkdir -p "$BASE"

# 1) Capture + export (dist / scaling_on / scaling_off)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$BASE/deepseek_phase_sl8192_patched_x1_dist" \
  bash -lc "MODE=distributed MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global MASTER_PORT=8611 bash examples/pretrain_deepseek_v3_moe.sh"
nsys export --type sqlite --force-overwrite=true \
  --output "$BASE/deepseek_phase_sl8192_patched_x1_dist" \
  "$BASE/deepseek_phase_sl8192_patched_x1_dist.nsys-rep"

nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on" \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=0 SCALE_GPU=7 MASTER_PORT=8621 bash examples/pretrain_deepseek_v3_moe.sh"
nsys export --type sqlite --force-overwrite=true \
  --output "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on" \
  "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on.nsys-rep"

nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off" \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=1 SCALE_GPU=7 MASTER_PORT=8631 bash examples/pretrain_deepseek_v3_moe.sh"
nsys export --type sqlite --force-overwrite=true \
  --output "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off" \
  "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off.nsys-rep"

# 2) NVTX structural-health gate (new)
python tests/performance/check_nsys_nvtx_structural_health.py \
  --sqlite "$BASE/deepseek_phase_sl8192_patched_x1_dist" \
  --label-prefix cmd_trace --ops forward_step,backward_step \
  --max-open-forward 0 --max-open-backward 0 --max-overlap-count 0 \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_dist.nvtx_gate.log"

python tests/performance/check_nsys_nvtx_structural_health.py \
  --sqlite "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on" \
  --label-prefix cmd_trace --ops forward_step,backward_step \
  --max-open-forward 0 --max-open-backward 0 --max-overlap-count 0 \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on.nvtx_gate.log"

python tests/performance/check_nsys_nvtx_structural_health.py \
  --sqlite "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off" \
  --label-prefix cmd_trace --ops forward_step,backward_step \
  --max-open-forward 0 --max-open-backward 0 --max-overlap-count 0 \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off.nvtx_gate.log"

# 3) Analyze
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite "$BASE/deepseek_phase_sl8192_patched_x1_dist" \
  --ops forward_step,backward_step,optimizer_step \
  --json-path "$BASE/deepseek_phase_sl8192_patched_x1_dist_breakdown.json" \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_dist_breakdown.md"

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on" \
  --ops forward_step,backward_step,optimizer_step \
  --json-path "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on_breakdown.json" \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on_breakdown.md"

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off" \
  --ops forward_step,backward_step,optimizer_step \
  --json-path "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off_breakdown.json" \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off_breakdown.md"

# 4) Compare
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json "$BASE/deepseek_phase_sl8192_patched_x1_dist_breakdown.json" \
  --scaling-json "$BASE/deepseek_phase_sl8192_patched_x1_scaling_on_breakdown.json" \
  --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union --kernel-scope shared --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_compare_on_pure_primary_union_shared.log"

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json "$BASE/deepseek_phase_sl8192_patched_x1_dist_breakdown.json" \
  --scaling-json "$BASE/deepseek_phase_sl8192_patched_x1_scaling_off_breakdown.json" \
  --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union --kernel-scope shared --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path "$BASE/deepseek_phase_sl8192_patched_x1_compare_off_pure_primary_union_shared.log"
```

### 3) Validation Criteria

1. Structural gate must PASS on all three captures.
2. analyzer contamination should remain zero.
3. Compare output should determine whether backward is still significantly above threshold.
4. If backward still high in clean x1, switch next priority to attention-family diagnosis.

### 4) Results and Evidence

#### 4.1 NVTX structural-health gate (new)

All three captures PASS with strict thresholds (`0/0/0`):

- `deepseek_phase_sl8192_patched_x1_dist.nvtx_gate.log`
  - `open_forward_step=0`
  - `open_backward_step=0`
  - `forward_backward_overlap_count=0`
- `deepseek_phase_sl8192_patched_x1_scaling_on.nvtx_gate.log`
  - `open_forward_step=0`
  - `open_backward_step=0`
  - `forward_backward_overlap_count=0`
- `deepseek_phase_sl8192_patched_x1_scaling_off.nvtx_gate.log`
  - `open_forward_step=0`
  - `open_backward_step=0`
  - `forward_backward_overlap_count=0`

Interpretation: patched run no longer exhibits the prior NVTX CMD structure corruption.

#### 4.2 Analyzer sanity

From `*.analyze.log`:

- all branches: `phase_window_parents=48`, `event_rows=72`, `aggregate_rows=24`
- all branches: contamination mean is `0.00%`.

Interpretation: phase-pure semantic cleanliness remains intact.

#### 4.3 Compare (`pure_primary_union`, shared primary stream)

- DDP-on (`...compare_on...log`):
  - `forward_step` rank-median diff: `15.63%`
  - `backward_step` rank-median diff: `17.42%`
  - `optimizer_step` rank-median diff: `3.51%`
  - script return code: FAIL

- DDP-off (`...compare_off...log`):
  - `forward_step` rank-median diff: `15.74%`
  - `backward_step` rank-median diff: `10.52%`
  - `optimizer_step` rank-median diff: `2.35%`
  - script return code: FAIL

Interpretation:
- Even after structural cleanup, backward remains clearly above 5% in x1.
- DDP-off improves backward (`17.42% -> 10.52%`) but remains insufficient.

#### 4.4 Attention-family evidence in patched clean x1

Additional stage1-backward kernel-family delta summary:

- file: `deepseek_phase_sl8192_patched_x1_stage1_backward_kernel_delta_summary.md`
- scope: `op=backward_step`, `state=steady`, `stage=1`, `rank=4..7`, paired rows `12/12`
- top delta in both on/off is still attention backward kernel:
  - scaling_on: `fmha_cutlassB ... +25.442 ms`
  - scaling_off: `fmha_cutlassB ... +20.650 ms`

Interpretation: attention-family remains dominant residual component after NVTX structural fix.

### 5) Verdict and Decision

1. Step-1/2 goals are met:
   - patched seq8192 x1 completed;
   - NVTX structural gate added and enforced; all three captures pass.
2. Step-3 condition is triggered:
   - clean x1 backward still significantly above threshold.
3. Therefore next priority should switch to attention-family diagnosis (debug-only tags/segmentation), not direct repeat-x5 freeze yet.

