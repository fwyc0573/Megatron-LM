## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added formal seq8192 phase-pure repeat-x5 freeze report (distributed + scaling DDP on/off), including per-run compare, kernel-family robust stats, and freeze verdict |

## Test Report: Seq8192 Phase-pure Repeat x5 Semantics Freeze (DDP on/off)

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Scope

1. Execute formal `repeat x5` protocol at `SEQ_LEN=8192` with unchanged phase-pure config.
2. For each run, collect:
   - distributed NSYS
   - scaling NSYS (`SCALING_DISABLE_DDP_WRAP=0`, DDP on)
   - scaling NSYS (`SCALING_DISABLE_DDP_WRAP=1`, DDP off)
3. Export/analyze/compare per run with contamination gating.
4. Produce `stage1 backward rank4..7` kernel-family robust statistics:
   - per-run top-k
   - cross-run median/IQR
5. Decide whether semantics can be frozen and whether priority should shift to attention-family diagnostics.

### 2) Reproducible Commands

```bash
# Repeat x5 end-to-end pipeline (profile + export + analyze + compare)
bash -lc '
set -euo pipefail
BASE=task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_repeat5
mkdir -p "$BASE"
for run in 1 2 3 4 5; do
  # distributed
  nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --output "$BASE/deepseek_phase_sl8192_run${run}_dist" \
    bash -lc "MODE=distributed MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global MASTER_PORT=$((7600 + run)) bash examples/pretrain_deepseek_v3_moe.sh"
  nsys export --type sqlite --force-overwrite=true \
    --output "$BASE/deepseek_phase_sl8192_run${run}_dist" \
    "$BASE/deepseek_phase_sl8192_run${run}_dist.nsys-rep"
  python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
    --sqlite "$BASE/deepseek_phase_sl8192_run${run}_dist" \
    --ops forward_step,backward_step,optimizer_step \
    --json-path "$BASE/deepseek_phase_sl8192_run${run}_dist_breakdown.json" \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_dist_breakdown.md"

  # scaling DDP on
  nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --output "$BASE/deepseek_phase_sl8192_run${run}_scaling_on" \
    bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=0 SCALE_GPU=7 MASTER_PORT=$((7800 + run * 10)) bash examples/pretrain_deepseek_v3_moe.sh"
  nsys export --type sqlite --force-overwrite=true \
    --output "$BASE/deepseek_phase_sl8192_run${run}_scaling_on" \
    "$BASE/deepseek_phase_sl8192_run${run}_scaling_on.nsys-rep"
  python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
    --sqlite "$BASE/deepseek_phase_sl8192_run${run}_scaling_on" \
    --ops forward_step,backward_step,optimizer_step \
    --json-path "$BASE/deepseek_phase_sl8192_run${run}_scaling_on_breakdown.json" \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_scaling_on_breakdown.md"

  # scaling DDP off
  nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --output "$BASE/deepseek_phase_sl8192_run${run}_scaling_off" \
    bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=1 SCALE_GPU=7 MASTER_PORT=$((7900 + run * 10)) bash examples/pretrain_deepseek_v3_moe.sh"
  nsys export --type sqlite --force-overwrite=true \
    --output "$BASE/deepseek_phase_sl8192_run${run}_scaling_off" \
    "$BASE/deepseek_phase_sl8192_run${run}_scaling_off.nsys-rep"
  python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
    --sqlite "$BASE/deepseek_phase_sl8192_run${run}_scaling_off" \
    --ops forward_step,backward_step,optimizer_step \
    --json-path "$BASE/deepseek_phase_sl8192_run${run}_scaling_off_breakdown.json" \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_scaling_off_breakdown.md"

  # compare (primary + union)
  python tests/performance/compare_qwen_nsys_compute_only.py \
    --distributed-json "$BASE/deepseek_phase_sl8192_run${run}_dist_breakdown.json" \
    --scaling-json "$BASE/deepseek_phase_sl8192_run${run}_scaling_on_breakdown.json" \
    --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step \
    --compute-metric pure_primary_union --kernel-scope shared --shared-kernel-source primary_stream \
    --require-low-contamination-pct 1 \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_compare_on_pure_primary_union_shared.log" || true
  python tests/performance/compare_qwen_nsys_compute_only.py \
    --distributed-json "$BASE/deepseek_phase_sl8192_run${run}_dist_breakdown.json" \
    --scaling-json "$BASE/deepseek_phase_sl8192_run${run}_scaling_off_breakdown.json" \
    --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step \
    --compute-metric pure_primary_union --kernel-scope shared --shared-kernel-source primary_stream \
    --require-low-contamination-pct 1 \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_compare_off_pure_primary_union_shared.log" || true
  python tests/performance/compare_qwen_nsys_compute_only.py \
    --distributed-json "$BASE/deepseek_phase_sl8192_run${run}_dist_breakdown.json" \
    --scaling-json "$BASE/deepseek_phase_sl8192_run${run}_scaling_on_breakdown.json" \
    --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step \
    --compute-metric pure_union --kernel-scope shared --shared-kernel-source primary_stream \
    --require-low-contamination-pct 1 \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_compare_on_pure_union_shared.log" || true
  python tests/performance/compare_qwen_nsys_compute_only.py \
    --distributed-json "$BASE/deepseek_phase_sl8192_run${run}_dist_breakdown.json" \
    --scaling-json "$BASE/deepseek_phase_sl8192_run${run}_scaling_off_breakdown.json" \
    --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step \
    --compute-metric pure_union --kernel-scope shared --shared-kernel-source primary_stream \
    --require-low-contamination-pct 1 \
    --report-path "$BASE/deepseek_phase_sl8192_run${run}_compare_off_pure_union_shared.log" || true

done
'

# Aggregate compare + kernel-family robust stats
python - <<'PY'
# (Executed inline during this round; outputs below)
# Produces:
# - logs/nsys_phase_repeat5/deepseek_phase_sl8192_repeat5_ddp_onoff_summary.json
# - logs/nsys_phase_repeat5/deepseek_phase_sl8192_repeat5_ddp_onoff_summary.md
PY
```

### 3) Validation Criteria

1. `repeat x5` artifacts are complete for distributed/scaling-on/scaling-off.
2. contamination gate is satisfied for all runs.
3. Evaluate DDP-off improvement robustness (median + IQR), not single-run.
4. Output kernel-family top-k per run and cross-run robust stats; verify whether `fmha_cutlassB` is consistently dominant.
5. Decide freeze status and next-priority direction.

### 4) Results and Evidence

#### 4.1 Artifact completeness

- All run artifacts exist under:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_repeat5/`
- Generated summaries:
  - `.../deepseek_phase_sl8192_repeat5_ddp_onoff_summary.json`
  - `.../deepseek_phase_sl8192_repeat5_ddp_onoff_summary.md`

#### 4.2 Contamination gate

Across all 5 runs:
- distributed `max_contam_pct = 0.0`
- scaling-on `max_contam_pct = 0.0`
- scaling-off `max_contam_pct = 0.0`

Interpretation: phase-pure semantics remains clean; residual is not from comm leakage into compute window.

#### 4.3 Compare repeat-x5 robustness (`pure_primary_union`, shared)

From `deepseek_phase_sl8192_repeat5_ddp_onoff_summary.md`:

- DDP-on (run1..5):
  - forward: `[8.01, 8.91, 3.63, 6.04, 6.28]`, median `6.28`, IQR `1.97`
  - backward: `[14.55, 19.16, 6.52, 13.02, 10.32]`, median `13.02`, IQR `4.23`
  - optimizer: `[5.07, 0.82, 2.14, 2.07, 3.24]`, median `2.14`, IQR `1.17`

- DDP-off (run1..5):
  - forward: `[12.55, 3.51, 5.85, 5.77, 4.47]`, median `5.77`, IQR `1.38`
  - backward: `[18.68, 2.42, 12.27, 14.19, 8.93]`, median `12.27`, IQR `5.26`
  - optimizer: `[4.74, 5.64, 3.71, 3.34, 2.45]`, median `3.71`, IQR `1.40`

Interpretation:
- DDP-off relative to DDP-on:
  - backward median improves only `13.02 -> 12.27` (partial, still far above 5%).
  - backward IQR worsens (`4.23 -> 5.26`), indicating stability did not improve.
  - optimizer median regresses (`2.14 -> 3.71`).

#### 4.4 Metric-mode sweep (`pure_union` vs `pure_primary_union`)

- For both DDP-on/off, all 5 runs are identical between `pure_union` and `pure_primary_union` for op-rank median values.
- Backward mode-sweep fluctuation = `0.00 pp` (<=2pp criterion satisfied).

#### 4.5 Stage1 backward kernel-family robust stats (rank4..7, steady)

From `deepseek_phase_sl8192_repeat5_ddp_onoff_summary.md`:

- Per-run top1 by absolute delta is `fmha_cutlassB...` for **all 5/5 runs** in both branches:
  - DDP-on: fmha top1 count `5/5`
  - DDP-off: fmha top1 count `5/5`

- `fmha_cutlassB` delta (scale - dist) across runs:
  - DDP-on: `[45.459, 41.118, 38.447, 23.553, 27.748]`, median/IQR = `38.447 / 13.370`
  - DDP-off: `[55.955, 30.666, 36.539, 23.782, 28.769]`, median/IQR = `30.666 / 7.770`

- Non-fmha kernels remain much smaller (typically low-single-digit ms medians).

Interpretation: `fmha_cutlassB` is consistently dominant and robustly larger than other families across runs.

### 5) Freeze Check (Current Round)

Using this formal repeat-x5 dataset:

1. Backward <=5%: **FAIL** (`12.27%` best median with DDP-off)
2. Backward IQR <=1.0pp: **FAIL** (`4.23` on / `5.26` off)
3. Mode sweep fluctuation <=2pp: **PASS** (`0.00pp`)
4. Forward/optimizer keep low error: **PARTIAL** (forward still >5 in median; optimizer branch-dependent)

### 6) Verdict

1. Phase-pure semantics is validated (contamination=0), but freeze conditions are not met.
2. Repeat-x5 confirms DDP-off is only a partial lever and cannot close backward residual alone.
3. `fmha_cutlassB` remains the stable dominant residual source (top1 in 5/5 runs, both branches).
4. Next priority should move to attention-family diagnostics (debug-only tags/segmentation), not further comm-adjacent emulation expansion.

