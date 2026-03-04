## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added seq8192 phase-pure DDP probe report: distributed x1 + scaling DDP-hook on/off x1 NSYS capture/export/analyze/compare, with kernel-family attribution and verdict |

## Test Report: Seq8192 Phase-pure DDP Probe A/B Validation

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Scope

1. Complete pending `seq8192` distributed NSYS probe capture and convert to phase-pure breakdown.
2. Run `seq8192` scaling NSYS A/B:
   - DDP hook accumulation **on** (`SCALING_DISABLE_DDP_WRAP=0`)
   - DDP hook accumulation **off** (`SCALING_DISABLE_DDP_WRAP=1`)
3. Verify contamination gating on new captures.
4. Quantify dist-vs-scale residual changes under phase-pure semantics.
5. Re-check dominant kernel-family source after DDP A/B.

### 2) Reproducible Commands

```bash
# 1) distributed seq8192 capture (already started in previous round; this round completed and collected artifact)
# output: logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe.nsys-rep

# 2) scaling seq8192 capture (DDP hook on)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=0 SCALE_GPU=7 MASTER_PORT=6560 bash examples/pretrain_deepseek_v3_moe.sh"

# 3) scaling seq8192 capture (DDP hook off)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=1 SCALE_GPU=7 MASTER_PORT=6570 bash examples/pretrain_deepseek_v3_moe.sh"

# 4) export sqlite
nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off.nsys-rep

# 5) phase-pure analyzer
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe_breakdown.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on_breakdown.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off_breakdown.md

# 6) compare (paper view: pure_primary_union + shared(primary_stream) + contamination gate)
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_on_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_compare_ddp_on_pure_primary_union_shared.log

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_dist_ddp_probe_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_scaling_ddp_off_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sl8192_compare_ddp_off_pure_primary_union_shared.log

# 7) quick regression for analyzer/compare unit paths
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29641 PYTHONPATH=$(pwd) \
python -m pytest \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py -q
```

### 3) Validation Criteria

1. New seq8192 captures must have valid phase windows and analyzable event rows.
2. contamination gate must pass (`contamination_pct <= 1%`) for all compared rows.
3. DDP-hook off should show measurable scaling-side kernel/time reduction if hook path is active.
4. Evaluate whether DDP-hook off materially closes dist-vs-scale backward residual.
5. Check whether top kernel-family deltas remain non-comm dominated.
6. Keep analyzer/compare unit path green after new artifact generation.

### 4) Results and Evidence

#### 4.1 Capture / analyzer sanity

- All three captures completed and generated `.nsys-rep` + sqlite + breakdown JSON/MD.
- Analyzer outputs are structurally valid for all three datasets:
  - `phase_window_parents=48`
  - `event_rows=72`
  - `aggregate_rows=24`
- contamination is clean in all rows (`contamination_pct=0.00%`).

#### 4.2 Dist-vs-scale compare (pure_primary_union, shared kernel scope)

- `DDP hook on` (`...compare_ddp_on_pure_primary_union_shared.log`):
  - `backward` rank-median diff: **16.88%** (FAIL)
  - `forward` rank-median diff: **7.20%** (FAIL)
  - `optimizer` rank-median diff: **5.45%** (FAIL)

- `DDP hook off` (`...compare_ddp_off_pure_primary_union_shared.log`):
  - `backward` rank-median diff: **9.00%** (FAIL)
  - `forward` rank-median diff: **4.91%** (PASS)
  - `optimizer` rank-median diff: **1.96%** (PASS)

Interpretation:
- relative to this seq8192 probe set, disabling scaling DDP hook accumulation improves all 3 ops, but **backward still fails 5% gate**.

#### 4.3 Stage1 backward steady (rank4..7) focused view

Using `compute_pure_primary_union_ms` per-rank means:

- DDP hook on vs dist:
  - rank diff% = `[+6.32, +13.73, +6.59, +11.08]`
  - median diff% = **8.83%**, IQR = **5.21pp**
- DDP hook off vs dist:
  - rank diff% = `[+2.70, +6.80, +5.27, +11.76]`
  - median diff% = **6.04%**, IQR = **3.41pp**

Interpretation:
- DDP-hook off reduces median and spread for stage1 backward, but still not enough to satisfy freeze condition (`<=5pp`).

#### 4.4 Scaling-only DDP A/B deltas (stage1 backward steady, rank4..7)

- `compute_pure_primary_union_ms` total:
  - on: `753.772 ms`
  - off: `734.473 ms`
  - delta: `-19.300 ms` (`-2.56%`)
- `kernel_count` total:
  - on: `7888`
  - off: `7300`
  - delta: `-588` (`-7.45%`)

Interpretation:
- this seq8192 run confirms DDP hook path contributes measurable kernels/time in scaling, but magnitude remains partial.

#### 4.5 Kernel-family attribution (dist vs scaling, stage1 backward steady rank4..7)

Artifact: `logs/nsys_phase_sanity/deepseek_phase_sl8192_kernel_family_delta_summary.json`

- Dist vs scaling-on top delta:
  - `fmha_cutlassB...`: **+40.965 ms** (scale - dist)
- Dist vs scaling-off top delta:
  - `fmha_cutlassB...`: **+32.653 ms** (scale - dist)

Other deltas are much smaller (single-digit ms).  
Interpretation: dominant residual component remains attention-family (`fmha_cutlassB`), consistent with non-comm-dominant diagnosis.

#### 4.6 Regression sanity

- Unit regression command passed:
  - `17 passed in 0.04s`
- Scope:
  - `tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py`

### 5) Verdict

1. Phase-pure path is healthy on seq8192 probe: contamination remains 0.00%, so residual is not comm-window leakage.
2. DDP hook accumulation contributes part of scaling backward cost; disabling it improves metrics in this probe, but backward still does not meet 5% criterion.
3. Dominant residual signal remains attention-family (`fmha_cutlassB`) rather than pure comm-adjacent copy/add kernels.
4. Current seq8192 x1 probe is still insufficient for semantic freeze; next required step remains fixed-protocol `rank7-cap + repeat x5` with phase-pure metric.
