## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added phase-label NSYS sanity x1 capture/export/analyze/compare report for DeepSeek distributed vs scaling, including contamination-gate evidence and residual-gap status |

## Test Report: Phase-level Comp-only Semantics Sanity (x1)

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, `/opt/anaconda/envs/myenv_yc/bin/python`)

### 1) Test Script Information

- Related scripts:
  - `examples/pretrain_deepseek_v3_moe.sh`
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/performance/compare_qwen_nsys_compute_only.py`
- Artifact directory:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/`

- Reproducible commands (exact):

```bash
# Distributed NSYS capture (rerun used for this report)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun \
  bash -lc "MODE=distributed MODEL_PROFILE=smoke SEQ_LEN=1024 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global MASTER_PORT=6500 bash examples/pretrain_deepseek_v3_moe.sh"

# Scaling NSYS capture (x1)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=1024 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 MASTER_PORT=6510 bash examples/pretrain_deepseek_v3_moe.sh"

# Export sqlite
nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling.nsys-rep

# Analyze phase-level breakdown
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun \
  --label-prefix cmd_trace --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun_breakdown.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling \
  --label-prefix cmd_trace --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_breakdown.md

# Compare (new official metric candidate)
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --compute-metric pure_primary_union \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_compare_dist_rerun_pure_primary_union.log
```

### 2) Validation Criteria

1. **Phase labels are actually present in new captures**:
   - analyzer must report `phase_window_parents > 0`.
2. **Pure compute path is uncontaminated**:
   - `contamination_pct` should be near zero and satisfy `--require-low-contamination-pct 1`.
3. **Metric consistency sanity**:
   - `pure_primary_union` should be compatible with analyzer outputs and compare should run without schema fallback.
4. **Accuracy gate check (diagnostic in this x1 run)**:
   - compare threshold `5%` to observe current residual shape (pass/fail expected to drive next repeat-x5).

### 3) Test Results and Evidence

| Check | Result | Evidence |
|------|--------|----------|
| Distributed NSYS capture finished | PASS | `deepseek_phase_sanity_dist_rerun.nsys-rep` generated |
| Scaling NSYS capture finished | PASS | `deepseek_phase_sanity_scaling.nsys-rep` generated |
| Analyzer parsed phase labels (dist) | PASS | `phase_window_parents=48`, `event_rows=72` |
| Analyzer parsed phase labels (scaling) | PASS | `phase_window_parents=48`, `event_rows=72` |
| Contamination gate data available | PASS | JSON contains `contamination_ms`, `contamination_pct` |
| Contamination gate threshold (`<=1%`) | PASS | all compared rows show `dist_contam_pct=0.00`, `scale_contam_pct=0.00` |
| `pure_primary_union` compare pipeline | PASS | compare command completed and produced report |
| 5% fidelity gate (x1 sanity) | FAIL | op-rank medians: `forward=10.25%`, `backward=17.97%`, `optimizer=5.75%` |

Key outputs:

- Analyzer outputs:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_rerun_breakdown.md`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_breakdown.md`
- Compare output:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_compare_dist_rerun_pure_primary_union.log`

### 4) Failure Diagnosis (x1)

- This run **did validate the new semantics path** (phase labels + contamination gate are active and clean).
- But the run **did not satisfy accuracy threshold** at seq1024 smoke x1.
- Evidence indicates the current fail is **not from comm contamination in compute-pure windows** (all contamination = 0), but from residual distributed-vs-scaling compute mismatch itself, especially for stage1 ranks.
- Therefore the next required step remains protocolized repeat-x5 (rank7-cap + seq8192) before freeze decisions.

### 5) Exit Code Evidence

- All capture/export/analyze commands returned exit code `0`.
- Compare command returned exit code `1` due threshold-based FAIL status (expected behavior for gating mode).
