## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added backward measurement-semantics validation on Round6-8 seq8192 batch: baseline subtract vs op-map/stage-aware subtract vs no-subtract views, with repeat-x5 robustness and bias-source conclusion |

## Test Report: Stage-2 Round6-8 Backward Measurement Semantics Validation

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Execution workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_regime`  
**Code state**: detached `3a50265d` (no code changes)  

---

### 1) Goal

Per user instruction, do **measurement-semantics-only** validation (no model/training code edits), and output in the same batch:

1. baseline `distributed_subtract_comm` view,
2. stage-aware/compute-only auxiliary views,
3. repeated-run robustness (`x5`, rank7 end-of-run cap).

Target is to identify whether backward regression comes from code path or from subtraction semantics.

---

### 2) Fixed Batch and Pairing

- Workload batch: Round6-8 smoke, `SEQ_LEN=8192`, same distributed/scaling batch already generated.
- Fixed pair timestamps (rank7 cap):
  - `20260228152645`
  - `20260228152924`
  - `20260228153204`
  - `20260228153444`
  - `20260228153723`

---

### 3) View Definitions

1. **baseline_subtract**  
   - `--distributed-subtract-comm`  
   - `--distributed-comm-scale-map ""` (implicit alpha=1.0)

2. **opmap_subtract** (compute-only auxiliary, op-level)  
   - `--distributed-subtract-comm`  
   - `--distributed-comm-scale-map "forward_step=0.787,backward_step=0.176"`

3. **stageaware_subtract** (compute-only auxiliary, stage-aware)  
   - `--distributed-subtract-comm`  
   - `--distributed-comm-scale-map "forward_step=0.787,backward_step=0.176,forward_step@stage1=0.787,backward_step@stage1=0.107"`

4. **nosubtract_total** (total-time auxiliary control)  
   - `--no-distributed-subtract-comm`

Notes on alpha source:
- alpha values derived from `--suggest-comm-scale` across the same five pairs.
- Aggregated suggestion summary:  
  `logs/deepseek_v3_stage2_round68_regime_sl8192_comm_scale_suggest_summary.json`

---

### 4) Repro Commands

```bash
# suggestion extraction (same pair, same batch)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp <rank7_end_ts> \
  --distributed-subtract-comm \
  --suggest-comm-scale \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run<k>_suggest.log

# stage-aware auxiliary view
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp <rank7_end_ts> \
  --distributed-subtract-comm \
  --distributed-comm-scale-map "forward_step=0.787,backward_step=0.176,forward_step@stage1=0.787,backward_step@stage1=0.107" \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run<k>_stageaware.log

# op-level auxiliary view
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp <rank7_end_ts> \
  --distributed-subtract-comm \
  --distributed-comm-scale-map "forward_step=0.787,backward_step=0.176" \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run<k>_opmap.log

# no-subtract control
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp <rank7_end_ts> \
  --no-distributed-subtract-comm \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run<k>_nosubtract.log
```

---

### 5) Results (repeat x5 robust summary)

#### 5.1 Median-of-runs by view

| View | forward_step | backward_step | optimizer_step | mean_3ops | max_3ops |
|---|---:|---:|---:|---:|---:|
| baseline_subtract | 2.00% | 62.73% | 6.72% | 24.67% | 62.73% |
| opmap_subtract | 1.74% | 5.85% | 6.72% | 4.97% | 6.72% |
| stageaware_subtract | 1.74% | 3.86% | 6.72% | 4.43% | 6.72% |
| nosubtract_total | 9.32% | 7.74% | 6.72% | 8.06% | 9.32% |

#### 5.2 Spread (range / IQR) on backward_step

| View | backward range | backward IQR |
|---|---:|---:|
| baseline_subtract | 34.41% | 6.84% |
| opmap_subtract | 0.75% | 0.47% |
| stageaware_subtract | 1.81% | 0.64% |
| nosubtract_total | 3.06% | 1.76% |

---

### 6) Evidence of Bias Location (example rows)

Run1, stage1 backward ranks (4~7):

- **baseline_subtract**:
  - rank4: `dist_total=65.42`, `dist_comm=43.62`, `dist_comp=21.80`, `scale_comp=54.09`, `diff=148.12%`
  - rank5: `dist_comp=22.58`, `scale_comp=62.66`, `diff=177.48%`

- **stageaware_subtract** (same batch, same pair):
  - rank4: `dist_eff_comm=4.6673`, `dist_comp=60.7527`, `scale_comp=54.09`, `diff=10.97%`
  - rank5: `dist_eff_comm=4.5832`, `dist_comp=60.8335`, `scale_comp=62.66`, `diff=3.01%`

This directly indicates backward inflation in baseline view is caused by over-subtraction of comm under overlap-heavy stage1 windows.

---

### 7) Conclusion

1. Backward failure under baseline (`62.73%`) is **not** a model-code regression signal; it is dominated by measurement semantics (`distributed_subtract_comm` over-subtraction in stage1).
2. Stage-aware/compute-only auxiliary views reduce backward median into stable low range (`3.86%`) with small spread.
3. Optimizer remains the true unresolved residual (`median 6.72%` across all views), so future code A/B should still be gated by optimizer convergence.
4. Decision aligned with user instruction:
   - keep code path frozen for now,
   - continue semantic stabilization first,
   - resume code-level single-variable A/B only after backward gate semantics is stable.

---

### 8) Artifacts

- Suggestion reports (5 runs):
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run1_suggest.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run2_suggest.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run3_suggest.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run4_suggest.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run5_suggest.log`
- View logs:
  - baseline: `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run{1..5}.log`
  - op-map: `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run{1..5}_opmap.log`
  - stage-aware: `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run{1..5}_stageaware.log`
  - no-subtract: `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run{1..5}_nosubtract.log`
- Aggregated summaries:
  - `logs/deepseek_v3_stage2_round68_regime_sl8192_comm_scale_suggest_summary.json`
  - `logs/deepseek_v3_stage2_round68_regime_sl8192_semantics_views_summary.json`
  - `logs/deepseek_v3_stage2_round68_regime_sl8192_semantics_views_summary.md`
