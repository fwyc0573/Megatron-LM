## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initial test/report artifact for DeepSeek-V3 comp gap analysis |
| 2026-03-05 | Added stable 8-GPU rerun validation for metadata-comm timing patch |
| 2026-03-05 | Added remove-comm-adjacent-copy validation and final fix selection evidence |

# Test Report: DeepSeek-V3 Scaling vs Distributed Comp Gap Analysis

**Date**: 2026-03-05  
**Environment**: `/opt/anaconda/envs/myenv_yc` (Python 3.9.18)

## 1. Test Script Information

- Script(s):
  - `tests/performance/compare_qwen_trace_comp.py`
  - `examples/pretrain_deepseek_v3_moe.sh` (reproduction attempt)
- Commands:
  ```bash
  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe/pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048/global_ranks_profile \
    --scaling-dir scaling_traces_h800_20260305_003852/profiler_log/pp2_tp1_ep4_expn32_dp8_nl32_hs2048_sl2048 \
    --ranks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ops forward_step,backward_step,optimizer_step \
    --threshold-pct 5 \
    --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/compare_trace_full_16ranks.md

  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe/pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048/global_ranks_profile \
    --scaling-dir scaling_traces_h800_20260305_003852/profiler_log/pp2_tp1_ep4_expn32_dp8_nl32_hs2048_sl2048 \
    --ranks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ops forward_step,backward_step,optimizer_step \
    --threshold-pct 5 \
    --no-distributed-subtract-comm \
    --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/compare_trace_no_subtract.md

  MODE=distributed MODEL_PROFILE=smoke PP=2 EP=4 TP=1 GPUS_PER_NODE=8 \
  TRAIN_ITERS=1 TRACE_START=1 SEQ_LEN=256 MICRO_BATCH_SIZE=1 \
  DO_TRACE=True TRACE_SUBOP_SYNC_MODE=global MASTER_PORT=6510 \
  bash examples/pretrain_deepseek_v3_moe.sh
  ```

## 2. Validation Criteria

- Primary metric extraction criteria:
  - For distributed mode: parse top-level op duration and subtract sub-op comm durations when `--distributed-subtract-comm` is enabled.
  - For scaling mode: use top-level op duration directly (comm sub-op durations are metadata-only 0 by design).
- Comparison coverage:
  - 16 ranks: `0..15`
  - Ops: `forward_step`, `backward_step`, `optimizer_step`
- Sanity checks:
  - Same `(rank, op, mg_state)` bucket alignment between modes.
  - Report contains both per-rank rows and op-level median summaries.

## 3. Test Results and Evidence

### 3.1 Trace Comparison Results

| Suite | Result | Details |
|------|--------|---------|
| `compare_trace_full_16ranks.md` (distributed subtract comm) | FAIL | 36 checks above 5% threshold |
| `compare_trace_no_subtract.md` (distributed no subtract) | FAIL | 36 checks above 5% threshold, but backward gap drops significantly |

Key numeric evidence:
- With distributed comm subtraction:
  - `forward_step` median diff ≈ 55.90%
  - `backward_step` median diff ≈ 35.96%
  - `optimizer_step` median diff ≈ 2.47%
- Without distributed comm subtraction:
  - `forward_step` median diff ≈ 36.35%
  - `backward_step` median diff ≈ 18.79%
  - `optimizer_step` median diff ≈ 2.47%

Artifacts:
- `task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/compare_trace_full_16ranks.md`
- `task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/compare_trace_no_subtract.md`
- `task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/per_rank_comp_summary.md`
- `task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/per_rank_comp_summary.csv`

### 3.2 Reproduction Attempt on 8-GPU A800

| Suite | Result | Details |
|------|--------|---------|
| 8-GPU distributed short run (`pretrain_deepseek_v3_moe.sh`) | FAIL | OOM due external GPU occupancy |

Failure evidence:
- `nvidia-smi` snapshot during attempt:
  - Each A800 had ~75GB used by pre-existing processes, only ~4.5-5.7GB free.
- Runtime failure:
  - `torch.cuda.OutOfMemoryError` during DDP/optimizer initialization.

Partial successful evidence before failure:
- Parsed startup output confirmed model parameter counts for smoke PP2 EP4:
  - stage0 total params: `695240704`
  - stage1 total params: `711309312`

## 4. Failure Diagnosis and Resolution Status

- Root cause of reproduction failure: shared host GPU contention, not code crash.
- Resolution in this session: not possible without freeing/reserving GPUs.
- Recommended next run conditions:
  - Reserve idle 8 GPUs, then rerun distributed/scaling scripts with same config.
  - Prefer `MODEL_PROFILE=smoke`, `TRACE_START=1`, `TRAIN_ITERS=1~3` for fast debug confirmation.

## 5. Stable Rerun Validation (After Temporary Patch)

**Date**: 2026-03-05  
**Environment**: `/opt/anaconda/envs/myenv_yc` (Python 3.9.18), 8x A800 GPUs idle  
**Patch under test**:
- `megatron/profiler/cmd.py` (optional timing for scaling metadata-only comm sub-ops)
- `megatron/training/arguments.py` (`--scaling-trace-metadata-comm-duration`)
- `examples/pretrain_deepseek_v3_moe.sh` (`SCALING_TRACE_METADATA_COMM_DURATION` gate)

### 5.1 Test Script Information

- Script(s):
  - `examples/pretrain_deepseek_v3_moe.sh`
  - `tests/performance/compare_qwen_trace_comp.py`
- Profile setup:
  - `MODEL_PROFILE=smoke`
  - `PP=2, TP=1, EP=4, FAKE_WORLD_SIZE=8`
  - `SEQ_LEN=256, MICRO_BATCH_SIZE=1`
  - Stable window: `TRAIN_ITERS=10`, `TRACE_START=10`
- Reproducible compare commands:
  ```bash
  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/dist_run/realistic_trace/pp2_tp1_exp4_expn32_dp4_nl32_hs2048_sl256 \
    --scaling-dir task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/scale_baseline/profiler_log/pp2_tp1_ep4_expn32_dp4_nl32_hs2048_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step \
    --threshold-pct 5 \
    --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_baseline_distSub_scaleNoSub.md

  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/dist_run/realistic_trace/pp2_tp1_exp4_expn32_dp4_nl32_hs2048_sl256 \
    --scaling-dir task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/scale_patched/profiler_log/pp2_tp1_ep4_expn32_dp4_nl32_hs2048_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step \
    --threshold-pct 5 \
    --scaling-subtract-comm \
    --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distSub_scaleSub.md
  ```

### 5.2 Validation Criteria

- Same rank/op/state pairing rule as prior report.
- Primary check:
  - baseline: `distributed_subtract_comm=True`, `scaling_subtract_comm=False`
  - patched: `distributed_subtract_comm=True`, `scaling_subtract_comm=True`
- Target threshold: `5%`.
- Focus metric: `op_rank_median_aux_summary(non-gating, recommended_for_paper)`.

### 5.3 Test Results and Evidence

| Suite | Result | Key Metrics (`op_rank_median_diff_pct`) |
|------|--------|------------------------------------------|
| baseline (`distSub + scaleNoSub`) | FAIL | `forward_step=8.88%`, `backward_step=11.67%`, `optimizer_step=7.05%` |
| patched (`distSub + scaleSub`) | FAIL | `forward_step=5.62%`, `backward_step=7.96%`, `optimizer_step=6.11%` |

Additional cross-checks:
- patched (`distSub + scaleNoSub`): `forward_step=6.47%`, `backward_step=8.16%`, `optimizer_step=6.11%`
- patched (`distNoSub + scaleSub`): `forward_step=19.61%`, `backward_step=18.33%`, `optimizer_step=6.11%`

Evidence:
- Compare script exited with code `1` for all above runs because rows exceeded 5% threshold (expected behavior under current residual gap).
- Reports were regenerated successfully:
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_baseline_distSub_scaleNoSub.md`
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distSub_scaleSub.md`
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distSub_scaleNoSub.md`
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distNoSub_scaleSub.md`

### 5.4 Failure Diagnosis and Resolution Status (Updated)

- Previous runtime blocker is resolved: stable distributed/scaling rerun completed on idle GPUs.
- Current remaining issue is metric-level, not runtime-level:
  - Patch improves fwd/bwd relative error but does not meet strict `<=5%` for all ops.
  - Residual mismatch is concentrated on selected ranks/ops (notably some stage-1 ranks and optimizer path).

## 6. Validation of Selected Fix (Remove Comm-Adjacent Copy in Scaling Mode)

**Date**: 2026-03-05  
**Environment**: `/opt/anaconda/envs/myenv_yc` (Python 3.9.18), 8x A800 GPUs idle  
**Patch under test**:
- `megatron/core/tensor_parallel/mappings.py`
  - function: `_profiled_all_to_all_single`
  - change: remove scaling-mode comm-adjacent `contiguous/copy_` materialization, keep metadata-only semantics.

### 6.1 Test Script Information

- Script(s):
  - `tests/performance/compare_qwen_trace_comp.py`
  - `python -m py_compile` for syntax validation
- Commands:
  ```bash
  python -m py_compile megatron/core/tensor_parallel/mappings.py

  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/dist_run/realistic_trace/pp2_tp1_exp4_expn32_dp4_nl32_hs2048_sl256 \
    --scaling-dir profiler_log/pp2_tp1_ep4_expn32_dp4_nl32_hs2048_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step \
    --threshold-pct 5 \
    --pair-timestamp 20260305081359 \
    --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_remove_comm_adjacent_copy/reports/compare_removedcopy_distSub_scaleNoSub_recheck.md
  ```

### 6.2 Validation Criteria

- Compare two candidate directions on same distributed baseline:
  - Option A: scaling-side comm attribution (`scaleSub`) from previous patch.
  - Option B: remove scaling comm-adjacent kernels and use `scaleNoSub`.
- Focus metric:
  - `op_rank_median_aux_summary(non-gating, recommended_for_paper)`.
- Acceptance threshold:
  - `<=5%` per op (strict target).

### 6.3 Test Results and Evidence

| Variant | Result | Key Metrics (`op_rank_median_diff_pct`) |
|------|--------|------------------------------------------|
| Previous best (Option A, `distSub + scaleSub`) | FAIL | `forward_step=5.62%`, `backward_step=7.96%`, `optimizer_step=6.11%` |
| Selected fix (Option B, `distSub + scaleNoSub`) | PARTIAL PASS | `forward_step=2.78%`, `backward_step=2.21%`, `optimizer_step=5.31%` |
| Option B + `SCALING_DISABLE_DDP_WRAP=1` (cross-check) | MIXED | `forward_step=5.80%`, `backward_step=4.70%`, `optimizer_step=4.59%` |

Evidence artifacts:
- `rerun_2026-03-05_remove_comm_adjacent_copy/reports/compare_removedcopy_distSub_scaleNoSub.md`
- `rerun_2026-03-05_remove_comm_adjacent_copy/reports/compare_removedcopy_distSub_scaleSub.md`
- `rerun_2026-03-05_remove_comm_adjacent_copy/reports/compare_removedcopy_ddpoff_distSub_scaleNoSub.md`
- `rerun_2026-03-05_remove_comm_adjacent_copy/reports/compare_removedcopy_distSub_scaleNoSub_recheck.md`

Execution evidence:
- `python -m py_compile megatron/core/tensor_parallel/mappings.py` exit code `0`.
- Compare command exited with code `1` because strict threshold remains unmet for a subset of rows; this is expected under current residual gap.

### 6.4 Final Conclusion for This Patch Round

- Selected solution: **Option B (remove comm-adjacent copy in scaling metadata-only comm path)**.
- Reason:
  - Largest improvement on `forward_step` and `backward_step` median gaps.
  - Better alignment with scaling-mode design intent (metadata-only comm should avoid payload movement kernels).
- Remaining delta:
  - `optimizer_step` median currently `5.31%`, marginally above `5%`.
