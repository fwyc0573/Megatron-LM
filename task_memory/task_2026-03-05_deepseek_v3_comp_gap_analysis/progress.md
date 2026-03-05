## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Started task and initialized tracking |
| 2026-03-05 | Completed trace comparison + root-cause analysis; documented reproduction blocker |
| 2026-03-05 | Added scaling metadata-comm timing patch and finished stable 8-GPU rerun |
| 2026-03-05 | Evaluated remove-comm-adjacent-copy fix, selected final solution, and rechecked key metrics |

# Progress

## 2026-03-05
- Created task directory and planning artifacts.
- Parsed and compared 16-rank traces for `forward_step`, `backward_step`, `optimizer_step`.
- Generated artifacts:
  - `logs/compare_trace_full_16ranks.md`
  - `logs/compare_trace_no_subtract.md`
  - `logs/per_rank_comp_summary.md`
  - `logs/per_rank_comp_summary.csv`
- Performed code audit for PP/MoE/sharding semantics and comm tracing behavior.
- Attempted 8-GPU distributed reproduction; blocked by OOM due shared machine occupancy.

## 2026-03-05 (stable rerun after patch)
- Implemented temporary debugging patch for scaling trace attribution:
  - Added CLI flag `--scaling-trace-metadata-comm-duration` in `megatron/training/arguments.py`.
  - Enabled optional measured duration for scaling metadata-only comm sub-ops in `megatron/profiler/cmd.py`.
  - Added script env gate `SCALING_TRACE_METADATA_COMM_DURATION=0/1` in `examples/pretrain_deepseek_v3_moe.sh`.
- Re-ran on current idle 8-GPU A800 with stable profile window (`TRAIN_ITERS=10`, `TRACE_START=10`, smoke profile, `pp2/tp1/ep4/dp4`):
  - Distributed trace generated successfully.
  - Scaling baseline trace generated successfully.
  - Scaling patched trace generated successfully.
- Generated comparison reports under:
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_baseline_distSub_scaleNoSub.md`
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distSub_scaleSub.md`
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distSub_scaleNoSub.md`
  - `rerun_2026-03-05_metadata_comm_fix_stable/reports/compare_patched_distNoSub_scaleSub.md`
- Key metric delta (`op_rank_median_diff_pct`, distributed subtract comm):
  - Baseline (`scale no subtract`): `forward_step=8.88%`, `backward_step=11.67%`, `optimizer_step=7.05%`.
  - Patched (`scale subtract`): `forward_step=5.62%`, `backward_step=7.96%`, `optimizer_step=6.11%`.
- Interim conclusion:
  - Patch reduces forward/backward mismatch, validating the timing-attribution root cause.
  - Residual gap remains (>5%, especially backward/optimizer), requiring additional attribution refinement.

## 2026-03-05 (remove comm-adjacent copy rerun and final selection)
- Implemented scaling-mode metadata-only communication cleanup in:
  - `megatron/core/tensor_parallel/mappings.py` (`_profiled_all_to_all_single`)
- Change details:
  - In `is_scaling_mode` path, removed comm-adjacent materialization/copy kernels (`contiguous`, `copy_`).
  - `output_split_sizes is None`: return `input_` directly (identity semantics).
  - `output_split_sizes` set: keep shape-only deterministic `new_zeros` output, without copying payload rows.
- Re-ran and compared with existing distributed baseline (`pair_timestamp=20260305081359`) under multiple accounting variants.
- Key comparison (op-level rank median diff, `distSub`):
  - Previous best (metadata-comm-duration patch, `scaleSub`): `forward=5.62%`, `backward=7.96%`, `optimizer=6.11%`.
  - New fix (`scaleNoSub`): `forward=2.78%`, `backward=2.21%`, `optimizer=5.31%`.
  - Additional check with `SCALING_DISABLE_DDP_WRAP=1`: `forward=5.80%`, `backward=4.70%`, `optimizer=4.59%`.
- Final selection:
  - Keep default DDP wrap.
  - Use remove-comm-adjacent-copy patch as primary fix.
  - Recommended comparison accounting for this fix: `distributed_subtract_comm=True`, `scaling_subtract_comm=False`.
