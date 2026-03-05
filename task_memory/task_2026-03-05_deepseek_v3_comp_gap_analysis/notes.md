## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initialize notes file |
| 2026-03-05 | Added comparison findings, code-level observations, and reproduction constraints |
| 2026-03-05 | Added fix-option comparison and final selection rationale |

# Notes

## Data Paths
- Distributed: `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe/pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048/global_ranks_profile`
- Scaling: `scaling_traces_h800_20260305_003852/profiler_log/pp2_tp1_ep4_expn32_dp8_nl32_hs2048_sl2048`

## Commands Executed
- Baseline comparison (distributed subtract comm):
  - `python tests/performance/compare_qwen_trace_comp.py --distributed-dir ... --scaling-dir ... --ranks 0..15 --ops forward_step,backward_step,optimizer_step --threshold-pct 5 --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/compare_trace_full_16ranks.md`
- Auxiliary comparison (no distributed comm subtraction):
  - `python tests/performance/compare_qwen_trace_comp.py --distributed-dir ... --scaling-dir ... --ranks 0..15 --ops forward_step,backward_step,optimizer_step --threshold-pct 5 --no-distributed-subtract-comm --report-path task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/logs/compare_trace_no_subtract.md`

## Key Numerical Findings
- With distributed comm subtraction:
  - forward median diff: ~56%
  - backward median diff: ~36%
  - optimizer median diff: ~2.5%
- Without distributed comm subtraction:
  - forward median diff: ~36%
  - backward median diff: ~18.8%
  - optimizer median diff: ~2.5%
- Stage-level trend:
  - Stage1 (`rank 8-15`) has significantly larger backward gap than stage0 when using comm-subtracted metric.
- Outlier:
  - `rank10 forward` shows a single-sample spike (`176.2 ms`) in scaling trace.

## Code-Level Observations
1. PP layer partition semantics are aligned by design:
   - `get_num_layers_to_build` uses `config.num_layers // config.fake_pp` in scaling and `// pipeline_world_size` in distributed.
   - `get_gpt_decoder_block_spec` slices by `pp_rank * num_layers_to_build` in both modes.
2. DeepSeek layer heterogeneity is explicitly driven by `--moe-layer-freq` and preserved by layer-spec slicing.
3. Major measurement bias source:
   - In scaling mode, comm sub-ops are marked metadata-only (`duration=0`) in `CMD.get_trace_decorator`, but wrapped functions still execute.
   - `_profiled_all_to_all_single` scaling branch executes tensor `contiguous/new_zeros/copy_` kernels, which are charged into parent `forward_step/backward_step` instead of comm duration.
4. Metric implication:
   - Subtracting distributed comm while scaling comm is zeroed (but not free) exaggerates comp gap, especially on ranks/stages with more MoE all-to-all sub-ops.

## Reproduction Attempt Notes (8-GPU A800)
- Attempted distributed run with `examples/pretrain_deepseek_v3_moe.sh` failed due GPU memory contention from existing external processes (each GPU had ~75GB occupied).
- Could extract startup evidence before failure:
  - model parameter counts printed as stage0 `695,240,704`, stage1 `711,309,312` for smoke PP2 EP4 config.

## Fix Option Comparison (Final)
### Option A: Continue comm attribution in scaling (`scaleSub`)
- Strength:
  - Reduced part of forward/backward gap versus baseline.
- Limitation:
  - Still above threshold for multiple ops in stable rerun.
  - Attributing metadata-only comm by timing wrappers can absorb synchronization/wait noise and over-correct comp.

### Option B: Remove comm-adjacent kernels in scaling metadata-only path
- Implementation point:
  - `megatron/core/tensor_parallel/mappings.py` line ~474 (`_profiled_all_to_all_single`).
- Semantics:
  - In scaling mode, comm should be metadata-only.
  - Therefore, comm-adjacent payload materialization (`contiguous`, `copy_`) should not execute in this path.
- Quantitative outcome (best accounting: `distSub + scaleNoSub`):
  - `forward_step`: `2.78%`
  - `backward_step`: `2.21%`
  - `optimizer_step`: `5.31%`

## Final Recommendation
- Adopt Option B as the primary fix.
- Keep comparison method as:
  - distributed: subtract comm sub-ops
  - scaling: do not subtract comm sub-ops
- Rationale:
  - More aligned with scaling-mode design intent (metadata-only communication).
  - Better and more stable fwd/bwd accuracy than Option A.
