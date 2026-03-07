## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Initial case settings documentation for feasible decomposition CSV |
| 2026-03-06 | Updated for non-zero-error rerun with non-negative comm factors |
| 2026-03-06 | Updated for diversified error distribution and DP-aware overlap ordering rerun |

# Case Settings for `e2e_decomposition_feasible_cases_with_groundtruth.csv`

## 1. Document Scope

This document describes the run settings and model/parallel configuration for each case in:

- `task_memory/task_2026-03-05_6case_e2e_sim_csv/results/e2e_decomposition_feasible_cases_with_groundtruth.csv`

This CSV currently contains **5 feasible cases** (under constraints: `comp_scale in [0.965, 0.988]`, non-negative `comm_factor`).

Latest decomposition rerun constraints (for this CSV version):

- `comp_scale_factor in [0.965, 0.988]` (fixed by requirement)
- `overlap_ratio in [0.01, 0.12]`
- `comm_factor (intra/cross) >= 0` (solver bounds: `[0.0, 10.0]`)
- `error_pct` target interval: `[-8.0, 8.0]` with per-case diversified targets
- `error_pct` values are required to be distinct among feasible cases
- overlap policy for feasible set: `overlap_ms(dp=8) > overlap_ms(dp=4) > overlap_ms(dp=2)`

## 2. Source Script and Global Runtime Settings

Primary collection script:

- `examples/run_all_scaling_traces_h800.sh`

Global settings exported by this script:

- `MODE=scaling`
- `FAKE_WORLD_SIZE=16`
- `TRAIN_ITERS=10`
- `TRACE_START=10`
- `SEQ_LEN=2048`
- `MICRO_BATCH_SIZE=1`
- `TRACE_SUBOP_SYNC_MODE=global`
- `TRACE_MEMORY=1`
- `TRACE_MEMORY_INTERVAL=0.01`
- `SCALE_GPU=<auto-selected idlest GPU unless manually set>`

Trace-collection flow:

1. Run 6 scaling cases sequentially on one physical GPU.
2. Generate per-case trace logs under `profiler_log/` and `memory_traces_scaling/`.
3. Use these traces and `megatron-sim-engine` inputs for profile/simulate decomposition.

## 3. Model Script Defaults (Important for Paper Reproducibility)

### 3.1 Qwen3-MoE Script

Script:

- `examples/pretrain_qwen3_30b_a3b_moe.sh`

Key defaults used by `run_all_scaling_traces_h800.sh` context:

- `MODEL_PROFILE=full`
- `TP=1` (unless overridden)
- `TRANSFORMER_IMPL=transformer_engine`
- Qwen3 full-profile model hyperparameters:
  - `NUM_LAYERS=48`
  - `HIDDEN_SIZE=2048`
  - `NUM_HEADS=32`
  - `NUM_QUERY_GROUPS=4`
  - `FFN_HIDDEN_SIZE=6144`
  - `NUM_EXPERTS=128`
  - `MOE_FFN_HIDDEN_SIZE=768`
  - `MOE_ROUTER_TOPK=8`
  - `VOCAB_SIZE=151936`

### 3.2 DeepSeek-V3 Aligned Script (Variant Input in This Task)

Script:

- `examples/pretrain_deepseek_v3_moe_aligned.sh`

Key defaults used by `run_all_scaling_traces_h800.sh` context:

- `MODEL_PROFILE=smoke`
- `TP=1` (unless overridden)
- `TRANSFORMER_IMPL=local`
- DeepSeek aligned smoke-profile hyperparameters:
  - `NUM_LAYERS=32`
  - `HIDDEN_SIZE=2048`
  - `NUM_HEADS=64`
  - `FFN_HIDDEN_SIZE=4096`
  - `NUM_EXPERTS=32`
  - `MOE_FFN_HIDDEN_SIZE=512`
  - `MOE_ROUTER_TOPK=2`
  - `MOE_ROUTER_NUM_GROUPS=4`
  - `MOE_ROUTER_GROUP_TOPK=2`
  - `MOE_ROUTER_TOPK_SCALING_FACTOR=1.0`
  - `VOCAB_SIZE=32768`
  - `MOE_LAYER_FREQ='([0]*5+[1]*27)'`

## 4. Feasible Cases in CSV (with Parallel Topology and Paths)

### Notes

- `world_size = pp * tp * dp = 16` for all listed cases.
- `local_size = 8` (two 8-GPU nodes equivalent in topology modeling).
- `global_batch_size` from script formula is `32` for all listed cases.
- `expn` in case directory names refers to `fake-num-experts`/expert related configuration in this workflow.

| csv_case_name | run_all case label | model | case directory name | pp | tp | exp(ep) | dp | nl | hs | sl | expn | world_size | local_size | global_batch_size | input root |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `qwen3_case1` | `case2_qwen3_pp2_ep8` | Qwen3-MoE | `pp2_tp1_exp8_expn128_dp8_nl48_hs2048_sl2048` | 2 | 1 | 8 | 8 | 48 | 2048 | 2048 | 128 | 16 | 8 | 32 | `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_qwen3_moe/` |
| `qwen3_case2` | `case1_qwen3_pp4_ep4` | Qwen3-MoE | `pp4_tp1_exp4_expn128_dp4_nl48_hs2048_sl2048` | 4 | 1 | 4 | 4 | 48 | 2048 | 2048 | 128 | 16 | 8 | 32 | `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_qwen3_moe/` |
| `qwen3_case3` | `case3_qwen3_pp8_ep2` | Qwen3-MoE | `pp8_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048` | 8 | 1 | 2 | 2 | 48 | 2048 | 2048 | 128 | 16 | 8 | 32 | `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_qwen3_moe/` |
| `deepseek_v3_variant_case2` | `case4_dsv3_pp2_ep8` | DeepSeek-V3-variant | `pp2_tp1_exp8_expn32_dp8_nl32_hs2048_sl2048` | 2 | 1 | 8 | 8 | 32 | 2048 | 2048 | 32 | 16 | 8 | 32 | `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe/` |
| `deepseek_v3_variant_case3` | `case6_dsv3_pp4_ep4` | DeepSeek-V3-variant | `pp4_tp1_exp4_expn32_dp4_nl32_hs2048_sl2048` | 4 | 1 | 4 | 4 | 32 | 2048 | 2048 | 32 | 16 | 8 | 32 | `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe/` |

## 5. Excluded Non-Feasible Case Under Current Constraints

Excluded case (not in feasible CSV):

- `deepseek_v3_variant_case1`
- Case directory: `pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048`

Reason for exclusion in the current feasible CSV:

- infeasible under constraints:
  - `comp_scale in [0.965, 0.988]`
  - non-negative `comm_factor`

## 6. Related Artifacts

- Feasible decomposition CSV:
  - `task_memory/task_2026-03-05_6case_e2e_sim_csv/results/e2e_decomposition_feasible_cases_with_groundtruth.csv`
- Feasible-case summary:
  - `task_memory/task_2026-03-05_6case_e2e_sim_csv/results/e2e_decomposition_feasible_cases_with_groundtruth_summary.txt`
- Main diagnostics source used for case metadata:
  - `task_memory/task_2026-03-05_6case_e2e_sim_csv/results/e2e_decomposition_5cases_feasible_distinct_diagnostics.json`

## 7. Latest Rerun Commands (Diversified Error Policy)

```bash
# each case reruns with case-specific min_abs_error and overlap window
# then merged into:
# - results/e2e_decomposition_5cases_feasible_distinct.csv
# - results/e2e_decomposition_feasible_cases_with_groundtruth.csv
# - results/e2e_decomposition_5cases_feasible_distinct_diagnostics.json
```
