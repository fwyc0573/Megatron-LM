## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Added H800 16-GPU Qwen3 + DeepSeek-V3-variant simulate/profile comparison report (6-case matrix) |
| 2026-03-04 | Re-ran with corrected Step1 GBS formula from runtime scripts and replaced official results |

## Test Report: H800 16-GPU Qwen3 + DeepSeek-V3-variant Simulate vs Profile (collective-sim)

**Date**: 2026-03-04  
**Environment**:
- Conda: `myenv_yc`
- Python: `3.9.18` (`/opt/anaconda/envs/myenv_yc/bin/python`)
- PYTHONPATH: `$(pwd)/megatron-sim-engine:$PYTHONPATH`

### 1) Test Script Information

- Dataset roots:
  - Qwen3: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_qwen3_moe`
  - DeepSeek-V3-variant: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe`
- Simulation backend:
  - `collective-sim`
- Execution speed mode:
  - in-process collective prediction cache enabled (canonical participant normalization)
- E2E extraction method:
  - For each mode (`simulate` / `profile`), run `SimulatorEngine.start_running()` and compute E2E as:
    - `max(op.finish_time)` across all ranks and comp+comm timeline ops.
- Step1 schedule formula correction (from runtime scripts):
  - `NUM_MICBATCH=4*PP`
  - `GBS=NUM_MICBATCH*MICRO_BATCH_SIZE*(GPUS_PER_NODE/TP/PP)`
  - with `MICRO_BATCH_SIZE=1`, `GPUS_PER_NODE=8`, `TP=1` -> `GBS=32` for all 6 cases

### 2) Validation Criteria

1. Each case completes both `simulate` and `profile` mode without runtime failure.
2. Final E2E latency is extracted with a consistent definition (rank timeline max finish time).
3. Gap metric is computed as:
   - `gap_pct = (simulate_e2e - profile_e2e) / profile_e2e * 100%`
4. Provide both per-case results and aggregate statistics (overall + per-model).

### 3) Test Results and Evidence

#### 3.1 Per-case results

| Model | Case | Simulate E2E (ms) | Profile E2E (ms) | Gap | Slowest Rank (Sim/Profile) |
|------|------|-------------------:|-----------------:|----:|----------------------------:|
| Qwen3 | pp2_tp1_exp8_expn128_dp8_nl48_hs2048_sl2048 | 1750.64 | 1467.29 | +19.31% | 5 / 2 |
| Qwen3 | pp4_tp1_exp4_expn128_dp4_nl48_hs2048_sl2048 | 1454.53 | 1619.90 | -10.21% | 3 / 2 |
| Qwen3 | pp8_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048 | 1467.29 | 1728.51 | -15.11% | 1 / 3 |
| DeepSeek-V3-variant | pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048 | 1605.61 | 860.53 | +86.58% | 5 / 8 |
| DeepSeek-V3-variant | pp2_tp1_exp8_expn32_dp8_nl32_hs2048_sl2048 | 1128.02 | 878.37 | +28.42% | 4 / 12 |
| DeepSeek-V3-variant | pp4_tp1_exp4_expn32_dp4_nl32_hs2048_sl2048 | 709.13 | 915.47 | -22.54% | 2 / 4 |

#### 3.2 Aggregate statistics (all 6 cases)

- Mean gap: `+14.41%`
- Mean absolute gap: `30.36%`
- Median absolute gap: `20.92%`
- Worst underestimation: `-22.54%` (`pp4_tp1_exp4_expn32_dp4_nl32_hs2048_sl2048`)
- Worst overestimation: `+86.58%` (`pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048`)

#### 3.3 Aggregate statistics by model

| Model | Cases | Mean Gap | Mean Abs Gap | Median Abs Gap | Worst Under | Worst Over |
|------|------:|---------:|-------------:|---------------:|------------:|-----------:|
| Qwen3 | 3 | -2.00% | 14.88% | 15.11% | -15.11% | +19.31% |
| DeepSeek-V3-variant | 3 | +30.82% | 45.85% | 28.42% | -22.54% | +86.58% |

#### 3.4 Runtime note

- All 6 cases completed successfully in this rerun.
- Total simulate runtime: `167.43s` (across 6 cases).
- Total profile runtime: `0.09s` (across 6 cases).
- collective-sim cache totals: hits=`9431`, misses=`55`

### 4) Evidence Artifacts

- Step1 schedule correction summary:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04_gbs_fix.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04_gbs_fix.md`
- Consolidated Step2 JSON summary (official replacement):
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_qwen3_deepseek_v3_variant_collective_sim_comparison_2026-03-04.json`
- Residual per-op report (post-fix):
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.md`

### 5) Findings

1. 根据实际运行脚本，Step1 调度的 GBS 口径应为固定 `32`（本批 6-case），而不是 `4*PP*DP`。
2. 应用正确 GBS 后，系统性过估显著收敛，但仍存在 residual gap。
3. residual 主要分布在部分 case 的 comp 偏差与通信等待链路（详见 residual per-op 报告）。
