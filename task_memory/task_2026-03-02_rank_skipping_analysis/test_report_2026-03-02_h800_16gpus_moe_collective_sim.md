## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Added H800 16-GPU MoE simulate/profile comparison validation report (8-case matrix) |

## Test Report: H800 16-GPU MoE Simulate vs Profile (collective-sim)

**Date**: 2026-03-02  
**Environment**:
- Conda: `myenv_yc`
- Python: `3.9.18` (`/opt/anaconda/envs/myenv_yc/bin/python`)
- PYTHONPATH: `$(pwd)/megatron-sim-engine:$PYTHONPATH`

### 1) Test Script Information

- Dataset root:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_moe`
- Simulation backend:
  - `collective-sim`
- E2E extraction method:
  - For each mode (`simulate` / `profile`), run `SimulatorEngine.start_running()` and compute E2E as:
    - `max(op.finish_time)` across all ranks and comp+comm timeline ops.

#### Reproducible command pattern

```bash
source /opt/anaconda/bin/activate myenv_yc
PYTHONPATH=$PWD/megatron-sim-engine:$PYTHONPATH \
CASE=<case_dir_name> PP=<pp> TP=1 EXP=<exp> WORLD=16 LOCAL=8 \
python - <<'PY'
# (Same snippet used in this session: build ParallelGroupManager/RankManager,
# run SimulatorEngine for MODE_SIMULATE and MODE_PROFILE, then print JSON result.)
PY
```

#### Cases covered (full 8-case matrix)

1. `pp2_tp1_exp2_expn8_dp8_nl8_hs4096_sl1024`
2. `pp2_tp1_exp4_expn16_dp8_nl8_hs4096_sl1024`
3. `pp2_tp1_exp4_expn8_dp8_nl8_hs4096_sl1024`
4. `pp2_tp1_exp8_expn16_dp8_nl8_hs4096_sl1024`
5. `pp2_tp1_exp8_expn8_dp8_nl8_hs4096_sl1024`
6. `pp4_tp1_exp2_expn8_dp4_nl8_hs4096_sl1024`
7. `pp4_tp1_exp4_expn16_dp4_nl8_hs4096_sl1024`
8. `pp4_tp1_exp4_expn8_dp4_nl8_hs4096_sl1024`

### 2) Validation Criteria

1. Each case can complete both `simulate` and `profile` mode without runtime failure.
2. Final E2E latency is extracted with a consistent definition (rank timeline max finish time).
3. Gap metric is computed as:
   - `gap_pct = (simulate_e2e - profile_e2e) / profile_e2e * 100%`
4. Provide both per-case results and aggregate statistics.

### 3) Test Results and Evidence

#### 3.1 Per-case results

| Case | Simulate E2E (ms) | Profile E2E (ms) | Gap | Slowest Rank (Sim/Profile) |
|------|-------------------:|-----------------:|----:|----------------------------:|
| pp2_tp1_exp2_expn8_dp8_nl8_hs4096_sl1024 | 673.59 | 692.95 | -2.79% | 0 / 1 |
| pp2_tp1_exp4_expn16_dp8_nl8_hs4096_sl1024 | 736.52 | 741.50 | -0.67% | 2 / 6 |
| pp2_tp1_exp4_expn8_dp8_nl8_hs4096_sl1024 | 584.07 | 583.40 | +0.11% | 6 / 5 |
| pp2_tp1_exp8_expn16_dp8_nl8_hs4096_sl1024 | 650.47 | 666.11 | -2.35% | 6 / 6 |
| pp2_tp1_exp8_expn8_dp8_nl8_hs4096_sl1024 | 730.84 | 669.47 | +9.17% | 4 / 6 |
| pp4_tp1_exp2_expn8_dp4_nl8_hs4096_sl1024 | 576.18 | 610.88 | -5.68% | 1 / 5 |
| pp4_tp1_exp4_expn16_dp4_nl8_hs4096_sl1024 | 625.76 | 657.02 | -4.76% | 1 / 6 |
| pp4_tp1_exp4_expn8_dp4_nl8_hs4096_sl1024 | 519.54 | 564.28 | -7.93% | 0 / 6 |

#### 3.2 Aggregate statistics (8 cases)

- Mean gap: `-1.86%`
- Mean absolute gap: `4.18%`
- Median absolute gap: `3.77%`
- Worst underestimation: `-7.93%` (`pp4_tp1_exp4_expn8_dp4_nl8_hs4096_sl1024`)
- Worst overestimation: `+9.17%` (`pp2_tp1_exp8_expn8_dp8_nl8_hs4096_sl1024`)

#### 3.3 Runtime note

- All 8 cases completed successfully in this run.
- Simulate mode runtime per case was ~7–13 minutes (profile mode ~0.3s/case).

### 4) Evidence Artifacts

- Consolidated JSON summary:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_moe_collective_sim_comparison_2026-03-02.json`

### 5) Findings

1. 对于 H800 16-GPU 数据集，当前 engine 修复后多数 case 的 simulate/profile gap 已收敛到 `~0%–8%` 区间。
2. 仍存在一个明显 overestimation case（`pp2_tp1_exp8_expn8_dp8...`, `+9.17%`），提示高 `EXP` + 特定 `expn` 组合下通信/等待建模仍有偏差。
3. 该偏差模式与此前 8-rank 结论一致：engine 语义级 bug 已修复，残余主要在 comm realism（非 rank 选择错误）。
