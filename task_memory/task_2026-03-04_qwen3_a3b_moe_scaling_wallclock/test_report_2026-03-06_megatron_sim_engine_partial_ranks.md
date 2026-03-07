## Test Report: Megatron-Sim-Engine Partial-Rank MoE Simulation

**Date**: 2026-03-06
**Environment**: `SIMULATOR_HARDWARE_TYPE=H800_SXM`, Python `python`, engine cwd `megatron-sim-engine/`

### Test Script Information
- Main command pattern: `python megatron-sim-engine/simu_main.py --mode simulate --cc-backend collective-sim --moe-rank-selection pp-ep --no-visualize ...`
- Cache file: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/collective_sim_cache_moe_partial_20260306.json`
- Result CSV: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/megatron_sim_engine_partial_ranks_20260306.csv`

### Validation Criteria
- All 6 cases exit with code `0`.
- CLI log contains `cc_backend=collective-sim` and `moe_rank_selection=pp-ep`.
- Recorded `selected_ranks_count` equals `PP * EP` for every case.
- `sim load time`, `sim execution time`, and external wall-clock are all parsed and stored.

### Test Results

| Case | Result | Selected ranks | Load (s) | Execution (s) | Outer wall-clock (s) |
|------|--------|----------------|----------|----------------|----------------------|
| qwen3_case1 | PASS | 16 | 0.785322 | 42.295990 | 43.328583 |
| qwen3_case2 | PASS | 16 | 0.757422 | 11.011104 | 12.004683 |
| qwen3_case3 | PASS | 16 | 0.767166 | 3.978587 | 4.990645 |
| deepseek_case1 | PASS | 8 | 0.424408 | 18.567982 | 19.166348 |
| deepseek_case2 | PASS | 16 | 0.653957 | 26.409293 | 27.246782 |
| deepseek_case3 | PASS | 16 | 0.504241 | 4.284021 | 4.993603 |

### Evidence
- `qwen3_case1` log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/qwen3_case1.stdout.log`
- `qwen3_case2` log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/qwen3_case2.stdout.log`
- `qwen3_case3` log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/qwen3_case3.stdout.log`
- `deepseek_case1` log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/deepseek_case1.stdout.log`
- `deepseek_case2` log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/deepseek_case2.stdout.log`
- `deepseek_case3` log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/deepseek_case3.stdout.log`

### A/B Acceleration Evidence

| Case | Policy | Selected ranks | Load (s) | Execution (s) | Outer wall-clock (s) |
|------|--------|----------------|----------|----------------|----------------------|
| `h800_16gpus_moe/pp2_tp1_exp2_expn8_dp8_nl8_hs4096_sl1024` | `all` | 16 | 0.318575 | 36.945331 | 37.487937 |
| `h800_16gpus_moe/pp2_tp1_exp2_expn8_dp8_nl8_hs4096_sl1024` | `pp-ep` | 4 | 0.103020 | 0.094953 | 0.359555 |

- A/B logs:
  - `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/ab_compare_all.stdout.log`
  - `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/megatron_sim_engine_partial_ranks_20260306/ab_compare_pp-ep.stdout.log`
