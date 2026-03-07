## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial execution plan for WS256 dense (H800) simulation phase |

# Plan: WS256 Dense (H800) Simulation Phase

## Scope
- Organize profiling inputs under `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2`.
- Generate schedule plan from `mg_scheduling` with parameters extracted from `examples/profile_gpt_dense_ws256_scaling.sh`.
- Run E2E simulation with `collective-sim` backend.
- Produce result summary and test report in this task directory.

## Execution Steps
1. Create target experiment directory and subdirectories (`database_profile/`, `schedule/`).
2. Copy profile data from `profiler_log/my_pp16_tp8_ep1_expnNone_dp2_nl96_hs12288_sl2048`.
3. Run `mg_test.py` with WS256/PP16/TP8/DP2/EP1/NL96/HS12288/SL2048/MBS1/GBS128.
4. Copy `stage0..stage15` scheduling files to target `schedule/`.
5. Run `simu_main.py` in simulate mode with `--cc-backend collective-sim`.
6. Extract and summarize iteration/throughput/rank-stage breakdown from simulation timelines.

## Acceptance Criteria
- `database_profile` contains 16 files with ranks `{0,16,...,240}`.
- `schedule` contains 16 files with stages `{0..15}`.
- Simulation command exits with code 0 and reports load/execution time.
- Result artifacts and a reproducible test report are saved under this task directory.
