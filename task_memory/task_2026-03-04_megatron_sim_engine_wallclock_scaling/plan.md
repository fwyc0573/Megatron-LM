## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initialized execution plan for megatron-sim-engine wall-clock scaling task |

# Plan: Megatron-Sim-Engine Wall-clock Scaling

## Objective
Measure simulator wall-clock at scales `8,16,32,64,256,512,1024,4096,8192` with:
- Dense rank-skipping (`tp=0, dp=0` representative rank per PP stage)
- CC backend set to `analytical` (direct `comm_sim`)
- Reused parallel configs from `task_memory/task_2026-02-28_simai_wallclock_scaling/parallel_configs.md` (new-plan table)

## Execution Steps
1. Parse and validate parallel configs from reference markdown.
2. Build stage-level schedule for each scale:
   - `PP>1`: generate via `mg_test.py`
   - `PP=1`: generate a manual stage0 schedule in matching format
3. Build dummy `database_profile` for representative ranks only.
4. Run `simu_main.py` with `--cc-backend analytical` and `--no-visualize`.
5. Parse and record:
   - `sim load time`
   - `sim execution time`
   - external process wall-clock
6. Emit CSV + markdown summary and capture per-scale logs.

## Acceptance Criteria
- All 9 scales finish successfully.
- Output CSV exists with one row per scale.
- Logs confirm `cc_backend=analytical`.
- Representative-rank count equals PP for each scale.
- No use of `collective-sim` backend.
