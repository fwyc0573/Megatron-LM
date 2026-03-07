## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial test report for megatron-sim-engine wall-clock scaling execution |

# Test Report: Megatron-Sim-Engine Wall-clock Scaling

**Date**: 2026-03-04  
**Environment**:
- Python binary: `/opt/anaconda/envs/myenv_yc/bin/python`
- Python version: `Python 3.9.18`
- Working directory: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

## 1. Test Script Information

- Runner script:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/tests/performance/wallclock_scaling/run_wallclock_scaling.py`
- Unit test script:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/tests/performance/wallclock_scaling/test_wallclock_scaling_runner.py`
- Output directory (full run):
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results`

### Reproducible Commands

```bash
# Unit tests
python -m unittest megatron-sim-engine/tests/performance/wallclock_scaling/test_wallclock_scaling_runner.py -v

# Smoke run
python megatron-sim-engine/tests/performance/wallclock_scaling/run_wallclock_scaling.py \
  --scales 8,16,32 \
  --output-root task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results_smoke

# Full run (required 9 scales)
python megatron-sim-engine/tests/performance/wallclock_scaling/run_wallclock_scaling.py \
  --scales 8,16,32,64,256,512,1024,4096,8192 \
  --output-root task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results
```

## 2. Validation Criteria

1. Reference config reuse:
   - Parse configs from `task_memory/task_2026-02-28_simai_wallclock_scaling/parallel_configs.md` new-plan table.
2. Input completeness:
   - `schedule` count must be `PP` (or `world_size`), and this run enforces stage-level `PP` files.
   - `database_profile` count must match representative ranks (`PP` for dense).
3. Rank-skipping rule:
   - Representative ranks must be `pp_idx * tp * dp`.
4. Backend correctness:
   - Run with `--cc-backend analytical` and logs must confirm backend init as analytical.
5. Timing outputs:
   - Each scale must emit parseable `sim load time` and `sim execution time`.
   - External wall-clock measured by runner must be recorded.
6. Scale coverage:
   - Exactly 9 result rows for `8/16/32/64/256/512/1024/4096/8192`.

## 3. Test Results and Evidence

### 3.1 Unit Tests

| Test Suite | Result | Details |
|------------|--------|---------|
| `unittest` for runner | PASS | `Ran 4 tests ... OK` |

Evidence excerpt:
- `test_parse_reference_new_plan_configs ... ok`
- `test_representative_ranks ... ok`
- `test_pp1_schedule_file_contains_required_ops ... ok`
- `test_dummy_database_profile_per_stage_rank ... ok`

### 3.2 Smoke Run (`8/16/32`)

| Scale | Result | Evidence |
|-------|--------|----------|
| 8     | PASS | `[done] world_size=8 ...` |
| 16    | PASS | `[done] world_size=16 ...` |
| 32    | PASS | `[done] world_size=32 ...` |

Output artifacts:
- `task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results_smoke/wallclock_scaling_megatron_sim_engine.csv`
- `task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results_smoke/wallclock_scaling_megatron_sim_engine.md`

### 3.3 Full Run (9 Scales)

| Scale | TP | PP | DP | sim_load_time_s | sim_execution_time_s | outer_wallclock_s | Result |
|------:|---:|---:|---:|----------------:|---------------------:|------------------:|--------|
| 8    | 8 | 1  | 1  | 0.001381 | 0.000002 | 0.144351 | PASS |
| 16   | 8 | 2  | 1  | 0.002721 | 0.000712 | 0.146191 | PASS |
| 32   | 8 | 4  | 1  | 0.004285 | 0.001267 | 0.153547 | PASS |
| 64   | 8 | 8  | 1  | 0.007310 | 0.002396 | 0.155715 | PASS |
| 256  | 8 | 8  | 4  | 0.007287 | 0.003207 | 0.167944 | PASS |
| 512  | 8 | 8  | 8  | 0.007329 | 0.003247 | 0.199272 | PASS |
| 1024 | 8 | 16 | 8  | 0.011968 | 0.005833 | 0.284925 | PASS |
| 4096 | 8 | 16 | 32 | 0.013830 | 0.006704 | 2.111493 | PASS |
| 8192 | 8 | 16 | 64 | 0.012798 | 0.006114 | 6.474381 | PASS |

Evidence checks:
- CSV line count:
  - `wc -l .../wallclock_scaling_megatron_sim_engine.csv` -> `10` (header + 9 rows).
- Backend verification (all logs):
  - `cc_backend=analytical`
  - `CC backend initialized: analytical`
- Input size verification:
  - `scale_8: schedule=1, db=1`
  - `scale_16: schedule=2, db=2`
  - `scale_32: schedule=4, db=4`
  - `scale_64: schedule=8, db=8`
  - `scale_256: schedule=8, db=8`
  - `scale_512: schedule=8, db=8`
  - `scale_1024: schedule=16, db=16`
  - `scale_4096: schedule=16, db=16`
  - `scale_8192: schedule=16, db=16`

## 4. Failure Handling Record

Initial smoke attempt failed at scale 8 before final fix:
- Error: `Invalid mode, cannot get dependency relationship.`
- Root cause: default strategy `1F1B-none_interleaved` is invalid for `PP=1`.
- Resolution:
  - Runner now selects `--strategy no-pipelining` for `PP=1`.
  - Re-ran unit tests + smoke + full run; all passed.

## 5. Final Artifacts

- Main CSV:
  - `task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results/wallclock_scaling_megatron_sim_engine.csv`
- Main markdown summary:
  - `task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results/wallclock_scaling_megatron_sim_engine.md`
- Per-scale logs:
  - `task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results/logs/scale_*.stdout.log`
  - `task_memory/task_2026-03-04_megatron_sim_engine_wallclock_scaling/results/logs/scale_*.stderr.log`
