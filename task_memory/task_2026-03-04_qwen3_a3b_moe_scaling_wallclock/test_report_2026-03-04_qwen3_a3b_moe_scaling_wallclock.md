## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-07 | Added execution-status check and setsid-managed rerun status for Config3/Config4 |
| 2026-03-04 | Initialized test report template for Qwen3-A3B MoE scaling wall-clock task |
| 2026-03-04 | Filled static, unit, and integration dry-run results with evidence |
| 2026-03-04 | Added real-run attempts, failures, and blocker diagnostics |
| 2026-03-04 | Added unit guard for `NUM_QUERY_GROUPS % tp_size == 0` and re-ran tests |
| 2026-03-04 | Added runtime-fix unit tests and latest real-run progress evidence |
| 2026-03-04 | Added resume-run evidence: Config2 completed, Config3/4 in progress |
| 2026-03-04 | Updated resume-run status: Config3 failed at rank 1400, Config4 not started |
| 2026-03-06 | Added rank1400 fix status, megatron-sim-engine partial-rank results, and background full-sweep tracking |

## Test Report: Qwen3-A3B MoE Scaling Wall-clock Scan

**Date**: 2026-03-04

### 1. Test Script Information
- Script(s):
  - `examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`
  - `tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh`
  - `tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh`
- Commands:
  ```bash
  # Static check
  bash -n examples/qwen3_a3b_moe_scaling_wallclock_scan.sh
  bash -n tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh
  bash -n tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh

  # Unit test
  bash tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh

  # Integration dry-run
  bash tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh

  # Runtime-fix unit tests
  pytest -q tests/unit_tests/transformer/test_tenorm_dtype_cast.py
  pytest -q tests/unit_tests/transformer/moe/test_token_dispatcher_shape_restore.py
  ```
- Environment:
  - Conda env: `myenv_yc`
  - Python: `3.9.18`
  - Python path: `/opt/anaconda/envs/myenv_yc/bin/python`

### 2. Validation Criteria
- Config arithmetic and MoE constraints are correct for all four fixed configurations.
- Representative rank mapping follows `rank = pp_stage * tp * dp + exp_rank * tp`.
- Dry-run generates CSV with exact header, exact field count, and expected `measured_ranks_count`.
- `estimated_5_iters_seconds` equals `single_iter_wallclock_seconds * 5` for each CSV row.
- `NUM_QUERY_GROUPS` in script must be divisible by each target `tp_size`.

### 3. Test Results and Evidence
| Suite | Result | Evidence |
|------|--------|----------|
| Static syntax check | PASS | `syntax_ok_script`, `syntax_ok_unit`, `syntax_ok_integration` |
| Unit test | PASS | `[PASS] Qwen3-A3B MoE scaling wall-clock config checks passed.` |
| Integration dry-run | PASS | `[PASS] Qwen3-A3B MoE dry-run integration checks passed.` |
| TENorm unit tests | PASS | `2 passed` in `test_tenorm_dtype_cast.py` |
| Token dispatcher shape-restore unit tests | PASS | `3 passed` in `test_token_dispatcher_shape_restore.py` |
| Real run attempt #1 | FAIL | `ValueError: num_query_groups (4) must be a multiple of tensor_model_parallel_size (8)` |
| Real run attempt #2 | FAIL | `RuntimeError: view size is not compatible ...` in TE `RMSNorm` |
| Real run attempt #3 | FAIL | Same TE `RMSNorm` runtime error after switching script default to `transformer_impl=local` |
| Real run attempt #4 | FAIL | `RuntimeError: shape '[2048, 1, 2048]' is invalid for input of size 524288` in MoE `token_unpermutation` |
| Real run attempt #5 | PARTIAL PASS | Config1 completed (`536.436879s`), then run was manually interrupted during Config2 for resume-mode switch |
| Real run attempt #6 | FAIL | Resume run failed in Config3 at rank `1400` with TE RMSNorm `view` runtime error |

#### Key dry-run evidence
- Artifact root: `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_084535_2712073`
- CSV path:
  `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_084535_2712073/qwen3_a3b_moe_scaling_wallclock_timing.csv`
- Latest rerun artifact:
  `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_085453_2734532/qwen3_a3b_moe_scaling_wallclock_timing.csv`
- Latest rerun artifact:
  `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_091234_2763725/qwen3_a3b_moe_scaling_wallclock_timing.csv`
- Expected `measured_ranks_count` by row:
  - `32`
  - `128`
  - `512`
  - `1024`

### 4. Real Measurement Status
- Status: **Blocked by Runtime Error**
- Attempted commands:
  ```bash
  SCALE_GPU=0 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_084838.log

  SCALE_GPU=0 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_084929.log

  SCALE_GPU=0 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_085126.log

  SCALE_GPU=0 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_090339.log

  SCALE_GPU=0 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_091026.log

  SCALE_GPU=0 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_091248.log

  SCALE_GPU=0 \
  APPEND_CSV=1 \
  CONFIG_START_INDEX=1 \
  CONFIG_END_INDEX=3 \
  OUTPUT_CSV=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/docs/data/qwen3_a3b_moe_scaling_wallclock_timing.csv \
  LOG_ROOT=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/log/qwen3_a3b_moe_scaling_wallclock \
  bash /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_resume_20260304_092804.log
  ```
- Failure evidence:
  - Log #1 (`...084838.log`): `ValueError: num_query_groups (4) must be a multiple of tensor_model_parallel_size (8)`
  - Log #2 (`...084929.log`): `RuntimeError: view size is not compatible ...` at `transformer_engine.pytorch.module.rmsnorm`
  - Log #3 (`...085126.log`): same TE RMSNorm `view` runtime error
  - Log #4 (`...090339.log`): `RuntimeError: shape '[2048, 1, 4096]' is invalid for input of size 1048576` (local dot-product path mismatch)
  - Log #5 (`...091026.log`): `RuntimeError: shape '[2048, 1, 2048]' is invalid for input of size 524288` at `token_dispatcher.token_unpermutation`
- Resolution status:
  - Query-group issue fixed in script (`NUM_QUERY_GROUPS=8`)
  - TE RMSNorm runtime error fixed
  - MoE token dispatcher scaling reshape mismatch fixed
  - Real run `...091248.log` completed Config1 and produced CSV row:
    `256,8,8,4,4,32,536.436879,2682.184395`
  - Resume run `...092804.log` completed Config2 and produced CSV row:
    `1024,8,8,16,16,128,2147.262902,10736.314510`
  - Resume run `...092804.log` failed in Config3 at rank `1400`:
    `RuntimeError: view size is not compatible with input tensor's size and stride ...`
  - Config3/Config4 final CSV rows are pending.

### 4.1 Current 2026-03-06 Runtime Status
- Rank1400 blocker is resolved in current workspace; evidence: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/repro_rank1400_20260306.log`.
- `megatron-sim-engine` partial-rank MoE simulation batch is complete; evidence: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/test_report_2026-03-06_megatron_sim_engine_partial_ranks.md`.
- Dedicated full sweeps are currently running in background for the remaining real wall-clock rows:
  - `Config3@GPU0`: log `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_20260306.log`, CSV `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_20260306.csv`
  - `Config4@GPU1`: log `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config4_20260306.log`, CSV `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_20260306.csv`
- Duplicate local `Config3` retry was cancelled to avoid contaminating single-GPU wall-clock measurements.

- On 2026-03-07, a fresh execution-status audit confirmed the previous `20260306` background attempts had not completed.
- Fresh mitigation: relaunch the remaining `Config3`/`Config4` full sweeps with `setsid`, not `nohup`, so the wrapper shell is re-parented to PID 1 and the `torchrun` children survive session cleanup.
- Active 2026-03-07 artifacts:
  - `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_setsid_20260307.csv`
  - `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_setsid_20260307.csv`
  - `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_setsid_20260307.log`
  - `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config4_setsid_20260307.log`

### 5. Interim Real Timing Snapshot
Current CSV (`docs/data/qwen3_a3b_moe_scaling_wallclock_timing.csv`) rows:

| world_size | pp_size | tp_size | ep_size | dp_size | measured_ranks_count | single_iter_wallclock_seconds | estimated_5_iters_seconds |
|-----------:|--------:|--------:|--------:|--------:|---------------------:|------------------------------:|--------------------------:|
| 256 | 8 | 8 | 4 | 4 | 32 | 536.436879 | 2682.184395 |
| 1024 | 8 | 8 | 16 | 16 | 128 | 2147.262902 | 10736.314510 |
