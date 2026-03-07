## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-07 | Diagnosed detached-run cleanup behavior and relaunched Config3/Config4 with setsid-managed sessions |
| 2026-03-04 | Created progress tracker and initialized task status |
| 2026-03-04 | Updated with script implementation and dry-run validation evidence |
| 2026-03-04 | Recorded real-run failures and blocker diagnostics |
| 2026-03-04 | Added query-group divisibility guard in unit test and re-validated |
| 2026-03-04 | Added TE RMSNorm + MoE scaling row-restore fixes and restarted real run |
| 2026-03-04 | Completed Config2 real measurement in resume run and entered Config3 |
| 2026-03-04 | Recorded new Config3 runtime failure at rank 1400 (TE RMSNorm view error) |
| 2026-03-06 | Fixed attention q/k layernorm contiguity path, added partial-rank MoE simulation switch, and validated rank1400 + engine runs |
| 2026-03-06 | Launched dedicated full Config3/Config4 wall-clock background runs on GPUs 0/1 and cancelled duplicate local retry |

# Progress

## Status
- [x] Plan confirmed
- [x] Core script implemented
- [x] Unit test added and passed
- [x] Integration dry-run test added and passed
- [ ] Real measurement completed (Config3/Config4 background full sweeps are running; final rows not written yet)
- [ ] Final CSV and report verified with full Config3/Config4 real timing numbers

## Timeline
- 2026-03-04: Added `examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`.
- 2026-03-04: Added `tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh`.
- 2026-03-04: Added `tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh`.
- 2026-03-04: Completed syntax checks for script/unit/integration shell tests.
- 2026-03-04: Unit test passed.
- 2026-03-04: Integration dry-run passed, artifacts saved under `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_084535_2712073` and rerun artifact `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_085453_2734532`.
- 2026-03-04: Real run attempt #1 failed early with `ValueError: num_query_groups (4) must be a multiple of tensor_model_parallel_size (8)`.
- 2026-03-04: Updated script `NUM_QUERY_GROUPS` from `4` to `8` and re-validated syntax/unit/integration.
- 2026-03-04: Real run attempt #2 and #3 failed at rank0 with TE RMSNorm runtime error (`view` on non-contiguous tensor).
- 2026-03-04: Added unit-test assertion `NUM_QUERY_GROUPS % tp_size == 0` to prevent regression.
- 2026-03-04: Added TENorm contiguous-input fix + unit test (`test_tenorm_dtype_cast.py`).
- 2026-03-04: Added MoE scaling row-restore fix + unit tests (`test_token_dispatcher_shape_restore.py`).
- 2026-03-04: Restarted real run with log `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_20260304_091248.log`; run progressed beyond previous rank0 blocker (observed >= rank40).
- 2026-03-04: Started resume run for remaining configs with log `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_resume_20260304_092804.log`.
- 2026-03-04: Resume run completed Config2 (`ws1024_pp8_tp8_ep16_dp16`) and wrote CSV row:
  `1024,8,8,16,16,128,2147.262902,10736.314510`.
- 2026-03-04: Resume run entered Config3 (`ws4096_pp16_tp8_ep32_dp32`), currently measuring `PP*EP=512` representative ranks.
- 2026-03-04: Resume run failed in Config3 at rank `1400` with TE RMSNorm runtime error:
  `RuntimeError: view size is not compatible with input tensor's size and stride ...`.

- 2026-03-06: Added `tests/unit_tests/transformer/test_attention_qk_layernorm_contiguous.py` and patched `megatron/core/transformer/attention.py` so q/k layernorm receives contiguous tensors before TE RMSNorm.
- 2026-03-06: Added `megatron-sim-engine` CLI switch `--moe-rank-selection {all,pp-ep}` and MoE representative-rank selection implementation in `megatron-sim-engine/src/core/simu_engine.py`.
- 2026-03-06: Validation passed for:
  - `pytest -q megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py megatron-sim-engine/tests/unit/test_simu_main_moe_rank_selection.py tests/unit_tests/transformer/test_attention_qk_layernorm_contiguous.py`
  - `pytest -q megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py`
  - `pytest -q megatron-sim-engine/tests/performance/wallclock_scaling/test_collective_sim_backend_cache.py`
- 2026-03-06: Real repro `fake_current_rank_id=1400` succeeded; previous TE RMSNorm view blocker is resolved in current workspace.
- 2026-03-06: Completed 6-case `megatron-sim-engine` partial-rank simulate runs with `collective-sim` cache; outputs recorded in CSV/report under current task directory.
- 2026-03-06: Started dedicated `Config3` full sweep on `GPU0` with output CSV `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_20260306.csv` and log `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_20260306.log`.
- 2026-03-06: Started dedicated `Config4` full sweep on `GPU1` with output CSV `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_20260306.csv` and log `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config4_20260306.log`.
- 2026-03-06: Aborted duplicate local `Config3` retry after confirming an existing `GPU0` full sweep was already running; this preserves single-GPU wall-clock purity for the official run.
- 2026-03-07: Checked background execution status and confirmed the previous `20260306` full sweeps had not completed; logs stopped at `Config3 rank 48` and `Config4 rank 8`, with no surviving worker processes.
- 2026-03-07: Identified an execution-environment issue: `nohup`-style detached runs were not surviving harness/session cleanup reliably in this container.
- 2026-03-07: Relaunched official full sweeps with `setsid` so wrapper shells are re-parented to PID 1 and keep `torchrun` children alive.
- 2026-03-07: Active official runs now write to `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_setsid_20260307.csv` and `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_setsid_20260307.csv`.
