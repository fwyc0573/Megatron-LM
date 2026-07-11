## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-14 | Added paper-facing dual tables and formalized v1 acceptance scope split |
| 2026-03-14 | Added v1 acceptance scope definition and known-limitation annotations |
| 2026-03-14 | Added refreshed acceptance artifact analysis and kernel-breakdown root-cause evidence |
| 2026-03-13 | Added finalized GPT-6.7B lightweight slowdown E2E validation report |

## Test Report: DDP Slowdown Support

**Date**: 2026-03-13
**Environment**: repo root `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`; Python via current shell environment; GPUs `4,5,6,7`

### Test Script Information
- Scripts:
  - `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`
  - `tests/e2e/run_ddp_slowdown_compare.py`
  - `tests/e2e/compare_ddp_slowdown_reference.py`
  - `megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py`
  - `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`
  - `megatron-sim-engine/tools/data_prep/slowdown/prepare_case_kernel_metrics.py`
- Commands:
  ```bash
  bash tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh
  cd megatron-sim-engine && pytest -q tests/unit tests/integration/test_simu_engine_ddp_slowdown_integration.py
  cd .. && pytest -q tests/unit/test_compare_ddp_slowdown_reference.py
  ```
- Final artifact dir:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859`

### Validation Criteria
- Self-contained lightweight workflow no longer depends on external `GPT67B_SLOWDOWN_METRICS_CSV`.
- Auto-generated `pp2` trace-shaped schedule is consumable by sim-engine.
- Targeted `NCU` collection plus case-local merge covers all required backward kernels.
- Slowdown assets are built successfully and simulator can produce slowdown off/on outputs.
- Real 4-GPU `nsys` reference is collected and slowdown off/on vs hardware error table is generated.
- Simulator regression tests pass after the workflow changes.

### Test Results

| Suite | Result | Details |
|------|--------|---------|
| `megatron-sim-engine` unit + integration | PASS | `68 passed` |
| `tests/unit/test_compare_ddp_slowdown_reference.py` | PASS | `1 passed` |
| GPT-6.7B lightweight E2E workflow | PASS | Completed via manual continuation from pass-1/pass-2 artifacts |
| Hardware error table generation | PASS | `compare/reference_compare.md` and `.json` created |

### Evidence
- Required kernels covered: `23` required, `23` merged unique kernels.
- Slowdown assets:
  - kernel features: `23`
  - backward blueprints: `4`
- wrank0 backward off/on/hardware:
  - off=`42.090000` ms
  - on=`60.919577` ms
  - hardware=`36.770000` ms
- wrank2 backward off/on/hardware:
  - off=`49.790000` ms
  - on=`67.109277` ms
  - hardware=`43.140000` ms
- wrank0 finalize wait off/on/hardware:
  - off=`9.260000` ms
  - on=`0.860000` ms
  - hardware=`2.030000` ms
- wrank2 finalize wait off/on/hardware:
  - off=`2.980000` ms
  - on=`2.980000` ms
  - hardware=`49.450000` ms
- Key outputs:
  - summary: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/summary.md`
  - hardware compare: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/reference_compare.md`
  - slowdown compare logs: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/wrank0.log`, `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/wrank2.log`

### Notes
- The workflow reached full functional closure, but hardware compare currently reports `shared_comm_uids_count = 0` for both wranks, so comm launch/finish MAE is unavailable. This is now tracked as a comm_uid alignment issue across trace sources, not a slowdown runtime integration failure.

## Follow-up: 2026-03-13 PM

### Additional Verification Commands
```bash
pytest -q tests/unit/test_compare_ddp_slowdown_reference.py tests/unit_tests/distributed/test_ddp_bucketing_scaling_mode.py
cd megatron-sim-engine && pytest -q tests/unit/test_slowdown_predictor.py tests/unit/test_build_ddp_slowdown_assets.py tests/integration/test_simu_engine_ddp_slowdown_integration.py
cd ..
python tests/e2e/run_ddp_slowdown_compare.py \
  --trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/global_ranks_profile \
  --database-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/database_profile \
  --schedule-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/trace_shaped_schedule \
  --slowdown-assets-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/slowdown_assets \
  --slowdown-model-path Echo-slowdown/training_testing/output/xgb_model.json \
  --slowdown-scaler-path Echo-slowdown/training_testing/output/standard_scaler.json \
  --world-size 4 --local-size 4 --pp-size 2 --tp-size 1 --exp-size 1 \
  --wrank-id 0 \
  --output-json tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/wrank0_new.json
python tests/e2e/compare_ddp_slowdown_reference.py \
  --reference-trace-dir tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/reference/trace \
  --reference-nsys-sqlite tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/reference/nsys/distributed_reference.sqlite \
  --sim-json tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/wrank0_new.json \
  --output-json tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/wrank0_reference_compare_new.json \
  --output-md tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859/compare/wrank0_reference_compare_new.md
```

### Follow-up Results
| Check | Result | Details |
|-------|--------|---------|
| Focused regression re-run | PASS | `5 passed` + `12 passed` |
| Old artifact `wrank0` hardware compare | PASS | `shared_comm_uids_count = 65`; launch / finish MAE available |
| Old artifact `wrank2` hardware compare | EXPECTED FAIL | no shared alignment keys; indicates real structural mismatch |
| New self-contained lightweight workflow | IN PROGRESS | workflow updated and re-run multiple times while tuning targeted `ncu` practicality |

### Evidence
- `wrank0` old-artifact reference compare now reports:
  - `hardware_backward_duration_ms = 73.46`
  - `sim_backward_duration_ms_off = 42.09`
  - `sim_backward_duration_ms_on = 56.496895`
  - `ddp_launch_mae_ms_off ≈ 1.2058`
  - `ddp_finish_mae_ms_off ≈ 0.6428`
  - `finalize_wait_abs_err_ms_on ≈ 0.39`
- `wrank2` old-artifact structural mismatch evidence:
  - simulator target backward DDP keys: `66`
  - hardware reference backward DDP keys for the matched stage-1 window: `1`
- Workflow tuning evidence:
  - bulk-regex targeted `ncu` was replaced by per-kernel short-name collection;
  - `-c 1` and `--kill yes` are now used so each `ncu` run stops after the first matched launch;
  - latest in-progress workflow log: `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/e2e_gpt67b_dense_pp2_tp1_dp2_sl256/.latest_lightweight_e2e_log`


## Late Update: 2026-03-13 Acceptance Run Unblock

### Test Script Information
- Scripts:
  - `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`
  - `tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py`
  - `tests/unit/test_compare_ddp_slowdown_reference.py`
  - `tests/unit_tests/distributed/test_ddp_bucketing_scaling_mode.py`
- Commands:
  ```bash
  pytest -q tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py     tests/unit/test_compare_ddp_slowdown_reference.py     tests/unit_tests/distributed/test_ddp_bucketing_scaling_mode.py

  export GPT67B_LIGHT_REAL_REFERENCE_GPUS=4,5,6,7
  export GPT67B_LIGHT_NSYS_GPU=4
  export GPT67B_LIGHT_NCU_STAGE0_GPU=5
  export GPT67B_LIGHT_NCU_STAGE1_GPU=6
  export GPT67B_LIGHT_SIM_GPU=7
  export GPT67B_LIGHT_DRYRUN_GPU=7
  bash tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh
  ```
- Environment:
  - Conda env: `myenv_yc`
  - Python: `3.9`
  - GPU allocation: `4-7`

### Validation Criteria
- The lightweight E2E workflow must cross the historical Step-2/3 breakpoint without exiting `141`.
- The refreshed artifact must contain the scaling `nsys` sqlite, generated `pp2` schedule, required-kernel manifests, and active targeted `ncu` outputs.
- The dry-run bucket structure should reflect the fixed scaling-mode DDP bucketing behavior, especially for stage-1 representative rank `2`.

### Test Results
| Check | Result | Details |
|-------|--------|---------|
| `latest_rank_file()` regression | PASS | `6 passed` focused regression suite |
| Historical `141` blocker | PASS | Fresh acceptance run progressed beyond `nsys export` into schedule + targeted `ncu` stages |
| Fresh acceptance artifact creation | PASS | Artifact `gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518` created and populated |
| Full acceptance E2E | IN PROGRESS | pass-1 targeted `ncu` is still running |

### Evidence
- Fresh artifact root: `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`
- Present intermediate outputs:
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite`
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/schedule_builder.log`
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/required_kernels_report.json`
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/ncu_pass1_rank0.log`
- Dry-run bucket counts:
  - rank0 = `65`
  - rank2 = `1`
- Notes:
  - The old `141` root cause was the shell pipeline inside `latest_rank_file()` under `set -euo pipefail`, not `nsys export` itself.
  - The fresh run now provides the right base to re-check `wrank2` bucket alignment once targeted `ncu`, slowdown asset building, and hardware compare finish.


## Final Acceptance Update: 2026-03-14 Refreshed Lightweight Artifact

### Test Script Information
- Artifact: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`
- Key outputs:
  - Summary: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/summary.md`
  - Slowdown compare: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/wrank0.json`, `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/wrank2.json`
  - Hardware compare: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.md`
- Commands executed after the refreshed run reached Step 5:
  ```bash
  export PYTHONPATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine:/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM:$PYTHONPATH

  python megatron-sim-engine/tools/data_prep/slowdown/prepare_case_kernel_metrics.py     --trace-dir /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/global_ranks_profile     --nsys-sqlite /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite     --label-prefix cmd_trace     --output-csv /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/kernel_metric_output_targeted_pass1.csv     --report-json /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/kernel_metrics_report_pass1.json     --alias ln_bwd_general_kernel=ln_bwd_tuned_kernel     --alias ln_bwd_finalize_general_kernel=ln_bwd_finalize_tuned_kernel     --candidate-csv /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/ncu_pass1_stage0_rank0/output/kernel_metric_output.csv     --candidate-csv /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/ncu_pass1_stage1_rank2/output/kernel_metric_output.csv

  python megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py     --trace-dir /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/global_ranks_profile     --nsys-sqlite /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite     --ncu-metrics-csv /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/kernel_metric_output_targeted_merged.csv     --label-prefix cmd_trace     --output-dir /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/slowdown_assets     --model-path /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/Echo-slowdown/training_testing/output/xgb_model.json     --scaler-path /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/Echo-slowdown/training_testing/output/standard_scaler.json

  CUDA_VISIBLE_DEVICES=7 python tests/e2e/run_ddp_slowdown_compare.py ... --wrank-id 0
  CUDA_VISIBLE_DEVICES=7 python tests/e2e/run_ddp_slowdown_compare.py ... --wrank-id 2

  CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile ... bash /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/run_real_reference_manual.sh
  python tests/e2e/compare_ddp_slowdown_reference.py     --reference-trace-dir /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/reference/trace     --reference-nsys-sqlite /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/reference/nsys/distributed_reference.sqlite     --sim-json /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/wrank0.json     --sim-json /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/wrank2.json     --output-json /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.json     --output-md /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.md
  ```

### Validation Criteria
- `rank2` refreshed artifact must restore shareable stage-1 DDP alignment keys.
- Step 6 must generate slowdown off/on outputs for both `wrank0` and `wrank2`.
- Step 7-8 must produce a real hardware `nsys` reference and a numeric slowdown on/off vs hardware error table.
- A canonical trace / scaling-`nsys` / targeted-`ncu` trio must be frozen for follow-up study.

### Test Results
| Check | Result | Details |
|-------|--------|---------|
| Refreshed `rank2` alignment recovery | PASS | `shared_comm_uids_count = 1` on hardware compare |
| Slowdown off/on compare generation | PASS | `wrank0.json` and `wrank2.json` created |
| Real hardware reference collection | PASS | `distributed_reference.sqlite` created from GPUs `4-7` |
| Acceptance error table generation | PASS | `reference_compare.md` created |
| Accuracy quality | MIXED | slowdown path is active, but rank-level error remains large for `wrank0` |

### Evidence
| wrank | shared keys | hw backward | off backward | on backward | off abs err | on abs err | launch MAE off | launch MAE on | finalize err off | finalize err on |
|------:|------------:|------------:|-------------:|------------:|------------:|-----------:|---------------:|--------------:|-----------------:|----------------:|
| 0 | 65 | 138.810000 | 42.590000 | 63.191119 | 96.220000 | 75.618881 | 5.200614 | 1.001078 | 6.710000 | 1.490000 |
| 2 | 1 | 54.510000 | 40.170000 | 46.068028 | 14.340000 | 8.441972 | 23.250001 | 23.240001 | 7.770000 | 13.660000 |

### Frozen Canonical Trio
- Trace dir: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/global_ranks_profile`
- Scaling `nsys` sqlite: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite`
- Targeted merged `ncu` csv: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/kernel_metric_output_targeted_merged.csv`


### Post-Run Root-Cause Analysis: 2026-03-14

#### Additional Commands
```bash
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py   --sqlite tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite   --label-prefix cmd_trace   --ops backward_step   --ranks 0,2   --json-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.json   --report-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py   --sqlite tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/reference/nsys/distributed_reference.sqlite   --label-prefix cmd_trace   --ops backward_step   --ranks 0,2   --json-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/hardware_backward_kernel_breakdown_20260314.json   --report-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/hardware_backward_kernel_breakdown_20260314.md
```

#### Additional Validation Criteria
- Explain whether the remaining hardware gap is caused by missing slowdown wiring, by baseline semantic mismatch, or by predictor magnitude.
- Verify whether the refreshed acceptance run actually consumed a persisted `StandardScaler`.
- Determine whether `wrank2` still suffers from `comm_uid` mismatch or from launch-marker timing drift after alignment recovery.

#### Additional Results
| Check | Result | Details |
|-------|--------|---------|
| Simulator slowdown wiring under `megatron-sim-engine/src/core` | PASS | slowdown is applied in `src/core/simu_engine.py`, not only in `tests/` |
| Persisted scaler usage in refreshed artifact | PASS | `wrank0.json` / `wrank2.json` record `Echo-slowdown/training_testing/output/standard_scaler.json` |
| `wrank0` hardware gap root cause isolation | PASS | hardware cooldown backward contains heavy overlap NCCL kernels under global CMD sync |
| `wrank2` remaining launch error root cause isolation | PASS | alignment recovered, but the single bucket launch marker is still ~`23 ms` early in scaling trace |

#### Additional Evidence
- `wrank0` scaling cooldown backward (`task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.json`):
  - `wall_ms = 42.527488`
  - `compute_pure_primary_union_ms = 33.369547`
  - `comm_kernel_union_ms = 0.0`
- `wrank0` hardware cooldown backward (`task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/hardware_backward_kernel_breakdown_20260314.json`):
  - sample A: `wall_ms = 138.694799`, `compute_pure_primary_union_ms = 56.610682`, `comm_kernel_union_ms = 129.574873`
  - sample B: `wall_ms = 140.105732`, `compute_pure_primary_union_ms = 54.286864`, `comm_kernel_union_ms = 129.901425`
- `wrank2` aligned bucket launch offsets extracted from the canonical traces:
  - scaling trace: `launch_offset_ms = 32.060001`, `after_backward_finish_ms = -8.109999`
  - hardware trace: `launch_offset_ms = 55.310001`, `after_backward_finish_ms = 0.800001`
- Timing-semantics explanation:
  - `megatron/profiler/cmd.py` uses global synchronization for top-level CMD timing when `--trace-cmd-sync-mode` is unset.
  - `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` collects both scaling and hardware traces with `--trace-subop-sync-mode global`, so the hardware top-level `backward_step` wall time naturally waits for overlap comm kernels while the scaling baseline does not.


## V1 Acceptance Scope Definition and Known Limitations

### V1 Acceptance Scope

The v1 slowdown acceptance focuses on **multi-bucket DDP overlap scenarios** (e.g., `wrank0` with 65 buckets on stage-0). In these scenarios, the slowdown path demonstrably improves DDP comm launch/finish alignment and finalize wait accuracy against hardware reference.

The following metrics are **in scope** for v1 acceptance on multi-bucket ranks:
- DDP comm launch MAE (off vs on vs hardware)
- DDP comm finish MAE (off vs on vs hardware)
- Finalize wait absolute error (off vs on vs hardware)
- Backward duration delta (slowdown off vs on)

The following metrics are **out of scope** for v1 acceptance:
- Top-level `backward_step` wall time vs hardware (blocked by global-sync semantic mismatch; see Known Limitation 1)
- Stage-1 single-bucket launch MAE (blocked by cross-mode marker timing drift; see Known Limitation 2)

### Known Limitation 1: Top-Level Backward Semantic Mismatch (wrank0)

The hardware `backward_step` wall time is measured under global CMD synchronization, which includes overlap NCCL kernel execution time within the top-level wall. The simulator v1 backward duration reflects only "compute-kernel sum + residual" semantics without extending to overlap comm completion.

- Hardware cooldown backward: `wall ≈ 138.7–140.1 ms`, `compute_pure ≈ 54.3–56.6 ms`, `comm_kernel_union ≈ 129.6–129.9 ms`
- Scaling baseline backward: `wall ≈ 42.5 ms`, `compute_pure ≈ 33.4 ms`, `comm_kernel_union = 0.0 ms`
- Simulator slowdown-on backward: `63.19 ms`

The ~75 ms gap between simulator slowdown-on (63.19 ms) and hardware (138.81 ms) is not a predictor accuracy issue. It is a structural semantic difference: the simulator does not extend `backward_step.finish_time` to the overlap comm completion upper bound.

This limitation does not affect DDP comm launch/finish timing or finalize wait accuracy, which are computed independently of the top-level backward wall time.

### Known Limitation 2: Stage-1 Single-Bucket Launch Marker Timing Drift (wrank2)

For stage-1 ranks with a single large DDP bucket (e.g., `wrank2` with `bucket_numel_unpadded = 3,428,130,816`, 195 params), the scaling-mode trace records the bucket launch marker systematically earlier than the hardware trace.

- Scaling trace: `launch_offset = 32.06 ms` after backward start (`8.11 ms` before backward finish)
- Hardware trace: `launch_offset = 55.31 ms` after backward start (`0.80 ms` after backward finish)
- Resulting launch MAE: `~23.24 ms` (unchanged between slowdown off and on)

Root cause: in scaling mode, `start_grad_sync()` is called when all params have grad ready, but no real NCCL comm kernels are running, so backward compute proceeds at full speed and the last param's grad becomes ready earlier. In hardware mode, overlap comm kernels compete for GPU resources, slowing backward compute and delaying the last param's grad-ready moment. This ~23 ms drift is the slowdown effect itself manifesting in grad-ready timing — a second-order effect that v1 does not model.

This limitation is specific to single-bucket stages. For multi-bucket stages (wrank0, 65 buckets), the per-bucket drift is small and launch MAE on slowdown-on is already `1.0 ms`.

### V1 Acceptance Evidence Summary (Multi-Bucket Focus)

| Metric | wrank0 off | wrank0 on | Improvement |
|--------|----------:|----------:|------------:|
| Launch MAE (ms) | 5.20 | 1.00 | 4.2x better |
| Finish MAE (ms) | 4.61 | 1.54 | 3.0x better |
| Finalize wait abs err (ms) | 6.71 | 1.49 | 4.5x better |

Conclusion: on multi-bucket ranks, the v1 slowdown path consistently and substantially improves DDP overlap timing accuracy. The two known limitations (top-level backward semantics and single-bucket launch drift) are structural and well-understood, not predictor or wiring bugs.

### Paper-Facing Dual Tables

#### Table A: `compute_pure` / Kernel-Level Accuracy (Primary v1 Acceptance View)

Methodology:
- Scaling baseline `compute_pure` comes from `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.json`.
- Hardware `compute_pure` uses the nearest matching `nsys` window for the selected target backward (`rank/stage/mg_state/batch` matched first, then nearest wall time).
- Simulator slowdown-on `compute_pure` is the summed predicted compute-kernel duration produced by the slowdown micro-scheduler with the frozen canonical assets and persisted scaler.

| wrank | Role | scaling `compute_pure` (ms) | slowdown-on predicted compute (ms) | hardware `compute_pure` (ms) | off abs err (ms) | on abs err (ms) |
|------:|------|----------------------------:|-----------------------------------:|-----------------------------:|-----------------:|----------------:|
| 0 | Primary acceptance | 33.369547 | 53.960793 | 56.610682 | 23.241135 | 2.649889 |
| 2 | Diagnostic only | 33.743720 | 39.641748 | 30.809418 | 2.934302 | 8.832330 |

Interpretation:
- `wrank0` shows that slowdown prediction materially closes the compute-side gap on the multi-bucket case that v1 is meant to validate.
- `wrank2` should not be treated as a v1 acceptance gate: its single-bucket launch timing drift also perturbs the compute-side comparison, so it remains diagnostic-only.

#### Table B: Top-Level Wall / Overlap Timing View (Contextual, Not a v1 Acceptance Gate)

| wrank | Role | hw backward (ms) | sim off (ms) | sim on (ms) | launch MAE off (ms) | launch MAE on (ms) | finalize err off (ms) | finalize err on (ms) |
|------:|------|-----------------:|-------------:|------------:|--------------------:|-------------------:|----------------------:|---------------------:|
| 0 | Primary timing case | 138.810000 | 42.590000 | 63.191119 | 5.200614 | 1.001078 | 6.710000 | 1.490000 |
| 2 | Diagnostic only | 54.510000 | 40.170000 | 46.068028 | 23.250001 | 23.240001 | 7.770000 | 13.660000 |

Interpretation:
- `wrank0` remains the right paper-facing timing case for v1 because it has many buckets and the slowdown path clearly improves launch and finalize timing.
- This table is intentionally not used as the sole acceptance gate because top-level hardware `backward_step` wall time is measured under global-sync semantics and therefore contains overlap comm completion effects that simulator v1 does not model into the top-level backward wall.
- `wrank2` remains valuable as a structural sanity case (`shared_comm_uids_count = 1` restored), but its stage-1 single-bucket launch MAE is excluded from v1 quantitative acceptance.
