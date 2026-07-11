## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-15 | Reviewed slowdown/overlap follow-up requirements, fixed no-overlap slowdown disable path, and added overlap/slowdown examples |
| 2026-03-14 | Added paper-facing dual tables and formalized primary-vs-diagnostic acceptance scope |
| 2026-03-14 | Added canonical-artifact root-cause analysis for wrank0/wrank2 hardware gaps |
| 2026-03-12 | Started auto schedule generation and hardware-reference validation workstream |
| 2026-03-12 | Added GPT-6.7B paper-like lightweight E2E slowdown validation results |
| 2026-03-12 | Started implementation and created task documentation |
| 2026-03-12 | Integrated slowdown runtime, tests, and verification evidence |
| 2026-03-12 | Added collective-sim recovery and Echo workflow execution evidence |
| 2026-03-12 | Added E2E smoke simulate validation progress tracking |
| 2026-03-12 | Recorded minimal E2E slowdown smoke evidence |

# Progress

## 2026-03-15
- Closed Phase 8c/8d as functionally complete after re-validating the workflow entrypoints instead of treating the remaining `NCU` wall-clock cost as a blocker.
- Added missing CLI-level regression coverage for the two workflow entry scripts and re-ran the refreshed 8c/8d validation set (`31 passed`):
  - `megatron-sim-engine/tests/unit/test_build_trace_shaped_pp_schedule.py` now covers the `main()` CLI path and emitted schedule file list.
  - `megatron-sim-engine/tests/unit/test_prepare_case_kernel_metrics.py` now covers the `main()` CLI path, alias parsing, prepared CSV emission, and JSON report output.
- Refreshed the discoverability docs for this workflow:
  - `task_memory/task_2026-03-12_sim_engine_slowdown_support/notes.md` now contains an example/workflow index for Qwen overlap tracing plus Phase 8c/8d entrypoints.
  - `megatron-sim-engine/tools/README.md` and `megatron-sim-engine/tests/README.md` now point to the trace-shaped schedule builder, case-local kernel metrics preparer, and the self-contained GPT-6.7B lightweight E2E script.
- Reviewed the follow-up requirement set against the current repository state and confirmed three gaps to fix:
  - Megatron did not auto-enable `trace_ddp_grad_overlap` purely from `overlap_grad_reduce + do_trace`.
  - `megatron-sim-engine/` had `--enable-slowdown` but no explicit overlap policy control.
  - The no-overlap + slowdown path still hard-failed instead of warning and continuing without slowdown.
- Fixed Megatron-side default behavior in `megatron/training/arguments.py` so `trace_ddp_grad_overlap` is auto-enabled whenever `overlap_grad_reduce` and tracing are both active.
- Added explicit overlap policy control in `megatron-sim-engine/`:
  - `simu_main.py` now exposes `--overlap-mode {auto,on,off}`.
  - `src/core/simulator_config.py` now carries `OverlapConfig`.
  - `src/core/simu_engine.py` now:
    - auto-enables overlap when trace overlay exists and policy is `auto`,
    - fail-fasts when overlap-aware traces are run with `--overlap-mode off`,
    - warns and keeps slowdown disabled when overlap metadata is absent.
- Locked the reviewed semantics with new/updated tests:
  - `megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`
  - `megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py`
  - existing distributed/profiler overlap tests were re-run to guard regressions.
- Added example coverage for discoverability and reproducibility:
  - `examples/pretrain_deepseek_v3_moe.sh` now accepts `OVERLAP_GRAD_REDUCE=1`.
  - Added `examples/pretrain_deepseek_v3_moe_ddp_overlap_trace.sh` as a minimal wrapper showing that DDP overlap tracing is auto-enabled without explicitly passing `--trace-ddp-grad-overlap`.
  - Added `megatron-sim-engine/examples/06_ddp_overlap_slowdown_modes.sh` covering:
    - slowdown disabled by default,
    - slowdown explicitly enabled,
    - overlap force-disabled fail-fast,
    - optional warning-only no-overlap slowdown run.
- Validation completed for the reviewed scope:
  - `pytest` passed for 37 targeted slowdown/overlap tests spanning unit, integration, distributed, and profiler coverage.
  - `python -m py_compile` passed for all touched Python files.
  - `bash -n` passed for all touched example scripts.
  - `python megatron-sim-engine/simu_main.py --help` shows both `--enable-slowdown` and `--overlap-mode {auto,on,off}`.

- Fixed the real `--do-trace` CLI parsing bug in `megatron/training/arguments.py`:
  - root cause was `type=bool`, so `--do-trace False` parsed as `True` because `bool("False")` is truthy;
  - replaced it with a strict boolean parser that accepts `true/false`, `1/0`, `on/off`, and `yes/no`.
- Expanded regression coverage for the trace CLI parser:
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py` now covers explicit `False`, explicit `0`, invalid bool token rejection, and the existing overlap auto-enable path.
- Aligned the Qwen MoE stage-1 example with the DeepSeek entry style:
  - `examples/pretrain_qwen3_30b_a3b_moe.sh` now accepts `OVERLAP_GRAD_REDUCE=1` as the primary env entry while keeping `ENABLE_DDP_OVERLAP` as a backward-compatible fallback;
  - updated the top-of-file usage example to show `OVERLAP_GRAD_REDUCE=1` and optional `DDP_BUCKET_SIZE`.
- Added a lightweight no-GPU script regression for Qwen MoE overlap/tracing arguments:
  - `tests/unit/test_qwen3_a3b_moe_overlap_script.py` stubs `torchrun` and validates that both `MODE=distributed` and `MODE=scaling` forward `--overlap-grad-reduce`, `--ddp-bucket-size`, and tracing flags correctly without explicitly passing `--trace-ddp-grad-overlap`.
- Continued the bool-parser hardening sweep across all active `type=bool` CLI sites in the workspace and verified no active runtime parser still uses raw `type=bool`; archived `backup/` and `legacy/` copies were intentionally left untouched.
- Added a dedicated regression suite `tests/unit_tests/test_strict_bool_argument_parsing.py` covering the remaining active strict-bool CLI entry points in:
  - `megatron/training/arguments.py` (`--onnx-safe`, `--lazy-mpu-init`)
  - `mg_scheduling/arguments.py`
  - `megatron-sim-engine/src/scheduler/mg_scheduling/arguments.py`
  - `tools/retro/sft/sft_retro.py` (`--reset_eval`)
- Added the Qwen DeepSeek-style wrapper entry `examples/pretrain_qwen3_30b_a3b_moe_ddp_overlap_trace.sh` and aligned it one-for-one with the DeepSeek wrapper semantics (same defaults, same messaging, different model/target script only).
- While validating the new strict-bool tests, `tools/retro/sft/sft_retro.py` exposed a hidden issue: the new `_argparse_bool()` used `argparse.ArgumentTypeError` without importing `argparse`. Fixed by adding the missing import and by isolating the unit test from unrelated top-level training dependencies.
- Re-ran focused strict-bool/Qwen-wrapper regressions (`23 passed`) and the expanded overlap/slowdown regression suite (`55 passed`) after the test isolation fix.
- Re-ran the expanded validation set after the parser and Qwen-script fixes:
  - `pytest -q megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py tests/unit_tests/test_trace_ddp_grad_overlap_args.py tests/unit_tests/distributed/test_ddp_overlap_trace.py tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py tests/unit/test_qwen3_a3b_moe_overlap_script.py`
  - `python -m py_compile megatron/training/arguments.py`
  - `bash -n examples/pretrain_qwen3_30b_a3b_moe.sh`
  - observed: `41 passed` and syntax checks passed.

## 2026-03-14
- Re-ran kernel-ground-truth breakdown on the frozen canonical trio using:
  - `python tests/performance/analyze_nsys_cmd_kernel_breakdown.py --sqlite tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite --label-prefix cmd_trace --ops backward_step --ranks 0,2 --json-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.json --report-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.md`
  - `python tests/performance/analyze_nsys_cmd_kernel_breakdown.py --sqlite tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/reference/nsys/distributed_reference.sqlite --label-prefix cmd_trace --ops backward_step --ranks 0,2 --json-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/hardware_backward_kernel_breakdown_20260314.json --report-path task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/hardware_backward_kernel_breakdown_20260314.md`
- Confirmed the slowdown path is wired into simulator runtime, not only tests:
  - `megatron-sim-engine/src/core/simu_engine.py` runs `_add_trace_driven_backward_slowdown()` for trace-driven `backward_step` and reuses the precomputed `comm_uid -> launch/finish` map in `_add_trace_driven_async_ddp_overlap_comm()`.
  - `megatron-sim-engine/src/extensions/slowdown_predictor.py` loads both `xgb_model.json` and `standard_scaler.json`; the refreshed `wrank0.json` / `wrank2.json` explicitly record the scaler path, so the current artifact is not missing `StandardScaler` at prediction time.
- Root-cause finding for `wrank0`:
  - Scaling cooldown backward is `wall=42.53 ms`, `compute_pure_primary_union=33.37 ms`, `comm_kernel_union=0.0 ms`.
  - Hardware cooldown backward is `wall≈138.69-140.11 ms`, `compute_pure_primary_union≈54.29-56.61 ms`, `comm_kernel_union≈129.57-129.90 ms`.
  - `megatron/profiler/cmd.py` still defaults top-level CMD timing to global synchronization when `--trace-cmd-sync-mode` is unset, and `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` collects both scaling and hardware traces with `--trace-subop-sync-mode global`.
  - Therefore the hardware `backward_step` wall time already includes overlap NCCL work launched on other streams, while the scaling baseline has no real comm kernels to wait for. This explains why `wrank0` hardware backward is much larger than both slowdown-off and slowdown-on simulator results even though launch/finalize alignment improved.
- Root-cause finding for `wrank2`:
  - The only shared bucket is aligned structurally, but its launch marker is semantically early in the scaling trace.
  - Scaling trace offset for the aligned bucket is `32.06 ms` after backward start and `8.11 ms` before backward finish.
  - Hardware trace offset for the aligned bucket is `55.31 ms` after backward start and `0.80 ms` after backward finish.
  - This explains why `shared_comm_uids_count` recovered to `1` but `launch_MAE` remains around `23.24 ms`: the remaining error is not a `comm_uid` mismatch anymore, but a cross-mode launch-marker semantic mismatch on stage-1's single giant bucket.
- Recorded the refreshed root-cause evidence in task docs before proposing any simulator calibration, because changing top-level backward semantics would be a real logic change rather than a pure bug fix.
- Formalized the v1 acceptance split in the report and artifact summary:
  - `wrank0` multi-bucket overlap is now the primary acceptance case.
  - `wrank2` stage-1 single-bucket launch MAE is diagnostic-only.
  - Added paper-facing dual tables separating `compute_pure` accuracy from top-level wall / overlap timing behavior.

## 2026-03-12
- Created task directory and planning docs.
- Confirmed `Echo-slowdown/` true feature set and mismatch vs paper doc.
- Confirmed simulator currently has DDP overlap replay but no slowdown runtime.
- Confirmed Megatron kernel-ground-truth NVTX label is missing `cmd_uid`.
- Added `cmd_uid` to kernel-ground-truth NVTX labels in `megatron/profiler/cmd.py`.
- Added offline slowdown asset builder under `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`.
- Added `src/extensions/slowdown_predictor.py` with:
  - manifest / blueprint / feature validation,
  - duplicate-key fail-fast JSON loading,
  - `Echo-slowdown`-compatible XGBoost adapter,
  - fixed-point overlap helpers.
- Extended `megatron-sim-engine/simu_main.py` with slowdown CLI flags:
  - `--enable-slowdown`
  - `--slowdown-assets-dir`
  - `--slowdown-model-path`
  - `--slowdown-max-iters`
  - `--slowdown-tol-ms`
- Extended `megatron-sim-engine/src/core/simulator_config.py` with `SlowdownConfig`.
- Integrated slowdown runtime into `megatron-sim-engine/src/core/simu_engine.py`:
  - load assets / model fail-fast,
  - precompute per-backward kernel slowdown schedule,
  - store slowdown-aware `comm_uid -> launch/finish` map,
  - consume precomputed comm schedule during overlay replay,
  - preserve existing finalize wait semantics,
  - fail fast on unsupported sub-op-expanded backward replay.
- Added unit tests for CLI validation, solver logic, loader/adapter behavior, and builder behavior.
- Added integration test for synthetic two-bucket backward slowdown replay.
- Restored `collective-sim` submodule under `megatron-sim-engine/src/core/cc_backend/collective-sim`.
- Restored minimal H800 measured P2P profile assets under `megatron-sim-engine/data/h800_dgx_roce_sendrecv/` so the default `collective-sim` backend can initialize in unit tests.
- Re-ran `cd megatron-sim-engine && pytest -q tests/unit` and got `57 passed in 0.57s`.
- Patched Echo workflow wrappers to remove the `jq` dependency, export the correct `PYTHONPATH`, and invoke Bash scripts with `bash` instead of `sh`.
- Patched `Echo-slowdown/slowdown_collection/run-nsys.sh` for this environment:
  - removed unsupported `nsys` flags,
  - accepted the observed `143` exit code when `.nsys-rep` was produced,
  - kept fail-fast checks on missing reports.
- Fixed `Echo-slowdown/merge/merge_script.py` to merge on normalized kernel name + occurrence index instead of the original brittle pointer walk.
- Added `tests/unit/test_echo_slowdown_merge.py` to lock the new merge behavior.
- Added `SKIP_KERNEL_METRIC=1` support to `Echo-slowdown/run_all.sh` so the practical workflow can reuse `merge/input/kernel_metric_output.csv`.
- Ran the full practical Echo workflow with:
  - `SKIP_KERNEL_METRIC=1 bash Echo-slowdown/run_all.sh`
  - output artifacts produced under `Echo-slowdown/slowdown_collection/output/`, `Echo-slowdown/merge/output/`, and `Echo-slowdown/training_testing/output/`.
- Verified direct Echo API usage with `training_testing/prediction_api.py` on an overlapped kernel row from `input/test_csv/merged_features.csv`.

- Added `tests/e2e/test_ddp_slowdown_simulate_smoke.sh` to orchestrate a trace-backed slowdown smoke:
  - sequential scaling-mode profiling for fake ranks `0..3` on one GPU,
  - a single combined `nsys` capture exported to sqlite,
  - schedule generation via `mg_test.py`,
  - slowdown asset building,
  - slowdown off/on simulate comparison through `tests/e2e/run_ddp_slowdown_compare.py`.
- Confirmed the dense smoke case must not pass `--fake-num-experts 1`; otherwise `pretrain_llama.py` enters the MoE path and fails in scaling mode with `SequentialMLP is not supported in scaling mode`.
- Corrected the E2E script to use the repo's real flag `--trace-kernel-boundary-sync-mode` and the dense scaling trace directory `profiler_log/pp1_tp1_ep1_expnNone_dp4_nl12_hs256_sl256`.

- Diagnosed the first full 4-rank smoke failure after real profiling:
  - `mg_test.py` cannot generate schedule for `PP=1`; the root cause matches the earlier wall-clock task notes (`forward_backward_func=None`).
  - `build_ddp_slowdown_assets.py` initially failed on real trace parsing because `dp_allreduce.description` contains an unquoted comma.
  - asset building on real traces also required short-name matching (`kShortName`) instead of full demangled names, consistent with `Echo-slowdown/merge/merge_script.py`.
- Fixed `megatron-sim-engine/src/core/simu_engine.py` so slowdown is applied on the `no-pipelining` replay path used by `PP=1` DDP smoke cases.
- Fixed `megatron-sim-engine/src/core/simu_engine.py` to preserve `backward_step` residual wall time not covered by kernel blueprints, preventing slowdown-on from shrinking the top-level op.
- Added regression coverage for:
  - unquoted-comma trace parsing in `test_build_ddp_slowdown_assets.py`,
  - short-name NCU feature matching in `test_build_ddp_slowdown_assets.py`,
  - residual backward duration preservation in `test_simu_engine_ddp_slowdown.py`,
  - `PP=1 no-pipelining` slowdown replay in `test_simu_engine_ddp_slowdown_integration.py`.
- Reused the existing NCU report `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/scale_rank0_ncu.ncu-rep` and exported a case-local metrics CSV with Echo's `ncu_report_process.py` logic.
- Built real slowdown assets for the minimal smoke case `megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/slowdown_assets`.
- Completed a real simulate off/on A/B on the minimal smoke case and verified slowdown injection evidence:
  - `backward_step`: `10.54 ms -> 10.541538 ms`
  - `ddp_grad_comm` launch: `8588878574.91 -> 8588878575.14 ms`
  - delayed comm uid: `ddpcomm-b2fd9ffefbea`
  - processed backward cmd uid: `cmd-91266ed41298`

## Current Focus
- Preserve the validated minimal E2E smoke artifacts and formalize the evidence in the test report.
- Optionally upgrade the newer 4-rank reprofiling script so it collects case-specific NCU metrics automatically instead of relying on a reused CSV.
- Keep acceptance-grade accuracy validation as a separate next step requiring a canonical matching trace/`nsys`/`ncu` trio.


## 2026-03-12 GPT-6.7B `pp2,tp1,dp2,sl256` Lightweight E2E
- Reconfirmed slowdown injection is implemented in simulator runtime rather than test-only glue:
  - CLI/config fail-fast plumbing in `megatron-sim-engine/simu_main.py` and `megatron-sim-engine/src/core/simulator_config.py`.
  - Predictor/assets loader in `megatron-sim-engine/src/extensions/slowdown_predictor.py`.
  - Backward micro-scheduler and slowdown-aware DDP replay in `megatron-sim-engine/src/core/simu_engine.py`.
- Added and executed `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` for a paper-like dense GPT case:
  - topology `pp2,tp1,dp2`, `seq=256`, `micro_batch_size=1`, `global_batch_size=8`, `num_layers=32`, `hidden_size=4096`.
  - GPU usage: bucket dry-run on GPU7, traced scaling run under `nsys` on GPU4.
  - Reused prepared lightweight metrics CSV `tests/e2e/artifacts/gpt67b_ddp_slowdown_e2e_bucket10000000_20260312_161902_928972/gpt67b_kernel_metric_output_lightweight.csv`, which had been assembled earlier from targeted NCU augment on GPU5/GPU6.
- Increased overlap opportunity by changing `--ddp-bucket-size` to `10000000`:
  - prior default-bucket trace evidence: rank0 `49` DDP buckets, rank2 `50` DDP buckets.
  - new dry-run evidence: rank0 `65`, rank2 `66` buckets.
  - traced case evidence: rank0/rank1 `65`, rank2/rank3 `66` `ddp_grad_comm` records.
- Materialized a trace-shaped manual PP dependency schedule because the compressed single-batch trace does not replay correctly with the full `mg_test.py` schedule.
- Generated new case-local slowdown assets:
  - case dir `megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015`
  - run dir `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015`
  - assets: `23` kernel feature rows, `4` backward kernel blueprints.
- Verified slowdown off/on compare for two representative wranks:
  - wrank0: `42.44 -> 61.035222 ms` (`+18.595222 ms`), delayed comms=`65`.
  - wrank2: `44.05 -> 62.695410 ms` (`+18.645410 ms`), delayed comms=`66`.
  - processed backward cmd_uids: `cmd-0a3db334730b`, `cmd-5fa4887655a6`.


## 2026-03-12 Auto Schedule + Self-Contained E2E Follow-up

- Completed one full GPT-6.7B lightweight slowdown E2E by manually continuing the failed shell flow from the produced pass-1/pass-2 artifacts.
- Confirmed the self-contained workflow now reaches all major stages on GPUs `4,5,6,7`:
  - scaling `nsys`,
  - trace-shaped `pp2` schedule generation,
  - targeted `NCU` pass1 + rank0-only pass2,
  - slowdown asset construction,
  - simulator slowdown off/on compare,
  - real 4-GPU hardware reference `nsys`,
  - reference error-table generation.
- Final artifact directory: `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_190346_1186859`.
- Final lightweight E2E headline results:
  - wrank0 backward `42.09 -> 60.919577 ms`, hardware `36.77 ms`.
  - wrank2 backward `49.79 -> 67.109277 ms`, hardware `43.14 ms`.
  - wrank0 finalize wait off/on/hardware = `9.26 / 0.86 / 2.03 ms`.
  - wrank2 finalize wait off/on/hardware = `2.98 / 2.98 / 49.45 ms`.
- Re-ran full simulator regression after the shell-flow changes:
  - `cd megatron-sim-engine && pytest -q tests/unit tests/integration/test_simu_engine_ddp_slowdown_integration.py` -> `68 passed`.
  - `pytest -q tests/unit/test_compare_ddp_slowdown_reference.py` -> `1 passed`.

- Continued the GPT-6.7B lightweight proving run on GPUs `4,5,6,7` after wiring the self-contained workflow.
- Diagnosed a real pass-1 failure in targeted `NCU`: rank2 hit `Unexpected number of profiled kernels` under application replay.
- Tightened `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` to use:
  - per-representative-rank required kernel regex (`rank0/1` vs `rank2/3`),
  - `prepare_case_kernel_metrics.py` for case-local filtering / aliasing / missing-by-rank reporting,
  - `--app-replay-mode relaxed` for targeted `NCU` replay.
- Added an explicit representative-rank validation step so the lightweight flow now fails fast if `rank0 != rank1` or `rank2 != rank3` in required kernel sets.
- Re-ran the most relevant fast regressions after the shell-flow rewrite:
  - `cd megatron-sim-engine && pytest -q tests/unit/test_prepare_case_kernel_metrics.py tests/unit/test_build_trace_shaped_pp_schedule.py tests/unit/test_build_ddp_slowdown_assets.py tests/unit/test_slowdown_predictor.py tests/unit/test_simu_engine_ddp_slowdown.py tests/integration/test_simu_engine_ddp_slowdown_integration.py`
  - `pytest -q tests/unit/test_compare_ddp_slowdown_reference.py`
  - observed: `21 passed` + `1 passed`.
- Started replacing the hand-written `manual_schedule_ppdeps` heredoc with an auto-generated trace-shaped PP schedule builder.
- Added a new schedule utility target path: `megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py`.
- Added a shared trace parser utility under `megatron-sim-engine/tools/data_prep/common/megatron_trace_utils.py` for data-prep tools.
- Added a new builder helper to enumerate required backward kernel short names directly from `trace + nsys` before NCU collection.
- Began upgrading `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` into a self-contained workflow with:
  - auto schedule generation,
  - targeted NCU collection on GPU5/GPU6,
  - a second-pass missing-kernel retry,
  - real 4-GPU reference `nsys` collection,
  - slowdown off/on vs hardware error-table generation.

## 2026-03-13 Follow-up
- Re-ran focused regression after the latest slowdown fixes:
  - `pytest -q tests/unit/test_compare_ddp_slowdown_reference.py tests/unit_tests/distributed/test_ddp_bucketing_scaling_mode.py` -> `5 passed`
  - `cd megatron-sim-engine && pytest -q tests/unit/test_slowdown_predictor.py tests/unit/test_build_ddp_slowdown_assets.py tests/integration/test_simu_engine_ddp_slowdown_integration.py` -> `12 passed`
- Recomputed the existing GPT-6.7B lightweight artifact with the new compare logic and new scaler path:
  - `wrank0` now aligns with hardware on `65` shared DDP alignment keys.
  - `wrank0` reference compare now reports launch / finish MAE and no longer shows `shared_comm_uids_count = 0`.
  - `wrank2` still fail-fast in hardware compare with `ValueError: Reference trace does not share any DDP comm alignment keys with the simulator target backward window.`
- Collected hard evidence that the old lightweight artifact still has a structural stage-1 mismatch on stage-1 ranks:
  - simulator `wrank2` target backward contains `66` DDP comm alignment keys;
  - hardware reference trace only exposes `1` matching DDP bucket for the selected stage-1 backward window.
- Confirmed the likely root cause is historical scaling-mode bucketing divergence before the fake-`pp_rank` fix in `megatron/core/distributed/distributed_data_parallel.py`.
- Updated `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` in three steps while continuing workflow validation:
  - serialized pass-1 / pass-2 targeted `ncu` collection to avoid parallel application-replay hangs;
  - replaced bulk regex `ncu` collection with per-kernel-short-name collection using rank-local kernel lists;
  - added `--kill yes` together with `-c 1` so each `ncu` run stops after the first matched launch.
- Verified locally with `bash -x` that the new workflow still reaches the dry-run and traced-scaling stages correctly and that per-kernel `ncu` logging advances kernel-by-kernel instead of stalling on one huge regex replay.
- Continued end-to-end lightweight rerun attempts on GPUs `4,5,6,7`; the workflow is still being tuned for wall-clock practicality, but the slowdown runtime path itself remains stable and already validated on the old artifact plus unit/integration coverage.


## 2026-03-13 Late Update
- Fixed the lightweight E2E `141` blocker in `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` by replacing `latest_rank_file()`'s `find | sort | head | cut` pipeline with a Python-based latest-file selector that is safe under `set -euo pipefail`.
- Added regression test `tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py` to execute the real `latest_rank_file()` function body under `set -euo pipefail` and verify it returns the newest matching rank trace path.
- Re-ran focused regression coverage:
  - `pytest -q tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py tests/unit/test_compare_ddp_slowdown_reference.py tests/unit_tests/distributed/test_ddp_bucketing_scaling_mode.py`
  - result: `6 passed`.
- Started a fresh GPU `4-7` acceptance run with the updated self-contained workflow:
  - artifact root: `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`
  - scaling `nsys` trace export already completed successfully.
  - auto trace-shaped `pp2` schedule generation already completed successfully.
  - targeted `ncu` pass-1 is running serially.
- Early correctness signal from the fresh run:
  - `dryrun_rank0` bucket count = `65`
  - `dryrun_rank2` bucket count = `1`
  - this matches the expected stage-1 asymmetry much better than the historical mismatched artifact and is the right direction for the `wrank2` bucket-alignment recheck.

- Milestone update on the fresh acceptance run (`tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`):
  - pass-1 representative `rank0` targeted `ncu` collection finished and produced `ncu_pass1_stage0_rank0/output/kernel_metric_output.csv`.
  - pass-1 representative `rank2` targeted `ncu` collection has started on GPU `6` and has already progressed beyond kernel indices `0` and `1`.
  - this confirms the self-contained workflow is no longer blocked at stage-0 collection and is actively exercising the stage-1 kernel family needed for `wrank2` hardware alignment re-validation.

- Completed the refreshed GPT-6.7B lightweight self-contained acceptance run on 2026-03-14 through Step 8 using artifact `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`.
- Completed targeted `ncu` collection for representative stage-1 rank by resuming the missing `rank2` kernel subset and merging pass-1 metrics successfully; no pass-2 collection was required.
- Completed Step 6 slowdown compare with the refreshed artifact:
  - `wrank0`: `backward_off=42.59 ms`, `backward_on=63.191119 ms`, `ddp_comm_count=65`, `alignment_keys=65`
  - `wrank2`: `backward_off=40.17 ms`, `backward_on=46.068028 ms`, `ddp_comm_count=1`, `alignment_keys=1`
- Completed Step 7 real 4-GPU hardware reference on GPUs `4-7` and exported sqlite successfully.
- Completed Step 8 hardware compare and generated:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.md`
- Key acceptance evidence from the refreshed artifact:
  - `wrank0`: `shared_comm_uids_count=65`, `launch_MAE_off=5.200614 ms`, `launch_MAE_on=1.001078 ms`, `finalize_err_off=6.71 ms`, `finalize_err_on=1.49 ms`
  - `wrank2`: `shared_comm_uids_count=1`, `launch_MAE_off=23.250001 ms`, `launch_MAE_on=23.240001 ms`, `finalize_err_off=7.77 ms`, `finalize_err_on=13.66 ms`
  - `wrank0`: `hardware_backward=138.81 ms`, `off_abs_err=96.22 ms`, `on_abs_err=75.618881 ms`
  - `wrank2`: `hardware_backward=54.51 ms`, `off_abs_err=14.34 ms`, `on_abs_err=8.441972 ms`
- Generated acceptance summary files:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/summary.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/summary.md`
