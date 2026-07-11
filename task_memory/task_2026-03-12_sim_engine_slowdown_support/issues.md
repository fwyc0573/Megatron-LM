## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-15 | Added follow-up review findings for overlap/slowdown control semantics and example coverage |
| 2026-03-15 | Marked Phase 8c/8d workflow closure and downgraded fresh NCU wall-clock to non-blocking limitation |
| 2026-03-14 | Formalized v1 acceptance to focus on wrank0 multi-bucket behavior and diagnostic-only wrank2 launch MAE |
| 2026-03-14 | Added root-cause findings for global-sync backward semantics and stage-1 launch-marker drift |
| 2026-03-12 | Added open items for self-contained NCU workflow and hardware reference parity |
| 2026-03-12 | Added GPT-6.7B lightweight E2E schedule and metrics dependency notes |
| 2026-03-12 | Created issue tracker for slowdown support task |
| 2026-03-12 | Updated blockers after implementation and regression runs |
| 2026-03-12 | Refreshed open/resolved issues after Echo workflow validation |
| 2026-03-12 | Added E2E smoke findings and follow-up gaps |

# Issues

## 2026-03-15 Follow-up Review

### Resolved Today

- **Megatron did not auto-enable DDP overlap tracing metadata from the runtime defaults alone**
  - Root cause: `trace_ddp_grad_overlap` still depended on explicit user input instead of following `overlap_grad_reduce + do_trace`.
  - Resolution: `megatron/training/arguments.py` now auto-enables `trace_ddp_grad_overlap` whenever DDP overlap and tracing are active.

- **`megatron-sim-engine` lacked explicit overlap policy control**
  - Root cause: slowdown enablement existed, but there was no user-facing way to declare whether overlap metadata should be auto-consumed, required, or rejected.
  - Resolution: added `--overlap-mode {auto,on,off}` plus `OverlapConfig` plumbing through the sim-engine CLI/config/runtime path.

- **`megatron-sim-engine` no-overlap slowdown path used to hard-fail instead of warning and continuing**
  - Root cause: slowdown initialization validated assets and overlap-trigger blueprints before finalizing whether the loaded trace actually contained overlap metadata.
  - Resolution: overlap policy is now applied first; if no overlap metadata is present, slowdown logs a warning and remains disabled without loading slowdown assets.

- **Example coverage for overlap / slowdown was incomplete**
  - Root cause: the DeepSeek-V3 example script did not expose a simple env-driven DDP overlap toggle, and `megatron-sim-engine/examples/` had no explicit overlap/slowdown walkthrough.
  - Resolution:
    - added `OVERLAP_GRAD_REDUCE` support to `examples/pretrain_deepseek_v3_moe.sh`,
    - added `examples/pretrain_deepseek_v3_moe_ddp_overlap_trace.sh`,
    - added `megatron-sim-engine/examples/06_ddp_overlap_slowdown_modes.sh`.

- **`--do-trace` used Python's raw `bool()` conversion and misparsed `False` as `True`**
  - Root cause: `argparse` was configured with `type=bool`, which treats any non-empty string as truthy.
  - Resolution: replaced the flag parser with a strict boolean converter and added regression coverage for `False`, `0`, and invalid tokens.

- **Qwen MoE example overlap entry drifted from the DeepSeek convention**
  - Root cause: `examples/pretrain_qwen3_30b_a3b_moe.sh` still used `ENABLE_DDP_OVERLAP` while the newer examples standardized on `OVERLAP_GRAD_REDUCE`.
  - Resolution: switched Qwen to `OVERLAP_GRAD_REDUCE` as the primary env entry, kept the old variable as a compatibility fallback, and added a stubbed script regression for both distributed and scaling modes.

## Open
- The refreshed canonical artifact has restored cross-source DDP alignment (`shared_comm_uids_count = 65` for `wrank0`, `1` for `wrank2`), so the main remaining accuracy gap is no longer `comm_uid` matching.
- `wrank0` hardware cooldown backward is measured under global CMD synchronization and therefore already includes overlap NCCL kernels inside the top-level `backward_step` wall time.
  - Evidence: `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/hardware_backward_kernel_breakdown_20260314.json` shows `compute_pure_primary_union≈54.29-56.61 ms` and `comm_kernel_union≈129.57-129.90 ms`, while the scaling baseline in `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/scaling_backward_kernel_breakdown_20260314.json` has `comm_kernel_union=0.0 ms`.
  - Impact: the current simulator slowdown path stretches compute kernels but does not redefine `backward_step.duration` to follow global-sync comm completion, so top-level backward wall accuracy against hardware remains structurally biased low on `wrank0`.
- `wrank2` still shows a stage-1 launch-marker semantic gap even after alignment recovery, but this is now explicitly classified as diagnostic-only for v1 acceptance.
  - Evidence: the aligned bucket launches at `32.06 ms` after backward start in the scaling trace but at `55.31 ms` in the hardware trace; the hardware launch is ~`0.80 ms` after backward finish, while the scaling marker is ~`8.11 ms` before backward finish.
  - Impact: `launch_MAE` stays around `23.24 ms` even though the shared alignment key is now correct; this row should remain in the report as a limitation note, not as a v1 pass/fail gate.
- The refreshed artifact already passes a persisted scaler path into the simulator (`Echo-slowdown/training_testing/output/standard_scaler.json`), so the remaining underestimation is not explained by a missing `StandardScaler` in this acceptance run.
- Self-contained targeted `NCU` collection remains expensive on this machine, but this is now classified as a non-blocking practicality limitation rather than an incompleteness blocker because the functional workflow is already closed. Future acceptance refreshes should continue to serialize `ncu` collection and prefer canonical artifact reuse unless new kernels are introduced.

## Resolved
- Phase 8c (auto trace-shaped PP schedule generation) is functionally closed by `megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py` plus unit and CLI-level regression coverage.
- Phase 8d (self-contained targeted `NCU` workflow) is functionally closed by `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`, `prepare_case_kernel_metrics.py`, serialized per-kernel replay, second-pass missing-kernel retry, and the completed canonical artifact flow.
- The full `pp2` schedule generated by `megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py` did not replay the compressed GPT-6.7B single-batch trace reliably; a manual trace-shaped PP dependency schedule resolved the mismatch and allowed stable slowdown off/on comparison.
- `PP=1 no-pipelining` replay used to bypass slowdown entirely because `_replay_profile_no_pipelining()` never called the slowdown hook path.
  - Root cause: slowdown hooks were only wired in `_add_operation_to_timeline()`, but `PP=1` simulate takes the no-pipelining fast path.
  - Resolution: no-pipelining replay now routes `backward_step`, `ddp_grad_comm`, and finalize wait through the slowdown-aware helpers when slowdown is enabled.
- Real trace parsing in `build_ddp_slowdown_assets.py` used to fail on `dp_allreduce.description=model_chunk.finish_grad_sync(), All-reduce ...` because the parser split every top-level comma blindly.
  - Resolution: ported the validated top-level field boundary logic from sim-engine's Megatron trace parser and added a regression test.
- Real NCU feature matching used to fail because builder preferred full demangled kernel names while `Echo-slowdown` joins features by `kShortName` / `Kernel Name`.
  - Resolution: builder now prefers `short_name` and has regression coverage for that behavior.
- Slowdown-on replay used to shrink `backward_step` on real traces when kernel blueprints covered only part of the top-level wall time.
  - Root cause: micro-scheduler replaced `backward_step.duration` with predicted kernel sum and dropped the uncovered residual wall time.
  - Resolution: micro-scheduler now preserves the residual baseline duration outside kernel coverage.
- Added `cmd_uid` to kernel-ground-truth NVTX labels so offline kernel blueprints can bind stably to simulator top-level `backward_step` ops.
- Added fail-fast simulator CLI/config plumbing for slowdown-enabled runs.
- Replaced fixed wall-clock DDP bucket launch replay with slowdown-aware precomputed launch/finish timestamps when slowdown is enabled.
- Added synthetic unit/integration coverage for slowdown solver, adapter, loader, asset builder, and finalize wait behavior.
- Restored the `collective-sim` submodule and the default H800 measured P2P profile assets needed by `megatron-sim-engine/tests/unit`.
- Fixed Echo shell compatibility issues caused by invoking Bash scripts through `sh`.
- Fixed Echo `nsys` compatibility issues on Nsight Systems `2023.1.2.43`.
- Fixed Echo merge failure on newly collected slowdown data by switching to normalized-name + occurrence-index matching.

## 2026-03-13 Follow-up Issues

### Resolved Today

- **Historical `wrank2` stage-1 bucket mismatch is resolved on the refreshed lightweight artifact**
  - Evidence: the refreshed simulator target backward now exposes `ddp_comm_count = 1` and `alignment_keys = 1` for `wrank2`, and the hardware compare reports `shared_comm_uids_count = 1` instead of failing fast with zero shared keys.
- **Canonical acceptance trio is now frozen**
  - Evidence: the refreshed artifact `gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518` includes a complete simulator trace dir, scaling `nsys` sqlite, targeted merged `ncu` csv, hardware reference `nsys` sqlite, and reference compare outputs.

- **`latest_rank_file()` used to terminate the proving run with exit code `141` under `set -euo pipefail`**
  - Root cause: the `find | sort | head -n 1 | cut` pipeline triggered `SIGPIPE` on upstream commands, which propagated as `141` and aborted the script right after `nsys export`.
  - Resolution: replaced the helper with a Python-based latest-file selector and added regression coverage in `tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py`.
- **Fresh acceptance run is now past the old Step-2/3 breakpoint**
  - Evidence: new artifact `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518` already contains `nsys/all_ranks_scaling.sqlite`, `schedule_builder.log`, required-kernel manifests, and active targeted `ncu` pass-1 state.
  - Next action: let the run complete, then regenerate `wrank0/wrank2` hardware compare and freeze the canonical trio from this artifact.
- **Cross-source DDP compare no longer depends on raw `comm_uid` identity**
  - Status: resolved for `wrank0` via stable alignment key + backward-window selection.
- **Parallel targeted `ncu` application replay can hang or stall**
  - Evidence: with representative rank0/rank2 launched concurrently, rank0 kept running while rank2 sat in `futex_wait_queue` / `ep_poll` with no log growth.
  - Resolution: workflow changed to serial targeted `ncu` execution.

### Open
- **Historical lightweight artifact still has real stage-1 bucket mismatch**
  - Evidence: `wrank2` simulator backward has `66` DDP keys, but the hardware reference backward candidates only expose `1` DDP key.
  - Impact: hardware launch/finish MAE and finalize-wait calibration cannot be computed for `wrank2` on the old artifact.
  - Next action: regenerate the lightweight traces using the fixed scaling-mode DDP bucketing logic and repeat hardware compare.
- **Self-contained targeted `ncu` collection remains the runtime bottleneck**
  - Latest evidence: the fresh acceptance run has completed representative `rank0` pass-1 and is now running representative `rank2` pass-1; the workflow is progressing correctly, but wall-clock remains high because every kernel short name still requires a full application replay.
  - Evidence: bulk-regex replay was too slow; per-kernel collection improved controllability, but full wall-clock is still being tuned.
  - Current mitigation: serialized `ncu`, per-kernel short-name lists, `-c 1`, and `--kill yes`.
  - Next action: continue the new detached / polled E2E run and record the first fully completed artifact once produced.
- **Detached long-running shell sessions are less reliable than direct log polling in this CLI harness**
  - Evidence: long `exec_command` sessions sometimes surfaced `141` even when intermediate artifacts were already valid.
  - Handling: use file-backed logs as the source of truth, and treat generated artifacts + explicit reruns as authoritative evidence.


- **Slowdown injection path is functionally validated, but accuracy remains mixed on the refreshed acceptance artifact**
  - Evidence:
    - `wrank0`: backward abs error improves from `96.22 ms` to `75.618881 ms`, launch MAE improves from `5.200614 ms` to `1.001078 ms`, finalize wait error improves from `6.71 ms` to `1.49 ms`.
    - `wrank2`: backward abs error improves from `14.34 ms` to `8.441972 ms`, but launch MAE remains ~`23.24 ms`, and finalize wait error worsens from `7.77 ms` to `13.66 ms`.
  - Impact: slowdown has clearly entered the simulator and affects overlap timing, but the current predictor + calibration stack is not yet acceptance-grade for paper-quality accuracy on all ranks.
  - Next action: analyze the refreshed artifact against kernel-level baseline timing and scaler/calibration assumptions before expanding beyond this v1 validation path.
