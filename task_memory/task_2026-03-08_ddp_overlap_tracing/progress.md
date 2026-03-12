## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-08 | Started implementation on branch `overlap-tracing` |
| 2026-03-08 | Completed implementation, regression fixes, and validation |
| 2026-03-09 | Added explicit scaling finalize-base trace emission and replay-consumption validation |

# Progress

## 2026-03-08
- Created branch `overlap-tracing`.
- Created task memory directory and planning docs.
- Extended `CMD` with `cmd_uid`, `op_semantics`, and generic trace-event serialization.
- Added `--trace-ddp-grad-overlap` CLI flag and runtime fail-fast validation.
- Instrumented DDP bucket lifecycle tracing in `param_and_grad_buffer.py` for launch, completion, and wait.
- Added distributed-mode completion tracking through `Work.get_future().then(...)`.
- Added scaling-mode metadata-only DDP overlap path that skips real DP collectives.
- Added stable `buffer_id` / `bucket_id` propagation through DDP buffer setup.
- Marked `dp_allreduce` top-level semantics as `wait_flush_only` or `metadata_placeholder` where applicable.
- Added unit tests for schema, distributed lifecycle, scaling metadata-only behavior, warmup gating, and CLI/runtime validation.
- Added `tests/integration/test_ddp_overlap_trace_smoke.sh` and verified both distributed and scaling smoke runs.

## Regression Fixes
- Restored backward compatibility for decorator-based sub-op tracing when the active CMD test double does not implement `phase_range()`.
- Fixed scaling warmup leakage so non-profiled iterations do not emit `ddp_grad_comm` events.
- Fixed scaling-mode warmup behavior so trace-enabled DDP overlap never falls back to a real collective.

## Current Focus
- Final handoff and delivery.

## 2026-03-09
- Extended `CMD` with top-level `finalize_base_duration_ms` serialization while keeping trailing `sub_operations=` intact.
- Added `_emit_scaling_dp_allreduce_placeholder(...)` in `megatron/training/training.py` to measure and emit the scaling-mode DDP finalize base window explicitly.
- Updated the scaling profiling loop so `dp_allreduce(metadata_placeholder)` is inserted at the original trace position but carries real `duration`, `timestamp`, and `finalize_base_duration_ms` before `optimizer_step` starts.
- Updated `megatron-sim-engine` replay to prefer explicit `finalize_base_duration_ms` over neighbor-gap inference whenever the field is present.
- Added unit coverage for schema emission and scaling placeholder insertion, plus simulator integration coverage for explicit-field precedence.
- Hardened `_emit_scaling_dp_allreduce_placeholder(...)` with an explicit fail-fast check for missing `cmd.stage_operations_trace_dict`.
- Tightened `_emit_scaling_dp_allreduce_placeholder(...)` contract to reject non-`dp_allreduce` commands and non-`metadata_placeholder` semantics.
- Re-ran the 2-rank tracing smoke test on GPUs `2,3` and confirmed fresh scaling traces now contain `finalize_base_duration_ms=...` on top-level `dp_allreduce(metadata_placeholder)`.


## 2026-03-11 Review Repair
- Re-reviewed the task against the current working tree and found that `megatron/training/training.py` still referenced `_emit_scaling_dp_allreduce_placeholder(...)` in the scaling profiling loop, but the helper was not actually defined in the module.
- Found the same working-tree gap for `_validate_trace_ddp_grad_overlap_runtime_args(...)`; the task report claimed runtime fail-fast validation existed, but the callable was missing.
- Reproduced the regression before fixing it:
  - `pytest -q tests/unit_tests/test_trace_ddp_grad_overlap_args.py tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py`
  - Result: `6 failed`, all due to missing helper/runtime-validation symbols.
- Restored both helpers in `megatron/training/training.py` and kept the fix minimal:
  - runtime fail-fast now rejects `--trace-ddp-grad-overlap` with `--scaling-disable-ddp-wrap` in scaling mode;
  - scaling finalize placeholder emission now validates command semantics, records explicit `finalize_base_duration_ms`, and inserts the record at the intended top-level trace position.
- Re-ran the full tracing regression suite successfully:
  - `python -m py_compile megatron/training/training.py`
  - `pytest -q tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py tests/unit_tests/test_trace_ddp_grad_overlap_args.py tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py tests/unit_tests/distributed/test_ddp_overlap_trace.py tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py`
  - Result: `15 passed`
- Re-ran the real two-mode smoke validation on fresh GPUs and confirmed fresh traces now contain the expected overlap fields again:
  - `TRACE_SMOKE_DIST_GPUS=0,1 TRACE_SMOKE_SCALE_GPU=0 bash tests/integration/test_ddp_overlap_trace_smoke.sh`
  - Result: PASS
  - Distributed trace: `realistic_trace/pp1_tp1_exp1_expnNone_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNumNone_l2_bs1_rank1_20260311171036.txt`
  - Scaling trace: `profiler_log/pp1_tp1_ep1_expn1_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNum1_numl2_bs1_rank0_20260311171051.txt`
