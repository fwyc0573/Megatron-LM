## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-08 | Created test report for DDP grad overlap tracing MVP |

# Test Report: DDP Grad Overlap Tracing MVP

**Date**: 2026-03-08  
**Environment**: `conda activate myenv_yc` (Python 3.9), `PYTHONPATH=$PWD`, GPUs: 8x NVIDIA A800-SXM4-80GB

## Test Script Information
- Scripts:
  - `tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py`
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py`
  - `tests/unit_tests/profiler/test_cmd_subop_sync_mode.py`
  - `tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py`
  - `tests/unit_tests/profiler/test_interception_comm_scaling_mode.py`
  - `tests/integration/test_ddp_overlap_trace_smoke.sh`
- Commands:
  ```bash
  PYTHONPATH=$PWD pytest -q \
    tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py \
    tests/unit_tests/distributed/test_ddp_overlap_trace.py \
    tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py \
    tests/unit_tests/test_trace_ddp_grad_overlap_args.py \
    tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
    tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
    tests/unit_tests/profiler/test_interception_comm_scaling_mode.py

  python -m py_compile \
    megatron/profiler/cmd.py \
    megatron/core/distributed/param_and_grad_buffer.py \
    megatron/core/distributed/distributed_data_parallel.py \
    megatron/core/distributed/finalize_model_grads.py \
    megatron/training/arguments.py \
    megatron/training/training.py

  bash -n tests/integration/test_ddp_overlap_trace_smoke.sh
  tests/integration/test_ddp_overlap_trace_smoke.sh
  ```

## Validation Criteria
- Top-level CMD lines serialize `cmd_uid` and optional `op_semantics` without breaking legacy `sub_operations` formatting.
- Each distributed-mode DDP bucket communication emits exactly one `ddp_grad_comm` record with launch, completion, and wait timestamps.
- Scaling mode emits metadata-only `ddp_grad_comm` records and does not execute real DP collectives.
- Warmup/non-profiled scaling iterations do not leak `ddp_grad_comm` records into the final trace payload.
- Existing profiler sub-op tracing behavior remains intact.
- Integration smoke produces real trace files and validates key distributed/scaling overlap fields directly from those files.

## Test Results

| Test Suite | Result | Details |
|------------|--------|---------|
| Targeted + regression unit tests | PASS | 22/22 passed |
| Python syntax check | PASS | 6 modified Python files compiled successfully |
| Integration smoke | PASS | Distributed + scaling real runs completed and trace assertions passed |

## Evidence
- Unit/regression command exit code: `0`
- Unit/regression summary:
  ```text
  ......................                                                   [100%]
  22 passed, 3 warnings in 8.63s
  ```
- Integration smoke command exit code: `0`
- Integration smoke summary:
  ```text
  [PASS] DDP overlap trace smoke test passed.
  [INFO] Artifacts: tests/integration/artifacts/ddp_overlap_trace_smoke_20260308_091608_2601666
  [INFO] Distributed trace: realistic_trace/pp1_tp1_exp1_expnNone_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNumNone_l2_bs1_rank1_20260308091621.txt
  [INFO] Scaling trace: profiler_log/pp1_tp1_ep1_expn1_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNum1_numl2_bs1_rank0_20260308091636.txt
  ```
- Verified distributed trace contains `ddp_grad_comm(...)`, `completion_observed_timestamp_ms=<number>`, `wait_cmd_uid=cmd-*`, and `op_semantics=wait_flush_only`.
- Verified scaling trace contains `ddp_grad_comm(...)`, `metadata_only=True`, `status=launch_only`, `trigger_op=backward_step`, and `op_semantics=metadata_placeholder`, with no leaked `trigger_op=loss_func` events.

## Failures Encountered and Resolutions
- `test_cmd_subop_sync_mode.py` initially failed because the updated decorator assumed every active CMD had `phase_range()`; fixed by making the decorator capability-aware and backward-compatible.
- Scaling warmup initially leaked metadata-only events under `loss_func`; fixed by aligning bucket-event gating with CMD trace activation and by keeping scaling-mode DP collectives metadata-only even during warmup.
