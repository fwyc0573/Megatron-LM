## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-09 | Added tracing-side validation report for explicit scaling finalize-base emission |

# Test Report: Scaling Finalize-Base Trace Emission

**Date**: 2026-03-09  
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)

## Test Script Information
- Scripts:
  - `tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py`
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py`
  - `tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py`
  - `tests/integration/test_ddp_overlap_trace_smoke.sh`
- Commands:
  ```bash
  pytest -q tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py \
    tests/unit_tests/distributed/test_ddp_overlap_trace.py \
    tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py \
    tests/unit_tests/test_trace_ddp_grad_overlap_args.py \
    tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py

  TRACE_SMOKE_DIST_GPUS=2,3 TRACE_SMOKE_SCALE_GPU=2 \
    bash tests/integration/test_ddp_overlap_trace_smoke.sh

  python -m py_compile megatron/profiler/cmd.py \
    megatron/training/training.py \
    tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py \
    tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py
  ```

## Validation Criteria
- `CMD` top-level trace line serializes `finalize_base_duration_ms` without breaking trailing `sub_operations=` layout.
- Scaling helper fail-fast rejects missing `cmd.stage_operations_trace_dict` instead of leaking `AttributeError`.
- Scaling helper fail-fast also rejects semantic misuse (`name_cmd != dp_allreduce` / `op_semantics != metadata_placeholder`).
- Scaling helper emits `dp_allreduce(metadata_placeholder)` with non-zero `duration`, non-zero `timestamp`, and explicit `finalize_base_duration_ms`.
- Existing DDP overlap tracing tests remain green.
- Fresh scaling smoke trace contains top-level `dp_allreduce(... finalize_base_duration_ms=...)`.

## Test Results

| Test Suite | Result | Details |
|------------|--------|---------|
| Main repo unit tests | PASS | `15 passed` |
| Tracing smoke test | PASS | Distributed + scaling smoke both passed |
| Syntax check | PASS | `py_compile` exit code `0` |

## Evidence
- Unit test output:
  - `15 passed, 3 warnings in 8.46s`
- Smoke output:
  - `[PASS] DDP overlap trace smoke test passed.`
  - Distributed trace: `realistic_trace/pp1_tp1_exp1_expnNone_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNumNone_l2_bs1_rank1_20260309115612.txt`
  - Scaling trace: `profiler_log/pp1_tp1_ep1_expn1_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNum1_numl2_bs1_rank0_20260309115627.txt`
- Fresh scaling trace excerpt:
  - `profiler_log/pp1_tp1_ep1_expn1_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNum1_numl2_bs1_rank0_20260309115627.txt:6`
  - Contains `op_semantics=metadata_placeholder` and non-zero `finalize_base_duration_ms` on top-level `dp_allreduce`; the smoke script now asserts this field directly.

## Notes
- Existing traces collected before this change still lack `finalize_base_duration_ms`; those historical traces continue to rely on replay-side fallback inference.
