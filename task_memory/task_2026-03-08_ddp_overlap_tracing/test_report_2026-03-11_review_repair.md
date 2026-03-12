## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-11 | Added review-repair regression report for DDP overlap tracing |

# Test Report: DDP Overlap Tracing Review Repair

**Date**: 2026-03-11  
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)

### Test Script Information
- Scripts:
  - `tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py`
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py`
  - `tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py`
  - `tests/integration/test_ddp_overlap_trace_smoke.sh`
- Commands:
  ```bash
  python -m py_compile megatron/training/training.py
  pytest -q tests/unit_tests/test_trace_ddp_grad_overlap_args.py tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py
  pytest -q tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py tests/unit_tests/test_trace_ddp_grad_overlap_args.py tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py tests/unit_tests/distributed/test_ddp_overlap_trace.py tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py
  TRACE_SMOKE_DIST_GPUS=0,1 TRACE_SMOKE_SCALE_GPU=0 bash tests/integration/test_ddp_overlap_trace_smoke.sh
  ```

### Validation Criteria
- `training.py` exposes `_validate_trace_ddp_grad_overlap_runtime_args(...)` and rejects invalid scaling/runtime combinations.
- `training.py` exposes `_emit_scaling_dp_allreduce_placeholder(...)` and the helper inserts a top-level `dp_allreduce(metadata_placeholder)` with explicit `finalize_base_duration_ms`.
- Root overlap unit tests cover schema serialization, distributed overlap lifecycle, scaling metadata-only behavior, and explicit finalize-base insertion.
- Fresh distributed/scaling smoke traces contain `ddp_grad_comm(...)`, correct overlap metadata, and scaling `finalize_base_duration_ms=...`.

### Test Results
| Test Suite | Result | Details |
|------------|--------|---------|
| Focused RED reproduction | PASS | Initial reproduction showed `6 failed` due to missing helper symbols in `training.py` |
| Root syntax check | PASS | `python -m py_compile megatron/training/training.py` |
| Root overlap unit suite | PASS | `15 passed` |
| Fresh distributed+scaling smoke | PASS | Script completed and located fresh trace artifacts |

### Evidence
- Initial failure signature before fix:
  - `AttributeError: module 'megatron.training.training' has no attribute '_validate_trace_ddp_grad_overlap_runtime_args'`
  - `AttributeError: module 'megatron.training.training' has no attribute '_emit_scaling_dp_allreduce_placeholder'`
- Post-fix unit result:
  - `15 passed, 3 warnings in 7.63s`
- Post-fix smoke result:
  - `[PASS] DDP overlap trace smoke test passed.`
  - Fresh distributed trace: `realistic_trace/pp1_tp1_exp1_expnNone_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNumNone_l2_bs1_rank1_20260311171036.txt`
  - Fresh scaling trace: `profiler_log/pp1_tp1_ep1_expn1_dp2_nl2_hs128_sl32/wd2_tp1_pp1_exp1_expNum1_numl2_bs1_rank0_20260311171051.txt`

### Failure Diagnosis and Resolution
- Root cause: the checked-out task state contained the scaling profiling call sites that referenced `_emit_scaling_dp_allreduce_placeholder(...)`, but the helper definition itself was absent from `megatron/training/training.py`; the runtime overlap validator callable was also absent.
- Resolution: restored both helpers with fail-fast validation, explicit finalize-base serialization, and no behavior changes outside the intended overlap path.
