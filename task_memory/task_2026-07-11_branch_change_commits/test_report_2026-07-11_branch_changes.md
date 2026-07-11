## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-11 | Added fresh verification evidence for branch change commits |

# Test Report: Branch Change Commit Preparation

**Date**: 2026-07-11
**Environment**: `myenv_yc` (`Python 3.9.18`, PyTorch 2.1.2)
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

## Test Script Information

- Main focused suites: argument parsing, overlap trace args, CMD NVTX labels, scaling DDP bucketing, example scripts, Echo merge, slowdown comparison, and e2e helpers.
- Sim-engine suites: `megatron-sim-engine/tests/unit` and `tests/integration/test_simu_engine_ddp_slowdown_integration.py`.
- Static checks: `bash -n` for changed shell scripts and `py_compile` for changed Python entrypoints.
- Fresh workflow: `tests/e2e/test_ddp_slowdown_simulate_smoke.sh`, followed by case-specific NCU coverage completion and replay using the same fresh trace/SQLite.

## Commands

```bash
export PYTHONPATH="$PWD/megatron-sim-engine:$PWD:$PYTHONPATH"
/opt/anaconda/envs/myenv_yc/bin/python -m pytest -q \
  tests/unit_tests/test_strict_bool_argument_parsing.py \
  tests/unit_tests/test_trace_ddp_grad_overlap_args.py \
  tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
  tests/unit_tests/distributed/test_ddp_bucketing_scaling_mode.py \
  tests/unit/test_qwen3_a3b_moe_overlap_script.py \
  tests/unit/test_echo_slowdown_merge.py \
  tests/unit/test_compare_ddp_slowdown_reference.py \
  tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py

cd megatron-sim-engine
/opt/anaconda/envs/myenv_yc/bin/python -m pytest -q \
  tests/unit tests/integration/test_simu_engine_ddp_slowdown_integration.py
```

## Validation Criteria

- All focused unit and integration tests pass.
- Changed Python and shell entrypoints pass syntax validation.
- Fresh trace collection emits four fake-rank traces and a usable Nsight SQLite database.
- Every compute kernel referenced by slowdown blueprints has NCU features.
- Slowdown-enabled replay processes the target backward CMD and delays its DDP communication schedule.

## Test Results

| Suite | Result | Evidence |
|-------|--------|----------|
| Main focused tests | PASS | `38 passed, 3 warnings in 9.45s` |
| Sim-engine unit + integration | PASS | `75 passed in 0.69s` |
| Shell syntax + Python compile | PASS | Exit code 0 |
| Fresh trace and Nsight export | PASS | Four rank traces; SQLite export completed |
| Initial smoke asset build | FAIL, resolved prerequisite | Historical metrics missed current kernels; fail-fast error identified exact names |
| Targeted NCU collection | PASS | Four BF16 GEMM reports generated; NCU exit code 0 |
| Kernel feature coverage | PASS | 24 required, 24 available, 0 missing across ranks 0-3 |
| Asset build + slowdown replay | PASS | Exit code 0; target `cmd-c8f6b9482185` processed and DDP launch delayed |

## Evidence

- The three warnings are existing Transformer Engine / `pkg_resources` deprecation warnings.
- The initial e2e failure was caused by a stale historical NCU fixture, not a source exception. No fallback was added and the asset builder's completeness check remained intact.
- Fresh replay result: `backward_duration_ms_off=27.27`, `backward_duration_ms_on=27.270069`, one shared and delayed DDP comm alignment key, and the target backward `cmd_uid` in `processed_backward_cmd_uids`.
- Generated trace, SQLite, NCU, metrics, and replay artifacts were kept outside the commit allowlists.
