## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-07-21 | Added functional two-model prebaked Task3 verification evidence. |

## Test Report: Functional Two-Model Prebaked Task3

**Date**: 2026-07-21  
**Environment**: repository worktree, Python 3.11 (`python3`), CPU-only synthetic simulator fixtures

### Test Script Information

- Unit tests: `tests/unit/test_sc26_ae_package_prebaked.py`
- Integration test: `tests/integration/test_sc26_ae_task3_functional_prebaked.sh`
- Commands:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q tests/unit/test_sc26_ae_package_prebaked.py
  tests/integration/test_sc26_ae_task3_functional_prebaked.sh
  ```

### Validation Criteria

- Functional package schema is `sc26-ae-functional-distribution-manifest-v1`.
- Bundle keys are exactly `gpt175b`, `dsv3`, and `shared_task2`.
- Both model bundles contain a non-empty model-local `ncu/kernel_metric_output.csv`.
- Functional Task3 requires explicit `TASK3_ALLOW_FUNCTIONAL_PREBAKED=1` and
  `TASK3_EXECUTION_MODE=synthetic`; real mode is rejected.
- CPU-only Task3 writes report, manifest, marker, and rank-0 NCU provenance for both models.
- Functional evidence remains `functional_prebaked_not_release_qualified`.

### Test Results

| Test Suite | Result | Evidence |
| --- | --- | --- |
| Package unit tests | PASS | 33/33 passed in 11.05 s |
| Functional Task3 integration | PASS | 5 contract checks; GPT + DSV3 model runs passed |

### Key Metrics

| Metric | GPT-175B | DeepSeek-V3 |
| --- | ---: | ---: |
| Functional model bundle count | 1 | 1 |
| Task1 trace rank count | 8 representative ranks | 256 full fake ranks |
| Model-local NCU feature files | 1 | 1 |
| Task3 simulator wall time (fixture) | 0.500 s | 0.500 s |
| Evidence class | functional_prebaked_not_release_qualified | functional_prebaked_not_release_qualified |

### Evidence

- Unit test exit code: `0`; output: `33 passed in 11.05s`.
- Integration output: `FUNCTIONAL_MODEL_PASS_COUNT=2`, `PASS_COUNT=5`.
- Each Task3 manifest reports `ncu_metrics_source=task1_rank0` and each marker reports
  `slowdown_trace_scope=global_rank_0`, `slowdown_trace_rank_ids=[0]`.
- Negative checks passed for missing explicit opt-in and real execution mode.
