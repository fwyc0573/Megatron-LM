## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-21 | Added focused functional prebaked package contract tests and recorded results. |

## Test Report: Functional Prebaked Package Contracts

**Date**: 2026-07-21  
**Environment**: system Python 3.12.3 (`/usr/bin/python3`), pytest 9.1.1

### Test Script Information

- Script: `tests/unit/test_sc26_ae_package_prebaked.py`
- Targeted command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
    tests/unit/test_sc26_ae_package_prebaked.py -k 'functional' -vv
  ```

- Full regression command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
    tests/unit/test_sc26_ae_package_prebaked.py
  ```

### Validation Criteria

- Functional builder emits exactly three bundles: `gpt175b`, `dsv3`, and `shared_task2`.
- Functional distribution uses schema `sc26-ae-functional-distribution-manifest-v1` and evidence class `functional_prebaked_not_release_qualified`.
- GPT Task1 capture inventory is exactly `[0,128,256,384,512,640,768,896]`.
- DSV3 Task1 capture inventory is exactly `0..255`.
- Each model bundle contains its own `ncu/kernel_metric_output.csv` and rank-0 provenance (`rank_scope=global_rank_0`, `rank_ids=[0]`, one physical GPU, zero missing kernels).
- Invalid rank inventories and missing model-local NCU CSVs fail closed.
- Legacy strict package tests remain passing.

### Test Results

| Test Suite | Result | Details |
|------------|--------|---------|
| Functional-focused tests | PASS | 6/6 passed; 24 legacy tests deselected |
| Full package unit suite | PASS | 30/30 passed |

### Key Metrics

| Metric | Expected | Observed | Delta |
|--------|----------|----------|-------|
| Functional bundle count | 3 | 3 | 0 |
| GPT selected rank count | 8 | 8 | 0 |
| DSV3 selected rank count | 256 | 256 | 0 |
| Functional-focused pass count | 6 | 6 | 0 |
| Full package pass count | 30 | 30 | 0 |

### Evidence

- Targeted pytest exit code: `0`; output: `6 passed, 24 deselected in 3.37s`.
- Full pytest exit code: `0`; output: `30 passed in 7.01s`.
- Tests exercise strict-verifier rejection of functional schema, model-specific rank-vector rejection, and model-local NCU requirement (preventing shared Task2 fallback).
