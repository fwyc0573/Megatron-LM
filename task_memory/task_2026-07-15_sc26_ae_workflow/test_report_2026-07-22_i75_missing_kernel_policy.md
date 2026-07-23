# Test Report: I75 Task3 Missing-Kernel Slowdown Policy

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-22 | Recorded TDD RED/GREEN evidence and affected regression for deterministic alias/skip slowdown handling |

**Date:** 2026-07-22  
**Result:** PASS for the local CPU-only Task3 slowdown scheduler policy.  
**Qualification boundary:** This report validates simulator behavior only. It does not qualify a
Fresh Task3 artifact, producer provenance, functional bundle, or release readiness.

## 1. Test Script Information

### Modified implementation and tests

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/src/core/simu_engine.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`

### Environment

| Item | Actual value |
|------|--------------|
| Working directory | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine` |
| Python | `/usr/bin/python`, Python 3.12.3 |
| pytest | 9.1.1 |
| torch | 2.5.1+cu124 |
| CUDA available / device count | `False` / `0` |
| Temporary roots | `/data/ycfeng/tmp/qwen3_missing_kernel_red`, `/data/ycfeng/tmp/qwen3_missing_kernel_green`, `/data/ycfeng/tmp/qwen3_missing_kernel_regression`, `/data/ycfeng/tmp/qwen3_missing_kernel_compile` |

No temporary file, log, or cache was written under `/tmp`.

## 2. Validation Criteria

1. Missing features do not call the predictor, do not raise, and preserve baseline duration.
2. A unique canonical alias calls the predictor with the resolved feature key.
3. Ambiguous canonical aliases do not call the predictor and use baseline duration.
4. Exact feature behavior and existing communication/residual/replay paths remain green.
5. Python compilation and diff hygiene pass.
6. Task2 data collection is not rerun or modified for this coverage policy.

## 3. Commands and Results

### TDD RED (before production edit)

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine
export TMPDIR=/data/ycfeng/tmp/qwen3_missing_kernel_red
mkdir -p "$TMPDIR"
/usr/bin/python -m pytest -q tests/unit/test_simu_engine_ddp_slowdown.py \
  -k 'skips_kernel_without_features or uses_unique_canonical_kernel_alias'
```

Observed result:

```text
2 failed, 11 deselected in 3.70s
exit code=1
ValueError: Missing slowdown kernel features for kernel 'kernel_missing'
ValueError: Missing slowdown kernel features for kernel 'ns::kernel_a'
```

This is the expected pre-fix failure: the old code required an exact feature key.

### Focused GREEN

```bash
export TMPDIR=/data/ycfeng/tmp/qwen3_missing_kernel_green
mkdir -p "$TMPDIR"
/usr/bin/python -m pytest -q tests/unit/test_simu_engine_ddp_slowdown.py \
  -k 'skips_kernel_without_features or uses_unique_canonical_kernel_alias or skips_ambiguous_canonical_alias'
```

Observed result:

```text
3 passed, 11 deselected in 0.91s
exit code=0
```

Numeric behavior evidence:

| Case | Baseline duration | Predictor invocation | Slowdown factor | Feature source |
|------|-------------------:|----------------------|----------------:|----------------|
| Missing feature | `4.0 ms` | no | `0.0` | `missing_skip` |
| Unique alias `ns::kernel_a → kernel_a` | `4.0 ms` | yes, key `kernel_a` | `0.5` | `canonical_alias:kernel_a` |
| Ambiguous aliases | `4.0 ms` | no | `0.0` | `missing_skip` |

### Affected regression and static checks

```bash
export TMPDIR=/data/ycfeng/tmp/qwen3_missing_kernel_regression
mkdir -p "$TMPDIR"
/usr/bin/python -m pytest -q \
  tests/unit/test_build_ddp_slowdown_assets.py \
  tests/unit/test_slowdown_predictor.py \
  tests/unit/test_simulator_config_cpu.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py

export TMPDIR=/data/ycfeng/tmp/qwen3_missing_kernel_compile
mkdir -p "$TMPDIR"
/usr/bin/python -m py_compile src/core/simu_engine.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py
git diff --check
```

Observed result:

```text
27 passed in 7.83s
pytest exit code=0
py_compile exit code=0
git diff --check exit code=0
```

## 4. Root-Cause Resolution

Task3 previously made an exact-key lookup mandatory for every kernel in the NCU slowdown blueprint.
The failure is a feature-coverage mismatch, not evidence that the Task2 predictor dataset must be
recollected. The resolver now follows an auditable sequence: exact key, one canonical alias, or
explicit `missing_skip`. It never selects one of multiple aliases, never invokes an uninitialized
or unrelated predictor row, and leaves communication scheduling and baseline timeline arithmetic
unchanged when slowdown is skipped.

## 5. Scope and Remaining Gates

- Task2 was **not rerun**, and no Task2 dataset, predictor, checksum, or manifest was edited.
- This local policy does not bypass Task1/Task2 producer-commit provenance checks.
- A new valid Fresh Task3 run must still produce `report.json`, `report.md`,
  `artifact_manifest.json`, and `run_marker.json` before functional/prebaked packaging.
- No claim is made here about distributed accuracy or AE/release readiness.
