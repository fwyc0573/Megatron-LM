# Test Report: Fake-Level AE README and Workflow Regression

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-21 | Recorded the final post-README fake-level regression and controller runtime boundary |

**Date**: 2026-07-21
**Environment**: CPU controller; Python `3.12.3` (`python3`); no fixed Megatron GPU worker runtime
**Evidence class**: `local_synthetic_not_gpu_qualification`

## Test Script Information

| Script | Exact command |
|---|---|
| Shell syntax | `bash -n examples/update_pretrain_gpt.sh SC26-AE/setup.sh SC26-AE/lib/task1_trace.sh SC26-AE/lib/task2_echo.sh SC26-AE/lib/task3_simulation.sh SC26-AE/task1_gpt175b.sh SC26-AE/task1_dsv3.sh SC26-AE/task2_gpt175b.sh SC26-AE/task2_dsv3.sh SC26-AE/task3_gpt175b.sh SC26-AE/task3_dsv3.sh` |
| Package unit suite | `PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q tests/unit/test_sc26_ae_package_prebaked.py` |
| Task1 contract | `bash tests/integration/test_sc26_ae_task1_contracts.sh` |
| Functional prebaked Task3 | `bash tests/integration/test_sc26_ae_task3_functional_prebaked.sh` |
| Task3 contract | `bash tests/integration/test_sc26_ae_task3_contract.sh` |
| Legacy CPU prebaked regression | `bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh` |

## Validation Criteria

1. Public shell wrappers parse with `bash -n`.
2. Functional package contains exactly `gpt175b`, `dsv3`, and `shared_task2` with model-local
   rank-0 NCU features.
3. GPT Task1 contract retains 8 representative ranks; DeepSeek-V3 retains 256-rank coverage;
   invalid rank vectors fail before workload execution.
4. Functional prebaked Task3 requires explicit `TASK3_ALLOW_FUNCTIONAL_PREBAKED=1`, rejects
   `TASK3_EXECUTION_MODE=real`, and emits report, manifest, marker, and rank-0 NCU provenance.
5. Existing CPU synthetic Task3 regression remains green for all legacy public model entries.
6. No command claims real distributed accuracy or upgrades the functional evidence class.

## Test Results and Evidence

| Test | Result | Numeric evidence |
|---|---|---:|
| Shell syntax | PASS | 11 public/helper/setup paths, exit `0` |
| Package unit suite | PASS | `33/33` tests, `10.75 s`, exit `0` |
| Task1 integration contract | PASS | `PASS_COUNT=45`, exit `0` |
| Functional prebaked Task3 | PASS | `PASS_COUNT=5`; functional model pass count `2`; exit `0` |
| Task3 contract | PASS | `PASS_COUNT=11`, exit `0` |
| Task2 contract | PASS | canonical manifest `13` files; alternate manifest `19` files; symlink/checksum/path-identity negatives passed; exit `0` |
| Legacy CPU prebaked Task3 | PASS | `MODEL_PASS_COUNT=3`, exit `0` |

The final post-README rerun produced these fixture values:

| Model | Rank0 step (ms) | Forward (ms) | Backward (ms) | Optimizer (ms) | Simulator wall (s) | Process wall (s) | Peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-175B | 18.5 | 5.0 | 9.0 | 2.0 | 0.5 | 1.402895 | 51,812 |
| Qwen3-A30B (legacy optional) | 22.5 | 6.0 | 11.0 | 2.5 | 0.5 | 1.430273 | 51,824 |
| DeepSeek-V3 | 24.5 | 6.5 | 12.0 | 3.0 | 0.5 | 1.417653 | 52,164 |

Functional Task3 evidence checks observed:

```text
FUNCTIONAL_MODEL_LOCAL_NCU=verified
EVIDENCE_CLASS=functional_prebaked_not_release_qualified
ncu_metrics_source=task1_rank0
slowdown_trace_scope=global_rank_0
slowdown_trace_rank_ids=[0]
```

Legacy CPU fixture metrics:

| Model | Rank0 step (ms) | Forward (ms) | Backward (ms) | Optimizer (ms) | Simulator wall (s) |
|---|---:|---:|---:|---:|---:|
| GPT-175B | 18.5 | 5.0 | 9.0 | 2.0 | 0.5 |
| Qwen3-A30B (legacy optional) | 22.5 | 6.0 | 11.0 | 2.5 | 0.5 |
| DeepSeek-V3 | 24.5 | 6.5 | 12.0 | 3.0 | 0.5 |

## Known Validation Boundary

Fresh GPT/DeepSeek-V3 Task1, exact-two-GPU Task2, and Fresh Task3 were not run on the
controller because `/opt/conda/envs/megatron_env/bin/python` and the required GPU/Nsight
runtime are unavailable. This is an environment blocker. No synthetic fixture, historical
output, shared Task2 CSV, fallback interpreter, or calibration factor was promoted to fresh
qualification. A clean commit/clean clone replay remains pending.

Controller preflight values recorded during this report:

| Probe | Observed value |
|---|---|
| Megatron Python | missing: `/opt/conda/envs/megatron_env/bin/python` |
| Megatron torchrun | missing: `/opt/conda/envs/megatron_env/bin/torchrun` |
| Echo Python | missing: `/opt/conda/envs/echo_slowdown/bin/python` |
| Nsight Systems | `/usr/local/bin/nsys` present |
| Nsight Compute | `ncu` absent |
| Controller torch | `2.5.1+cu124` |
| CUDA availability | `False` |
| CUDA device count | `0` |
