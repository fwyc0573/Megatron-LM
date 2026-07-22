# Test Report: Task2 Source Compatibility

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Initial RED→GREEN, real-artifact, baseline-failure, and independent-review evidence |

**Date:** 2026-07-23  
**Environment:** controller shell, Python 3, no GPU commands; temporary root `/data/ycfeng/tmp`

## 1. Test Script Information

- Scripts:
  - `/data/ycfeng/sc26_ae_task3_qwen/tests/unit/test_sc26_ae_task3_source_compatibility.sh`
  - `/data/ycfeng/sc26_ae_task3_qwen/tests/unit/test_sc26_ae_task3_contracts.sh`
  - `/data/ycfeng/sc26_ae_task3_qwen/tests/integration/test_sc26_ae_task3_contract.sh`
- Commands:
  ```bash
  bash tests/unit/test_sc26_ae_task3_source_compatibility.sh
  SC26_AE_TMP_ROOT=/data/ycfeng/tmp/sc26_task2_compat_20260723/unit \
    bash tests/unit/test_sc26_ae_task3_contracts.sh
  SC26_AE_TMP_ROOT=/data/ycfeng/tmp/sc26_task2_compat_20260723/integration \
    bash tests/integration/test_sc26_ae_task3_contract.sh
  python3 -B SC26-AE/tools/artifact_manifest.py verify \
    --root /data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/_shared/task2/runs/task2-20260722T142810Z-192-11368 \
    --manifest /data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/_shared/task2/runs/task2-20260722T142810Z-192-11368/artifact_manifest.json
  ```

## 2. Validation Criteria

- Task1 remains strict after dense tracing source changes.
- Verified prior Task2 is accepted only when outer ancestry, exact Echo identity, recorded gitlink bindings, and seven Task2 source blobs agree.
- Wrong Echo, non-ancestor outer, Task2 source drift, and inconsistent recorded simulator fail.
- Real Task2 marker, manifest, 18 listed files, model, scaler, dataset, and checksums remain valid.
- No Task2 collection/training command runs.

## 3. Test Results and Evidence

| Test | Result | Actual |
|------|--------|--------|
| TDD RED | PASS | exit `127`; missing `task3_task2_source_compatibility_mode` |
| Focused source compatibility | PASS | `14/14` |
| Task3 unit contracts | PASS | `9/9` |
| Real Task2 manifest | PASS | `18` files; SHA256 `d344fbfc0f4e56286efe9dd5ee6ac3f125ed3ad34fe8e4599bc9a71f67dda76e` |
| Real Task2 compatibility | PASS | `task2_producer_equivalent_reuse` |
| Independent StepCode review | PASS | `APPROVE` |
| Broader Task3 synthetic integration | BASELINE FAIL | observed Qwen traces `1`; required `32` |

### Key Metrics

| Metric | Expected | Actual | Delta / Status |
|--------|----------|--------|----------------|
| Dataset rows | `> 0` | `727` | PASS |
| Validation MSE | finite, nonnegative | `0.04124828706619175` | PASS |
| Test MSE | finite, nonnegative | `0.061428837844613504` | PASS |
| Task2 elapsed seconds | `> 0` | `1041.532698287` | PASS |
| Model bytes | `> 0` | `412174` | PASS |
| Scaler bytes | `> 0` | `616` | PASS |
| GPU IDs | exactly two | `0,1` | PASS |
| Task2 commands executed in this repair | `0` | `0` | PASS |

The integration failure is reproduced unchanged from a clean `9baafdf` clone. Its one-trace Qwen
fixture predates the current 32-rank product contract; it is not caused by this change and is
deferred under the user's instruction to avoid spending substantial time on test harnesses.
