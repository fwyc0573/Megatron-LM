# Test Report: I55 Task2 Outer Producer Provenance

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Recorded the focused Task2 outer-producer RED→GREEN cycle, affected regressions, failure diagnosis, and independent design/implementation reviews |

**Date:** 2026-07-20  
**Repository:** `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`  
**Baseline HEAD:** `0e0d8c4c57c8c241b46b36390eeec632b6f40492`  
**Evidence class:** `local_synthetic_not_gpu_qualification`

## 1. Test Script Information

### Implementation and tests

- Implementation: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/lib/task2_echo.sh`
- Focused unit test: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_source_provenance.sh`
- Affected integration test: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task2_contract.sh`
- Existing interpreter regression: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_interpreter_contract.sh`

Current source identities:

| Path | SHA256 |
|------|--------|
| `SC26-AE/lib/task2_echo.sh` | `842fdd5998a6a2dea265f2fe8ff9d1bb563bdebd29d3fb91c321f9961850c8a2` |
| `tests/unit/test_sc26_ae_task2_source_provenance.sh` | `aa623edb6a54a08761833a08d34924ed49ab1d11fbd401eb6a20fee65ccfc66c` |
| `tests/integration/test_sc26_ae_task2_contract.sh` | `86aca2bc41a392495bd028f5004580743b5fa97722d2b101390f56271f9a3281` |
| `tests/unit/test_sc26_ae_task2_interpreter_contract.sh` | `e97511fe7e625a15457b51cf2ad1d3c891405c20a3e50fc0c0495336bbe9acb2` |

### Exact reproducible commands

Focused unit RED and GREEN used the same command before and after the implementation changes:

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp \
  bash tests/unit/test_sc26_ae_task2_source_provenance.sh
```

Affected integration regression:

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp \
  bash tests/integration/test_sc26_ae_task2_contract.sh
```

Existing real-interpreter contract regression:

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp \
  bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
```

Focused syntax and diff verification:

```bash
bash -n SC26-AE/lib/task2_echo.sh
bash -n tests/unit/test_sc26_ae_task2_source_provenance.sh
bash -n tests/integration/test_sc26_ae_task2_contract.sh

git diff --check -- \
  SC26-AE/lib/task2_echo.sh \
  tests/unit/test_sc26_ae_task2_source_provenance.sh \
  tests/integration/test_sc26_ae_task2_contract.sh
```

## 2. Environment

| Item | Observed value |
|------|----------------|
| Conda environment | None active |
| Python executable | `/usr/bin/python` |
| Python | `3.12.3` |
| pytest | `9.1.1` |
| Torch | `2.5.1+cu124` |
| CUDA available | `False` |
| CUDA device count | `0` |

The controller has no visible GPU. Therefore all results in this report are local control-plane or
synthetic evidence and cannot qualify H800 execution, real datasets, release datasets, or AE
readiness.

## 3. Validation Criteria

1. One frozen `TASK2_MAIN_COMMIT` must govern Task2 outer metadata and both nested gitlink lookups.
2. The fixed producer surface must contain exactly the three public Task2 wrappers,
   `SC26-AE/lib/common.sh`, `SC26-AE/lib/task2_echo.sh`,
   `SC26-AE/tools/artifact_manifest.py`, and `SC26-AE/tools/echo_metrics.py`.
3. For each of the seven files, the helper must reject symlinks/non-regular paths, missing commit
   paths, non-blob Git objects, and live bytes whose `git hash-object --no-filters` value differs
   from the recorded commit blob.
4. Real mode must check the producer bytes before build/reuse and independently at the first line
   of model-marker and shared-pointer publication. Synthetic mode must retain its existing local
   evidence behavior.
5. The focused unit must pass one clean case, the recorded-commit static invariant, seven producer
   mutation rejections, and two publication rejection cases: total `PASS_COUNT=11`.
6. The affected Task2 integration and existing interpreter contract must remain green without a
   production fallback.
7. The result must not claim an authority-approved interpreter/image digest, canonical worker,
   issuer authentication, H800 qualification, real/release dataset qualification, or AE readiness.

## 4. RED→GREEN Evidence

### RED-0: non-authoritative test-harness failure

| Evidence | Exit | Bytes | SHA256 |
|----------|-----:|------:|--------|
| `logs/session58-i55-task2-outer-provenance-red-20260720.log` | `1` | `276` | `e71c5114f5b381f0cbeef74c53ea812a3fa974808e6da9e2c099fdf9fc600f0e` |

The extracted production library executed `set +e`; the first test draft did not restore
`errexit`, so it continued after the first missing-helper error. The test harness was corrected by
restoring `set -euo pipefail`. This transcript is retained as failure evidence but is not the
authoritative product RED.

### RED-1: missing producer-provenance helper

| Evidence | Exit | Bytes | SHA256 | Key output |
|----------|-----:|------:|--------|------------|
| `logs/session58-i55-task2-outer-provenance-red-v2-20260720.log` | `127` | `164` | `9ec4241faca180340d3bfa521edab0c62a37d206849489a1e6775ff36261ec80` | `task2_assert_outer_source_provenance: command not found` |

### RED-2: live-HEAD commit drift

| Evidence | Exit | Bytes | SHA256 | Key output |
|----------|-----:|------:|--------|------------|
| `logs/session58-i55-task2-recorded-commit-red-20260720.log` | `1` | `102` | `b3c118e75890b27f815a6dd6e638b8f3e3a6927e017bb9c279335bd6c11dd5d5` | `Task2 outer gitlink lookup bypasses the recorded main commit` |

The first implementation patch still resolved two sim-engine gitlinks from live `HEAD:`. The
second RED proved that a single Task2 run could mix two outer commits. Both reads were changed to
`${TASK2_MAIN_COMMIT}:megatron-sim-engine`.

### GREEN

| Evidence | Exit | Bytes | SHA256 | Key output |
|----------|-----:|------:|--------|------------|
| `logs/session58-i55-task2-outer-provenance-green-v2-20260720.log` | `0` | `144` | `1fc1c1945ff2190e64bd2b897e2226ccff0fcd0e42d27ae8d72293f50dee37ef` | `PASS_COUNT=11` |
| `logs/session58-i55-task2-integration-green-v3-20260720.log` | `0` | `1,395` | `635ba2e9649356a8dcff7995f3fcabf3f2e98ade8eb14b3bb9875ae06fea32b5` | `10` explicit `PASS:` lines; alternate manifest `verified`, `19` files |
| `logs/session58-i55-task2-real-contract-green-20260720.log` | `0` | `3,196` | `841e19aec22ec8eb52cfbedc5501e9f20c5b5a0724d5e227055aa96da470da34` | `PASS_COUNT=12`; parser `11`; duplicate-key `4`; sidecar-tamper `9` |

## 5. Test Results

| Test / metric | Expected | Actual | Delta | Result |
|---------------|---------:|-------:|------:|--------|
| Focused clean case | `1` | `1` | `0` | PASS |
| Recorded-commit invariant | `1` | `1` | `0` | PASS |
| Producer mutation rejections | `7` | `7` | `0` | PASS |
| Publication rejection cases | `2` | `2` | `0` | PASS |
| Focused `PASS_COUNT` | `11` | `11` | `0` | PASS |
| Task2 integration explicit PASS lines | `10` | `10` | `0` | PASS |
| Alternate manifest verified files | `19` | `19` | `0` | PASS |
| Interpreter contract `PASS_COUNT` | `12` | `12` | `0` | PASS |
| Parser negative cases | `11` | `11` | `0` | PASS |
| Duplicate-key negative cases | `4` | `4` | `0` | PASS |
| Sidecar-tamper negative cases | `9` | `9` | `0` | PASS |
| Test failures after final fixes | `0` | `0` | `0` | PASS |

## 6. Key Metrics

- Fixed outer producer files: `7`.
- Real-mode entry checks: `1` before build/reuse.
- Independent publication check sites: `2` functions (`task2_write_marker` and
  `task2_write_shared_pointer`).
- Focused assertions passed: `11/11`.
- Affected integration PASS lines: `10/10`.
- Existing interpreter contract: `12/12` with `24` combined parser/duplicate-key/sidecar-tamper
  negative cases (`11 + 4 + 9`).
- Final failed test cases across the three current GREEN transcripts: `0`.

No latency, accuracy, or GPU-performance metric was measured because the controller exposes
`0` CUDA devices.

## 7. Failure Diagnosis and Resolution

| Failure | Root cause | Resolution |
|---------|------------|------------|
| First RED continued after a missing command | Sourcing the production library disabled `errexit` in the test shell | Restore `set -euo pipefail` in the focused test after sourcing the extracted library |
| Missing-helper RED | Task2 had no outer working-tree-to-commit producer-byte comparison | Add the fixed seven-file, fail-fast Git blob helper and real-only entry/publication checks |
| Recorded-commit RED | Two sim-engine gitlink reads still used live `HEAD:` | Resolve both gitlinks from frozen `TASK2_MAIN_COMMIT` |
| First integration rerun: `recorded Megatron-LM commit is invalid while verifying Task2 bundle` | A direct-library fixture bypassed `task2_main`, so it did not set the newly required explicit commit | Set `TASK2_MAIN_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)` in that fixture; no production fallback was added |

## 8. Independent Review

### Design review

- Artifact: `.omx/artifacts/claude-act-as-an-independent-read-only-reviewer-for-the-sc26-ae-i55-2026-07-20T07-45-16-468Z.md`
- Provider exit: `0`
- Bytes: `7,729`
- SHA256: `33c984474aed308d262b78194954ab0b0d16bfe0b49a6cb363437989729ee855`
- Verdict: `APPROVE`

The reviewer required a complete fixed list, real-entry verification, independent reuse/build
publication checks, and all three Git object guards before live-hash comparison. The implemented
scope satisfies those corrections.

### Implementation review

- Artifact: `.omx/artifacts/claude-act-as-an-independent-read-only-implementation-reviewer-for--2026-07-20T07-57-46-511Z.md`
- Provider exit: `0`
- Bytes: `7,291`
- SHA256: `638403b9dc0010551e366355e6ce115b2501e9299507acba9cc06e64c106ae23`
- Verdict: `APPROVE`

The implementation reviewer reported no `BLOCK` or `HIGH` findings. Two `LOW` WATCH items remain:

1. The final check and file creation are not descriptor-anchored, so a non-adversarial TOCTOU
   window remains. Frozen/descriptor-anchored closure belongs to I57.
2. The focused unit does not separately cover invalid-repository or empty-commit publication
   inputs. The reviewed fail-fast implementation already handles them; adding cosmetic negatives
   is not required for this focused closure.

## 9. Non-Claims / Remaining Blockers

This report closes only the locally actionable Task2 outer-producer integrity subfinding. It does
not supply or prove:

- an authority-approved immutable interpreter or image digest;
- canonical-worker execution identity;
- cryptographic issuer authentication;
- adversarial TOCTOU resistance or a frozen descriptor-anchored input snapshot;
- real H800 Task2 execution or performance;
- `real_pre_dataset` or `release_pre_dataset` qualification;
- Gate B1 completion or AE readiness.

I55 therefore remains `OPEN / HIGH / BLOCK`. Gate B1 remains `BLOCKED`, both pre-datasets remain
`NOT QUALIFIED`, `AE-ready=NO`, and the overall workflow remains `INCOMPLETE`.
