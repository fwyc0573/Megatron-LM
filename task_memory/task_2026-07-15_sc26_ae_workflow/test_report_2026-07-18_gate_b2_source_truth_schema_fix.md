## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-18 | Recorded the terminal B2 source-of-truth schema failure, A3 RED -> GREEN evidence, current reproducibility boundary, and local-vs-real qualification status |

# Test Report: Gate B2 Source-of-Truth Schema Fix and A3 Self-Repair Boundary

**Date**: 2026-07-18  
**Task**: `task_2026-07-15_sc26_ae_workflow`  
**Scope**: Test/audit/control-plane repair only; no product or runtime qualification change.

## 1. Test Script Information

### Historical runtime evidence

The evidence was produced in the clean execution runtime (before the later local AE implementation changed its Git state):

```text
/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/
```

Scripts and logs:

- `test_worker_source_truth_contract.py` -> `tdd_source_truth_fix_green.log`
- `test_generate_b2_harness.py` -> `lifecycle_contract_green.log`
- `test_static_validator_derivation.py`, `test_validate_sealed_zero_drift.py`, and the remaining controller suites -> `final_controller_regression_green.log`
- `test_post_validate_fixtures.py` -> `final_post_validate_pre_generation.log` and its `fixture_summary.json`
- Terminal live evidence -> `live_only.log`, `live_only_result.env`

### Reproducible command forms

Run from the runtime directory with the fixed interpreter and bytecode suppression:

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B -m pytest -q test_worker_source_truth_contract.py
/usr/bin/python3 -B test_generate_b2_harness.py
/usr/bin/python3 -B test_static_validator_derivation.py
/usr/bin/python3 -B test_validate_sealed_zero_drift.py
/usr/bin/python3 -B test_predict_semantic_validator.py
/usr/bin/python3 -B test_post_validate_fixtures.py
```

The historical aggregate command covered the same controller suites and recorded the aggregate result below. The live command is retained only as immutable evidence; it must **not** be rerun for `20260718T101257Z`:

```text
RLAUNCH_EXIT=1
TEE_EXIT=0
ELAPSED_SECONDS=114
LIVE_INVOCATION_COUNT=1
```

### Environment

- Interpreter: `/usr/bin/python3`, Python `3.12.3`
- Test runner: `pytest 9.1.1` for the function-style source-truth contract suite; Python `unittest` entry points for the controller suites
- `PYTHONDONTWRITEBYTECODE=1` and `-B`
- No GPU, scheduler, image, or workload was used by the local contract tests.

## 2. Validation Criteria

1. The emitted worker must accept the producer's valid `source_of_truth.json`, including the D47 `d47_authority` object.
2. The worker must reject an unexpected top-level key and an unexpected nested D47 key.
3. Lifecycle, static-validator, zero-drift, and semantic fixture contracts must remain strict; no assertion weakening or hash skipping is allowed.
4. All original failure evidence and the consumed identity must remain immutable.
5. Local GREEN must be reported only as `[LOCAL-SYNTHETIC]` test/audit evidence. It must not be promoted to Gate B2/B3/B4/B5, real GPU qualification, runtime evidence, or pre-dataset qualification.

## 3. Test Results and Evidence

### 3.1 Terminal failure (the block that initiated the repair)

| Metric | Expected | Actual | Delta / status |
|---|---:|---:|---|
| Live invocations for `RUN_ID=20260718T101257Z` | 1 | 1 | exact-once boundary met |
| `rlaunch` exit | 0 | 1 | `-1`, terminal failure |
| `tee` exit | 0 | 0 | log capture completed |
| Elapsed time | positive | 114 s | evidence preserved |
| Worker top-level source-of-truth keys | producer and consumer equal | mismatch | `d47_authority` was rejected |
| Workload start | must reach workload for B2 qualification | not reached | failure was pre-workload |

Observed worker error:

```text
RuntimeError: Unexpected source-of-truth keys
```

Failure boundary: before `nvidia-smi`, GPU identity inventory, Megatron product import, Qwen workload, trace capture, memory capture, capture marker, and `qualification_result.json`.

### 3.2 RED -> GREEN contract evidence

| Suite / artifact | Expected | Actual | Result |
|---|---:|---:|---|
| Focused source-truth contract | 3/3 pass | 3/3 pass | PASS |
| Lifecycle contract | 19 tests, 50 subtests | 19/19, 50 subtests | PASS |
| Aggregate controller regression | all tests pass | 57 passed, 70 subtests | PASS |
| Post-validate fixture matrix | 17 cases; 1 positive, 16 negative | 17/17; 1 positive, 16 negative | PASS |

The earlier `55 passed` plus `2 expected state-bound failures` log remains preserved. Those two failures are intentionally not reclassified as GREEN for the consumed identity: they prove that an already-sealed identity cannot be patched or resealed in place.

### 3.3 Current reproducibility check

On 2026-07-18, a fresh local rerun of the focused command:

```bash
/usr/bin/python3 -B -m pytest -q test_worker_source_truth_contract.py
```

did not reach the schema assertions. The generator's clean-worktree precondition stopped all three cases because the execution worktree now contains local AE implementation changes:

```text
RuntimeError: Clean execution worktree has tracked changes: ['megatron-sim-engine']
```

Current status also shows the expected in-progress local files (`SC26-AE/` and AE test files). This is a reproducibility/environment-boundary failure, not evidence that the schema fix regressed. It must be resolved by the owning implementation lane or by an explicitly reviewed test/control-plane contract adjustment; do not clean, revert, or hide those changes.

## 4. Root-Cause Analysis and Resolution

**Root cause**: `generate_b2_harness.py` correctly added the D47 provenance object `d47_authority` to `source_of_truth.json`, but the emitted `WORKER_TEMPLATE` consumer retained the old exact top-level key set. A valid producer payload was rejected by its own consumer.

**Resolution**: The producer and emitted consumer now share canonical top-level and D47 nested schema constants. Positive and negative tests exercise the real rendered consumer. The consumed sealed identity, its raw log, and all prior failure artifacts were not edited.

## 5. Qualification Boundary and Pending Work

- A3 permits self-diagnosis and self-repair for this test/audit/control-plane defect because it directly serves the AE one-click shell and pre-dataset workflow.
- Real GPU, quota, image, scheduler, product, runtime, workload, and data-quality failures remain fail-fast and require a fresh identity for any future qualification.
- `RUN_ID=20260718T101257Z` is terminal and non-reusable; no patch, retry, reuse, or reseal is allowed.
- Local Phase 1–5 shell/manifest/scheduler/reporter work may continue only with `[LOCAL-SYNTHETIC]` labels. B5 interface reconciliation remains mandatory.
- Final delivery still requires real runtime evidence, contract-compliant pre-dataset/assets, and reproducible AE-facing scripts. The local numbers in this report do not satisfy those final criteria.

## 6. Evidence Paths and Hashes

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `live_only.log` | 10,470 | `c2a3c9c1e0ce237736caffdc81dcfc3936418498efee9f7ac0dee1d23cde5c52` |
| `live_only_result.env` | 194 | `2f25e0023fd80714d46799a2f2664784e9dda2cf3280915f8597bb3228cf3e58` |
| `source_truth_testfix_diagnosis.md` | 1,907 | `c07817586134261d95f0b4e1b98a9dc7b1c2c3b254519d2aca8d8c1d93c6e5fb` |

**Overall report status**: `PASS FOR LOCAL TEST/AUDIT REPAIR; REAL GATE B2 REMAINS BLOCKED/UNQUALIFIED`.
