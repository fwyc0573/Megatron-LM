# Test Report: Task1 Memory-Artifact Negative Coverage

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added deterministic integration coverage for invalid, non-positive, and incomplete Task1 memory artifacts; real qualification status remains unchanged |

## Scope and evidence boundary

This report covers a test-only hardening of the Task1 synthetic integration fixture. It does not
change the production acceptance contract, promote a marker, or qualify a GPU run. Every result in
this report is classified as:

```text
local_synthetic_not_gpu_qualification
```

The change exercises the existing validator branches identified in the Session 45 read-only audit:
empty payloads/samples, NaN/Infinity/non-positive peaks, negative memory values, all-zero samples,
and missing/duplicate rank inventory. No release-level I51-I58 or CR-01 design was implemented.

## Test Script Information

### Environment

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Shell: GNU bash `5.2.21`
- Python: `3.12.3` (`/usr/bin/python`)
- pytest: `9.1.1`
- Hardware evidence: controller-only synthetic fixtures; CUDA device count `0`; no RJob/GPU/Docker was started

### Scripts and commands

1. Shell syntax:

   ```bash
   bash -n tests/integration/test_sc26_ae_task1_contracts.sh
   ```

2. Focused Task1 integration contract:

   ```bash
   bash tests/integration/test_sc26_ae_task1_contracts.sh
   ```

3. Public Task1 synthetic smoke:

   ```bash
   bash tests/e2e/test_sc26_ae_task1_smoke.sh
   ```

4. Fresh synthetic Task1 → Task2 → Task3 chain:

   ```bash
   bash tests/e2e/test_sc26_ae_fresh_chain.sh
   ```

Raw focused output is retained at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/session45-task1-memory-negative-coverage.log
```

The focused log is `2,168` bytes with SHA256
`accaa663b58e0a3e0f9108eb3edba090adbee417b6d30b1c991672c576c6dc0d` and exit `0`.

The affected local matrix output is retained at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/session45-task1-memory-affected-matrix.log
```

It covers `20` SC26-AE shell contract/integration/e2e scripts and the four Python unit modules;
the Python suite reports `73 passed in 5.22 s`, `MATRIX_STATUS=PASS`, and exit `0`. The matrix log
is `17,303` bytes with SHA256
`cc02a5fada02efd207dcd3dce8b30b9eb3f1d8d39759f04ee03a0c7bedbf5d3f`.

## Validation Criteria

The focused integration test must:

1. execute all pre-existing Task1 adapter, rank-scope, trace-semantic, source-failure, and
   existing-destination checks;
2. execute ten memory-failure cases;
3. reject every malformed memory bundle before `capture_marker.json` is published;
4. report exactly `PASS_COUNT=31`;
5. retain the existing fail-fast and no-fallback behavior.

The e2e checks must continue to pass, and their numeric report values must remain finite and
internally consistent. Passing these local checks must not alter the real/release qualification
boundary.

## Test Results and Evidence

| Test | Result | Observed evidence |
|------|--------|--------------------|
| `bash -n` | PASS | Shell syntax accepted, exit `0` |
| Task1 integration contract | PASS | `PASS_COUNT=31`; ten new memory negative cases; exit `0` |
| Task1 public smoke | PASS | `SMOKE_PASS_COUNT=1`; `REAL_GPU_WORKLOAD_COUNT=0`; exit `0` |
| Fresh synthetic chain | PASS | `CHAIN_PASS_COUNT=1`; exit `0` |
| Affected local matrix | PASS | `20` shell scripts; `73` Python tests passed in `5.22 s`; exit `0` |

### Memory negative cases

| Case | Expected rejection | Observed result |
|------|--------------------|-----------------|
| Empty payload | `Empty memory payload` | Rejected before marker publication |
| Empty samples | `Memory samples are empty` | Rejected before marker publication |
| NaN peak | `Non-positive peak_allocated_MB` | Rejected before marker publication |
| Infinity peak | `Non-positive peak_allocated_MB` | Rejected before marker publication |
| Zero peak | `Non-positive peak_allocated_MB` | Rejected before marker publication |
| Negative reserved memory | `Invalid reserved_memory_MB` | Rejected before marker publication |
| Negative allocated memory | `Invalid allocated_memory_MB` | Rejected before marker publication |
| All-zero samples | `Memory samples never report positive usage` | Rejected before marker publication |
| Missing rank file | `Memory rank inventory mismatch` | Rejected before marker publication |
| Duplicate rank file | `Memory rank inventory mismatch` | Rejected before marker publication |

### Fresh-chain numeric evidence

| Metric | Expected / acceptance | Actual | Delta / derived value |
|--------|----------------------|--------|----------------------|
| Task1 trace files | `4` selected synthetic ranks | `4` | `0` |
| Task1 memory JSON files | `4` selected synthetic ranks | `4` | `0` |
| Task2 dataset rows | Fixture contract requires `2` | `2` | `0` |
| Task2 average validation MSE | Finite fixture value | `3.0` | finite |
| Task2 test MSE | Finite fixture value | `0.5` | finite |
| Reload max absolute prediction delta | `0.0` | `0.0` | exact match |
| Rank0 step time | Positive finite | `22.5 ms` | — |
| Forward duration | Positive finite | `6.0 ms` | — |
| Backward duration | Positive finite | `11.0 ms` | — |
| Optimizer duration | Positive finite | `2.5 ms` | — |
| Simulator load time | Non-negative | `0.125 s` | — |
| Simulator execution time | Non-negative | `0.375 s` | — |
| Simulator wall-clock | `load + execution` | `0.5 s` | `0.0 s` |
| Process wall-clock | Observed host value | `0.901412 s` | — |
| Peak RSS | Host observation | `51,536 KiB` (`0.04914856 GiB`) | — |
| Tested host allocation | Fixture declaration | `32 MiB` | — |

## RED → root cause → GREEN

The first run after adding the ten assertions produced the intended RED:

```text
FAIL: empty-payload memory semantics were unexpectedly accepted
```

Root cause was in the test fixture, not production validation: the fake `torchrun` always emitted
the valid memory JSON and ignored the newly supplied `FAKE_MEMORY_MODE`. The minimal repair added
fixture-only payload modes and rank-inventory mutations, without weakening any assertion or adding
fallback behavior. The rerun produced `PASS_COUNT=31` and exit `0`.

## Current qualification boundary

This test hardening closes only a local coverage gap. It does **not** close I53 as a release issue,
because real SQLite/NVTX semantics, canonical `nsys` identity, D16 timing fields, complete producer
provenance, and real H800 evidence remain unresolved. The authoritative status remains:

```text
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```
