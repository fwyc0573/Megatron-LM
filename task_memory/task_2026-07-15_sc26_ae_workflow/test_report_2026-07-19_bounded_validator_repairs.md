# Test Report: Session 45 Bounded Validator Repairs

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added fresh post-handoff current-state regression/static evidence, numeric e2e metrics, independent review identity, and explicit controller grouped-gemm non-qualification |
| 2026-07-19 | Added the I57 clean-clone stale-count RED/GREEN addendum, current 73-test SC26-AE matrix, standalone e2e metrics, and controller grouped-gemm dependency result |
| 2026-07-19 | Superseded the initial 21-case regression transcript with the current 31-case Task1 memory-negative-coverage regression; package/Task3 results remain green |
| 2026-07-19 | Recorded RED/GREEN and affected regression evidence for Task1 trace semantics, Task3 marker identity, and package summary convergence; release gates remain blocked |

## Scope and Boundary

This report covers only local, synthetic/controller validation for three bounded validator repairs
already present in the working tree:

1. Task1 rejects trace files that cannot satisfy the existing Task3 operation/metadata contract
   before publishing a verified marker.
2. The prebaked package consumer enforces exact Task3 marker/manifest identity and path-free
   `simulation_run_id` values before publishing a distribution manifest.
3. The package summary uses a bounded fixed-point write sequence so `total_size_bytes` and
   `distribution_medium` describe the final staged tree, and verification recomputes both values.

These checks do **not** qualify real GPUs, close producer snapshot/provenance gaps, authenticate an
issuer, publish a qualified shared pointer, or change any acceptance threshold/evidence class.

## Test Script Information

- Primary regression log: `logs/session45-bounded-repairs-regression.log`
- Exact command:

  ```bash
  ROOT=/data/ycfeng/sc26-ae-test-tmp/session45-bounded-repairs-20260719
  mkdir -p "$ROOT"
  SC26_AE_TMP_ROOT="$ROOT/task1" TMPDIR="$ROOT/task1" bash tests/integration/test_sc26_ae_task1_contracts.sh
  SC26_AE_TMP_ROOT="$ROOT/task2" TMPDIR="$ROOT/task2" bash tests/integration/test_sc26_ae_task2_contract.sh
  SC26_AE_TMP_ROOT="$ROOT/task3-contract" TMPDIR="$ROOT/task3-contract" bash tests/integration/test_sc26_ae_task3_contract.sh
  SC26_AE_TMP_ROOT="$ROOT/task3-portability" TMPDIR="$ROOT/task3-portability" bash tests/integration/test_sc26_ae_task3_portability.sh
  SC26_AE_TMP_ROOT="$ROOT/task3-provenance" TMPDIR="$ROOT/task3-provenance" bash tests/unit/test_sc26_ae_task3_provenance.sh
  pytest -q tests/unit/test_sc26_ae_artifact_manifest.py tests/unit/test_sc26_ae_package_prebaked.py tests/unit/test_sc26_ae_seal_qualification.py
  ```

- Full reproducible regression command, including syntax checks, is archived in the log-producing
  shell command for `logs/session45-bounded-repairs-regression.log`.
- Environment: `/usr/bin/python3` **3.12.3**, `pytest` **9.1.1**; no GPU, RJob, Docker, external
  attestation, publication, commit, or push was used.

## Validation Criteria

- The new Task1 semantic negative cases must fail before `capture_marker.json` publication.
- The package identity negative cases must fail before `distribution_manifest.json` publication.
- A built distribution must report the exact final staged byte sum and the medium selected from that
  final tree.
- A tampered `total_size_bytes` field must be rejected by `verify_distribution()`.
- A deliberately non-convergent summary medium must fail after a finite iteration bound.
- Existing artifact, package, sealer, Task1, Task2, and Task3 contracts must remain green.
- Shell/Python syntax and `git diff --check` must pass.

## Test Results and Evidence

| Suite / check | Result | Numeric evidence |
|---------------|--------|------------------|
| Artifact + package + sealer unit tests | PASS | **68 passed**, exit `0`, 7.33 s |
| Task1 integration contract | PASS | `PASS_COUNT=21`, exit `0` |
| Task2 integration contract | PASS | all contract assertions passed, exit `0` |
| Task3 integration contract | PASS | `PASS_COUNT=10`, exit `0` |
| Task3 portability | PASS | `PASS_COUNT=17`, exit `0` |
| Task3 provenance | PASS | `PROVENANCE_TEST_STATUS=PASS`, exit `0` |
| Shell syntax | PASS | **73** files checked with `bash -n` |
| Python syntax | PASS | **160** files checked with `python3 -m py_compile` |
| Diff hygiene | PASS | `git diff --check` exit `0` |

## RED → GREEN Evidence

### Task1 trace semantic gate

- RED: `logs/task1-trace-semantic-red-20260719.log`, exit `1`; the pre-repair validator accepted
  a trace missing `forward_step` and proceeded toward marker publication.
- GREEN: `logs/task1-trace-semantic-green-20260719.log`, exit `0`; `PASS_COUNT=21`, including ten
  semantic negative cases.
- The affected regression requires missing operation/metadata cases to fail before marker output;
  all ten cases passed.

### Task3 marker identity gate

- RED: `logs/task3_marker_identity_red.log`, exit `1`; five marker/manifest split-brain cases were
  accepted before the repair (`DID NOT RAISE`).
- GREEN: `logs/task3_marker_identity_targeted_green.log`, exit `0`; **5/5** targeted cases passed.
- The package full suite in the current combined working tree is included in the **68 passed**
  result above, rather than relying on the earlier pre-convergence-change count.

### Package final-size convergence gate

- RED: the new test observed declared `total_size_bytes=51,138` while the final staged tree was
  `51,207` bytes, an absolute discrepancy of **69 bytes**.
- GREEN: `test_build_records_exact_final_distribution_size` passed after the fixed-point write
  sequence; the stale-total and non-convergence edge tests also passed.
- The final implementation fails explicitly after **16** summary iterations if no fixed point is
  reached; it does not apply a scaling factor or fallback.

## Current Qualification Disposition

The local validator repairs are GREEN, but they are not qualification evidence:

```text
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

Open release findings I51–I58 and CR-01 remain open. In particular, the package summary repair
does not close the frozen-input/trusted-root architecture, and the Task1 repair does not establish
real SQLite/Nsight semantics, D16 timing fields, producer snapshot identity, or full-rank
promotion. The grouped-gemm controller prerequisite remains a separate `ModuleNotFoundError:
grouped_gemm` collection block.

## Evidence Identity

- Regression log bytes: **5,871**
- Regression log SHA256:
  `4f730106c05864f21e98d2fc1a5d10008654c2cc44b9d284fec7b06e057fc4a2`
- Current production/test file SHA256 values are recorded in the Session 45 progress/review
  addenda and must be re-hashed if any further edits occur.

## Recheck correction — current working-tree bytes

A later bounded Task1 test lane added ten memory-artifact negative cases (empty payload/samples,
non-finite or zero peaks, negative counters, all-zero samples, missing rank, and duplicate rank).
Because those tests changed the current working-tree bytes after the first matrix, the initial
`21/21` Task1 count is retained as historical evidence and is superseded for the final current
matrix by the recheck below.

- Current Task1 integration: `PASS_COUNT=31`, exit `0`.
- Current artifact/package/sealer unit suite: `68 passed`, exit `0`.
- Current Task3 contract/portability/provenance: `10/10`, `17/17`, and provenance PASS.
- Current shell/Python syntax: `73` and `160` files.
- Current regression log: `logs/session45-bounded-repairs-regression-v2.log`, bytes `6,558`,
  SHA256 `3ea96feba83eb0cc22b40239a7945a3f9b0a10bcdd9f6f9b05ca7cf2b42fa75c`.

The additional memory checks are test-only coverage. They do not change the Task1 acceptance
criteria or promote local synthetic evidence to a GPU qualification result.

## Superseding I57 portability and current-matrix addendum — 2026-07-19

### Motivation

The clean-clone replay was rerun after Task1 semantic and memory-negative cases changed the public
Task1 contract from `11` to `31`. The first replay reached `[PASS]` for every internal case but
failed its stale final grep; the same continuation also required validating the new intermediate
Task1-root symlink rejection.

### Expectation

The replay harness must assert the current `PASS_COUNT=31`, all isolated producer repositories must
remain clean, and the current Task3 portability suite must reject the symlink escape before resolving
fresh inputs. Synthetic evidence must remain explicitly non-qualifying.

### Method and RED evidence

The immutable RED transcript is
`logs/task3-clean-clone-followup-20260719.log` (910 bytes, SHA256
`703ff0f937529c5afe65a1028f989978298f49d389f587b27cfd0ea83daf592e`, exit `1`). It shows setup,
Task1, Task2, Task3-prebaked, and fresh-chain internal `[PASS]` lines followed by
`CLEAN_CLONE_RC=1`; the stale `grep -Fq 'PASS_COUNT=11'` was the root cause.

### Minimal repair and GREEN evidence

Only `tests/e2e/test_sc26_ae_clean_clone_replay.sh` changed for the harness mismatch:
`PASS_COUNT=11` became `PASS_COUNT=31`. The production trusted-root check and portability negative
case are the existing narrow I57 repair; no evidence class, threshold, source, fallback, or schema
was changed. The corrected replay
`logs/task3-clean-clone-followup-green-20260719.log` exited `0`, reports public entries `3/3/3`,
setup cases `6`, fresh chain `1`, four clean clone/submodule statuses, and
`EVIDENCE_CLASS=local_synthetic_not_gpu_qualification` (1,385 bytes, SHA256
`1a19a88e525ca602fa888892295908cea56a85cc31929e564272cb061e9cde9c`).

### Current e2e numeric evidence

| Suite | Result | Key values |
|---|---|---|
| Fresh chain | PASS | chain `1`; traces/memory `4/4`; rows `2`; MSE `3.0/0.5`; reload delta `0.0`; rank0 `22.5 ms`; wall `0.5 s`; peak RSS `51,432 KiB` |
| Task1 smoke | PASS | `SMOKE_PASS_COUNT=1`; real workload `0`; Task1 `31` |
| Task2 smoke | PASS | three public entries; identity/snapshot contract PASS |
| Task3 prebaked CPU | PASS | models `3/3`; rank0 step `18.5/22.5/24.5 ms`; manifests `22/18/18` |
| Clean clone | PASS | public `3/3/3`; setup `6`; chain `1`; four clean statuses |

### Current affected matrix and static evidence

`logs/session45-task3-symlink-affected-regression-v2-20260719.log` exited `0` (19,632 bytes,
SHA256 `5a58a47013efabb7e17aa9a92c906c39f4e2196b8f66da2c10e7f009a726bd00`) with Python unit
`73 passed in 5.18 s`, Task1 `PASS_COUNT=31`, Task3 contract `10/10`, portability `18/18`, and
provenance PASS across 20 SC26-AE shell/integration/e2e scripts. The broad static log
`logs/session45-task3-symlink-static-validation-20260719.log` exited `0` (183 bytes, SHA256
`f545a96bbac62907b325a4d5c00dc85bda0a87420731c4313bda70a7d22dcb3c`) with shell syntax `73`,
Python syntax `160`, temporary-root scan PASS, and `git diff --check` PASS.

The modified grouped-gemm setup test (`37/37`) and GPT example mock integration (`22/22`) pass,
but the grouped-gemm runtime test is blocked at collection by `ModuleNotFoundError: grouped_gemm`;
this controller dependency gap is recorded in
`logs/session45-grouped-gemm-affected-regression-20260719.log` (4,276 bytes, SHA256
`6484172a9d9931f917a85d73177ecc677618bc3ca31ea562ff08511eb7259c12`, exit `2`).

### Boundary

All values above are local synthetic/controller evidence. I57 remains open for the approved frozen
input snapshot/trusted-root architecture; I51-I58 and CR-01 remain open as documented. Gate B1 is
`BLOCKED`, real/release pre-datasets are `NOT QUALIFIED`, and `AE-ready` is `NO`.

## Fresh post-handoff current-state verification — 2026-07-19

### 1. Test Script Information

**Environment**

- Worktree: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Python: `/usr/bin/python3`, Python `3.12.3`
- Pytest: `/home/i-fengyicheng/.local/bin/pytest`, pytest `9.1.1`
- Conda environment: none active (`CONDA_DEFAULT_ENV=none`)
- Temporary root:
  `/data/ycfeng/sc26-ae-test-tmp/session45-final-current-state-v18-20260719`
- GPU/runtime boundary: controller-only; no GPU, RJob, Docker, package installation, or network
  qualification was attempted.

**Python unit command**

```bash
pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py \
  tests/unit/test_sc26_ae_package_prebaked.py \
  tests/unit/test_sc26_ae_seal_qualification.py
```

**Shell unit/integration/e2e commands**

```bash
for test_file in \
  tests/unit/test_sc26_ae_common.sh \
  tests/unit/test_sc26_ae_docs_contract.sh \
  tests/unit/test_sc26_ae_setup_runtime.sh \
  tests/unit/test_sc26_ae_task1_source_provenance.sh \
  tests/unit/test_sc26_ae_task2_evidence_mode.sh \
  tests/unit/test_sc26_ae_task2_interpreter_contract.sh \
  tests/unit/test_sc26_ae_task2_snapshot.sh \
  tests/unit/test_sc26_ae_task3_contracts.sh \
  tests/unit/test_sc26_ae_task3_interpreter_contract.sh \
  tests/unit/test_sc26_ae_task3_provenance.sh \
  tests/integration/test_sc26_ae_setup.sh \
  tests/integration/test_sc26_ae_task1_contracts.sh \
  tests/integration/test_sc26_ae_task2_contract.sh \
  tests/integration/test_sc26_ae_task3_contract.sh \
  tests/integration/test_sc26_ae_task3_portability.sh \
  tests/e2e/test_sc26_ae_clean_clone_replay.sh \
  tests/e2e/test_sc26_ae_fresh_chain.sh \
  tests/e2e/test_sc26_ae_task1_smoke.sh \
  tests/e2e/test_sc26_ae_task2_smoke.sh \
  tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh; do
    bash "$test_file"
done

bash tests/unit/test_setup_grouped_gemm_v1.sh
bash tests/integration/test_gpt_example_mock_mode.sh
```

**Static commands**

```bash
find SC26-AE tests tools -type f -name '*.sh' -print | sort
# Run bash -n on all 73 listed files.

find SC26-AE tests -type f -name '*.py' -print | sort
# Parse all 160 listed files with Python ast.parse().

rg -n '/tmp/(sc26-ae|grouped-gemm)' SC26-AE tests tools \
  --glob '*.sh' --glob '*.py'
git diff --check
```

**Evidence logs**

- Regression:
  `logs/session45-final-current-state-regression-v18-20260719.log`, 23,288 bytes, SHA256
  `d1bbd254759fb2efb2cbc9ce0f1a9b8efaeecc9702b22287b98b81e59b435098`.
- Static:
  `logs/session45-final-current-state-static-v18-20260719.log`, 251 bytes, SHA256
  `653ca2bbab392d034c723f205a2b2f02a424853dd51f50236d41ec70c1ae2e3c`.
- Independent review:
  `.omx/artifacts/claude-you-are-an-independent-review-lane-for-an-sc-26-artifact-eva-2026-07-19T14-52-09-639Z.md`,
  7,443 bytes, SHA256
  `211c2c11c851238f3caf291fb3a8085cded7bfcd1faec9a5d19402ac995c21e3`.

### 2. Validation Criteria

1. All 73 Python unit cases pass.
2. All ten SC26-AE unit-shell, five integration, and five e2e scripts exit `0`.
3. Task1 reports exactly 31 cases; Task3 contract reports 10; Task3 portability reports 18;
   Task3 provenance reports PASS.
4. Clean-clone replay reports public Task1/Task2/Task3 counts `3/3/3`, setup `6`, fresh chain `1`,
   and four clean Git statuses.
5. Grouped-gemm setup and GPT mock contracts pass `37/37` and `22/22` without treating the absent
   runtime module as qualification.
6. Shell/Python static scopes remain exactly `73/160`; the hard-coded temporary-root scan and
   `git diff --check` pass.
7. All evidence remains explicitly local synthetic/controller evidence; release boundary phrases
   and open I51-I58/CR-01 findings remain unchanged.

### 3. Test Results and Evidence

| Suite | Result | Actual numeric evidence |
|---|---|---|
| Python unit | PASS | `73 passed in 4.55 s` |
| Unit shell | PASS | `10/10` scripts |
| Integration | PASS | `5/5` scripts; Task1 `31`; Task3 `10`; portability `18`; provenance PASS |
| E2E | PASS | `5/5` scripts; public entries `3/3/3`; fresh chain `1`; models `3/3` |
| Grouped-gemm setup | PASS | `37/37` installer-contract cases |
| GPT mock integration | PASS | `22/22` cases |
| Grouped-gemm runtime | NOT RUN / NOT QUALIFIED | CUDA devices `0`; module available `False` |
| Static | PASS | shell `73/73`; Python `160/160`; temp-root scan PASS; diff check PASS |

### Key metrics

| Metric | Expected | Actual | Delta |
|---|---:|---:|---:|
| Task1 contract cases | 31 | 31 | 0 |
| Task3 contract cases | 10 | 10 | 0 |
| Task3 portability cases | 18 | 18 | 0 |
| Fresh-chain Task1 trace files | 4 | 4 | 0 |
| Fresh-chain Task1 memory files | 4 | 4 | 0 |
| Task2 synthetic dataset rows | 2 | 2 | 0 |
| Task2 validation MSE | 3.0 | 3.0 | 0.0 |
| Task2 test MSE | 0.5 | 0.5 | 0.0 |
| Task2 reload max absolute prediction delta | 0.0 | 0.0 | 0.0 |
| Fresh Task3 rank0 step (ms) | 22.5 | 22.5 | 0.0 |
| Fresh Task3 forward/backward/optimizer (ms) | 6.0 / 11.0 / 2.5 | 6.0 / 11.0 / 2.5 | 0.0 / 0.0 / 0.0 |
| Fresh Task3 simulator wall time (s) | 0.5 | 0.5 | 0.0 |
| Fresh Task3 peak RSS (KiB) | diagnostic | 51,704 | n/a |
| Prebaked rank0 step GPT/Qwen/DSV3 (ms) | 18.5 / 22.5 / 24.5 | 18.5 / 22.5 / 24.5 | 0.0 / 0.0 / 0.0 |
| Prebaked manifest files GPT/Qwen/DSV3 | 22 / 18 / 18 | 22 / 18 / 18 | 0 / 0 / 0 |

### Harness incidents

- A read-only audit pipeline initially exited `141` because `pipefail` exposed `head` closing a
  sorted producer early. Replacing `head` with `sed -n` corrected the audit command; no project
  test or source file was involved.
- The first tool-wrapper matrix launch was rejected before shell execution because two shell
  `${...}` expressions were not escaped in the JavaScript template. No evidence log was created.
  The corrected wrapper executed the exact command above and produced the immutable GREEN log.

### Boundary

The fresh matrix is `local_synthetic_not_gpu_qualification`. It does not qualify grouped-gemm
runtime, H800 execution, Task2 exact-two-H800 evidence, release packaging, or issuer authentication.
I51-I58 and CR-01 remain open. Overall status remains `INCOMPLETE`; Gate B1 is `BLOCKED`;
`real_pre_dataset` and `release_pre_dataset` are `NOT QUALIFIED`; `AE-ready=NO`.
