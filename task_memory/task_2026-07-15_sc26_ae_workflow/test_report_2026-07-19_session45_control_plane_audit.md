# Test Report: Session 45 Control-Plane Audit and Alias Repair

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Superseded the overwritten Session 45 verifier reference with immutable v3 RED/GREEN logs and recorded the refreshed document-hash inventory |
| 2026-07-19 | Recorded fresh post-audit local regression, static/document gate, read-only provenance findings, and the controller grouped-gemm prerequisite block |

## Scope and status boundary

This report covers the local Task2 checksum-alias repair and a read-only audit of the Task1/Task2/
Task3 qualification handoff. It does not claim a real GPU qualification, external attestation,
release pre-dataset, or AE readiness.

```text
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

The D45 semantic quota remains `gpu : 129/128` (CLI exit `0`, semantic `FAIL`). No GPU/RJob,
Docker, publication, commit, push, reset, `rm`, `mv`, or submodule mutation was performed.

## 1. Test Script Information

### Environment

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Outer `HEAD`: `c217ce93156e7c37e065da2989c1a482f12ecebc`
- Nested sim-engine `HEAD`: `39755169f73f6c748e8d7376c3a2158c6569436b`
- Nested sim-engine status: clean at audit time
- Python: `Python 3.12.3`
- pytest: `pytest 9.1.1`
- Temporary root: `/data/ycfeng/sc26-ae-test-tmp/session45-post-audit`

### Exact commands

1. Focused/full local regression (all commands were run with `SC26_AE_TMP_ROOT` and `TMPDIR`
   under `/data/ycfeng/sc26-ae-test-tmp/session45-post-audit`):

   ```bash
   bash tests/unit/test_sc26_ae_docs_contract.sh
   bash tests/unit/test_sc26_ae_common.sh
   bash tests/unit/test_sc26_ae_setup_runtime.sh
   bash tests/unit/test_sc26_ae_task1_source_provenance.sh
   bash tests/unit/test_sc26_ae_task2_evidence_mode.sh
   bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
   bash tests/unit/test_sc26_ae_task2_snapshot.sh
   bash tests/unit/test_sc26_ae_task3_contracts.sh
   bash tests/unit/test_sc26_ae_task3_interpreter_contract.sh
   bash tests/unit/test_sc26_ae_task3_provenance.sh
   pytest -q tests/unit/test_sc26_ae_artifact_manifest.py \
     tests/unit/test_sc26_ae_echo_metrics.py \
     tests/unit/test_sc26_ae_package_prebaked.py \
     tests/unit/test_sc26_ae_seal_qualification.py
   bash tests/integration/test_sc26_ae_setup.sh
   bash tests/integration/test_sc26_ae_task1_contracts.sh
   bash tests/integration/test_sc26_ae_task2_contract.sh
   bash tests/integration/test_sc26_ae_task3_contract.sh
   bash tests/integration/test_sc26_ae_task3_portability.sh
   bash tests/unit/test_setup_grouped_gemm_v1.sh
   bash tests/integration/test_gpt_example_mock_mode.sh
   bash tests/e2e/test_sc26_ae_task1_smoke.sh
   bash tests/e2e/test_sc26_ae_task2_smoke.sh
   bash tests/e2e/test_sc26_ae_fresh_chain.sh
   bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
   bash tests/e2e/test_sc26_ae_clean_clone_replay.sh
   ```

   Transcript: `logs/session45-post-audit-regression.log`  
   SHA256: `fced7bb7614dc3c093427496d69f7d23bfa0ce6e0986049efd02611796f49871`  
   Size: `17,334` bytes / `325` lines  
   Exit: `0`

2. Documentation/static gate:

   ```bash
   bash tests/unit/test_sc26_ae_docs_contract.sh
   # semantic duplicate/future-status probe
   # bash -n over the fixed 52-file shell scope
   # python3 -m py_compile over the fixed 35-file Python scope
   # hard-coded temporary-root scan
   git diff --check
   ```

   Transcript: `logs/session45-static-doc-gate.log`  
   SHA256: `476f357063b71ff67d9904e6d4eb35968de9f773612157bd90ff92883b2d277c`  
   Size: `362` bytes / `14` lines  
   Exit: `0`

3. Read-only control-plane audit:

   ```bash
   # git identity/status, producer-path HEAD membership, source snippets,
   # Task1 rank/timing/nsys fields, Task2 lifecycle, Task3 root/snapshot/package fields
   ```

   Transcript: `logs/session45-control-plane-audit-raw.log`  
   SHA256: `14342fc38a909854712de101518ccc7c39828e7a149b1d0fa8637a0a05d6c40a`  
   Size: `119,279` bytes / `1,271` lines  
   Exit: `0` (read-only inspection)

4. Controller grouped-gemm runtime probe:

   ```bash
   pytest -q tests/integration/test_grouped_gemm_v1_runtime.py
   ```

   Transcript: `logs/session45-grouped-gemm-runtime-probe.log`  
   SHA256: `2db2cdd457fa880d915a1bf37aa0e79fd736ed8df98fead8b5ac28862804233e`  
   Exit: `2` during collection because `grouped_gemm` is not installed. This is recorded as a
   controller prerequisite block, not as a qualification result.

## 2. Validation Criteria

- The second Task2 checksum alias must be mandatory and equal to the actual manifest digest.
- Changed Task2 producer/consumer paths and affected regressions must pass without fallback.
- All local synthetic entries, fresh/prebaked chains, and clean-clone rehearsal must remain
  evidence-classed as `local_synthetic_not_gpu_qualification`.
- Shell/Python syntax, documentation contract, temporary-root scan, and `git diff --check` must
  pass.
- The audit must preserve, rather than hide, source-binding, qualified-reuse, rank-scope, trusted
  path, snapshot, issuer, and external-quota blockers.

## 3. Test Results and Evidence

| Suite / gate | Result | Numeric evidence |
|--------------|--------|------------------|
| Documentation contract | PASS | public entries `9`; paper suggestions `10` |
| Local Python unit package matrix | PASS | `65 passed` in `2.86 s` |
| Task1 source provenance | PASS | `2/2` |
| Task1 integration | PASS | `11/11` |
| Task2 evidence mode | PASS | `4/4` |
| Task2 interpreter contract | PASS | `11` cases |
| Task2 integration incl. alias negative | PASS | model attachments `3/3`; manifest files `13` |
| Task2 snapshot contract | PASS | synthetic snapshot/reuse/rebuild/dirty/new-path checks |
| Task3 contracts/integration/portability | PASS | `9/9`, `10/10`, `17/17` |
| Setup integration | PASS | `6/6`; real installer executions `0` |
| Grouped-gemm setup unit | PASS | `37/37` |
| GPT mock integration | PASS | `22/22`; real GPU workload count `0` |
| Fresh synthetic Task1→Task2→Task3 chain | PASS | chain `1/1`; traces/memory `4/4`; rows `2` |
| Prebaked Task3 CPU rehearsal | PASS | models `3/3`; rank0 steps `18.5/22.5/24.5 ms` |
| Clean-clone-style replay | PASS | public entries `3/3/3`; fresh chain `1`; clone statuses `4` clean |
| Shell syntax | PASS | `52` files |
| Python syntax | PASS | `35` files |
| Hard-coded temporary-root scan | PASS | matches `0` |
| `git diff --check` | PASS | exit `0` |
| Grouped-gemm runtime import | BLOCKED | `ModuleNotFoundError: grouped_gemm`, collection exit `2` |

### Numeric synthetic workflow metrics

| Metric | Observed value |
|--------|---------------:|
| Fresh Task1 trace files | `4` |
| Fresh Task1 memory JSON files | `4` |
| Fresh Task2 dataset rows | `2` |
| Fresh Task2 average validation MSE | `3.0` |
| Fresh Task2 test MSE | `0.5` |
| Fresh Task2 reload max absolute prediction delta | `0.0` |
| Fresh Task3 rank0 step | `22.5 ms` |
| Fresh Task3 forward/backward/optimizer | `6.0/11.0/2.5 ms` |
| Fresh Task3 simulator load/execution/wall | `0.125/0.375/0.5 s` |
| Fresh Task3 peak RSS | `51,332 KiB` (`0.04895401 GiB`) |
| Tested host allocation | `32 MiB` |

These values come from deterministic fixtures and are not GPU performance or release-quality
metrics.

## 4. RED → GREEN Repair Evidence

### Task2 alias defect

**RED:** `logs/task2-pointer-alias-red-session45.log` exited `1` because a pointer with a correct
`manifest_sha256` and tampered `artifact_manifest_sha256` alias was accepted.

**Root cause:** the shared-pointer parser did not emit the second alias to the shell comparison.

**Minimal repair:** `SC26-AE/lib/task2_echo.sh` now parses both aliases and requires both to equal
the verified manifest digest. The integration test adds the second-alias mutation case and restores
the pointer before continuing.

**GREEN:** `logs/task2-pointer-alias-green-session45-final.log` exited `0`; the full affected
regression above also exited `0`. No threshold, fallback, source-selection, evidence label, or
release rule changed.

## 5. Audit findings that remain open

The read-only audit is archived in `phase10_control_plane_audit_2026-07-19.md`. It confirms:

1. Task1/Task2 AE producer paths are not all represented by the recorded outer commit.
2. MoE QUICK rank scope is not machine-gated against full release promotion.
3. Task1 trace/SQLite semantics, canonical real `nsys`, and D16 timing fields are incomplete.
4. Qualified Task2 reuse, sealer publication, and predictor-ID identity have no closed lifecycle.
5. Echo subordinate interpreter binding and Task2 producer provenance are incomplete.
6. Task2 trusted paths/cross-file identity and Task3 trusted-root/frozen-input handling have gaps.
7. Package summary/schema/prebaked semantic checks need a design-consistent contract.
8. CR-01 cryptographic issuer authentication remains an external governance blocker.

These findings are recorded as I51-I58 and F10-01--F10-12. They are not waived by the local
regression and require design approval before implementation.

## 6. Pending tasks, newly discovered issues, and recommended next steps

### Pending tasks

1. Approve the producer snapshot and full-rank promotion design.
2. Define the Task2 raw→attested→qualified state machine and canonical pointer publication.
3. Define trusted-root/frozen-input and schema-specific package contracts.
4. Obtain issuer-authentication governance and external two-H800 authority.
5. Only after the above, implement and rerun the corresponding negative/positive tests, then run
   real qualification in an authorized environment.

### Newly discovered issues

- The current outer `HEAD` omits the untracked AE producer overlay, so a current real manifest
  would not be reproducible from its recorded commit.
- The plan names four D16 timing fields that the Task1 runner does not currently emit.
- The qualified Task2 evidence class cannot pass every existing verifier/marker stage.
- The sealer has no canonical qualified shared-pointer publication path.
- The controller-side grouped-gemm runtime import is unavailable.

### Recommended next steps

1. Keep Gate B1, real/release pre-datasets, and AE-ready status unchanged; do not submit another
   RJob from this controller.
2. Have the owner approve the six design decisions listed in the Session 45 plan addendum.
3. Implement one design-approved control-plane change at a time with RED→GREEN tests and fresh
   regression evidence.
4. Re-run the grouped-gemm runtime probe only in the designated qualified environment.


## 7. Final verifier after report/hash addenda

The final status-aware verifier was rerun after this report, the phase10 audit, and the task
planning documents were finalized:

- Log: `logs/session45-final-verification.log`
- Bytes/lines: `631` / `16`
- SHA256: `c129643dab177c398331b1be4f9fe057b73feae5de5ecf8f5b40e97ad0e9fe07`
- Exit: `0`
- Observed: docs=`9/10`, shell=`52`, Python=`35`, issue headings `I50..I58`, hard-coded
  temporary templates=`0`, `git diff --check=PASS`.

This final verifier closes only the local document/static checkpoint. It does not change the
external quota result or any real/release qualification label.

## 8. Immutable verifier-log supersession

### Test Script Information

- Working directory: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Environment: system `Python 3.12.3`; no conda environment activated
- Temporary root: `/data/ycfeng/sc26-ae-test-tmp/session45-final-v3-green`, supplied through both
  `SC26_AE_TMP_ROOT` and `TMPDIR`
- Commands:

  ```bash
  SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/session45-final-v3-green \
    TMPDIR=/data/ycfeng/sc26-ae-test-tmp/session45-final-v3-green \
    bash tests/unit/test_sc26_ae_docs_contract.sh
  # inline semantic/document hash probe, fixed 52-shell bash -n scope,
  # fixed 35-file Python py_compile scope, temporary-root scan, and:
  git diff --check
  ```

### Validation Criteria

1. The verifier must fail fast on a real predicate failure and preserve the failed transcript.
2. The corrected run must report documentation=`9/10`, issue headings `I50..I58`, shell/Python
   syntax=`52/35`, zero hard-coded temporary templates, and `git diff --check`=`PASS`.
3. The document hash inventory must cover nine non-self-referential task documents and must not
   alter the blocked real/release disposition.

### Test Results and Evidence

The first immutable replacement log, `logs/session45-final-verification-v3.log`, is retained as
RED evidence: bytes=`236`, lines=`7`, SHA256=`7bbe56d8e43591cf30adfd2b4de59b15454d9b965b0fca9e6d5b84126049150c`,
exit=`1`. It stopped after the issue-heading check because the verifier regex used one extra
escape level. The correction was limited to that verifier predicate.

The corrected immutable GREEN log is:

| Metric | Expected | Actual | Result |
|--------|----------|--------|--------|
| Documentation contract | `9/10` | `9/10` | PASS |
| Plan adjacent duplicates | `0` | `0` | PASS |
| Future I39 revalidation scope | PASS | PASS | PASS |
| Issue headings | `I50..I58` | `I50..I58` | PASS |
| Status boundary | blocked/not qualified | exact tokens present | PASS |
| Document hash scope | `9` | `9` | PASS |
| Shell syntax | `52` files | `52` files | PASS |
| Python syntax | `35` files | `35` files | PASS |
| Hard-coded temporary templates | `0` | `0` | PASS |
| `git diff --check` | exit `0` | exit `0` | PASS |

Stable GREEN transcript: `logs/session45-final-verification-v3-green.log`, bytes=`1,934`,
SHA256=`aefb43b93d1e6da860970599ca08ceeefaf6a4170c9b5e9c0362775f2594f5cd`, exit=`0`.

The old `logs/session45-final-verification.log` hash remains only in historical paragraphs; it is
not reused as the current identity. No source code, test assertion, threshold, fallback,
provenance rule, GPU/RJob action, publication, commit, push, reset, `rm`, `mv`, or submodule
mutation occurred. Gate B1 remains `BLOCKED`; `real_pre_dataset` and `release_pre_dataset` remain
`NOT QUALIFIED`; `AE-ready` remains `NO`.
