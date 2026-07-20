# Test Report: Session 47 V21 Strict Documentation Verifier

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Added the penultimate exact-log tracked-snapshot PASS, command-transport interruption diagnosis, and local I59 closure evidence |
| 2026-07-20 | Added Session 56 exact-log staging, runtime-output scope RED→GREEN, tracked-snapshot V21, and independent approval evidence |
| 2026-07-19 | Added the final document-hash refresh and independently verified the final V21 identity block |
| 2026-07-19 | Added the fail-fast V21 verifier evidence, inventory reconciliation, and post-append status-boundary checks |

## Test Script Information

- Shell script: `tests/integration/test_sc26_ae_v21_verifier.sh`
- Python checker: `tests/integration/sc26_ae_v21_verifier.py`
- Documentation contract: `tests/unit/test_sc26_ae_docs_contract.sh`
- Environment: Python 3.12.3; pytest 9.1.1; PyTorch 2.5.1+cu124; controller `CUDA_AVAILABLE=False`,
  `CUDA_DEVICE_COUNT=0`; NVIDIA driver and grouped-gemm runtime unavailable.
- Commands:

  ```bash
  bash tests/integration/test_sc26_ae_v21_verifier.sh --expected-status VERIFIER_PENDING \
    > task_memory/task_2026-07-15_sc26_ae_workflow/logs/session47-final-verification-v21-preappend-20260720-final.log 2>&1
  bash tests/integration/test_sc26_ae_v21_verifier.sh --expected-status PASS \
    > task_memory/task_2026-07-15_sc26_ae_workflow/logs/session47-final-verification-v21.log 2>&1
  bash tests/integration/test_sc26_ae_v21_verifier.sh --expected-status PASS \
    > task_memory/task_2026-07-15_sc26_ae_workflow/logs/session47-final-verification-v21-post-append-20260720.log 2>&1
  ```

## Validation Criteria

1. Parse exactly one uniquely marked V21 artifact table (`7` rows) and one authoritative
   non-self-referential document table (`10` rows).
2. Recompute every listed byte count and SHA256; resolve and verify all `17` supplemental
   identities.
3. Require issue headings `I50` through `I58`, the explicit `I56 = PARTIAL / OPEN` marker, and
   the unchanged `INCOMPLETE` / Gate B1 / pre-dataset / `AE-ready` boundary.
4. Run the public documentation contract (`9` entries and `10` paper suggestions).
5. Re-run the established static scope: shell syntax `52/52`, Python syntax `35/35`, production
   temporary-root matches `0`, and `git diff --check=PASS`.
6. Confirm numeric synthetic evidence without relabeling it as GPU qualification: traces/memory,
   MSEs, reload delta, timing values, simulator wall time, and the six CUDA-only unit failures.
7. Prove fail-fast behavior by requiring a non-zero exit when an intentionally wrong expected
   status is supplied.

## Test Results and Evidence

| Check | Result | Evidence |
|---|---:|---|
| Pre-append V21 verifier | PASS, exit `0` | `logs/session47-final-verification-v21-preappend-20260720-final.log`, `6162` bytes, SHA256 `c255ac762e12126325b97061b0e6d7ba0f2481a2154d3af9a7131075c4c409c7` |
| Intentional wrong-status fail-fast probe | PASS (expected RED), exit `1` | `logs/session47-final-verification-v21-failfast-red-20260720.log`, `5568` bytes, SHA256 `be7ee9e2a285852567d4d9aa62eeb46a82a613e2940b0f37fd152b02aa3e566c` |
| V21 verifier after initial status append | PASS, exit `0` | `logs/session47-final-verification-v21.log`, `6150` bytes, SHA256 `f8480ee9178bd2bbc56c528a42b7c02658efdb77949446104fddd6157594b17a` |
| Final V21 verifier after document-hash refresh | PASS, exit `0` | `logs/session47-final-verification-v21-final-20260720.log`, `6312` bytes, SHA256 `b53246fd285fda1b6df0c167c4e1cc979ec51c9db2d40368ad47d3a7a3ae0f27` |
| Read-only post-final verifier | PASS, exit `0` | `logs/session47-final-verification-v21-post-final-20260720.log`, `6327` bytes, SHA256 `8f91ce2f5ea0593eba94cd1993b74fd54a8848aef161a17d1568093ef9144c91` |

### Key Numeric Metrics

| Metric | Expected / acceptance | Observed |
|---|---:|---:|
| V21 artifact rows | `7` | `7` |
| V21 authoritative document rows | `10` | `10` |
| Supplemental identities | all listed | `17` verified |
| Public documentation entries | `9` | `9` |
| Paper suggestions | `10` | `10` |
| Shell static scope/syntax | `52/52` | `52/52` |
| Python static scope/syntax | `35/35` | `35/35` |
| Production temporary-root matches | `0` | `0` |
| Task1 trace / memory files | `4/4` | `4/4` |
| Task2 dataset rows | `2` | `2` |
| Average validation MSE | numeric, synthetic-only | `3.0` |
| Test MSE | numeric, synthetic-only | `0.5` |
| Model reload max absolute prediction delta | `0.0` | `0.0` |
| Task3 rank0 step | synthetic diagnostic | `22.5 ms` |
| Forward / backward / optimizer | synthetic diagnostics | `6.0 / 11.0 / 2.5 ms` |
| Simulator wall time | synthetic diagnostic | `0.5 s` |
| Full unit CUDA-only failures | environment limitation retained | `6` (with `113` non-GPU passes) |

## Evidence Boundary

All outputs in this report are local synthetic/controller evidence. The verifier does not qualify an
H800 run, grouped-gemm runtime, issuer attestation, release package, `real_pre_dataset`,
`release_pre_dataset`, or `AE-ready`. The authoritative disposition remains:

```text
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## Session 56 Test Report: V21 Clean-Clone Provenance Remediation

### Test Script Information

- Unit test: `tests/unit/test_sc26_ae_v21_scope.py`
- Integration wrapper: `tests/integration/test_sc26_ae_v21_verifier.sh`
- Python verifier: `tests/integration/sc26_ae_v21_verifier.py`
- Candidate snapshot tree: `841042300c32dc737d429feb333283fb53f7fbd0`
- Candidate snapshot repository:
  `/data/ycfeng/sc26-ae-test-tmp/session56-candidate-snapshot-green-20260720-sSHNUC`
- Environment: `CONDA_DEFAULT_ENV=none`; Python `3.12.3`; pytest `9.1.1`; controller-only,
  no GPU/RJob/Docker/network execution.
- Commands:

  ```bash
  python3 -m pytest tests/unit/test_sc26_ae_v21_scope.py -q
  bash tests/integration/test_sc26_ae_v21_verifier.sh --expected-status PASS
  git diff --cached --check
  git archive "$(git write-tree)" | tar -x -C <fresh-snapshot>
  git -C <fresh-snapshot> init
  git -C <fresh-snapshot> add -A
  git -C <fresh-snapshot> commit -m 'Create ephemeral staged-tree verification snapshot'
  bash <fresh-snapshot>/tests/integration/test_sc26_ae_v21_verifier.sh \
    --repo-root <fresh-snapshot> \
    --expected-status PASS
  ```

### Validation Criteria

1. Resolve `65` unique required files: `6` task-root reports and exactly `59` logs.
2. Require staged logs to equal the computed allowlist with missing=`0`, extra=`0`; unrelated logs
   must remain ignored and untracked.
3. Preserve immutable evidence hashes while requiring `git diff --cached --check=PASS` for the full
   staged tree and normal whitespace behavior outside the task archive.
4. Exclude ignored `SC26-AE/output/` runtime copies from source shell enumeration; retain hard
   source counts and syntax checks.
5. Observe unit RED before implementation and GREEN after the minimal collector repair.
6. Reproduce V21 from a clean Git repository exported from the staged tree.
7. Keep all local synthetic/external qualification boundaries unchanged.

### Test Results and Evidence

| Check | Result | Evidence |
|---|---:|---|
| Unit scope test before implementation | PASS as expected RED; pytest exit `1` | `/data/ycfeng/sc26-ae-test-tmp/session56-v21-scope-unit-red-20260720.log`; `1,442` bytes; SHA256 `98856c69ca61532dd032333a66bfd39b17dce2fc580fbf7e71551d7e61934548`; expected missing `collect_shell_paths` |
| Unit scope test after implementation | PASS; `1/1`, exit `0` | `/data/ycfeng/sc26-ae-test-tmp/session56-v21-scope-unit-green-20260720.log`; `136` bytes; SHA256 `d7471a0ab80a0376f9403187ced0f0d22d37abf5b6b1dc802485740c4f721f11` |
| First candidate snapshot | PASS as diagnostic RED; V21 exit `1` | `/data/ycfeng/sc26-ae-test-tmp/session56-candidate-snapshot-v21-20260720-143309.log`; `13,287` bytes; SHA256 `59dfa2a8dd7f44ebac7e5c3121f9ff817f5e0e4b94631f63f731328f8a3c948e`; observed clean scope `47` vs stale expected `53` |
| Local V21 after scope repair | PASS; exit `0` | `/data/ycfeng/sc26-ae-test-tmp/session56-v21-runtime-output-scope-green-20260720.log`; `13,529` bytes; SHA256 `8335d2ccb26531c81b121397ccf39e5af63a9988565746201dab57e0c22beb0f` |
| Candidate staged-tree snapshot V21 | PASS; exit `0` | `/data/ycfeng/sc26-ae-test-tmp/session56-candidate-snapshot-v21-green-20260720-143608.log`; `13,529` bytes; SHA256 `512972d9716e06ff50c4754bcf5155e92e825aa7618a0a88311e465d0d1b44fb` |
| Penultimate command transport | FAIL before verifier evidence | Snapshot commit was created, but `/data/ycfeng/sc26-ae-test-tmp/session56-penultimate-snapshot-v21-20260720-Z5XvbR.log` remained `0` bytes; SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; not counted as PASS |
| Penultimate exact-log snapshot V21 rerun | PASS; exit `0` | Tree `6c5cf790c62b021e1504621ae7489986a29990ec`; snapshot commit `26f89b4df53760df8c38ac9ab62bfcf4ff0d6349`; `/data/ycfeng/sc26-ae-test-tmp/session56-penultimate-snapshot-v21-rerun-20260720-iXxBlF.log`; `13,524` bytes; SHA256 `5e68330fc19eeead6eb6e1f52a046b9d7f0427e3a3cc32ffe05723c7f964ab39` |
| Candidate staged audit | PASS | required/staged logs=`59/59`; missing/extra=`0/0`; runtime-state/output/credential/secret/large violations=`0`; staged blobs=`150`; staged bytes=`2,528,112` |
| Independent follow-up review | APPROVE | `.omx/artifacts/claude-act-as-an-independent-read-only-reviewer-for-the-sc26-ae-loc-2026-07-20T06-40-51-467Z.md` |

### Key Numeric Metrics

| Metric | Expected / acceptance | Observed | Delta |
|---|---:|---:|---:|
| Unique V21 dependencies | `65` | `65` | `0` |
| Required task-root reports | `6` | `6` | `0` |
| Required / staged logs | `59 / 59` | `59 / 59` | `0 / 0` |
| Required log bytes before current-identity replacement | `228,172` | `228,172` | `0` |
| Missing / extra staged logs | `0 / 0` | `0 / 0` | `0 / 0` |
| V21 artifact / document rows | `7 / 10` | `7 / 10` | `0 / 0` |
| Supplemental identities | `64` | `64` | `0` |
| Source shell scope / syntax | `47 / 47` | `47 / 47` | `0 / 0` |
| Runtime-output shell exclusions | `>= 1` marker | `1` | `0` |
| Python scope / syntax | `36 / 36` | `36 / 36` | `0 / 0` |
| Unit tests | `1` pass | `1` pass | `0` failures |
| Candidate snapshot V21 exit | `0` | `0` | `0` |
| Penultimate snapshot V21 exit | `0` | `0` | `0` |

### Failure Root Cause and Resolution

The initial clean snapshot omitted six ignored runtime copies below
`SC26-AE/output/_work/.../source/`, while the local verifier had counted them as authored source.
The root cause was state-dependent source enumeration, not missing commit content. The fix adds a
tested collector that excludes the ignored runtime-output root and continues to hard-assert the
real source count and syntax. No runtime output was added, no historical evidence file was
rewritten, and no threshold or qualification predicate was weakened.

The later penultimate replay had one command-transport interruption after its ephemeral Git commit
was created and before verifier output was written. Process, lock, index, commit, and clean-status
diagnostics proved the snapshot itself was valid and clean. The complete verifier was rerun without
changing repository bytes and passed with exit `0`; the empty first log is retained only as failure
evidence and is not used as a successful verifier identity.

### Evidence Boundary

This Session 56 evidence proves only local tracked-tree reproducibility and resolves I59 locally.
I55 remains
`OPEN / HIGH / BLOCK`; I53 remains `OPEN / HIGH / WATCH`; I54/I56 remain `PARTIAL / OPEN`;
Gate B1 remains `BLOCKED`; `real_pre_dataset` and `release_pre_dataset` remain `NOT QUALIFIED`;
`AE-ready=NO`; workflow remains `INCOMPLETE`.
