# Test Report: I54 Qualified-Evidence Predicate Consistency

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Recorded the deterministic RED→GREEN repair for the Task2 canonical qualified-evidence predicate and the affected local regression |
| 2026-07-20 | Added a fresh post-documentation regression transcript with environment, syntax, and documentation-contract evidence |

## Scope and Safety Boundary

This report covers one local Task2 control-plane defect: the canonical verifier rejected the
terminal evidence value `real_exact_two_h800_qualified` even though the mode-specific reuse
validator accepted it. The repair does **not** create, authenticate, or promote a real two-H800
qualification. All fixtures and outputs in this report are synthetic/controller evidence with
evidence class `local_synthetic_not_gpu_qualification`.

## Test Script Information

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Production file: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/lib/task2_echo.sh`
- Integration script: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task2_contract.sh`
- Unit scripts:
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_evidence_mode.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_snapshot.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_interpreter_contract.sh`
- Evidence logs:
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i54-qualified-verify-red-20260720.log`
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i54-qualified-verify-green-20260720.log`
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i54-affected-regression-20260720.log`
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/session48-task2-regression-20260720.log`
- Environment: Python `3.12.3`, pytest `9.1.1`, PyTorch `2.5.1+cu124`; controller reports
  `torch.cuda.is_available()=False` and `torch.cuda.device_count()=0`.

### Reproducible commands

Current (post-repair) regression:

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717
bash tests/unit/test_sc26_ae_task2_evidence_mode.sh
bash tests/unit/test_sc26_ae_task2_snapshot.sh
bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
```

The RED command was the same integration contract executed against the pre-repair working tree,
before `real_exact_two_h800_qualified` was added to the canonical allowlist. Its preserved output
is the immutable RED log listed above; rerunning the current tree is expected to be GREEN.

## Validation Criteria

1. Before the repair, a manifest with `execution_evidence=real_exact_two_h800_qualified` must
   reach canonical verification and fail with the known invalid-evidence error (RED).
2. After the repair, the same terminal evidence state must be accepted by canonical verification
   (GREEN), while pending and synthetic evidence rules remain unchanged.
3. Existing Task2 negative contracts (dirty source, excluded tracked paths, marker/pointer escape,
   identity mismatch, checksum alias mismatch, and unverified pointer) must remain enforced.
4. The affected unit and integration commands must exit successfully without changing release
   labels or evidence classes.

## Root-Cause and Change Record

### Root cause

`task2_validate_reuse_evidence()` and the embedded Python predicate in `task2_verify_run()` had
different accepted terminal-state sets. This produced a deterministic split-brain path: reuse
precheck success followed by canonical verification failure.

### Minimal change

`SC26-AE/lib/task2_echo.sh` now includes `real_exact_two_h800_qualified` in the canonical verifier's
accepted terminal set. The mode-specific predicate remains strict: real mode still requires that
value, synthetic mode still accepts only its explicit local/qualified values, and pending evidence
is not accepted for real reuse. No fallback or source switching was added.

## Test Results and Evidence

| Test/evidence | Result | Numeric details |
|---|---|---|
| Pre-repair qualified verification | RED | exit `1`; `MANIFEST_STATUS=verified`; `MANIFEST_FILE_COUNT=13`; `QUALIFIED_VERIFY_STATUS=rejected` |
| Pre-repair root error | Expected RED | exact text: `Task2 artifact manifest execution evidence is invalid` |
| Post-repair qualified verification | GREEN | exit `0`; `MANIFEST_STATUS=verified`; `MANIFEST_FILE_COUNT=13`; `QUALIFIED_VERIFY_STATUS=accepted` |
| Evidence-mode unit | PASS | `PASS_COUNT=4` |
| Snapshot unit | PASS | exit `0`; both intended negative cases remained rejected |
| Interpreter contract | PASS | `PASS_COUNT=11` |
| Task2 integration | PASS | exit `0`; `7` `PASS:` lines in the qualified/negative/snapshot section |
| Fresh post-documentation regression | PASS | exit `0`; `git diff --check=PASS`; changed-shell syntax `PASS`; docs public entries=`9`; paper suggestions=`10` |

### Evidence identities

```text
i54-qualified-verify-red-20260720.log bytes=464 sha256=f41384926a0221f221bc6a3d9dcff9ad70d7fd51876fe44061fe14b8a3b3b413
i54-qualified-verify-green-20260720.log bytes=1050 sha256=2172ad1e26d40c7c6e5ffad77a460524b3a22f15d8e62900d87317a0025592a7
i54-affected-regression-20260720.log bytes=2403 sha256=731192731ce4cb21e0e5b97e705509327961d0d460094063d75ac181fefe2a71
session48-task2-regression-20260720.log bytes=2725 sha256=eb3fed5ebe37ee498eee37347285caa2ac1b640a4e786a476f9d76c0dd14da79
```

Representative GREEN output:

```text
PASS: canonical Task2 verification accepts qualified evidence
PASS: Task2 rejects an intermediate symlink escape in a model marker
PASS: Task2 rejects an intermediate symlink escape in a shared pointer
PASS: Task2 rejects a model marker whose predictor identity differs from its run path
PASS: Task2 rejects a shared pointer with a mismatched artifact_manifest_sha256 alias
PASS: Task2 rejects an unverified shared predictor pointer before attachment
PASS: Task2 snapshot-only execution, predictor identity, and provenance contract
```

## Residual Issues and Interpretation

- This is a local predicate-consistency repair only. It does not prove that any real
  `real_exact_two_h800_qualified` manifest exists or is externally authenticated.
- I54 remains `PARTIAL / OPEN` because the qualified shared-pointer lifecycle, destination
  identity, and cross-file provenance state machine are not frozen.
- I55 remains `OPEN / HIGH/BLOCK`; a separate read-only synthetic probe showed nested interpreter
  escape (`EXTERNAL_INVOCATIONS=1`, `NESTED_SENTINEL=created`, `PROBE_RC=1`) and was not used here as
  qualification evidence.
- Global status remains `INCOMPLETE`; Gate B1 remains `BLOCKED`; `real_pre_dataset` and
  `release_pre_dataset` remain `NOT QUALIFIED`; `AE-ready=NO`.
