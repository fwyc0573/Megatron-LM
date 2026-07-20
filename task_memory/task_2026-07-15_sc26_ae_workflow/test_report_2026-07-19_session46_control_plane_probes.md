# Test Report: Session 46 Control-Plane Behavioral Probes

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Recorded read-only reproductions of the Task2 shared-pointer symlink containment gap and qualified-evidence lifecycle contradiction; no production or release status change |

## Test Script Information

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Probe roots (intentionally retained; no destructive cleanup was performed):
  - `/data/ycfeng/sc26-ae-test-tmp/session46-task2-symlink-probe-20260719`
  - `/data/ycfeng/sc26-ae-test-tmp/session46-task2-qualified-lifecycle-probe-20260719`
- Source under test: `SC26-AE/lib/task2_echo.sh`
- Supporting verifier: `SC26-AE/tools/artifact_manifest.py`
- Environment: `/usr/bin/python3` 3.12.3; local CPU controller; no CUDA, RJob, Docker, or external issuer
- Execution style: isolated synthetic fixtures; no repository files or submodules were changed

### Reproducible commands

The first probe sourced the current Task2 functions through the `task2_main` boundary, created a
lexically safe `_shared/task2/runs/<predictor_run_id>` symlink to a run outside the probe output
root, and invoked `task2_run_reuse`. The exact captured output is:

```text
session46-task2-symlink-probe-20260719/probe-v5.log
PROBE_RC=1
MANIFEST_STATUS=verified
MANIFEST_FILE_COUNT=13
Traceback (most recent call last):
  ...
ValueError: '<external run path>' is not in the subpath of '<probe output root>'
```

The second probe sourced `task2_validate_reuse_evidence` and evaluated the exact evidence predicate
embedded in `task2_verify_run` against a manifest containing
`execution_evidence=real_exact_two_h800_qualified`.

## Validation Criteria

1. A Task2 shared pointer must reject a symlink or any resolved path outside the canonical output
   root **before** opening manifest, provenance, or metrics files.
2. The same evidence state must be accepted consistently by reuse precheck and canonical manifest
   verification; a qualified evidence label must not be accepted by one stage and rejected by the
   next.
3. No probe may promote synthetic data, alter acceptance thresholds, add fallback behavior, or
   change `Gate B1`, `real_pre_dataset`, `release_pre_dataset`, or `AE-ready` status.

## Test Results and Evidence

| Probe | Result | Observed numeric/status evidence |
|---|---|---|
| Shared-pointer symlink containment | **FAIL (expected finding reproduction)** | `PROBE_RC=1`; manifest verifier reached `MANIFEST_STATUS=verified`; `MANIFEST_FILE_COUNT=13`; final rejection occurred only when `Path.relative_to(output_root)` saw the external target |
| Qualified lifecycle consistency | **FAIL (expected finding reproduction)** | `REUSE_VALIDATOR_RC=0`; `VERIFY_EVIDENCE_PREDICATE_RC=1`; verifier message `Task2 artifact manifest execution evidence is invalid` |

### Artifact identities

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Symlink probe report | 1,970 | `0fe8e03b677e071d327f8f021140693a241273954ee2099200738db98cf9772f` |
| Symlink probe raw log | 529 | `74c73cd35a31a53ceb0eb76a60c05dd3cee02d38bd500441a8ce78192c3e9285` |
| Qualified lifecycle probe report | 1,300 | `fd47157747bc5d19a4886a82d7c76e974b645340187aa090c9df3cbd0dcd7255` |
| Qualified lifecycle result log | 250 | `86a53e1c8af9eae68e5eae8a65cb218bc1e4660cda8e6f5dd6b99728401c221f` |

## Root-Cause Analysis

- **I56/F10-09:** The shared-pointer consumer performs lexical `..` checks and then constructs a
  path from `output_root.resolve() / rel`. It does not reject a symlink component or establish
  resolved containment before reading the pointed-to bundle. The later marker writer rejects the
  external path, but too late to serve as a trusted-root gate.
- **I54/F10-05:** `task2_validate_reuse_evidence()` accepts
  `real_exact_two_h800_qualified`, while the inline canonical-manifest predicate in
  `task2_verify_run()` accepts only raw/pending evidence classes. The two stages therefore cannot
  form a qualified reuse chain.

## Disposition

These are deterministic control-plane findings, not successful qualification results. The probes
do not authorize a local patch to the I54/I56 contracts: the current handoff requires an
owner-approved evidence-state/trusted-path design before changing any I51–I58 production contract.
The release boundary remains:

```text
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## Recommended next steps

1. Approve one evidence-state machine and one canonical trusted-path seam for I54/I56.
2. Implement the smallest RED→GREEN changes and negative tests under that approved design.
3. Re-run the affected full local matrix, then request independent review before any release-status change.
