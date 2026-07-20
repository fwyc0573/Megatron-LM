## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Superseded the stale Task1 `PASS_COUNT=11` clone assertion with the current `31`-case contract and recorded fresh clean-clone GREEN evidence |
| 2026-07-19 | Added local clean-clone-style replay evidence for the setup boundary, all nine public entries, and one synthetic Task1→Task2→Task3 chain |

# Test Report: SC26 AE Local Clean-Clone-Style Replay

**Date:** 2026-07-19  
**Worktree:** `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`  
**Evidence class:** `local_synthetic_not_gpu_qualification`

## Scope and root cause

The Phase 8 audit initially found no executable clean-clone/replay harness in the
AE surface. Existing tests covered Task1/Task2 contracts and Task3 portability,
but they ran from the current worktree and did not prove that all nine public
entries survive an isolated clone with clean pinned producers.

The workflow defect was a missing rehearsal seam, not a failed acceptance
threshold. The minimal repair was to add
`tests/e2e/test_sc26_ae_clean_clone_replay.sh`. It creates an isolated outer
clone, populates the pinned Echo, sim-engine, and nested collective-sim commits
from local object stores (no network fallback), copies only the evaluator-facing
AE/test surface (not ignored captures), commits that surface as an ephemeral
synthetic overlay, and executes the existing contract/e2e tests from a separate
outside CWD. It asserts clean outer and nested Git status before and after
replay. No production threshold, checksum rule, provenance rule, source choice,
or evidence label was weakened.

## Test Script Information

- Script: `tests/e2e/test_sc26_ae_clean_clone_replay.sh`
- Exact command:

  ```bash
  bash tests/e2e/test_sc26_ae_clean_clone_replay.sh
  ```

- Validation log: `task_memory/task_2026-07-15_sc26_ae_workflow/logs/clean-clone-replay-20260719T191118Z.log`
- Environment: host `/usr/bin/bash`, Python `python3` (synthetic fixtures); no GPU, RJob, container, or network qualification was attempted.
- The harness's ephemeral replay root was:
  `/tmp/sc26-ae-clean-clone-replay.3hKtyq`

## Validation Criteria

1. The isolated outer clone and all three pinned producer repositories are clean
   before and after tests.
2. The outer gitlinks equal the expected Echo and sim-engine commits; the nested
   collective-sim commit is also pinned and clean.
3. The current ignored `SC26-AE/output/` captures are not copied into the clone.
4. Setup's explicit source contract passes all six cases without executing the
   real installer.
5. All three Task1, all three Task2, and all three Task3 public entries are
   exercised from outside the repository CWD.
6. One fresh synthetic Task1→Task2→Task3 chain completes with verified markers,
   manifests, checksums, and report fields.
7. Synthetic evidence remains explicitly non-qualification evidence.

## Test Results and Evidence

| Suite / invariant | Result | Numeric evidence |
|---|---|---:|
| Setup public contract | PASS | `6/6` cases; real installer execution count `0` |
| Task1 public entries | PASS | `3/3` entries; contract `PASS_COUNT=9` |
| Task2 public entries | PASS | `3/3` entries; isolated snapshot/predictor contract PASS |
| Task3 prebaked public entries | PASS | `3/3` entries; `MODEL_PASS_COUNT=3` |
| Fresh atomic chain | PASS | `1/1` chain |
| Outer clone status | PASS | `0` porcelain lines before/after replay |
| Echo-slowdown status | PASS | `0` porcelain lines before/after replay |
| Sim-engine status | PASS | `0` porcelain lines before/after replay |
| Nested collective-sim status | PASS | `0` porcelain lines before/after replay |
| Source evidence boundary | PASS | `EVIDENCE_CLASS=local_synthetic_not_gpu_qualification` |

The harness exited with code `0`. The final log excerpt is:

```text
PUBLIC_TASK1_ENTRY_COUNT=3
PUBLIC_TASK2_ENTRY_COUNT=3
PUBLIC_TASK3_ENTRY_COUNT=3
SETUP_CONTRACT_CASE_COUNT=6
FRESH_ATOMIC_CHAIN_COUNT=1
OUTER_CLONE_STATUS=clean
ECHO_SUBMODULE_STATUS=clean
SIM_ENGINE_STATUS=clean
COLLECTIVE_SIM_STATUS=clean
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
PASS: isolated clean-clone-style nine-entry synthetic replay completed.
```

## RED → GREEN record

- **RED / audit observation:** before this change, a repository search found no
  `clean-clone`, `distribution`, or `fetch_prebaked` replay harness under
  `SC26-AE` or `tests`; Phase 8.2/8.3 remained unexercised locally.
- **Root cause:** no isolated source-snapshot runner bound the public entries to
  clean outer/submodule state and an outside working directory.
- **Minimal fix:** add one e2e harness only; reuse existing contract fixtures and
  preserve all fail-fast checks. No fallback, threshold change, or real-data
  substitution was introduced.
- **GREEN:** the command above completed all setup/task suites and clean-state
  assertions with exit `0`; shell syntax also passed:

  ```bash
  bash -n tests/e2e/test_sc26_ae_clean_clone_replay.sh
  git diff --check -- tests/e2e/test_sc26_ae_clean_clone_replay.sh
  ```

## Boundary and remaining blockers

This report proves only local synthetic portability and control-flow behavior. It
does **not** prove:

- public default-branch or Release fetchability;
- immutable `v1.2-ae` image digest or clean-container qualification;
- exact-two-H800 Echo qualification;
- real Task1 traces, real Task2 slowdown data, or real Task3 reports;
- a release-qualified three-model × three-task pre-dataset.

Accordingly, `real pre-dataset=NOT QUALIFIED`, `release_pre_dataset=NOT
QUALIFIED`, and `AE-ready=NO` remain unchanged.

## Superseding replay addendum — 2026-07-19

### RED and root cause

After the Task1 semantic and memory-negative expansions, the current contract emits `PASS_COUNT=31`.
The first continuation replay ran all five internal cases successfully but exited `1` because the
harness still grepped for `PASS_COUNT=11`. The immutable RED log is
`logs/task3-clean-clone-followup-20260719.log` (910 bytes, SHA256
`703ff0f937529c5afe65a1028f989978298f49d389f587b27cfd0ea83daf592e`).

### Minimal GREEN repair

The only harness change was the expected-count literal in
`tests/e2e/test_sc26_ae_clean_clone_replay.sh`. The corrected replay log
`logs/task3-clean-clone-followup-green-20260719.log` is 1,385 bytes,
SHA256 `1a19a88e525ca602fa888892295908cea56a85cc31929e564272cb061e9cde9c`, and exits `0` with:

- public Task1/Task2/Task3 entries `3/3/3`;
- setup contract cases `6` and fresh atomic chain `1`;
- outer clone, Echo-slowdown, sim-engine, and nested collective-sim statuses all `clean`;
- `EVIDENCE_CLASS=local_synthetic_not_gpu_qualification`.

This report supersedes only the stale count assertion and does not convert the replay into public
release, exact-two-H800, or AE-ready evidence.
