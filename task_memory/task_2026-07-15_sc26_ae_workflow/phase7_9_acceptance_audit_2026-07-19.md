# Phase 7/9 AE Documentation and Acceptance Audit

## Modification History

| Date | Summary of Changes |
|------|--------------------|
| 2026-07-19 | Added Session 43 superseding reconciliation for the canonical aggregate report, final local regression, summary inventory, and unchanged external qualification blocks |
| 2026-07-19 | Audited the nine public entries, README evidence boundaries, paper suggestion traceability, Phase 8/9 qualification prerequisites, and local documentation regression results |

## Audit scope and evidence boundary

This is an independent, read-only-first audit of the documentation and final-acceptance
requirements in:

- `task_memory/task_2026-07-15_sc26_ae_workflow/requirements.md`
- `task_memory/task_2026-07-15_sc26_ae_workflow/harness.md`
- `task_memory/task_2026-07-15_sc26_ae_workflow/plan.md` (Phase 7--9)
- `SC26-AE/README.md`
- `task_memory/task_2026-07-15_sc26_ae_workflow/tex_change_suggestions.md`
- `task_memory/task_2026-07-15_sc26_ae_workflow/summary.md`

The local documentation regression uses deterministic repository fixtures only. Its evidence
class is:

```text
local_synthetic_not_gpu_qualification
```

This audit does not promote any local result to a real H800 qualification, a reusable
`release_pre_dataset`, or `AE-ready` status.

## Requirement-to-evidence matrix

### Phase 7 — README and paper suggestions

| Requirement | Authoritative evidence inspected | Current result | Boundary / action |
|-------------|----------------------------------|----------------|-------------------|
| System/image identity and pinned commits | `SC26-AE/README.md` “Repository and source identities”; `git ls-tree HEAD`; `git submodule status --recursive` | **PASS locally** | README now records main `c217ce93156e7c37e065da2989c1a482f12ecebc`, Echo `1390b4416ded08bc1b9cd0620d329d81d4470bf9`, sim-engine `39755169f73f6c748e8d7376c3a2158c6569436b`, and collective-sim `6e06e3f5140cd4e2e7c12a35586ebcdc0f410df0`. The image digest remains pending. |
| Current image is `v1.2-ae`, not historical `v1.1` | README qualification section and setup table | **PASS as documentation** | Tag is recorded; immutable digest and clean-container qualification remain external blockers. |
| Task1 hardware/topology and fixed execution policy | README hardware table, Task1 contract (`9/9`) | **PASS locally** | Full real-rank evidence is still required for release; QUICK remains smoke-only. |
| Task2 exactly two GPUs and fixed interpreter | README Task2 section, Task2 integration/interpreter tests | **PASS as contract** | Synthetic test proves schema only; real exact-two-H800 run remains unqualified. |
| Task3 CPU prebaked path, `LOCAL_SIZE=8`, analytical backend, overlap on | README Task3 command/flag sections, Task3 integration (`6/6`), portability (`10/10`) | **PASS locally** | CPU fixture values are not model-performance claims. |
| All nine public commands are directly discoverable | Nine files under `SC26-AE/`, README complete matrix, docs contract (`PUBLIC_ENTRY_COUNT=9`) | **PASS locally** | No dispatcher is used. The matrix lists all nine scripts explicitly; Task3 fresh/prebaked source selection remains explicit. |
| QUICK semantics and no extrapolation | README Task1 section and `plan.md` I1/D8 rows | **PASS as documentation** | QUICK is explicitly a four-rank smoke subset for MoE; no full qualification claim. |
| Exact Task3 report fields and overlap semantics | README exact field block; Task3 validator and integration fixture | **PASS locally** | Field names are exact; `comp+comm` remains diagnostic-only. |
| Output tree, manifest, checksums, separate identities | README output/provenance section; artifact manifest tests (`29` pytest tests in the latest local run) | **PASS locally** | A complete real portable distribution is still missing. |
| Canonical distribution path and explicit fetch behavior | `plan.md` D21/Phase 6; packaging work is audited separately | **PENDING** | The measured `regular_git` versus `github_release` decision and complete three-model bundle are not yet real-qualified. README must be updated with the final measured path before release. |
| Fail-fast/no-fallback and evidence boundary | README fail-fast section, Task3 portability/provenance tests, D30 rules | **PASS locally** | No assertions, thresholds, provenance checks, or real-vs-synthetic boundaries were weakened. |
| Paper suggestions contain copied old wording, replacement, evidence path, reason | `tex_change_suggestions.md`, docs contract (`PAPER_SUGGESTION_COUNT=10`) | **PASS locally** | The source paper remains untouched; the user applies suggestions after real evidence. |

### Phase 8 — GPU dry-run and clean-clone rehearsal

| Requirement | Evidence inspected | Current result | What is still required |
|-------------|-------------------|----------------|------------------------|
| Current `v1.2-ae` immutable image digest and fixed runtime/toolchain | `plan.md` Phase 8.1, `container_dependency_inventory.md`, existing setup reports | **BLOCKED** | Resolve the digest and run the complete clean-container interpreter/CUDA/NVML/Nsight qualification. |
| Exact-two-H800 predict-only then live Task2 gate | Gate B1 reports and current issues ledger | **BLOCKED externally** | Quota/resource semantic gate must pass; a CLI exit `0` alone is insufficient. |
| Real Task1 captures for GPT/Qwen/DeepSeek | Phase 8.2 checklist and current summary | **NOT QUALIFIED** | Need source-bound traces, memory JSON, Nsight/SQLite where enabled, full-rank coverage, and checksums for each model. |
| Real Task2 predictor and numeric dataset | Phase 8.2/Task 3.1 requirements and Task2 reports | **NOT QUALIFIED** | Need one clean exact-two-GPU run, predictor metrics, reload parity, source snapshot identity, and portable bundle. |
| Real Task3 reports for all models | Phase 8.2 checklist and Task3 report contract | **NOT QUALIFIED** | Need verified real Task1 + Task2 inputs and successful real or qualified prebaked simulation. |
| Nine-entry clean-clone replay | Phase 8.3 checklist; clean-clone lane report (when available) | **PENDING/BLOCKED** | Requires approved publication SHA/assets and an immutable image; no local fixture can substitute. |
| CPU RSS and tested allocation evidence | Phase 8.1/8.2 and synthetic Task3 report | **LOCAL ONLY** | Current synthetic peak RSS is approximately `51,308--51,356 KiB` (`0.0489--0.0490 GiB`) with a `32 MiB` fixture allocation; these are wiring values, not a release memory minimum. |

### Phase 9 — final validation, review, and archive

| Requirement | Evidence inspected | Current result | Gap |
|-------------|-------------------|----------------|-----|
| Targeted unit/integration/e2e/syntax regression | Local logs and affected suites | **PASS locally** | Latest focused evidence below; all results remain synthetic/control-plane evidence. |
| Final report at the plan's canonical path | `plan.md` §18.1 requires `test_report_2026-07-15_sc26_ae_workflow.md`; directory listing | **MISSING** | Component reports exist, but the canonical aggregate report is not yet present. Parent integration should create it with local metrics plus explicit real-gate blocks; do not label it a qualification pass. |
| Independent review records required fields | `review.md` entries and current agent reviews | **PARTIAL** | Existing review history is extensive, but the final Phase 9 reviewer sign-off remains pending until Phase 8 evidence is complete. |
| Summary inventory with exact paths and SHA256 | `summary.md` and current file hashes | **PARTIAL / STALE AFTER THIS AUDIT** | Summary contains historical/addendum inventories, but the newly audited README, tex suggestions, docs test, and packaging/clean-clone artifacts need final hashes after integration. |
| Lessons/open-items preserve unresolved gates | `lessons.md`, `future.md`, summary status | **PASS as boundary** | The task remains `INCOMPLETE`, real pre-dataset `NOT QUALIFIED`, and `AE-ready=NO`. |

## RED → GREEN documentation repair

### Observed RED

The new documentation contract was intentionally run before the documentation repair:

```text
DOC_CONTRACT_FAIL: README entry task3_gpt175b.sh is missing: SC26-AE/task3_gpt175b.sh
exit=1
```

The README described the Task3 examples as “the same two commands apply” instead of listing all
nine public commands, and it did not expose the concrete current source pins or exact report
field names. The paper suggestion entries also lacked machine-auditable `Evidence path` and
`Reason` labels.

### Root cause

The implementation had the nine entry files and runtime contracts, but Phase 7 documentation
was less explicit than the Phase 7 acceptance checklist. This was a documentation/test-seam gap,
not evidence that the missing real qualification had passed.

### Minimal repair

The repair is limited to AE-facing documentation and its unit contract:

1. Added a complete nine-entry command matrix to `SC26-AE/README.md`.
2. Added current source/gitlink identities and an explicit pending image-digest boundary.
3. Added exact Task3 JSON field names and explicit local evidence status.
4. Added `Evidence path` and `Reason` labels to all ten paper suggestions without modifying
   `sc26-ad.tex`.
5. Added `tests/unit/test_sc26_ae_docs_contract.sh`, which checks file existence/executability,
   command discoverability, full SHA-1 identities, no-fallback/evidence wording, report fields,
   and paper-suggestion traceability.

No acceptance threshold, assertion, checksum rule, provenance rule, source-selection rule, or
real qualification criterion changed.

### GREEN evidence

```text
DOC_CONTRACT_STATUS=PASS
PUBLIC_ENTRY_COUNT=9
PAPER_SUGGESTION_COUNT=10
```

Additional local checks:

| Check | Result | Numeric evidence |
|-------|--------|------------------|
| Documentation contract | PASS | `1/1`, exit `0` |
| Documentation test shell syntax | PASS | `1/1`, exit `0` |
| Public entry shell syntax | PASS | `9/9` entries; `15` shell files checked; exit `0` |
| Task1 affected contract | PASS | `9/9`, exit `0` |
| Task2 affected contract | PASS | `3/3` model attachments, manifest files=`13`, exit `0` |
| Task3 affected contract | PASS | `6/6`, evidence class=`local_synthetic_not_gpu_qualification`, exit `0` |
| Diff hygiene | PASS | `git diff --check`, exit `0` |

Artifact hashes at the time of this audit:

| Artifact | Bytes | SHA256 |
|----------|------:|--------|
| `SC26-AE/README.md` | `13,518` | `2838cf2fdc45ad5ce50156b646584eaebbc6e1f2143f62d8a915fe4caeaa05dd` |
| `task_memory/task_2026-07-15_sc26_ae_workflow/tex_change_suggestions.md` | `11,351` | `2cb43f632b7350e85f5553a5fd839e40e209a9ea6dd5eec78d75eac1106e3de7` |
| `tests/unit/test_sc26_ae_docs_contract.sh` | `4,407` | `eb1aa16328722714eeafb36cb9b61c4af1e3f0f2a2819538a1f67c617eff1527` |

## Final disposition for parent integration

This audit closes the local Phase 7 documentation contract only. It does **not** close Phase 6
packaging, Phase 8 real GPU/clean-clone qualification, or Phase 9 final release evidence. The
parent agent should integrate the three artifacts above, rerun the aggregate regression after
packaging/clean-clone lanes settle, update the canonical Phase 9 report and summary hashes, and
retain:

```text
Echo exact-two-H800 = BLOCK until semantic resource evidence passes
real_pre_dataset = NOT_QUALIFIED
AE-ready = NO
task = INCOMPLETE
```

## Session 43 Superseding Reconciliation — 2026-07-19

This append-only section corrects only the **current-state interpretation** of the earlier matrix.
The original audit rows and their checkpoint evidence remain intact for auditability.

### Phase 9 document status

The earlier row 67 said that the canonical aggregate report was missing. That was true at the time
of the first audit, but it is no longer current. The required path now exists:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-15_sc26_ae_workflow.md
```

It contains the required environment, reproducible commands, validation criteria, PASS/BLOCK counts,
actual numeric metrics, evidence boundary, open items, and a Session 43 superseding section. The
latest detailed regression report is:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-19_sc26_ae_final_regression.md
SHA256=dd9f9bef66a7755b269446aebd8ab79af2bfc158791eb0d072c5c1b6b8dba9a5
```

Accordingly, the current result for the canonical-report requirement is **PASS LOCALLY WITH REAL
RELEASE GATES OPEN**, not `MISSING`.

### Current inventory and review interpretation

- Session 43 summary inventory verification is current: `20` listed rows were rehashed and passed;
  the `progress.md` row was corrected to `153,299` bytes and SHA256
  `2b13049370677edf640e61d2cf15f9ebebc6287a78d83c2a1294f9c97214f5f2`.
- The current-success marker audit covered `7` markers (`4` fresh and `3` prebaked) and found
  alias mismatch count `0`; every marker had `verified=true` and equal manifest checksums.
- The earlier summary `PARTIAL/STALE AFTER THIS AUDIT` wording is historical. The summary now
  carries superseding hash rows and explicitly excludes its own self-referential final hash.
- Review remains **PARTIAL for release**: the independent code-review lane returned `REQUEST
  CHANGES` because external issuer authentication is unresolved, and the independent architecture
  lane is unavailable. A primary-lane check is not represented as independent approval.

### Current Phase 7--9 disposition

| Requirement | Current result | Evidence boundary |
|---|---|---|
| Phase 7 documentation contract | PASS locally | `DOC_CONTRACT_STATUS=PASS`, entries=`9`, suggestions=`10` |
| Phase 9 canonical aggregate report | PASS locally | canonical report plus Session 43 supersession |
| Phase 9 local regression/static gate | PASS locally | final report: pytest=`65`, shell=`52`, Python=`35` |
| Phase 8 immutable image/exact-two-H800 | BLOCKED | D45 semantic quota `gpu : 129/128`; controller lacks worker dependencies |
| Real 3-model × 3-task pre-dataset | NOT QUALIFIED | no complete source-bound real chain |
| Release/architecture governance | BLOCKED/PARTIAL | CR-01 issuer authentication open; architecture approval unavailable |

The task remains `INCOMPLETE`; `real_pre_dataset` and `release_pre_dataset` remain `NOT QUALIFIED`,
and `AE-ready=NO`.

