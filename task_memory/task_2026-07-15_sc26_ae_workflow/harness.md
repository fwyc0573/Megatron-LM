# Harness — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Added the reduced two-model functional acceptance gates and Task2 non-rerun/kernel-skip invariants |
| 2026-07-19 | Added Session 45 control-plane gates for producer binding, full-rank promotion, qualified pointer publication, frozen inputs, and issuer authentication; synthetic evidence remains non-qualifying |
| 2026-07-19 | Added the D30 writable-temp-root portability invariant; test environment repair does not relax qualification gates |
| 2026-07-19 | Reconciled the current Echo/B1 resource block with D45 semantic quota evidence (`129/128`); historical D26 PASS remains historical |
| 2026-07-19 | Reconciled the current gate snapshot with the verified clean `megatron-sim-engine` producer provenance; historical dirty-source text remains historical only |
| 2026-07-19 | Added the D30 latest-user overlay: every test/validation/rehearsal failure serving the AE scripts or reusable pre-dataset may be self-repaired, while acceptance and release evidence gates remain strict |
| 2026-07-19 | Appended the D42/D43 evidence-class overlay, terminal retry accounting, provenance WATCH, and final 3x3 release block |
| 2026-07-19 | Added the task-specific D29 harness, evidence classes, self-repair contract, and current incomplete-gate snapshot |

## 1. Purpose and status

This harness defines the acceptance boundaries for the SC'26 Artifact Evaluation (AE) workflow.
The core objective is to give AE users nine model-specific, one-click shell entry points and a
reusable pre-dataset whose source, runtime, checksums, and quality evidence are reproducible.

**Overall status: INCOMPLETE.** Local controller/synthetic workflow evidence is green, but the
real qualification and release pre-dataset gates are not closed. This file is a gate contract, not
a claim that the workflow is AE-ready.

## 2. Evidence classes

| Evidence class | What it proves | What it cannot prove |
|----------------|----------------|----------------------|
| `local_synthetic_not_gpu_qualification` | Shell orchestration, schema, manifest, fail-fast, and synthetic Task1→Task2→Task3 control flow | H800 behavior, real traces, real predictor quality, or a release pre-dataset |
| `local_synthetic_not_two_gpu_qualification` | Synthetic Task2 wiring and deterministic local predictor checks | Exact-two-GPU CUDA/NCCL execution or real slowdown data |
| `real_gpu_qualification` | A fresh, source-bound run on the qualified image and required H800 topology | Publication or AE release by itself; packaging and checksum gates still apply |
| `release_pre_dataset` | A complete, portable, checksum-verified bundle with real-quality evidence | Any artifact not listed and verified by its manifest |

Evidence may only be promoted when the stronger class has been independently produced. Synthetic
metrics must never be relabeled as real qualification.

## 3. D29 autonomy boundary

D29 permits autonomous diagnosis, repair, and decision-making when the root cause is confined to an
AE-serving **test, audit, schema, validator, documentation, or control-plane orchestration**
surface. The repair must directly advance the one-click scripts or reusable pre-dataset.

Every such repair records:

1. Motivation and expected behavior;
2. Observed RED and root-cause evidence;
3. The smallest contract-preserving repair;
4. Observed GREEN;
5. Affected regression commands and exit codes;
6. Numeric evidence and the resulting evidence class.

D29 does **not** authorize:

- weakening assertions, acceptance thresholds, or data-quality requirements;
- fallback or source/version switching;
- bypassing checksum, provenance, or clean-submodule checks;
- converting local/synthetic evidence into real evidence;
- overriding real GPU, image, quota, scheduler, product/runtime/workload, security,
  destructive-operation, external-publication, or real pre-dataset failures.

## 3.1 D30 latest-user test-failure autonomy overlay

D30 supersedes the narrow scope interpretation above for the current execution. **Any error or
problem exposed by a test, validation, rehearsal, audit, or qualification check may be diagnosed,
decided, and repaired autonomously** when the work directly advances the two AE deliverables:

1. one-click model/task shell scripts; and
2. a complete, reusable, quality-qualified pre-dataset for AE users.

The defect may be in the test itself, a validator/schema/document, an orchestration/control-plane
path, a shell entry, or a task-scoped implementation path that the failing test proves is required
for those deliverables. A test failure is therefore **not** a user-approval BLOCK. The agent must
still repair the root cause rather than mask the symptom, and must rerun the affected tests before
promoting the corresponding gate.

D30 changes who may resolve a test-detected defect; it does not change what constitutes PASS. The
agent must preserve all assertions and acceptance thresholds, checksum/provenance and clean-source
checks, real-vs-synthetic evidence labels, no-fallback/source-selection rules, and data-quality
requirements. A failed real qualification or data-quality check remains an unmet gate until the
underlying problem is fixed and the check passes; it is not waived or relabeled. Actual external
resource unavailability, destructive or irreversible actions, credentials, external publication,
or a materially scope-changing refactor remain outside this local autonomy boundary.

For every D30 repair, record motivation, expectation, observed RED and root cause, minimal method,
GREEN result, affected regression commands/exit codes, numeric evidence, and the resulting evidence
class. Do not claim `AE-ready`, `real_gpu_qualification`, or `release_pre_dataset` from a local
test pass alone.

## 4. Mandatory invariants

1. Task1, Task2, and Task3 use explicit, recorded source and runtime bindings.
2. `ARTIFACT_SOURCE` is explicit (`fresh` or `prebaked`); a missing, partial, corrupt, or
   wrong-provenance selected source fails fast and never falls back.
3. A submodule's worktree must be clean when a manifest claims a pinned commit. An outer-repository
   gitlink alone is insufficient provenance.
4. Every distributed file has a size and SHA256 entry. Verified markers are written only after all
   semantic reports and manifests pass.
5. Fresh and prebaked runs use immutable, non-overlapping run roots; an existing destination is an
   error rather than an overwrite.
6. Rank0 reports contain positive, finite `forward_step`, `backward_step`, and `optimizer_step`
   values and agree with the machine-readable report.
7. Numeric claims include the compared values and the derived metric (for example MSE, delta, or
   peak RSS), not only a PASS label.
8. Historical image or run evidence remains labeled historical. The current image target is
   `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`; its immutable digest is required before
   qualification.

## 5. Current gate snapshot

| Gate | Current status | Release consequence |
|------|----------------|---------------------|
| D30 latest test-failure autonomy overlay | Synchronized | Test/validation/rehearsal defects may be self-repaired; no acceptance gate is relaxed |
| D29 governance overlay | Synchronized | Test/control-plane defects may be self-repaired with evidence |
| Local Task1 contracts | PASS (synthetic/controller) | Does not open Gate B |
| Local Task3/sim-engine contracts | PASS (synthetic/controller) | Does not open Gate B |
| D27 one-H800 MemoryTracker branch | PASS | Only one qualification component is closed |
| Echo exact-two-H800 | BLOCK | Integrated B1 remains blocked |
| Integrated Gate B1 | BLOCK | B2–B5 and implementation phases remain blocked |
| `megatron-sim-engine` provenance | CLOSED/RESOLVED | Outer gitlink and clean nested producer both equal `39755169f73f6c748e8d7376c3a2158c6569436b`; this closes source cleanliness only, not real pre-dataset qualification |
| Fresh real pre-dataset | NOT QUALIFIED | `AE-ready` cannot be claimed |

**Current resource evidence:** D45's v1.2-ae 2-GPU predict-only command returned CLI exit=`0` but
printed the authoritative quota failure `gpu : 129/128`; semantic status is `FAIL`. The Echo
exact-two-H800 and integrated B1 rows therefore remain closed. The older D26 two-GPU PASS is
historical and cannot override D45; no test change or one-GPU substitution may bypass this external
resource gate.

## 6. Verification contract

The minimum local regression set is:

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717
bash tests/integration/test_sc26_ae_task3_contract.sh
python -m pytest \
  megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py \
  megatron-sim-engine/tests/unit/test_rank0_report.py \
  megatron-sim-engine/tests/integration/test_rank0_report_integration.py \
  megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py \
  megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py \
  megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py -q
```

The 2026-07-19 report observed Task3 unit=`6/6`, integration=`6/6`, prebaked CPU e2e=`3/3`,
fresh chain=`1/1`, and sim-engine=`45/45` in `10.82 s`, all with exit code `0`. The report labels
the evidence `local_synthetic_not_gpu_qualification`.

## 7. Stop and handoff rules

- Continue autonomously for any test/validation/rehearsal defect that satisfies D30, including a
  task-scoped implementation defect proven by the failing test; use RED→root cause→minimal fix→
  GREEN→regression evidence.
- Keep the corresponding gate closed while real qualification, data quality, provenance, or
  completeness evidence is still failing; repair it rather than self-waiving it.
- Stop and escalate only for actual external resource/authority problems, destructive or
  irreversible actions, credentials, external publication, materially scope-changing refactors,
  or an ambiguity that cannot be resolved from repository facts.
- Do not mark Phase 1–9 complete while integrated B1 and real pre-dataset quality remain open.
- A final AE handoff requires clean source provenance, immutable image digest, real H800 evidence,
  complete trace/predictor/simulator artifacts, portable manifests, and checksum/data-quality
  validation.

## 8. D42/D43 superseding evidence overlay

This overlay supersedes the current-status interpretation of the older D28/B1 snapshot in Section
5 without erasing its historical record. The sealed Retry-1 root establishes a new, deliberately
narrow evidence class:

| Evidence class | Required facts | Promotion limit |
|----------------|----------------|-----------------|
| `sealed_retry1_functional_with_provenance_watch` | Exact two-H800 resource PASS, Qwen rank-0 smoke PASS, Echo standalone pipeline PASS, and complete byte/hash inventory | May support narrow functional readiness only; cannot be promoted to clean-source Gate B, complete Task1→Task2→Task3, `release_pre_dataset`, or `AE-ready` |

Observed D42/D43 facts are:

- predict/live normalized argv=`16/16`, equal=`true`; predict/live/worker exits=`0/0/0`;
- requested/visible H800=`2/2`, distinct UUIDs=`2`;
- Qwen forward/backward/optimizer counts=`1/1/1`, durations=`11.95/7.93/2.99 ms`, real trace
  files=`1`, trace bytes=`4,248`;
- Echo rows=`727`, validation/test MSE=`0.0031091272501499075/0.0033649328512874955`, reload
  match=`true`;
- sealed inventory files/bytes=`2,184/419,329,007`, all audited mismatch counts=`0`.

The corresponding guardrails are mandatory:

1. `D42 Retry-1 identity=CONSUMED_TERMINAL_NO_REUSE`.
2. `old D28 replacement budget=UNCONSUMED_SUPERSEDED_NOT_NEEDED` under the audited sibling D33
   disposition. It is not an available retry, and no new live action is implied.
3. Source provenance remains `WATCH / PARTIAL`: the D42 controller had `13` dirty paths without a
   bound diff, and the Echo tar has no producer commit in `qualification_result.json`.
4. The active sim-engine's separate `3` modified plus `3` untracked paths remain a fail-fast
   provenance problem for current Task3 manifests.
5. Qwen ran only rank `0`; its three replay `.pt` files are not rank traces. Qwen and Echo were
   separate probes, not an atomic Task1→Task2 chain.
6. Final `3x3` status remains `BLOCK`: all three models still require complete, source-bound,
   portable, checksum-verified Task1→Task2→Task3 evidence and nine-entry clean-clone execution.

Under D30, every test/validation/rehearsal defect encountered while building that evidence may be
repaired autonomously when it serves the AE deliverables, including defects that require a
task-scoped implementation change. Dirty-source equivalence, missing real data, missing
ranks/tasks, external GPU authority, and final data quality remain unmet acceptance conditions:
they may be fixed and re-tested, but never self-waived or promoted without evidence.

## 9. Setup Runtime Verifier and Provenance Reconciliation — 2026-07-19

The setup control-plane gate is locally closed with synthetic contract evidence:

| Check | Result |
|-------|--------|
| Fixed runtime verifier | PASS, 21/21 negative/positive cases |
| Setup source/status integration | PASS, 6/6; real installer execution count=0 |
| Grouped-gemm installer regression | PASS, 37/37 |
| Outer/nested sim-engine provenance | RESOLVED, outer gitlink and nested HEAD=39755169f73f6c748e8d7376c3a2158c6569436b; nested status clean |

I39 is CLOSED/RESOLVED. The old dirty-producer text remains historical only and must not be
interpreted as current status. The closure is evidence-based and does not weaken the clean-source
contract.

The D30 test-autonomy overlay remains active: a test/validation/rehearsal/audit defect serving the
one-click scripts or reusable pre-dataset may be autonomously diagnosed and repaired with
RED→root cause→minimal fix→GREEN and affected regression evidence. No assertion, threshold,
checksum/provenance, no-fallback, real-vs-synthetic, or data-quality rule may be weakened.

The remaining hard block is not a local setup test block:

- Echo exact-two-H800 and integrated Gate B1 are not qualified;
- the complete real GPT-175B, Qwen3-A30B, and DeepSeek-V3 Task1→Task2→Task3 matrix is absent;
- full-rank coverage, atomic provenance, portable manifest/checksum/data-quality validation, and
  clean-clone replay remain incomplete.

Consequently the harness status remains INCOMPLETE, real pre-dataset remains NOT QUALIFIED, and
AE-ready remains NO.
## 3.2 D30 continuation: writable temporary-root invariant

The test-autonomy gate includes the operational test environment needed to exercise the AE
deliverables. Every AE-facing test, validator, rehearsal, and setup fixture may select a writable
temporary root through SC26_AE_TMP_ROOT, then TMPDIR, and only then the platform default /tmp.
This is a portability requirement for the test harness, not a fallback for product artifacts.

A test must fail if its selected root cannot be created or written. The root selection does not
permit source switching, acceptance relaxation, checksum/provenance bypass, or evidence promotion.
The continuation repair changed only test fixture temp-root paths and preserved all release
invariants. Current local evidence is green, but the real GPU and release gates remain closed.

## Session 45 control-plane gates

The following gates are now explicit and fail-closed for any future release attempt:

1. Task1/Task2 producer bytes must be tracked or captured in an immutable source snapshot; an outer
   repository `HEAD` alone is insufficient when AE wrappers/tools are untracked.
2. MoE QUICK rank sets are smoke-only; full Qwen/DSV3 release evidence requires the exact full rank
   inventory and matching trace/memory counts.
3. A Task2 qualified evidence label must pass the same state machine through manifest verification,
   reuse, marker writing, and shared-pointer publication.
4. Task3 and packaging must validate a trusted canonical root before resolving links and consume one
   frozen input expectation snapshot; live-path drift is a failure.
5. Issuer authentication is an external governance gate. Schema/checksum validation cannot promote
   a bundle to qualified evidence.

These gates supplement, and do not weaken, the existing no-fallback, fail-fast, checksum,
provenance, real-vs-synthetic, quota, and data-quality rules. Synthetic local tests remain
`local_synthetic_not_gpu_qualification`.

## Session 57 reduced functional gates

- Representative models are exactly `gpt175b` and `qwen3_a30b`; DeepSeek-V3 is deferred.
- GPT Task1 must contain the 8 PP representative ranks; Qwen Task1 must contain all 32 PP×EP representatives.
- NCU provenance must be Task1 global rank 0 only.
- Task2 must retain real two-GPU evidence, verified marker/manifest/checksums, model, scaler, and dataset; Task3 kernel gaps never authorize a Task2 rerun.
- Task3 must enable slowdown and produce report, manifest, and marker with finite nonnegative timings.
- Missing slowdown features use exact, then one unambiguous canonical alias, otherwise `missing_skip` with baseline duration unchanged.
- Functional distribution and CPU-only Task3 remain mandatory before AE-ready; fidelity and real distributed ground-truth accuracy are outside this functional gate.
