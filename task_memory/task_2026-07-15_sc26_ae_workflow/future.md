# Future Work — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Deferred DeepSeek-V3 repair and fidelity/distributed validation outside the current two-model functional AE session |
| 2026-07-19 | Added future design work for Task1/Task2/Task3 source snapshots, qualified evidence state, trusted paths, package schema, and issuer authentication |
| 2026-07-19 | Superseded the stale I39 dirty-worktree future item with a current clean-provenance status and future revalidation-only scope |
| 2026-07-19 | Listed work explicitly outside the current docs/local-validation session; no future item is treated as completed evidence |

## Scope boundary

The following items are intentionally outside the current session. They remain release gates or
future extensions and must not be inferred from the local synthetic reports.

## 1. Revalidate clean sim-engine provenance (I39 resolved)

The historical I39 dirty-producer finding is **CLOSED/RESOLVED** for the current producer boundary:
the outer gitlink and nested `megatron-sim-engine` HEAD both equal
`39755169f73f6c748e8d7376c3a2158c6569436b`, and the nested worktree is clean. Future work is limited
to re-running the provenance and regression checks whenever a new release bundle is captured; it
must fail fast if a newly executed nested source is dirty or differs from the bound gitlink.

The original dirty-worktree wording is retained in the issue and plan histories as audit evidence;
it is not a current blocker and must not be read as a pending repair.

## 2. Complete current-image and exact-two-H800 qualification

For `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`, resolve and record the immutable digest,
run the reviewed clean preflight/predict-only contract, and complete the permitted exact-two-H800
Echo retry. Preserve the existing D27=`PASS`, Echo=`BLOCK`, and integrated-B1=`BLOCK` history until
fresh evidence changes those verdicts.

## 3. Collect the real three-task evidence

After Gate B is genuinely open, collect real Task1 traces/memory/SQLite/NCU artifacts, real Task2
slowdown data and predictor/scaler metrics, and real Task3 simulator reports for GPT-175B,
Qwen3-A30B, and DeepSeek-V3. Bind every artifact to source, image, runtime, topology, and
checksum metadata.

## 4. Qualify and package the reusable pre-dataset

Run the data-quality, completeness, relocation, size, checksum, and provenance gates on a fresh
real bundle. Choose the approved distribution path only after measuring the actual artifact sizes.
Then rehearse a clean clone using the exact documented commands. External push, release creation,
or publication remains a separate explicit approval action.

## 5. Close downstream phases and archive the final summary

Only after integrated B1 and the real workflow pass may B2–B5 and Phases 1–9 be advanced. At that
point update this task's summary and lessons with final hashes and numeric qualification matrices;
until then, this task remains `INCOMPLETE`.

## 6. Optional extensions (not required for the current handoff)

- Add a reusable clean-submodule provenance helper shared by Task1 and Task3.
- Add a repository-level validator that checks all eleven required task artifacts rather than only
  the historical seven/eight-document subset.
- Add a formal real-vs-synthetic evidence dashboard for future AE revisions.

## Session 45 design work outside the current verified branch

The following items remain future work until the corresponding design is explicitly approved and
implemented with RED→GREEN tests:

- bind the complete Task1/Task2 AE producer surface to a tracked immutable source snapshot;
- add model-specific Task1 `capture_scope` and full-rank promotion gates;
- define the raw→attested→qualified Task2 evidence state machine and atomic qualified-pointer
  publication;
- bind every Echo subordinate module to the fixed interpreter;
- introduce trusted-root and frozen-input snapshot helpers for Task3/package consumers;
- unify schema-specific manifest semantics, summary recomputation, exact marker identity, and
  fresh/prebaked trace semantic validation;
- obtain an approved cryptographic issuer-authentication protocol.

These are not completed deliverables, and no synthetic fixture can close them.

## Deferred model and fidelity work

- DeepSeek-V3 tracing/runtime RCA and repair are deferred; the current functional release uses GPT-175B and Qwen3-A30B.
- Real distributed multi-node/multi-GPU accuracy comparison and paper-number fidelity are outside the current AE functional-badge scope.
- Broader similar-kernel alias research is unnecessary unless a future workload has exactly one defensible canonical match.
