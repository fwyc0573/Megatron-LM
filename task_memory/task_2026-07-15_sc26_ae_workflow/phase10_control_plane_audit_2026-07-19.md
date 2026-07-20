# Phase 10 Control-Plane and Qualification-Handoff Audit

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added a read-only audit of Task1 producer binding, rank-scope promotion, Task2 qualified reuse, Task3 trusted-input handling, packaging metadata, and issuer-governance boundaries; no release gate was promoted |

## Audit scope and evidence boundary

This audit is a read-only review of the current SC'26 AE control plane. It covers:

- `SC26-AE/lib/task1_trace.sh`
- `SC26-AE/lib/task2_echo.sh`
- `SC26-AE/lib/task3_simulation.sh`
- `SC26-AE/tools/artifact_manifest.py`
- `SC26-AE/tools/package_prebaked.py`
- `SC26-AE/tools/seal_qualification.py`
- the related Task1/Task2/Task3 tests and task-memory contracts

The raw command transcript is:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/session45-control-plane-audit-raw.log
SHA256=14342fc38a909854712de101518ccc7c39828e7a149b1d0fa8637a0a05d6c40a
bytes=119279
lines=1271
```

The audit did not run a GPU, RJob, Docker, real Nsight capture, Echo two-GPU workload, external
attestation, publication, commit, push, `rm`, `mv`, reset, or submodule mutation. It does not
promote any synthetic marker to a real qualification label.

Authoritative current boundary:

```text
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## Baseline identity

| Identity | Observed value |
|----------|----------------|
| Outer `HEAD` | `c217ce93156e7c37e065da2989c1a482f12ecebc` |
| Outer `megatron-sim-engine` gitlink | `39755169f73f6c748e8d7376c3a2158c6569436b` |
| Nested `megatron-sim-engine` `HEAD` | `39755169f73f6c748e8d7376c3a2158c6569436b` |
| Nested status | clean in this audit (`git status --porcelain` empty) |
| D45 quota semantic result | `gpu : 129/128`, CLI exit `0`, semantic `FAIL` |

The clean nested sim-engine state closes the historical I39 current-producer discrepancy; it does
not close the new source-binding and release-handoff findings below.

## Findings

### F10-01 — Task1 source pin does not cover the executed AE producer surface (**BLOCK**)

`ae_task1_assert_source_provenance()` currently checks the model source script plus five Megatron
files (`pretrain_llama.py`, `training.py`, `cmd.py`, and `arguments.py`). The actual AE execution
and manifest path also loads the following files:

- `SC26-AE/task1_gpt175b.sh`
- `SC26-AE/task1_qwen3_a30b.sh`
- `SC26-AE/task1_dsv3.sh`
- `SC26-AE/lib/common.sh`
- `SC26-AE/lib/task1_trace.sh`
- `SC26-AE/tools/artifact_manifest.py`

The raw audit found all six AE paths `HEAD_MISSING` at the current outer `HEAD`, while the working
tree contains them as an untracked overlay. The runner still records only `git rev-parse HEAD` in
the Task1 metadata. Therefore that commit cannot reproduce the bytes that generated a real Task1
bundle.

**Root cause:** the provenance contract binds selected runtime source files but not the complete
AE control-plane producer and manifest tool.

**Closure condition:** define one canonical producer snapshot, bind every load-bearing AE wrapper,
helper, manifest/metrics tool, and consumed Megatron source to tracked bytes, and execute from the
same snapshot. A real marker must be rejected until the recorded source identity can reproduce the
executed bytes. Do not solve this by merely copying a new commit string into JSON.

### F10-02 — MoE QUICK scope can pass the synthetic chain without a full-rank release gate (**BLOCK**)

The runner records `selected_rank_ids` and `selected_rank_count`, but the current manifest schema,
sealer, and packager do not enforce model-specific rank promotion rules. A Qwen/DSV3 QUICK capture
can therefore retain `profile=full` and `world_size=256` while containing only ranks
`0,64,128,192`. The local synthetic chain correctly exercises QUICK as a smoke subset; the defect
is that no independent machine gate prevents later promotion to `real_single_h800_qualified` or a
release bundle.

**Closure condition:** add a contract that distinguishes `capture_scope=quick|full`, validates the
exact selected-rank set, matches trace/memory counts, and rejects MoE QUICK evidence at sealer and
packager boundaries. Full Qwen/DSV3 release evidence must require the exact `0..255` set. This is a
design/contract change and is not implemented in this audit.

### F10-03 — Task1 `verified` marker has weak trace/SQLite semantics and PATH-selectable real `nsys` (**HIGH/WATCH**)

Task1 currently checks trace filename/rank inventory and that Nsight files are non-empty. It does
not, before publishing the marker, require the trace content to contain all three target operations
(`forward_step`, `backward_step`, `optimizer_step`), valid timing/identity fields, or the required
communication/DDP trigger metadata. The integration fixture can publish a verified marker from a
one-line trace and a plain-text file named `.sqlite`; this proves that the current marker means
structural inventory, not consumable Nsight semantics.

In real mode, `nsys` is selected with `command -v nsys`; the README/setup contract names a canonical
`/usr/local/bin/nsys` and version, but the Task1 metadata does not record the resolved path,
version, or capture argv digest.

**Closure condition:** either move the minimum trace/SQLite semantic validator before marker
publication, or rename/document the marker as structural-only and require a strict semantic gate in
the next consumer. Bind real `nsys` to the canonical executable/version and record its identity.

### F10-04 — D16 timing fields are specified in the plan but not emitted by Task1 (**OPEN**)

The plan requires `single_rank_elapsed_seconds`, `estimated_full_seconds`,
`fresh_capture_gate_threshold_seconds`, and `fresh_capture_gate_result`. The current Task1 source
emits only `capture_elapsed_seconds`; the raw audit found the other names in `plan.md` but not in
`SC26-AE/lib/task1_trace.sh`. No synthetic timing has been inserted to fill this gap.

**Closure condition:** decide the authoritative timing measurement boundary, emit the required
fields from a real capture, and test the `pass|prebaked_required` decision without using estimated
values as provenance for a missing full capture.

### F10-05 — Task2 qualified reuse lifecycle is internally contradictory (**BLOCK**)

Real reuse first requires `real_exact_two_h800_qualified` through
`task2_validate_reuse_evidence()`, but the subsequent `task2_verify_run()` and `task2_write_marker()`
paths only accept raw/pending classes (`local_synthetic_not_two_gpu_qualification` and
`runtime_measurement_requires_external_two_gpu_qualification`). A real build intentionally emits
the pending class, while a sealed manifest can emit the qualified class. There is no accepted
qualified path through the complete verifier/marker chain.

**Closure condition:** define one explicit evidence state machine for raw, externally attested, and
qualified bundles; make the same state machine authoritative in manifest verification, reuse, and
marker publication. Do not broaden a single conditional without reconciling all cross-file evidence
fields.

### F10-06 — Sealer does not publish a canonical qualified shared pointer and identity has no closed loop (**BLOCK**)

`seal_qualification.py` publishes a new sealed tree and receipt, but no
`_shared/task2/predictor_marker.json`. The raw builder owns that pointer. The sealer refuses an
existing destination, while a new destination basename conflicts with the strict equality between
pointer ID, directory basename, and manifest `predictor_run_id`. Consequently, the repository has
no machine-enforced path from external attestation to a canonical qualified pointer consumed by
remaining model attachments and the packager.

**Closure condition:** specify the immutable qualified-run naming and atomic pointer publication
protocol first, then implement and test it. Any external issuer/authentication design must be
approved separately; local JSON edits are not qualification.

### F10-07 — Task2 fixed interpreter stops at the outer launcher (**HIGH**)

The wrapper starts `update_configs.py` with the fixed interpreter, but `run_all.sh` and subordinate
modules use `which python`, bare `python` in `json_get()`, or a config-injected `python_path`. The
current interpreter test exits before entering those subordinate modules, so it does not prove that
profiling, training, analysis, merge, and prediction all use the fixed executable.

**Closure condition:** bind the complete subordinate execution chain to one fixed executable and add
a test that reaches at least one subordinate module and fails on a deliberate interpreter mismatch.

### F10-08 — Task2 manifest records outer `HEAD`, not all actual producer bytes (**BLOCK**)

The Task2 manifest records `git rev-parse HEAD` as `source_commits.megatron_lm`, while the actual
Task2 wrappers/helpers/tools are untracked in the current outer commit. Echo snapshot provenance is
stronger, but it does not cover the outer AE producer. This is the same reproducibility class as
F10-01 and independently blocks release provenance.

**Closure condition:** bind the full outer producer surface to a tracked commit/snapshot and verify
working-tree bytes before and after execution; include the manifest/metrics producer itself.

### F10-09 — Task2 self-consumer path and cross-file provenance checks are weaker than packaging (**MEDIUM**)

Task2 marker/pointer consumers reject lexical `..` and absolute paths but do not reject intermediate
symlinks or prove resolved containment inside the canonical output root. Reuse verifies schema,
Echo commit, and `automatic_fallback`, but does not fully cross-check model, predictor ID, execution
mode/evidence, CUDA IDs, rebuild/command identity, and `run_path`/`run_relative_path` across marker,
manifest, and provenance.

**Closure condition:** share the trusted-path and cross-file identity helper with packaging and add
negative tests for symlink escape, alias divergence, and split-brain provenance.

### F10-10 — Task3 fresh resolver validates a root after `resolve()` and does not freeze consumed bytes (**HIGH**)

The fresh resolver resolves `output_root` and `task1_dir` before checking lexical symlinks or exact
canonical relationship. A symlinked Task1 root can therefore become the apparent trusted root. After
validation, resolver metadata and slowdown assets are copied while the builder/simulator continue to
read live source paths; there is no immutable expectation snapshot shared by validation, copy, and
consumption. This leaves a root-boundary and TOCTOU gap even though individual copy checks are strong.

**Closure condition:** establish lexical no-symlink/exact-root containment first, create one frozen
manifest expectation snapshot, and make every subsequent copy and simulator input consume that
snapshot (or fail if bytes change).

### F10-11 — Packaging summary/identity and generic manifest semantics remain incomplete (**MEDIUM/HIGH**)

The package builder computes `total_size_bytes` and medium before its final manifest write, so the
reported total can differ from the final manifest's own size. Marker source paths are prefix-checked
but not always exact run-identity checked. The generic manifest verifier does not uniformly require
Task3/prebaked fields such as `execution_evidence`, `simulation_run_id`, communication backend,
overlap mode, and database/trace identity. Prebaked trace validation is weaker than fresh semantic
validation.

**Closure condition:** define one schema for each artifact class, recompute summary fields during
verification, require exact marker/run identity, and run the same semantic trace checks for fresh and
prebaked inputs.

### F10-12 — Cryptographic issuer authentication remains an external governance blocker (**BLOCK**)

The sealer validates schema, checksums, and attestation fields but has no trusted issuer key,
signature verification, or allowlist. Integrity proves bytes; it does not prove who issued the
qualification. This is tracked as review finding CR-01 and is intentionally not implemented by
local synthetic work.

**Closure condition:** obtain an approved issuer protocol and trusted-key distribution before
promoting an externally sealed bundle to release qualification. Do not invent a local signature
scheme or infer issuer trust from a JSON field.

## Disposition and next gate

| Area | Current disposition | Permitted next action |
|------|---------------------|-----------------------|
| Local alias repair | Closed with RED→GREEN evidence | Keep both checksum aliases mandatory |
| Task1 producer/rank/semantic controls | BLOCK/HIGH/OPEN | Design and implement only after the contract is approved; add negative tests |
| Task2 qualified reuse/pointer/provenance | BLOCK/HIGH | Resolve state machine and identity publication before real reuse |
| Task3 trusted root/snapshot/package | HIGH/MEDIUM | Design frozen-input seam and schema before changing consumer code |
| Issuer authentication | External BLOCK | Obtain governance decision and trusted issuer material |
| Real GPU/RJob qualification | Not started in this audit | Remains blocked by D45 and unresolved control-plane gates |

No finding in this audit justifies changing acceptance thresholds, adding fallback/source switching,
relabeling synthetic evidence, or promoting a real/release marker.

