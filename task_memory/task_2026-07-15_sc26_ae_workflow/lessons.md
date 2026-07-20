# Lessons — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added D30 lesson: broaden test-detected defect autonomy to task-scoped AE fixes while preserving RED→GREEN, evidence classes, and release gates |
| 2026-07-19 | Added double-checked lessons on evidence classes, provenance, fail-fast source selection, and D29 self-repair discipline |

## Vetted reusable lessons

### 1. Separate control-plane green from qualification green

A shell workflow can pass syntax, schema, fixture, and simulator tests while the real GPU and data
gates remain closed. Keep an explicit evidence-class field in every local report. The current
Task3 report demonstrates this with `local_synthetic_not_gpu_qualification`, despite `6/6` unit,
`6/6` integration, `3/3` prebaked e2e, `1/1` fresh-chain, and `45/45` sim-engine tests passing.

### 2. Inspect nested worktrees, not only outer gitlinks

An outer repository gitlink identifies the intended submodule commit, not whether the checked-out
submodule contains uncommitted changes. A reproducibility validator must inspect both the gitlink and
`git -C <submodule> status --porcelain`. Otherwise a manifest can claim a clean commit while the
actual execution used dirty code.

### 3. Make source selection explicit and fail fast

`fresh` and `prebaked` are different provenance paths. Automatic fallback can hide the actual
failure and produce a bundle whose inputs are unclear. Require the caller to select one source,
verify that source completely, and stop on missing, partial, corrupt, or mismatched artifacts.

### 4. Publish markers only after complete verification

A run marker is an assertion that the corresponding report and manifest are complete. Writing it
before nested manifests, semantic fields, checksums, and provenance are verified allows partial
runs to masquerade as reusable data. Marker publication must therefore be the final operation in a
successful run.

### 5. Record numeric evidence, not only verdicts

AE reviewers need scale and relationships. Reports should include observed durations, MSE/deltas,
peak RSS, allocation size, file counts, byte totals, and hashes. For example, the current synthetic
Task3 report records rank0 step values (`18.5/22.5/24.5 ms` for the three models), `45/45` tests,
`10.82 s` pytest elapsed time, and a touched `32 MiB` allocation while clearly withholding real
qualification claims.

### 6. D29 enables repair without lowering the bar

Allowing autonomous test/control-plane repair reduces approval latency, but it is safe only when
the repair is root-cause based and acceptance-preserving. The required RED→GREEN record prevents a
test from being “fixed” by deleting an assertion, widening a threshold, adding a fallback, or
relabeling synthetic output as real evidence.

### 7. D30 separates test-failure ownership from gate promotion

When a test, validation, rehearsal, or qualification check exposes a defect in an AE shell,
control-plane path, or task-scoped implementation, the same agent can repair the root cause without
waiting for a separate approval handoff. This does not make the failed check pass: the original
assertion, provenance/checksum/data-quality contract, and evidence class stay intact, and promotion
requires a fresh RED→GREEN regression with numeric evidence. External authority/resource failures,
irreversible actions, publication, and materially scope-changing refactors remain separate gates.

### 7. Treat image tags as pointers until digest qualification exists

An image tag can move or identify an unqualified runtime. Record the immutable digest and fresh
worker/tool evidence before calling an image qualified. Historical image evidence must remain
historical and must not silently support a current release claim.

### 8. Keep run roots immutable and non-overlapping

Fixed CWD-relative output directories allow stale traces, caches, or reports to enter a later
manifest. Versioned run roots plus “destination already exists” failure make provenance boundaries
observable and prevent accidental mixing of fresh and prebaked artifacts.

### 9. Preserve failure evidence as carefully as success evidence

Qualification incidents, invalid predict-only probes, dirty provenance, and incomplete real data are
part of the audit trail. Do not delete or overwrite them to make the latest status look cleaner; add
a new evidence root and record the transition instead.
