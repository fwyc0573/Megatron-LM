# claude advisor artifact

- Provider: claude
- Exit code: 0
- Created at: 2026-07-23T03:22:35.590Z

## Original task

Provide an independent read-only review of this branch/provenance decision. We have a complete detached source at commit 3b1b51e, while an existing functional prebaked bundle was built at exact producer c7288c66 and verification requires exact commit equality. We need one canonical branch containing GPT-175B and Qwen3-A3B scripts, docs, compact evidence manifests/reports/checksums copied from /data/ycfeng/tmp, and operator index. Compare: A) branch exact c7288c66 and keep all new evidence outside branch; B) branch current 3b1b51e, commit the compact index/evidence, then rebuild only the functional bundle against the final commit (never rerun Task2). Recommend one and state provenance risks and validation steps. Do not edit repository files.

## Final prompt

Provide an independent read-only review of this branch/provenance decision. We have a complete detached source at commit 3b1b51e, while an existing functional prebaked bundle was built at exact producer c7288c66 and verification requires exact commit equality. We need one canonical branch containing GPT-175B and Qwen3-A3B scripts, docs, compact evidence manifests/reports/checksums copied from /data/ycfeng/tmp, and operator index. Compare: A) branch exact c7288c66 and keep all new evidence outside branch; B) branch current 3b1b51e, commit the compact index/evidence, then rebuild only the functional bundle against the final commit (never rerun Task2). Recommend one and state provenance risks and validation steps. Do not edit repository files.

## Raw output

```text
I now have enough context to deliver the review. Here's my independent assessment:

---

## Provenance Decision Review

### Situation Summary

| Commit | Role | Content delta from predecessor |
|--------|------|-------------------------------|
| `c7288c66` | Functional bundle producer | Scripts, libs, tools, bundle-sealing code — the commit at which `package_prebaked.py` sealed the distribution manifest |
| `3b1b51e` | Documentation descendant | 8 files changed, all `.md` (README, task_memory progress/review/summary/test_report) — zero code or bundle changes |

The critical constraint is in `SC26-AE/lib/task3_simulation.sh:483` — verification resolves `git rev-parse HEAD` and compares it against `producer_commits` in the distribution manifest. The README at 3b1b51e explicitly states: *"Functional distribution verification deliberately requires the current checkout to equal the bundle producer recorded in `distribution_manifest.json`; a later docs-only descendant is not accepted."*

---

### Option A: Branch at exact c7288c66

**Mechanism**: Create the AE delivery branch directly at the producer commit. Evidence manifests, compact reports, and checksums from `/data/ycfeng/tmp` (~988K) remain outside the branch — delivered as a companion archive or side-channel.

| Dimension | Assessment |
|-----------|-----------|
| Bundle verification | Trivially passes — HEAD IS the producer |
| Task2 safety | Zero risk — nothing is rerun |
| Self-containment | Weak — AE reviewer needs the branch + external evidence |
| Provenance chain | Pristine — no commit mutation, no re-sealing |

**Risks**:
- External evidence files in `/data/ycfeng/tmp` are ephemeral; if lost before archival, the audit trail is gone.
- AE reviewers may expect a single self-contained artifact.

---

### Option B: Branch at 3b1b51e, commit evidence, rebuild bundle

**Mechanism**: Take the current HEAD (which has all docs), add compact evidence, then re-run `package_prebaked.py` to re-seal the bundle with the new final commit as producer.

| Dimension | Assessment |
|-----------|-----------|
| Bundle verification | Requires re-sealing — the distribution manifest must be regenerated with the new HEAD as `producer_commits` |
| Task2 safety | `package_prebaked.py` is documented as "a sealing step, not an evidence promotion step" — it should accept existing bundle artifacts. But the re-seal changes the manifest hash, which means every downstream checksum/marker binding is invalidated. |
| Self-containment | Strong — single branch, everything in one place |
| Provenance chain | **Broken and re-anchored** — the bundle was originally verified at c7288c6; re-sealing at a new commit creates a gap where the bundle content was produced under one commit but the manifest claims another |

**Risks**:
1. **Provenance gap**: The prebaked `.pkl`/profile data was generated at c7288c6. Re-sealing claims a new producer. An AE reviewer who inspects file timestamps vs. commit dates will see the disconnect.
2. **Silent regression**: If `package_prebaked.py` validation checks (e.g., hash of existing bundle entries against their recorded state) have any path-dependency on the original sealing environment, the re-seal may fail or require debugging.
3. **Cascade invalidation**: Every marker, checksum, and evidence binding recorded in the existing test reports references c7288c6. Committing evidence that says "producer = c7288c6" into a branch whose manifest says "producer = <new-hash>" creates a contradiction.
4. **"Never rerun Task2" violated in spirit**: While Task2 itself isn't rerun, the re-sealing step is architecturally equivalent to saying "the current commit produced this bundle" — which it didn't.

---

### Recommendation: **Option A**

**Rationale**: The bundle's real provenance is c7288c6. That fact is embedded in the actual data (file creation timestamps, log provenance markers, the two-GPU Task2 evidence that was never re-run). Option B would make the manifest *lie* about provenance to gain self-containment — a tradeoff that weakens the primary value of exact-commit verification.

The self-containment weakness of Option A is solvable without provenance compromise:

### Validation Steps for Option A

1. **Create the AE branch** at `c7288c66` exactly.
2. **Archive evidence separately**: tar the compact evidence files from `/data/ycfeng/tmp/sc26_ae*` into a checksummed archive (e.g., `sc26_ae_evidence_c7288c6.tar.gz` with SHA-256). Store alongside the branch (e.g., as a GitHub release asset, or a `prebaked_evidence/` directory that is documented but not part of the manifest check path).
3. **Verify bundle from clean clone**:
   ```bash
   git clone --branch <ae-branch> <repo> /tmp/ae_verify
   cd /tmp/ae_verify
   # HEAD should be c7288c66
   git rev-parse HEAD  # must equal c7288c66f0a6c3d0445edc841a6e5982d3b22f09
   # Run verify-functional or the CPU-only Task3 path
   ```
4. **Operator index**: Include a concise `OPERATOR.md` at the branch root that tells the AE reviewer: (a) the branch is self-sufficient for running the workflow, (b) the evidence archive is supplementary audit material with its own checksum, (c) the bundle was produced and verified at this exact commit.
5. **Cross-check**: Confirm `distribution_manifest.json` inside the prebaked bundle records `c7288c66` as `producer_commits` and that the clean-clone Task3 CPU path passes without modification.

### If Self-Containment is Mandatory

If the AE committee requires a single git checkout with everything, there is a narrow safe path: commit the compact evidence/index files into an `evidence/` directory that is **explicitly excluded from the manifest verification path** (i.e., `task3_simulation.sh` only checks code/bundle hashes, not docs). This would be a commit on top of c7288c6 that does NOT trigger a re-seal because it touches no manifest-relevant files. Verify that `package_prebaked.py` and `task3_simulation.sh` do not include this directory in their hash scope before attempting it.
ℹ Log files: CLI debug log /home/i-fengyicheng/.stepcode/logs/stepcode-1784776835682.log | Session log /home/i-fengyicheng/.stepcode/sessions/a90f262a-2f0b-4b64-a30b-a6430f2a46c9.jsonl

```

## Concise summary

Provider completed successfully. Review the raw output for details.

## Action items

- Review the response and extract decisions you want to apply.
- Capture follow-up implementation tasks if needed.
