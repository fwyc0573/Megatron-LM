# Test Report — I55 Interpreter Chain Semantic Hardening

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Corrected the final V21 historical/current wording before the v7 reconciliation; only summary.md's sole identity block remains current |
| 2026-07-20 | Corrected stale V21 v3/v4 current/final wording; only the summary's sole identity block is current |
| 2026-07-20 | Corrected stale final-v2/current and 1,272-byte transcript claims identified by the independent evidence audit; the final-v5 affected regression and marker-complete alternate-manifest v3 evidence remain current; V21 verifier v3 is historical |
| 2026-07-20 | Corrected the durable alternate-manifest evidence marker and superseded final-v2 with the final-v5 affected regression identity (`94 passed in 4.37 s`) |
| 2026-07-20 | Added final-v2 affected regression evidence (`94 passed in 4.75 s`), durable log identity, and explicit separation of initial versus current synthetic-only results |
| 2026-07-20 | Added a complete generic-manifest integration negative for the checksum-coherent alternate requested path; the unit fixture is explicitly scoped to the sidecar seam and I55 remains open |
| 2026-07-20 | Closed the synthetic real-evidence fixed-requested-path gap with a RED→GREEN alternate-path fixture, refreshed the affected regression, and corrected the archived-config SHA transcription; I55 remains open and no qualification gate moved |
| 2026-07-20 | Recorded the final RED→GREEN interpreter-chain checks, coherent semantic-tamper evidence, and affected local regression; I55 remains open and no qualification gate moved |

## Scope and disposition

This report covers the wrapper-owned Task2 interpreter-chain contract in
`SC26-AE/lib/task2_echo.sh` and its CPU-only control-plane tests. The change binds and checks the
outer fixed interpreter, the `PATH` lookup used by pinned Echo, all four generated nested
configuration files, archived configuration bytes, the interpreter sidecar, provenance, and the
artifact manifest. It also rejects duplicate JSON keys, binds artifact-only real-evidence bundles
to the wrapper's fixed requested interpreter path, and rejects a non-canonical executable path even
when an attacker coherently rewrites every checksum.

This is **not** a hardware or release qualification report. No GPU, RJob, Docker, H800, or
external worker qualification was run. No pinned `Echo-slowdown` source was modified.

The evidence class and release boundary are intentionally unchanged:

```text
evidence class = local_synthetic_not_gpu_qualification
I55 = OPEN / HIGH / BLOCK
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## Test Script Information

### Repository and scripts

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Wrapper under test: `SC26-AE/lib/task2_echo.sh`
- Unit contract: `tests/unit/test_sc26_ae_task2_interpreter_contract.sh`
- Evidence-mode unit: `tests/unit/test_sc26_ae_task2_evidence_mode.sh`
- Snapshot unit: `tests/unit/test_sc26_ae_task2_snapshot.sh`
- Integration contract: `tests/integration/test_sc26_ae_task2_contract.sh`
- Smoke e2e: `tests/e2e/test_sc26_ae_task2_smoke.sh`
- Fresh-chain e2e: `tests/e2e/test_sc26_ae_fresh_chain.sh`
- Related Python unit suites:
  - `tests/unit/test_sc26_ae_artifact_manifest.py`
  - `tests/unit/test_sc26_ae_seal_qualification.py`
  - `tests/unit/test_sc26_ae_package_prebaked.py`

### Reproducible commands

The final-v2 affected regression was run as one fail-fast matrix and persisted at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/i55-current-affected-regression-final-v2-20260720.log
```

The earlier Session 55 matrix remains retained at
`logs/i55-current-affected-regression-final-20260720.log` as historical evidence; it was not
overwritten.

The commands in that matrix were:

```bash
bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
bash tests/unit/test_sc26_ae_task2_evidence_mode.sh
bash tests/unit/test_sc26_ae_task2_snapshot.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/e2e/test_sc26_ae_task2_smoke.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh

python -m pytest \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_seal_qualification.py \
  tests/unit/test_sc26_ae_package_prebaked.py -q

bash -n SC26-AE/lib/task2_echo.sh
bash -n tests/unit/test_sc26_ae_task2_interpreter_contract.sh
bash -n tests/unit/test_sc26_ae_task2_evidence_mode.sh
bash -n tests/integration/test_sc26_ae_task2_contract.sh
git diff --check
test -z "$(git -C Echo-slowdown status --porcelain)"
```

The complete generic-manifest alternate-path integration case was also rerun independently and
persisted at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/i55-qualified-alternate-path-integration-green-20260720.log
```

### Environment

The regression ran on the controller, not a GPU worker:

| Item | Observed value |
|------|----------------|
| `python` | `/usr/bin/python` |
| Python | `3.12.3` |
| pytest | `9.1.1` |
| Torch | `2.5.1+cu124` |
| `torch.cuda.is_available()` | `False` |
| `torch.cuda.device_count()` | `0` |
| Pinned Echo worktree status | empty (`git status --porcelain` produced no bytes) |

The production real-mode literal remains `/opt/conda/envs/echo_slowdown/bin/python`. That path is
not present on this controller and was not replaced with another interpreter. The positive
interpreter-chain fixture therefore uses an explicitly isolated synthetic executable whose
identity is recorded below; the fixture identity must not be called an approved worker-image
digest.

## Validation Criteria

1. All changed interpreter-chain paths must be exercised, including positive binding, missing and
   malformed inputs, duplicate keys, path tampering, archive/provenance/manifest mismatches, and
   missing function arguments.
2. Every expected negative case must fail non-zero before nested producer work or marker/pointer
   publication.
3. A coherent tamper that updates all generic manifest checksums must still fail the independent
   semantic verifier.
4. A valid real-evidence sidecar must be checked by a synthetic reuse caller; synthetic mode must
   not bypass real-bundle artifact semantics or the wrapper's fixed requested path.
5. A complete checksum-coherent real bundle naming an alternate requested interpreter path must
   pass the generic manifest verifier and be rejected by the independent semantic verifier. The
   unit fixture isolates the sidecar seam; the integration fixture covers the complete manifest.
6. The existing publication-ordering case must reject semantic tampering before any marker or
   pointer publication.
7. Existing Task2 snapshot, smoke, fresh-chain, artifact/sealer/package, shell-syntax, and pinned
   Echo cleanliness checks must remain green.
8. No result may be promoted from local synthetic evidence to H800 qualification or release
   readiness.

## Test Results and Evidence

### Final regression matrix (retained initial Session 55 rerun)

| Test | Exit code | Key observed values |
|------|----------:|---------------------|
| `test_sc26_ae_task2_interpreter_contract.sh` | `0` | `ALTERNATE_FIXED_PATH_NEGATIVE=1`; `PARSER_NEGATIVES=11`; `DUPLICATE_KEY_NEGATIVES=4`; `SIDECAR_TAMPER_NEGATIVES=9`; `PASS_COUNT=12` |
| `test_sc26_ae_task2_evidence_mode.sh` | `0` | `PASS_COUNT=5` |
| `test_sc26_ae_task2_snapshot.sh` | `0` | Generic manifest status `verified`; `MANIFEST_FILE_COUNT=12`; synthetic snapshot/reuse/rebuild contract passed |
| `test_sc26_ae_task2_contract.sh` | `0` | Generic manifest status `verified`; complete alternate-path fixture `MANIFEST_FILE_COUNT=19` then semantic rejection; coherent publication tamper rejected; model attachments completed for `qwen3_a30b` and `dsv3` |
| `test_sc26_ae_task2_smoke.sh` | `0` | Public Task2 smoke contract passed; coherent semantic tamper and symlink-escape negatives passed |
| `test_sc26_ae_fresh_chain.sh` | `0` | `TASK1_TRACE_FILES=4`; `TASK1_MEMORY_JSON=4`; `TASK2_DATASET_ROWS=2`; chain `1/1` |
| Artifact/sealer/package pytest | `0` | `94 passed in 5.13 s` |
| Shell syntax checks | `0` | Wrapper, interpreter unit, and integration shells parsed successfully |
| `git diff --check` | `0` | No whitespace errors |
| Pinned Echo cleanliness | `0` | Empty `git -C Echo-slowdown status --porcelain` |

The retained initial matrix log is `logs/i55-current-affected-regression-final-20260720.log`,
`8,416` bytes with SHA256
`3aafe89acc8b7f718ae7711a7e00cef77d14d24764e27b6d46cf1544dd6aff53`; every command returned
exit `0` and the matrix ended with `SUMMARY PASS=12 FAIL=0`. The older
`logs/i55-final-regression-20260720.log` remains retained as historical evidence and is not
rewritten.


### Final-v2 affected regression (historical; superseded by final-v5)

The final-v2 matrix was run after the evidence-quality correction and did not overwrite any prior
transcript. Its durable log identity is retained below for historical comparison. The v2 log does
not contain a `FINAL_V2_EXIT` marker or per-command exit markers; the current durable aggregate exit
evidence is the final-v5 log later in this report.

```text
logs/i55-current-affected-regression-final-v2-20260720.log
bytes=7650
sha256=a78ea30c22667747d7d6f7a978c65fcda22b4b56b3a328e827f9b0b97d18ee84
FINAL_V2_EXIT_MARKER=ABSENT_FROM_V2_LOG
```

| Metric | Actual value |
|--------|--------------:|
| Matrix command sequence | `16` commands were exercised; the v2 durable log lacks a persisted aggregate exit marker, so v2 is historical rather than current closure evidence |
| Interpreter contract `PASS_COUNT` | `12` |
| Alternate fixed-path negative | `1` |
| Parser negatives | `11` |
| Duplicate-key negatives | `4` |
| Sidecar-tamper negatives | `9` |
| Evidence-mode `PASS_COUNT` | `5` |
| Generic-manifest integration file count | `19` before semantic rejection |
| Artifact/sealer/package pytest | `94 passed in 4.75 s` |
| Shell syntax / diff / pinned Echo checks | `0` exit / clean |
| CUDA availability / device count | `False / 0` |

The complete integration fixture independently reports `MANIFEST_STATUS=verified` and
`MANIFEST_FILE_COUNT=19` for the checksum-coherent alternate requested-path bundle, then the
semantic validator rejects `sidecar fixed requested path differs from wrapper fixed interpreter`.
This is local controller evidence only; it does not establish canonical-worker authenticity,
approved immutable digest, H800 qualification, or release readiness.

### Fresh-chain numeric evidence

The fresh-chain e2e run reported the following actual values; they are synthetic scale checks, not
performance claims:

| Metric | Actual value |
|--------|--------------:|
| Task2 average validation MSE | `3.0` |
| Task2 test MSE | `0.5` |
| Model reload max absolute prediction delta | `0.0` |
| Task3 rank-0 step | `22.5 ms` |
| Task3 forward | `6.0 ms` |
| Task3 backward | `11.0 ms` |
| Task3 optimizer | `2.5 ms` |
| Simulator load | `0.125 s` |
| Simulator execution | `0.375 s` |
| Simulator wall clock | `0.5 s` |
| Tested host allocation | `32 MiB` |
| CUDA devices | `0` |

### Interpreter identity and archived artifacts

The following values come from the positive isolated fixture retained by the final unit run at
`/data/ycfeng/tmp/sc26-ae-task2-runtime-binding.Ig39XA/nested-chain-binding/run`. The fixture's
requested path and canonical path are identical because it deliberately uses a regular executable
file rather than a symlink:

| Identity | Value |
|----------|-------|
| Fixed requested path (synthetic fixture) | `/data/ycfeng/tmp/sc26-ae-task2-runtime-binding.Ig39XA/nested-chain-binding/bin/python` |
| Fixed canonical path (synthetic fixture) | `/data/ycfeng/tmp/sc26-ae-task2-runtime-binding.Ig39XA/nested-chain-binding/bin/python` |
| Observed executable SHA256 | `7850db0d7accdd7833faf675726b9210fa659ffe5f33335c3793339baa1851af` |
| Executable size | `668` bytes |
| Production real-mode requested literal | `/opt/conda/envs/echo_slowdown/bin/python` |
| Approved immutable digest | **Not supplied; I55 remains open** |

All four archived configs were checked. Their values are intentionally identical in this minimal
fixture:

| Archived config | Size | SHA256 |
|-----------------|-----:|--------|
| `provenance/interpreter_configs/kernel_metric.global_config.json` | `104` bytes | `ecefbb83146a5401a75ef42d4115d67176b51496449fb255562765577db0db35` |
| `provenance/interpreter_configs/merge.global_config.json` | `104` bytes | `ecefbb83146a5401a75ef42d4115d67176b51496449fb255562765577db0db35` |
| `provenance/interpreter_configs/slowdown_collection.global_config.json` | `104` bytes | `ecefbb83146a5401a75ef42d4115d67176b51496449fb255562765577db0db35` |
| `provenance/interpreter_configs/training_testing.global_config.json` | `104` bytes | `ecefbb83146a5401a75ef42d4115d67176b51496449fb255562765577db0db35` |

Cross-file artifacts in the same fixture were:

| Artifact | Size | SHA256 |
|----------|-----:|--------|
| `interpreter_binding.json` | `3,258` bytes | `3f3ded71c458829b7bb3b3f822fca4a066c2c550276bf4ecaedf15a68c368799` |
| `provenance.json` | `515` bytes | `a7868a9dcdce069b0a13a78bd2911ba33f4d6bd5c89c4acbeebdf74f10e0c6f3` |
| `artifact_manifest.json` | `1,056` bytes | `adeeb735a96970e02b4f89cc1e029ac48bfd49599cbcb2c6ba8bbfa9ac823345` |

The provenance sidecar reference and manifest entry both contain sidecar SHA256
`3f3ded71c458829b7bb3b3f822fca4a066c2c550276bf4ecaedf15a68c368799`; the four manifest archive
entries each contain size `104` and the archive SHA256 shown above.

### RED → GREEN records

| Contract | RED evidence | GREEN evidence |
|----------|--------------|----------------|
| Initial sidecar verifier | `logs/i55-sidecar-verifier-red-20260720.log`, `232` bytes, SHA256 `c1c3b5984c2909cec2d47afdd18f5a79d9870faa35bcd70a6aefe627fbc1d358`, exit `1` (`command not found`) | `logs/i55-sidecar-verifier-green-attempt2-20260720.log`, `192` bytes, SHA256 `ef85b18bb3020003c6811a1dc4fbd133d49ed34f50ee9701536164b7ffc71375`, exit `0`, `PASS_COUNT=12` |
| Coherent canonical-path tamper | `logs/i55-coherent-canonical-red-20260720.log`, `313` bytes, exit `1`; generic verifier still printed `MANIFEST_STATUS=verified` | `logs/i55-coherent-canonical-green-20260720.log`, `1,177` bytes, SHA256 `2896476173e78a8d9a5db52cb5d48fa9a342a76ffb308698ec61ce08d4b2da81`, exit `0`; semantic tamper failed before publication |
| Duplicate top-level reuse evidence key | `logs/i55-reuse-duplicate-red-20260720.log`, `60` bytes, SHA256 `8c49e2a71a200c332f1e452f13bdf0745427866067a480e5b323d0372d499a42`, exit `1` | `logs/i55-reuse-duplicate-green-20260720.log`, `13` bytes, SHA256 `2a37c8d41b8bf0edc757526a84a8a2b84f0e99a2c908bc4c78fa13e2a9c315f6`, exit `0`, `PASS_COUNT=5` |
| Missing sidecar-validator argument | `logs/i55-sidecar-no-arg-red-20260720-v2.log`, `299` bytes, SHA256 `81eede81977e32398e59d968e4ff25451a38b9c207c6fd75a282b579b67cb5a1`, exit `1` due to the old `set -u` `$1` failure | `logs/i55-sidecar-no-arg-green-20260720.log`, `1,574` bytes, SHA256 `6bc085d6bcd9eea759c3a30bb919e129d068fcfdfcb534fdee2a39c380f3dc66`, exit `0`, parser/duplicate/tamper counts passed |
| Durable nested-interpreter escape | `logs/i55-nested-interpreter-red-20260720.log`, `1,431` bytes, SHA256 `0551da75f50e9801e875ba55ab036ce31b42c54d187c2239766da2050ac551f4`, `PROBE_RC=1`, `EXTERNAL_INVOCATIONS=1`, `NESTED_SENTINEL=created`, external path in `4/4` configs | Wrapper-only semantic checks now pass in the isolated fixture; this does not claim a real worker GREEN |
| Synthetic real-bundle fixed requested path (sidecar unit seam) | `logs/i55-alternate-fixed-path-red-20260720.log`, `179` bytes, SHA256 `523bc4512efa12b1ba5f89a3d82f259a13e965b82a1b80ab3040ac28c4deaf8e`, exit `1`; sidecar-coherent alternate path was accepted | `logs/i55-alternate-fixed-path-green-20260720.log`, `3,103` bytes, SHA256 `4b1f2a6453d603fb4e455a1e1b89e8874d6e44af6240771e8cac23ababbaf37f`, exit `0`; `ALTERNATE_FIXED_PATH_NEGATIVE=1` |

### Complete generic-manifest alternate-path integration

The integration contract now copies the complete qualified-real-shaped fixture, adds the alternate
executable to the manifest, rewrites all four archived configs plus sidecar/provenance fields, and
recomputes every listed checksum. The generic verifier accepted this complete bundle with
`MANIFEST_STATUS=verified` and `MANIFEST_FILE_COUNT=19`. The synthetic semantic validator then
rejected it with `sidecar fixed requested path differs from wrapper fixed interpreter`.

The older standalone transcript is
`logs/i55-qualified-alternate-path-integration-green-20260720.log` (1,272 bytes, SHA256
`cfda836abc419db5fa7a3bffe4eea6c861bdb0065f4810a6c790d33b715c6a37`, exit `0`), but it only prints
the surrounding fixture's `MANIFEST_FILE_COUNT=13` and therefore does **not** close the evidence-
quality gap. The marker-complete integration evidence is instead
`logs/i55-qualified-alternate-path-integration-green-v3-20260720.log` (1,340 bytes, SHA256
`70006af6138aa16b40d93c15409f42c96517e6f75f309f787c397f82dfb4f8d1`), which prints
`ALTERNATE_MANIFEST_STATUS=verified` and `ALTERNATE_MANIFEST_FILE_COUNT=19` before semantic
rejection. The unit case remains a focused sidecar-seam RED→GREEN, while the v3 integration case
proves the same rejection against a generic-manifest-coherent real bundle. It is still CPU-only
synthetic evidence and does not establish worker authenticity or publication qualification by
itself. The supplemental independent audit transcript is
`logs/i55-independent-requested-path-audit-20260720.log` (9,617 bytes, SHA256
`41a9a2b7d9ea8138096965093c68afcdae8db2670070f2fd1cfc83e3b8fe0118`); it records the same
local evidence-quality boundary and is not an approved worker digest.

### Synthetic real-bundle fixed-path root cause and resolution

The previous validator used `live_required=0` for a synthetic caller. It still checked the
sidecar's internal equalities, but it did not compare `fixed_requested_path` with the wrapper's
fixed `/opt/conda/envs/echo_slowdown/bin/python` literal. A bundle could therefore rewrite its
requested path, canonical path, SHA256, archived configs, provenance, and manifest coherently and
pass artifact-only validation. The original RED fixture isolated the sidecar seam and observed
exit `0` before the fix; the new integration fixture repeats the attack against a complete
generic-manifest bundle and records the independent generic-pass/semantic-reject split.

The minimal production repair performs the requested-path comparison for every pending/qualified
real-evidence bundle, regardless of caller mode. Worker-local canonical existence, executable
regularity, and current-byte SHA256 checks remain restricted to `live_required=1`; no local
synthetic digest is treated as an approved image digest. The GREEN fixture now rejects the same
checksum-coherent alternate bundle with the explicit wrapper-contract diagnostic. The integration
fixture uses the fixed requested literal for its valid positive state, while its dedicated negative
case adds the alternate executable to the generic manifest so no hidden unlisted-file assumption
is involved; its canonical executable is still controller-local and synthetic.

### Coherent tamper publication boundary

The integration test copied a qualified-real-shaped fixture, changed all four archived config
bytes, changed the sidecar to a non-canonical `/./` path, updated the provenance sidecar hash, and
rebuilt every generic manifest checksum. The generic verifier returned `MANIFEST_STATUS=verified`
with `MANIFEST_FILE_COUNT=13`. The independent semantic verifier then returned non-zero with
`canonical interpreter path is not lexically canonical`.

The test recorded the shared pointer SHA256 before and after the rejected reuse attempt as equal
(`83688bdb81ba0bfabd925f3ac02ae0805bc81d423fa7fdb692dd0d52d5812424`) and asserted that the
model-level marker did not exist. The failure evidence was preserved at:

```text
/data/ycfeng/tmp/sc26-ae-task2-integration.AidWLu/coherent-tamper-output/_work/task2-failure-20260719T190514Z-1021819-2423/failure.json
/data/ycfeng/tmp/sc26-ae-task2-integration.AidWLu/coherent-tamper-output/_work/task2-failure-20260719T190514Z-1021819-2423/failure.log
```

The failure files were `325` and `54` bytes respectively. This proves semantic rejection and
publication ordering in the fixture; it does not prove trusted-root or TOCTOU closure (I56).

## Residual risks and explicit non-claims

- The observed executable SHA256 is a local fixture measurement, not an authority-approved image
  digest. I55 cannot close until an external owner supplies and validates that immutable digest.
- The controller has no CUDA device and cannot validate the fixed `/opt/conda` interpreter on the
  canonical worker image.
- The sidecar checks lexical canonical form and current bytes; it does not by itself solve
  descriptor-anchored TOCTOU, frozen input snapshots, producer source snapshots, or issuer
  authentication. Those remain I56, I57, I51, and I58 boundaries.
- Nsight tool binding (I53) is outside this report.
- `artifact_manifest.py` still has broader duplicate-key readers outside the I55 semantic paths;
  this report does not claim generic manifest duplicate-key closure beyond the exercised contract.
- No evidence threshold, source selection, fallback policy, pointer lifecycle, or release label was
  changed.

## Final disposition

The wrapper-owned I55 semantic hardening is locally GREEN for the exercised CPU fixtures and the
refreshed affected regression, including the synthetic fixed-path negative. I55 itself remains
`OPEN / HIGH / BLOCK`; Gate B1 remains blocked, both
pre-datasets remain unqualified, and `AE-ready` remains `NO`.

## Evidence correction and final-v5 affected regression — 2026-07-20

### Why v5 supersedes v2/v3/v4

The earlier standalone transcript
`logs/i55-qualified-alternate-path-integration-green-20260720.log` is retained for historical
continuity, but its `MANIFEST_FILE_COUNT=13` lines describe the surrounding positive fixture. It
must not be cited as proof that the alternate bundle had `19` files. The v2 transcript added the
alternate count (`1,305` bytes, SHA256
`22b45f5aa77da46858c3c08b8c48809d491eb76a9259d21f109febef3aed8a76`), and v3 added the explicit
status marker (`1,340` bytes, SHA256
`70006af6138aa16b40d93c15409f42c96517e6f75f309f787c397f82dfb4f8d1`). The v4 regression remains
historical (`7,787` bytes, SHA256
`afabf8967e331e5647cdb5123925c9925b90a94c4dfa1b26c95216f19edf4de1`, `94 passed in 4.67s`).

The marker omission was first made RED by removing the expected marker assertions. The durable RED
identity is:

```text
logs/i55-alternate-manifest-marker-red-20260720.log
bytes=257
sha256=8e50f7954f787344114fc1ccad74d85287482eb48f615f416e247fd67cfb4399
exit=1
```

The failure is the intended harness assertion (`expected explicit alternate manifest count marker
was absent`), not a production validator failure. After the minimal assertion/output correction,
the v3 integration transcript contains:

```text
ALTERNATE_MANIFEST_STATUS=verified
ALTERNATE_MANIFEST_FILE_COUNT=19
PASS: checksum-coherent alternate requested path is generic-verified but semantically rejected
```

### Current test script information and reproducible command

The current affected matrix is persisted at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/i55-current-affected-regression-final-v5-20260720.log
```

The following fail-fast command sequence reproduces the v5 coverage and appends the aggregate
`PIPESTATUS` result to the durable log:

```bash
set -euo pipefail
LOG=task_memory/task_2026-07-15_sc26_ae_workflow/logs/i55-current-affected-regression-final-v5-20260720.log
{
  bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
  bash tests/unit/test_sc26_ae_task2_evidence_mode.sh
  bash tests/unit/test_sc26_ae_task2_snapshot.sh
  bash tests/integration/test_sc26_ae_task2_contract.sh
  bash tests/e2e/test_sc26_ae_task2_smoke.sh
  bash tests/e2e/test_sc26_ae_fresh_chain.sh
  python3 -m pytest \
    tests/unit/test_sc26_ae_artifact_manifest.py \
    tests/unit/test_sc26_ae_seal_qualification.py \
    tests/unit/test_sc26_ae_package_prebaked.py -q
  bash -n SC26-AE/lib/task2_echo.sh
  bash -n tests/unit/test_sc26_ae_task2_interpreter_contract.sh
  bash -n tests/unit/test_sc26_ae_task2_evidence_mode.sh
  bash -n tests/unit/test_sc26_ae_task2_snapshot.sh
  bash -n tests/integration/test_sc26_ae_task2_contract.sh
  bash -n tests/e2e/test_sc26_ae_task2_smoke.sh
  bash -n tests/e2e/test_sc26_ae_fresh_chain.sh
  git diff --check
  test -z "$(git -C Echo-slowdown status --porcelain)"
} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'FINAL_V5_EXIT=%s\n' "$rc" >> "$LOG"
exit "$rc"
```

Environment was the controller (`Python 3.12.3`, pytest `9.1.1`, Torch `2.5.1+cu124`, CUDA
available `False`, device count `0`). The production wrapper and integration source identities at
this checkpoint are:

```text
SC26-AE/lib/task2_echo.sh
  bytes=72002
  sha256=207188ec2656fe60334ce97debdbdbfbb4440215b1a055c88de652c7ca844752
tests/unit/test_sc26_ae_task2_interpreter_contract.sh
  bytes=32377
  sha256=e97511fe7e625a15457b51cf2ad1d3c891405c20a3e50fc0c0495336bbe9acb2
tests/integration/test_sc26_ae_task2_contract.sh
  bytes=28197
  sha256=1722e4d5d15be761a8eb4a81c37375421d12672634bf2cad825e1c76f8476578
```

### Final-v5 results and numeric evidence

The current durable identity is:

```text
logs/i55-current-affected-regression-final-v5-20260720.log
bytes=7801
sha256=1ded471a0304d85300822f92ff6e717d842b9ea2ba69ee2bd998bae65a83eb32
FINAL_V5_EXIT=0
```

The log contains the independent alternate-bundle markers and the related pytest result:

```text
ALTERNATE_MANIFEST_STATUS=verified
ALTERNATE_MANIFEST_FILE_COUNT=19
94 passed in 4.37s
FINAL_V5_EXIT=0
```

| Test/metric | Actual result | Acceptance interpretation |
|-------------|---------------|---------------------------|
| Interpreter contract | exit `0`; `PASS_COUNT=12` | All positive/negative interpreter-chain cases in the unit contract passed |
| Alternate fixed-path negative | `1` | A checksum-coherent non-production requested path is rejected semantically |
| Parser negatives | `11` | Malformed/unsafe parser inputs rejected |
| Duplicate-key negatives | `4` | Duplicate JSON keys rejected in the exercised paths |
| Sidecar-tamper negatives | `9` | Sidecar/config/provenance tampering rejected |
| Evidence-mode unit | exit `0`; `PASS_COUNT=5` | Evidence-mode lifecycle cases passed locally |
| Complete alternate manifest | `MANIFEST_STATUS=verified`; `FILE_COUNT=19` | Generic checksums pass before semantic rejection |
| Semantic alternate-path check | explicit rejection | Wrapper fixed requested literal remains authoritative |
| Related pytest | `94 passed in 4.37s` | Artifact/sealer/package regression passed |
| Shell syntax, diff, pinned Echo | all exit `0`; Echo status empty | Static and source-cleanliness checks passed |
| CUDA | available `False`; devices `0` | No GPU qualification claim is permitted |

### Evidence boundary and final disposition

The v5 log is controller-only synthetic evidence. It demonstrates the generic-manifest/semantic
validator split and the narrow wrapper requested-path repair; it does **not** authenticate the
canonical worker, supply an approved immutable interpreter/image digest, prove H800 execution,
qualify a real pre-dataset, qualify a release pre-dataset, or set `AE-ready`. The current boundary
remains:

```text
I55 = OPEN / HIGH / BLOCK
I53 = OPEN / HIGH / WATCH
I54 = PARTIAL / OPEN
I51/I56/I57/I58/CR-01 = OPEN or PARTIAL
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
overall workflow = INCOMPLETE
```

## V21 reconciliation evidence — 2026-07-20

After the v5 regression, the live documentation inventory was rebuilt from the current tree. The
strict command

```bash
bash tests/integration/test_sc26_ae_v21_verifier.sh --expected-status PASS
```

returned exit `0` with artifact/document rows `7/10`, supplemental identities `64`, issue headings
`I50..I58`, shell/Python scope `53/35`, `TMP_ROOT_SCAN=PASS`, and `GIT_DIFF_CHECK=PASS`. The durable
identity log is:

```text
logs/i55-v21-final-reconciliation-v2-20260720.log
bytes=13465
sha256=10f31078cb2744350cb0efa587ab27a3138786571550a6c2f0fd9a4f9245ef02
CURRENT_V21_EXIT=0
```

A second run after inserting that identity into `summary.md` also returned `0`; it was captured in
`/tmp/i55-v21-post-identity-v2-20260720.log` with `POST_IDENTITY_EXIT=0`, bytes `13,470`, and SHA256
`c419ec11e320614e5801a1f4d5c6b5ca67532c0b3e01c661bab204e832015da8`. This V21 result is a local
control-plane check only and does not alter the I55, Gate B1, or release status boundary above.

### Historical V21 identity v3 (superseded by v4 and v5)

The v2 V21 identity in the preceding section is retained as historical evidence. After the V21
reconciliation checkpoint was appended to the authoritative documents, the inventory was refreshed
and a historical v3 transcript was generated:

```text
logs/i55-v21-final-reconciliation-v3-20260720.log
bytes=13471
sha256=8f8e21f5e333a963457d776f3547a5fe008a5ed28314ff5ddb3b6006386e3aea
CURRENT_V21_V3_EXIT=0
```

The post-identity v3 run returned `0` and was captured at
`/tmp/i55-v21-post-identity-v3-20260720.log` (13,473 bytes, SHA256
`ba616240aa5094db1d29dcb7b7a914df18f8adc430416eb655a7386dd9fbbbc7`,
`POST_IDENTITY_V3_EXIT=0`). The v3 transcript is historical rather than current. The later v4
transcript is also historical; the v5 and v6 transcripts are superseded snapshots as well. The
sole current V21 identity is maintained only in the single identity block in `summary.md`.

The v4 summary identity snapshot is retained for historical comparison:

```text
logs/i55-v21-final-reconciliation-v4-20260720.log
bytes=13471
sha256=ddca7a0613100efa65455a6c0710effec4c53eddc8ec5a201efe9ec36bc4c517
CURRENT_V21_V4_EXIT=0
```

The sole current V21 identity is the single identity block in `summary.md`, generated only after
the latest authoritative-document snapshot; no other report section is a current identity source.
