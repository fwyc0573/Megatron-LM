# Test Report: Gate B2 Worker Authority-Path Failure

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-18 | Added fresh generator/static/zero-drift visibility TDD and generic fixture metrics |
| 2026-07-18 | Recorded 12/12 final unit GREEN and terminal documentation candidate integration PASS with exact metrics |
| 2026-07-18 | Added terminal RUN_ID=20260718T085830Z evidence, root-cause analysis, TDD results, and candidate validation criteria |

**Date:** 2026-07-18  
**Gate result:** Gate B2 | FAIL / BLOCKED  
**Terminal status:** `BLOCKED_WORKER_INVISIBLE_AUTHORITY_PATH`  
**Run identity:** `RUN_ID=20260718T085830Z`  
**RJob:** `sc26-gb-b2-fr-20260718t085830z`

## 1. Test Script Information

### Scripts

- `/data/ycfeng/Megatron-LM-sc26-ae/tests/unit_tests/sc26_ae/test_gate_b2_terminal_docs_validator.py`
- `/data/ycfeng/Megatron-LM-sc26-ae/tests/integration/sc26_ae/validate_gate_b2_terminal_docs.py`
- Sealed worker entry: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T085830Z/worker_entry.sh`
- Terminal audit: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T085830Z/post_live_terminal_audit.json`

### Reproducible local commands

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -B -m unittest -v \
  tests/unit_tests/sc26_ae/test_gate_b2_terminal_docs_validator.py

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -B \
  tests/integration/sc26_ae/validate_gate_b2_terminal_docs.py
```

The sealed GPU commands are preserved in the immutable harness and are not repeated because `retry_allowed=false` and `further_live_authorized=false`.

### Environment

- Controller Python: `/usr/bin/python`, Python `3.12.3`
- Local validator mode: `-B` and `PYTHONDONTWRITEBYTECODE=1`
- Authority worktree: `/data/ycfeng/Megatron-LM-sc26-ae`, branch `sc26-ae`, HEAD `3c91d15bc035d49216161c9cac874f2453b69cb9`
- Execution worktree: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`, branch `sc26-ae-exec-clean-20260717`, same HEAD
- Sealed GPU image: `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`
- Worker mount: `/data:/data`

## 2. Validation Criteria

1. Predict process and generic semantic validation each run exactly once and pass.
2. At least one eligible H800 candidate exists before the one live authorization.
3. Live runs exactly once. A nonzero live exit retains Gate B2 as FAIL.
4. All `23` authority hashes must be readable and match inside the worker before context, GPU, product, or workload execution.
5. Capture and `qualification_result.json` must exist only after the workload reaches its output path; their absence proves no B2 success.
6. Terminal evidence binds the exact worker files, active-RJob count, D38 `4/4` identities, Session 70 frozen identities, clean execution Git status, and bytecode count.
7. Documentation states no patch/retry/reuse/second identity and preserves B3/B4/B5 plus Phase 1–9 as blocked.

## 3. Test Results and Evidence

### Gate execution matrix

```text
Eligible H800 candidates | 10
Authority hashes | 23 | 22 | 1 failed
Workload started | false
```

| Check | Expected for B2 PASS | Actual | Delta / Result |
|---|---:|---:|---|
| Predict invocation count | 1 | 1 | 0 — PASS |
| Semantic validator count | 1 | 1 | 0 — PASS |
| Live invocation count | 1 | 1 | 0 — executed once |
| Predict process exit | 0 | 0 | 0 — PASS |
| Semantic status | PASS | PASS | match |
| Eligible H800 candidates | >= 1 | 10 | +9 — PASS |
| Available GPUs per candidate | >= 1 | min=4, max=8 | min margin=+3 |
| Live process exit | 0 | 1 | +1 — FAIL |
| Authority hashes | 23 | 22 | 1 failed |
| Source manifest hashes | 6 | 6 | 0 — PASS |
| Workload started | true | false | required path not reached |
| Capture root exists | true | false | required output absent |
| `qualification_result.json` exists | true | false | required output absent |
| Worker evidence files | 3 terminal files | 3 | 0 |
| Active RJob name matches after exit | 0 | 0 | 0 |
| Runtime bytecode count | 0 | 0 | 0 |
| Gate B2 | PASS | FAIL / BLOCKED | acceptance not met |

### Timing and artifact metrics

| Metric | Expected / Bound | Actual | Delta / Note |
|---|---:|---:|---|
| Predict elapsed | > 0s | 1s | positive |
| Live elapsed | terminal observation | 133s | worker entered, then preflight failed |
| Terminal audit | exact identity | 11,106 bytes | SHA256 `4114c6bbbb3c00111f55a42a4c4f6f4c45e8aa2d4518369227bca654bd4f40f2` |
| Live log | exact identity | 9,904 bytes | SHA256 `9cb84c032c80e8431c4c6e6099c7ca7bebfb669771ddbce37060e4931a8b75a2` |
| Predict semantic JSON | exact identity | 2,119 bytes | SHA256 `26241496ea5b79aa18e1b4895ba8600b266ac00ca5deb7df88d051885414e69e` |
| Independent postmortem | provider exit 0 | 8,935 bytes | APPROVE; SHA256 `525f2875b3b067d8f066c09ef9cad5c0a78c630f352904c6a1695ea94bd9f2a8` |
| D38 exact identities | 4 | 4 | 0 drift |
| Session 70 frozen identities checked | 8 | 8 | 0 drift |

### Worker evidence identities

| File | Bytes | SHA256 | Mode | UID:GID |
|---|---:|---|---:|---:|
| `seal_verification.log` | 55 | `e2d70e19f479e60f69af1e7839f8219b9cb6c60a5d5fd08a3ec761fd9284ad06` | `0644` | `0:0` |
| `source_manifest_check.log` | 201 | `4b2ff00c20af427c6ae23243aa53e0b5d243eb4f4822761b52b2d60a65a8c478` | `0644` | `0:0` |
| `authority_hash_check.log` | 2,775 | `ee75fb4496fd04b9a1101abb7fa6b57268b74686fd1c6f815e771052d1bb871b` | `0644` | `0:0` |

### Root-cause evidence

- Authority hashes | 23 | 22 | 1 failed.
- The failed controller artifact is under `/home/i-fengyicheng/...`, bytes/SHA256=`3,463/d50c371d0f4c887ad092777841c1eaadc6ea195ec37790fbfe04c99eea071170`.
- The worker mounts only `/data:/data`; `/home` is not worker-visible.
- Failure occurred in the worker authority-hash check before context verification, GPU identity, product import, workload, capture, trace, or memory tracing.
- Root-cause category: `controller_only_authority_path_not_mounted_in_gpu_worker`.

### TDD and local validator results

| Stage | Tests / Checks | Result | Evidence |
|---|---:|---|---|
| Missing validator RED | import | PASS as RED; exit=1 | `logs/gate_b2_terminal_docs_tdd_red_20260718.log` = `1,517/46cb0b09f07a38cd8813815ba00a2495de2cf4d737b7b5cb2a7500a471b9d7e6` |
| First unit attempt | 10 | FAIL, 9/10 | Success fixture lacked Modification History; implementation rule retained |
| Unit GREEN Retry-1 | 10 | PASS, 10/10 | `2,105/132e4438b08e5ca61d7464b39a8ff541cbd8553b08868f1b9ba4a5111ef320a9` |
| Semantic schema RED | 1 | PASS as RED; exit=1 | Missing exact flat-schema helper |
| Intermediate unit GREEN | 11 | PASS, 11/11 | `2,324/d271bc088c95f4f80c1a5ca677bea37d169fca306b4c0a33cae981fc36437ec7` |
| First integration attempt | evidence schema | FAIL; exit=1 | Incorrect nested `summary` assumption; terminal evidence unchanged |
| Intended document RED | terminal doc tokens | PASS as RED; exit=1 | `269/da43dbaf5012e1c6e15ab6cb890cd8a44e419cf60d55aaeb8fb68a660e61e179` |
| Heading-scope regression RED | 1 | PASS as RED; exit=1 | Inline heading token exposed raw substring counting |
| Final unit GREEN | 12 | PASS, 12/12 | `2,543/7bb231f3633413d96c9ae59dc92cf6b097dce43ef0ec038149af955593bbbc64` |
| Candidate integration Attempt 1 | 7 documents | FAIL; exit=1 | Anchored heading scope required |
| Candidate integration Retry-1 | 7 documents | FAIL; exit=1 | Missing terminal status in `progress.md` |
| Candidate integration Retry-2 | 7 documents | FAIL; exit=1 | Incorrect `/home` negative-test quoting in `issues.md` |
| Candidate integration Retry-3 | 7 documents / 36 tokens | PASS | `645/dc3818082a4fdcdaf4e6065cbfa89e88395e46803082aa1d01f4a279af46f058` |

Candidate Retry-3 actual metrics: Python/`tee`=`0/0`; elapsed=`149,614,032ns`; current bound identities=`16`; D38=`4/4`; Session 70 frozen identities=`8/8`; worker files=`3`; eligible H800=`10`; capture absent; bytecode=`0`; Gate B2=`FAIL_PRE_WORKLOAD_AUTHORITY_PREFLIGHT`. Process evidence is `248/a9ac065a4a008d192dd6a6d811a13e81fa8bf341512e489129b5b6da987b2e2e` (bytes/SHA256). A final no-further-edit validation is required after this result is recorded.

## 4. Failure Diagnosis and Resolution

The B2 failure is not a GPU scheduling, image, conda, Megatron, Qwen, trace, or memory failure. Those workload stages were never reached. The generator and static review chain omitted one cross-boundary invariant: every path that the worker must hash must be reachable inside the exact sealed mount set.

The future root fix is structural and fail-fast:

1. Reject non-worker-visible authority paths before generation and seal.
2. Add the exact negative test `/home authority path must be rejected before generation and seal`.
3. Bind D47 authority under `/data` or inside the sealed harness.
4. Do not add a temporary mount and do not skip a hash.
5. Use an entirely fresh identity in a later authorized continuation.

## 5. Terminal Containment

- `retry_allowed=false`
- `further_live_authorized=false`
- `second_identity_allowed_this_continuation=false`
- `capture_root_exists=false`
- `qualification_result_exists=false`
- B3/B4/B5=`BLOCKED_NOT_RUN`
- Phase 1–9=`BLOCKED_NOT_RUN`
- Persistent task goal remains active.

## 6. Fresh Worker-Visible-Authority Root-Fix TDD

**Fresh identity:** `RUN_ID=20260718T101257Z`  
**Current scope:** generator and external static-validator source/tests only  
**Gate B2 status:** still `FAIL / BLOCKED`; no candidate generation, seal, predict, semantic, or live action has run for this identity.

### 6.1 Test Script Information

- Generator test: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/test_generate_b2_harness.py`
- Generator: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/generate_b2_harness.py`
- Static test: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/test_static_validator_derivation.py`
- Static validator: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/validate_generated_b2_harness.py`
- Environment: system `python3`, Python bytecode disabled through `python3 -B` plus `PYTHONDONTWRITEBYTECODE=1`; clean execution branch/HEAD=`sc26-ae-exec-clean-20260717/3c91d15bc035d49216161c9cac874f2453b69cb9`.

Reproducible commands:

```bash
RUNTIME=/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z
PYTHONDONTWRITEBYTECODE=1 python3 -B "$RUNTIME/test_generate_b2_harness.py"
PYTHONDONTWRITEBYTECODE=1 python3 -B "$RUNTIME/test_static_validator_derivation.py"
```

### 6.2 Validation Criteria

1. Controller D47 source remains the exact `/home` artifact, but the worker manifest contains only the exact sealed copy under the fresh `/data/.../HARNESS_ROOT`.
2. Canonical volume parsing accepts `/data:/data` and rejects missing, malformed, duplicate, traversal, root, and non-identity mappings.
3. Worker paths reject `/home`, `/database`, `/data2`, relative, traversal, NUL, duplicate, missing, and symlink-escape targets.
4. Worker manifest must be nonempty, LF-canonical, unique, exact-order/exact-set, lowercase-SHA256, absolute-path, and byte/hash consistent.
5. `d47_question_authority.json` must be exactly `3,463` bytes with SHA256 `d50c371d0f4c887ad092777841c1eaadc6ea195ec37790fbfe04c99eea071170`; any actual candidate byte drift must fail.
6. Static validation must implement its volume/path logic independently rather than call the generator's corresponding helpers.
7. Terminal `20260718T085830Z`, Session 70, D38, clean worktree, fresh output absence, and runtime bytecode invariants must remain unchanged.

### 6.3 Test Results and Evidence

| Stage | Expected | Actual | Result / Delta |
|---|---:|---:|---|
| Static missing-implementation RED | validator absent; nonzero exit | `0` tests; Python/`tee=5/0` | PASS as RED |
| Static RED elapsed | positive | `74,257,828ns` | positive |
| GREEN Attempt 1 | strict authority binding | `0` tests; `progress.md` old/actual bytes=`250,156/251,884` | expected fail-fast drift |
| Controller authority refresh | same path/cardinality/order | `27/27`; progress=`251,884/58b69139...fc5c53` | PASS |
| Generator regression | `12/12` | `12/12`, Python/`tee=0/0` | PASS |
| Generator regression elapsed | positive | `429,307,005ns` | positive |
| Static Attempt 2 | expose remaining defect | `21/22` | fixture bug found |
| Static final GREEN | `22/22` | `22/22`, Python/`tee=0/0` | PASS |
| Static final elapsed | positive | `434,255,744ns` | positive |
| D47 expected vs actual bytes | `3,463` | `3,463` | delta=`0` |
| D47 expected vs actual SHA256 | `d50c...1170` | `d50c...1170` | match |
| Terminal audit expected vs actual bytes | `11,106` | `11,106` | delta=`0` |
| Terminal audit expected vs actual SHA256 | `4114c6...40f2` | `4114c6...40f2` | match |
| Fresh candidate/context/capture | absent/absent/absent | absent/absent/absent | PASS |
| Fresh runtime bytecode | `0` | `0` | PASS |

Final evidence identities:

| File | Bytes | SHA256 |
|---|---:|---|
| `test_static_validator_derivation.py` | `20,939` | `83befc993b6814cdfd1c2691f6a90f5f0e0cd71721f58db7f3d6a8a0bf46c346` |
| `validate_generated_b2_harness.py` | `45,008` | `31026cf737f1ee9659786a97fb1c49a46f705431b366d3be8f0cbe52df3b3c9a` |
| `tdd_static_validator_red.log` | `1,211` | `2901923eec88a50b8ce45101686bac85cc0d3836bef922aa3f5a9d761a80114b` |
| `tdd_static_validator_green_attempt3.log` | `3,881` | `584d530ccd1c3f80d4fc4450b004506b942f1bbc79b315fe86ab10607cbe06f2` |

### 6.4 Failure Resolution

- Attempt 1 was not weakened or bypassed: the authority checkpoint itself changed `progress.md`, so the generator correctly rejected the old frozen identity. The refresh changed only the `27` current bytes/SHA256 pairs.
- Attempt 2 was a test-fixture defect. Removing the only predict volume correctly exercised the missing-volume branch. The intended drift test now supplies two individually valid but different mappings (`/scratch:/scratch` versus `/data:/data`), so the validator reaches and proves the predict/live drift branch.
- The fresh identity remains pre-generation. These local GREEN results do not authorize seal, predict, live, Gate B2 PASS, or downstream gates.

### 6.5 Zero-Drift Visibility and Generic Fixture Results

Additional scripts:

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/test_validate_sealed_zero_drift.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/validate_sealed_zero_drift.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/test_predict_semantic_validator.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T101257Z/test_post_validate_fixtures.py`

| Suite / Metric | Expected | Actual | Result / Delta |
|---|---:|---:|---|
| Zero-drift visibility missing-module RED | nonzero | Python/`tee=5/0`; `0` tests | PASS as RED |
| Zero-drift RED elapsed | positive | `67,342,935ns` | positive |
| Zero-drift visibility GREEN | `7/7` | `7/7`; Python/`tee=0/0` | PASS |
| Zero-drift GREEN elapsed | positive | `119,713,687ns` | positive |
| Predict semantic unit suite | `13/13` | `13/13`; Python/`tee=0/0` | PASS |
| Predict semantic elapsed | positive | `182,419,249ns` | positive |
| Post-validate cases | `17` | `17` | delta=`0` |
| Post-validate positive cases | `1` | `1` | delta=`0` |
| Post-validate negative cases | `16` | `16` | delta=`0` |
| Post-validate elapsed | positive | `1,014,906,836ns` | positive |
| Formal zero-drift before seal/review | fail fast | `seal/review binding is not finalized` | PASS; no guessed binding |

The zero-drift visibility helper reads the actual `predict_command.sh`, `live_command.sh`, and `authority_hashes.txt`, then invokes the independent static parser/projection/resolution functions. The formal validator is intentionally incomplete until the actual self-excluding seal and independent full-harness review have immutable bytes/SHA256 identities. This is a strict phase boundary, not a fallback.

The authority-refresh helper also received an idempotence correction after a compile-before-write failure exposed whole-line AST splice drift. Two immediate runs now produce the same generator identity: expected/actual bytes=`73,610/73,610`, delta=`0`; expected/actual SHA256=`a0df77bd79e92dfd78fdc1298e129abaafaa8f5f88bc246bdc2c2e3fcec93d58/a0df77bd79e92dfd78fdc1298e129abaafaa8f5f88bc246bdc2c2e3fcec93d58`, match.
