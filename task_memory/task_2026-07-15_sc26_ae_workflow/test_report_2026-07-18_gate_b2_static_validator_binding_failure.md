# Test Report: Gate B2 Fresh Static-Validator Binding Failure

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-18 | Recorded the rejected unsealed fresh identity, exact RED/GREEN/generation/static results, root cause, containment, and required next-session repair |

**Date:** 2026-07-18  
**Result:** **FAIL — identity rejected before seal, review, predict, or live**

## 1. Test Script Information

### Identity and paths

- Run ID: `20260718T073639Z`
- Runtime root: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z`
- Harness root: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T073639Z`
- Context: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/context/sc26-ae-gate-b2-fresh-20260718T073639Z.md`
- Intended capture root: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/output/_work/recon-task1-qwen-20260718T073639Z`
- Intended RJob: `sc26-gb-b2-fr-20260718t073639z`

### Scripts

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/test_generate_b2_harness.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/generate_b2_harness.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/seal_b2_harness.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/validate_generated_b2_harness.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/test_predict_semantic_validator.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/validate_predict_semantics.py`

### Exact commands

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
  .omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/test_generate_b2_harness.py

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
  .omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/test_predict_semantic_validator.py

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
  .omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/generate_b2_harness.py \
  --identity .omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/controller_identity.env

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
  .omx/runtime/sc26-ae-gate-b2-fresh-20260718T073639Z/validate_generated_b2_harness.py
```

All commands ran from:

```text
/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717
```

### Environment

- Controller Python: `/usr/bin/python3`, Python `3.12.3`
- Shell: GNU Bash `5.2.21(1)-release`
- Python bytecode controls: `PYTHONDONTWRITEBYTECODE=1` and `-B`
- Git branch/HEAD: `sc26-ae-exec-clean-20260717` / `3c91d15bc035d49216161c9cac874f2453b69cb9`
- Recursive submodules: Echo `1390b4416ded08bc1b9cd0620d329d81d4470bf9`; simulator `2044cccc8fff222172b7f91571a617886841001f`; collective-sim `6e06e3f5140cd4e2e7c12a35586ebcdc0f410df0`
- Functional container/GPU environment: **NOT RUN** because the host static gate failed before scheduler admission.

## 2. Validation Criteria

1. Observe generator contract RED while the generator is absent.
2. After adding the new identity-bound external generator/sealer, pass all `8` generator contract tests.
3. Re-run the byte-identical generic predict semantic validator suite and pass all `13` tests.
4. Generate exactly `18` flat pre-seal files with no subdirectory, symlink, `__pycache__`, `.pyc`, or `.pyo` entry.
5. Generate a new self-contained context and keep the capture root absent.
6. The external static validator must return exit `0` by comparing the generated candidate to the new identity's rendered expected bundle.
7. Do not run fixtures, seal, independent full-harness review, zero-drift, predict-only, live, Docker, GPU, or Qwen after any static failure.
8. Preserve D38, T175804Z, and the Session 64 sealed identity exactly.

## 3. Test Results and Evidence

### Outcome summary

| Stage | Expected | Actual | Result |
|---|---:|---:|---|
| Generator RED | Nonzero test exit while generator is absent | Exit `5`; `0` tests ran; missing-generator error | PASS |
| Generator GREEN | `8/8` tests pass | `8/8` passed in `0.177s`; exit=`0` | PASS |
| Predict semantic validator | `13/13` tests pass | `13/13` passed in `0.109s`; exit=`0` | PASS |
| Generation | Exit `0`; 18 flat files; capture absent | Exit=`0`; files=`18`; bytes=`33,741`; capture=`false` | PASS |
| Static validator | Exit `0` | Exit=`1`; stopped at stale payload-byte assertion | **FAIL** |
| Post-validator fixtures | Run only after static PASS | `NOT RUN` | HELD |
| Seal | Run only after all pre-seal gates PASS | `NOT RUN`; `seal.json` absent | HELD |
| Independent full-harness review | Run only after seal | `NOT RUN` | HELD |
| Predict-only | Run only after review and zero-drift | `NOT RUN` | HELD |
| Live B2 | Run only after process-plus-semantic predict PASS | `NOT RUN` | HELD |

### Key metrics

| Metric | Stale validator value / acceptance | Actual | Delta / Error |
|---|---:|---:|---:|
| Pre-seal file count | `18` | `18` | `0` |
| Pre-seal payload bytes | `33,068` | `33,741` | `+673` bytes (`+2.035200%`) |
| Context bytes | `7,761` | `8,940` | `+1,179` bytes (`+15.191341%`) |
| Context SHA256 | `73439f6ec477310c1960894866639396484d533ccff8ad9f41a06360a86682c5` | `a76a2c42ddafc5a7fce2cc264b1fed62fa9f293bf083d96288aba9d9f9a651e3` | Mismatch, as expected for a new context |
| Harness subdirectories | `0` | `0` | `0` |
| Harness symlinks | `0` | `0` | `0` |
| Harness bytecode files | `0` | `0` | `0` |
| Runtime bytecode files | `0` | `0` | `0` |
| Seal files | `0` before seal | `0` | `0` |
| Capture root | Absent before live | Absent | `0` created |
| RJob / GPU / workload | `0` before predict/live gates | `0 / 0 / 0` | `0` |

### Root cause

`validate_generated_b2_harness.py` was copied as if it were generic, but it is identity-bound. It retained three Session 64 constants:

```text
pre-seal bytes = 33068
context bytes = 7761
context SHA256 = 73439f6ec477310c1960894866639396484d533ccff8ad9f41a06360a86682c5
```

The new generator correctly expanded the authority snapshot to `16` entries and added the Session 68 WATCH context, so the candidate became `33,741` bytes and the context became `8,940` bytes with a new SHA256. The validator failed first at its stale `33,068` assertion before reaching the two latent stale context assertions.

This is an external controller validator re-binding defect. It is not a defect in Megatron, the canonical image, conda environments, H800 resources, Qwen, trace, memory, or the generated worker payload.

### Required root-cause repair

Do not replace the stale constants with another manually copied byte total. In the next permitted session and a wholly new identity, the static validator must derive its expected candidate bytes and context bytes/SHA256 from `generate_b2_harness.py`'s freshly rendered `RenderedBundle`, then compare every emitted file byte-for-byte. This removes the identity-transfer error at its source rather than applying a scaling factor or fallback.

### Evidence identities

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `tdd_red.log` | `1,161` | `80ab929dd45f9a3dd39910cb598b2c9585dfdf41277ca4a16d2ce781d534ef42` |
| `tdd_green_attempt1.log` | `1,208` | `e5a069e82b2f824cf2255b93d1a55c7fab41ef5d8b6645794c10a9a4a022b0af` |
| `predict_semantic_tests_attempt1.log` | `1,779` | `4fc2609c77fff51dfc9140faf6c1220af1d26325192854bb3324a59959b95ed7` |
| `generation.log` | `559` | `80f7f2986fed064243882d4b1ea4cd2e27cc42054f62a3f4249df108ebd4cf2b` |
| `static_validation_attempt1.log` | `927` | `d7e2baa5ba89f36526ec17800a7376ed35c44b13be948c1477b9eb461bd41cc8` |
| `failure_snapshot.json` | `4,758` | `e8617a4ae84922453839d1b88525ba44ff53b5b9a1ab3bed1495634c4382c347` |

### Containment evidence

- Failed harness remains unsealed and immutable-by-disposition: `18` files / `33,741` bytes, subdirectories/symlinks/bytecode=`0/0/0`.
- Context remains `8,940` bytes at SHA256 `a76a2c42ddafc5a7fce2cc264b1fed62fa9f293bf083d96288aba9d9f9a651e3`.
- `seal.json`, capture root, predict logs/results, live logs/results, RJob, GPU worker, and workload artifacts are absent.
- Session 64 seal remains `2,948` bytes at SHA256 `7db55d97cb07c6ed2bddb951510f906ea8dbc8bde226c8f622232ded540da395`.
- T175804Z remains `24` files / `100,490` bytes / `1` subdirectory / `2` `.pyc` / `43,165` bytecode bytes.
- Clean execution worktree status remains only `?? SC26-AE/`; `git diff --check` exits `0`.

## 4. Final Disposition

`FAIL — REJECTED UNSEALED IDENTITY.`

The static gate failed, so this identity must not be patched, sealed, reviewed for execution, predicted, or run live. Session 68 permits only one identity in this session; therefore no second identity is generated now. B2 remains open, and B3/B4/B5 plus Phases 1–9 remain blocked.
