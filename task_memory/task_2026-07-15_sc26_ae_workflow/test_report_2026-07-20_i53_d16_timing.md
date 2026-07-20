# Test Report — I53/F10-04 D16 Model-Aware Timing Contract

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Added V21 harness-drift evidence, shell-verifier metrics, and final local documentation/static boundary to the I53 report |
| 2026-07-20 | Recorded the model-aware GPT/MoE D16 timing contract, RED→GREEN evidence, synthetic numeric values, and the remaining independent-preflight boundary |

## Scope and disposition

This report covers the Task1 timing metadata and validator contract in
`SC26-AE/lib/task1_trace.sh`. The change separates GPT-175B's eight-rank representative timing
estimate from the frozen 256-rank MoE D16 arithmetic used for Qwen3-A30B and DeepSeek-V3.

This is a **local synthetic/controller report**, not a hardware or release qualification report.
No GPU, RJob, Docker, H800, real Nsight Systems SQLite/NVTX qualification, or independent
rank-0-only preflight was run. The pinned `Echo-slowdown` checkout was not modified.

The current boundary is intentionally unchanged:

```text
evidence class = local_synthetic_not_gpu_qualification
I53 = OPEN / HIGH / WATCH
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## Test Script Information

### Repository and changed surface

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Production contract: `SC26-AE/lib/task1_trace.sh`
- D16 unit contract: `tests/unit/test_sc26_ae_task1_d16_timing.sh`
- Task1 integration contract: `tests/integration/test_sc26_ae_task1_contracts.sh`
- Source-provenance unit: `tests/unit/test_sc26_ae_task1_source_provenance.sh`
- Task1 smoke: `tests/e2e/test_sc26_ae_task1_smoke.sh`
- Fresh chain: `tests/e2e/test_sc26_ae_fresh_chain.sh`
- Clean-clone replay: `tests/e2e/test_sc26_ae_clean_clone_replay.sh`
- Related Task3 regression: `tests/integration/test_sc26_ae_task3_contract.sh`

### Exact commands

All commands below were run from the repository root. The final local matrix was persisted at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-final-local-verification-20260720.log
```

```bash
bash -n SC26-AE/lib/task1_trace.sh
bash -n tests/unit/test_sc26_ae_task1_d16_timing.sh
bash -n tests/integration/test_sc26_ae_task1_contracts.sh
bash -n tests/e2e/test_sc26_ae_fresh_chain.sh
bash -n tests/e2e/test_sc26_ae_clean_clone_replay.sh
git diff --check
test -z "$(git -C Echo-slowdown status --porcelain --untracked-files=all)"

bash tests/unit/test_sc26_ae_task1_source_provenance.sh
bash tests/unit/test_sc26_ae_task1_d16_timing.sh
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/e2e/test_sc26_ae_task1_smoke.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
bash tests/e2e/test_sc26_ae_clean_clone_replay.sh

python -m pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_seal_qualification.py \
  tests/unit/test_sc26_ae_package_prebaked.py
```

### Environment

| Item | Observed value |
|------|----------------|
| `CONDA_DEFAULT_ENV` | `none` / controller shell |
| Python executable | `/usr/bin/python` |
| Python version | `3.12.3` |
| pytest | `9.1.1` |
| Torch | `2.5.1+cu124` |
| `torch.cuda.is_available()` | `False` |
| CUDA device count | `0` |
| Pinned Echo status | empty `git status --porcelain --untracked-files=all` |

## Root cause and expected behavior

The pre-fix writer applied `rank0 timing × 256` and the `7200`-second fresh-capture gate to every
model. GPT-175B captures eight PP-stage representative ranks, so that field was not a valid GPT
estimate. The measured rank-0 interval is also collected during an already-running selected-rank
capture; it cannot serve as a decision made before starting a full capture.

The corrected contract is:

| Model | `d16_gate_applicable` | `estimate_rank_count` | Gate fields |
|-------|----------------------:|----------------------:|-------------|
| GPT-175B | `false` | `8` | absent; diagnostic representative estimate only |
| Qwen3-A30B | `true` | `256` | integer threshold `7200`, result `pass` or `prebaked_required` |
| DeepSeek-V3 | `true` | `256` | integer threshold `7200`, result `pass` or `prebaked_required` |

Every model must satisfy:

```text
estimated_full_seconds = single_rank_elapsed_seconds × estimate_rank_count
```

The validator rejects unsupported models, non-boolean applicability, model/count mismatch,
forbidden GPT gate fields, missing or inconsistent MoE gate fields, non-positive or non-finite
timing, malformed/duplicate rank timing, rank inventory mismatch, missing fake rank zero, and
summary/metadata disagreement. No automatic source switch or fallback is implemented.

## Validation Criteria

1. GPT metadata must explicitly state that D16 is not applicable, use the eight selected
   representative ranks, and omit the MoE-only threshold/result fields.
2. Qwen3-A30B and DeepSeek-V3 metadata must retain the 256-rank estimate and exact `7200`-second
   gate arithmetic, including the `pass`/`prebaked_required` boundary.
3. Negative inputs and cross-file inconsistencies must fail closed before marker publication.
4. Existing Task1 semantic, memory, source-provenance, Task3, fresh-chain, and clean-clone
   contracts must remain green.
5. All changed shell paths must pass `bash -n`, `git diff --check` must pass, and pinned Echo must
   remain clean.
6. Results must remain labeled synthetic/controller evidence and must not promote any qualification
   or release status.

## RED → GREEN evidence

The first targeted run intentionally exposed the contract mismatch:

| Stage | Log | Exit | Evidence |
|-------|-----|-----:|----------|
| RED | `logs/i53-d16-model-aware-red-20260720.log` | `1` | Existing validator rejected the writer's missing `fresh_capture_gate_threshold_seconds` field; bytes `129`, SHA256 `cc753674701a70b68a1de453e40ef6808884580fdd6cc3d26c1365556975ccea` |
| GREEN unit | `logs/i53-d16-model-aware-green-unit-20260720.log` | `0` | `PASS_COUNT=29`; bytes `1,256`, SHA256 `8d412b844d392055e40bb5292b23915951c48d48f4c16722eed6ae5143052e13` |
| GREEN integration | `logs/i53-d16-model-aware-green-integration-20260720.log` | `0` | `PASS_COUNT=35`; bytes `2,381`, SHA256 `2a32a75a52a9782ad45317e949ea7ca2ac0139861e444bc1df641bb473803594` |

The RED was a harness/contract mismatch, not evidence of a hardware failure. The fix made the
validator model-aware and added the GPT/MoE negative cases before rerunning the affected suites.

## Test Results and Evidence

### Targeted suite matrix

| Test suite | Exit | Actual result |
|------------|-----:|---------------|
| `test_sc26_ae_task1_source_provenance.sh` | `0` | `PASS_COUNT=2` |
| `test_sc26_ae_task1_d16_timing.sh` | `0` | `PASS_COUNT=29` |
| `test_sc26_ae_task1_contracts.sh` | `0` | `PASS_COUNT=35` |
| `test_sc26_ae_task3_contract.sh` | `0` | `PASS_COUNT=10` |
| `test_sc26_ae_task1_smoke.sh` | `0` | `SMOKE_PASS_COUNT=1`, `REAL_GPU_WORKLOAD_COUNT=0` |
| `test_sc26_ae_fresh_chain.sh` | `0` | `CHAIN_PASS_COUNT=1` |
| `test_sc26_ae_clean_clone_replay.sh` | `0` | public entries `3/3/3`, setup cases `6`, chain `1`, all clone statuses clean |
| Artifact/sealer/package pytest subset | `0` | `94 passed in 49.60 s` |

The complete fresh local matrix log is `10,193` bytes with SHA256
`b94f3c9d61f13641e7564ad2c27ec122d11d8d02fcb09ebb5b26970f2245b586`. The D16 regression matrix
log is `5,212` bytes with SHA256
`730e0d6df02318b04c0fea900411e7bd4be932231b4ad0932a74c16f17df3d51`.

### Synthetic D16 numeric metrics

The Task1 integration fixture's actual metadata values were read back from the generated JSON,
not recomputed for this report:

| Model | Rank-0 elapsed (s) | Estimate count | Estimated full seconds | Threshold (s) | Result |
|-------|-------------------:|---------------:|-----------------------:|---------------:|--------|
| GPT-175B | `0.009844431` | `8` | `0.078755448` | not applicable | not applicable |
| Qwen3-A30B | `0.013280299` | `256` | `3.399756544` | `7200` | `pass` |
| DeepSeek-V3 | `0.010431187` | `256` | `2.670383872` | `7200` | `pass` |

The fixture metadata root was
`/data/ycfeng/tmp/sc26-ae-task1-contracts.uk1UdS`. These are synthetic values and cannot establish
real elapsed time, full-rank coverage, or a pre-capture decision.

### Fresh-chain numeric metrics

The fresh synthetic Task1→Task2→Task3 chain reported:

| Metric | Actual |
|--------|-------:|
| Task1 trace files | `4` |
| Task1 memory JSON files | `4` |
| Task2 dataset rows | `2` |
| Task2 validation MSE | `3.0` |
| Task2 test MSE | `0.5` |
| Model reload max absolute prediction delta | `0.0` |
| Task3 rank-0 step | `22.5 ms` |
| Task3 forward/backward/optimizer | `6.0 / 11.0 / 2.5 ms` |
| Simulator load/execution/wall | `0.125 / 0.375 / 0.5 s` |
| Process wall clock | `0.954087 s` |
| Peak RSS | `51,528 KiB` (`0.04914093 GiB`) |
| Tested host allocation | `32 MiB` |

Fresh-chain root:
`/data/ycfeng/tmp/sc26-ae-fresh-chain.rSRT1u`; clean-clone replay root:
`/data/ycfeng/tmp/sc26-ae-clean-clone-replay.0vg0sF`. Both are temporary synthetic fixtures, not
release artifacts.

### Static and source checks

The final matrix observed exit `0` for all five `bash -n` commands, `git diff --check`, and the
empty pinned Echo status assertion. No GPU workload was attempted (`REAL_GPU_WORKLOAD_COUNT=0`).

## Remaining issues and required next steps

The MoE `fresh_capture_gate_result=pass` values above are arithmetic observations from timing
inside a selected capture. They do **not** mean that a rank-0-only probe ran before full capture,
that a full 256-rank fresh capture completed within `7200` seconds, or that a source was switched
to a prebaked bundle. Closure still requires:

1. An independent rank-0-only preflight implementation and real decision gate before full capture.
2. Real H800 timing and complete real trace/memory/SQLite/NVTX semantics.
3. Canonical `nsys` executable identity/version and producer source snapshot/provenance.
4. Independent I51/I54/I55/I56/I57/I58/CR-01 controls and Gate B1 approval.
5. A complete real and release-qualified pre-dataset.

Until those items are proven, the authoritative disposition remains `I53=OPEN`,
`Gate B1=BLOCKED`, both pre-datasets `NOT QUALIFIED`, and `AE-ready=NO`.


## V21 documentation/static harness drift checkpoint — 2026-07-20

This addendum records the final local V21 verifier correction required after the D16 change. It is
part of the I53 controller report and does not expand the I53 qualification claim.

### Test Script Information

- Verifier implementation: `tests/integration/sc26_ae_v21_verifier.py`
- Required shell entry point: `tests/integration/test_sc26_ae_v21_verifier.sh`
- Evidence logs:
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-v21-shell-scope-red-20260720.log`
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-v21-shell-scope-green-20260720.log`
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-v21-verifier-green-20260720.log`
  - `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-v21-shell-verifier-pass-20260720.log`

Exact commands from the repository root:

```bash
python3 -m py_compile tests/integration/sc26_ae_v21_verifier.py
git diff --check
bash tests/integration/test_sc26_ae_v21_verifier.sh --expected-status PASS
```

Environment: controller shell, Python `3.12.3`; no CUDA device was available. `Echo-slowdown`
status remained empty.

### Validation Criteria

1. Retained Session 47 transcript markers must remain `SHELL_SCOPE_COUNT=52` and
   `SHELL_SYNTAX_COUNT=52`.
2. Current live shell enumeration must be `53/53` after adding the D16 unit shell file; the V21
   verifier shell is self-excluded.
3. Current Python syntax scope must remain `35/35`.
4. V21 must return exit `0`, preserve artifact/document rows `7/10`, and preserve the blocked
   synthetic-only status boundary.

### RED → GREEN and actual metrics

Before inventory refresh, the post-documentation shell run correctly failed closed on a stale
authoritative digest: `logs/i53-v21-post-doc-stale-inventory-red-20260720.log`, exit `1`, bytes
`706` and SHA256 `be966073732b7ea6cc394c6478859d69a8de61a02673740b1d33af96e7999696`; the first
error was `progress.md` expected `241505` bytes versus actual `244249`. This RED is retained as
harness evidence, not a product failure.

| Check | Exit | Actual evidence |
|---|---:|---|
| Stale live-scope expectation | `1` | `53` live files versus hard-coded `52`; log `9482` bytes, SHA256 `51d84029119161b276e6e1b3252e8d3f89cb524c8c58e64e217ed53332e441cf` |
| Historical-marker drift attempt | `1` | archived transcript still has `52`; log `9176` bytes, SHA256 `dd651767b6c8930ad5d94df509b28728acdcd417a2cd6e91bc512bb8a318fcf5` |
| Direct Python verifier | `0` | rows `7/10`, supplemental `40`, live shell/Python `53/35`; log `9531` bytes, SHA256 `33113ad76df4b220ec9af8c023b3946b2c64278f4adf6b356a4e33e96168425f` |
| Required shell verifier | `0` | `SESSION47_FINAL_VERIFICATION_V21=PASS`, rows `7/10`, live shell/Python `53/35`; log `9531` bytes, SHA256 `f95275a5526a97cab931d64b6f6da3f6b5529c30e70f24f1ac3ec2901c48c3f2` |
| Python compile / diff check | `0 / 0` | empty-output logs; each is `0` bytes with SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The file named `i53-v21-shell-scope-green-20260720.log` is explicitly a failed attempt, not a
GREEN result. The corrected implementation keeps historical marker validation at `52` and live
scope validation at `53`.

### Result and boundary

The V21 documentation/static harness is locally GREEN. This does not represent a real worker,
H800, SQLite/NVTX, independent preflight, full-rank capture, or release qualification. The
authoritative boundary remains:

```text
evidence class = local_synthetic_not_gpu_qualification
I53 = OPEN / HIGH / WATCH
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## Final identity and independent-review checkpoint — 2026-07-20

The independent reviewer `/root/audit_i54_i55` confirmed the D16 model contract and the
historical-52/current-53 split, while correctly blocking the transient snapshot that still had
concurrent document drift. After removing the duplicate status substring and refreshing the
inventory, candidate2 returned exit `0` and became the non-self-referential V21 identity:

```text
path=task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-v21-final-candidate2-20260720.log
bytes=10134
sha256=9d51b3371eec34f4f4d0bb3d34f7223dfde724a1f359057803aa8eb66ae97a6c
exit=0
```

A subsequent shell verification after that identity update returned exit `0` in
`logs/i53-v21-final-post-identity-20260720.log` (`10138` bytes,
SHA256 `077285e55c10100fc9687b6ce49102d6c7ab6365357ccb2d06a0c8452abf9e34`). It reported
artifact/document rows `7/10`, supplemental identities `43`, historical markers `52/52`, live
shell/syntax `53/53`, Python scope/syntax `35/35`, and `SESSION47_FINAL_VERIFICATION_V21=PASS`.

The reviewer also confirmed that D16 remains selected-capture observation rather than an
independent rank-0 preflight. Therefore the report's local GREEN result does not promote any
hardware, qualification, source-selection, or release status.
