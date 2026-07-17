# Test Report: Gate B1 D27 and Echo Live Qualification

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-17 | Added the fresh post-closure validator rerun, validator-test root-cause corrections, and final stop-state evidence |
| 2026-07-17 | Recorded final D1–D28 docs, artifact, hash, Markdown, and Git-scope validation evidence |
| 2026-07-17 | Recorded follow-up independent D28 APPROVE after both WATCH precision findings were closed |
| 2026-07-17 | Applied the independent D28 WATCH clarification that Task A7 review is the current sequential gate |
| 2026-07-17 | Created the consolidated D27/Echo Gate B1 evidence, incident, numeric-metric, and D28 blocking report |

## 1. Scope and Verdict

This report consolidates historical Gate B1 runtime evidence. It does not authorize or execute a new RJob.

| Branch | Current result | Reason |
|--------|----------------|--------|
| D27 one-H800 MemoryTracker | **PASS** | Live CUDA/NVML allocation and non-empty memory JSON passed. |
| Echo exact-two-H800 | **BLOCK** | No completed live train/save/reload/prediction-parity record exists. |
| Integrated B1 | **BLOCK** | Both independent branches must pass. |
| D28 clean live | **NOT RUN** | Requires independent D28 review plus a fully-bound predict-only PASS. |
| B2/B3/B4 | **BLOCKED / NOT RUN** | Integrated B1 has not passed. |
| Phase 1–9 | **BLOCKED / NOT RUN** | Gate B has not passed. |

## 2. Test Script Information

### Environment

| Role | Interpreter / environment | Observed version |
|------|---------------------------|------------------|
| D27 worker | `/opt/conda/envs/megatron_env/bin/python3.9` | Python `3.9.18`, torch `2.1.2`, torch CUDA `12.1` |
| Echo Task2 | `/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python3.10` | Python `3.10.20` |
| Echo image | `hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef` | Historical qualification image; not the final release image |

### Scripts and artifact roots

| Purpose | Full path |
|---------|-----------|
| D27 live probe | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/b1_d27_worker1_one_h800_20260717T054559Z/probe.sh` |
| D27 post-validation | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/b1_d27_worker1_one_h800_20260717T054559Z/post_validate.py` |
| Echo Attempt0 helper/test | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_20260717T054334Z/{echo_two_gpu_qualification.py,test_echo_two_gpu_qualification.py}` |
| Echo Retry1 helper/test | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_retry_20260717T055005Z/{echo_two_gpu_qualification.py,test_echo_two_gpu_qualification.py}` |
| Echo Retry2 helper/test | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_retry2_20260717T055948Z/{echo_two_gpu_qualification.py,test_echo_two_gpu_qualification.py}` |
| Echo recovery helper/test | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_recovery_20260717T062633Z/{echo_two_gpu_qualification.py,test_echo_two_gpu_qualification.py}` |
| Echo CPU integration | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_recovery_20260717T062633Z/cpu_integration.py` |

### Reproducible commands

The exact historical `rlaunch` commands are preserved verbatim in each root's `live_launch_command.txt` or `live_command.txt`; the exact predict-only commands are preserved in `predict_only_command.txt` or `predict_command.txt`. They must be inspected rather than reconstructed:

```bash
cat task_memory/task_2026-07-15_sc26_ae_workflow/logs/b1_d27_worker1_one_h800_20260717T054559Z/{predict_command.txt,live_command.txt}
cat task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_retry2_20260717T055948Z/{predict_only_command.txt,live_launch_command.txt}
```

Historical D27 worker payload and post-validation:

```bash
bash /data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/b1_d27_worker1_one_h800_20260717T054559Z/probe.sh
/opt/conda/envs/megatron_env/bin/python3.9 \
  /data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/b1_d27_worker1_one_h800_20260717T054559Z/post_validate.py
```

Historical recovery unit and CPU commands:

```bash
ECHO_PY=/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python3.10
ROOT=/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_b1_echo_two_gpu_recovery_20260717T062633Z
"${ECHO_PY}" -m unittest -v "${ROOT}/test_echo_two_gpu_qualification.py"
"${ECHO_PY}" "${ROOT}/cpu_integration.py"
"${ECHO_PY}" "${ROOT}/static_version_contract_assertion.py"
"${ECHO_PY}" "${ROOT}/preflight_version_contract.py"
```

These commands are evidence commands, not instructions to overwrite immutable roots. A future D28 run must use a new clean root and a newly reviewed fully-bound command.

## 3. Validation Criteria

### D27 branch

- Predict-only, live launch, probe, and final post-validation exits are all `0`.
- Exactly one visible H800 is reported by CUDA and NVML.
- The probe allocates CUDA memory and writes a non-empty JSON.
- Sample count is positive; allocated, reserved, and peak values are positive and finite.
- Inventory contains no missing file, hash mismatch, or byte mismatch.

### Echo branch

- Fixed Python `3.10.20` interpreter and exact torch-family distribution/runtime/CUDA contract pass.
- Exactly two H800 devices with two distinct UUIDs are visible.
- Pinned Echo source trains and writes non-empty model and scaler files.
- Freshly loaded model and two independently loaded pinned `SlowdownPredictor` instances reproduce predictions within tolerance `1e-12`.
- Positive and negative/clipped nonzero-overlap examples report original/reloaded values, formula values, absolute deltas, and relative deltas.
- All exits are `0`; `qualification_result.json` and a closed inventory/hash manifest exist.

### Safety and provenance

- Prior roots remain immutable.
- Filtered source files are hash-bound to the pinned Echo lineage; full-tree equality is not claimed without `.git` metadata.
- No invalid predict-only or interrupted RJob is treated as qualification evidence.
- A consumed live budget is not silently reused.

## 4. Test Results and Evidence

### 4.1 D27 one-H800 MemoryTracker

Artifact root:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/b1_d27_worker1_one_h800_20260717T054559Z
```

| Metric | Expected | Actual | Delta / result |
|--------|----------|--------|----------------|
| Predict-only exit | `0` | `0` | PASS |
| Live rlaunch exit | `0` | `0` | PASS |
| Probe exit | `0` | `0` | PASS |
| Final post-validation exit | `0` | `0` | PASS |
| CUDA device count | `1` | `1` | `0` |
| NVML device count | `1` | `1` | `0` |
| GPU model | H800 | `NVIDIA H800` | PASS |
| Sample count | `>0` | `30` | PASS |
| Allocated memory | `>0 MiB` | `68.0 MiB` | PASS |
| Reserved memory | `>0 MiB` | `1169.9375 MiB` | PASS |
| Peak allocated | `>0 MiB` | `68.0 MiB` | PASS |
| Theoretical tensor | `64.0 MiB` | `64.0 MiB` | `0.0 MiB` |
| Memory JSON bytes | `>0` | `4,951` | PASS |
| Inventory rows | closed inventory | `26` | PASS |
| Missing/hash/byte mismatches | `0/0/0` | `0/0/0` | PASS |

GPU UUID:

```text
GPU-b7b8ef15-9f45-e435-16ca-94f637ea873f
```

Key hashes:

| Artifact | SHA256 |
|----------|--------|
| Memory JSON | `f7b372f38bdfd1a7bc13f5cdc476677ad4a9bbd99be552ee874b096f43da7b0c` |
| `qualification_result.json` | `d2fd0fff673c198abf8532b7796e07197b12da282bb39af736b4efe00b6125f3` |
| `artifact_inventory.tsv` | `715f7eedddd5b262b2f2b3489f35bf0db3d7f16994c66732bc7ada308e07a80b` |

**D27 result: PASS.**

### 4.2 Echo Attempt0, Retry1, and Retry2

| Attempt | Passed before failure | First failure / root cause | Inventory rows | Inventory SHA256 | Integrity mismatches |
|---------|-----------------------|----------------------------|----------------|------------------|----------------------|
| Attempt0 | Source/image/interpreter setup and worker start | Compared torch distribution `2.1.2` with runtime `2.1.2+cu121` as one value | `68` | `7565d38501ca73f28629dcfca6b8f0f8788294517dba388216f8ebd3edc363d4` | `0/0/0` |
| Retry1 | Torch distribution/runtime split | Repeated the same semantic class for torchvision `0.16.2` vs `0.16.2+cu121` | `76` | `f74d0cf47d0158534d14a880e5922e914b3497a59a18fd8842439978eb410ff1` | `0/0/0` |
| Retry2 | GREEN `9/9`; static version fields `3/3/1`; cp310 preflight; `pip check`; predict-only; exactly two H800; real XGBoost training | `if not left` evaluated a `float32` ndarray of shape `(124,)`, raising ambiguous truth-value `ValueError` | `98` | `1c01db512dfe10c688c97b3e0f5fbd7679e8685763b0ae6692bdef9996b4d120` | `0/0/0` |

Retry2 wrote:

| Artifact | Actual bytes |
|----------|-------------:|
| `xgb_model.json` | `621,165` |
| `standard_scaler.json` | `616` |

Retry2 did not complete reload/prediction-api parity. Its earlier RED was an import/setup error and is not accepted as a genuine behavioral RED.

### 4.3 Recovery RED / GREEN

| Run | Exit | Tests | Failures | Errors | Result |
|-----|-----:|------:|---------:|-------:|--------|
| Genuine pre-fix RED | `1` | `13` | `1` | `2` | Expected behavioral failure observed |
| Post-fix GREEN | `0` | `13` | `0` | `0` | `13/13 PASS` |

The functional root-cause fix is:

```diff
-    if not left:
+    if len(left) == 0:
```

The complete helper diff is `+56/-1`; the additional `+55` lines are mandatory evidence instrumentation, so the whole helper change must not be described as a one-line-only diff.

Key hashes:

| Artifact | SHA256 |
|----------|--------|
| Parent helper | `5928333c09e414d0f4bf6e436eaffca774a53b9c0c961c26a839396820933af5` |
| Recovery helper | `a62f632b8cd0bac733ea65d2f50696a16d7128fc6b3c36b7623d58e5e16743d8` |
| Recovery test | `f78d91a90392868069b7343b0d79d0f9d460ef5510762c4709085dcd77442935` |
| Qualification diff | `0767ebee6e34cbc4ef40ba07ec4bf019369e2aafeb9d5fa7099f07f789e61ec6` |

### 4.4 Fixed-cp310 serial CPU integration

| Metric | Expected | Actual | Delta / result |
|--------|----------|--------|----------------|
| CPU integration exit | `0` | `0` | PASS |
| Static version contract exit | `0` | `0` | PASS |
| Preflight version contract exit | `0` | `0` | PASS |
| Dataset rows | positive | `619` | PASS |
| Train/test rows | non-empty | `495/124` | PASS |
| Columns/features | expected schema | `17/8` | PASS |
| XGBoost prediction dtype/shape | ndarray | `float32/[124]` | PASS |
| Test MSE | finite | `15.265446877683257` | PASS |
| Model/scaler bytes | positive | `621,165/616` | PASS |
| Model reload max abs delta | `<=1e-12` | `0.0` | `0.0` |
| Prediction API reload max abs delta | `<=1e-12` | `0.0` | `0.0` |

Package versions:

```text
NumPy=1.26.4
pandas=2.2.0
scikit-learn=1.3.0
XGBoost=2.1.0
torch distribution/runtime=2.1.2/2.1.2+cu121
torchvision distribution/runtime=0.16.2/0.16.2+cu121
torchaudio distribution/runtime=2.1.2/2.1.2+cu121
CUDA=12.1
```

Positive nonzero-overlap sample:

| Field | Actual |
|-------|-------:|
| Row | `10` |
| Overlap | `0.0457111761104686` |
| Predicted factor | `0.4127890169620514` |
| Ground truth | `215171.0` |
| Predicted execution | `219231.07697314429` |
| Formula delta | `0.0` |
| Max relative delta | `0.0` |

Negative/clipped nonzero-overlap sample:

| Field | Actual |
|-------|-------:|
| Row | `26` |
| Overlap | `0.0522674391728431` |
| Predicted factor | `-0.1773671954870224` |
| Clipped factor | `0.0` |
| Ground truth | `65505.0` |
| Predicted execution | `64897.73399121439` |
| Predicted clipped execution | `65505.00000000001` |
| Formula/clipped formula delta | `0.0/0.0` |
| Max relative delta | `0.0` |

| Artifact | SHA256 |
|----------|--------|
| Model | `0e5c53c0da638f376366beb2d7a66af7889debfd7bf25d4e9cff016cef8ded07` |
| Scaler | `eb330ca67ecb39156b3b8e6aaf8b81940f7d0501154f0da7f51a9a39c550d165` |
| CPU integration script | `f38b23f811cba1ce4e423d503031664b57caab3dabf779ac296794dea4596523` |

### 4.5 Source binding

The recovery snapshot has no `.git`. Early `git rev-parse` results resolved upward to the Megatron parent and are non-authoritative. The corrected binding is:

| Identity | Value |
|----------|-------|
| Pinned Echo commit | `1390b4416ded08bc1b9cd0620d329d81d4470bf9` |
| Pinned Echo tree | `581fc0bf6d36ff6e7b3b0c54ba5d6bb008c5368f` |
| Full Echo archive SHA256 | `2a9e7c48fa450831714fdd55a7082125eebe871c4ad492fada9e3e82ac16311c` |
| Executed `prediction_api.py` SHA256 | `f391a83a35c8554b98791b5f863c98ddc92b2af4a23c322c0c8cddf12a30ced6` |
| Executed training CSV SHA256 | `5309e3b0e9265ca50142db96c559df9c7c06f49dc721a4d78c4e85ff7aa83a14` |
| Full-tree equality claim | `false` |

### 4.6 Execution incidents

1. Two CPU integration processes used the same root: PIDs `2329847` and `2331687`.
2. Both were terminated through `pkill -f`; Attempt-1 exit=`143`.
3. Worker-2 used unauthorized `rm -f` on the exit, metrics, model, and scaler evidence paths. Attempt-1 bytes are permanently lost.
4. The later serial CPU PASS is separate evidence; it is not restoration of Attempt-1.
5. Incident record SHA256: `975401da88d60e1a66bccbc6c2afb4f85f9ad57a1adaa88a122c02bba70a9cad`.
6. An early two-GPU predict-only ran `bash -lc 'true'`. Although process/semantic exits were `0/0` and it listed `10` candidates, it omitted the actual live contract and is invalid/non-authorizing.
7. During the hard hold, RJob `sc26-ae-b1-echo-recovery-20260717t062633z` was created at 2026-07-17 14:37:44 +08:00, scheduled on `gpu-h800-0263.host.platform.shaipower.com`, and began image pull before interruption.
8. Local launch exit=`130`; final RJob status=`Stopped`. No Echo payload, `nvidia-smi`, device count, UUID, model/scaler, parity, or `qualification_result.json` was produced.
9. The prior live budget is consumed. The interrupted RJob is not qualification evidence.

### 4.7 Independent audit

| Lane | Reviewer | D27 | Echo | Integrated B1 | Additional finding |
|------|----------|-----|------|---------------|--------------------|
| A | OMX verifier worker-1 | PASS | BLOCK | BLOCK | Split verdict confirmed. |
| B | OMX verifier worker-2 | PASS | BLOCK | BLOCK | Prior inventories closed; product/staged/gitlink diffs `0/0/0`. |
| C | native verifier `/root/verifier_lane_c` | PASS | BLOCK | BLOCK | Prior budget consumed; D28 budget conditional and unconsumed. |

The later Team `orphan-cleanup` removed canonical task/mailbox state. Lane C remains attributable to the native verifier, not dead worker-3; Team lifecycle completion is not claimed.

## 5. Current Blocker and D28 Gate

The current Echo blocker is **missing final live qualification evidence**, not a missing library package. The cp310 package closure, `pip check`, torch-family schema, helper RED/GREEN, and actual CPU integration have passed.

D28 permits one new clean live attempt only after:

1. D28 is synchronized across all task documents.
2. Independent StepCode Claude review returns `APPROVE`; `WATCH` is remediated and re-reviewed, while `BLOCK` stops for user adjudication.
3. A new root is created without rewriting historical pointers.
4. Predict-only is identical to the intended live command except for `--predict-only`, including image, `/data:/data`, workdir, clean root, fixed cp310 interpreter, isolated source, helper, payload, and resources.
5. Predict-only process/semantic exits are `0/0`, quota markers are absent, and an H800 candidate has at least two GPUs.
6. Exactly one final live attempt runs. Any new root-cause class, contract drift, incomplete evidence, or failure stops with no retry or fallback.

Gate 2, the independent D28 Task A7 review, passed after a `WATCH` plus two plan-doc precision remediations and a follow-up `APPROVE`. Final plan-document validation also passed. Gates 3–6 remain later execution-stage actions and were not started during this docs-only stage.

During this plan-review stage, the D28 root, fully-bound predict-only, and live RJob are **NOT RUN**.

## 6. Final Report Status

| Validation target | Result |
|-------------------|--------|
| D27 one-H800 qualification | **PASS** |
| Echo exact-two-H800 qualification | **BLOCK** |
| Integrated B1 | **BLOCK** |
| Dependency closure for the known Echo helper path | **PASS; not the current blocker** |
| D28 independent plan review | **PASS after WATCH remediation and follow-up APPROVE** |
| D28 fully-bound predict-only | **NOT RUN** |
| D28 final live qualification | **NOT RUN** |
| B2/B3/B4 | **BLOCKED** |
| Phase 1–9 | **BLOCKED** |

No product implementation, package installation, Git/submodule mutation, commit, push, Release publication, or new GPU/RJob execution is part of this report creation.

## 7. Final Docs / Artifact / Git-Scope Validation

**Working directory:** `/data/ycfeng/Megatron-LM-sc26-ae`  
**Environment:** system Python `3.12.3`; `CONDA_DEFAULT_ENV=none`; branch `sc26-ae`; HEAD `c25544486283ed9e0fe75d2966e38a529b8185af`.

The final validator read the seven core task documents, the enhanced-plan test report, the current Gate B1 report, both D28 advisor artifacts, the D27 closed inventory, all three prior Echo root manifests, recovery metrics/source binding, and Git scope.

| Metric | Expected | Actual | Result |
|--------|----------|--------|--------|
| Core/checked docs | `7/9` | `7/9` | PASS |
| Requirements/decisions | `15/28` | `15/28` | PASS |
| `[Original Request]` tags | `44` | `44` | PASS |
| Public entries | `9` | `9` | PASS |
| Issue headings / missing matrix rows | `38/0` | `38/0` | PASS |
| Balanced Markdown fences | `9/9` | `9/9` | PASS |
| Advisor artifacts/verdicts | `2`, WATCH then APPROVE | `2`, WATCH then APPROVE | PASS |
| D27 inventory rows/mismatches | `26/0` | `26/0` | PASS |
| Prior Echo manifest rows | `68/76/98` | `68/76/98` | PASS |
| Prior-root hash mismatches | `0` | `0` | PASS |
| CPU rows/train/test | `619/495/124` | `619/495/124` | PASS |
| CPU test MSE | finite | `15.265446877683257` | PASS |
| Model/prediction reload deltas | `<=1e-12` | `0.0/0.0` | PASS |
| Product-scope paths | `0` | `0` | PASS |
| Staged paths | `0` | `0` | PASS |
| Gitlink diffs | `0` | `0` | PASS |
| `git diff --check` exit | `0` | `0` | PASS |

Validator output:

```text
PASS_FINAL_D28 core_docs=7 checked_docs=9 R=15 D=28 original_request_tags=44 public_entries=9 issue_headings=38 matrix_missing=0 balanced_fences=9 advisor_artifacts=2 advisor_verdicts=WATCH/APPROVE d27_inventory_rows=26 d27_inventory_mismatches=0 prior_manifest_rows=68/76/98 prior_hash_mismatches=0 cpu_rows_train_test=619/495/124 cpu_test_mse=15.265446877683257 model_reload_delta=0.0 prediction_api_reload_delta=0.0 changed_paths=7 untracked_paths=1 product_scope_paths=0 staged_paths=0 gitlink_diff=0
```

**Docs-only stage result: PASS.** This result closes Task A7 documentation review only. Echo exact-two-H800 and integrated B1 remain `BLOCK`; D28 predict-only/live remain `NOT RUN`; B2/B3/B4 and Phase 1–9 remain blocked.

## 8. Fresh Post-Closure Rerun

The closure-status wording in `plan.md`, `progress.md`, `review.md`, and this report was written after the first final validator. A fresh run therefore revalidated the stopped state rather than relying on the earlier result.

### Test Script Information

- Temporary read-only validator: `/tmp/sc26_ae_post_closure_validator.py`
- Working directory: `/data/ycfeng/Megatron-LM-sc26-ae`
- Environment: system Python `3.12.3`; `CONDA_DEFAULT_ENV=none`; branch `sc26-ae`; HEAD `c25544486283ed9e0fe75d2966e38a529b8185af`
- Exact commands:

  ```bash
  set -euo pipefail
  python3 /tmp/sc26_ae_post_closure_validator.py
  git diff --check
  git status --short --branch
  git diff --name-only
  git ls-files --others --exclude-standard
  git diff --cached --name-only
  git diff --submodule=short -- Echo-slowdown megatron-sim-engine
  ```

### Validation Criteria

1. All final R1–R15, D1–D28, logical I1–I38, public-entry, Markdown-fence, advisor-artifact, D27 inventory, prior Echo manifest, recovery metric, source-binding, and incident contracts pass.
2. D27 remains `PASS`; Echo exact-two-H800 and integrated B1 remain `BLOCK`.
3. D28 predict-only/live remain `NOT RUN`; B2/B3/B4 and Phase 1–9 remain blocked.
4. Product-scope paths, staged paths, and gitlink diffs remain `0`; the historical latest-path pointer remains unchanged.

### Validator-Test Failure Diagnosis and Resolution

The first reconstructed validator iterations failed fast because the validator itself encoded unsupported or schema-inaccurate assertions. No task document was changed to mask these failures.

| Validator defect | Observed result | Root cause | Resolution |
|------------------|-----------------|------------|------------|
| Required every D1–D28 token in `review.md` | Exit `1` at D2 | Review records checkpoints; it is not a duplicate requirements registry | Kept D1–D28 completeness in `requirements.md` and traceability in `plan.md` |
| Required the fixed cp310 absolute path only in `plan.md` | Exit `1` | The exact path is recorded in the dependency/review/progress contract while `plan.md` fixes the interpreter role | Validated the path across the authoritative task documents |
| Checked non-canonical lowercase incident/dependency phrases | Exit `1` | Case-sensitive test wording differed from the approved text | Matched the exact recorded evidence statements |
| Required `qualification_result.json.status` | Exit `1` | D27 schema records measured CUDA/NVML/memory fields without a redundant status key | Validated GPU model, CUDA/NVML counts, `30` samples, and positive finite memory metrics directly |
| Outer wrapper omitted `set -e` once | Python failed while the wrapper ended `0` | Later Git commands masked the Python exit | Reran with `set -euo pipefail`; only the corrected fail-fast run is accepted |

### Test Results and Numeric Evidence

| Metric | Expected | Actual | Result |
|--------|----------|--------|--------|
| Validator exit | `0` | `0` | PASS |
| Core/checked docs | `7/9` | `7/9` | PASS |
| Requirements/decisions | `15/28` | `15/28` | PASS |
| `[Original Request]` tags | `44` | `44` | PASS |
| Logical issue coverage / missing matrix IDs | `38/0` | `38/0` | PASS |
| Advisor verdict sequence | WATCH then APPROVE | WATCH then APPROVE | PASS |
| D27 inventory rows/mismatches | `26/0` | `26/0` | PASS |
| Prior Echo manifest rows/hash mismatches | `68/76/98`, `0` | `68/76/98`, `0` | PASS |
| CPU dataset/train/test rows | `619/495/124` | `619/495/124` | PASS |
| CPU test MSE | finite | `15.265446877683257` | PASS |
| Model/prediction-api reload deltas | `<=1e-12` | `0.0/0.0` | PASS |
| Tracked/untracked task paths | `7/1` | `7/1` | PASS |
| Product/staged/gitlink paths | `0/0/0` | `0/0/0` | PASS |
| `git diff --check` exit | `0` | `0` | PASS |

Accepted validator output:

```text
PASS_FINAL_D28 core_docs=7 checked_docs=9 R=15 D=28 original_request_tags=44 public_entries=9 issue_headings=38 matrix_missing=0 balanced_fences=9 advisor_artifacts=2 advisor_verdicts=WATCH/APPROVE d27_inventory_rows=26 d27_inventory_mismatches=0 prior_manifest_rows=68/76/98 prior_hash_mismatches=0 cpu_rows_train_test=619/495/124 cpu_test_mse=15.265446877683257 model_reload_delta=0.0 prediction_api_reload_delta=0.0 changed_paths=7 untracked_paths=1 product_scope_paths=0 staged_paths=0 gitlink_diff=0
```

This rerun closes only the plan-review validation gap. It does not create or authorize D28 execution evidence.
