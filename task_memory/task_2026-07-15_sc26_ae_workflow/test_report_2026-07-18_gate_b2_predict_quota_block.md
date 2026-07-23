# Test Report — Gate B2 Fresh Sealed Predict-Only Quota Block

## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-18 | Added Session 70 documentation-validator TDD, stale Gate B checkpoint correction, and 199-check candidate PASS evidence. |
| 2026-07-18 | Added Session 70 RenderedBundle-derived fresh identity, full-harness review, zero-drift TDD, sole predict semantic quota failure, and terminal containment evidence. |
| 2026-07-18 | Recorded fresh harness validation, semantic-parser RED/GREEN, sealed predict-only process/semantic evidence, quota root cause, containment, and downstream stop state. |

## 1. Verdict

**BLOCKED.** The wholly fresh Gate B2 harness passed generation tests, fixture validation, static validation, self-excluding seal, independent full-harness technical review, external zero-drift, and predict semantic-parser tests. The exact sealed `rlaunch --predict-only` command then completed with process exit `0`, but its content failed admission because current `codesign` GPU quota was `129/128`.

The semantic validator correctly returned exit `1` and status `FAIL`. No live RJob was submitted, no GPU worker started, and the Qwen3 full-profile workload did not run. Therefore Gate B2 is not a workload failure and not a PASS; it is blocked before live by external dynamic resource state.

## 2. Execution Identity

| Field | Value |
|---|---|
| UTC `RUN_ID` | `20260718T055534Z` |
| Job name | `sc26-gb-b2-fr-20260718t055534z` |
| Clean execution worktree | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717` |
| Branch / HEAD | `sc26-ae-exec-clean-20260717` / `3c91d15bc035d49216161c9cac874f2453b69cb9` |
| Harness root | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T055534Z` |
| Runtime root | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z` |
| Context | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/context/sc26-ae-gate-b2-fresh-20260718T055534Z.md` |
| Intended capture root | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/output/_work/recon-task1-qwen-20260718T055534Z` |
| Canonical image | `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4` |
| Intended worker interpreter | `/opt/conda/envs/megatron_env/bin/python` |
| Requested resources | GPU=`1`, CPU=`16`, memory=`65536 MiB`, H800, `codesign`, private group machine |

## 3. Test Script Information

### 3.1 Scripts

| Script | Purpose | Bytes | SHA256 |
|---|---|---:|---|
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z/test_generate_b2_harness.py` | Fresh generator TDD and exact bundle/command contract | `11,473` | Recorded in the sealed preparation runtime evidence |
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z/test_post_validate_fixtures.py` | Valid and negative post-validation fixtures | `11,579` | Recorded in the sealed preparation runtime evidence |
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z/validate_generated_b2_harness.py` | Generated-bytes static/semantic validation | `11,218` | Recorded in the sealed preparation runtime evidence |
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z/validate_sealed_zero_drift.py` | Exact seal, render, manifest, review, mode, and path validation | `7,995` | Recorded in the runtime evidence |
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z/test_predict_semantic_validator.py` | Predict parser unit and CLI overwrite tests | `7,496` | `69243ca60e2c2f724ec7b149f1f97eaf8ab2567df8c14c3fc6ad9e5dd5b143c3` |
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z/validate_predict_semantics.py` | Process/content/resource semantic gate | `7,110` | `9da84c1a3498d8de74518bc71fbb4fa698271bf96f54d2fceb877156f837db68` |
| `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T055534Z/predict_command.sh` | Exact sealed predict-only command | `642` | Bound by `seal.json` |

### 3.2 Environment

| Item | Actual value |
|---|---|
| Controller interpreter | `/usr/bin/python3` |
| Controller Python | `3.12.3` |
| Controller conda env | `none` |
| Bytecode controls | `PYTHONDONTWRITEBYTECODE=1`, Python `-B` |
| Worker allocation | `NOT_CREATED` because predict semantics failed |
| Worker GPU/device | `NOT_RUN` |

### 3.3 Reproducible commands

#### Sealed zero-drift

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B validate_sealed_zero_drift.py \
  2>&1 | tee zero_drift_attempt1.log
statuses=("${PIPESTATUS[@]}")
printf 'ZERO_DRIFT_EXIT=%s\nTEE_EXIT=%s\n' "${statuses[0]}" "${statuses[1]}"
```

#### Predict semantic validator RED/GREEN

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest -v \
  test_predict_semantic_validator.py
```

#### Exact sealed predict-only

```bash
/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T055534Z/predict_command.sh
```

The sealed script invokes:

```bash
/kubebrain/rlaunch --predict-only \
  --name=sc26-gb-b2-fr-20260718t055534z \
  --charged-group=codesign \
  --private-machine=group \
  --positive-tags=h800 \
  --gpu=1 \
  --cpu=16 \
  --memory=65536 \
  --predict-node-num=10 \
  --backoff-limit=1 \
  --image-pull-policy=Always \
  --image=hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4 \
  --volume=/data:/data \
  --workdir=/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717 \
  -- env RUN_ID=20260718T055534Z bash \
  /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T055534Z/worker_entry.sh
```

#### Semantic evaluation of the actual predict log

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B validate_predict_semantics.py \
  --log-path predict_process.log \
  --process-exit-code 0 \
  --elapsed-seconds 2 \
  --requested-gpu-count 1 \
  --requested-cpu-count 16 \
  --requested-memory-mib 65536 \
  --predict-node-num 10 \
  --output-path predict_semantic.json
```

## 4. Validation Criteria

| Check | Acceptance criterion |
|---|---|
| Fresh identity | All harness/runtime/context/capture/RJob identities are new and absent before creation |
| Candidate integrity | Flat regular-file root; subdirectories/symlinks/bytecode=`0/0/0` before seal |
| Generator tests | All exact-identity, D43-schema, command, syntax, and negative-path tests pass |
| Post-validator fixtures | One valid fixture passes; every negative case fails for the expected root cause |
| Seal | Self-excluding payload identity matches; root becomes read-only; no post-seal write |
| Independent review | Fresh full-harness technical verdict `APPROVE`; `WATCH=0`, `BLOCK=0` |
| Zero-drift | Seal/rendered bytes/manifests/context/review/modes match; capture and worker roots absent |
| Predict parser | Unit tests cover positive, process failure, elapsed boundary, quota/image/auth/error markers, headers, candidates, resource eligibility, malformed rows, limit, and overwrite |
| Predict process | `rlaunch` exit=`0`, `tee` exit=`0`, elapsed>0 |
| Predict semantics | No failure marker; exactly one resource section; at least one H800 candidate satisfying GPU/CPU/memory=`1/16/65536 MiB` |
| Live eligibility | Only process PASS **and** semantic PASS open the single live command |
| Containment | Any predict semantic failure leaves capture/worker/live absent and stops without retry/fallback |

## 5. Results and Evidence

### 5.1 Test summary

| Suite / gate | Expected | Actual | Result |
|---|---:|---:|---|
| Generator TDD | all pass | `8/8` | PASS |
| Post-validator fixtures | `17/17` expected outcomes | `17/17`; valid=`1`, negative rejected=`16` | PASS |
| Generated-bytes static validation | exit `0` | exit=`0`, status=`PASS` | PASS |
| Independent full-harness review | `APPROVE`, no WATCH/BLOCK | `APPROVE`, `0/0` | PASS |
| Initial zero-drift | exits `0/0` | validator/`tee`=`0/0` | PASS |
| Predict parser RED | fail because implementation absent | `13` tests, `13` expected failures | PASS (RED observed) |
| Predict parser GREEN | all pass | `13/13`, `0.128s` | PASS |
| Known real-log integration | valid/invalid=`PASS/FAIL` | `PASS/FAIL` | PASS |
| Pre-predict zero-drift | exit `0` | exit=`0` | PASS |
| Predict process | exits=`0/0`, elapsed>0 | exits=`0/0`, elapsed=`2s` | PASS |
| Predict semantics | status=`PASS` | exit=`1`, status=`FAIL` | **FAIL** |
| Live submission | only after semantic PASS | `NOT_RUN` | BLOCKED |
| Post-predict containment | no capture/worker/live; seal unchanged | all absent; seal unchanged | PASS |

### 5.2 Key numeric metrics

| Metric | Expected | Actual | Delta / interpretation |
|---|---:|---:|---|
| Sealed payload files | `18` | `18` | `0` |
| Sealed payload bytes | `33,068` | `33,068` | `0` |
| Seal bytes | `2,948` | `2,948` | `0` |
| Seal SHA256 | frozen value | `7db55d97cb07c6ed2bddb951510f906ea8dbc8bde226c8f622232ded540da395` | exact match |
| Context bytes | `7,761` | `7,761` | `0` |
| Context SHA256 | frozen value | `73439f6ec477310c1960894866639396484d533ccff8ad9f41a06360a86682c5` | exact match |
| Runtime bytecode count | `0` | `0` | `0` |
| Predict process exit | `0` | `0` | `0` |
| Predict `tee` exit | `0` | `0` | `0` |
| Predict elapsed | `>0s` | `2s` | positive |
| Quota requested-plus-used | `<=128` | `129` | `+1` over quota |
| Quota total | `128` | `128` | fully used |
| Resource headers | `1` | `0` | `-1`; no admission candidates returned |
| Candidate nodes | `>=1` | `0` | no candidate |
| H800 candidates | `>=1` | `0` | no candidate |
| Eligible H800 candidates | `>=1` | `0` | live gate closed |
| Available GPU min/max | `>=1` for an eligible candidate | `0/0` | no resource rows |
| Live submissions | `1` only after predict PASS | `0` | correctly withheld |
| Capture files/bytes | nonempty only after live | `0/0` | workload `NOT_RUN` |

### 5.3 Exact failure evidence

```text
time="2026-07-18T14:26:09+08:00" level=info msg="Checking image..." image="hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4"
fail to pass quota check:
gpu : 129/128; current value + has used value: 129; total value: 128,
```

Semantic result:

```json
{
  "candidate_count": 0,
  "elapsed_seconds": 2,
  "eligible_candidate_count": 0,
  "failure_markers": ["quota_failure"],
  "h800_candidate_count": 0,
  "maximum_available_marker_count": 0,
  "process_exit_code": 0,
  "semantic_status": "FAIL"
}
```

## 6. Root-Cause Analysis

The direct cause is external dynamic GPU quota state:

```text
current used GPU quota = 128
requested additional GPU = 1
requested plus used = 129
quota total = 128
```

`rlaunch --predict-only` itself completed normally and therefore returned process exit `0`; its output correctly reported that the requested resource cannot be admitted. The parser prevented this process-level success from being misreported as scheduler eligibility.

This evidence does **not** establish any defect in:

- the immutable image or registry access;
- the selected conda environment or package set;
- the D43 dependency result (`required_package_gap_count=0`, `broken_conda_env_count=0`);
- the fresh harness, seal, or review chain;
- Megatron-LM, Qwen3 MoE, MemoryTracker, trace capture, memory capture, or the full-profile workload.

Those worker/workload paths were never entered.

## 7. Containment and Stop State

- Sealed harness remained read-only and byte-identical.
- Post-predict containment/zero-drift exit=`0`.
- Runtime bytecode and `__pycache__` counts remained `0/0`.
- Intended capture root does not exist.
- Worker runtime root does not exist.
- Live process log/result does not exist.
- No live RJob, GPU worker, Qwen execution, trace, memory JSON, or 256-rank estimate exists.
- No retry, resource reduction, quotagroup/tag/private-machine/image/interpreter/source switch, fallback, D38 adoption, product/test edit, `rm`, or `mv` occurred.

## 8. Evidence Identities

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `zero_drift_attempt1.log` | `1,186` | `6ea2da847d638f898e5946d7c8c237665be8042f0affe895acb9395241b74428` |
| `predict_semantic_tdd_red.log` | `18,500` | `954debd0f01c2a1d21adf01264dd9d6cbcc09848d1e659f7e1d04196e4080357` |
| `predict_semantic_tdd_green_attempt1.log` | `2,078` | `7b3bbe154979b86a8b7f62efa347f881e9959aa1be7c96e43b2e87548f5d9252` |
| `predict_semantic_integration_validation.log` | `3,233` | `b14ab32c2927e5634cd3b0867a79328e90e6764ffeedc1ab9e2fd5666b7f80a4` |
| `zero_drift_prepredict.log` | `1,186` | `6ea2da847d638f898e5946d7c8c237665be8042f0affe895acb9395241b74428` |
| `predict_process.log` | `286` | `2ace346a8f4cc971fbb4f1b5acce7aeb3adba2a699120868625d40ad322b1d9b` |
| `predict_process.env` | `157` | `2419da4765c42e7cf5b5efbdd2cbe5f39111bc1b2b0d8d518e2d407845269d1e` |
| `predict_semantic.json` | `644` | `1343cb299a097ed37f8b28cf076c1ae1cb3d6bc79d003f14ccce025c3cfb0bda` |
| `predict_semantic_validation.log` | `644` | `1343cb299a097ed37f8b28cf076c1ae1cb3d6bc79d003f14ccce025c3cfb0bda` |
| `post_predict_containment.log` | `1,411` | `f895834707a70de5c21160c202ab67202dd2f8ecdd82dccc02cc13daaa42d8c5` |

All paths in this table are relative to:

```text
/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T055534Z
```

## 9. Required Adjudication Before Continuation

This failed identity is preserved and non-reusable. The one-live B2 submission was not consumed, but live is not authorized for this identity.

Continuation requires all of the following:

1. Explicit user adjudication to resume after external `codesign` GPU quota becomes available.
2. No resource/tag/quotagroup/private-machine/image/interpreter/source fallback.
3. A wholly new identity, authority snapshot, harness/runtime/context/capture/RJob paths.
4. Complete generator TDD, fixtures, static validation, self-excluding seal, fresh independent full-harness review, and external zero-drift.
5. A new exact sealed predict-only with both process and semantic PASS.
6. Only then, one live B2 submission followed by trace/memory/communication/containment validation.

B3 live, B4, B5, Phase 1–9, D38 adoption, and product/test implementation remain blocked until those gates pass in order.

## 10. Session 70 Fresh-Chain Revalidation

### 10.1 Verdict

**BLOCKED — EXTERNAL GPU QUOTA.** A second, wholly fresh and non-reused Gate B2 identity fixed Session 69's static-validator root cause and passed generation, positive/negative validation, seal, independent full-harness review, and external zero-drift. Its exact sealed predict-only then reported the same external `codesign` quota state: requested-plus-used GPU=`129`, quota total=`128`, delta=`+1` GPU.

The predict CLI process returned `0`, but semantic validation returned exit `1` and `FAIL`. No live command was run. This is not a Qwen, Megatron-LM, image, interpreter, dependency, trace, memory, or workload failure because the worker path was never entered.

### 10.2 Execution Identity

| Field | Value |
|---|---|
| UTC `RUN_ID` | `20260718T075949Z` |
| Job name | `sc26-gb-b2-fr-20260718t075949z` |
| Branch / HEAD | `sc26-ae-exec-clean-20260717` / `3c91d15bc035d49216161c9cac874f2453b69cb9` |
| Harness root | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T075949Z` |
| Runtime root | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z` |
| Context path | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/context/sc26-ae-gate-b2-fresh-20260718T075949Z.md` |
| Intended capture root | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/output/_work/recon-task1-qwen-20260718T075949Z` |
| Canonical image | `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4` |
| Intended worker interpreter | `/opt/conda/envs/megatron_env/bin/python` |
| Requested resources | GPU=`1`, CPU=`16`, memory=`65536 MiB`, candidates=`10`, H800, `codesign`, private group machine |

### 10.3 Test Script Information

**Controller environment**

- Host conda environment: none.
- Python: `/usr/bin/python3`, Python `3.12.3`.
- Bash: GNU bash `5.2.21(1)-release`.
- Timezone: `HKT`.
- GPU workload environment: not entered because predict semantics failed before a worker was launched.

**Scripts**

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/test_validate_sealed_zero_drift.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/validate_sealed_zero_drift.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/validate_predict_semantics.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/test_build_post_predict_terminal_audit.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/build_post_predict_terminal_audit.py`
- Sealed scheduler entry: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T075949Z/predict_command.sh`

**Reproducible commands**

```bash
R=/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z
H=/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/evidence/sc26-ae-gate-b2-fresh-20260718T075949Z

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/test_validate_sealed_zero_drift.py"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/validate_sealed_zero_drift.py"

# This sealed predict command was run exactly once for this identity. Do not rerun it.
bash "$H/predict_command.sh"

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/validate_predict_semantics.py" \
  --log-path "$R/predict_once_raw.log" \
  --process-exit-code 0 \
  --elapsed-seconds 1 \
  --requested-gpu-count 1 \
  --requested-cpu-count 16 \
  --requested-memory-mib 65536 \
  --predict-node-num 10 \
  --output-path "$R/predict_semantic_result.json"

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/test_build_post_predict_terminal_audit.py"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/build_post_predict_terminal_audit.py"
```

The semantic command is expected to return exit `1` for the preserved quota-failed log. Its JSON output, rather than a zero process status from `rlaunch`, is the admission decision.

### 10.4 Validation Criteria

1. Static expectations derive from the fresh generator's current `RenderedBundle`; file additions/removals/byte changes and context changes are detected.
2. Candidate root is flat, regular, bytecode-free, exactly mode-bound, and byte-identical to the rendered bundle.
3. Self-excluding seal, payload totals, external context, source/authority/context manifests, and bindings all match.
4. Independent artifact is exact and semantically contains current `APPROVE`, `BLOCK=None`, exactly one expected WATCH, and explicit zero-drift/predict-only scope.
5. Wrong review hash, wrong seal hash, wrong payload total, and obsolete Session 64 approval wording fail explicitly.
6. Predict process and `tee` exit `0/0`, elapsed is positive, failure markers are absent, exactly one resource header exists, and at least one H800 candidate satisfies GPU/CPU/memory.
7. Live remains unauthorized unless all semantic criteria pass.
8. Post-predict seal/review/context/manifests remain exact; capture/worker/bytecode remain absent after a failed predict.

### 10.5 Test Results

| Test / Gate | Result | Actual Evidence |
|---|---|---|
| Generator suite | PASS | `8/8` |
| RenderedBundle-derived static suite | PASS | `10/10` |
| Generic predict semantic parser suite | PASS | `13/13` |
| Post-validator fixtures | PASS | `17/17`; valid=`1`, expected negative rejection=`16` |
| One-shot generation | PASS | payload=`18/34,861`; context=`10,243/09d52a7c...31c2` |
| Self-excluding seal | PASS | `2,948/54a1d912...c315`; root files including seal=`19` |
| Independent full-harness review | PASS with WATCH | verdict=`APPROVE`; WATCH/BLOCK=`1/0` |
| Zero-drift TDD RED | PASS | missing validator rejected; Python/`tee`=`5/0` |
| Zero-drift GREEN Attempt 1 | FAIL then diagnosed | `5/6`; old-review fixture lacked the advisor raw-output envelope |
| Zero-drift GREEN Attempt 2 | PASS | `6/6`; Python/`tee`=`0/0`; elapsed=`798,734,869ns` |
| Formal initial zero-drift | PASS | Python/`tee`=`0/0`; elapsed=`238,867,591ns` |
| Sealed predict-only | PROCESS PASS / SEMANTIC FAIL | process/`tee`=`0/0`; elapsed=`830,557,225ns`; quota=`129/128` |
| Predict semantic validation | EXPECTED FAIL | validator/`tee`=`1/0`; marker=`quota_failure` |
| Post-predict zero-drift | PASS | Python/`tee`=`0/0`; seal/context/review exact |
| Terminal audit builder TDD | PASS | RED exit=`5`; GREEN=`1/1` |
| Terminal audit creation | PASS | process/`tee`=`0/0`; elapsed=`97,968,854ns`; audit=`6,143/d8b13f72...e394` |
| Live execution | NOT RUN | invocation count=`0`; worker/capture absent |

### 10.6 Key Numeric Metrics

| Metric | Expected / Threshold | Actual | Delta / Error | Result |
|---|---:|---:|---:|---|
| Payload file count | `18` | `18` | `0` | PASS |
| Payload total bytes | `34,861` | `34,861` | `0` | PASS |
| Root files including seal | `19` | `19` | `0` | PASS |
| Seal bytes | `2,948` | `2,948` | `0` | PASS |
| Context bytes | `10,243` | `10,243` | `0` | PASS |
| Authority/source/context manifests | `22/6/1` | `22/6/1` | `0/0/0` | PASS |
| Review WATCH/BLOCK | `1/0` | `1/0` | `0/0` | PASS |
| Runtime bytecode | `0` | `0` | `0` | PASS |
| Predict process/`tee` exit | `0/0` | `0/0` | `0/0` | PASS |
| Predict elapsed | `>0ns` | `830,557,225ns` | positive | PASS |
| Quota requested-plus-used | `<=128` | `129` | `+1 GPU` | FAIL |
| Quota total | `128` | `128` | `0` | observed |
| Resource header count | `1` | `0` | `-1` | FAIL |
| Candidate/H800/eligible | `>=1/>=1/>=1` | `0/0/0` | `-1/-1/-1` minimum | FAIL |
| Predict/semantic/live invocations | `1/1/0` after semantic FAIL | `1/1/0` | `0/0/0` | PASS |
| Capture/worker existence | `false/false` | `false/false` | none | PASS |

### 10.7 Root Cause, Controller Incidents, and Containment

The blocking cause is external dynamic quota state: `128` GPUs were already counted against total quota `128`; adding the exact one-GPU request produces `129`. The only supported pre-allocation query is `rlaunch --predict-only`, and its process-level exit `0` does not override explicit quota-failure text.

One environment/controller issue was resolved before zero-drift: OMC wrote `.omc/state/.../last-tool-error-state.json`, temporarily adding `?? .omc/` to the status-bound clean worktree. Adding `.omc/` next to the existing `.omx/` in repository-local Git `info/exclude` restored the exact status without deleting/moving runtime state or weakening validation. Generator authority validation changed from exit `1` to `0`; status entries changed from `2` to `1`; authority/source remained `22/6`; seal remained unchanged.

Terminal audit assembly then had two preserved controller-only failures:

1. wrong parent index selected `/data/ycfeng` for `git status`, returning `128`;
2. a long inline here-document retained patch `+` prefixes and failed Python parsing.

Neither incident invoked the scheduler, changed semantic output, wrote capture/worker data, or modified the sealed harness. The test-first builder resolved both and created the final audit once. There was no fallback, request reduction, identity reuse, predict retry, live submission, product/test change, D38 adoption, `rm`, or `mv`.

### 10.8 Evidence Identities

All paths below are relative to `.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/` in the clean execution worktree.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `pre_zero_drift_snapshot_red.log` | `870` | `aa36b9673fbd89f6b8f8ae0a5241ab6afa4dc3a50477013c22a719c4d30eee99` |
| `pre_zero_drift_snapshot_green.log` | `234` | `9f421117e8e8ab044c3b6c65ba04858d16bd51e8e72b29a3956f3987da46371b` |
| `tdd_zero_drift_red.log` | `1,204` | `6f074961550174e9339735a0d9e95f370027cef09f80059f10c4eff7683cd82c` |
| `tdd_zero_drift_green_attempt1.log` | `1,808` | `040537c02d71e6e289ef813c001d7c28cf158081b9781c7a88fa520137266def` |
| `tdd_zero_drift_green_attempt2.log` | `906` | `70871531ee680da0cbc1b7b60cccab563f06a57ceaf917f6d5de21ac02326e74` |
| `zero_drift_initial.json` | `1,677` | `0eb255a5d52cccbf0e5e2f799b58f6f91ee688b3a22e949d936da15d4fd97f55` |
| `predict_once_raw.log` | `286` | `6214fb205a6247b9cb66b62f1851fb61b9be1961b1380750436e1539c3ae14a3` |
| `predict_once_process.env` | `150` | `de9eeaecb481bb1b5c92035a90a0b6f3843c7a383bb8efbef8f7aadd06ea9ee0` |
| `predict_semantic_result.json` | `644` | `b3ff7c9cdae765f4bef64313f5129b08453d8eb07237db8538a831384c1c87e3` |
| `zero_drift_post_predict.json` | `1,677` | `0eb255a5d52cccbf0e5e2f799b58f6f91ee688b3a22e949d936da15d4fd97f55` |
| `tdd_terminal_audit_red.log` | `1,247` | `82257ce56ab7c031bcb3c70127ce317fc4ec7e539056b3ae37dbb09dd845ba00` |
| `tdd_terminal_audit_green.log` | `241` | `5385b76fe210b0e4fef0271ac059ffc6987c5d85614b57f9a08b7fe7aee72b73` |
| `post_predict_terminal_audit.json` | `6,143` | `d8b13f725275e074ae90a01309b53ddcb894431925b97abcff5658322b3be394` |

### 10.9 Stop State

`RUN_ID=20260718T075949Z` is terminal, preserved, and non-reusable. `retry_allowed=false`; `live_authorized=false`. A later session may proceed only after external quota availability and another wholly fresh complete B2 chain. B3 live, B4, B5, Phase 1–9, D38 adoption, product/test implementation, clean-clone rehearsal, and final archive remain pending. The overall persistent goal remains active, not complete.

### 10.10 Documentation Validator Candidate

**Environment and scripts**

- Host conda environment: none.
- Python: `/usr/bin/python3`, Python `3.12.3`; bytecode disabled with `PYTHONDONTWRITEBYTECODE=1` and `-B`.
- Test: `.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/test_validate_session70_docs.py`.
- Validator: `.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z/validate_session70_docs.py`.

**Exact commands**

```bash
R=/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/runtime/sc26-ae-gate-b2-fresh-20260718T075949Z
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/test_validate_session70_docs.py"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$R/validate_session70_docs.py" --phase candidate
```

**Validation criteria**

1. All eight changed documents have a current Modification History row, final newline, balanced fences, no trailing whitespace, and exactly one relevant Session 70 terminal section.
2. Section 36 and the top-level Gate B row/checkpoint agree on terminal quota `129/128`; I62 remains open; I64, I67, and I68 retain their exact resolved scopes.
3. The terminal review includes all six required review fields; Section 10 includes environment, commands, criteria, expected/actual/delta metrics, evidence hashes, and stop state.
4. D38 identities are `4/4` exact; Session 70 seal/context/review/terminal-audit/predict-semantic/review-binding identities are `6/6` exact.
5. Authority changes remain limited to task docs/env handbook plus D38; staged paths and scoped submodule diff are empty; clean execution status is exactly the branch line plus `?? SC26-AE/`.
6. Runtime bytecode/live artifacts are `0/0`; capture and worker remain absent; exactly one quota report exists.

**Observed TDD and candidate results**

| Check | Expected | Actual | Delta | Result |
|---|---:|---:|---:|---|
| Missing-validator RED exit | nonzero | `1` | meets | PASS |
| Final helper tests | `9` | `9` | `0` | PASS |
| Candidate checks | all pass | `199/199` | `0` failures | PASS |
| Documents / Markdown failures | `8/0` | `8/0` | `0/0` | PASS |
| D38 exact identities | `4` | `4` | `0` | PASS |
| Session 70 exact identities | `6` | `6` | `0` | PASS |
| Authority status / unexpected / staged | task+D38 only / `0` / `0` | `21/0/0` | `0/0` violations | PASS |
| Scoped submodule diff bytes | `0` | `0` | `0` | PASS |
| Runtime bytecode / live artifacts | `0/0` | `0/0` | `0/0` | PASS |
| Capture / worker existence | `false/false` | `false/false` | none | PASS |
| Duplicate quota reports | `1` | `1` | `0` | PASS |
| Candidate Python / `tee` exit | `0/0` | `0/0` | `0/0` | PASS |
| Candidate elapsed | `>0ns` | `262,481,560ns` | positive | PASS |

`SESSION70_DOCS_VALIDATION=PASS`. candidate log bytes/SHA256=`4087/63cb26fa33024b17fe6ab860c387dbb32111c4b869a74f81754ff6851d723de6`; candidate process bytes/SHA256=`102/346d185e8910d3161945702fb5afaec9443c4034fc5ee372c5192f3222d2662f`. Candidate evidence modes are `0444/0444`.

**Failure history and resolution**

- Candidate Attempt 1 reported `20` failures because it correctly caught the stale top-level Gate B checkpoint while its own parser also mistook a fenced Bash comment for a Markdown heading. The parser received a fenced-code regression test and root-cause fix; the plan checkpoint was reconciled to Session 70.
- The first named candidate then reported one over-narrow case-sensitive literal mismatch despite the exact `129/128` identity being present. The contract now binds the exact numeric token rather than duplicating prose capitalization. Both failed logs are preserved; neither touched scheduler, harness, D38, capture, worker, product/test code, or live state.

This candidate PASS authorizes only the final no-further-edit documentation validation. It does not alter the terminal stop state: Gate B2 remains blocked by external `codesign` quota `129/128`, retry/live remain false, and the persistent goal remains active.
