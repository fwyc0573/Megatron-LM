# Test Report: SC'26 AE Task3 Local Workflow and Sim-Engine Verification

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Created the formal report for fresh Task3 unit/integration/e2e and sim-engine verification evidence |

## 1. Test Scope and Evidence Classification

This report records the fresh local verification of the SC'26 AE Task3 shell workflow and the directly related `megatron-sim-engine` unit/integration suites.

The Task3 fixture evidence is classified exactly as:

```text
local_synthetic_not_gpu_qualification
```

The fresh-chain Task2 metrics additionally identify themselves as:

```text
local_synthetic_not_two_gpu_qualification
```

The process-memory measurement records:

```text
local_synthetic_fixture
```

These tests validate local orchestration, explicit `fresh`/`prebaked` source selection, scheduler/simulator argument contracts, rank0 report semantics, manifest/checksum/provenance handling, fail-fast behavior, marker publication ordering, and a synthetic Task1-to-Task2-to-Task3 chain. They do **not** qualify a real GPU run, the current AE image, a real Nsight/NCU capture, a real slowdown predictor dataset, or a release-ready pre-dataset.

## 2. Environment

| Field | Actual Value |
|-------|--------------|
| Worktree | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717` |
| Branch | `sc26-ae-exec-clean-20260717` |
| Main-repository HEAD | `3c91d15bc035d49216161c9cac874f2453b69cb9` |
| Python executable | `/usr/bin/python` |
| Python version | `3.12.3` |
| Conda environment | `none` (`CONDA_DEFAULT_ENV` unset) |
| Execution hardware used by these tests | CPU/controller only; no GPU qualification was performed |
| Test date | `2026-07-19` |

Peak RSS was measured by the synthetic fixture with Python `resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss`. The controller did not provide `/usr/bin/time`, so this report does not describe the numbers as `/usr/bin/time -v` measurements.

## 3. Test Script Information

### 3.1 Task3 shell and fixture tests

| Test Type | Full Path |
|-----------|-----------|
| Unit | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task3_contracts.sh` |
| Integration | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task3_contract.sh` |
| Prebaked CPU e2e | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh` |
| Fresh-chain e2e | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/e2e/test_sc26_ae_fresh_chain.sh` |
| Synthetic fixture generator | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/fixtures/sc26_ae_task3_fixture.py` |

The shell syntax check also covered the three public Task3 entries and their shared library:

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/task3_gpt175b.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/task3_qwen3_a30b.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/task3_dsv3.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/lib/task3_simulation.sh`

### 3.2 Sim-engine tests

| Test Type | Full Path |
|-----------|-----------|
| Scheduler unit contract | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py` |
| Rank0 report unit contract | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/unit/test_rank0_report.py` |
| Rank0 report integration | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/integration/test_rank0_report_integration.py` |
| Slowdown asset builder unit | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py` |
| Slowdown simulation unit | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py` |
| Slowdown simulation integration | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py` |

## 4. Exact Reproducible Commands

### 4.1 Sim-engine unit/integration regression

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717

set -o pipefail
python --version
python -m pytest \
  megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py \
  megatron-sim-engine/tests/unit/test_rank0_report.py \
  megatron-sim-engine/tests/integration/test_rank0_report_integration.py \
  megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py \
  megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py \
  megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  -q
```

Observed command exit code: `0`.

### 4.2 Task3 shell syntax and local workflow regression

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717

set -euo pipefail
bash -n \
  SC26-AE/task3_gpt175b.sh \
  SC26-AE/task3_qwen3_a30b.sh \
  SC26-AE/task3_dsv3.sh \
  SC26-AE/lib/task3_simulation.sh \
  tests/unit/test_sc26_ae_task3_contracts.sh \
  tests/integration/test_sc26_ae_task3_contract.sh \
  tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh \
  tests/e2e/test_sc26_ae_fresh_chain.sh

bash tests/unit/test_sc26_ae_task3_contracts.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
```

Observed combined command exit code: `0`. Because the command used `set -euo pipefail` and reached the final fresh-chain PASS output, the syntax check and each of the four scripts exited `0`.

## 5. Validation Criteria

### 5.1 Task3 contract criteria

1. All three public Task3 entries and the shared library parse as valid shell.
2. `ARTIFACT_SOURCE` must be explicitly `fresh` or `prebaked`; missing, invalid, missing-selected-source, or cross-mixed inputs must fail without fallback.
3. Every scheduler and simulator invocation must carry the required explicit arguments, including `LOCAL_SIZE=8`, analytical communication, overlap enabled, and explicit slowdown model/scaler locations.
4. `TRACE_DIR` and `DATABASE_DIR` must resolve to the same canonical path.
5. Rank0 JSON and Markdown must agree and include positive `forward_step`, `backward_step`, and `optimizer_step` metrics.
6. Nested and outer manifests must verify size and SHA256 metadata before a verified marker is published.
7. Existing run destinations, corrupt checksums, builder failures, and incomplete report schemas must fail fast and must not publish or overwrite a verified marker.
8. The prebaked e2e must complete for all three public models.
9. The fresh e2e must connect synthetic Task1 artifacts, the shared Task2 predictor, exact backward `cmd_uid` assets, and Task3 through their public entries.
10. CPU memory evidence must exercise a real 32 MiB touched host allocation and report measured peak RSS.

### 5.2 Sim-engine criteria

1. Scheduler bf16/output-directory contracts pass.
2. Rank0 report calculations, schema validation, and fail-fast branches pass.
3. The canonical scheduler-generated integration path produces a valid rank0 report.
4. Existing slowdown builder and simulator unit/integration regressions remain green.
5. All collected tests pass with exit code `0`.

### 5.3 Acceptance boundary

The local acceptance condition is all syntax/unit/integration/e2e and sim-engine tests passing with verified synthetic artifacts. It is explicitly **not** the real AE acceptance condition. Real qualification still requires the target image and actual H800 execution, actual Task1 traces/memory/SQLite/NCU inputs, actual Task2 predictor evidence, actual Task3 runtime, and final pre-dataset checksum/provenance/data-quality qualification.

## 6. Test Results Summary

| Suite | Result | Passed / Total | Exit Code | Evidence |
|-------|--------|----------------|-----------|----------|
| Shell syntax | PASS | 8 paths parsed | `0` | Console output `SHELL_SYNTAX_EXIT=0` |
| Task3 unit | PASS | `6/6` | `0` | `/tmp/sc26-ae-task3-unit.Lb864B` |
| Task3 integration | PASS | `6/6` | `0` | `/tmp/sc26-ae-task3-integration.teSSpE` |
| Task3 prebaked CPU e2e | PASS | `3/3` models | `0` | `/tmp/sc26-ae-task3-prebaked-e2e.Rjcg1M` |
| Fresh Task1-to-Task2-to-Task3 e2e | PASS | `1/1` chain | `0` | `/tmp/sc26-ae-fresh-chain.ELEfpw` |
| Sim-engine related unit/integration | PASS | `45/45` | `0` | Fresh pytest console output |

Fresh sim-engine output:

```text
Python 3.12.3
.............................................                            [100%]
45 passed in 10.82s
```

### 6.1 Evidence-root inventory

| Evidence Root | File Count | Total File Bytes | Notes |
|---------------|-----------:|-----------------:|-------|
| `/tmp/sc26-ae-task3-unit.Lb864B` | `0` | `0` | Unit assertions report through stdout and intentionally leave no payload files |
| `/tmp/sc26-ae-task3-integration.teSSpE` | `148` | `119,183` | Positive and negative integration evidence |
| `/tmp/sc26-ae-task3-prebaked-e2e.Rjcg1M` | `114` | `105,041` | Three-model prebaked workflow and summary |
| `/tmp/sc26-ae-fresh-chain.ELEfpw` | `126` | `161,987` | Synthetic Task1/Task2/Task3 chain |

These `/tmp` roots are local transient evidence, not release artifacts.

## 7. Task3 Prebaked CPU E2E Numeric Results

| Metric | GPT-175B | Qwen3-A30B | DeepSeek-V3 |
|--------|---------:|------------:|------------:|
| Rank0 step time | `18.5 ms` | `22.5 ms` | `24.5 ms` |
| Forward duration sum | `5.0 ms` | `6.0 ms` | `6.5 ms` |
| Backward duration sum | `9.0 ms` | `11.0 ms` | `12.0 ms` |
| Optimizer duration sum | `2.0 ms` | `2.5 ms` | `3.0 ms` |
| Comp+comm diagnostic | `16.25 ms` | `19.75 ms` | `21.75 ms` |
| Simulator load time | `0.125 s` | `0.125 s` | `0.125 s` |
| Simulator execution time | `0.375 s` | `0.375 s` | `0.375 s` |
| Simulator wall-clock | `0.5 s` | `0.5 s` | `0.5 s` |
| Measured process wall-clock | `0.908497 s` | `0.899241 s` | `0.910960 s` |
| Peak RSS | `51,412 KiB` | `51,236 KiB` | `51,464 KiB` |
| Peak RSS | `0.049030304 GiB` | `0.048862457 GiB` | `0.049079895 GiB` |
| Touched host allocation | `32 MiB` | `32 MiB` | `32 MiB` |
| Outer manifest file count | `22` | `18` | `18` |
| Outer manifest SHA256 | `21c6740e8f0c808cc3cf174a003d39fc2464f2ce60c964d53050a65e9ab43484` | `5a144fc0c60dd45a2f1de8f7c50fd2864bd986c73ccbe5c3a0e464c25a3213a8` | `d2ebd62132f08fd42ad8f9923412058b22bd443d971fe3fbc7a91546c1c10714` |

The three rank0 component sums are diagnostic components and are not required to equal the overlap-aware rank0 step span.

## 8. Fresh Task1-to-Task2-to-Task3 Chain Results

### 8.1 Chain coverage and Task2 metrics

| Metric | Actual Value |
|--------|-------------:|
| Task1 trace files | `4` |
| Task1 memory JSON files | `4` |
| Task2 dataset rows | `2` |
| Task2 validation folds | `[1.0, 2.0, 3.0, 4.0, 5.0]` |
| Task2 average validation MSE | `3.0` |
| Task2 test MSE | `0.5` |
| Task2 model reload maximum absolute prediction delta | `0.0` |
| Task2 scaler feature/mean/scale counts | `2 / 2 / 2` |
| Task2 nonzero scale count | `2` |
| Task2 synthetic `run_all` elapsed time | `0.033742143 s` |
| Covered backward `cmd_uid` count | `4` |

The exact covered backward identities were:

```text
bwd-0
bwd-64
bwd-128
bwd-192
```

### 8.2 Fresh Task3 report and process metrics

| Metric | Actual Value |
|--------|-------------:|
| Artifact source | `fresh` |
| Model | `qwen3_a30b` |
| Rank0 step time | `22.5 ms` |
| Forward duration sum | `6.0 ms` |
| Backward duration sum | `11.0 ms` |
| Optimizer duration sum | `2.5 ms` |
| Comp+comm diagnostic | `19.75 ms` |
| Simulator load time | `0.125 s` |
| Simulator execution time | `0.375 s` |
| Simulator wall-clock | `0.5 s` |
| Measured process wall-clock | `0.874247 s` |
| Peak RSS | `51,248 KiB` (`0.048873901 GiB`) |
| Touched host allocation | `33,554,432 bytes` (`32 MiB`) |

### 8.3 Fresh-chain manifest evidence

| Artifact | Manifest File Count | Manifest Bytes | Manifest SHA256 |
|----------|--------------------:|---------------:|-----------------|
| Task1 | `13` | `3,619` | `f8456ecf7dbdaa077fd89bd4e0fd66fe9cc35e6e5d6f89212271f31ed9b8af37` |
| Task2 | `13` | `2,654` | `e2d538834890e600cb87eecfdfc42ba8496203442e1767cc6e66120e8a447f11` |
| Task3 | `17` | `3,656` | `cfaeae50589d1a9211ffe44034970ca9ee1fff853c821871aadbd0f5f0f75d9a` |

All three manifests returned `MANIFEST_STATUS=verified`. The Task3 builder consumed explicit trace, SQLite, NCU metrics, model, and scaler paths. The synthetic fixture checked exact backward coverage and did not read the prebaked distribution manifest on the fresh path.

## 9. Integration Fail-Fast Evidence

The `6/6` integration PASS covered the following positive and negative branches:

| Branch | Expected | Actual Evidence |
|--------|----------|-----------------|
| Relocated prebaked bundle | Portable resolution and complete explicit argv | PASS; scheduler and simulator each invoked with the required explicit fields |
| Successful Task3 run | Report, provenance, outer manifest, then marker | PASS; `18` manifest files verified |
| Existing run ID | Fail without stale-output or marker reuse | PASS; destination-exists error emitted |
| Corrupt prebaked payload | Fail before simulator and marker | PASS; file-size mismatch detected for `bundles/dsv3/trace/rank0.txt` |
| Fresh builder exits `37` | Propagate exact root-cause exit; no fallback | PASS; wrapper propagated `37`; empty stdout log remained valid |
| Missing optimizer report field | Reject report and do not publish marker | PASS; observed schema omitted `rank0_optimizer_step_duration_sum_ms` and was rejected |

The successful integration manifest SHA256 was:

```text
4cf54bfce2f1dc10f4ea9b84e52d8a3602f741984321153a06f0120a49e6d75f
```

Two successful synthetic tools intentionally produced no stdout, so their captured stdout logs were valid zero-byte files. The manifest contract allows `size_bytes == 0`; semantic payloads such as commands, reports, schedules, provenance, and slowdown assets were separately required to be non-empty and checksum-valid.

## 10. RED-to-GREEN Root-Cause Record

The earlier Task3 implementation cycle recorded these test defects before the fresh all-green run:

1. The four Task3 fixture/test files initially did not exist; invoking them returned exit `127`. The root-cause fix was to add the bounded fixture, integration, prebaked e2e, and fresh-chain e2e test assets.
2. Integration and prebaked e2e initially required every manifest file to be non-empty. The canonical manifest contract permits zero-byte files, and the synthetic scheduler/simulator legitimately produced no stdout. The tests were corrected to require `size_bytes >= 0` and a 64-character SHA256 for every entry, while requiring each semantic payload to be non-empty. No fake stdout content was added.
3. The fresh-chain test initially read the nonexistent key `model_reload_max_abs_delta`. The canonical Task2 schema key is `model_reload_max_abs_prediction_delta`; the test was corrected to the canonical key without changing the metrics schema.

The final fresh regression then passed all `6/6` unit, `6/6` integration, `3/3` prebaked model, `1/1` fresh-chain, and `45/45` sim-engine checks.

## 11. Evidence Excerpts

Task3 unit:

```text
PASS_COUNT=6
```

Task3 integration:

```text
PASS_COUNT=6
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
EVIDENCE_ROOT=/tmp/sc26-ae-task3-integration.teSSpE
```

Prebaked CPU e2e:

```text
PASS: all three Task3 public entries completed the strict prebaked CPU synthetic workflow.
MODEL_PASS_COUNT=3
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
EVIDENCE_ROOT=/tmp/sc26-ae-task3-prebaked-e2e.Rjcg1M
```

Fresh chain:

```text
PASS: Task1 capture, shared Task2 predictor, and fresh Task3 simulation formed one verified synthetic chain.
CHAIN_PASS_COUNT=1
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
EVIDENCE_ROOT=/tmp/sc26-ae-fresh-chain.ELEfpw
```

## 12. Synthetic-vs-Real Qualification Boundary

### Proven by this report

- The three Task3 public shell entries can execute the strict synthetic prebaked workflow.
- The Qwen3-A30B public Task1, Task2, and Task3 entries can form one synthetic fresh chain.
- Explicit source selection, no-fallback behavior, checksum/provenance verification, immutable run destinations, report validation, and verified-marker ordering are locally covered.
- The canonical sim-engine scheduler/report/slowdown unit and integration paths pass `45/45` tests.
- Local CPU process and host-allocation metrics are numerically recorded.

### Not proven by this report

- No H800 worker or multi-GPU topology was used.
- The current AE image, image digest, CUDA/NVML, `nsys`, and `ncu` runtime were not qualified.
- Synthetic traces, SQLite, NCU CSV, model, scaler, and slowdown assets are not the paper's real pre-dataset.
- No real performance or accuracy claim is supported by the synthetic timings.
- No distribution size gate, public clean clone, GitHub Release asset, or final AE bundle was qualified.
- This report must not be used to claim `AE-ready`, real GPU PASS, or final pre-dataset qualification.

## 13. Final Verdict

```text
LOCAL TASK3/SIM-ENGINE VERIFICATION: PASS
REAL GPU QUALIFICATION: NOT PERFORMED BY THIS REPORT
REAL PRE-DATASET QUALIFICATION: NOT PERFORMED BY THIS REPORT
AE-READY CLAIM: NOT AUTHORIZED BY THIS REPORT
```

There is no remaining Task3 **local synthetic test blocker** in the covered scope. Real GPU execution and final reusable pre-dataset qualification remain separate required gates.
