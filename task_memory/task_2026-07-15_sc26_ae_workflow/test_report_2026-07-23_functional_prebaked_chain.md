# Test Report: SC'26 Functional Prebaked Chain

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Added exact-producer clean-clone build/verify and final GPT/Qwen CPU-only Task3 evidence |
| 2026-07-23 | Recorded final after-action documentation/static verification without rerunning Task2 or the heavy workflows |
| 2026-07-23 | Added final shell/Python/static checks and the 40-test functional packaging regression |
| 2026-07-23 | Recorded verified GPT/Qwen Fresh artifacts, functional bundle verification, and both CPU-only Task3 executions |

**Date:** 2026-07-23  
**Repository:** `/data/ycfeng/sc26_ae_task3_qwen`  
**Exact producer commit under test:** `c7288c66f0a6c3d0445edc841a6e5982d3b22f09`  
**Clean clone:** `/data/ycfeng/tmp/sc26_ae_clean_clone_20260723T060352_c7288c6`  
**Nested Echo commit:** `1390b4416ded08bc1b9cd0620d329d81d4470bf9`  
**Nested sim-engine commit:** `51eed0404635632fd52a99b3f372d5830b1d73b4`

## 1. Test Script Information

### Scripts and tools

- `/data/ycfeng/sc26_ae_task3_qwen/SC26-AE/task3_gpt175b.sh`
- `/data/ycfeng/sc26_ae_task3_qwen/SC26-AE/task3_qwen3_a30b.sh`
- `/data/ycfeng/sc26_ae_task3_qwen/SC26-AE/lib/task3_simulation.sh`
- `/data/ycfeng/sc26_ae_task3_qwen/SC26-AE/tools/package_prebaked.py`
- `/data/ycfeng/sc26_ae_task3_qwen/SC26-AE/tools/artifact_manifest.py`

### Environment

- Controller OS Python: `/usr/bin/python3`, Python `3.12.3`
- `numpy=2.4.6`
- `pandas=3.0.3`
- `scipy=1.17.1`
- `scikit-learn=1.9.0`
- Task3 XGBoost layer:
  `/data/ycfeng/tmp/sc26_ae_cpu_task3_pydeps_xgboost210_20260723`
- `xgboost=2.1.0`
- CPU-only runs set `CUDA_VISIBLE_DEVICES=""`.
- All temporary paths, caches, logs, outputs, and staging roots use `/data/ycfeng/tmp`; `/tmp` was
  not used.

### Intermediate Session 60 functional bundle verification command

```bash
cd /data/ycfeng/sc26_ae_task3_qwen
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1

python3 SC26-AE/tools/package_prebaked.py verify-functional \
  --repo-root "$PWD" \
  --prebaked-root /data/ycfeng/tmp/sc26_ae_functional_prebaked_20260722T210958Z
```

### Intermediate Session 60 GPT CPU-only Task3 command

```bash
cd /data/ycfeng/sc26_ae_task3_qwen
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/data/ycfeng/tmp/sc26_ae_cpu_task3_pydeps_xgboost210_20260723${PYTHONPATH:+:$PYTHONPATH}"

AE_OUTPUT_ROOT=/data/ycfeng/tmp/sc26_ae_cpu_prebaked_20260722T213203Z \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/data/ycfeng/tmp/sc26_ae_functional_prebaked_20260722T210958Z \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
TASK3_SIMULATION_RUN_ID=gpt175b-prebaked-cpu-20260722T213203Z \
bash SC26-AE/task3_gpt175b.sh
```

### Intermediate Session 60 Qwen CPU-only Task3 command

```bash
cd /data/ycfeng/sc26_ae_task3_qwen
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/data/ycfeng/tmp/sc26_ae_cpu_task3_pydeps_xgboost210_20260723${PYTHONPATH:+:$PYTHONPATH}"

AE_OUTPUT_ROOT=/data/ycfeng/tmp/sc26_ae_cpu_prebaked_20260722T213415Z \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/data/ycfeng/tmp/sc26_ae_functional_prebaked_20260722T210958Z \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
TASK3_SIMULATION_RUN_ID=qwen3_a30b-prebaked-cpu-20260722T213415Z \
bash SC26-AE/task3_qwen3_a30b.sh
```

### Artifact verification command

```bash
python3 SC26-AE/tools/artifact_manifest.py verify \
  --root <task3-run-directory> \
  --manifest <task3-run-directory>/artifact_manifest.json
```

No `SC26-AE/task2_*.sh` command was executed during this completion phase.

### Exact Session 61 clean-clone bundle commands

```bash
cd /data/ycfeng/tmp/sc26_ae_clean_clone_20260723T060352_c7288c6
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1

python3 SC26-AE/tools/package_prebaked.py build-functional \
  --repo-root "$PWD" \
  --output-root /data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4 \
  --staging-root /data/ycfeng/tmp/sc26_ae_functional_prebaked_clean_20260723T063212_c7288c6 \
  --distribution-id sc26-ae-functional-clean-20260723T063212-c7288c6 \
  --result-json /data/ycfeng/tmp/sc26_ae_functional_clean_build_results/sc26-ae-functional-clean-20260723T063212-c7288c6.json

python3 SC26-AE/tools/package_prebaked.py verify-functional \
  --repo-root "$PWD" \
  --prebaked-root /data/ycfeng/tmp/sc26_ae_functional_prebaked_clean_20260723T063212_c7288c6
```

### Exact Session 61 GPT CPU-only Task3 command

```bash
cd /data/ycfeng/tmp/sc26_ae_clean_clone_20260723T060352_c7288c6
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/data/ycfeng/tmp/sc26_ae_cpu_task3_pydeps_xgboost210_20260723${PYTHONPATH:+:$PYTHONPATH}"

AE_OUTPUT_ROOT=/data/ycfeng/tmp/sc26_ae_clean_final_cpu_20260723T063628_gpt \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/data/ycfeng/tmp/sc26_ae_functional_prebaked_clean_20260723T063212_c7288c6 \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
TASK3_SIMULATION_RUN_ID=gpt175b-clean-final-cpu-20260723T063628 \
bash SC26-AE/task3_gpt175b.sh
```

### Exact Session 61 Qwen CPU-only Task3 command

```bash
cd /data/ycfeng/tmp/sc26_ae_clean_clone_20260723T060352_c7288c6
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/data/ycfeng/tmp/sc26_ae_cpu_task3_pydeps_xgboost210_20260723${PYTHONPATH:+:$PYTHONPATH}"

AE_OUTPUT_ROOT=/data/ycfeng/tmp/sc26_ae_clean_final_cpu_20260723T063749_qwen \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/data/ycfeng/tmp/sc26_ae_functional_prebaked_clean_20260723T063212_c7288c6 \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
TASK3_SIMULATION_RUN_ID=qwen3_a30b-clean-final-cpu-20260723T063749 \
bash SC26-AE/task3_qwen3_a30b.sh
```

## 2. Validation Criteria

1. GPT Task1 contains exactly ranks `0,128,256,384,512,640,768,896`; Qwen Task1 contains exactly
   ranks `0,8,16,...,248`.
2. Both Task1 captures use global rank 0 for NCU/slowdown provenance.
3. The reused Task2 artifact records two real GPUs and passes its manifest/checksum validation.
4. GPT and Qwen Fresh Task3 each contain a non-empty report, manifest, and marker.
5. The functional distribution verifies with exactly three bundles: `gpt175b`, `qwen3_a30b`, and
   `shared_task2`.
6. GPT and Qwen CPU-only Task3 exit `0`, explicitly run with slowdown enabled, and produce verified
   report/manifest/marker artifacts.
7. Every report timing is finite and nonnegative.
8. `simulator_wall_clock_s` equals `simulator_load_time_s + simulator_execution_time_s` within
   floating-point tolerance.
9. Each marker binds the exact artifact-manifest SHA256 and records:
   `artifact_source=prebaked`, `execution_evidence=local_synthetic_not_gpu_qualification`,
   `ncu_metrics_source=task1_rank0`, `slowdown_trace_scope=global_rank_0`, and
   `slowdown_trace_rank_ids=[0]`.
10. Task2 commands executed in this phase equal zero. Accuracy/fidelity is not an acceptance
    criterion for this functional workflow.

## 3. Test Results and Evidence

### Session 60 intermediate result summary

| Test / artifact | Result | Numeric evidence |
|-----------------|--------|------------------|
| Shared two-GPU Task2 artifact | PASS / reused | GPUs `0,1`; manifest files `18`; dataset rows `727`; model/scaler bytes `412174/616` |
| GPT Fresh Task1 | PASS | traces `8`; memory files `8`; NCU feature rows `6270` |
| Qwen Fresh Task1 | PASS | traces `32`; rank vector `0,8,...,248`; NCU feature rows `24` |
| GPT Fresh Task3 | PASS | manifest files `1049`; slowdown occurrences `4856`; missing-skip `0` |
| Qwen Fresh Task3 | PASS | manifest files `281`; exact occurrences `24288`; missing-skip `0` |
| Functional distribution | PASS | bundles `3`; files `375`; total bytes `6,554,852,341` |
| GPT CPU-only Task3 | PASS | exit `0`; outer wall `62 s`; manifest files `1049` |
| Qwen CPU-only Task3 | PASS | exit `0`; outer wall `1176 s`; manifest files `281` |
| Task2 entry commands in current phase | PASS | `0` |

### Session 61 exact-producer clean-clone result summary

| Test / artifact | Result | Numeric evidence |
|-----------------|--------|------------------|
| Older bundle from clean clone | EXPECTED FAIL | exit `1`; exact producer `4dad1774...` differs from checkout `c7288c66...` |
| Final functional distribution build | PASS | exit `0`; bundles/files/bytes `3/375/6,554,852,354` |
| Final functional distribution verify | PASS | exit `0`; evidence `functional_prebaked_not_release_qualified` |
| Final GPT CPU-only Task3 | PASS | exit `0`; outer wall `63 s`; manifest files `1049` |
| Final Qwen CPU-only Task3 | PASS | exit `0`; outer wall `1184 s`; manifest files `281` |
| Clean tracked status | PASS | outer/Echo/sim-engine status lines `0/0/0` |
| Task2 entry commands/processes | PASS | `0/0` |

### Shared Task2 metrics

| Metric | Expected | Actual | Delta / interpretation |
|--------|----------|--------|------------------------|
| Physical GPUs | exactly `2` | `2` (`CUDA_VISIBLE_DEVICES=0,1`) | `0` |
| Dataset rows | positive | `727` | positive |
| Average validation MSE | finite, nonnegative | `0.04124828706619175` | functional only |
| Test MSE | finite, nonnegative | `0.061428837844613504` | functional only |

Task2 manifest SHA256:
`d344fbfc0f4e56286efe9dd5ee6ac3f125ed3ad34fe8e4599bc9a71f67dda76e`.

### Fresh Task3 metrics

| Metric | GPT-175B | Qwen3-A3B |
|--------|----------|------------|
| `rank0_step_time_ms` | `8276.64` | `3051.24` |
| `rank0_forward_step_duration_sum_ms` | `1905.12` | `0.32` |
| `rank0_backward_step_duration_sum_ms` | `99.24` | `26.29` |
| `rank0_optimizer_step_duration_sum_ms` | `53.61` | `3.62` |
| `simulator_load_time_s` | `14.201178` | `70.935776` |
| `simulator_execution_time_s` | `18.494841` | `760.9432` |
| `simulator_wall_clock_s` | `32.696019` | `831.878976` |

GPT Fresh Task3 manifest SHA256:
`02b89c32f2d3c55628858709b8519933a73dd1a5d7339e1602bcab5125bd161f`.
Qwen Fresh Task3 manifest SHA256:
`805e646704ec9680481722f75d8df132ccffbab99afcee41f4c4a414b8512a9b`.

### Session 60 intermediate CPU-only prebaked Task3 metrics

| Metric | GPT-175B | Qwen3-A3B |
|--------|----------|------------|
| `rank0_step_time_ms` | `8276.64` | `3051.24` |
| `rank0_forward_step_duration_sum_ms` | `1905.12` | `0.32` |
| `rank0_backward_step_duration_sum_ms` | `99.24` | `26.29` |
| `rank0_optimizer_step_duration_sum_ms` | `53.61` | `3.62` |
| `rank0_comp_plus_comm_diagnostic_ms` | `13682.09` | `6221.86` |
| `simulator_load_time_s` | `17.566988` | `84.0241` |
| `simulator_execution_time_s` | `23.052294` | `1066.334488` |
| Expected wall clock (`load + execution`) | `40.619282` | `1150.358588` |
| Actual `simulator_wall_clock_s` | `40.619282` | `1150.358588` |
| Absolute wall-clock delta | `0.0` | `0.0` |

All values above are finite and nonnegative. They demonstrate functional data flow only and are not
accuracy or fidelity claims.

### Session 61 final clean-clone CPU-only Task3 metrics

| Metric | GPT-175B | Qwen3-A3B |
|--------|----------|------------|
| `rank0_step_time_ms` | `8276.64` | `3051.24` |
| `rank0_forward_step_duration_sum_ms` | `1905.12` | `0.32` |
| `rank0_backward_step_duration_sum_ms` | `99.24` | `26.29` |
| `rank0_optimizer_step_duration_sum_ms` | `53.61` | `3.62` |
| `rank0_comp_plus_comm_diagnostic_ms` | `13682.09` | `6221.86` |
| Stored `simulator_load_time_s` | `18.015626` | `84.063936` |
| Stored `simulator_execution_time_s` | `23.576013` | `1072.67156` |
| Derived expected wall clock (`load + execution`) | `41.591639` | `1156.735496` |
| Stored `simulator_wall_clock_s` | `41.591639` | `1156.735496` |
| Derived absolute wall-clock delta | `0.0` | `0.0` |

All stored report values are finite and nonnegative. The expected wall clock and absolute delta are
derived during validation; they are not fields in `sc26-ae-rank0-report-v1`.

### Session 60 intermediate artifact hashes

| Artifact | SHA256 |
|----------|--------|
| Functional distribution manifest | `4e07f8f705c7662a60452f0992b01d9a817adb22db40b80b5f6e7a874c972985` |
| GPT CPU report | `5490e933ab564ce4b168684b5301fa525bbffee174b0c819c6e27446f6a4e8b3` |
| GPT CPU manifest | `083a92a613df3538fbfc259b95e470df363f64988d5ad578e27c9918692d8f04` |
| GPT CPU marker | `5095b4100c1dd4b2b0a76f44b4120bead5f8b7255e5be991d47ece386ae20302` |
| Qwen CPU report | `00982a081c9385eca97554e21ccdd1c835736f3c36ac6b20c7ac489e3d6d0dca` |
| Qwen CPU manifest | `0cbb754e43f91bcf93eb1581442235314006ceef82834e8132c241a795a8520d` |
| Qwen CPU marker | `89e058d04128948428083718fca8fa8e5683bce3e873bbb2de5164ba5a1cf8c1` |

Manifest verification output:

```text
GPT:  MANIFEST_STATUS=verified; MANIFEST_FILE_COUNT=1049
Qwen: MANIFEST_STATUS=verified; MANIFEST_FILE_COUNT=281
Distribution: DISTRIBUTION_STATUS=verified; DISTRIBUTION_BUNDLE_COUNT=3;
              DISTRIBUTION_FILE_COUNT=375; DISTRIBUTION_TOTAL_SIZE_BYTES=6554852341
```

### Session 61 final artifact hashes

| Artifact | SHA256 |
|----------|--------|
| Final distribution manifest | `6e9ab347df9107b9f2ada1900db47e56f246f6b445ed06d62b5e2b4960a50468` |
| Final build result | `bc2947922b1711d5f17db386ddde3c279fd11f8d6bb84750bb07b17609888109` |
| Final build log | `ee6c53c951d1665c7ad33a2085241af69b81c170328f94df4ec1a07aea1e65a0` |
| Final verify log | `02e3f4f5a876ba8eec446dba76520197931dced767d079615e85f516adfad351` |
| GPT final report JSON | `20ea1637fe418915be987a51caa5cc7e3c37378c4a4e718c02becda97e485ecb` |
| GPT final report Markdown | `0033ce8e86a77982defbee55f3458788d0a020e2656402a21d031356ef104650` |
| GPT final manifest | `d2f2838d4f605645b9258b2caf26250a7956a4c73fe850ed63d64ce6a5f55534` |
| GPT final marker | `396096f052448b25b87ad20a28ed3309ad8095a86e4e81591aec08e5a0292d9e` |
| Qwen final report JSON | `08700cba92a3459362bb682ba5e09069c1f47575b7cbe4d753ef51c48ecefc57` |
| Qwen final report Markdown | `0f6321af799311b30e062044b1b99655c9d4f04577589e3da74f96cfa9d72a70` |
| Qwen final manifest | `4f2903707633e88c62c6e55d98e9fb9ecd9cb4acf9098e2a733aee2c46886a89` |
| Qwen final marker | `2c0c3d0c40e60da84648ffd9c9985d2a4d3b4e74100a337aa7668ebf520d9a04` |
| Preserved exact-producer failure log | `64646269b1945f2e45612a519d220d7e10c855aa2da3ca663769a0fcaecba96e` |

Final manifest verification output:

```text
GPT:  MANIFEST_STATUS=verified; MANIFEST_FILE_COUNT=1049
Qwen: MANIFEST_STATUS=verified; MANIFEST_FILE_COUNT=281
Distribution: DISTRIBUTION_STATUS=verified; DISTRIBUTION_BUNDLE_COUNT=3;
              DISTRIBUTION_FILE_COUNT=375; DISTRIBUTION_TOTAL_SIZE_BYTES=6554852354
```

### Failure and root-cause record

The first GPT CPU run at
`/data/ycfeng/tmp/sc26_ae_cpu_prebaked_20260722T212843Z` failed after scheduler/materialization
because controller Python lacked `xgboost`. The failure was not caused by missing kernel features.
Installing the exact Task2 XGBoost version in a dedicated `/data/ycfeng/tmp` target directory and
binding `PYTHONPATH` resolved the runtime dependency. The successful GPT and Qwen runs then passed
without modifying or rerunning Task2. The failed run and partial venv directory were preserved.

The first clean-clone verification used the older Session 60 bundle and failed with
`gpt175b functional bundle producer differs from the current checkout`. Root cause: functional
verification deliberately requires exact equality between the bundle producer and current
checkout, and the older producer was `4dad1774...` while the final clean clone was `c7288c66...`.
The resolution was to rebuild only the functional distribution from `c7288c66...` using the same
verified Fresh and Task2 inputs. The verifier was not relaxed, no descendant fallback was added,
and Task2 was not rerun.

### Final current-worktree regression

Commands:

```bash
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONPYCACHEPREFIX=/data/ycfeng/tmp/sc26_ae_pycache_final_20260723_current
export PYTHONDONTWRITEBYTECODE=1

bash -n \
  SC26-AE/task3_gpt175b.sh \
  SC26-AE/task3_qwen3_a30b.sh \
  SC26-AE/lib/task3_simulation.sh

python3 -m py_compile \
  SC26-AE/tools/package_prebaked.py \
  SC26-AE/tools/artifact_manifest.py

git diff --check

python3 -B -m pytest -q tests/unit/test_sc26_ae_package_prebaked.py
```

Results:

| Check | Result | Numeric evidence |
|-------|--------|------------------|
| Shell syntax | PASS | 3 changed-path dependencies, exit `0` |
| Python compile | PASS | 2 tools, exit `0` |
| Git whitespace check | PASS | exit `0` |
| Package regression | PASS | `40/40` tests in `10.81 s`, exit `0` |
| `sc26-ad.tex` unchanged | PASS | changed-path count `0` |
| Running Task2 script commands | PASS | process count `0` |

Pytest transcript:
`/data/ycfeng/tmp/sc26_ae_package_prebaked_pytest_final_20260723.log`, SHA256
`788ca2e1bf6077a73a3913ae12e1ac38f6dc659650e8653b875d4d77bb163752`.

### Final after-action documentation/static verification

The Session 61 after-action update changed documentation only. The final targeted checks were:

| Check | Result | Numeric evidence |
|-------|--------|------------------|
| Task3 shell syntax | PASS | `3/3`, exit `0` |
| Package-tool Python compilation | PASS | `2/2`, exit `0` |
| Documentation semantic assertions | PASS | `7/7`, exit `0` |
| Git whitespace check | PASS | exit `0` |
| `sc26-ad.tex` changed paths | PASS | `0` |
| Running Task2 script processes | PASS | `0` |
| Echo/sim-engine tracked status lines | PASS | `0/0` |

The `40/40` package regression above was not repeated because no executable source changed after
that run. No Task2 command, GPU workflow, simulator execution, or large harness was rerun during
the documentation closeout.

## 4. Final Boundary

The two-model Fresh and functional paths, including exact-producer clean-clone replay, pass for the
requested fake-level functional scope. Therefore:

```text
Goal 1=PASS
Goal 2=PASS
functional-fake-level-AE-ready=YES
release-ready=NO
distributed-accuracy-qualified=NO
paper-fidelity-reproduced=NO
Task2 commands rerun=0
```
