## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Closed I48 local documentation/static harness with scope-corrected status-aware EXIT=0 evidence and retained all RED logs |
| 2026-07-19 | Recorded a transient final-verifier shell quoting error and its corrected rerun; qualification boundary remains unchanged |
| 2026-07-19 | Added Session 43 current-state supersession after the final control-plane regression; earlier continuation metrics remain historical |
| 2026-07-19 | Added continuation aggregate evidence after D30 temp-root portability repair; real qualification boundary remains open |

# Test Report: SC'26 AE Workflow Continuation Aggregate

**Date:** 2026-07-19  
**Worktree:** /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717  
**Branch:** sc26-ae-exec-clean-20260717  
**Outer HEAD:** c217ce93156e7c37e065da2989c1a482f12ecebc  
**Nested megatron-sim-engine HEAD:** 39755169f73f6c748e8d7376c3a2158c6569436b  
**Environment:** Python 3.12.3, pytest 9.1.1, CONDA_DEFAULT_ENV=none  
**Test temporary root:** SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp, TMPDIR=/data/ycfeng/sc26-ae-test-tmp  
**Evidence boundary:** all local rows below are synthetic/control-plane evidence and are not H800 qualification.

## 1. Test Script Information

All commands were run from:

    cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717
    export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
    export TMPDIR=/data/ycfeng/sc26-ae-test-tmp
    mkdir -p "$SC26_AE_TMP_ROOT"

### Targeted local regression

    bash tests/unit/test_sc26_ae_docs_contract.sh
    bash tests/unit/test_sc26_ae_setup_runtime.sh
    bash tests/integration/test_sc26_ae_setup.sh
    bash tests/unit/test_sc26_ae_common.sh
    bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
    bash tests/unit/test_sc26_ae_task2_snapshot.sh
    bash tests/unit/test_sc26_ae_task3_provenance.sh
    bash tests/unit/test_sc26_ae_task3_contracts.sh
    bash tests/integration/test_sc26_ae_task1_contracts.sh
    bash tests/integration/test_sc26_ae_task2_contract.sh
    bash tests/integration/test_sc26_ae_task3_contract.sh
    bash tests/integration/test_sc26_ae_task3_portability.sh
    python3 -B -m pytest -q tests/unit/test_sc26_ae_artifact_manifest.py tests/unit/test_sc26_ae_echo_metrics.py tests/unit/test_sc26_ae_package_prebaked.py tests/unit/test_sc26_ae_seal_qualification.py

Exact transcript: task_memory/task_2026-07-15_sc26_ae_workflow/logs/local-regression-20260719-continuation.log  
Bytes: 8,885  
SHA256: be9147197753d3059159967a6ecdaf5bf3aaa6866d009f2da85e4ac2b1cff14c

### Public-entry and portability e2e

    bash tests/e2e/test_sc26_ae_task1_smoke.sh
    bash tests/e2e/test_sc26_ae_task2_smoke.sh
    bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
    bash tests/e2e/test_sc26_ae_fresh_chain.sh
    bash tests/e2e/test_sc26_ae_clean_clone_replay.sh

Transcript: task_memory/task_2026-07-15_sc26_ae_workflow/logs/e2e-regression-20260719-continuation.log  
Bytes: 5,054  
SHA256: 1a89cf300c3aa7fbeeb30714e5b81400f4c38c94619c90842aa04b6a54f55056

### Setup and affected existing integration suites

    bash tests/unit/test_setup_grouped_gemm_v1.sh
    TEST_ROOT=/data/ycfeng/sc26-ae-test-tmp/gpt-example-mock-20260719 bash tests/integration/test_gpt_example_mock_mode.sh

Transcripts:

- logs/grouped-gemm-unit-20260719-continuation.log: 1,788 bytes, SHA256 d14e2cdc7e6f541a1fe640274f6b3fdc3dc8977d8633d8645519b91b526422b6
- logs/gpt-example-mock-20260719-continuation.log: 1,315 bytes, SHA256 c66d2b8c8ec49a6f7b8362bc45d7498cd8b273c9efb6179e16f8c80ea3d7bdf4

### Static checks

    bash -n SC26-AE/*.sh SC26-AE/lib/*.sh tools/ae/setup_grouped_gemm_v1.sh tests/unit/test_sc26_ae_*.sh tests/unit/test_setup_grouped_gemm_v1.sh tests/integration/test_sc26_ae_*.sh tests/e2e/test_sc26_ae_*.sh
    python3 AST syntax walk over SC26-AE and SC26-AE test Python files
    git diff --check

Transcript: logs/static-shell-python-20260719-continuation.log  
Bytes: 99  
SHA256: 68ae7f77215c5d183f47b476fc201f234e2060bb5fbc309fbbb418a0873933d4

## 2. Validation Criteria

- All changed test paths execute from the configured writable temporary root.
- Nine public entries remain present: 3 Task1 + 3 Task2 + 3 Task3.
- Test failures remain fail-fast; no assertion, threshold, checksum, provenance, source-selection, or evidence-class rule is weakened.
- Fresh Task1 to Task2 to Task3 output contains verified markers and manifests.
- Relocated/prebaked paths remain portable and reject unsafe paths, symlinks, stale output, and wrong evidence classes.
- Numeric Task2 and Task3 fields are finite, positive where required, and internally consistent.
- Local results remain explicitly local_synthetic_not_gpu_qualification.
- No real H800 qualification is claimed from this run.

## 3. Test Results and Evidence

| Suite | Result | Numeric evidence | Exit |
|---|---|---:|---:|
| Documentation contract | PASS | public entries=9; paper suggestions=10 | 0 |
| Fixed-runtime setup verifier | PASS | 21/21 | 0 |
| Setup source/status integration | PASS | 6/6; real installer executions=0 | 0 |
| Common AE helpers | PASS | 7/7 | 0 |
| Task2 interpreter contract | PASS | fixed interpreter binding=1/1 | 0 |
| Task2 snapshot/reuse contract | PASS | manifest files=12 per synthetic snapshot; gpt/qwen/dsv3 paths exercised | 0 |
| Task3 provenance unit | PASS | 5/5; PROVENANCE_TEST_STATUS=PASS | 0 |
| Task3 unit contract | PASS | 6/6 | 0 |
| Task1 integration contract | PASS | 10/10; 3 model entries | 0 |
| Task2 integration contract | PASS | manifest files=13; model attachments=3/3 | 0 |
| Task3 integration contract | PASS | 6/6 | 0 |
| Task3 portability/provenance | PASS | 14/14 | 0 |
| Artifact/metrics/package/sealing pytest | PASS | 60 passed in 76.24 s | 0 |
| Grouped-gemm setup unit | PASS | 37/37 | 0 |
| GPT example integration | PASS | 22/22 | 0 |
| Task1 public smoke | PASS | 10/10; SMOKE_PASS_COUNT=1 | 0 |
| Task2 public smoke | PASS | 3 public entries; manifest files=13 | 0 |
| Task3 prebaked CPU e2e | PASS | models=3/3 | 0 |
| Fresh chain e2e | PASS | CHAIN_PASS_COUNT=1 | 0 |
| Clean-clone-style replay | PASS | Task1/Task2/Task3 entries=3/3/3; setup cases=6; chain=1; all clone statuses clean | 0 |
| Shell syntax | PASS | 33 scripts | 0 |
| Python AST syntax | PASS | 27 files | 0 |
| Diff hygiene | PASS | git diff --check | 0 |

## 4. Key Numeric Metrics

### Task1 local synthetic smoke

    REAL_GPU_WORKLOAD_COUNT=0
    Task1 contract cases=10
    public Task1 entries=3

### Task2 local synthetic chain

    TASK2_DATASET_ROWS=2
    TASK2_AVERAGE_VALIDATION_MSE=3.0
    TASK2_TEST_MSE=0.5
    TASK2_MODEL_RELOAD_MAX_ABS_PREDICTION_DELTA=0.0

These values validate schema and producer/consumer wiring only; they are not real predictor-quality claims.

### Task3 prebaked CPU synthetic runs

| Model | Rank0 step (ms) | Forward (ms) | Backward (ms) | Optimizer (ms) | Load (s) | Execution (s) | Simulator wall (s) | Process wall (s) | Peak RSS (KiB) | Peak RSS (GiB) | Tested allocation (MiB) | Manifest files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-175B | 18.5 | 5.0 | 9.0 | 2.0 | 0.125 | 0.375 | 0.5 | 0.913777 | 51,344 | 0.048965454 | 32 | 22 |
| Qwen3-A30B | 22.5 | 6.0 | 11.0 | 2.5 | 0.125 | 0.375 | 0.5 | 0.901115 | 51,340 | 0.048961639 | 32 | 18 |
| DeepSeek-V3 | 24.5 | 6.5 | 12.0 | 3.0 | 0.125 | 0.375 | 0.5 | 0.910986 | 51,316 | 0.048938751 | 32 | 18 |

Fresh-chain Qwen3-A30B repeated metrics:

    Task1 trace files=4
    Task1 memory JSON=4
    Task3 backward CMD UID count=4
    rank0 step/forward/backward/optimizer=22.5/6.0/11.0/2.5 ms
    simulator load/execution/wall=0.125/0.375/0.5 s
    process wall=1.258732 s
    peak RSS=51,252 KiB=0.048877716 GiB
    tested host allocation=32 MiB

### Clean-clone replay

    PUBLIC_TASK1_ENTRY_COUNT=3
    PUBLIC_TASK2_ENTRY_COUNT=3
    PUBLIC_TASK3_ENTRY_COUNT=3
    SETUP_CONTRACT_CASE_COUNT=6
    FRESH_ATOMIC_CHAIN_COUNT=1
    OUTER_CLONE_STATUS=clean
    ECHO_SUBMODULE_STATUS=clean
    SIM_ENGINE_STATUS=clean
    COLLECTIVE_SIM_STATUS=clean
    EVIDENCE_CLASS=local_synthetic_not_gpu_qualification

## 5. D30 RED to GREEN Repair Record

### Motivation and expectation

The environment had no free /tmp inodes (IUse=100%). AE test scripts that hard-coded mktemp /tmp templates could not run even when a writable project-local temporary root was supplied. The expectation was that every AE-facing test/rehearsal would honor the configured SC26_AE_TMP_ROOT/TMPDIR without changing any acceptance or evidence rule.

### Observed RED

    mktemp: failed to create directory via template /tmp/sc26-ae-setup-integration.XXXXXX: No space left on device
    exit=1

The same root cause was present in the grouped-gemm setup unit's hard-coded temporary roots.

### Root cause

The test fixtures bypassed the documented writable temporary-root contract by embedding /tmp directly in mktemp templates. This was a test portability defect, not a product/runtime or qualification result.

### Minimal method

- Added a TMP_PARENT selection from SC26_AE_TMP_ROOT, TMPDIR, or /tmp and created the directory before mktemp.
- Changed mktemp templates to use the selected TMP_PARENT.
- Applied the same isolated-root change to the 37 grouped-gemm setup unit cases.
- Preserved all assertions, thresholds, source-selection rules, checksum/provenance checks, and evidence labels.
- Corrected one mechanical quote error immediately after its RED shell-syntax observation: bash -n initially reported unexpected EOF while looking for matching quote; all 37 templates were closed and bash -n rerun.

### GREEN and affected regression

    SC26_HARDCODED_TMP_COUNT=0
    SHELL_SYNTAX_COUNT=33
    PYTHON_AST_SYNTAX_COUNT=27
    DIFF_CHECK=PASS
    local regression aggregate=0
    e2e aggregate=0
    grouped-gemm setup=37/37
    GPT example integration=22/22

Evidence class: local_synthetic_setup_contract / local_synthetic_not_gpu_qualification.

## 6. Environment and Qualification Boundary

The CPU controller exposes nvidia-smi and rlaunch, but nvidia-smi -L listed no GPU. No new RJob was submitted in this continuation. The H800 runtime test tests/integration/test_grouped_gemm_v1_runtime.py therefore remains a real-H800 qualification check, not a local synthetic test.

The newest authoritative D45 predict-only evidence remains:

    CLI exit=0
    semantic output=fail to pass quota check: gpu : 129/128
    normalized semantic quota=gpu=129/128
    semantic status=FAIL

This keeps:

    Echo exact-two-H800=BLOCK
    Integrated Gate B1=BLOCK
    real_pre_dataset=NOT QUALIFIED
    release_pre_dataset=NOT QUALIFIED
    AE-ready=NO
    Task=INCOMPLETE

## 7. Open Items

1. Obtain external quota/resource availability and run the authorized exact-two-H800 qualification with the pinned image hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae.
2. Produce complete real GPT-175B, Qwen3-A30B, and DeepSeek-V3 Task1 to Task2 to Task3 chains with full-rank coverage, atomic producer/consumer provenance, checksums, data-quality evidence, and portable manifests.
3. Run final real-container and public clean-clone/release matrix after the required external publication authorization.
4. Keep local synthetic evidence explicitly separate from the release pre-dataset.

## Session 43 Current-State Supersession — 2026-07-19

The preceding sections are the continuation checkpoint captured before the Session 43 final
control-plane regression. This append-only section is the current interpretation of this canonical
report; earlier numbers remain historical and are not silently rewritten.

### Latest local regression evidence

The detailed current report is:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-19_sc26_ae_final_regression.md
SHA256=dd9f9bef66a7755b269446aebd8ab79af2bfc158791eb0d072c5c1b6b8dba9a5
bytes=12617
```

The latest local checks recorded:

| Check | Result | Numeric evidence | Boundary |
|---|---|---:|---|
| Documentation contract | PASS | public entries=`9`; paper suggestions=`10` | local control-plane |
| Artifact/metrics/package/sealer pytest | PASS | `65 passed` in `3.46 s` | local synthetic |
| Task1/Task2/Task3 focused contracts | PASS | `2/2`, `4/4`, `3/3`; integration/provenance/portability suites GREEN | local synthetic |
| Fresh synthetic chain | PASS | chain=`1/1`; traces/memory=`4/4`; MSE=`3.0/0.5`; reload delta=`0.0`; rank0 step=`22.5 ms` | local synthetic |
| Prebaked CPU entries | PASS | models=`3/3`; rank0 step=`18.5/22.5/24.5 ms` | local synthetic |
| Clean-clone-style replay | PASS | public entries=`3/3/3`; setup=`6`; chain=`1`; clone statuses=`4` clean | local synthetic |
| Shell/Python/static checks | PASS | shell=`52`; Python=`35`; temp templates=`0`; `git diff --check`=`0` | local static |
| Grouped-gemm runtime import | BLOCKED | controller collection `ModuleNotFoundError: grouped_gemm` | worker prerequisite gap |

### Current release boundary

All local rows remain `local_synthetic_not_gpu_qualification` unless explicitly identified as a
setup-contract check. D45 semantic quota evidence remains `gpu : 129/128` (CLI exit `0`, semantic
`FAIL`), so `Gate B1=BLOCKED`, `real_pre_dataset=NOT QUALIFIED`,
`release_pre_dataset=NOT QUALIFIED`, and `AE-ready=NO`. No local result substitutes for the
authorized exact-two-H800 run or external issuer authentication.
## Session 43 Final Rerun Evidence — 2026-07-19

This is the final verification record for the documentation reconciliation and EOF repair. The
commands were run from the repository root with:

```bash
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/session43-doc-reconcile-final
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp/session43-doc-reconcile-final
```

### Reproducible evidence logs

- Full local regression:
  `logs/local-regression-20260719-session43-doc-reconcile-final.log`
  (`bytes=19,933`, `SHA256=477b98195103107252d4fa04d98899f21f1dba4180fd67ff619897545f0e696e`).
- Final documentation/static gate:
  `logs/final-doc-static-gate-20260719-session43-final.log`
  (`bytes=583`, `SHA256=45e20f70b3995986e52dc0a8a9f650e52b09db7eb2079a5ef4eb8e3f56c88415`).

### Final numeric results

| Check | Result | Observed value |
|---|---|---:|
| Documentation contract | PASS | entries=`9`; suggestions=`10` |
| Fixed-runtime/setup contracts | PASS | runtime=`21/21`; setup=`6/6`; grouped-gemm setup=`37/37` |
| Task1/Task2/Task3 focused/integration/portability | PASS | Task1=`11/11`; Task2=`3/3`; Task3=`9/9`, provenance=`10/10`, portability=`17/17` |
| Artifact/metrics/package/sealer pytest | PASS | `65 passed` in `5.25 s` |
| GPT mock integration | PASS | `22/22` |
| Fresh synthetic chain | PASS | traces/memory=`4/4`; rows=`2`; MSE=`3.0/0.5`; reload delta=`0.0`; rank0=`22.5 ms`; fwd/bwd/opt=`6.0/11.0/2.5 ms`; load/exec/wall=`0.125/0.375/0.5 s`; peak RSS=`51,416 KiB`; allocation=`32 MiB` |
| Prebaked CPU entries | PASS | models=`3/3`; rank0=`18.5/22.5/24.5 ms`; fwd/bwd/opt=`5.0/9.0/2.0`, `6.0/11.0/2.5`, `6.5/12.0/3.0 ms`; manifests=`22/18/18` |
| Clean-clone-style replay | PASS | public entries=`3/3/3`; setup=`6`; chain=`1`; clean statuses=`4` |
| Shell/Python/static | PASS | shell=`52`; Python=`35`; temp templates=`0`; diff check=`PASS` |

All local values remain `local_synthetic_not_gpu_qualification` or setup-contract evidence. The
controller `grouped_gemm` runtime collection gap and D45 semantic quota block remain unchanged.

## Session 43 I48 Transient Verifier Harness Failure — 2026-07-19

The first final-document verifier attempt is retained as a RED harness event. Its shell command
ended with:

```text
/bin/bash: -c: line 32: unexpected EOF while looking for matching `''
```

The root cause was an unmatched single quote in the verifier's final `printf` statement. The
failure occurred before the final marker and was not emitted by a repository test, fixture,
producer, consumer, or qualification workload. No product logic, threshold, source-selection,
provenance, quota, or evidence-class rule changed.

The corrected command uses the balanced form:

```bash
printf '%s\n' 'FINAL_DOC_VERIFICATION=PASS'
```

The corrected rerun is required before I48 is closed. Until that evidence is captured, the global
status remains `INCOMPLETE`, `Gate B1=BLOCKED`, `real_pre_dataset=NOT QUALIFIED`,
`release_pre_dataset=NOT QUALIFIED`, and `AE-ready=NO`.

## I48 corrected-rerun attempt 1: pipefail/no-match harness RED — 2026-07-19

The balanced verifier passed the documentation contract, current inventory/hash checks, syntax,
`git diff --check`, and the seven current-success marker checks. It then stopped at the expected
zero-match temporary-root scan with exit `1`. The cause was an unguarded `rg | wc -l` pipeline under
`set -o pipefail`: `rg` correctly returns `1` when no hard-coded template matches, but the verifier
treated that expected condition as a failure before printing its count.

Retained RED log:

```text
logs/final-doc-verification-20260719-session43-i48-corrected.log
bytes=550
SHA256=4819c49007f744a68f967099a91051afeb5bde2a99b0c2947877a01141345fa9
```

This is a verifier-only control-flow defect. The next command uses an explicit status-aware
conditional and does not change any product or release qualification boundary.

## I48 final local documentation/static closure — 2026-07-19

### Test Script Information

- Documentation contract: tests/unit/test_sc26_ae_docs_contract.sh.
- Status-aware verifier evidence:
  logs/final-doc-verification-20260719-session43-i48-status-aware-v2.log.
- Working directory:
  /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717.
- Environment:
  CONDA_DEFAULT_ENV unset, Python 3.12.3, GNU bash 5.2.21, ripgrep 15.1.0.
- Commands and scopes used:

      bash tests/unit/test_sc26_ae_docs_contract.sh
      find SC26-AE tests/unit tests/integration tests/e2e tools/ae -type f -name '*.sh'
      find SC26-AE/tools tests/unit tests/integration tests/e2e tests/performance tools/ae -type f -name '*.py'
      git diff --check

  pretrain_llama.py was included explicitly in the Python AST parse. The final temporary-template
  check used this exact status contract:

      if matches=$(rg -n 'mktemp -d /tmp|mktemp /tmp|mktemp -p /tmp|mktemp .* /tmp' SC26-AE tests tools/ae); then
        count=$(printf '%s
' "$matches" | wc -l)
      else
        rg_status=$?
        if [ "$rg_status" -eq 1 ]; then count=0; else exit "$rg_status"; fi
      fi
      test "$count" -eq 0

### Validation Criteria

1. Documentation contract passes with nine public entries and ten paper suggestions.
2. The established Session 43 static scopes contain exactly 52 shell files and 35 Python files,
   all parse successfully, and git diff check is clean.
3. The current summary inventory verifies 20 exact byte/hash rows and the current I48 baseline
   verifies seven non-self-referential document rows.
4. Seven current-success markers are verified, their manifest aliases equal the actual referenced
   manifest hashes, and mismatch count is zero.
5. The hard-coded temporary-template search reports zero; only rg status 1 is accepted as the
   no-match case; the verifier emits its final PASS marker and exits zero.

### Test Results and Evidence

| Check | Expected | Actual | Result |
|---|---:|---:|---|
| Public task entries | 9 | 9 | PASS |
| Paper suggestions | 10 | 10 | PASS |
| Shell files | 52 | 52 | PASS |
| Python files | 35 | 35 | PASS |
| Inventory hash rows | 20 | 20 | PASS |
| Final document hash rows | 7 | 7 | PASS |
| Current-success markers | 7 | 7 | PASS |
| Fresh/prebaked markers | 4/3 | 4/3 | PASS |
| Marker alias mismatch | 0 | 0 | PASS |
| Hard-coded temporary templates | 0 | 0 | PASS |
| Final verifier exit | 0 | 0 | PASS |

The final GREEN log is bytes=628 and
SHA256=32878844222ac152d41b770f5fae3a78c5dbe4883c681bf56006c80fe7be1786.
The preceding scope-enumeration RED is retained at
logs/final-doc-verification-20260719-session43-i48-status-aware.log, bytes=148,
SHA256=7f43991e021af9fdd006b40e7427cb8a7d92462a4ab2af869b49c01d35c778d8.
Its 69-versus-52 discrepancy came from a broadened verifier file scope, not a syntax error or
product regression.

### Evidence Boundary

I48 is CLOSED only for the local documentation/static harness. All values remain
local_synthetic_not_gpu_qualification. D45 semantic quota remains gpu : 129/128 with CLI exit 0
and semantic FAIL; Gate B1 is BLOCKED; real_pre_dataset and release_pre_dataset are NOT QUALIFIED;
AE-ready is NO.
