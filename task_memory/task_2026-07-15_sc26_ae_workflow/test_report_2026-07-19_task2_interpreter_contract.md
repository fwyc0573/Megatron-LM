# Test Report — Task2 v1.2-ae Interpreter Contract

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added the final verification-command RED→GREEN and fresh full-regression rerun |
| 2026-07-19 | Recorded the Task2 fixed-interpreter RED→GREEN repair and affected regression evidence |

## 1. Scope and Evidence Classification

This report validates the one-click Task2 control-plane binding only. It does not
run a GPU worker or RJob and does not qualify a real pre-dataset.

```text
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
REAL_PRE_DATASET_QUALIFIED=NO
AE_READY=NO
```

The v1.2-ae D42/D43 Retry-1 worker evidence records the Echo interpreter as:

```text
/opt/conda/envs/echo_slowdown/bin/python
```

The pre-fix runner instead defaulted to the unverified path
`/opt/conda/envs/echo_py310/bin/python`.

## 2. Root Cause and Repair

### Motivation

An AE reviewer invoking any public Task2 shell without an override would receive
the wrong absolute interpreter path, even though the current v1.2-ae image evidence
and all generated Echo `global_config.json` files bind the worker to
`/opt/conda/envs/echo_slowdown/bin/python`.

### Expectation

The default must equal the exact worker path. It must remain a direct fixed
binding: no filesystem search, environment discovery, alternate path, or runtime
fallback may be added.

### Observed RED

The deterministic unit contract was added before the production change and run
against the original assignment:

```text
Task2 default interpreter mismatch: expected=/opt/conda/envs/echo_slowdown/bin/python actual=/opt/conda/envs/echo_py310/bin/python
RED_EXIT_CODE=1
```

This is the expected behavioral failure, not a test syntax or environment error.

### Root Cause

The runner encoded an environment name (`echo_py310`) that was used for a
controller-side provisioning prefix, not the name actually installed and exercised
inside the v1.2-ae worker (`echo_slowdown`). Synthetic tests always set
`TASK2_PYTHON=python3`, so the incorrect real-mode default had no prior direct
coverage.

### Minimal Fix

Changed the one default assignment in `SC26-AE/lib/task2_echo.sh` to
`/opt/conda/envs/echo_slowdown/bin/python`. Added no discovery, source switch,
fallback, retry, calibration, or GPU behavior. Updated the evaluator README and
dependency inventory to distinguish the worker runtime from the controller-only
cp310 provisioning prefix.

### GREEN

```text
PASS: Task2 default interpreter=/opt/conda/envs/echo_slowdown/bin/python
interpreter_contract_EXIT_CODE=0
```

## 3. Test Script Information

### Environment

| Item | Actual value |
|------|--------------|
| Worktree | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717` |
| Branch | `sc26-ae-exec-clean-20260717` |
| Conda environment | unset |
| Test Python | `/usr/bin/python3`, Python `3.12.3` |
| GPU/RJob actions | `0` |

### Test Scripts

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_interpreter_contract.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_snapshot.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_echo_metrics.py`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task2_contract.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/e2e/test_sc26_ae_task2_smoke.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/e2e/test_sc26_ae_fresh_chain.sh`

### Exact Reproducible Commands

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717

bash -n \
  SC26-AE/lib/task2_echo.sh \
  SC26-AE/task2_gpt175b.sh \
  SC26-AE/task2_qwen3_a30b.sh \
  SC26-AE/task2_dsv3.sh \
  tests/unit/test_sc26_ae_task2_interpreter_contract.sh \
  tests/unit/test_sc26_ae_task2_snapshot.sh \
  tests/integration/test_sc26_ae_task2_contract.sh \
  tests/e2e/test_sc26_ae_task2_smoke.sh \
  tests/e2e/test_sc26_ae_fresh_chain.sh

bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
bash tests/unit/test_sc26_ae_task2_snapshot.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  tests/unit/test_sc26_ae_echo_metrics.py
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/e2e/test_sc26_ae_task2_smoke.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
```

## 4. Validation Criteria

1. The Task2 default evaluates to exactly
   `/opt/conda/envs/echo_slowdown/bin/python`.
2. The fixed-path test must first fail against the original default and then pass
   after the one-line production repair.
3. All Task2 snapshot branches remain covered: build, reuse, rebuild, dirty-source
   failure, and newly tracked excluded-path failure.
4. The three public Task2 entries continue to share one verified predictor identity.
5. Task2 metric extraction and save/reload validation remain unchanged.
6. The local Task1→Task2→Task3 chain remains semantically valid and explicitly
   labelled synthetic rather than GPU-qualified.

## 5. Test Results and Evidence

| Suite | Result | Actual numeric result | Exit code |
|-------|--------|-----------------------|-----------|
| Shell syntax | PASS | `9` shell paths checked | `0` |
| Fixed interpreter unit contract | PASS | expected path count=`1`; matched path count=`1` | `0` |
| Task2 snapshot unit | PASS | fresh predictor builds=`2`; verified manifest file count=`12`; negative branches=`2/2` | `0` |
| Echo metrics unit | PASS | `5/5` tests in `0.77 s` | `0` |
| Task2 integration | PASS | public model attachments=`3/3`; verified manifest file count=`13` | `0` |
| Task2 public-entry e2e | PASS | public entries=`3/3` | `0` |
| Fresh Task1→Task2→Task3 chain | PASS | chain count=`1/1` | `0` |

### Key Numeric Metrics

| Metric | Expected / acceptance | Actual | Delta / disposition |
|--------|-----------------------|--------|---------------------|
| Task2 dataset rows | positive | `2` | PASS |
| Average validation MSE | finite and internally consistent | `3.0` | PASS |
| Test MSE | finite and nonnegative | `0.5` | PASS |
| Model reload max absolute prediction delta | `0.0` | `0.0` | `0.0` |
| Task1 trace files in chain | `4` fixture files | `4` | `0` |
| Task1 memory JSON files in chain | `4` fixture files | `4` | `0` |
| Task3 backward `cmd_uid` count | positive | `4` | PASS |
| Task3 rank0 step | positive finite | `22.5 ms` | PASS |
| Task3 forward/backward/optimizer | positive finite | `6.0 / 11.0 / 2.5 ms` | PASS |
| Task3 simulator load/execution/wall | positive finite | `0.125 / 0.375 / 0.5 s` | PASS |
| Task3 measured process wall | positive finite | `0.894594 s` | PASS |
| Task3 peak RSS | within local fixture allocation | `51,232 KiB` (`0.048858643 GiB`) | informational |

### Final Verification-Command Failure and Resolution

The first final aggregate command completed all repository unit, integration, and
e2e suites, then failed in its inline documentation validator:

```text
File "<stdin>", line 1
  +from pathlib import Path
   ^^^^
SyntaxError: invalid syntax
```

The root cause was a test-command transcription error: patch-style leading `+`
characters were accidentally included in the inline Python heredoc. No repository
file or acceptance criterion caused the failure. The minimal correction removed
only those characters from the command. The complete affected suite was then run
again rather than reusing the earlier partial aggregate result. The corrected
final run returned:

```text
DOC_PATHS=4
DOC_TRAILING_WHITESPACE=0
DOC_ODD_FENCE_FILES=0
DOCUMENTATION_CONTRACT_EXIT=0
GIT_DIFF_CHECK_EXIT=0
```

The corrected aggregate command exit was `0`; its log is
`/tmp/sc26-task2-path-fix-final2-20260719.log`.

### Expected Negative-Branch Evidence

The snapshot unit deliberately observed and rejected both unsafe conditions:

```text
[ERROR] Echo-slowdown checkout is dirty: ?? dirty.txt
[ERROR] new tracked path under excluded output contract: merge/output/new.csv
```

Both negative branches preserved failure evidence and the suite exited `0` only
after confirming the failures occurred.

## 6. Files Covered by This Repair

- `SC26-AE/lib/task2_echo.sh`
- `tests/unit/test_sc26_ae_task2_interpreter_contract.sh`
- `SC26-AE/README.md`
- `task_memory/task_2026-07-15_sc26_ae_workflow/container_dependency_inventory.md`

## 7. Remaining Boundary

This repair removes the Task2 interpreter **control-plane mismatch**. It does not
close the dirty-source provenance WATCH, missing full-rank/model coverage, missing
atomic real Task1→Task2→Task3 chain, or final distribution/clean-clone gates.
Consequently, the real pre-dataset remains `NOT QUALIFIED` and the task remains
`INCOMPLETE`.
