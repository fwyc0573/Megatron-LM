# Test Report — SC'26 AE Setup Runtime Verifier

## Modification History

| Date | Summary of Changes |
|------|--------------------|
| 2026-07-19 | Added RED→GREEN evidence for the fixed-runtime setup verifier, installer-boundary regression, and affected AE contract suites |

## 1. Test Script Information

**Worktree**: \`/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717\`

**Environment**:

- Shell: \`/usr/bin/bash\` with \`set -euo pipefail\`
- Python: \`python3\` (CPython \`3.12.x\` controller-side test process)
- Test mode: deterministic local fixtures; no package installation, GPU allocation, RJob, or
  external publication
- Evidence class: \`local_synthetic_setup_contract\` and
  \`local_synthetic_setup_runtime_contract\`; downstream workflow checks remain
  \`local_synthetic_not_gpu_qualification\`

**Primary scripts**:

- \`SC26-AE/setup.sh\`
- \`tests/unit/test_sc26_ae_setup_runtime.sh\`
- \`tests/integration/test_sc26_ae_setup.sh\`
- \`tests/unit/test_setup_grouped_gemm_v1.sh\`

**Reproducible command** (run from the worktree above):

\`\`\`bash
set -o pipefail
bash -n SC26-AE/setup.sh
bash -n tests/unit/test_sc26_ae_setup_runtime.sh
bash -n tests/integration/test_sc26_ae_setup.sh
bash tests/integration/test_sc26_ae_setup.sh
bash tests/unit/test_sc26_ae_setup_runtime.sh
bash tests/unit/test_sc26_ae_common.sh
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/unit/test_setup_grouped_gemm_v1.sh
bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
python3 -m pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py
bash tests/unit/test_sc26_ae_task2_snapshot.sh
bash tests/unit/test_sc26_ae_task3_contracts.sh
bash tests/unit/test_sc26_ae_task3_provenance.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
git diff --check
\`\`\`

The complete captured log is \`/tmp/sc26_ae_setup_runtime_final_20260719.log\`.

## 2. Validation Criteria

The setup gate is accepted only when all of the following hold:

1. The setup entry has no shell syntax errors.
2. \`GROUPED_GEMM_SOURCE\` is explicit (\`vcs\` or \`archive\`); missing/invalid values fail before
   installer invocation.
3. The runtime verifier checks the fixed Megatron and Echo interpreters, exact Python/torch/CUDA
   contracts, CUDA/NVML visibility, required imports, pinned \`SlowdownPredictor\`, and fixed
   Nsight Systems/Compute paths, versions, and capabilities.
4. The grouped-gemm post-install check validates the import and backend file through the fixed
   Megatron interpreter.
5. A selected installer failure propagates unchanged and does not trigger source switching.
6. The success marker \`SC26_AE_SETUP_STATUS=verified\` is emitted only after all checks pass.
7. No test fixture may discover a host interpreter/tool through \`PATH\` or perform a real install.
8. Downstream Task1/Task2/Task3 contract regressions remain green without changing their
   acceptance thresholds or evidence class.

## 3. RED → Root Cause → Minimal Repair → GREEN

### RED observation

Before the test-harness repair, the affected regression was run as:

\`\`\`bash
bash tests/unit/test_setup_grouped_gemm_v1.sh
\`\`\`

Observed result:

\`\`\`text
36 setup/installer cases printed PASS
ERROR: Python executable is missing or not executable: /opt/conda/envs/megatron_env/bin/python
exit=1
\`\`\`

The failure occurred in the final setup-forwarding case, before the expected final summary line.

### Root cause

The production setup entry now deliberately verifies fixed runtime paths before invoking the
installer. The older unit fixture invoked \`SC26-AE/setup.sh\` as a child process and only supplied
fake installer environment variables; it did not isolate the fixed runtime verifier. Consequently
the test reached the real fixed interpreter path and then the real installer boundary instead of
testing source forwarding. This was a test-seam defect, not a production setup defect.

### Minimal repair

Only \`tests/unit/test_setup_grouped_gemm_v1.sh\` was changed:

- \`run_setup\` now sources the setup entry through \`/usr/bin/bash -c\`, stubs the two dedicated
  runtime/post-install verifier functions, and calls \`sc26_ae_setup_main\`, matching the existing
  integration test's isolation contract.
- The final setup-forwarding case adds an explicit \`/usr/bin/bash\` fake installer boundary (with
  an absolute shebang to avoid PATH recursion) and asserts that the selected \`archive\` source and
  fixed Megatron interpreter \`/opt/conda/envs/megatron_env/bin/python\` are forwarded.

No production assertion, threshold, source-selection rule, runtime constant, provenance check, or
fallback behavior was changed.

### GREEN observation

The repaired unit test now reports:

\`\`\`text
PASS: 37/37 grouped_gemm setup unit cases.
exit=0
\`\`\`

## 4. Test Results and Evidence

| Suite | Result | Numeric evidence | Exit |
|-------|--------|------------------|------|
| Shell syntax (\`setup.sh\` + 2 setup tests) | PASS | \`3/3\` commands | \`0\` |
| Setup integration | PASS | \`6/6\`; \`REAL_INSTALLER_EXECUTION_COUNT=0\` | \`0\` |
| Fixed runtime verifier unit | PASS | \`21/21\` cases | \`0\` |
| Common AE helpers | PASS | \`7/7\` | \`0\` |
| Task1 contracts | PASS | \`9/9\` | \`0\` |
| Task2 integration contract | PASS | verified manifest files=\`13\`; models attached=\`3\` | \`0\` |
| Task3 integration contract | PASS | \`6/6\` | \`0\` |
| Grouped-gemm installer unit | PASS | \`37/37\` | \`0\` |
| Task2 interpreter contract | PASS | expected/actual fixed interpreter=\`1/1\` | \`0\` |
| Artifact/metrics Python unit tests | PASS | \`27 passed\` in \`1.81 s\` | \`0\` |
| Task2 snapshot unit | PASS | snapshot/reuse/rebuild/dirty/new-path checks complete | \`0\` |
| Task3 contract unit | PASS | \`6/6\` | \`0\` |
| Task3 provenance unit | PASS | \`PROVENANCE_TEST_STATUS=PASS\` | \`0\` |
| Task3 prebaked CPU e2e | PASS | \`3/3\` models; \`MODEL_PASS_COUNT=3\` | \`0\` |
| Fresh Task1→Task2→Task3 chain | PASS | \`CHAIN_PASS_COUNT=1\` | \`0\` |
| Diff hygiene | PASS | \`git diff --check\` | \`0\` |

### Key synthetic chain metrics

These values are recorded for scale and schema validation only; they are not GPU qualification:

| Metric | Observed value |
|--------|----------------|
| Task1 trace files | \`4\` |
| Task1 memory JSON files | \`4\` |
| Task2 dataset rows | \`2\` |
| Task2 average validation MSE | \`3.0\` |
| Task2 test MSE | \`0.5\` |
| Task2 reload max absolute prediction delta | \`0.0\` |
| Task3 backward CMD UID count | \`4\` |
| Task3 rank0 step / forward / backward / optimizer | \`22.5 / 6.0 / 11.0 / 2.5 ms\` |
| Task3 load / execution / simulated wall | \`0.125 / 0.375 / 0.5 s\` |
| Task3 peak RSS | \`51,372 KiB\` (\`0.048992157 GiB\`) |

## 5. Scope and Remaining Gate

The setup control-plane/test block is now **\`CLOSED_LOCALLY_WITH_SYNTHETIC_CONTRACT_EVIDENCE\`**.
The nested \`megatron-sim-engine\` producer is clean at \`39755169f73f6c748e8d7376c3a2158c6569436b\`,
and the outer gitlink is closed at the same SHA in commit \`c217ce93156e7c37e065da2989c1a482f12ecebc\`.

This report does **not** close integrated Gate B1 or final AE qualification. The exact-two-H800
qualification and complete real \`3 models × 3 tasks\` Task1→Task2→Task3 pre-dataset remain unmet;
therefore \`real pre-dataset=NOT QUALIFIED\` and \`AE-ready=NO\` remain unchanged.

