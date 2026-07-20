# Test Report: Task2 Canonical Output-Root Containment

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added primary-agent independent recheck identities for Task2 containment, affected e2e chains, and the full local matrix; I56 remains PARTIAL and all qualification boundaries remain unchanged |
| 2026-07-19 | Reconciled independent post-implementation WATCH findings and narrowed the RED claim to the directly observed model-marker branch |
| 2026-07-19 | Added RED/GREEN, affected-regression, static-validation, and harness-failure evidence for the narrow Task2 canonical-containment repair |

## Scope and disposition

This report covers only the test-exposed intermediate-symlink containment defect in the Task2
model-marker and shared-pointer reuse resolvers. It does **not** qualify a GPU run, promote an
evidence class, close I54/I55/I57/I58, or close full I56. The resulting disposition is:

```text
I56 PARTIAL — post-resolution canonical containment only
Overall disposition: INCOMPLETE
Gate B1: BLOCKED
real_pre_dataset: NOT QUALIFIED
release_pre_dataset: NOT QUALIFIED
AE-ready: NO
Evidence class: local_synthetic_not_gpu_qualification
```

## 1. Test Script Information

### Environment

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Python: `/usr/bin/python3` — `Python 3.12.3`
- pytest: `pytest 9.1.1`
- Conda environment: `none` (`ENV_CONDA_DEFAULT_ENV=none`)
- Temporary root: `/data/ycfeng/sc26-ae-test-tmp/`
- CUDA/GPU: not used; controller has no CUDA device
- RJob/Docker/publication: not used

### RED command (before production guard)

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/task2-containment-fixture-red-20260719 \
TMPDIR=/data/ycfeng/sc26-ae-test-tmp/task2-containment-fixture-red-20260719 \
bash tests/integration/test_sc26_ae_task2_contract.sh
```

The fixture first builds two complete synthetic runs outside the canonical output root, so their
predictor IDs, manifests, metrics, provenance, and checksums are internally consistent. It then
places lexical-safe intermediate symlinks under `OUT`.

This is a historical pre-fix command: it was run against the unmodified resolver before the
production guard was added. Evidence: `logs/task2-containment-fixture-red-20260719.log` (269 bytes,
SHA256 `a95d18fcf0102bacc5d39eb1e8d72ec3707a6a9c4d863652cff83553be207041`).

### Targeted GREEN command

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/task2-containment-green-20260719 \
TMPDIR=/data/ycfeng/sc26-ae-test-tmp/task2-containment-green-20260719 \
bash tests/integration/test_sc26_ae_task2_contract.sh
```

Evidence: `logs/task2-containment-green-20260719.log` (1,034 bytes,
SHA256 `03bbf12ca01a27dc97b1bd0f7fb722dca1e31c6e30f02c3c8bd2f72dac49c7e8`).

### Affected regression commands

```bash
# Task2 integration and public smoke
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/task2-affected-20260719/task2_integration \
TMPDIR=/data/ycfeng/sc26-ae-test-tmp/task2-affected-20260719/task2_integration \
bash tests/integration/test_sc26_ae_task2_contract.sh
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/task2-affected-20260719/task2_smoke \
TMPDIR=/data/ycfeng/sc26-ae-test-tmp/task2-affected-20260719/task2_smoke \
bash tests/e2e/test_sc26_ae_task2_smoke.sh

# Fresh chain
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/fresh-chain-containment-20260719 \
TMPDIR=/data/ycfeng/sc26-ae-test-tmp/fresh-chain-containment-20260719 \
bash tests/e2e/test_sc26_ae_fresh_chain.sh

# Isolated clean-clone replay
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/clean-clone-containment-20260719 \
TMPDIR=/data/ycfeng/sc26-ae-test-tmp/clean-clone-containment-20260719 \
bash tests/e2e/test_sc26_ae_clean_clone_replay.sh
```

### Full local regression command

The exact rerun was captured in
`logs/task2-full-regression-rerun-20260719.log` and consisted of:

```bash
TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/task2-full-regression-rerun-20260719
mkdir -p "$TMP_ROOT"
/usr/bin/python3 -m pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py \
  tests/unit/test_sc26_ae_package_prebaked.py \
  tests/unit/test_sc26_ae_seal_qualification.py

# Then, with a per-script writable SC26_AE_TMP_ROOT/TMPDIR:
for script in \
  tests/unit/test_sc26_ae_common.sh \
  tests/unit/test_sc26_ae_docs_contract.sh \
  tests/unit/test_sc26_ae_setup_runtime.sh \
  tests/unit/test_sc26_ae_task1_source_provenance.sh \
  tests/unit/test_sc26_ae_task2_evidence_mode.sh \
  tests/unit/test_sc26_ae_task2_interpreter_contract.sh \
  tests/unit/test_sc26_ae_task2_snapshot.sh \
  tests/unit/test_sc26_ae_task3_contracts.sh \
  tests/unit/test_sc26_ae_task3_interpreter_contract.sh \
  tests/unit/test_sc26_ae_task3_provenance.sh \
  tests/integration/test_sc26_ae_setup.sh \
  tests/integration/test_sc26_ae_task1_contracts.sh \
  tests/integration/test_sc26_ae_task2_contract.sh \
  tests/integration/test_sc26_ae_task3_contract.sh \
  tests/integration/test_sc26_ae_task3_portability.sh \
  tests/e2e/test_sc26_ae_clean_clone_replay.sh \
  tests/e2e/test_sc26_ae_fresh_chain.sh \
  tests/e2e/test_sc26_ae_task1_smoke.sh \
  tests/e2e/test_sc26_ae_task2_smoke.sh \
  tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh \
  tests/unit/test_setup_grouped_gemm_v1.sh \
  tests/integration/test_gpt_example_mock_mode.sh; do
  mkdir -p "$TMP_ROOT/$(basename "$script" .sh)"
  SC26_AE_TMP_ROOT="$TMP_ROOT/$(basename "$script" .sh)" \
  TMPDIR="$TMP_ROOT/$(basename "$script" .sh)" bash "$script"
done
```

### Static validation command

```bash
# bash -n for every shell file under SC26-AE, tests, and tools
while IFS= read -r path; do bash -n "$path"; done < <(
  find SC26-AE tests tools -type f -name '*.sh' | sort
)
# compile() for every Python file under SC26-AE, tests, and tools (no pyc writes)
/usr/bin/python3 - <<'PY'
from pathlib import Path
for root in (Path('SC26-AE'), Path('tests'), Path('tools')):
    for path in sorted(root.rglob('*.py')):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
PY
# production temp-root scan limited to SC26-AE/lib and SC26-AE/tools
if rg -n '/data/ycfeng/sc26-ae-test-tmp|/tmp/' SC26-AE/lib SC26-AE/tools; then
  exit 1
fi
git diff --check
```

Evidence: `logs/task2-static-validation-green-20260719.log` (215 bytes,
SHA256 `b792b960aff0da3214fa4d5ce06885ee2f58e97d0fc169d085246b7edc4caa53`).

### Post-review documentation/affected regression

After the append-only WATCH reconciliation, the affected Task2 integration, documentation
contract, and diff check were rerun with task-scoped writable temporary roots:

```bash
TASK_DIR=task_memory/task_2026-07-15_sc26_ae_workflow
TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp/task2-post-review-doc-regression-20260719
mkdir -p "$TMP_ROOT/task2-integration" "$TMP_ROOT/docs-contract"
SC26_AE_TMP_ROOT="$TMP_ROOT/task2-integration" \
TMPDIR="$TMP_ROOT/task2-integration" \
bash tests/integration/test_sc26_ae_task2_contract.sh
SC26_AE_TMP_ROOT="$TMP_ROOT/docs-contract" \
TMPDIR="$TMP_ROOT/docs-contract" \
bash tests/unit/test_sc26_ae_docs_contract.sh
git diff --check
```

Evidence: `logs/task2-post-review-doc-regression-20260719.log` (1,769 bytes,
SHA256 `fc75ac5114ff54c936ab35f27397036fd15ecce01a6016e498e7c37e77f0f445`). The command exited
`0`; Task2 integration passed, the documentation contract reported `PUBLIC_ENTRY_COUNT=9` and
`PAPER_SUGGESTION_COUNT=10`, and `git diff --check` passed. This was an affected docs/test rerun,
not a new GPU or full qualification run.

## 2. Validation Criteria

1. A lexical-safe intermediate symlink whose canonical target is outside `AE_OUTPUT_ROOT` must
   fail in both resolvers before any external bundle file is consumed.
2. The pre-fix test must fail because the old implementation accepts the escaped run; this proves
   the negative test is meaningful.
3. The post-fix test must reject both escapes and preserve a valid in-root three-model reuse flow.
4. Existing evidence classes, schemas, checksums, source selection, and no-fallback behavior must
   remain unchanged.
5. All affected local tests, full SC26-AE regression tests, syntax checks, and diff checks must pass.
6. Results must remain explicitly synthetic/controller evidence, never real qualification.

## 3. Test Results and Evidence

### RED/GREEN matrix

| Check | Expected | Actual | Result |
|---|---|---|---|
| Pre-fix model-marker symlink case | Old code accepts escape, test fails | Acceptance branch reached; `RC=1` | PASS (direct RED) |
| Post-fix model-marker symlink case | Reject with containment error | Rejected; marker not attached | PASS |
| Post-fix shared-pointer symlink case | Reject with containment error | Rejected; model marker absent | PASS |
| Valid synthetic Task2 reuse | Remains accepted | 3 model attachments accepted | PASS |
| Task2 integration + smoke | All cases pass | `2` suites, exit `0` | PASS |
| Post-review Task2/docs/diff rerun | Affected checks remain green | Task2 exit `0`; docs `9/10`; diff exit `0` | PASS |

### Full regression matrix

| Suite | Actual result | Exit |
|---|---:|---:|
| Python unit tests | `73 passed` | 0 |
| SC26-AE shell unit suites | all listed scripts pass | 0 |
| Task1 integration contract | `PASS_COUNT=31` | 0 |
| Task2 integration contract | containment + existing cases pass | 0 |
| Task3 integration contract | `PASS_COUNT=10` | 0 |
| Task3 portability | `PASS_COUNT=18` | 0 |
| Clean-clone replay | public entries `3/3/3`, setup `6`, chain `1`, repositories clean | 0 |
| Fresh chain | `CHAIN_PASS_COUNT=1` | 0 |
| Task1 smoke | `SMOKE_PASS_COUNT=1` | 0 |
| Task2 smoke | pass | 0 |
| Task3 prebaked CPU | `MODEL_PASS_COUNT=3` | 0 |
| Grouped-gemm setup contract | `37/37` | 0 |
| GPT mock integration | `22/22` | 0 |
| Static shell syntax | `73/73` | 0 |
| Static Python compile | `201/201` | 0 |
| Production temp-root scan | PASS | 0 |
| `git diff --check` | PASS | 0 |

### Key numeric workflow metrics

| Metric | Value |
|---|---:|
| Task1 trace files | 4 |
| Task1 memory JSON files | 4 |
| Task2 dataset rows | 2 |
| Task2 average validation MSE | 3.0 |
| Task2 test MSE | 0.5 |
| Task2 model reload max absolute prediction delta | 0.0 |
| Task3 rank0 step | 22.5 ms |
| Task3 forward/backward/optimizer | 6.0 / 11.0 / 2.5 ms |
| Task3 simulator wall | 0.5 s |
| Task3 peak RSS (fresh-chain rerun) | 51,684 KiB |
| Tested host allocation | 32 MiB |

These values are synthetic fixture outputs and are not GPU timing or qualification measurements.

### RED/GREEN evidence-scope note

The retained RED transcript directly observes the old model-marker acceptance branch and exits
before the shared-pointer case. The new GREEN transcript independently exercises both new guards
and the valid in-root path. The older Session 46 shared-pointer symlink probe
(`test_report_2026-07-19_session46_control_plane_probes.md`) is supporting pre-fix evidence of the
shared-pointer gap, but is not claimed as an independent RED transcript for this exact fixture.

### Post-implementation reviewer WATCH findings

The independent review accepted this narrow repair with `COMMENT / APPROVE WITH WATCH`. Remaining
boundaries are explicitly outside this report's closure claim:

1. `Path(...).resolve()` does not enforce a lexical non-symlink `AE_OUTPUT_ROOT`, exact
   `_shared/task2/runs/<predictor_run_id>` shape, or `run_path == run_relative_path`.
2. Ordinary pathname reads after containment still leave a check/use TOCTOU window; no immutable
   snapshot or descriptor-anchored traversal is implemented.
3. These findings keep `I56=PARTIAL / OPEN` and do not change the synthetic evidence class or any
   Gate B/release status.

## 4. Failure Diagnosis and Resolution

- **Fixture-only RED (resolved before production edit):** the initial copied external run retained
  `predictor_run_id=integration-one`, so the old resolver failed its earlier identity check instead
  of exercising containment. The fixture now builds complete external runs with IDs matching their
  symlink basenames and recomputed producer metadata.
- **Regression-wrapper failure (not product):** the first full rerun did not create the per-case
  `gpt-mock` parent, so `mktemp` failed. Adding `mkdir -p` before each case and rerunning the full
  matrix produced the authoritative exit-0 log.
- **Static-wrapper false positive (not product):** the first scan treated intentional negative
  fixture literals `/tmp/other-python` as production temp paths. Restricting the scan to production
  `SC26-AE/lib` and `SC26-AE/tools` produced the recorded GREEN result.

No fallback, source switching, evidence relabeling, schema extension, or qualification claim was
introduced.

## 5. Changed Files

- `SC26-AE/lib/task2_echo.sh` — canonical containment guards in both reuse resolvers;
  SHA256 `e75c78140d8eed49bb558c64440b5e632150916cdc1fb892fe3789d78473d5b3`.
- `tests/integration/test_sc26_ae_task2_contract.sh` — complete external-run symlink fixtures;
  SHA256 `4969740e39c8f1e00f549881fe6606651eed0d90c99f632f2299da8538522937`.

The independent design review artifact is
`.omx/artifacts/claude-you-are-an-independent-security-code-design-reviewer-in-repo-2026-07-19T15-17-15-855Z.md`,
SHA256 `0e407d25f545cd1c826236b4540bce292c2771c4152097306ae6dcd4af0f3694`.

Post-implementation review identity and conclusions are recorded in
`review.md` under “Session 46 post-implementation WATCH reconciliation”; no production code was
changed in response to that review.

## 6. Primary-agent independent recheck — 2026-07-19

This section records a fresh rerun by the integrating `/root` agent after the earlier lane report.
It is an independent verification of the current working tree, not a new production change and
not a replacement for the retained RED/GREEN history above.

### Commands and identities

All commands used a new controller-only temporary root under
`/data/ycfeng/sc26-ae-test-tmp/session46-*` and wrote new, non-overwriting logs:

| Check | Exit | Log bytes | SHA256 |
|---|---:|---:|---|
| Task2 integration contract | `0` | `1086` | `7b2bbea97c816110d050067a754033b915bf9f150f5eda84fcecaecb19eb718d` |
| Task2 public smoke | `0` | `1145` | `7dc98413038c9ab89b8f4656f5d0fd6eb2691cc5e6f7d8311e27f13be358c358` |
| Fresh Task1→Task2→Task3 chain | `0` | `789` | `3b230e159ea0c34d21f1bbf943dcf72e69c715e9e101edf1dbdcc33ef6ae815c` |
| Clean-clone-style replay | `0` | `1408` | `0e64fa6439a1380808304a91507affd477928c29a017c87c5335a354bffcabc0` |
| Full local matrix + static checks | `0` | `23476` | `a01816d3416045865c2476e3ddf6c2bb4c785068d72f2f69317a0ac69d680d8e` |

Environment was `/usr/bin/python3` `3.12.3`, `pytest 9.1.1`, no Conda environment, and no CUDA
device. No GPU, RJob, Docker, publication, commit, push, `rm`, or `mv` operation was used.

### Observed numeric results

- Task2 integration and smoke each rejected both intermediate-symlink escapes, the marker identity
  mismatch, the checksum-alias mismatch, and the unverified pointer; the valid path attached all
  three models.
- Fresh chain: trace/memory files=`4/4`, dataset rows=`2`, validation/test MSE=`3.0/0.5`, model
  reload maximum absolute prediction delta=`0.0`, rank0 step=`22.5 ms`, forward/backward/optimizer
  durations=`6.0/11.0/2.5 ms`, simulator wall=`0.5 s`, peak RSS=`51,336 KiB`.
- Full matrix: Python unit tests=`73 passed`; Task1=`31` cases; Task3 contract=`10`; Task3
  portability=`18`; clean-clone public entries=`3/3/3`, setup cases=`6`, chain=`1`; Task3
  prebaked models=`3/3`; grouped-gemm setup=`37/37`; GPT mock=`22/22`; shell syntax=`73/73`;
  Python compile=`201/201`; production temporary-root scan=`PASS`; `git diff --check=PASS`.

### Independent disposition

The rerun confirms only the narrow post-resolution canonical-containment behavior. It does not
establish a lexical no-symlink trusted root, exact canonical run identity, cross-file equality,
TOCTOU/frozen-input protection, qualified-evidence lifecycle, external issuer authentication, or
real GPU qualification. Therefore the disposition remains:

```text
I56 PARTIAL / OPEN
Evidence class: local_synthetic_not_gpu_qualification
INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```
