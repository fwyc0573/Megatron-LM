# Test Report: Task2 Shared Predictor Verification Marker

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Added the post-repair full local regression evidence and log path |
| 2026-07-19 | Recorded the D30 RED→GREEN repair for the Task2 shared predictor verification marker and affected regressions |

## Test Script Information

- Worktree: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Focused integration test: `tests/integration/test_sc26_ae_task2_contract.sh`
- Fresh-chain e2e test: `tests/e2e/test_sc26_ae_fresh_chain.sh`
- Commands:

  ```bash
  bash tests/integration/test_sc26_ae_task2_contract.sh
  bash tests/e2e/test_sc26_ae_fresh_chain.sh
  ```

- Environment: host Bash, Python interpreter selected by the test fixture; no GPU or live RJob was
  used. Both runs are synthetic/controller evidence only.

## Validation Criteria

1. The shared Task2 pointer must contain `schema_version=sc26-ae-task2-shared-pointer-v1` and
   `verified=true`.
2. Task3's fresh-source resolver must accept the shared pointer only when the pointer is verified,
   checksum-consistent, and points to the expected Task2 predictor bundle.
3. The Task2 model attachment contract must remain intact for all three model keys.
4. The fresh Task1→Task2→Task3 chain must complete without source switching or fabricated
   artifacts.
5. Existing acceptance, provenance, checksum, data-quality, and evidence-class rules must remain
   unchanged.

## RED, Root Cause, and Minimal Repair

### Observed RED

After Task3 was hardened to fail fast on an unverified shared predictor marker, the fresh-chain
regression exited `1` with:

```text
[ERROR] Task2 shared predictor marker is not verified
```

The Task2 model-level marker already contained `verified=true`, but the shared pointer consumed by
Task3 did not contain that field.

### Root cause

`SC26-AE/lib/task2_echo.sh::task2_write_shared_pointer()` serialized the shared predictor identity
and checksums but omitted the semantic verification flag. The producer/consumer schema contract was
therefore asymmetric: Task3 correctly rejected the pointer, while Task2 incorrectly emitted one
that could not be consumed as a verified source.

### Minimal repair

Added only:

```python
"verified": True,
```

to the shared-pointer payload. No assertion, threshold, checksum, provenance rule, fallback, source
selection rule, or evidence label was changed.

## Test Results and Evidence

| Test | Result | Numeric evidence | Exit |
|------|--------|------------------|------|
| Task2 contract | PASS | model attachments=`3/3`; manifest files=`13`; verified pointer accepted=`1` | `0` |
| Fresh Task1→Task2→Task3 chain | PASS | chain pass count=`1`; Task1 trace files=`4`; Task1 memory JSON=`4`; Task2 rows=`2`; validation/test MSE=`3.0/0.5`; reload max abs delta=`0.0` | `0` |

Fresh-chain Task3 metrics were rank0 step=`22.5 ms`, forward/backward/optimizer=
`6.0/11.0/2.5 ms`, simulator load/execution/wall=`0.125/0.375/0.5 s`, process wall=
`1.106935 s`, and peak RSS=`51,292 KiB` (`0.048915863 GiB`). These values are scale and wiring
checks only, not GPU qualification or performance claims.

The resulting evidence class is:

```text
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
```

## Regression Boundary and Current Gate

The focused repair closes the local Task2 producer/Task3 consumer schema defect. It does not close
Echo exact-two-H800, integrated Gate B1, the complete real 3-model×3-task chain, portable
provenance/checksum/data-quality validation, or clean-clone nine-entry qualification. Therefore:

```text
task = INCOMPLETE
real pre-dataset = NOT QUALIFIED
AE-ready = NO
```

## Post-Repair Full Regression

**Date/Environment:** 2026-07-19 UTC, worktree
`/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`, host Python `3.12.3`; no GPU/RJob.

**Exact command:**

```bash
bash tests/unit/test_sc26_ae_setup_runtime.sh
bash tests/integration/test_sc26_ae_setup.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py \
  megatron-sim-engine/tests/unit/test_rank0_report.py \
  megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py \
  megatron-sim-engine/tests/integration/test_rank0_report_integration.py \
  megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py \
  megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py \
  megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/integration/test_sc26_ae_task3_portability.sh
bash tests/e2e/test_sc26_ae_task1_smoke.sh
bash tests/e2e/test_sc26_ae_task2_smoke.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
bash -n SC26-AE/*.sh SC26-AE/lib/*.sh
git diff --check
```

**Log:** `task_memory/task_2026-07-15_sc26_ae_workflow/logs/local-regression-20260719-d30-task2-pointer.log`

**Results:** all commands exited `0`. The Python suite reported `74 passed in 41.86 s`; setup
runtime=`21/21`; setup integration=`6/6`; Task1 contract=`9/9`; Task2 contract=`3/3` model
attachments with `13` manifest files; Task3 contract=`6/6`; Task3 portability=`10/10`; Task1
smoke=`1/1`; Task2 smoke=`1/1`; prebaked Task3=`3/3`; fresh-chain=`1/1`; shell syntax and
`git diff --check` both passed. Fresh-chain numeric evidence was Task2 validation/test MSE=`3.0/0.5`,
reload delta=`0.0`, Task3 rank0 step=`22.5 ms`, simulator wall=`0.5 s`, process wall=`1.093753 s`,
and peak RSS=`51,444 KiB`.

All results remain `local_synthetic_not_gpu_qualification`; no real GPU, quota, image, or release
gate is promoted by this regression.

## Documentation/Gate Validation

The D30 document validator `/tmp/sc26_ae_d30_gate_validator_20260719.py` was rerun after the
append-only evidence updates. It passed with required docs=`11/11`, `[Original Request]` tags=`47`,
D30 sync=`6/6`, Markdown fence lines=`112`, trailing-whitespace lines=`0`, and
`git diff --check` exit=`0`. The validator log is
`task_memory/task_2026-07-15_sc26_ae_workflow/logs/d30-task2-pointer-doc-validator.log`
(SHA256=`66b78e8b51252b7c5126e09b784ffd9cbc03a913caae81fa6c6948d83b6333a1`, bytes=`1,100`).

## Final Post-D45 Reconciliation Regression

After correcting the stale D26 quota wording, the complete local matrix was rerun with the same
fixed commands plus the stale-status guard. Log:
`task_memory/task_2026-07-15_sc26_ae_workflow/logs/local-regression-20260719-d30-final.log`
(SHA256=`ed2f22d4301d7612ffc3d006ae3cb842294da2a1b8d0e7b1bb82cd681f0f7962`, bytes=`9,215`).

Results were all exit `0`: stale-status guard=`PASS`; D30 docs=`11/11` and sync=`6/6`; fixed
runtime=`21/21`; setup=`6/6`; Python suite=`74 passed in 35.25 s`; Task1=`9/9`; Task2=`3/3`;
Task3=`6/6`; Task3 portability=`10/10`; Task1 smoke=`1/1`; Task2 smoke=`1/1`; prebaked Task3=
`3/3`; fresh chain=`1/1`; shell syntax and `git diff --check`=`PASS`. Final fresh-chain metrics:
Task2 validation/test MSE=`3.0/0.5`, reload delta=`0.0`, Task3 rank0 step=`22.5 ms`, simulator
wall=`0.5 s`, process wall=`0.999560 s`, and peak RSS=`51,452 KiB`.

This is still `local_synthetic_not_gpu_qualification`; the D45 external quota failure remains
unresolved and no real qualification or release status is promoted.

The final post-append D30 validator rerun remained GREEN with required docs=`11/11`, original
request tags=`47`, D30 sync=`6/6`, fence lines=`114`, trailing whitespace=`0`, and diff-check
exit=`0`. The final validator log SHA256 is
`5f1737a44a35b8834534227bf2c462f1edf0cc64514904f236038147a8cdafde` (bytes=`1,100`).
