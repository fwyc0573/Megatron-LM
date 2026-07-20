# Test Report — Task3 Portability, Provenance, and Fresh-Chain Verification

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Recorded Task3 portability/provenance RED→GREEN evidence, cross-task marker verification, full affected regression, and local-evidence limitations |

## 1. Test Script Information

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Environment: host/controller shell, `CONDA_DEFAULT_ENV=<none>`, Python `3.12.3`
- Evidence class: `local_synthetic_not_gpu_qualification`
- Real GPU workloads executed by this lane: `0`
- Test scripts:
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task1_contracts.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task2_contract.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task3_contracts.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task3_provenance.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task3_contract.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task3_portability.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/e2e/test_sc26_ae_fresh_chain.sh`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_artifact_manifest.py`
  - `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_echo_metrics.py`

### Reproducible commands

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717

bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/unit/test_sc26_ae_task3_contracts.sh
bash tests/unit/test_sc26_ae_task3_provenance.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/integration/test_sc26_ae_task3_portability.sh
bash tests/e2e/test_sc26_ae_task1_smoke.sh
bash tests/e2e/test_sc26_ae_task2_smoke.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh

PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py
```

Captured aggregate log:

```text
/tmp/sc26-ae-task3-final-affected-regression-20260719.log
```

## 2. Validation Criteria

1. Task1 and shared Task2 markers are published only after manifest verification and contain JSON
   boolean `verified=true`.
2. Fresh Task3 rejects unverified Task1 or Task2 producers, checksum/provenance mismatch, and
   fresh/prebaked source mixing without fallback.
3. Task3 output cannot escape `AE_OUTPUT_ROOT` through parent, destination, or dangling symlinks;
   immutable run destinations cannot be reused.
4. Prebaked outer and nested manifest paths are relative, safe, and relocatable; absolute or
   traversal paths fail before simulation.
5. Prebaked manifests require the frozen model profile, `bf16`, and `ddp_overlap=true`; the shared
   predictor manifest explicitly declares `artifact_source=prebaked`.
6. Rank0 reports require an integer, non-boolean `rank_id=0` and exact forward, backward,
   optimizer, step-span, load, execution, and wall-clock fields.
7. The synthetic fresh chain and all three prebaked Task3 entries complete with verified manifests
   and markers while remaining labeled as non-GPU qualification evidence.
8. Assertions, checksum/provenance checks, no-fallback rules, and release-data-quality requirements
   remain unchanged.

## 3. RED Evidence and Root-Cause Resolution

| Defect | Observed RED | Root cause | Minimal repair |
|--------|--------------|------------|----------------|
| Task1 terminal marker state | Task1 contract exit `1`; `KeyError: 'verified'` | Manifest verification completed, but the capture marker omitted the explicit terminal state required by Task3 | Added `verified: true` to publication and asserted it in the Task1 contract |
| Unverified fresh Task1 accepted | A checksum-consistent marker with `verified=false` reached Task3 publication | Fresh resolver checked schema, identity, and digest but not terminal verification | Required Task1 marker `verified is true` before resolving artifacts |
| Shared Task2 terminal marker state | Fresh chain exit `1` with `[ERROR] Task2 shared predictor marker is not verified`; direct Task2 contract exit `1` with `KeyError: 'verified'` | Task3 correctly required the field, but `task2_write_shared_pointer()` omitted it | Added one `verified: true` field and a checksum-bound producer assertion |
| Output subtree symlink escape | `ESCAPED_MANIFEST_PRESENT=1`; `ESCAPED_MARKER_PRESENT=1` | Only the final marker pathname was checked; parent output components were not checked | Added component-by-component non-symlink, directory, and canonical-containment checks |
| Dangling destination symlink | An existing dangling run root passed an `! -e` check | Dangling symlinks have `-e=false` while still controlling the pathname | Immutable destinations now reject both existing entries and `-L` symlinks |
| Boolean rank identity | JSON `rank_id=false` passed validation | Python evaluates `False == 0` | Required an integer that is not `bool`, with exact value `0` |
| Prebaked source identity | A checksum-resealed shared manifest declaring `artifact_source=fresh` was accepted | Nested checksum verification did not enforce source-class semantics | Required prebaked shared manifest `artifact_source=prebaked` |
| Prebaked compute semantics | A checksum-resealed model manifest with `ddp_overlap=false` was accepted | Generic schema did not enforce the frozen AE profile, precision, and overlap mode | Required a frozen profile, `precision=bf16`, and `ddp_overlap=true` |

Relevant RED evidence:

```text
/tmp/sc26-ae-task3-marker-red.Ks9aEx
/tmp/sc26-ae-task3-output-red.CZYukA
/tmp/sc26-ae-task3-report-red.qgpXDu
/tmp/sc26-ae-task3-prebaked-source-red.qSK5Kd
/tmp/sc26-ae-task3-prebaked-overlap-red.Vkd8Pi
/tmp/sc26-ae-task3-portability-red.log
/tmp/sc26-ae-task2-shared-pointer-red-20260719.log
```

## 4. Test Results and Evidence

### Suite summary

| Test suite | Result | Actual result |
|------------|--------|---------------|
| Shell syntax | PASS | All affected Task1/Task2/Task3 libraries, entries, and tests parsed; exit `0` |
| Task1 contracts | PASS | `9/9`; exit `0` |
| Task2 integration contract | PASS | Shared build plus three model attachments; manifest files=`13`; exit `0` |
| Task3 unit contracts | PASS | `6/6`; exit `0` |
| Task3 provenance | PASS | Dirty, non-canonical, and gitlink-mismatch producers rejected; clean canonical producer accepted |
| Task3 integration contract | PASS | `6/6`; exit `0` |
| Task3 portability/provenance | PASS | `10/10`; exit `0` |
| Task1 public-entry smoke | PASS | Contract=`9/9`, smoke=`1/1`, real GPU workloads=`0` |
| Task2 public-entry smoke | PASS | All three entries share the verified synthetic predictor contract |
| Task3 prebaked CPU e2e | PASS | `3/3` models; exit `0` |
| Fresh Task1→Task2→Task3 e2e | PASS | `1/1` verified synthetic chain; exit `0` |
| Artifact-manifest and Echo-metrics pytest | PASS | `29 passed in 0.98 s`; exit `0` |

### Numeric prebaked Task3 results

| Model | Step ms | Forward ms | Backward ms | Optimizer ms | Simulator wall s | Process wall s | Peak RSS KiB | Manifest files |
|-------|--------:|-----------:|------------:|-------------:|-----------------:|---------------:|-------------:|---------------:|
| GPT-175B | `18.5` | `5.0` | `9.0` | `2.0` | `0.5` | `0.937961` | `51,308` | `22` |
| Qwen3-A30B | `22.5` | `6.0` | `11.0` | `2.5` | `0.5` | `0.946505` | `51,356` | `18` |
| DeepSeek-V3 | `24.5` | `6.5` | `12.0` | `3.0` | `0.5` | `0.942943` | `51,336` | `18` |

Every model used a tested host allocation of `32 MiB`; simulator load/execution were
`0.125/0.375 s`. These are synthetic fixture values that validate report plumbing and relative
scale, not model-performance measurements.

### Numeric fresh-chain results

| Metric | Actual |
|--------|-------:|
| Task1 trace files / memory JSON files | `4 / 4` |
| Task2 dataset rows | `2` |
| Task2 average validation MSE / test MSE | `3.0 / 0.5` |
| Task2 reload maximum absolute prediction delta | `0.0` |
| Task3 backward command UIDs | `4` |
| Task3 rank0 step / forward / backward / optimizer | `22.5 / 6.0 / 11.0 / 2.5 ms` |
| Task3 simulator load / execution / wall | `0.125 / 0.375 / 0.5 s` |
| Task3 process wall | `0.917871 s` |
| Task3 peak RSS | `51,336 KiB` (`0.048957825 GiB`) |

## 5. Evidence Boundary and Remaining Qualification

This report closes the identified local control-plane, schema, provenance, and portability test
gaps. It does not qualify a real GPU workload or release pre-dataset. The evidence remains:

```text
local_synthetic_not_gpu_qualification
```

Therefore `AE-ready=NO`, the real three-model × three-task pre-dataset is `NOT QUALIFIED`, and the
task remains `INCOMPLETE` until real source-bound captures, predictor assets, full-rank data,
portable distribution manifests, checksum/data-quality validation, and clean-clone replay pass.
