# Test Report: SC'26 AE Control-Plane Final Regression (2026-07-19)

## Modification History

| Date | Summary of Changes |
|------|--------------------|
| 2026-07-19 | Added the post-closure I49 documentation verifier identity and seven-row hash/marker validation evidence |
| 2026-07-19 | Added Session 44 documentation-consistency RED→GREEN evidence, full local rerun, static counts, and explicit grouped-gemm controller prerequisite result |
| 2026-07-19 | Recorded the post-alias, provenance, interpreter-binding, evidence-boundary, package-copy, and clean-clone regression evidence. |

## Scope and status boundary

This report validates the local SC'26 AE control plane and synthetic fixtures after the
Task1/Task2/Task3 provenance and evidence-boundary repairs. It does **not** claim a real GPU
qualification, a release pre-dataset, or AE readiness.

Authoritative release status at report time:

```text
INCOMPLETE
Gate B1 = BLOCKED by D45 semantic quota failure
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

The external D45 result is semantic `gpu : 129/128` with CLI exit `0`; the semantic failure is
authoritative and is not overridden by the CLI status.

## 1. Test Script Information

### Environment

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Outer HEAD: `c217ce93156e7c37e065da2989c1a482f12ecebc`
- Nested `megatron-sim-engine` HEAD: `39755169f73f6c748e8d7376c3a2158c6569436b`
- Nested status lines: `0`
- Python: `Python 3.12.3`
- pytest: `pytest 9.1.1`
- Temporary root:
  `/data/ycfeng/sc26-ae-test-tmp` (`SC26_AE_TMP_ROOT` and `TMPDIR`)
- Controller prerequisites intentionally unavailable:
  `/opt/conda/envs/echo_slowdown/bin/python`,
  `/opt/conda/envs/megatron_env/bin/python`, and the `grouped_gemm` module.

### Exact commands

Focused control-plane regression:

```bash
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp
bash tests/unit/test_sc26_ae_task1_source_provenance.sh
bash tests/unit/test_sc26_ae_task2_evidence_mode.sh
bash tests/unit/test_sc26_ae_task3_interpreter_contract.sh
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/unit/test_sc26_ae_task3_contracts.sh
bash tests/unit/test_sc26_ae_task3_provenance.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/integration/test_sc26_ae_task3_portability.sh
pytest -q tests/unit/test_sc26_ae_package_prebaked.py
```

Public e2e and relocation regression:

```bash
bash tests/e2e/test_sc26_ae_fresh_chain.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
bash tests/e2e/test_sc26_ae_clean_clone_replay.sh
```

Full local control-plane matrix (the grouped-gemm runtime import is reported separately because
the controller has no installed `grouped_gemm` module):

```bash
bash tests/unit/test_sc26_ae_docs_contract.sh
bash tests/unit/test_sc26_ae_common.sh
bash tests/unit/test_sc26_ae_setup_runtime.sh
bash tests/unit/test_sc26_ae_task1_source_provenance.sh
bash tests/unit/test_sc26_ae_task2_evidence_mode.sh
bash tests/unit/test_sc26_ae_task2_interpreter_contract.sh
bash tests/unit/test_sc26_ae_task2_snapshot.sh
bash tests/unit/test_sc26_ae_task3_contracts.sh
bash tests/unit/test_sc26_ae_task3_interpreter_contract.sh
bash tests/unit/test_sc26_ae_task3_provenance.sh
pytest -q tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py \
  tests/unit/test_sc26_ae_package_prebaked.py \
  tests/unit/test_sc26_ae_seal_qualification.py
bash tests/integration/test_sc26_ae_setup.sh
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/integration/test_sc26_ae_task3_portability.sh
pytest -q tests/integration/test_grouped_gemm_v1_runtime.py
bash tests/unit/test_setup_grouped_gemm_v1.sh
bash tests/integration/test_gpt_example_mock_mode.sh
bash tests/e2e/test_sc26_ae_task1_smoke.sh
bash tests/e2e/test_sc26_ae_task2_smoke.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
bash tests/e2e/test_sc26_ae_clean_clone_replay.sh
```

Static validation:

```bash
find SC26-AE tests/e2e tests/integration tests/unit tools/ae -type f -name '*.sh' -print
for script in ...; do bash -n "$script"; done
python3 -m py_compile <all SC26-AE and AE-test Python files>
git diff --check
rg -n 'mktemp -d /tmp|mktemp /tmp|mktemp -p /tmp|mktemp .* /tmp' SC26-AE tests tools/ae
```

## 2. Validation Criteria

1. Every changed producer/consumer path must fail fast on wrong source, interpreter, provenance,
   evidence class, checksum alias, path containment, schedule shape/dtype, and stale output.
2. Task1/Task2/Task3 synthetic contracts must preserve explicit evidence labels and must never
   self-promote to a real qualification label.
3. Package verification must bind source bytes, destination bytes, and both checksum aliases.
4. All public synthetic entries must complete without fallback or stale-marker reuse.
5. Static shell/Python syntax and `git diff --check` must pass.
6. Any real-runtime or external-qualification claim must remain blocked when its prerequisite is
   absent.

## 3. Test Results and Evidence

| Suite | Result | Numeric evidence | Evidence class / limit |
|-------|--------|------------------|------------------------|
| Task1 source-provenance unit | PASS | `2/2` | local synthetic contract |
| Task2 evidence-mode unit | PASS | `4/4` | local synthetic contract |
| Task3 interpreter-binding unit | PASS | `3/3` | local synthetic contract |
| Task1 integration | PASS | `11/11` | local synthetic contract |
| Task2 integration | PASS | model attachments `3/3`, manifest files `13` | local synthetic contract |
| Task3 unit contracts | PASS | `9/9` | local synthetic contract |
| Task3 provenance | PASS | `10/10` | local synthetic contract |
| Task3 integration | PASS | `10/10` | local synthetic contract |
| Task3 portability | PASS | `17/17` | local synthetic contract |
| Artifact/metrics/package/sealer pytest | PASS | `65 passed` in `3.46 s` | local synthetic contract |
| Setup integration | PASS | `6/6`, installer executions `0` | local synthetic setup contract |
| Grouped-gemm setup unit | PASS | `37/37` | local synthetic setup contract |
| GPT example integration | PASS | `22/22` | local mock contract |
| Task1 public smoke | PASS | `1/1`, real workload count `0` | local synthetic, no GPU claim |
| Task2 public smoke | PASS | `1/1` | local synthetic, no GPU claim |
| Fresh Task1→Task2→Task3 chain | PASS | chain `1/1`; traces/memory `4/4`; rows `2`; MSE `3.0/0.5`; reload delta `0.0` | `local_synthetic_not_gpu_qualification` |
| Prebaked Task3 CPU e2e | PASS | models `3/3`; manifests `22/18/18` files | `local_synthetic_not_gpu_qualification` |
| Clean-clone-style replay | PASS | public entries `3/3/3`; setup `6`; chain `1`; clone statuses `4` clean | `local_synthetic_not_gpu_qualification` |
| Shell syntax | PASS | `52` scripts | local static validation |
| Python AST syntax | PASS | `35` files | local static validation |
| Temporary-root scan | PASS | hard-coded templates `0` | local static validation |
| `git diff --check` | PASS | exit `0` | local static validation |
| `tests/integration/test_grouped_gemm_v1_runtime.py` | BLOCKED | collection error: `ModuleNotFoundError: grouped_gemm` | controller dependency gap; not a qualification result |

### Numeric synthetic workflow evidence

The fresh chain produced:

| Metric | Value |
|--------|------:|
| Task1 trace files | `4` |
| Task1 memory JSON files | `4` |
| Task2 dataset rows | `2` |
| Task2 average validation MSE | `3.0` |
| Task2 test MSE | `0.5` |
| Task2 reload max absolute prediction delta | `0.0` |
| Task3 rank0 step | `22.5 ms` |
| Task3 forward/backward/optimizer | `6.0/11.0/2.5 ms` |
| Task3 simulator load/execution/wall | `0.125/0.375/0.5 s` |
| Task3 tested host allocation | `32 MiB` |

The prebaked CPU run recorded rank0 step times of GPT-175B=`18.5 ms`, Qwen3-A30B=`22.5 ms`, and
DeepSeek-V3=`24.5 ms`; forward/backward/optimizer values were respectively
`5.0/9.0/2.0 ms`, `6.0/11.0/2.5 ms`, and `6.5/12.0/3.0 ms`. These are synthetic fixture values,
not measured GPU performance.

## 4. RED Root Cause and GREEN Resolution

### Evidence/provenance/package RED (from the preceding audit checkpoint)

The preceding audit recorded four focused package failures before the verified-copy and strict
checksum-alias repair. The underlying defects were: a real Task3 interpreter override seam, Task1
working-tree bytes not bound to tracked HEAD blobs, a source-verification/copy TOCTOU window, and
the package helper accepting either checksum alias. A separate Task2 real-reuse check also exposed
that synthetic predictor evidence could cross the real reuse boundary.

### Minimal GREEN repairs

- Real Task3 now binds both metadata and simulator execution to the fixed executable regular file
  `/opt/conda/envs/megatron_env/bin/python`, rejecting overrides and symlinks.
- Real Task1 validates the load-bearing producer files against the pinned HEAD blobs before and
  after execution.
- Package copy verifies source size/SHA256 before and after copy and destination size/SHA256,
  failing on source mutation or destination mismatch.
- Marker consumers require both `manifest_sha256` and `artifact_manifest_sha256`, with both aliases
  equal to the actual manifest digest.
- Real Task2 reuse accepts only `real_exact_two_h800_qualified`; synthetic evidence is rejected at
  the consumer boundary.

The focused and full local suites above provide the GREEN evidence. No acceptance threshold,
source-selection rule, fallback, qualification label, or external resource state was weakened.

## 5. Review and Governance Evidence

- Independent code review recommendation: `REQUEST CHANGES`.
- The unresolved CRITICAL finding is external issuer authentication for qualification evidence;
  checksum/integrity validation does not prove issuer identity. Implementing a cryptographic issuer
  protocol, key allowlist, or external qualification service was not authorized and was not done.
- The HIGH/MEDIUM findings for interpreter binding, Task1 source provenance, verified package copy,
  Task2 evidence-mode separation, and checksum aliases are locally repaired and regression-tested.
- Independent architecture review lane was unavailable. The primary lane did not record its own
  inspection as independent architecture approval.

## 6. External blockers and stop condition

The controller lacks the fixed real interpreters, `grouped_gemm`, visible GPU output, and the
external exact-two-H800 quota. D45's authoritative semantic output remains `gpu : 129/128` with
CLI exit `0`. Therefore no real Task1/Task2/Task3 3×3 chain, release distribution, or AE-ready
claim can be made from this report.

### Pending tasks

1. Obtain authorized external exact-two-H800 qualification with a valid immutable image digest.
2. Run and seal the real three-model × three-task chain with complete provenance and data-quality
   evidence.
3. Resolve external issuer-authentication governance before promoting any `real_*_qualified`
   label.
4. Re-run the grouped-gemm runtime test only in the designated qualified environment.

### Newly discovered issues

1. The controller-side grouped-gemm runtime import is unavailable; the setup contract remains
   green, but runtime execution is not testable here.
2. Local synthetic tests remain non-qualifying by design.

### Recommended next steps

1. Do not submit another RJob or request quota from this controller; preserve D45 as the current
   semantic blocker.
2. Use the approved H800 worker/image path to collect the external sealing evidence.
3. After external evidence exists, rerun the exact commands in Section 1 and update the release
   gate only from the resulting artifacts.

## Evidence logs

- Focused regression: `logs/focused-regression-20260719-followup.log`
  (`SHA256=df0c8e93f584c70ec02a4fc97bdc04b6f5d9123e44c62f80a26128cc81b63b7e`).
- E2E regression: `logs/e2e-regression-20260719-followup.log`
  (`SHA256=ee930882b9a8b8868c11520f28df06dd3ef2b0c58a8b9ec5565c7ca87fa18a0e`).
- Static validation: `logs/static-validation-20260719-followup.log`
  (`SHA256=c6c12b0215498841b6e93f67b459e6995d2641b3e4af783e92c88784ce2e134d`).
- Full matrix before the known dependency gap: `logs/full-local-regression-20260719-followup.log`
  (`SHA256=86fa90fda33fe1ed3997f52e3d85d00edacde9b8e831e213b9a4683a32dc2d48`).
- Full matrix after the dependency-gap command: `logs/full-local-regression-20260719-followup-rest.log`
  (`SHA256=b5b85f5fc8b61062ec323be8a00b1e7054dff40b01487337950bdbe77a0cfff8`).
- Final documentation/static gate: `logs/final-doc-static-gate-20260719.log`
  (`SHA256=91d7b67da09bba943ffe0817ea6e5b057f9e73fa9f2d246b1322ba7c7df713fb`).

## 7. Session 44 documentation-consistency regression

### Test script information

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Environment: `Python 3.12.3`, `pytest 9.1.1`; local controller only. No GPU/RJob was started.
- Temporary root: `/data/ycfeng/sc26-ae-test-tmp/session44-doc-full` via both
  `SC26_AE_TMP_ROOT` and `TMPDIR`.
- Documentation consistency probe: inline Python check over `plan.md`/`future.md`.
- Targeted static gate: `tests/unit/test_sc26_ae_docs_contract.sh`, `bash -n` over the fixed 52-file
  shell scope, AST parsing over the fixed 35-file Python scope, hard-coded temporary-root scan,
  and `git diff --check`.
- Full local rerun: the focused contract/e2e commands listed in Section 1, plus the full local
  matrix excluding the separately reported unavailable grouped-gemm runtime collection test.
- Exact grouped-gemm prerequisite command:

  ```bash
  pytest -q tests/integration/test_grouped_gemm_v1_runtime.py
  ```

### Validation criteria

1. `plan.md` has zero adjacent duplicate non-empty lines.
2. `future.md` states that I39 is `CLOSED/RESOLVED` and limits future work to revalidation of new
   release bundles.
3. Documentation contract, shell syntax, Python syntax, temporary-root scan, and `git diff --check`
   pass with the established counts.
4. Existing local contracts and synthetic e2e flows remain green without fallback or evidence-class
   promotion.
5. The missing controller `grouped_gemm` dependency is reported explicitly rather than converted
   into a false runtime qualification pass.

### Test results and evidence

| Check | Result | Numeric evidence / boundary |
|---|---|---|
| Pre-repair consistency probe | RED (expected historical evidence) | duplicate line=`1835`; I39 correction=`False`; exit=`1` |
| First post-repair literal probe | RED (verifier-only) | exit=`1`; valid bold status was rejected by an over-literal predicate |
| Corrected semantic consistency probe | PASS | duplicate=`0`; I39 closed=`True`; revalidation=`True`; exit=`0` |
| Documentation contract | PASS | public entries=`9`; paper suggestions=`10` |
| Shell syntax | PASS | `52` files |
| Python AST syntax | PASS | `35` files |
| Hard-coded temporary-root scan | PASS | matches=`0` |
| `git diff --check` | PASS | exit=`0` |
| Focused local regression | PASS | fresh chain=`1/1`; prebaked=`3/3`; clean-clone entries=`3/3/3`; exit=`0` |
| Full local control-plane matrix | PASS | pytest=`65 passed in 3.10 s`; setup runtime=`21/21`; grouped-gemm setup=`37/37`; GPT mock=`22/22`; exit=`0` |
| Grouped-gemm runtime collection | BLOCKED | `ModuleNotFoundError: grouped_gemm`; collection exit=`2`; controller prerequisite only |

### Evidence logs

- Pre-repair observation: `logs/session44-doc-consistency-initial-red.log`, bytes=`142`,
  SHA256=`292fe3c2f84019e30d5599e6f001523b13b4fe72a3db6d3ba877db6291122605`.
- Literal-predicate RED: `logs/session44-doc-consistency-green.log`, bytes=`116`,
  SHA256=`b479a416b26c583928485cfb6204a6f2942b0a8df7e71ff75a57325200453e0`.
- Semantic GREEN: `logs/session44-doc-consistency-green-v2.log`, bytes=`129`,
  SHA256=`cbc5d3b76ad5b4bb3e123bf4a1dd9899b75e20b69dd3beac2197e0938e90f1cf`.
- Targeted static gate: `logs/session44-doc-static-targeted.log`, bytes=`399`,
  SHA256=`eb16478646931602075d6ef1f3b94e8bad0dfe26f0de13246a80cc85e6a971fd`.
- Final static gate: `logs/session44-doc-static-final.log`, bytes=`291`,
  SHA256=`2f00d89e3ae0d468c4e43378bd12d317ea04565f7ee9171b570b371bd68137e4`.
- Focused regression: `logs/session44-doc-focused-regression.log`, bytes=`9,085`,
  SHA256=`a18057e88199095871830e45af9c64a03aa46f89c906894ce5e536ea008a2641`.
- Full local regression: `logs/session44-doc-full-regression.log`, bytes=`17,147`,
  SHA256=`108e9bd41fa73d1032f78e605b05b613644cef2e846f1e1c2135c29f1e584275`.
- Runtime prerequisite probe: `logs/session44-grouped-gemm-runtime-probe.log`, bytes=`857`,
  SHA256=`ebf048b581e9a2be2d8cb3a2cfda5cd83a44a2fcf0449f0cd5c482a3577802ef`.

### Status boundary

This report closes only the local documentation/static item. It does not qualify an H800 run,
real pre-dataset, release pre-dataset, issuer identity, or AE readiness. D45 semantic quota remains
`gpu : 129/128`; Gate B1 remains `BLOCKED`; `real_pre_dataset` and `release_pre_dataset` remain
`NOT QUALIFIED`; `AE-ready` remains `NO`.

### Final post-closure document verifier

The reproducible final verifier reran the documentation contract, fixed shell/Python syntax scopes,
summary hash inventory, semantic I39/I49 status checks, current marker aliases, temporary-root
scan, and `git diff --check`. It selected the newest Session 44 synthetic roots under
`/data/ycfeng/sc26-ae-test-tmp/session44-doc-full` and exited `0`.

| Metric | Observed value |
|---|---:|
| Public entries / paper suggestions | `9 / 10` |
| Shell / Python files | `52 / 35` |
| Final document hash rows verified | `7` |
| Current successful markers | `7` (`fresh=4`, `prebaked=3`) |
| Marker alias mismatches | `0` |
| Hard-coded temporary templates | `0` |
| `git diff --check` | `PASS` |
| Final verifier exit | `0` |

Final log: `logs/final-doc-verification-20260719-session44-i49-final.log`, bytes=`659`,
SHA256=`97acda6299ac7ed3fddc13a521505d7a732ced564df425e2b2f7e3f191a9c3fe`.
The preceding verifier-only RED is retained at
`logs/final-doc-verification-20260719-session44-i49-attempt1.log`, bytes=`283`,
SHA256=`447c882b3b825e49b8bd7d753e8e223a845ce3fc5596a73fd38bdf4d8bb1e997`.
