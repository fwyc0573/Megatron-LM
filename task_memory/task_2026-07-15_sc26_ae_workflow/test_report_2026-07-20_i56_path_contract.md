# Test Report: I56 Task2 Reuse Path Contract

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Added focused RED/GREEN, validation-order failure diagnosis, minimal affected regression, pre/post-implementation independent reviews, and explicit non-qualification boundaries for the Task2 path-alias/canonical-shape repair |

## Scope and disposition

This report covers one local I56 subfinding in the two `task2_run_reuse` resolvers:

1. `run_path` and `run_relative_path` must both be non-empty strings and exactly equal; and
2. the stored path must have the canonical shape
   `_shared/task2/runs/<predictor_run_id>` before bundle files are opened.

The existing resolved-containment checks remain unchanged. The repair does not alter evidence
states, marker/pointer schemas, receipt binding, source provenance, qualification, or release
semantics. The resulting disposition is:

```text
I56 = PARTIAL / OPEN
Evidence class = local_synthetic_not_gpu_qualification
Overall workflow = INCOMPLETE
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

## 1. Test Script Information

### Environment

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Conda environment: `none`
- Python: `/usr/bin/python` — `Python 3.12.3`
- pytest: `pytest 9.1.1`
- Torch: `2.5.1+cu124`
- CUDA available: `False`
- CUDA device count: `0`
- GPU, RJob, Docker, network qualification, publication, push, `rm`, and `mv`: not used

### Modified production and test paths

- Production: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/lib/task2_echo.sh`
- Focused integration:
  `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task2_contract.sh`
- Affected source-provenance regression:
  `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_source_provenance.sh`
- Affected snapshot/reuse regression:
  `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task2_snapshot.sh`

### Reproducible commands

Focused Task2 integration:

```bash
TMP_ROOT=$(mktemp -d \
  /data/ycfeng/sc26-ae-test-tmp/i56-task2-path-contract.XXXXXX)
SC26_AE_TMP_ROOT="${TMP_ROOT}/runtime" \
TMPDIR="${TMP_ROOT}/runtime" \
bash tests/integration/test_sc26_ae_task2_contract.sh
```

Minimal affected regressions:

```bash
TMP_ROOT=$(mktemp -d \
  /data/ycfeng/sc26-ae-test-tmp/i56-task2-affected.XXXXXX)
mkdir -p "${TMP_ROOT}/source-provenance" "${TMP_ROOT}/snapshot"
SC26_AE_TMP_ROOT="${TMP_ROOT}/source-provenance" \
TMPDIR="${TMP_ROOT}/source-provenance" \
bash tests/unit/test_sc26_ae_task2_source_provenance.sh
SC26_AE_TMP_ROOT="${TMP_ROOT}/snapshot" \
TMPDIR="${TMP_ROOT}/snapshot" \
bash tests/unit/test_sc26_ae_task2_snapshot.sh
```

Focused static checks:

```bash
bash -n SC26-AE/lib/task2_echo.sh
bash -n tests/integration/test_sc26_ae_task2_contract.sh
git diff --check -- \
  SC26-AE/lib/task2_echo.sh \
  tests/integration/test_sc26_ae_task2_contract.sh
```

## 2. Validation Criteria

1. The pre-fix implementation must accept all four intentionally invalid path payloads so the new
   test produces a meaningful RED:
   - model marker with divergent aliases;
   - model marker with a noncanonical but in-root path;
   - shared pointer with divergent aliases; and
   - shared pointer with a noncanonical but in-root path.
2. The fixed implementation must reject all four before opening the selected bundle.
3. Existing intermediate-symlink containment negatives must remain green.
4. The existing model-marker `predictor_run_id` mismatch must retain its prior specific error,
   rather than being silently shadowed by the new shape check.
5. Valid shared-pointer reuse must still attach Qwen3-A30B and DSV3 to the same verified predictor.
6. The seven-file outer-source provenance contract and snapshot/reuse contract must remain green.
7. Shell syntax and focused whitespace checks must pass.
8. No local result may be promoted to H800 qualification, a real/release pre-dataset, or AE-ready.

## 3. Test Results and Evidence

### RED/GREEN matrix

| Check | Expected | Actual | Result |
|---|---:|---:|---|
| Pre-fix path-contract RED | Non-zero; four invalid payloads accepted | exit `1`; `unexpected_acceptances=4` | PASS (meaningful RED) |
| First post-fix integration | New four-case block passes and all prior cases remain green | exit `1`; new path block passed, but an older predictor-ID error was shadowed | FAIL, diagnosed |
| Corrected focused integration | All new and existing cases pass | exit `0`; new cases `4/4`; explicit `PASS:` lines `11`; `[PASS]` lines `1`; verified-manifest markers `4` | PASS |
| Outer-source provenance | Existing fixed producer bracket remains green | `PASS_COUNT=11`, exit `0` | PASS |
| Snapshot/reuse | Valid build/reuse/rebuild plus two intentional source negatives pass | exit `0`; verified-manifest markers `5`; manifest file count `12` | PASS |
| Shell syntax | Both modified shell paths parse | `2/2` | PASS |
| Focused `git diff --check` | No whitespace error | exit `0` | PASS |

### Numeric identities

| Evidence | Exit | Bytes | SHA256 |
|---|---:|---:|---|
| RED integration | `1` | `734` | `7dc04c48e52740a3e6bfda509e3a65adbf3e8d82bdfd2ac037872306256984a7` |
| First GREEN attempt | `1` | `783` | `44c1434d34c8406e2b3a16ea3bbce350af48565c18e3d0acf307514d8bed953b` |
| First GREEN predictor-mismatch detail | `1` | `320` | `a7acef03945698803fbe9082ce58b860707664908fd129471437930039ee1bf2` |
| Corrected GREEN integration | `0` | `1,510` | `2e93899a5dcb2dbde9fe73ec9479f5d4bb5b94aa37face3559d1bb5169c6962c` |
| Minimal affected regression | `0` | `1,532` | `3fa170bcb2107d224c6f4136c6264b6ed7243a407d909a01b58ce9e1805ac9a6` |

Evidence roots were intentionally retained outside the repository:

```text
/data/ycfeng/sc26-ae-test-tmp/session59-i56-path-contract-red-20260720.eyop5q
/data/ycfeng/sc26-ae-test-tmp/session59-i56-path-contract-green-20260720.DVY7lO
/data/ycfeng/sc26-ae-test-tmp/session59-i56-path-contract-green-v2-20260720.yWSAOS
/data/ycfeng/sc26-ae-test-tmp/session59-i56-affected-regression-20260720.Sx83lz
```

### Failure diagnosis and resolution

The first post-fix integration did not expose a product acceptance failure. The new marker shape
check ran before the existing comparison between `predictor_run_id` and the resolved run basename.
Consequently, the older predictor-ID negative was still rejected, but it reported
`existing Task2 marker path is not canonical for predictor_run_id` instead of its established
`existing Task2 marker predictor_run_id does not match run path` root cause.

The minimal correction moved only the marker exact-shape check after the existing resolved
containment and predictor-ID comparison, while keeping it before `artifact_manifest.json`,
`metrics.json`, or `provenance.json` is opened. The corrected rerun passed the new four-case path
contract and every pre-existing integration assertion. No assertion was weakened and no fallback
was added.

### Pre-implementation independent review

StepCode Claude (`claude-opus-4-6[1m]`, effort `max`) returned `APPROVE` for this path-only repair:

```text
.omx/artifacts/claude-act-as-an-independent-read-only-reviewer-for-the-current-sc2-2026-07-20T08-27-42-415Z.md
```

- Bytes: `11,542`
- SHA256: `2c92e19024f8a028ad201fb7ec9f87d801ee1477a337b1344fd6be63fe0e2529`
- Provider exit: `0`

The reviewer confirmed that alias/shape validation is independent of I54. It also confirmed that
rejecting a symlinked `AE_OUTPUT_ROOT` is a deployment-policy decision and is not required for this
repair.

### Post-implementation independent review

After the corrected GREEN and focused document verifier, StepCode Claude
(`claude-opus-4-6[1m]`, effort `max`) inspected the current resolver order, focused tests, report,
and I56 status sections directly. The overall verdict was `APPROVE`:

```text
.omx/artifacts/claude-act-as-an-independent-read-only-post-implementation-reviewer-2026-07-20T08-43-47-365Z.md
```

- Bytes: `9,361`
- SHA256: `e5e8998b7d6996662dcbd29a6ee39390868dc60d1e597d6a1ee71799f60003cb`
- Provider exit: `0`

The review confirmed that the marker resolver preserves the established predictor-ID diagnostic,
all path checks still precede bundle-file reads, no I54/I57 scope crossing occurred, no
evidence/schema/fallback/release boundary weakened, the four focused negatives are adequate, and
all current documents keep I56 `PARTIAL / OPEN`. Its single non-blocking `WATCH` records an
intentional ordering asymmetry: the shared-pointer shape check precedes resolved containment,
whereas the marker check follows its established resolved predictor-ID comparison. The pointer
resolver has no corresponding pre-existing Python diagnostic to preserve, and both branches remain
fail-closed before opening `artifact_manifest.json`; no code or test change is required.

## 4. Current file identities

| File | Bytes | SHA256 |
|---|---:|---|
| `SC26-AE/lib/task2_echo.sh` | `76,402` | `b1be6ea9bda248c9c11fe5a786a43fae83882f5dd47750d3b94e292eef93f5ca` |
| `tests/integration/test_sc26_ae_task2_contract.sh` | `31,577` | `9013ed44c25663bb516bb56ef44a62aa85e1c2704d51bfb8266200d681bcf678` |

## 5. Residual boundaries

I56 remains `PARTIAL / OPEN` after this repair:

1. Whether `AE_OUTPUT_ROOT` itself must be lexically non-symlinked remains an explicit deployment
   policy/design decision.
2. Ordinary pathname reads are not descriptor-anchored and retain a non-adversarial TOCTOU window.
3. Frozen-input/run-content immutability remains part of the I57 snapshot design.
4. Qualified evidence, receipt binding, and canonical post-seal publication remain I54 work and
   were deliberately not changed.
5. Canonical worker/image authority and cryptographic issuer authentication remain I55/I58 work.

These local tests do not change Gate B1, either pre-dataset, issuer governance, `AE-ready`, or the
overall `INCOMPLETE` state.
