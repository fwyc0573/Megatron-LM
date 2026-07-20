# Test Report — SC'26 AE Task1 Local Contracts

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-18 | Recorded final local Task1/shared unit, integration, syntax, and compile evidence after autonomous test-harness repair |

**Date:** 2026-07-18  
**Execution worktree:** `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`  
**Environment:** `CONDA_DEFAULT_ENV=none`; `/usr/bin/python3`; Python `3.12.3`  
**Scope:** local deterministic mocks and contract validation only; no real GPU, `rlaunch`, RJob, publication, submodule mutation, or real pre-dataset qualification was performed.

## 1. Test Script Information

### Scripts

- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task1_contracts.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_common.sh`
- `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_artifact_manifest.py`

### Reproducible commands

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717

PYTHONDONTWRITEBYTECODE=1 \
  bash tests/integration/test_sc26_ae_task1_contracts.sh

PYTHONDONTWRITEBYTECODE=1 \
  bash tests/unit/test_sc26_ae_common.sh

PYTHONDONTWRITEBYTECODE=1 \
  python3 -B -m pytest -q tests/unit/test_sc26_ae_artifact_manifest.py

bash -n \
  SC26-AE/lib/common.sh \
  SC26-AE/lib/task1_trace.sh \
  SC26-AE/task1_gpt175b.sh \
  SC26-AE/task1_qwen3_a30b.sh \
  SC26-AE/task1_dsv3.sh \
  tests/unit/test_sc26_ae_common.sh \
  tests/integration/test_sc26_ae_task1_contracts.sh

PYTHONDONTWRITEBYTECODE=1 python3 -B - <<'PY'
from pathlib import Path

source = Path("SC26-AE/tools/artifact_manifest.py")
compile(source.read_text(encoding="utf-8"), str(source), "exec")
print("PYTHON_COMPILE_FILES=1")
PY
```

## 2. Validation Criteria

1. The three public Task1 entries resolve the frozen full/QUICK rank sets.
2. Local mock runs create immutable model-specific run roots, verified manifests, memory JSON, trace files, and post-verification capture markers.
3. Final mock `torchrun` arguments contain bf16, mock data, overlap, memory tracing, kernel-ground-truth tracing, warmup `3`, and profile `1`; they omit fp16.
4. One Nsight Systems `profile` call wraps the complete selected-rank loop, and `export` produces a non-empty SQLite fixture.
5. Source failure propagates a nonzero exit and does not publish a capture marker.
6. Existing run destinations are rejected rather than reused.
7. Shared common-shell validation covers valid and invalid enums, positive integers, commands, file/directory paths, safe output paths, gitlink identity, clean submodules, and dirty-submodule rejection.
8. Artifact-manifest validation covers canonical paths, hashes, byte sizes, missing/extra files, symlinks/special files, duplicate paths, metadata invariants, and distribution-size boundaries.
9. Every modified shell file parses with `bash -n`, and `artifact_manifest.py` compiles without producing repository bytecode artifacts.

## 3. Test Results and Evidence

### Outcome summary

| Suite | Expected | Actual | Result |
|-------|---------:|-------:|--------|
| Task1 integration contract cases | 6 | 6 | PASS |
| Common-shell unit cases | 7 | 7 | PASS |
| Artifact-manifest pytest cases | 22 | 22 | PASS |
| Shell syntax files | 7 | 7 | PASS |
| Python compile files | 1 | 1 | PASS |
| Final command exit codes | 0 | 0 | PASS |

### Mock capture metrics

| Model | Selected ranks | Trace files | Memory JSON files | Expected peak allocated | Actual peak allocated | Delta | Manifest payload files | Manifest bytes | Nsight report / SQLite |
|-------|---------------:|------------:|------------------:|------------------------:|----------------------:|------:|-----------------------:|---------------:|-------------------------|
| GPT-175B | 8 | 8 | 8 | 90.0 MB | 90.0 MB | 0.0 MB | 26 | 6,912 | 0 / 0, not requested |
| Qwen3-A30B | 4 | 4 | 4 | 90.0 MB | 90.0 MB | 0.0 MB | 12 | 3,501 | 1 / 1 |
| DeepSeek-V3 | 4 | 4 | 4 | 90.0 MB | 90.0 MB | 0.0 MB | 10 | 3,169 | 0 / 0, not requested |

The three positive model runs executed `16` fake-rank invocations (`8 + 4 + 4`). The complete integration script ended with `21` fake `torchrun` invocations after including the deliberate source-failure case and the existing-destination negative case. Nsight used exactly `2` fake commands: one `profile` and one `export`.

### Failure diagnosis and resolution

#### Failure 1 — Nsight fake-log serialization mismatch

- **Observed:** the integration test expected `--trace=cuda,nvtx,osrt` in `nsys.log`.
- **Actual:** the fake logger intentionally serialized shell arguments with `printf '%q'`, producing `--trace=cuda\,nvtx\,osrt`.
- **Root cause:** the assertion compared an unescaped presentation against a shell-escaped audit representation; the production Task1 command was already correct.
- **Fix:** changed only the test assertion to match the logger's documented `%q` representation.
- **Result:** the Nsight case advanced and verified one profile call around the complete rank loop plus one SQLite export.

#### Failure 2 — stale terminal case count

- **Observed:** all printed cases passed, but the script ended with `PASS_COUNT=6` and `expected 7 cases`.
- **Root cause:** the script contained exactly six `pass` calls; the terminal assertion had drifted to seven without a seventh case.
- **Fix:** corrected the expected terminal count from `7` to `6`; no functional assertion or acceptance criterion was removed.
- **Result:** final Task1 integration exit code `0`, `PASS_COUNT=6`.

### Key log excerpts

```text
PASS: frozen full and QUICK rank scopes
PASS: three Task1 entries produce isolated verified mock bundles
PASS: adapter enforces bf16, mock data, overlap, trace memory, and warmup/profile
PASS: one Nsight invocation wraps the complete selected-rank source loop
PASS: source failure leaves the partial run unverified and publishes no marker
PASS: existing run destinations are never reused
PASS_COUNT=6
```

```text
PASS_COUNT=7
......................                                                   [100%]
22 passed in 0.93s
SHELL_SYNTAX_FILES=7
PYTHON_COMPILE_FILES=1
```

### Local log hashes

| Log | SHA256 |
|-----|--------|
| `/tmp/sc26_ae_task1_integration_20260718_rerun.log` | `4dce7c258a3dac1c06f27785736f859565125d56288b935e6a29992713adf310` |
| `/tmp/sc26_ae_common_20260718.log` | `9eef893114385ce4936c0ea3f164ee2ff02b1116ec47c02d113e3110484fc93a` |
| `/tmp/sc26_ae_manifest_20260718.log` | `3e904ddb1779028057e21a79cf879e7a240fa7611363320eba48dee7d6fb517d` |

## 4. Artifact Hygiene and Remaining Qualification Gap

- Repository-tracked `.pyc` files: `0`.
- Untracked, non-ignored `.pyc` files: `0`.
- One concurrent Task2 bytecode file existed at `SC26-AE/tools/__pycache__/echo_metrics.cpython-312.pyc`; `.gitignore:47` ignores all `__pycache__/` content, so it is excluded from Git/artifact inputs. This Task1 lane did not modify or delete it.
- These results prove local Task1/shared contracts only. Real H800 execution, real trace/memory/Nsight capture, pre-dataset completeness, runtime elapsed time, and release qualification remain pending and must not be inferred from mock metrics.
