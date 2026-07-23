# Test Report: SC'26 AE Artifact Consolidation

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Added the staged-archive completeness audit and documented byte-preserving evidence whitespace handling |
| 2026-07-23 | Completed the post-report consistency rerun and separated the process guard from syntax command arguments |
| 2026-07-23 | Added the compact-evidence inventory, checksum, documentation, and branch-scope validation results |

**Date:** 2026-07-23  
**Repository:** `/data/ycfeng/sc26_ae_task3_qwen`  
**Branch under test:** `sc26-ae-functional`  
**Archive base commit:** `3b1b51eec0162bd00b694c054dc9527016690c9a`

## 1. Test Script Information

### Scripts and files

- `SC26-AE/evidence/INDEX.md`
- `SC26-AE/evidence/index.json`
- `SC26-AE/evidence/checksums.sha256`
- `SC26-AE/README.md`
- `tests/unit/test_sc26_ae_docs_contract.sh`
- `tests/unit/test_sc26_ae_package_prebaked.py`
- `SC26-AE/tools/artifact_manifest.py`
- `SC26-AE/tools/echo_metrics.py`
- `SC26-AE/tools/normalize_ncu_metrics.py`
- `SC26-AE/tools/package_prebaked.py`

### Reproducible commands

Run from the repository root:

```bash
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/data/ycfeng/tmp/sc26_ae_artifact_consolidation_pycache

bash -n \
  SC26-AE/setup.sh SC26-AE/lib/common.sh \
  SC26-AE/lib/task1_trace.sh SC26-AE/lib/task2_echo.sh \
  SC26-AE/lib/task3_simulation.sh \
  SC26-AE/task1_gpt175b.sh SC26-AE/task1_qwen3_a30b.sh \
  SC26-AE/task2_gpt175b.sh SC26-AE/task2_qwen3_a30b.sh \
  SC26-AE/task3_gpt175b.sh SC26-AE/task3_qwen3_a30b.sh \
  examples/update_pretrain_gpt.sh \
  examples/qwen3_a3b_moe_scaling_wallclock_scan.sh

python3 -m py_compile \
  SC26-AE/tools/artifact_manifest.py \
  SC26-AE/tools/echo_metrics.py \
  SC26-AE/tools/normalize_ncu_metrics.py \
  SC26-AE/tools/package_prebaked.py

bash tests/unit/test_sc26_ae_docs_contract.sh
python3 -m pytest -q tests/unit/test_sc26_ae_package_prebaked.py

( cd SC26-AE/evidence && sha256sum -c checksums.sha256 )
# Runtime evidence intentionally preserves source log/CSV whitespace. Check the staged
# source/docs scope separately without rewriting those validated bytes.
git diff --cached --check -- ':!SC26-AE/evidence'
```

Run the Task2 process guard separately from the syntax command (so the parent command line cannot
contain the names being searched):

```bash
python3 - <<'PY'
from pathlib import Path
prefix = ''.join(('SC26-AE/task2_',))
suffixes = {''.join(('gpt175b', '.sh')), ''.join(('qwen3_a30b', '.sh')), ''.join(('dsv3', '.sh'))}
found=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        cmd=(entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (FileNotFoundError,PermissionError):
        continue
    if any(prefix+s in cmd for s in suffixes):
        found.append((entry.name,cmd))
assert not found, found
print('TASK2_REAL_PROCESS_COUNT=0')
PY
```

The inventory consistency check is the Python block in `SC26-AE/evidence/INDEX.md` section 6. It
also verifies every indexed file's byte count and SHA256, every JSON document, exact GPT/Qwen rank
vectors, and the two-GPU Task2 metadata. The Task2 process check reads `/proc/*/cmdline` and builds
the target suffixes at runtime so it cannot match its own command text.

### Environment

- Controller Python: `Python 3.12.3`
- `CONDA_DEFAULT_ENV`: unset; this report validates docs/archive files and does not execute a
  Megatron or Echo workload.
- All temporary, cache, and bytecode paths: `/data/ycfeng/tmp`
- `/tmp`: not used.
- No GPU, distributed run, Nsight capture, Task1 run, Task2 run, or Task3 run was started by this
  consolidation test.

## 2. Validation Criteria

1. The canonical branch is `sc26-ae-functional`, with no model-specific delivery branch.
2. The compact archive contains exactly 177 indexed files totaling 16,048,905 bytes.
3. All 177 checksums pass, and the only unindexed files are the three inventory metadata files
   (`INDEX.md`, `index.json`, and `checksums.sha256`).
4. GPT Task1 has 8 PP representatives and 8 memory files; Qwen Task1 has 32 PP×EP
   representatives and 32 memory files; each model's NCU vector is global rank 0 only.
5. The shared Task2 dataset has 727 rows and records exactly two physical GPUs. Consolidation runs
   zero Task2 commands.
6. All formal scripts pass shell syntax, all four packaging/metric tools compile, the README
   contract passes, and the existing functional package unit suite remains green.
7. `sc26-ad.tex` has zero changed paths, and no Task2 process is running.
8. Numeric Task3 report values in the indexed metrics are finite and nonnegative.
9. Every indexed artifact is present in the staged Git name set; preserved runtime evidence is
   checked by SHA256 rather than by a whitespace-normalizing diff.

## 3. Test Results and Evidence

| Test suite | Result | Evidence |
|------------|--------|----------|
| Shell syntax | PASS | `SHELL_SYNTAX_PASS_COUNT=13`, exit `0` |
| Python compilation | PASS | `PYTHON_COMPILE_PASS_COUNT=4`, exit `0` |
| README/docs contract | PASS | `DOC_CONTRACT_STATUS=PASS`, `PUBLIC_ENTRY_COUNT=9`, `PAPER_SUGGESTION_COUNT=10` |
| Evidence inventory | PASS | `ARCHIVED_FILE_COUNT=177`, `ARCHIVED_TOTAL_BYTES=16048905`, filesystem files `180` including 3 metadata files |
| Rank coverage | PASS | GPT traces/memory `8/8`; Qwen traces/memory `32/32`; NCU rank count per model `1` |
| Task2 provenance | PASS | dataset rows `727`; physical GPUs `2`; consolidation Task2 commands `0` |
| SHA256 verification | PASS | `CHECKSUM_PASS_COUNT=177` (`sha256sum -c` exit `0`) |
| Functional package unit suite (intermediate run) | PASS | `40 passed in 9.61 s` |
| Paper scope guard | PASS | `SC26_AD_TEX_CHANGED_PATH_COUNT=0` |
| Task2 process guard | PASS | `TASK2_REAL_PROCESS_COUNT=0`, `PROCESS_SCOPE_STATUS=PASS` |
| Staged source/docs whitespace guard | PASS | `git diff --cached --check -- ':!SC26-AE/evidence'` exit `0`; evidence bytes are checksum-checked |

The post-report rerun passed the document-existence assertion with
`INDEX_DOCUMENT_COUNT=5` and `INDEX_CONSISTENCY_STATUS=PASS`. The separate process guard returned
`TASK2_REAL_PROCESS_COUNT=0` and `PROCESS_SCOPE_STATUS=PASS`.

The final consolidation rerun superseded the intermediate package timing above and completed the
same suite in `8.93 s`; this is the current package-regression value recorded in
`progress.md`.

The pre-commit staged-scope audit found that normal Git staging omitted `68` compact evidence
files because historical ignore rules match `logs/`, `*.txt`, and `*.log`. Force-adding only the
curated evidence directory restored the promised archive without changing any bytes:
`INDEXED_ARTIFACTS_STAGED=177`, `EVIDENCE_FILES_STAGED=180`, and ignored evidence count `0`.

### Key Task2 metrics retained in the archive

| Metric | Actual |
|--------|--------:|
| Dataset rows | `727` |
| Physical GPUs | `2` (`CUDA_VISIBLE_DEVICES=0,1`) |
| Average validation MSE | `0.04124828706619175` |
| Test MSE | `0.061428837844613504` |
| Model reload max absolute prediction delta | `0.0` |

### Key Task3 values retained in the archive

| Model/path | Rank-0 step (ms) | Forward/backward/optimizer sums (ms) | Simulator load/execution/wall (s) |
|------------|-----------------:|--------------------------------------:|----------------------------------:|
| GPT Fresh | `8276.64` | `1905.12 / 99.24 / 53.61` | `14.201178 / 18.494841 / 32.696019` |
| GPT CPU prebaked | `8276.64` | `1905.12 / 99.24 / 53.61` | `18.015626 / 23.576013 / 41.591639` |
| Qwen Fresh | `3051.24` | `0.32 / 26.29 / 3.62` | `70.935776 / 760.9432 / 831.878976` |
| Qwen CPU prebaked | `3051.24` | `0.32 / 26.29 / 3.62` | `84.063936 / 1072.67156 / 1156.735496` |

These values validate archive/report wiring and finite numeric output only. They do not claim
distributed accuracy, release qualification, or paper fidelity.

## 4. Failure and RCA Records

### Documentation contract RED

The first docs-contract run failed with `README entry task1_dsv3.sh is missing`. The canonical base
README was missing 18 legacy contract anchors, not just the deferred entry. The root cause was a
stale contract versus an incomplete public README. The README was repaired with an explicit
deferred-only DeepSeek block, source/status/metric fields, and fail-fast wording; the unchanged
contract then returned `PASS`.

### Process guard false positive RED

The first full command counted two Task2 processes because its literal `ps | awk` pattern appeared
in its own command line. Raw process inspection found no Task2 process. The guard was replaced by a
runtime-constructed `/proc/*/cmdline` scan and returned `TASK2_REAL_PROCESS_COUNT=0`.

### Inventory document ordering RED

The first final consistency pass failed because `index.json` listed this report before the report
file had been created. Creating this report closed the dangling-document condition; the
post-report consistency rerun passed with `INDEX_DOCUMENT_COUNT=5`.

## 5. Scope and non-claims

- This is a compact-archive/documentation validation, not a new workload execution.
- Task2 was not rerun; the verified two-GPU predictor is reused unchanged.
- DeepSeek-V3 remains deferred and is not a formal evidence model.
- The 6.2 GiB functional distribution and multi-gigabyte raw captures remain external and
  manifest-anchored.
- `functional-fake-level-AE-ready=YES` is limited to fake-level workflow wiring;
  `release-ready=NO`, `distributed-accuracy-qualified=NO`, and `paper-fidelity-reproduced=NO`.
