# Test Report — I52 Task1 MoE Full-Rank Promotion Gate

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Recorded I52 RED→GREEN evidence, inventory edge coverage, package/sealer boundaries, Task1 scope propagation, and affected local regression |

## 1. Test Script Information

**Environment**

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Python: `3.12.3`
- PyTorch: `2.5.1+cu124`
- pytest: `9.1.1`
- `CUDA_AVAILABLE=False`, `CUDA_DEVICE_COUNT=0`; no NVIDIA driver is visible to this controller.
- Evidence class for every fixture/e2e run: `local_synthetic_not_gpu_qualification`.

**Changed production and test files**

| File | SHA256 |
|---|---|
| `SC26-AE/tools/artifact_manifest.py` | `6b781a15292bb18f2aa773b1fed9946f8516e64e9c9d63299bd35e07301ccb8f` |
| `SC26-AE/tools/package_prebaked.py` | `2407906f0e6596e6ecb11df46601abc101a7e06c9567ca2dfcd8762cf7645f57` |
| `SC26-AE/tools/seal_qualification.py` | `4cda1b4aeae2c0ffe88a0a9ba056afa465cc0ee88c55134200b0ffe2c0f94eb9` |
| `SC26-AE/lib/task1_trace.sh` | `3f163f5ded1f009eab41d56078429617d3c4f46f442c4127ce2f580103e9940c` |
| `tests/unit/test_sc26_ae_artifact_manifest.py` | `443e85d04d4031dca54ded71ed97377c1808bc3984c0e4dd51aad73f1ddccec9` |
| `tests/unit/test_sc26_ae_package_prebaked.py` | `ee3d4660edf7e398292d8291fc0b3629758ebabda290cf2ec65e316b411de416` |
| `tests/unit/test_sc26_ae_seal_qualification.py` | `1feb7ddd80db21074de1291223d84469bf31102aac079e5f5346b6ff5de007cb` |
| `tests/integration/test_sc26_ae_task1_contracts.sh` | `f42bd000ffc799be6a0faa6947695e5f7bd08948655bf768629a8b4a0ff2254f` |

**Reproducible commands**

```bash
python3 -m pytest -q \
  tests/unit/test_sc26_ae_package_prebaked.py \
  -k real_moe_task1_promotion_rejects_quick_rank_subset

python3 -m pytest -q tests/unit/test_sc26_ae_artifact_manifest.py \
  -k 'task1_moe_promotion or gpt_representative'

python3 -m pytest -q tests/unit/test_sc26_ae_package_prebaked.py \
  -k 'real_moe_task1_promotion'

python3 -m pytest -q tests/unit/test_sc26_ae_seal_qualification.py \
  -k 'moe and (quick or full)'

python3 -m pytest -q \
  tests/unit/test_sc26_ae_package_prebaked.py \
  tests/unit/test_sc26_ae_seal_qualification.py

python3 -m pytest -q tests/unit/test_sc26_ae_artifact_manifest.py
python3 -m pytest -q tests/unit/test_sc26_ae_*.py
python3 -m pytest -q tests/unit --ignore=tests/unit/test_backward_layer_count.py

bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
bash tests/e2e/test_sc26_ae_clean_clone_replay.sh

find SC26-AE tests -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python3 -m compileall -q SC26-AE/tools tests/unit tests/integration
git diff --check
```

## 2. Validation Criteria

1. A Qwen3/DSV3 Task1 manifest may use QUICK ranks for local smoke testing, but a pending external
   or real-qualified promotion must fail unless all of the following are true:
   - `simulation_topology.world_size == 256`;
   - `capture_summary.capture_scope == "full"`;
   - `selected_rank_ids == [0, 1, ..., 255]` in exact order, with no duplicates or booleans;
   - `selected_rank_count == 256`;
   - `trace_file_count == 256`;
   - `memory_json_count == 256`.
2. The same predicate must be used by the fresh package boundary and the external qualification
   sealer.
3. GPT-175B must retain its paper-defined representative ranks
   `[0,128,256,384,512,640,768,896]`; it must not be forced into a 1024-rank MoE rule.
4. Task1 metadata must preserve scope values: GPT=`representative`, Qwen3/DSV3 QUICK=`quick`,
   and Qwen3/DSV3 full=`full`.
5. Local tests must not be reported as GPU qualification. Global release status must remain:
   `INCOMPLETE`, Gate B1=`BLOCKED`, `real_pre_dataset=NOT QUALIFIED`,
   `release_pre_dataset=NOT QUALIFIED`, `AE-ready=NO`.

## 3. Test Results and Evidence

### 3.1 RED→GREEN and focused boundaries

| Check | Result | Numeric evidence / log |
|---|---|---|
| Original QUICK package regression (pre-repair) | **RED** | exit `1`; `2 failed, 19 deselected`; both `DID NOT RAISE`; `logs/i52-rank-gate-red-20260719.log` |
| Focused QUICK rejection after repair | **PASS** | `2 passed, 19 deselected`; exit `0`; log SHA256 `94a5c7689527cebfa8465c132d1eb147c71bb1c715ef1cb3f57e40319e899a70` |
| Artifact positive/negative/edge inventory | **PASS** | `18 passed, 24 deselected`; exit `0`; log SHA256 `013e54aa1425ad2f0601c7561bf2d45cbcb2a0017e11421b93c9663d4c201745` |
| Package positive + negative boundary | **PASS** | `4 passed, 19 deselected`; exit `0`; log SHA256 `a78d8eb26b4a27fcd398caf13f689084b281ea05f68768ce8da5f24012eb4b7d` |
| Sealer QUICK rejection + full control-plane acceptance | **PASS** | `4 passed, 25 deselected`; exit `0`; log SHA256 `13efb6610802acec463a9a34adb2d933e1fbb405fbb36f10d1ca3833d34755fe` |

The first sealer rerun used the invalid pytest expression `-k 'moe_(quick|full)'` and returned
exit `4` with no tests collected. The corrected command above returned `4 passed`; this was a
harness syntax error and did not alter production code.

### 3.2 Unit and integration regression

| Suite | Result | Evidence |
|---|---|---|
| Package + sealer full units | **PASS** | `52 passed in 4.45s`, exit `0`; log SHA256 `01d80656d3c2112a1dbfbb4dc98dd3e22a0175bc080601fff098d3861ab70f7b` |
| Artifact manifest full units | **PASS** | `37 passed in 0.88s`, exit `0`; log SHA256 `ee62b2493cb9e3a9b0e4a1308e42d69efb64b107175719ab5867b90bac1335ad` |
| All SC26-AE Python units | **PASS** | `99 passed in 5.58s`, exit `0`; log SHA256 `2fff6071a6dce0469bd953e51cdcb361f27c4d5240628365ff2f44dca753c3da` |
| All non-GPU unit files | **PASS** | `113 passed in 13.63s`, exit `0`; log SHA256 `bc7a9fb8fa5f9f4ff22b701c5f2e8eab50235453557df204d53472dd8cbb749d` |
| Task1 integration | **PASS** | `PASS_COUNT=31`; exit `0`; log SHA256 `3fd5fb369ec5a6c0cf2dd7d3facb4e3803220cf7b13e082a86973490b9a752c3` |
| Task2 integration | **PASS** | exit `0`; log SHA256 `2c397f2ec20289dd70a2dcdd63e692bb4f9167fe2263f688f21c7fbff212ae4` |
| Task3 integration | **PASS** | `PASS_COUNT=10`; exit `0`; log SHA256 `6e08702c3b070f96a7759b67fd5a22c2ea88292ddb74cdfa178f7d8029d6f9f3` |

Task1 scope assertions observed the following values:

| Model / mode | Scope | Selected ranks | Trace files | Memory JSON |
|---|---|---:|---:|---:|
| GPT-175B QUICK or full | `representative` | `8` | `8` | `8` |
| Qwen3-A30B QUICK | `quick` | `4` | `4` | `4` |
| DSV3 QUICK | `quick` | `4` | `4` | `4` |
| Qwen3/DSV3 full promotion fixture | `full` | `256` | `256` | `256` |

The full promotion fixtures used rank endpoints `0` and `255` and rejected missing, duplicate,
out-of-order, boolean, count-mismatch, wrong-topology, and QUICK inputs.

### 3.3 End-to-end and static checks

| Check | Result | Key numeric output / evidence |
|---|---|---|
| Fresh Task1→Task2→Task3 chain | **PASS** | exit `0`; traces/memory=`4/4`; validation/test MSE=`3.0/0.5`; reload delta=`0.0`; rank0 step=`22.5 ms`; forward/backward/optimizer=`6.0/11.0/2.5 ms`; log SHA256 `2c406a2bb40f74b60178b4942d1a133de9ccf843d1baafc3fc586d987ba563c7` |
| Clean-clone replay | **PASS** | exit `0`; public Task1/Task2/Task3=`3/3/3`; setup=`6`; fresh chain=`1`; log SHA256 `54bcaa9856679bc12870fb52734f461eb4221d416384f42060a63e4ae246b7a9` |
| SC26-AE shell matrix | **PASS** | exit `0`; all 20 listed shell/unit/integration/e2e cases passed; log SHA256 `d9793d2f57e9cfecb41b26a1ffb54417f19a149199930ebe15f28b8d8804d335` |
| Shell syntax / Python compile / diff check | **PASS** | exit `0`; log SHA256 `55050741105186b45dca44b203f551679fa2e38beb26e3c6597d92031aa7a07e` |

### 3.4 Environment-limited check

The intentionally broad command `python3 -m pytest -q tests/unit` returned exit `1` with
`113 passed, 6 failed`. All six failures are in the pre-existing CUDA-only
`tests/unit/test_backward_layer_count.py` and fail at CUDA initialization with
`RuntimeError: Found no NVIDIA driver on your system`. This is consistent with the measured
environment (`CUDA_AVAILABLE=False`, `CUDA_DEVICE_COUNT=0`). It is not an I52 regression and is
not treated as a qualification result; the complete non-GPU scope passed `113/113`.

## 4. Disposition

`I52 narrow local promotion gate: GREEN after regression.` The change closes only the identified
QUICK-to-real-promotion validation omission. It does **not** qualify Qwen3/DSV3 on H800 hardware,
does not establish producer snapshot provenance (I51), and does not close I54/I55/I56/I57/I58.

Global disposition remains:

```text
Overall disposition: INCOMPLETE
Gate B1: BLOCKED
real_pre_dataset: NOT QUALIFIED
release_pre_dataset: NOT QUALIFIED
AE-ready: NO
Evidence class: local_synthetic_not_gpu_qualification
```

## 5. Final-tree independent rerun — 2026-07-19/20

**Motivation:** The Task1 integration fixture and several I52 tests were edited after the first
report. This section records a fresh rerun from the stopped parallel-agent working tree; earlier
hashes in Section 1 remain historical and are not used for the V21 inventory.

**Commands and environment:** Python `3.12.3`, pytest `9.1.1`, controller `CUDA_AVAILABLE=False`,
`CUDA_DEVICE_COUNT=0`. Commands were run with `set -euo pipefail` where the expected result was
success; the intentionally broad unit probe was captured with its non-zero result so the six
CUDA-only failures remained visible.

| Command / scope | Result | Log bytes / SHA256 |
|---|---|---|
| `pytest -q tests/unit/test_sc26_ae_package_prebaked.py tests/unit/test_sc26_ae_seal_qualification.py` | `52 passed` | `247 / 013f8ac894e6ee0c989cbbe6cab74217ddf8c01f90c4e1b4ed57fec24110cf56` |
| `pytest -q tests/unit/test_sc26_ae_artifact_manifest.py` | `42 passed` | same log |
| `bash tests/integration/test_sc26_ae_task1_contracts.sh` | `PASS_COUNT=31` | `2176 / d79a1f0fc98e97b21626d44f39c265d88aa316a46f8fb42341e0712d0eac9060` |
| Task2 + Task3 integration | Task2 `0`, Task3 `PASS_COUNT=10` | combined log `session47-post-agent-task2-task3-e2e-20260720.log`, SHA256 `f388f1f7346c629403ca2454a44f5ca2a3c45f25ff4849cfb7ee39b27029e615` |
| Fresh Task1→Task2→Task3 chain | chain=`1`, traces/memory=`4/4`, MSE=`3.0/0.5`, reload delta=`0.0`, rank0=`22.5 ms`, F/B/O=`6.0/11.0/2.5 ms`, simulator wall=`0.5 s`, peak RSS=`51,640 KiB` | `719 / 89dae5799b9d275f56b7af9b2f3618a0e2c23c4f6ae0a9500ad5d246e1cbe878` |
| Clean-clone replay | public=`3/3/3`, setup=`6`, chain=`1`, statuses clean | `1339 / a0c4d24e6e687b28d0d29d2255ae8386f5f70ed3552557cbf9e11b7ad3a5deef` |
| `pytest -q tests/unit/test_sc26_ae_*.py` | `99 passed` | `197 / 9384686630dc735f6be4352f2359afe3d22a194299a34bb91da67744860b41b5` |
| `pytest -q tests/unit --ignore=tests/unit/test_backward_layer_count.py` | `113 passed` | `198 / 59ec80e40e3243874e61c282e0f919143c7a7a24a63a30c73543b0fa44020eae` |
| Broad `pytest -q tests/unit` | `113 passed, 6 failed` (all no-driver CUDA initialization) | `21721 / 822d20aaf8e667201ffd0ca42db9884fa20477161b42214f3d401fc7c7692e72` |
| Established AE static scope | shell=`52/52`, Python=`35/35`, temp-root=`PASS`, diff=`PASS` | `c6b210eb8e1f8d1850ce95496ad08081212ab558775e76bc1b5405cbbaa75e34` |
| `bash tests/unit/test_sc26_ae_docs_contract.sh` | public=`9`, paper=`10` | `91 / 6a1f69b8a0c281ab8fdc62ae8e99c177dab0588542a15b74e4a4bc6134537210` |

The exploratory broad shell scan found 17 legacy example files with literal `<Specify path>`
placeholders. They were not edited and are outside the established AE syntax scope; claiming
`73/73` for that unfiltered set would be false. The current I52 result remains a controller-only
local validation and does not qualify GPU, grouped-gemm, or release evidence.
