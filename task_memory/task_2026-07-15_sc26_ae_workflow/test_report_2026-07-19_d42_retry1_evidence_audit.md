# Test Report: D42/D43 Retry-1 Evidence Audit

## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-19 | Added an independent read-only audit of the sealed D42/D43 Retry-1 evidence, D28 budget disposition, source-provenance limits, and final pre-dataset boundary |

## 1. Scope and Executive Verdict

This report audits the sealed evidence root:

```text
/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_hub_i_basemind_mg_echo_gpu_qualification_retry1_20260717T144123Z
```

The audit is read-only. It did not submit a new `rlaunch` job, modify the sealed root, change any
gate, overwrite any prior report, or modify product/test code.

| Question | Verdict | Precise disposition |
|---|---|---|
| Did the D42 Retry-1 live attempt execute? | **PASS** | Yes. One exact-two-H800 live RJob completed with controller/live/worker exits `0/0/0`. That Retry-1 identity is used, terminal, and must not be reused or resubmitted. |
| Did it satisfy the exact resource gate? | **PASS** | Predict/live commands normalize identically; two distinct H800 UUIDs were observed under the immutable enterprise image digest. |
| Did it satisfy a strict clean-source/clean-commit gate? | **WATCH / PARTIAL** | The image and role-bound environments were not mutated during qualification, and the executed snapshot bytes are sealed by `final_inventory.json`. However, the Megatron controller tree had `13` dirty paths, no dirty diff was bound to the declared commit, and the Echo tar has no producer commit in `qualification_result.json`. Commit-equivalent clean-source provenance is therefore not established by this root alone. |
| Was the old D28 clean-retry budget consumed? | **NO — UNCONSUMED, SUPERSEDED, AND NO LONGER NEEDED** | The root has no D28 attribution or budget-consumption record. D33 explicitly states that a passing equivalent-or-stricter D29/D42 path leaves the old D28 budget unconsumed and unnecessary. It is not a free live slot; any future submission would require new explicit authority. |
| Can the root support the narrow D42/D43 canonical-image functional qualification? | **PASS WITH WATCH** | It supports the already-recorded narrow qualification: Qwen rank-0 Scaling smoke plus the complete standalone Echo slowdown pipeline on the exact enterprise image. The source-provenance and wording limitations in this report remain attached. |
| Can it be promoted to final AE-ready or complete three-model-by-three-task pre-dataset qualification? | **BLOCK** | No complete Task1→Task2→Task3 chain exists for any model, and the required three-model artifacts, atomic profiler alignment, portable packaging, nine-entry execution matrix, and final data-quality qualification are absent. |

The central distinction is:

1. **D42 Retry-1 identity:** consumed and terminal;
2. **old D28 replacement budget:** not consumed, but retired by the D33/D43 supersession path and
   unavailable for an implicit retry;
3. **D42/D43 functional image qualification:** supportable with the listed WATCH limitations;
4. **final AE pre-dataset qualification:** not achieved.

## 2. Governing Requirement and Attribution

The authoritative D33 requirement states:

> [Original Request] 用户在 grilling 中选择 `d29_supersedes_d28_for_b1`：若 D29 clean qualification 满足与 D28 同等或更严格的证据闭包并经独立审计通过，则 D29 替代旧镜像 D28 路径并关闭 integrated B1；旧 D28 replacement budget 保持未消费并转为不再需要，不再重复运行旧镜像 qualification。

Source:

```text
/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/requirements.md:194-195
```

The sealed root itself contains none of the following fields:

- `d28_budget_id`;
- `authority_decision`;
- `budget_disposition`;
- an explicit statement that the live attempt was charged to D28.

Its declared scope is instead:

```text
D42 enterprise-image exact-two-H800 clean qualification: Qwen3 MoE Scaling Mode smoke plus complete Echo slowdown pipeline
```

Therefore:

- **Evidence from the root:** one D42 Retry-1 live attempt ran and completed;
- **Unknown from the root alone:** a machine-verifiable D28 budget charge;
- **Disposition from D33 plus the canonical D43 task record:** D28 remained unconsumed, was
  superseded, and became unnecessary;
- **Safety consequence:** neither the used D42 identity nor the retired D28 budget authorizes
  another live job.

## 3. Test Script Information

### 3.1 Environment

| Item | Observed value |
|---|---|
| Audit host Python | `/usr/bin/python`, Python `3.12.3` |
| Active conda environment | unset |
| Audit worktree | `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717` |
| Evidence root | `/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_hub_i_basemind_mg_echo_gpu_qualification_retry1_20260717T144123Z` |
| GPU/runtime action during this audit | none |

The historical worker environments recorded in `qualification_result.json` were:

| Role | Interpreter | Python | Torch / CUDA |
|---|---|---:|---|
| Megatron/Qwen | `/opt/conda/envs/megatron_env/bin/python` | `3.9.18` | torch `2.1.2`, CUDA `12.1` |
| Echo slowdown | `/opt/conda/envs/echo_slowdown/bin/python` | `3.10.20` | torch `2.1.2+cu121`, CUDA `12.1` |

### 3.2 Exact Reproducible Commands

Key-file hashes:

```bash
ROOT=/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_hub_i_basemind_mg_echo_gpu_qualification_retry1_20260717T144123Z
sha256sum \
  "$ROOT/qualification_result.json" \
  "$ROOT/final_inventory.json" \
  "$ROOT/input/controller_git_status.txt" \
  "$ROOT/worker/qwen/trace_inventory.txt" \
  "$ROOT/worker/qwen/trace_validation.txt" \
  "$ROOT/worker/echo/output_inventory.txt" \
  "$ROOT/worker/echo/output_validation.txt" \
  "$ROOT/input/echo_source.tar" \
  "$ROOT/input/echo_source.tar.sha256" \
  "$ROOT/input/echo_source_verify.log"
```

Full self-excluding inventory verification:

```bash
python - <<'PY'
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

root = Path("/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_hub_i_basemind_mg_echo_gpu_qualification_retry1_20260717T144123Z")
inventory = json.loads((root / "final_inventory.json").read_text())
entries = inventory["entries"]
seen = set()
duplicates = []
unsafe = []
missing = []
size_mismatches = []
hash_mismatches = []

for row in entries:
    relative = row["path"]
    pure = PurePosixPath(relative)
    if relative in seen:
        duplicates.append(relative)
    seen.add(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        unsafe.append(relative)
    path = root / relative
    if not path.is_file():
        missing.append(relative)
        continue
    if path.stat().st_size != row["bytes"]:
        size_mismatches.append(relative)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["sha256"]:
        hash_mismatches.append(relative)

actual = []
symlinks = []
special = []
for directory, child_directories, files in os.walk(root, followlinks=False):
    for name in list(child_directories) + list(files):
        path = Path(directory) / name
        mode = os.lstat(path).st_mode
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(mode):
            symlinks.append(relative)
        elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            special.append(relative)
    for name in files:
        relative = (Path(directory) / name).relative_to(root).as_posix()
        if relative != "final_inventory.json":
            actual.append(relative)

unexpected = sorted(set(actual) - seen)
listed_but_absent = sorted(seen - set(actual))
assert len(entries) == inventory["listed_file_count"] == 2184
assert sum(row["bytes"] for row in entries) == inventory["listed_total_bytes"] == 419329007
assert not any((duplicates, unsafe, missing, size_mismatches, hash_mismatches,
                unexpected, listed_but_absent, symlinks, special))
print("PASS_FULL_SHA256_INVENTORY_AUDIT")
PY
```

Predict/live command equivalence:

```bash
python - <<'PY'
import shlex
from pathlib import Path

root = Path("/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_hub_i_basemind_mg_echo_gpu_qualification_retry1_20260717T144123Z")
predict = shlex.split((root / "controller/predict_command.sh").read_text())
live = shlex.split((root / "controller/live_command.sh").read_text())

def normalize(argv):
    return [value for value in argv
            if value != "--predict-only" and not value.startswith("--name=")]

assert normalize(predict) == normalize(live)
print("PASS_PREDICT_LIVE_COMMAND_EQUIVALENCE")
PY
```

Echo output binding:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("/data/ycfeng/Megatron-LM-sc26-ae/task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_hub_i_basemind_mg_echo_gpu_qualification_retry1_20260717T144123Z")
result = json.loads((root / "qualification_result.json").read_text())
outputs = result["echo_slowdown"]["outputs"]
assert len(outputs) == 10
assert sum(row["bytes"] for row in outputs) == 1133084
for row in outputs:
    path = root / row["path"]
    assert path.stat().st_size == row["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
print("PASS_ECHO_BOUND_OUTPUT_AUDIT")
PY
```

## 4. Validation Criteria

1. The root-wide inventory must contain no duplicate, unsafe, missing, extra, size-mismatched,
   hash-mismatched, symlink, or special-file entry.
2. Predict and live commands must preserve the same image, resources, mount, workdir, and payload
   after removing only `--predict-only` and the intentionally different RJob name.
3. The worker must observe exactly two H800 devices with two unique UUIDs and complete with exit
   code `0`.
4. Qwen qualification evidence must be interpreted according to its declared rank and memory mode;
   replay files must not be mislabeled as independent rank traces.
5. Echo output paths declared in `qualification_result.json` must exist and match declared sizes and
   SHA256 values.
6. Clean-image qualification must not be silently broadened into clean-commit provenance.
7. D28/D42 attempt accounting must distinguish an executed D42 live identity from the separate old
   D28 replacement budget.
8. A final AE/pre-dataset claim requires complete, atomically linked Task1→Task2→Task3 evidence for
   GPT-175B, Qwen3-A30B, and DeepSeek-V3, plus the release and reproducibility gates.

## 5. Test Results and Evidence

### 5.1 Sealed Inventory Integrity

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| Listed entries | `2,184` | `2,184` | PASS |
| Listed bytes | `419,329,007` | `419,329,007` | PASS |
| Inventory file bytes | recorded separately | `507,588` | PASS |
| Total files including inventory | `2,185` | `2,185` | PASS |
| Total bytes including inventory | `419,836,595` | `419,836,595` | PASS |
| Total directories including root | recorded | `327` | evidence |
| Duplicate paths | `0` | `0` | PASS |
| Unsafe paths | `0` | `0` | PASS |
| Missing files | `0` | `0` | PASS |
| Size mismatches | `0` | `0` | PASS |
| SHA256 mismatches | `0` | `0` | PASS |
| Unexpected files | `0` | `0` | PASS |
| Symlinks / special files | `0 / 0` | `0 / 0` | PASS |

The inventory is self-excluding: it lists every regular file except `final_inventory.json` itself.
This proves post-seal byte integrity of the archived snapshot. It does not by itself prove that the
snapshot equals a clean Git commit.

### 5.2 Key Artifact Hashes

| Artifact | SHA256 |
|---|---|
| `qualification_result.json` | `5624cafc972a615776cc8411edfe69ebb7555c436e405fc9c4ea841e65a102d0` |
| `final_inventory.json` | `762134c26031ff2c1fa8dbcd07a84d0dd4263db7952fdf378e10d0d7929af513` |
| `input/controller_git_status.txt` | `2c01497ea78dfdefb39495bb8b6df76fcea4b41fd08945bbb303e79e143a310c` |
| `worker/qwen/trace_inventory.txt` | `7ae00e17c0d723fe5594471f2f3ca3c46e33b67ba645126c61a9af098bdb7b7b` |
| `worker/qwen/trace_validation.txt` | `dcd553337d09bc113f3ff995ab70bd43709646ff13dbe2ef2718bc58038f4cc0` |
| `worker/echo/output_inventory.txt` | `acc8a020d95d0b700e2534bd4c024ccbebb0158364d504106beb80665ba6420a` |
| `worker/echo/output_validation.txt` | `01128c03f496a3e509b4928b84503a699453d2b5f5996c3ce192f50fd4940e22` |
| `input/echo_source.tar` | `2a9e7c48fa450831714fdd55a7082125eebe871c4ad492fada9e3e82ac16311c` |

### 5.3 Exact Resource and Controller Contract

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| Predict process exit | `0` | `0` | PASS |
| Predict elapsed | recorded | `1 s` | evidence |
| H800 candidates | at least one exact-two capable node | `10` | PASS |
| Live process exit | `0` | `0` | PASS |
| Worker exit | `0` | `0` | PASS |
| Live elapsed | recorded | `1,338 s` | evidence |
| Requested GPUs | `2` | `2` | PASS |
| Visible H800 GPUs | `2` | `2` | PASS |
| Unique GPU UUIDs | `2` | `2` | PASS |
| Memory per GPU | positive H800 capacity | `81,559 MiB` | PASS |
| Post-Qwen GPU health exit | `0` | `0` | PASS |
| Predict argv tokens before normalization | recorded | `18` | evidence |
| Live argv tokens before normalization | recorded | `17` | evidence |
| Normalized argv tokens | equal | `16 / 16` | PASS |

The resource contract was:

```text
--gpu=2
--cpu=16
--memory=65536
--charged-group=codesign
--private-machine=group
--positive-tags=h800
--backoff-limit=1
```

The immutable image was:

```text
hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4
```

### 5.4 Qwen Qualification Evidence

| Field | Actual |
|---|---:|
| Profile | `smoke` |
| Mode | `scaling` |
| Fake world size | `8` |
| Executed fake rank order | `[0]` |
| Physical GPU index | `0` |
| Topology | `PP=4, TP=1, EP=2, DP=2` |
| Hidden size / layers / experts | `1,024 / 12 / 32` |
| Warmup / profile iterations | `3 / 1` |
| Train iterations | `2` |
| Runtime exit / elapsed | `0 / 27 s` |
| Current qualified trace files | `1` |
| Current replay activation files | `3` |
| Current trace bytes | `4,248` |
| Current memory JSON | `0` (`TRACE_MEMORY=0`) |
| Current Qwen-correlated SQLite / `.ncu-rep` / `.nsys-rep` | `0 / 0 / 0` |

| Operation | Count | Duration |
|---|---:|---:|
| `forward_step` | `1` | `11.95 ms` |
| `backward_step` | `1` | `7.93 ms` |
| `optimizer_step` | `1` | `2.99 ms` |

`qualification_result.json` reports `new_trace_file_count=4`. The authoritative
`worker/qwen/trace_inventory.txt` shows that those four new files are:

- one real trace text file of `4,248` bytes;
- three replay activation `.pt` files of `525,622` bytes each.

Therefore `new_trace_file_count=4` must not be described as four rank traces. The source snapshot
also contains historical profiler and memory artifacts, but they are not current Retry-1 outputs and
must not be used to fill the current Qwen memory/SQLite/NCU gaps.

### 5.5 Echo Qualification Evidence

| Metric | Actual | Result |
|---|---:|---|
| `update_configs` exit / elapsed | `0 / 2 s` | PASS |
| `run_all` exit / elapsed | `0 / 991 s` | PASS |
| Merged rows | `727` | evidence |
| Feature shape | `[727, 8]` | PASS |
| Target shape | `[727]` | PASS |
| Average validation MSE | `0.0031091272501499075` | evidence |
| Test MSE | `0.0033649328512874955` | evidence |
| Model reload prediction match | `true` | PASS |
| Bound output files | `10` | PASS |
| Bound output bytes | `1,133,084` | PASS |
| Missing / size / SHA256 mismatches | `0 / 0 / 0` | PASS |

| Slowdown metric | Original | Clipped |
|---|---:|---:|
| MAE | `1.1174428876295175` | `1.1163005969562176` |
| MSE | `15.038093600726505` | `15.010386688475997` |
| RMSE | `3.8778980905545346` | `3.874324029876179` |

The Echo snapshot contains five `.sqlite`, one `.ncu-rep`, and two `.nsys-rep` files. These are
Echo workload assets, not an atomic Qwen Task1 capture linked by current trace identity or `cmd_uid`.
They cannot satisfy the missing Qwen-aligned profiler evidence.

### 5.6 Source and Provenance Boundary

#### Megatron

`qualification_result.json` records:

```text
megatron_commit=3c91d15bc035d49216161c9cac874f2453b69cb9
```

`input/controller_git_status.txt` records `13` dirty paths:

```text
 M megatron/profiler/trace_memory.py
 M task_memory/env_handbook.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/container_dependency_inventory.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/issues.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/notes.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/plan.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/progress.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/requirements.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/review.md
 M task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-17_gate_b1_live_qualification.md
 M tests/unit_tests/profiler/test_interception_comm_scaling_mode.py
 M tests/unit_tests/profiler/test_scaling_replay_cache_paths.py
?? tests/unit_tests/profiler/test_trace_memory.py
```

The complete executed snapshot is preserved and hash-sealed by `final_inventory.json`, which is
valuable byte-level evidence. However:

- the dirty diff is not stored or hash-bound as a distinct provenance object;
- `qualification_result.json` does not declare that the runtime tree equals the recorded commit;
- source-file hashes are not summarized into an explicit producer-source manifest;
- the only modified production path shown is `megatron/profiler/trace_memory.py`; current Qwen ran
  with `TRACE_MEMORY=0`, but the root does not establish a clean-commit equivalence claim.

This is why the correct wording is **sealed executed snapshot with dirty-source disclosure**, not
**clean commit checkout**.

#### Echo

The Echo source tar is independently bound by:

```text
bytes=1,822,720
sha256=2a9e7c48fa450831714fdd55a7082125eebe871c4ad492fada9e3e82ac16311c
```

The root does not record an Echo producer commit in `qualification_result.json`. In addition,
`input/echo_source.tar.sha256` and `input/echo_source_verify.log` contain the historical path:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/sc26_d29_image_v1_2_qualification_20260717T091427Z/input/echo_source.tar
```

The tar bytes remain verifiable, but the text records are not relocation-safe and do not themselves
prove tar-to-commit identity.

### 5.7 Final Three-Model-by-Three-Task Pre-Dataset Gap

This root does not contain a complete pre-dataset workflow cell for any model. The Qwen smoke and
Echo pipeline are separate qualification branches, not one atomically linked Task1→Task2→Task3
chain. A numeric claim such as “one of nine cells complete” would therefore be misleading.

Missing final-qualification evidence includes:

1. GPT-175B Task1, Task2, and Task3;
2. DeepSeek-V3 Task1, Task2, and Task3;
3. the complete target Qwen fake-rank set rather than rank `0` only;
4. current Qwen Task1 memory JSON;
5. current Qwen-aligned Nsight SQLite;
6. current Qwen-aligned NCU metrics and `.ncu-rep`;
7. trace-specific forward/backward `cmd_uid` blueprint binding;
8. Task2 slowdown assets built from the same Task1 input chain;
9. Task3 scheduler, simulator, and rank-0 report outputs;
10. portable outer manifests with relative paths and full SHA256/size verification;
11. producer/consumer commit compatibility evidence;
12. distribution manifest, strict size gate, and selected regular-Git or Release path;
13. the real-container execution matrix for all nine public shell entries;
14. final checksum, provenance, relocation, data-quality, and clean-clone qualification.

The correct final verdict is therefore `BLOCK` for AE-ready or complete pre-dataset qualification.
This does not invalidate the narrower D42/D43 image-function evidence.

## 6. Evidence, Inference, and Unknown Matrix

| Classification | Finding |
|---|---|
| **Evidence** | Exact enterprise digest, predict/live commands, two H800 UUIDs, Qwen rank-0 trace, Echo outputs, controller exits, worker exit, and root-wide file hashes are preserved. |
| **Evidence** | The D42 Retry-1 identity ran once and completed; it is terminal and non-reusable. |
| **Evidence** | The root has no field charging the attempt to D28. |
| **Evidence** | Megatron controller status had `13` dirty paths; Echo tar commit identity is absent from `qualification_result.json`. |
| **Evidence** | No complete three-model Task1→Task2→Task3 pre-dataset chain exists in this root. |
| **Inference governed by D33** | The old D28 replacement budget remained unconsumed and became unnecessary after the accepted D42/D43 supersession path. |
| **Inference prohibited** | The unconsumed D28 budget is not a free future slot and does not authorize another live job. |
| **Inference prohibited** | `new_trace_file_count=4` is not evidence of four rank traces. |
| **Inference prohibited** | Historical Qwen artifacts copied inside the runtime snapshot cannot fill current Retry-1 memory/profiler gaps. |
| **Unknown from this root alone** | A machine-verifiable D28 budget ledger entry or D28/D42 charge identifier. |
| **Unknown from this root alone** | Exact mapping from the dirty Megatron snapshot to the recorded commit plus diff. |
| **Unknown from this root alone** | Echo tar-to-producer-commit identity. |

## 7. Independent Review and Reconciliation

An independent StepCode Claude read-only review was run outside the repository:

```text
/tmp/sc26_d42_retry1_audit_advisor/.omx/artifacts/claude-act-as-an-independent-read-only-governance-and-evidence-revi-2026-07-18T17-29-30-674Z.md
bytes=11,421
sha256=344b760841af92a671f70608d978f4cd807918aa3f79b431d7a7dc7ef7c7e34d
```

Its verdicts were:

- D28 budget: `APPROVE — not consumed`;
- D42 functional evidence / D33 supersession: `WATCH` because the root alone lacks formal budget
  attribution and clean-source provenance;
- final AE-ready / complete three-by-three pre-dataset: `BLOCK`.

Reconciliation:

1. The reviewer correctly identifies the root-alone attribution and provenance gaps.
2. The broader task's D33 requirement and canonical D43 review already provide the policy-level
   supersession disposition. This audit therefore retains the canonical narrow functional `PASS`
   but attaches the reviewer's source-provenance `WATCH`.
3. No root-alone or task-level record supports promotion to final pre-dataset qualification.
4. The reviewer's rough coverage fraction is not adopted because the standalone Echo qualification
   is not atomically linked to the Qwen smoke as a completed AE workflow cell.

## 8. Final Disposition

### Accepted claim

> The immutable enterprise image completed one exact-two-H800 D42/D43 Retry-1 functional
> qualification consisting of a Qwen3 rank-0 Scaling Mode smoke and the complete standalone Echo
> slowdown pipeline. The sealed root has zero inventory integrity mismatches. The executed source
> snapshot is byte-sealed but not proven equivalent to a clean Megatron commit, and Echo producer
> commit provenance is incomplete.

### Required D28 wording

> The D42 Retry-1 identity is consumed and terminal. The old D28 replacement budget was not
> consumed; under D33/D43 it was superseded and became unnecessary. It must not be treated as an
> available retry slot. No further live submission is authorized by this evidence.

### Prohibited claim

> This root is not final AE-ready evidence, not a complete three-model-by-three-task pre-dataset,
> and not proof that all nine public shell entries run successfully in a clean canonical container.

## 9. Pending Tasks, Newly Discovered Issues, and Recommended Next Steps

### Pending tasks

- Build and qualify the missing real Task1→Task2→Task3 chains for all three models.
- Produce the current atomic trace/memory/SQLite/NCU bindings and Task3 outputs.
- Complete portable manifests, distribution, nine-entry execution, clean-clone, and final
  data-quality qualification.

### Newly discovered issues

1. `new_trace_file_count=4` represents one trace plus three replay tensors, not four rank traces.
2. The executed Megatron snapshot is sealed but controller Git status is dirty and not bound as a
   commit-plus-diff provenance object.
3. Echo tar bytes are bound, but the root omits the producer commit and retains two stale historical
   path strings.
4. Historical artifacts present inside copied source trees can be miscounted as current Retry-1
   outputs unless the run-owned inventory is used.
5. Echo profiler assets and the Qwen smoke are not an atomic Task1/Task2 capture chain.

### Recommended next steps

1. Treat this root as immutable, terminal evidence; do not patch, reuse, or resubmit it.
2. Record D28 as `UNCONSUMED_SUPERSEDED_NOT_NEEDED`, not `CONSUMED` and not `AVAILABLE_RETRY`.
3. Preserve the narrow D42/D43 functional `PASS` with an explicit clean-source/provenance `WATCH`.
4. Keep final AE/pre-dataset qualification `BLOCK` until the missing real three-model workflow,
   profiler alignment, packaging, and data-quality evidence exists.
5. Require new explicit authority before any future live job; this audit authorizes none.
