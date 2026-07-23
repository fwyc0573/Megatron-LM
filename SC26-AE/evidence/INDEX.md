# SC'26 AE Artifact Checklist

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Created the canonical-branch checklist for scripts, compact evidence, external artifacts, and verification commands |

## 1. Canonical delivery identity

Use exactly one delivery branch for both supported models:

```text
branch=sc26-ae-functional
branch_base_commit=3b1b51eec0162bd00b694c054dc9527016690c9a
formal_models=gpt175b,qwen3_a30b
deferred_model=dsv3
```

The local worktree used to assemble this branch is:

```text
/data/ycfeng/sc26_ae_task3_qwen
```

The worktree path is machine-local. AE users should locate the content by branch name, not by this
absolute path:

```bash
git switch sc26-ae-functional
```

There is no per-model final branch. Detached GPT/Qwen worktrees and historical `sc26-ae*`
branches are development or audit history, not current AE entry points.

The retained claims are deliberately narrow:

```text
functional_fake_level_only=YES
release_qualified=NO
distributed_accuracy_qualified=NO
paper_fidelity_reproduced=NO
```

DeepSeek-V3 records remain deferred in the task documentation and are not part of this inventory.

## 2. How this archive is organized

This directory contains the compact files that are useful for review and reuse. The original
validated runtime trees are preserved in place because removing files from them would invalidate
their manifests. The files here are checksum-verified copies, not renamed evidence.

| Inventory file | Purpose |
|----------------|---------|
| `SC26-AE/evidence/INDEX.md` | Human-readable checklist and AE orientation |
| `SC26-AE/evidence/index.json` | Complete machine-readable inventory, source paths, bytes, hashes, metrics, scripts, and external roots |
| `SC26-AE/evidence/checksums.sha256` | SHA256 list for all 177 compact archived files |

The machine-readable inventory records:

```text
archived_file_count=177
archived_total_bytes=16048905
workload_trace_count=40
memory_trace_count=40
task2_dataset_rows=727
task2_physical_gpu_count=2
task2_commands_executed_during_consolidation=0
external_artifact_root_count=7
```

## 3. Script checklist

### 3.1 AE entry points

| Task | GPT-175B | Qwen3-A3B | Hardware contract |
|------|----------|------------|-------------------|
| Task1 workload tracing | `SC26-AE/task1_gpt175b.sh` | `SC26-AE/task1_qwen3_a30b.sh` | One physical GPU; sequential fake ranks |
| Task2 slowdown dataset/predictor | `SC26-AE/task2_gpt175b.sh` | `SC26-AE/task2_qwen3_a30b.sh` | Exactly two physical GPUs |
| Task3 Fresh/prebaked simulation | `SC26-AE/task3_gpt175b.sh` | `SC26-AE/task3_qwen3_a30b.sh` | Fresh inputs from Task1/2; functional prebaked consumer may be CPU-only |

### 3.2 Shared implementation and workload sources

| Role | Path |
|------|------|
| Task1 orchestration | `SC26-AE/lib/task1_trace.sh` |
| Task2 orchestration | `SC26-AE/lib/task2_echo.sh` |
| Task3 orchestration | `SC26-AE/lib/task3_simulation.sh` |
| GPT dense workload source | `examples/update_pretrain_gpt.sh` |
| Qwen3 MoE workload source | `examples/qwen3_a3b_moe_scaling_wallclock_scan.sh` |
| Functional bundle builder/verifier | `SC26-AE/tools/package_prebaked.py` |
| Task2 metric extraction | `SC26-AE/tools/echo_metrics.py` |
| Rank-0 NCU normalization | `SC26-AE/tools/normalize_ncu_metrics.py` |
| Operator guide | `SC26-AE/README.md` |

Exact byte counts and SHA256 values for these files are stored under the `scripts` field of
`SC26-AE/evidence/index.json`.

## 4. Compact artifact inventory

| Scope | Files | Required contents | Directory |
|-------|------:|-------------------|-----------|
| GPT Task1 | 23 | marker, manifest, 8 workload traces, 8 memory traces, rank-0 NCU CSV, 4 logs | `gpt175b/task1/` |
| Qwen Task1 | 71 | marker, manifest, 32 workload traces, 32 memory traces, rank-0 NCU CSV, 4 logs | `qwen3_a30b/task1/` |
| Shared Task2 | 20 | two-GPU dataset input, 727-row dataset, model, scaler, metrics, marker, manifest, provenance, logs | `shared_task2/` |
| GPT Fresh Task3 | 16 | JSON/Markdown reports, marker, manifest, rank-0 slowdown trace/assets, provenance, logs | `gpt175b/task3_fresh/` |
| Qwen Fresh Task3 | 16 | JSON/Markdown reports, marker, manifest, rank-0 slowdown trace/assets, provenance, logs | `qwen3_a30b/task3_fresh/` |
| GPT CPU prebaked Task3 | 13 | JSON/Markdown reports, marker, manifest, provenance, logs | `gpt175b/task3_cpu_prebaked/` |
| Qwen CPU prebaked Task3 | 13 | JSON/Markdown reports, marker, manifest, provenance, logs | `qwen3_a30b/task3_cpu_prebaked/` |
| Historical exact-producer functional record | 4 | distribution manifest, build result, build log, verify log | `functional_distribution/c7288c6/` |
| Independent branch/provenance review | 1 | read-only review record | `test_records/` |
| **Total** | **177** | — | `SC26-AE/evidence/` |

### 4.1 Task1 topology and rank coverage

| Model | Topology | Required rank vector | Trace files | Memory files | NCU scope |
|-------|----------|----------------------|------------:|-------------:|-----------|
| GPT-175B | world=`1024`, PP=`8`, TP=`8`, EP=`1`, DP=`16` | `0,128,256,384,512,640,768,896` | 8 | 8 | global rank 0 only |
| Qwen3-A3B | world=`256`, PP=`8`, TP=`8`, EP=`4`, DP=`4` | `0,8,16,...,248` | 32 | 32 | global rank 0 only |

Key Task1 files:

| Model | Manifest | Rank-0 NCU feature CSV |
|-------|----------|------------------------|
| GPT-175B | `gpt175b/task1/artifact_manifest.json` | `gpt175b/task1/ncu/kernel_metric_output.csv` |
| Qwen3-A3B | `qwen3_a30b/task1/artifact_manifest.json` | `qwen3_a30b/task1/ncu/kernel_metric_output.csv` |

### 4.2 Shared Task2 dataset and predictor

Task2 provenance is fixed to a real two-GPU run:

```text
predictor_run_id=task2-20260722T142810Z-192-11368
CUDA_VISIBLE_DEVICES=0,1
physical_gpu_count=2
dataset_rows=727
average_validation_mse=0.04124828706619175
test_mse=0.061428837844613504
model_reload_max_abs_prediction_delta=0.0
```

| Asset | Path | SHA256 |
|-------|------|--------|
| Dataset input | `shared_task2/dataset/kernel_metric_output.csv` | `bf0c082d33d8ca001f534e4739e6a05a7a0a3c6fded523794786a2d4133fe173` |
| Training dataset | `shared_task2/dataset/train_dataset.csv` | `3f6fd7be758f016eb32cb864611348edee0d6e08d9289bb839c57c87879d7eb3` |
| XGBoost predictor | `shared_task2/predictor/xgb_model.json` | `6f9474775b1c60a0489abf1f314af1f9366a87d515bb629f9430a12dc605e06e` |
| Standard scaler | `shared_task2/predictor/standard_scaler.json` | `71fdebff4a797f860f2f9f4088c9f0bdf6ca83ae01df303ca6ddb573df9fa16b` |
| Task2 manifest | `shared_task2/artifact_manifest.json` | `d344fbfc0f4e56286efe9dd5ee6ac3f125ed3ad34fe8e4599bc9a71f67dda76e` |
| Metrics | `shared_task2/metrics.json` | `6cda47a5f011ec763443f8a31ccaa7fd27ac7398016a62b2b1bf6832003a8761` |

Do not rerun Task2 merely because Task3 encounters an unseen kernel. Task3 uses the documented
exact/unique-alias/`missing_skip` policy and leaves unmatched kernels at baseline slowdown.

### 4.3 Task3 reports and observed values

| Model/path | `rank0_step_time_ms` | Forward sum (ms) | Backward sum (ms) | Optimizer sum (ms) | Simulator load / execution / wall (s) |
|------------|---------------------:|-----------------:|------------------:|-------------------:|----------------------------------------:|
| GPT Fresh | 8276.64 | 1905.12 | 99.24 | 53.61 | 14.201178 / 18.494841 / 32.696019 |
| GPT CPU prebaked | 8276.64 | 1905.12 | 99.24 | 53.61 | 18.015626 / 23.576013 / 41.591639 |
| Qwen Fresh | 3051.24 | 0.32 | 26.29 | 3.62 | 70.935776 / 760.9432 / 831.878976 |
| Qwen CPU prebaked | 3051.24 | 0.32 | 26.29 | 3.62 | 84.063936 / 1072.67156 / 1156.735496 |

The reports are:

```text
gpt175b/task3_fresh/report.json
gpt175b/task3_cpu_prebaked/report.json
qwen3_a30b/task3_fresh/report.json
qwen3_a30b/task3_cpu_prebaked/report.json
```

Each directory also includes `report.md`, `artifact_manifest.json`, `run_marker.json`, selected
runtime logs, and provenance. These values check workflow behavior and general numeric sanity; they
are not accuracy or fidelity results.

## 5. Large artifacts retained outside ordinary Git

The following roots are intentionally external. Their anchor manifests and hashes are recorded in
`index.json`; selected compact records are already present in this directory.

| Artifact root | Bytes | Anchor manifest SHA256 |
|---------------|------:|------------------------|
| `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/gpt175b/task1/runs/gpt175b-20260722T181829Z` | 4,105,724,380 | `490bf26101edbb4594b7c21d14a3a7b858d5aa654b7bfa706224d660fdbc77bd` |
| `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/gpt175b/task3/runs/gpt175b-20260722T194650Z-296-25436` | 59,834,307 | `02b89c32f2d3c55628858709b8519933a73dd1a5d7339e1602bcab5125bd161f` |
| `/data/ycfeng/tmp/sc26_ae_clean_final_cpu_20260723T063628_gpt/gpt175b/task3/runs/gpt175b-clean-final-cpu-20260723T063628` | 59,993,775 | `d2f2838d4f605645b9258b2caf26250a7956a4c73fe850ed63d64ce6a5f55534` |
| `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/qwen3_a30b/task1/runs/qwen3_a30b-20260722T112049Z` | 2,437,829,861 | `a29941939c7b9b19b5d7cc2508ac5be94fafc926934f171d1d7a2602dd6fd123` |
| `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/qwen3_a30b/task3/runs/qwen3_a30b-20260722T173544Z-303-7078` | 1,416,193,192 | `805e646704ec9680481722f75d8df132ccffbab99afcee41f4c4a414b8512a9b` |
| `/data/ycfeng/tmp/sc26_ae_clean_final_cpu_20260723T063749_qwen/qwen3_a30b/task3/runs/qwen3_a30b-clean-final-cpu-20260723T063749` | 1,416,350,451 | `4f2903707633e88c62c6e55d98e9fb9ecd9cb4acf9098e2a733aee2c46886a89` |
| `/data/ycfeng/tmp/sc26_ae_functional_prebaked_clean_20260723T063212_c7288c6` | 6,554,852,354 | `6e9ab347df9107b9f2ada1900db47e56f246f6b445ed06d62b5e2b4960a50468` |

The last bundle was built and verified at exact producer commit
`c7288c66f0a6c3d0445edc841a6e5982d3b22f09`. It is retained as historical verified evidence.
After the archive/document commit is finalized, the complete external bundle must be rebuilt from
that new exact commit; its distribution manifest cannot be committed into its own producer commit
without creating a commit-hash self-reference.

## 6. Quick verification commands

Run from the repository root after checking out `sc26-ae-functional`:

```bash
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp

test "$(git branch --show-current)" = "sc26-ae-functional"

(
  cd SC26-AE/evidence
  sha256sum -c checksums.sha256
)

python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("SC26-AE/evidence")
index = json.loads((root / "index.json").read_text())
artifacts = index["artifacts"]

assert index["canonical_branch"] == "sc26-ae-functional"
assert len(artifacts) == 177
assert sum(item["bytes"] for item in artifacts) == 16048905
assert sum(item["kind"] == "workload_trace" for item in artifacts) == 40
assert sum(item["kind"] == "memory_trace" for item in artifacts) == 40
assert index["summary"]["slowdown_dataset_rows"] == 727
assert index["summary"]["task2_physical_gpu_count"] == 2
assert index["summary"]["task2_commands_executed_during_consolidation"] == 0

for item in artifacts:
    path = root / item["path"]
    assert path.is_file(), path
    data = path.read_bytes()
    assert len(data) == item["bytes"], path
    assert hashlib.sha256(data).hexdigest() == item["sha256"], path

print("ARCHIVED_FILE_COUNT=177")
print("ARCHIVED_TOTAL_BYTES=16048905")
print("WORKLOAD_TRACE_COUNT=40")
print("MEMORY_TRACE_COUNT=40")
print("TASK2_DATASET_ROWS=727")
print("TASK2_PHYSICAL_GPU_COUNT=2")
print("EVIDENCE_INDEX_STATUS=PASS")
PY
```

Expected rank checks:

```text
GPT_TASK1_RANKS=0,128,256,384,512,640,768,896
GPT_TASK1_TRACE_COUNT=8
GPT_TASK1_MEMORY_COUNT=8
QWEN_TASK1_RANKS=0,8,16,...,248
QWEN_TASK1_TRACE_COUNT=32
QWEN_TASK1_MEMORY_COUNT=32
NCU_RANK_IDS=0
```

## 7. Reviewer sign-off checklist

- [x] One canonical branch is named: `sc26-ae-functional`.
- [x] GPT-175B and Qwen3-A3B use the same branch.
- [x] Six formal Task1/2/3 entry scripts are identified.
- [x] GPT Task1 contains all 8 PP representative traces and memory files.
- [x] Qwen Task1 contains all 32 PP×EP representative traces and memory files.
- [x] Both model-local NCU feature files are global-rank-0 records.
- [x] Task2 dataset and predictor preserve two-GPU provenance and were not rerun for consolidation.
- [x] Fresh Task3 report, manifest, marker, provenance, and slowdown records exist for both models.
- [x] CPU-only prebaked Task3 report, manifest, marker, and provenance exist for both models.
- [x] Large raw runtime trees and the complete functional bundle remain external and manifest-anchored.
- [x] `sc26-ad.tex` is outside this consolidation and remains unmodified.
- [ ] Rebuild and verify the external functional bundle from the final archive commit; record it outside the commit to avoid self-reference.
