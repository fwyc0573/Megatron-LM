# SC'26 AE Fake-Level Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-23 | Relaxed `verify-functional` to usability checks and added the external-bundle AE usage guide |
| 2026-07-23 | Declared `sc26-ae-functional` as the single GPT/Qwen delivery branch and linked the checksum-backed artifact checklist |
| 2026-07-23 | Closed the clean committed-clone replay and recorded the exact functional bundle producer commit |
| 2026-07-23 | Recorded verified GPT/Qwen Fresh chains, the real functional bundle, both CPU-only Task3 runs, and the remaining clean-clone gate |
| 2026-07-23 | Bound CPU-only functional Task3 to an explicit `python3` runtime with XGBoost and `/data/ycfeng/tmp` temporary storage |
| 2026-07-21 | Replaced the deferred DeepSeek-V3 path with the Qwen3-A3B fake-level workflow (`world_size=256`, `PP=8`, `TP=8`, `EP=4`, `DP=4`) and documented the current evidence boundary |
| 2026-07-21 | Documented Qwen3 PP×EP representative tracing with independent rank-0 Nsight Systems/Compute provenance |
| 2026-07-20 | Clarified fake-level D16 behavior and rank-scoped Task1 capture |
| 2026-07-19 | Added explicit artifact-source and fail-fast workflow notes |

This is the SC'26 AE operator entry point. The intended shell-driven workflow is:

```text
Task1 workload tracing
  -> Task2 slowdown dataset / predictor
  -> Task3 end-to-end simulation
```

A second path accepts a stored functional profiling/predictor bundle and runs Task3 directly on
the CPU. This README describes the requested fake-level functional workflow. It does not claim
multi-node distributed accuracy and does not modify
`2026-SC-first-submission/sc25-ad-ae/for-paper-authors/sc26-ad.tex`.

## Canonical branch and artifact map

The single AE delivery branch for both supported models is:

```bash
git switch sc26-ae-functional
```

Do not select a branch by model name. Other `sc26-ae*` branches, detached worktrees, and OMX worker
worktrees are historical development or audit state. The canonical branch contains the formal
Task1/2/3 scripts for GPT-175B and Qwen3-A3B together.

Use the following files to locate and verify the retained artifacts:

| File | Purpose |
|------|---------|
| [`SC26-AE/evidence/INDEX.md`](evidence/INDEX.md) | Human-readable script/artifact checklist, topology, key metrics, and verification commands |
| [`SC26-AE/evidence/index.json`](evidence/index.json) | Machine-readable list of 177 compact artifacts plus seven external large roots |
| [`SC26-AE/evidence/checksums.sha256`](evidence/checksums.sha256) | SHA256 verification list for the compact archive |

The compact archive contains selected workload traces, memory files, model-local rank-0 NCU
features, the real two-GPU Task2 dataset and predictor, Fresh/CPU Task3 reports, manifests,
markers, provenance, and key logs. Multi-gigabyte NCU reports, Nsight databases, replay caches,
expanded simulator output, and complete functional bundles remain external and are anchored by
manifest path and SHA256 in `index.json`. Original validated output trees are preserved intact;
the in-branch evidence files are checksum-verified copies, not files removed from those runs.

## Current scope and evidence boundary

The formal AE model scope is:

* **GPT-175B dense**
* **Qwen3-A3B MoE**

The deferred DeepSeek-V3 path remains available only for historical diagnostics. It is not part of
the current AE chain and must not be used as a substitute for Qwen3 evidence.

The three historical DeepSeek entry points remain discoverable for audit purposes only. Do not run
these commands as part of the current formal chain:

```bash
# DEFERRED / HISTORICAL ONLY -- not a formal AE command
bash SC26-AE/task1_dsv3.sh
bash SC26-AE/task2_dsv3.sh
bash SC26-AE/task3_dsv3.sh
```

Current fake-level local evidence status:

```text
fresh_execution_evidence=runtime_measurement_requires_external_single_gpu_qualification
functional_prebaked_evidence=functional_prebaked_not_release_qualified
real_fresh_task1=GPT_VERIFIED; QWEN3_VERIFIED
real_two_gpu_task2=VERIFIED_REUSED; TASK2_COMMANDS_EXECUTED=0
real_fresh_task3=GPT_VERIFIED; QWEN3_VERIFIED
functional_bundle=VERIFIED
cpu_prebaked_task3=GPT_VERIFIED; QWEN3_VERIFIED
archived_functional_bundle_producer_commit=c7288c66f0a6c3d0445edc841a6e5982d3b22f09
clean_committed_clone=VERIFIED
functional-fake-level-AE-ready=YES
release-ready=NO
```

The Fresh chains and functional prebaked path are closed, including a replay from the exact clean
producer commit above. Functional distribution verification checks that the bundle's dataset and
predictor files are complete and checksum-intact and that the bundle's Echo-slowdown and
megatron-sim-engine source identities match the checked-in `.source_commit` files; it does not
require the checkout to equal the bundle's recorded outer producer commit. Historical output may be retained for audit,
but must not be relabeled as new qualified evidence. This reduced workflow never promotes the
result to distributed-accuracy or release qualification.

## Hardware and topology

| Task | Runtime hardware | Fake/simulator topology |
|------|------------------|-------------------------|
| Task1 GPT-175B | One physical GPU; execute fake ranks sequentially | `fake_world_size=1024`, `PP=8`, `TP=8`, `DP=16`; trace only `0,128,256,384,512,640,768,896` |
| Task1 Qwen3-A3B | One physical GPU; execute fake ranks sequentially | `fake_world_size=256`, `PP=8`, `TP=8`, `EP=4`, `DP=4`; trace `0,8,16,...,248` (PP×EP representatives) |
| Task2 | **Exactly two GPUs**; never downgrade to one GPU | Echo-slowdown dataset, training, reload, and prediction |
| Task3 Fresh | Inputs produced by Task1/Task2; simulator may run on CPU | Analytical backend, `local_size=8`, overlap enabled |
| Task3 functional prebaked | CPU-only | Verified functional bundle with synthetic execution |

Task1 NCU/slowdown workload collection is limited to **global rank 0** because the slowdown
dataset needs one representative DDP rank:

```json
{
  "rank_scope": "global_rank_0",
  "rank_ids": [0],
  "physical_gpu_count": 1,
  "missing_kernel_count": 0
}
```

This does not reduce model-level Task1 coverage: GPT retains eight PP representative ranks, while
Qwen3-A3B retains all 32 PP×EP representative ranks for its 256-rank topology.

## Environment requirements

Real runs use the reviewed worker image/runtime. A controller without these fixed paths must fail
immediately; do not switch interpreters or fabricate GPU evidence:

| Purpose | Fixed path/requirement |
|---------|------------------------|
| Task1 and Fresh Task3 | `/opt/conda/envs/megatron_env/bin/python` and its matching `torchrun` |
| Task2 | `/opt/conda/envs/echo_slowdown/bin/python` |
| Functional prebaked Task3 on a CPU controller | `python3` with `numpy`, `pandas`, and `xgboost==2.1.0` |
| Nsight Systems | `nsys` |
| Nsight Compute | `ncu` |

Real Task1 requires at least one physical GPU. Real Task2 requires two different decimal GPU IDs
in `CUDA_VISIBLE_DEVICES`. If the controller lacks the fixed Megatron interpreter or a GPU, that
is an environment blocker; do not bypass it with `TASK2_SKIP_HARDWARE_CHECK`, synthetic features,
or another fallback.

On a reviewed worker, run setup once before the task scripts and choose the source explicitly:

```bash
GROUPED_GEMM_SOURCE=archive bash SC26-AE/setup.sh
```

Setup validates the fixed runtime bindings; it does not discover or silently replace a missing
interpreter/tool.

## Source identity and qualification boundary

The compact archive and historical functional record use these source identities:

| Component | Commit / status |
|-----------|----------------|
| Main repository commit (archive base) | `3b1b51eec0162bd00b694c054dc9527016690c9a` |
| Archived exact functional producer | `c7288c66f0a6c3d0445edc841a6e5982d3b22f09` |
| Echo-slowdown | `1390b4416ded08bc1b9cd0620d329d81d4470bf9` |
| megatron-sim-engine | `51eed0404635632fd52a99b3f372d5830b1d73b4` |
| collective-sim | optional historical backend; not vendored in the AE path |
| Current worker image | `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae` (immutable digest unresolved; no release qualification) |

The current status is intentionally explicit:

```text
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
QUICK_TASK3_STATUS=LOCAL_SMOKE_COMPATIBLE_NOT_RELEASE_QUALIFIED
REAL_RUNTIME_EVIDENCE_STATUS=PENDING_H800_QUALIFICATION
CANONICAL_PREBAKED_DISTRIBUTION=NOT_SELECTED
functional-fake-level-AE-ready=YES
AE-ready=NO
release-ready=NO
```

`AE-ready=NO` refers to release/distributed qualification. The functional fake-level workflow is
usable, but it must not be promoted to a release or accuracy claim. The image tag alone is never
an immutable qualification proof; resolve a digest before any future real-worker qualification.

## Task1: workload tracing

Use a new absolute `AE_OUTPUT_ROOT` for every run. The scripts create traces, memory JSON,
manifests, and markers; Nsight artifacts are created when the corresponding capture flags are
enabled. Real Task1 must set `CAPTURE_NCU=1`; otherwise it fails before workload execution because
Fresh Task3 requires workload-aligned kernel features.

### GPT-175B dense (eight PP representative ranks)

```bash
AE_OUTPUT_ROOT="$PWD/SC26-AE/output_gpt175b" \
SCALE_GPU=0 \
CAPTURE_NSYS=1 \
CAPTURE_NCU=1 \
QUICK=1 \
bash SC26-AE/task1_gpt175b.sh
```

The fixed rank order is:

```text
0,128,256,384,512,640,768,896
```

The GPT source is `examples/update_pretrain_gpt.sh`. It accepts a strictly validated
`FAKE_RANK_ORDER`, so the NCU invocation does not accidentally execute all 1024 fake ranks.
The older `examples/update_pretrain_gpt-copy.sh` remains a reference script; the AE wrapper uses
the validated `update_pretrain_gpt.sh` implementation.

### Qwen3-A3B MoE (32 PP×EP representative ranks)

```bash
AE_OUTPUT_ROOT="$PWD/SC26-AE/output_qwen3_a30b" \
SCALE_GPU=0 \
CAPTURE_NSYS=1 \
CAPTURE_NCU=1 \
QUICK=0 \
bash SC26-AE/task1_qwen3_a30b.sh
```

The formal fake-level chain uses `QUICK=0` and the 32-rank order `0,8,16,24,...,248`. This is
the PP×EP representative scope for the 256-rank topology; it avoids duplicating identical TP/DP
graphs while retaining every PP stage and expert-parallel group. The source script is
`examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`.

After a successful Task1 run, the model output contains:

```text
<output>/<model>/task1/runs/<capture_id>/
  runtime/profiler_log/           # eight GPT files; 32 Qwen3 PP×EP files
  runtime/memory_traces_scaling/  # memory JSON files for the selected ranks
  logs/                            # source, rank-timing, and summary logs
  nsys/                            # when CAPTURE_NSYS=1
  ncu/kernel_metric_output.csv    # workload-aligned, global rank 0
  artifact_manifest.json
<output>/<model>/task1/capture_marker.json
```

Fresh Task3 accepts only the model-local `ncu/kernel_metric_output.csv`. A shared Task2 NCU CSV,
synthetic/default kernel feature, manual scaling, or calibration factor is not a substitute.

## Task2: slowdown dataset and predictor

Task2 **must use exactly two GPUs**. Replace `0,1` with the two different IDs allocated on the
worker:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
REBUILD=1 \
bash SC26-AE/task2_gpt175b.sh

CUDA_VISIBLE_DEVICES=0,1 \
bash SC26-AE/task2_qwen3_a30b.sh
```

The first command creates the shared predictor bundle; the later model entry reuses the same
verified `predictor_run_id`. The wrapper checks:

* exactly two distinct decimal IDs in `CUDA_VISIBLE_DEVICES`;
* the fixed Echo interpreter and two visible CUDA devices;
* NCU collection, slowdown merge, training, save/reload, prediction, and metrics;
* complete `xgb_model.json`, `standard_scaler.json`, source manifest, and checksums.

If only one GPU is available, Task2 must fail. A single-GPU synthetic fixture is not Task2
qualification. Task2 output is written to:

```text
<output>/_shared/task2/runs/<predictor_run_id>/
  dataset/
  predictor/xgb_model.json
  predictor/standard_scaler.json
  metrics.json
  metrics.md
  predictor_marker.json
  artifact_manifest.json
```

The reusable Task2 report preserves these machine-readable fields: `task2_run_all_elapsed_seconds`,
`dataset_row_count`, `validation_mse_by_fold`, `average_validation_mse`, `test_mse`,
`model_reload_max_abs_prediction_delta`, `scaler_feature_count`, `scaler_mean_count`,
`scaler_scale_count`, `scaler_nonzero_scale_count`, and `prediction_sample`.

## Task3 path A: Fresh chain

Run Fresh Task3 only after the corresponding Fresh Task1 and shared Task2 have completed. The
source and execution mode are explicit; the analytical simulator itself may execute on CPU:

```bash
AE_OUTPUT_ROOT="$PWD/SC26-AE/output_gpt175b" \
ARTIFACT_SOURCE=fresh \
TASK3_EXECUTION_MODE=real \
SIMULATOR_HARDWARE_TYPE=cpu \
bash SC26-AE/task3_gpt175b.sh

AE_OUTPUT_ROOT="$PWD/SC26-AE/output_qwen3_a30b" \
ARTIFACT_SOURCE=fresh \
TASK3_EXECUTION_MODE=real \
SIMULATOR_HARDWARE_TYPE=cpu \
bash SC26-AE/task3_qwen3_a30b.sh
```

The Fresh resolver binds the same chain's:

* Task1 capture marker, SQLite, rank-0 slowdown trace, and rank-0 NCU CSV;
* Task2 predictor marker, `xgb_model.json`, and `standard_scaler.json`;
* source commit, topology, profile, file SHA256, and nested manifests.

Missing, partial, stale, or checksum-mismatched inputs fail fast. The wrapper never switches to a
prebaked source automatically. A successful run emits:

```text
<output>/<model>/task3/runs/<simulation_run_id>/
  report.json
  report.md
  artifact_manifest.json
<output>/<model>/task3/run_marker.json
```

The report includes `rank0_step_time_ms`, forward/backward/optimizer scheduled sums, and
simulator load/execution/wall-clock values. These values demonstrate workflow output; they are not
distributed-accuracy claims for this task.

The exact report fields are `rank0_step_time_ms`, `rank0_forward_step_duration_sum_ms`,
`rank0_backward_step_duration_sum_ms`, `rank0_optimizer_step_duration_sum_ms`,
`rank0_comp_plus_comm_diagnostic_ms`, `simulator_load_time_s`, `simulator_execution_time_s`, and
`simulator_wall_clock_s`.

## Task3 path B: functional prebaked (CPU-only)

### For AE reviewers: run Task3 from the distributed external bundle

If you received an external functional bundle (a directory such as
`sc26_ae_functional_final_<timestamp>_<commit>` containing `distribution_manifest.json` and
`bundles/{gpt175b,qwen3_a30b,shared_task2}/`), you can run the end-to-end Task3 simulation on a
CPU-only machine without any GPU. The bundle is a pure data package: it carries the Task1 traces
and the Task2 dataset/predictor weights, and it intentionally contains no shell scripts. All
entry-point scripts come from this repository.

Step-by-step:

```bash
# 1. Get the code (the bundle itself has no scripts)
git clone <repository-url> && cd <repository>
git switch sc26-ae-functional
# Echo-slowdown and megatron-sim-engine are vendored directories in this branch.
# No submodule initialization is required.

# 2. Check the CPU-only Python runtime: python3 with numpy, pandas, and xgboost==2.1.0
python3 -c 'import numpy, pandas, xgboost; print(xgboost.__version__)'   # must print 2.1.0

# 3. Point temporary storage away from /tmp if your host requires it
export TMPDIR=/path/to/scratch TEMP=/path/to/scratch TMP=/path/to/scratch

# 4. (Optional) verify the bundle's integrity before running
python3 SC26-AE/tools/package_prebaked.py verify-functional \
  --repo-root "$PWD" \
  --prebaked-root /absolute/path/to/external-bundle
# Expected: DISTRIBUTION_STATUS=verified

# 5. Run CPU-only Task3 for both models
AE_OUTPUT_ROOT=/path/to/scratch/sc26_ae_output \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/absolute/path/to/external-bundle \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
bash SC26-AE/task3_gpt175b.sh

AE_OUTPUT_ROOT=/path/to/scratch/sc26_ae_output \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/absolute/path/to/external-bundle \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
bash SC26-AE/task3_qwen3_a30b.sh
```

Each run ends with a `[PASS] Task3 model=... artifact_source=prebaked` line and writes
`report.json`, `report.md`, `run_marker.json`, and `artifact_manifest.json` under
`$AE_OUTPUT_ROOT/<model>/task3/`. As a rough guide on one CPU controller, the GPT-175B simulation
finishes in a few minutes, while the Qwen3-A3B simulation takes roughly 20 minutes
(`simulator_execution_time_s` around 1000-1250 s); do not interrupt it early.

The rest of this section documents how a bundle producer builds and verifies such a distribution.

### Building and verifying a functional distribution

After the real Fresh Task3 chains are closed, create a functional distribution from the real
artifacts:

```bash
python3 SC26-AE/tools/package_prebaked.py build-functional \
  --repo-root "$PWD" \
  --output-root <fresh-output-root> \
  --staging-root <new-functional-bundle-root> \
  --distribution-id <distribution-id> \
  --result-json <result-json>

python3 SC26-AE/tools/package_prebaked.py verify-functional \
  --repo-root "$PWD" \
  --prebaked-root <functional-bundle-root>
```

`build-functional` packages only `gpt175b`, `qwen3_a30b`, and `shared_task2`. It checks GPT's eight
representative ranks, Qwen3-A3B's 32 PP×EP representative ranks, and an independent rank-0 NCU
CSV for each model.
The bundle evidence class is fixed to:

```text
functional_prebaked_not_release_qualified
```

Use a functional bundle only with explicit opt-in and CPU/synthetic execution. The exact run
commands are listed in "For AE reviewers: run Task3 from the distributed external bundle" above;
the same commands apply to a locally built bundle by pointing `PREBAKED_ROOT` at the staging root.
On this host, use `/data/ycfeng/tmp` for `TMPDIR`/`TEMP`/`TMP` and `AE_OUTPUT_ROOT`; if
`python3 -c 'import xgboost; print(xgboost.__version__)'` does not print `2.1.0`, follow
`task_memory/env_handbook.md`.

The functional path must also emit report, manifest, and marker files. These outputs prove fake-
level workflow wiring only; they do not become H800, Fresh, or release qualification.

The packager validates bundle structure, dataset/predictor completeness, and file checksums, and
checks that the bundle's Echo-slowdown and megatron-sim-engine source identities match the
`.source_commit` files in the current checkout. It does not require the checkout to equal the bundle's recorded producer commit.
There is no automatic fresh-to-prebaked fallback: select
`ARTIFACT_SOURCE=fresh` or `ARTIFACT_SOURCE=prebaked` explicitly. A missing kernel follows the
documented exact-match, unique-alias, or `missing_skip` policy; it does not trigger Task2 rerun.
The functional package is a local/synthetic distribution, not a release package.

## Fail-fast troubleshooting

* Missing fixed interpreter, GPU, Nsight tool, manifest, marker, or checksum: stop and report the
  root cause; do not switch interpreters or fabricate evidence.
* One visible GPU for Task2: stop; Task2 requires two distinct physical GPU IDs.
* A missing Task3 kernel: use exact match, one unambiguous alias, or `missing_skip` baseline;
  do not recollect the predictor dataset.
* A source-identity mismatch (`Echo-slowdown` or `megatron-sim-engine`): switch to the canonical
  `sc26-ae-functional` branch and verify the two `.source_commit` files; the bundle only depends on
  these two vendored components, not on the outer commit.

`collective-sim` is optional background infrastructure for historical simulator work. The formal
fake-level AE path explicitly uses the analytical backend and does not silently fall back to
`collective-sim`.

## Validation commands (current checkout)

These checks do not require a distributed run and do not make an accuracy claim:

```bash
bash -n \
  examples/update_pretrain_gpt.sh \
  SC26-AE/lib/task1_trace.sh \
  SC26-AE/lib/task2_echo.sh \
  SC26-AE/lib/task3_simulation.sh \
  SC26-AE/task1_gpt175b.sh SC26-AE/task1_qwen3_a30b.sh \
  SC26-AE/task2_gpt175b.sh SC26-AE/task2_qwen3_a30b.sh \
  SC26-AE/task3_gpt175b.sh SC26-AE/task3_qwen3_a30b.sh

PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  tests/unit/test_sc26_ae_package_prebaked.py

bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task3_functional_prebaked.sh
bash tests/integration/test_sc26_ae_task3_contract.sh
bash tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
```

Current fake-level results are: package unit `40 passed`; Task1 contract `PASS_COUNT=45`;
functional Task3 `PASS_COUNT=5` (one successful GPT and one successful Qwen3-A3B chain);
Task3 contract `PASS_COUNT=11`; and legacy prebaked CPU regression `3/3` models. Fixture
numbers such as rank0 step time validate report schema and simulator output only.

## Real Fresh and clean-clone closeout order

Before claiming the **functional fake-level AE workflow** is ready, execute the following sequence:

1. Complete GPT Task1 on the reviewed worker (eight traces plus rank-0 NCU).
2. Complete Qwen3-A3B Task1 on the reviewed worker (32 PP×EP traces plus rank-0 NCU).
3. Complete Task2 with `CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1>` and exactly two GPUs.
4. Run GPT and Qwen3-A3B Fresh Task3 and retain report, manifest, and marker for each.
5. Only after both Fresh Task3 chains close, build and verify the real functional distribution.
6. Run CPU-only Task3 with that real bundle and recheck reports, manifests, markers, and checksums.
7. Re-run the minimal fake-level matrix from a clean commit/clean clone and save the final report.

If the controller lacks the fixed interpreter, GPU, or Nsight tool, record the environment blocker
and stop the real chain. Do not relabel old partial output, synthetic fixtures, or a shared Task2
CSV as Fresh evidence.

The sequence above is a functional reproducibility gate only. It does not establish distributed
accuracy, paper-number fidelity, or release qualification; `release-ready` remains `NO` in this
scope.
