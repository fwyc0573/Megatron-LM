# SC'26 AE Fake-Level Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
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

## Current scope and evidence boundary

The formal AE model scope is:

* **GPT-175B dense**
* **Qwen3-A3B MoE**

The deferred DeepSeek-V3 path remains available only for historical diagnostics. It is not part of
the current AE chain and must not be used as a substitute for Qwen3 evidence.

Current fake-level local evidence status:

```text
fresh_execution_evidence=runtime_measurement_requires_external_single_gpu_qualification
functional_prebaked_evidence=functional_prebaked_not_release_qualified
real_fresh_task1=GPT_VERIFIED; QWEN3_VERIFIED
real_two_gpu_task2=VERIFIED_REUSED; TASK2_COMMANDS_EXECUTED=0
real_fresh_task3=GPT_VERIFIED; QWEN3_VERIFIED
functional_bundle=VERIFIED
cpu_prebaked_task3=GPT_VERIFIED; QWEN3_VERIFIED
clean_committed_clone=PENDING
functional-AE-ready=NO
release-ready=NO
```

The Fresh chains and functional prebaked path are closed. Keep `functional-AE-ready=NO` until the
same commands are reproduced from a clean committed clone. Historical output may be retained for
audit, but must not be relabeled as new qualified evidence. This reduced workflow never promotes
the result to distributed-accuracy or release qualification.

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

## Task3 path B: functional prebaked (CPU-only)

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

Use a functional bundle only with explicit opt-in and CPU/synthetic execution:

```bash
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp

# This must print 2.1.0. If it fails, follow task_memory/env_handbook.md.
python3 -c 'import xgboost; print(xgboost.__version__)'

AE_OUTPUT_ROOT=/data/ycfeng/tmp/sc26_ae_output_functional \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/absolute/path/to/functional-bundle \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
bash SC26-AE/task3_gpt175b.sh

AE_OUTPUT_ROOT=/data/ycfeng/tmp/sc26_ae_output_functional \
ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT=/absolute/path/to/functional-bundle \
TASK3_EXECUTION_MODE=synthetic \
TASK3_ALLOW_FUNCTIONAL_PREBAKED=1 \
SIMULATOR_HARDWARE_TYPE=cpu \
TASK3_META_PYTHON=python3 \
TASK3_SIMULATOR_PYTHON=python3 \
bash SC26-AE/task3_qwen3_a30b.sh
```

The functional path must also emit report, manifest, and marker files. These outputs prove fake-
level workflow wiring only; they do not become H800, Fresh, or release qualification.

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
