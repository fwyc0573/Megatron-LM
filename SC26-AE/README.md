# SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Clarified that MoE D16 timing is an isolated rank-0 preflight: QUICK records a non-gating observation, while full applies the 7200-second gate before any 256-rank selected capture |
| 2026-07-20 | Clarified that D16 applies only to the two 256-rank MoE models; GPT representative-rank timing is diagnostic and carries no 7200-second gate |
| 2026-07-19 | Documented that qualification metrics are integrity-bound opaque issuer payloads; task-specific semantics remain with the external issuer |
| 2026-07-19 | Closed the Phase 7 documentation gaps for paper traceability, QUICK compatibility, Task2 metrics, runtime/distribution status, communication scope, and root-cause troubleshooting |
| 2026-07-19 | Added strict prebaked packager instructions and an explicit synthetic-fixture boundary |
| 2026-07-19 | Added explicit source-pin identities, a complete nine-entry command matrix, and exact Task3 report field names for auditability |
| 2026-07-19 | Added fixed v1.2-ae runtime verification for both task interpreters and Nsight tools; setup now fails before task commands on contract drift |
| 2026-07-19 | Aligned the Task2 default interpreter with the fixed v1.2-ae worker path |
| 2026-07-19 | Clarified D29 test-issue self-repair scope and v1.2-ae current-image boundary |
| 2026-07-19 | Added the evaluator-facing nine-entry workflow, explicit artifact-source commands, and evidence-bound qualification notes |

This directory is the evaluator-facing entry point for the SC'26 AE workflow.
It exposes three explicit model entries for each of the three tasks. The scripts
are intentionally thin and fail fast: they do not infer a model, switch an
interpreter, change an artifact source, retry a failed runtime, or overwrite a
versioned result.

The evaluator workflow is derived from the current paper draft at
`2026-SC-first-submission/sc25-ad-ae/for-paper-authors/sc26-ad.tex`. This task does
not edit that file. Auditable, evidence-bound wording changes are listed in
`task_memory/task_2026-07-15_sc26_ae_workflow/tex_change_suggestions.md` for the
paper author to apply after the corresponding real qualification evidence exists.

## Qualification status

The local contract and synthetic workflows are implemented and testable. They are
not a substitute for the real H800 qualification. In particular, the current
worktree must not be described as containing a qualified pre-dataset until a fresh
real run records GPU identity, source/image/interpreter provenance, trace and memory
evidence, checksums, and semantic validation for the selected model matrix.

The canonical worker image reference for the eventual qualification is:

```text
hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae
```

The immutable digest must be resolved and recorded by the reviewed qualification
preflight; the tag alone is not treated as a digest claim.
Historical Gate B1 evidence produced with `v1.1-image-11c794ef` remains audit
history only and is not a current qualification or pre-dataset claim.

### Repository and source identities

These are the declared compatibility pins in the current checkout. The main
repository row identifies the clean baseline commit; uncommitted local changes
must be committed and re-verified before any release or clean-clone claim.

| Component | Pinned identity |
|-----------|-----------------|
| Main repository commit | `c217ce93156e7c37e065da2989c1a482f12ecebc` |
| Echo-slowdown | `1390b4416ded08bc1b9cd0620d329d81d4470bf9` |
| megatron-sim-engine | `39755169f73f6c748e8d7376c3a2158c6569436b` |
| collective-sim | `6e06e3f5140cd4e2e7c12a35586ebcdc0f410df0` |

The current image reference is `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`;
its immutable digest is intentionally **pending qualification** and must be
recorded from the approved worker preflight. A tag, a local checkout, or a
synthetic fixture is not an immutable image or release identity.

The current local evidence boundary is explicit:

```text
EVIDENCE_CLASS=local_synthetic_not_gpu_qualification
real_pre_dataset=NOT_QUALIFIED
AE-ready=NO
```

## Hardware and topology

| Task | Real hardware | Virtual/simulator topology |
|------|---------------|----------------------------|
| Task1 | One physical GPU; one fake rank at a time | GPT-175B `1024/PP8/TP8/DP16`; Qwen3-A30B and DSV3 `256/PP4/TP8/DP8/EP8` |
| Task2 | Exactly two visible GPUs for a real Echo run | One shared predictor bundle with a model-labelled marker |
| Task3 fresh | Megatron/sim-engine environment plus validated Task1/Task2 inputs | `--cc-backend analytical`, `--overlap-mode on`, `--local-size 8` |
| Task3 prebaked | CPU-only is sufficient for the analytical simulation path after the bundle is verified | `LOCAL_SIZE=8`; measured host RSS is recorded by the e2e report |

All three model profiles use bf16, mock data, DDP gradient overlap, scaling warmup
`3`, and profile iterations `1`. The two MoE models use a full 256-rank Task1
capture for release evidence; `QUICK=1` selects the documented four-rank smoke
subset and is not silently extrapolated into a full qualification.

The current QUICK compatibility boundary is machine-readable:

```text
QUICK_TASK3_STATUS=LOCAL_SMOKE_COMPATIBLE_NOT_RELEASE_QUALIFIED
```

The local fresh-chain fixture proves that a four-rank QUICK capture can pass the
Task3 structural, provenance, and report contracts. It does not prove that the
subset represents a complete 256-rank MoE trace or a real H800 result. Release
packaging therefore still requires the full qualified Task1 source selected by
the D16 timing gate.

## Setup

Run from the repository root inside the reviewed AE image/environment:

```bash
GROUPED_GEMM_SOURCE=archive bash SC26-AE/setup.sh
```

`GROUPED_GEMM_SOURCE` is mandatory and must be either `vcs` or `archive`. The
archive path verifies the pinned grouped-gemm and CUTLASS SHA256 values before
installation. Before invoking the installer, setup validates the fixed worker
runtime bindings below; it does not install or discover core runtime dependencies:

| Role | Fixed path | Required contract |
|------|------------|-------------------|
| Megatron / Task1 / Task3 | `/opt/conda/envs/megatron_env/bin/python` | Python `3.9.18`, torch `2.1.2`, torch CUDA `12.1`, live CUDA/NVML, required scientific imports |
| Echo / Task2 | `/opt/conda/envs/echo_slowdown/bin/python` | Python `3.10.20`, torch `2.1.2+cu121`, torch CUDA `12.1`, torchvision `0.16.2+cu121`, torchaudio `2.1.2+cu121`, pinned `SlowdownPredictor` import |
| Nsight Systems | `/usr/local/bin/nsys` | version `2024.4.2.133`, working `profile` and `export` |
| Nsight Compute | `/usr/local/cuda/bin/ncu` | version `2024.3.2.3`, `--csv` and `--log-file` query capabilities |

The fixed Echo predictor source is the checked-out
`Echo-slowdown/training_testing/prediction_api.py`; setup does not select another
checkout or interpreter. A missing path, wrong version, failed import, unavailable
CUDA/NVML query, or unavailable Nsight command fails fast. Only after the explicit
`GROUPED_GEMM_SOURCE` installer succeeds and grouped-gemm import/backend verification
passes does setup print `SC26_AE_SETUP_STATUS=verified`.

This status proves only local runtime/setup readiness. It is not a GPU qualification,
source-provenance approval, or final reusable `release_pre_dataset` claim. The setup
script never switches source methods automatically, changes thresholds, or adds a
fallback interpreter/tool.

## Public commands

Every command accepts `AE_OUTPUT_ROOT=/absolute/path` and `QUICK=0|1` where
applicable. The output root should be new for a qualification run.

### Task1: workload tracing

Task1 runs compute for selected fake ranks on one physical GPU and records the
communication metadata needed by the simulator. Set `SCALE_GPU` explicitly when
more than one GPU is visible:

```bash
SCALE_GPU=0 AE_OUTPUT_ROOT="$PWD/SC26-AE/output" \
  bash SC26-AE/task1_gpt175b.sh

SCALE_GPU=0 QUICK=1 AE_OUTPUT_ROOT="$PWD/SC26-AE/output" \
  bash SC26-AE/task1_qwen3_a30b.sh

SCALE_GPU=0 QUICK=1 AE_OUTPUT_ROOT="$PWD/SC26-AE/output" \
  bash SC26-AE/task1_dsv3.sh
```

The release path enables the one-shot Nsight boundary only after its GPU/runtime
qualification has been reviewed:

```bash
CAPTURE_NSYS=1 SCALE_GPU=0 bash SC26-AE/task1_qwen3_a30b.sh
```

Each successful run creates a fresh `capture_id`, a summary, memory JSON files,
optional Nsight/SQLite artifacts, and a verified `artifact_manifest.json` before
publishing `capture_marker.json`. A partial or existing run directory is an error.

### Task2: slowdown collection and predictor training

Real Task2 requires exactly two distinct visible GPU IDs and a clean pinned
Echo-slowdown checkout:

```text
/opt/conda/envs/echo_slowdown/bin/python
```

This is the fixed default interpreter recorded by the v1.2-ae worker contract
(Python `3.10.20`, torch `2.1.2+cu121`, CUDA `12.1`). The runner does not search
for another environment or switch interpreters. If this exact worker path is
missing or cannot observe exactly two CUDA devices, real Task2 fails before
executing the Echo source.

```bash
CUDA_VISIBLE_DEVICES=0,1 REBUILD=1 \
  bash SC26-AE/task2_gpt175b.sh

CUDA_VISIBLE_DEVICES=0,1 \
  bash SC26-AE/task2_qwen3_a30b.sh

CUDA_VISIBLE_DEVICES=0,1 \
  bash SC26-AE/task2_dsv3.sh
```

The first build writes one shared verified predictor bundle and a
`predictor_run_id`; later model entries attach to that exact bundle. `REBUILD=1`
always uses a new run identity. `REBUILD=0` never silently rebuilds a missing or
invalid predictor.

For local contract testing only, the test fixtures set
`TASK2_EXECUTION_MODE=synthetic` and explicitly mark their output as
`local_synthetic_not_two_gpu_qualification`.

Each real Task2 run must publish `metrics.json` and `metrics.md` with these exact
audit fields:

```text
`task2_run_all_elapsed_seconds`
`dataset_row_count`
`validation_mse_by_fold`
`average_validation_mse`
`test_mse`
`model_reload_max_abs_prediction_delta`
`scaler_feature_count`
`scaler_mean_count`
`scaler_scale_count`
`scaler_nonzero_scale_count`
`prediction_sample`
```

The five fold values must be finite and nonnegative, and their arithmetic mean
must equal `average_validation_mse`. `test_mse` is reported separately. The
reload delta proves that two independent loads give the same deterministic
sample prediction; the scaler counts bind the model to the dataset feature
schema. These checks establish a reusable numeric contract, not accuracy by
assertion: the real dataset values and prediction sample remain subject to the
reviewed exact-two-H800 qualification.

### Task3: timeline simulation

Task3 requires an explicit artifact source. The rule is **no automatic fresh-to-prebaked fallback**:

```bash
ARTIFACT_SOURCE=fresh \
  bash SC26-AE/task3_qwen3_a30b.sh

ARTIFACT_SOURCE=prebaked \
PREBAKED_ROOT="$PWD/SC26-AE/prebaked" \
SIMULATOR_HARDWARE_TYPE=cpu \
  bash SC26-AE/task3_qwen3_a30b.sh
```

The same two commands apply to `task3_gpt175b.sh` and `task3_dsv3.sh`. A fresh
run validates the Task1 capture marker, the independent Task2 predictor marker,
source commits, topology, profile, checksums, and slowdown inputs before building
assets. A prebaked run validates `PREBAKED_ROOT`, the shared predictor, the model
bundle, and the outer distribution manifest without inspecting fresh output.

The wrapper always passes these values explicitly to the canonical sim-engine:

```text
--cc-backend analytical
--overlap-mode on
--local-size 8
--database-dir <same resolved path as --trace-dir>
--slowdown-model-path <verified model path>
--slowdown-scaler-path <verified scaler path>
--report-output-dir <new simulation run>
```

The primary report field is `rank0_step_time_ms`, the final rank0 timeline span.
The report also contains exact-name scheduled sums for `forward_step`,
`backward_step`, and `optimizer_step`, plus simulator load, execution, and
wall-clock times. Overlap means the three operation sums are not required to equal
the final span; `comp+comm` is diagnostic-only.

The JSON report uses these exact field names (consumers must not infer aliases):

```text
`rank0_step_time_ms`
`rank0_forward_step_duration_sum_ms`
`rank0_backward_step_duration_sum_ms`
`rank0_optimizer_step_duration_sum_ms`
`rank0_comp_plus_comm_diagnostic_ms`
`simulator_load_time_s`
`simulator_execution_time_s`
`simulator_wall_clock_s`
```

Communication is intentionally weak-validated in this AE path. The canonical
backend is `analytical` with overlap enabled. `collective-sim` is optional background infrastructure
only after its public dependency and pinned commit have been independently
verified; it is neither an implicit fallback nor required to support the primary
AE result.

### Runtime and release-source evidence

Real evaluator-time estimates are not inferred from local synthetic fixtures.
They remain pending until the reviewed image and required H800 topology produce
fresh measurements:

```text
REAL_RUNTIME_EVIDENCE_STATUS=PENDING_H800_QUALIFICATION
```

| Surface | Current release decision | Measurement required before publication |
|---------|--------------------------|-----------------------------------------|
| Setup | `PENDING_CLEAN_IMAGE_QUALIFICATION` | Clean-container setup elapsed time plus immutable image digest |
| GPT-175B Task1 | `PENDING_H800_QUALIFICATION` | Single-rank probe and complete selected-rank capture elapsed time |
| Qwen3-A30B Task1 | `PENDING_D16_7200_SECOND_GATE` | Rank-0 probe, rank0×256 estimate, and full-capture result when the estimate is at most 7200 seconds |
| DeepSeek-V3 Task1 | `PENDING_D16_7200_SECOND_GATE` | Rank-0 probe, rank0×256 estimate, and full-capture result when the estimate is at most 7200 seconds |
| Shared Task2 | `PENDING_EXACT_TWO_H800_RUN` | Collection, training, save/reload, and prediction elapsed time on exactly two H800 GPUs |
| Task3 | `PENDING_REAL_INPUT_BUNDLES` | Per-model simulator load, execution, wall-clock, and peak RSS from qualified inputs |

If either MoE estimate exceeds 7200 seconds, the documented reviewer path uses
the complete, independently qualified prebaked source. The runner never changes
source because a timing gate fails. Until these rows are replaced by measured
values, the README makes no setup-duration, task-duration, CPU-memory-minimum, or
fresh-full-capture support claim.

The D16 7200-second gate applies only to Qwen3-A30B and DeepSeek-V3, whose Task1
topology has 256 fake ranks. GPT-175B captures eight representative PP-stage
ranks and therefore records `d16_gate_applicable=false`, an eight-rank diagnostic
estimate, and no D16 threshold/result fields. For each MoE invocation, the wrapper
first runs an isolated rank-0 preflight and records
`d16_timing_source=independent_rank0_probe`. With `QUICK=1`, the probe is an
observation only (`d16_gate_enforced=false`, `gate_decision_applied=false`): even
an estimate above 7200 seconds does not fail the four-rank smoke path. With
`QUICK=0` (full), the same independently measured value is multiplied by 256 and
the gate is enforced (`d16_gate_enforced=true`, `gate_decision_applied=true`)
before the complete 256-rank selected capture is started. An above-threshold full
probe fails closed without creating a full run root, starting any selected-rank
loop, publishing a marker, or switching to another source. Synthetic/controller
output must not be described as H800 qualification.

### Complete public entry matrix

The nine public entries are deliberately listed here rather than hidden behind a
dispatcher. Each line is an executable command shape; set the documented output,
hardware, and artifact-source variables before running it.

```bash
# Task1 — one physical GPU, sequential fake ranks
SCALE_GPU=0 bash SC26-AE/task1_gpt175b.sh
SCALE_GPU=0 bash SC26-AE/task1_qwen3_a30b.sh
SCALE_GPU=0 bash SC26-AE/task1_dsv3.sh

# Task2 — real mode requires exactly two distinct visible GPUs
CUDA_VISIBLE_DEVICES=0,1 REBUILD=1 bash SC26-AE/task2_gpt175b.sh
CUDA_VISIBLE_DEVICES=0,1 bash SC26-AE/task2_qwen3_a30b.sh
CUDA_VISIBLE_DEVICES=0,1 bash SC26-AE/task2_dsv3.sh

# Task3 — source selection is explicit; no automatic fallback
ARTIFACT_SOURCE=fresh bash SC26-AE/task3_gpt175b.sh
ARTIFACT_SOURCE=fresh bash SC26-AE/task3_qwen3_a30b.sh
ARTIFACT_SOURCE=fresh bash SC26-AE/task3_dsv3.sh
```

For a prebaked run, replace the corresponding Task3 line with
`ARTIFACT_SOURCE=prebaked PREBAKED_ROOT=<verified-root> bash
SC26-AE/task3_<model>.sh` and set `SIMULATOR_HARDWARE_TYPE=cpu` as shown above.
The selected source must be complete and verified before the command starts;
the wrapper never changes source or downloads artifacts on the operator's behalf.

## Output and provenance

The stable tree is:

```text
SC26-AE/output/
  _work/
  _shared/task2/runs/<predictor_run_id>/
  gpt175b/{task1,task2,task3}/
  qwen3_a30b/{task1,task2,task3}/
  dsv3/{task1,task2,task3}/
```

Task1 and Task3 runs use fresh identities and are never overwritten. Manifests use
root-relative POSIX paths and record file size plus SHA256. A Task3/prebaked
manifest records both the independent `capture_id` and `predictor_run_id`; they
must not be made equal merely for convenience.

Verify an artifact bundle directly with:

```bash
python SC26-AE/tools/artifact_manifest.py verify \
  --root <bundle-root> \
  --manifest <bundle-root>/artifact_manifest.json
```

The distribution size decision is measured over the complete staged tree:

```bash
python SC26-AE/tools/artifact_manifest.py size-gate \
  --root <staged-distribution-root>
```

Only the measured `regular_git` or `github_release` result is documented as the
canonical distribution path. A release asset, if required by the measured gate,
must be fetched explicitly and verified before it is passed as `PREBAKED_ROOT`.

No real three-model distribution has passed that gate yet:

```text
CANONICAL_PREBAKED_DISTRIBUTION=NOT_SELECTED
release_pre_dataset=NOT_QUALIFIED
```

Accordingly, `SC26-AE/prebaked` in the command example is an expected verified
root, not a claim that this worktree currently ships qualified data. After the
real staged tree is sealed and measured, this section must be replaced with
exactly one path: either the checked-in `regular_git` root, or an explicit
versioned Release fetch command plus its asset SHA256 and a separate
`PREBAKED_ROOT=<verified-root> ARTIFACT_SOURCE=prebaked ...` invocation. Task3
must never download or select that path automatically.

After all three models have independently completed the real Task1→Task2→Task3
qualification chain, seal a reusable distribution with the strict packager:

```bash
python SC26-AE/tools/package_prebaked.py build \
  --repo-root "$PWD" \
  --output-root <fresh-output-root> \
  --staging-root <new-prebaked-root> \
  --distribution-id <immutable-distribution-id> \
  --result-json <result-json-outside-staging-root>

python SC26-AE/tools/package_prebaked.py verify \
  --repo-root "$PWD" \
  --prebaked-root <verified-prebaked-root>
```

The packager is a sealing step, not an evidence-promotion step. It requires
`real_single_h800_qualified` for every Task1 source and
`real_exact_two_h800_qualified` for the shared Task2 source; a local/synthetic
fixture is rejected by the production CLI. It copies evidence classes without
upgrading them, records per-file bytes/MiB/SHA256 plus nested manifest hashes, and
fails if the destination already exists. The unit suite may use an explicit
in-process contract seam to exercise copy/inventory mechanics with synthetic
labels; that seam is not exposed by the AE command and cannot qualify a release.

The qualification sealer treats `qualification_metrics.json` as an integrity-bound
issuer payload rather than a universal task-semantics parser. It requires the
metrics path to resolve to a regular, non-empty file, verifies its SHA256 before
and after copying, and stores the exact bytes under sealed `provenance/`. The
external qualification producer is responsible for the stable Task1/Task2/Task3
metrics schema and acceptance thresholds. An opaque metrics payload therefore
cannot bypass the external issuer's semantic gate, and adding task-specific
checks to this wrapper requires a separately reviewed, versioned schema contract.

## Local validation

The following commands exercise local contracts; their output must remain labelled
non-qualification evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  tests/unit/test_sc26_ae_artifact_manifest.py \
  tests/unit/test_sc26_ae_echo_metrics.py \
  tests/unit/test_sc26_ae_package_prebaked.py
bash tests/unit/test_sc26_ae_common.sh
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/integration/test_sc26_ae_task2_contract.sh
bash tests/e2e/test_sc26_ae_task2_smoke.sh
```

The sim-engine report contract is tested from both the repository root and the
sim-engine directory so that imports do not depend on the caller's current working
directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py \
  megatron-sim-engine/tests/unit/test_rank0_report.py \
  megatron-sim-engine/tests/integration/test_rank0_report_integration.py
```

## Fail-fast troubleshooting

Troubleshoot the recorded root cause; do not bypass the failed contract or change
artifact source as a recovery mechanism.

| Failure symptom | Root cause to verify | Required action |
|-----------------|----------------------|-----------------|
| Setup rejects Python, torch/CUDA, NVML, Nsight, or `SlowdownPredictor` | The worker is not the reviewed fixed runtime or the image contract has drifted | Preserve the verifier output and qualify the fixed `v1.2-ae` image/runtime; do not switch interpreters or tools |
| Task1 rejects GPU selection | More than one GPU is visible and `SCALE_GPU` is absent, or the selected ID is invalid | Start a new run with exactly one selected physical GPU; do not emulate the physical-GPU check |
| Task2 rejects CUDA visibility | The allocation does not expose exactly two distinct decimal GPU IDs | Obtain an exact-two-GPU allocation and rerun with `CUDA_VISIBLE_DEVICES=<id0>,<id1>`; one GPU is not a substitute |
| Fresh or prebaked Task3 validation fails | The explicitly selected bundle is missing, partial, stale, mixed, or inconsistent with its manifest | Preserve the evidence and repair or reproduce that selected source; never retry through the other source |
| Manifest size or SHA256 differs | Payload bytes changed after sealing or the wrong root was supplied | Start from the correct immutable producer output and rebuild a new destination; do not edit the manifest to match |
| Task3 has no valid rank0 span or target operation | The schedule/report input is incomplete or semantically invalid | Fix the producer or scheduler root cause and rerun; never substitute the diagnostic `comp+comm` value |
| Output destination already exists | The requested run identity or staging root is not fresh | Choose a new output/run identity; never overwrite a prior result |
| Evidence class is local/synthetic | Only contract wiring has been exercised | Keep qualification and release gates closed until the required real run produces stronger evidence |

## Fail-fast rules

Do not treat a local synthetic PASS as real qualification. Missing files,
partial/extra files, checksum or provenance drift, dirty source, incompatible
topology, invalid metrics, missing report operations, unwritable output, and
unavailable required hardware all fail with evidence. The workflow does not add a
fallback, hidden retry, source switch, interpreter switch, calibration factor, or
silent rebuild.
