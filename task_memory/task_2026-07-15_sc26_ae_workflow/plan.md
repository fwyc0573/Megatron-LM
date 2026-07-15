# SC'26 Artifact Evaluation Workflow Implementation Plan

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-16 | Recorded explicit user approval of Gate A and opened Phase 0 while retaining all downstream gates |
| 2026-07-15 | Integrated the independent Claude WATCH verdict: resolved I16 without MoE source edits, froze AE model-size labels, added canonical optimizer integration evidence, and guarded analytical 8-GPU topology coupling |
| 2026-07-15 | Author self-review isolated versioned Task1/Task3 runs, fixed Task3 database binding, hardened prebaked provenance/distribution/publication gates, and replaced assumed CPU RAM with measured qualification |
| 2026-07-15 | Corrected the Gate B Qwen rank-selection variable and replaced its timestamp placeholder with a reproducible UTC `RUN_ID` command |
| 2026-07-15 | Restored the R9-required pre-implementation three-task runtime reconnaissance as Gate B before any AE infrastructure or wrapper edit |
| 2026-07-15 | Author self-review separated trace `capture_id` from Task2 `predictor_run_id`, closed the D16 prebaked-generation gap, defined manifest self-file handling, froze dense `NUM_EXPERTS=1`, and clarified report rounding |
| 2026-07-15 | Froze Task3 `LOCAL_SIZE=8` from sim-engine/analytical-backend facts and added I16 as an independent-review WATCH for the MoE Task1 fake-node-size discrepancy |
| 2026-07-15 | Replaced the initial D1–D15 outline with an implementation-ready, review-gated plan covering R1–R15, D1–D23, I1–I15, exact interfaces, TDD steps, provenance, packaging, and validation |
| 2026-07-15 | Initial plan after the first grilling session |

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` (recommended for independent bounded tasks) or `executing-plans` to implement this plan task-by-task. Every implementation task follows RED → observe FAIL → GREEN → observe PASS. Do not execute any implementation task until Gate A has explicit user approval.

**Goal:** Deliver a reproducible SC'26 AE workflow with exactly nine per-model task entry scripts, centralized versioned artifacts, trace-compatible slowdown provenance, an overlap-aware rank0 simulator report, explicit fresh/prebaked source selection, and reviewer-facing documentation.

**Architecture:** Keep orchestration in the main repository under `SC26-AE/`, reuse the three existing model scripts as Task1 compute/tracing sources, execute Echo-slowdown from an isolated `git archive` snapshot for Task2, and use only the sim-engine built-in scheduler plus simulator for Task3. Task1 and Task3 write immutable versioned runs and expose only verified markers. A portable outer manifest ties every artifact to checksums, producer commits, simulation topology, capture runtime, profile, capture identity, and explicit artifact source; selected-source failures stop immediately rather than switching paths.

**Tech Stack:** Bash with `set -euo pipefail`, Python 3.9, PyTorch/Megatron-LM, Nsight Systems (`nsys`), Nsight Compute (`ncu`), XGBoost, JSON/Markdown reports, pytest, shell integration tests, Git submodules, Docker image `hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef`, and the internal `rlaunch` H800 platform.

---

## 1. Current Status and Hard Stop

| Gate / Phase | Status | Entry condition | Exit condition |
|--------------|--------|-----------------|----------------|
| **Gate A — enhanced plan and independent review** | **APPROVED 2026-07-16** | R1–R15 and D1–D23 captured | Author self-review, StepCode Claude review, document validation, and explicit user approval |
| Phase 0 — safety baseline and isolated worktree | **COMPLETED 2026-07-16** | Gate A approved | Protected baseline committed and branches/worktree ready |
| **Gate B — existing Task1/Task2/Task3 runtime reconnaissance** | **IN PROGRESS** | Phase 0 complete; no feature edit started | Existing three-task chain executed in the AE image, interfaces/evidence recorded, plan delta reviewed |
| Phase 1 — shared AE infrastructure | BLOCKED | Gate B complete | Setup, common shell contracts, and manifest helper tested |
| Phase 2 — Task1 tracing and atomic capture | BLOCKED | Phase 1 complete | Three Task1 entries and provenance outputs verified |
| Phase 3 — Task2 isolated slowdown workflow | BLOCKED | Phase 1 complete | Shared predictor bundle and numeric evidence verified |
| Phase 4 — canonical scheduler and rank0 reporter | BLOCKED | Phases 1–3 interfaces frozen | Scheduler/reporter tests pass in sim-engine |
| Phase 5 — Task3 end-to-end wrappers | BLOCKED | Phases 2–4 complete | Fresh and prebaked explicit-source paths verified |
| Phase 6 — prebaked packaging | BLOCKED | Real artifact sizes measured | Exactly one D21 distribution path selected and verified |
| Phase 7 — AE documentation and paper suggestions | BLOCKED | Runtime commands and outputs stable | README and tex suggestions match evidence |
| Phase 8 — GPU dry-run and clean-clone rehearsal | BLOCKED | Phases 1–7 locally verified | Nine entries rehearsed with recorded metrics |
| Phase 9 — final review, evidence, and archive | BLOCKED | Phase 8 complete | Tests/reviews complete; summary and lessons archived |

**Stop rule for this session:** finish Gate A only. Do not modify implementation files, run GPU jobs, commit, create/check out branches, create worktrees, push, or change submodules.

---

## 2. Global Constraints

1. **Exactly nine public task entries; no dispatcher:**
   - `SC26-AE/task1_gpt175b.sh`
   - `SC26-AE/task1_qwen3_a30b.sh`
   - `SC26-AE/task1_dsv3.sh`
   - `SC26-AE/task2_gpt175b.sh`
   - `SC26-AE/task2_qwen3_a30b.sh`
   - `SC26-AE/task2_dsv3.sh`
   - `SC26-AE/task3_gpt175b.sh`
   - `SC26-AE/task3_qwen3_a30b.sh`
   - `SC26-AE/task3_dsv3.sh`
2. **Explicit source selection:** Task3 requires `ARTIFACT_SOURCE=fresh` or `ARTIFACT_SOURCE=prebaked`. Missing, partial, corrupt, checksum-mismatched, or provenance-mismatched inputs fail immediately. No fresh-to-prebaked switching is permitted.
3. **Root-cause fixes only:** do not add calibration factors, hidden recovery paths, silent rebuilds, silent source switching, or output guessing.
4. **Precision and workload:** all three models use bf16, mock-data, and DDP gradient overlap. Task3 explicitly passes `--overlap-mode on`.
5. **Communication scope:** the canonical AE backend is explicitly `--cc-backend analytical`. `collective-sim` is documentation-only and not an acceptance dependency.
6. **Canonical scheduler:** only `megatron-sim-engine/src/scheduler/mg_scheduling/` is used at runtime. Root `mg_scheduling/` remains legacy/reference and is not kept equivalent.
7. **Echo-slowdown isolation:** do not execute collection/training inside the pinned submodule checkout. Build every run from `git -C Echo-slowdown archive <gitlink_commit>` under `SC26-AE/output/_work/`.
8. **Submodule policy:** Echo-slowdown remains at its existing upstream URL and pinned commit with no AE-required internal commit. Sim-engine changes live on a public `sc26-ae` branch and the main-repo gitlink must point to its reachable commit.
9. **Repository safety:** no `rm` or `mv`; no bulk replacement; do not touch worktree branch `task/ddp-overlap-comprehensive-review-20260713`; protect all pre-existing local modifications.
10. **Environment:** before GPU work, read `/data/ycfeng/stepfun-env-handbook/guidence.md`; use `--charged-group=codesign --private-machine=group --positive-tags=h800 --backoff-limit=1`, and run `--predict-only` before material allocations.
11. **Setup source:** `GROUPED_GEMM_SOURCE` must be explicitly `vcs` or `archive`. The README recommends `archive` because its two source archives have pinned SHA256 values. A selected-source failure is final.
12. **Testing:** every logic change starts with an observed failing unit test and ends with targeted tests plus affected integration/e2e regression. Numeric evidence is recorded rather than only asserting file existence.
13. **Output locality:** runtime output goes under `SC26-AE/output/`; temporary reconnaissance, capture probes, Task2 snapshots, and packaging staging roots go under versioned `SC26-AE/output/_work/` paths; no temporary document is created in the repository root.
14. **Documentation:** code/comments use formal English; `SC26-AE/README.md` and final `summary.md` are English; task-management discussion may use Chinese.
15. **Git history:** future commits follow the Lore commit protocol and occur only after relevant tests pass. No commit is made during Gate A.
16. **Virtual node topology:** all Task3 scheduler/simulator invocations fix `LOCAL_SIZE=8`, matching the H800 platform, sim-engine hardware presets, and canonical analytical backend's 8-GPU-per-node model. The value is serialized in every outer topology manifest and is never inherited from a CLI default.
17. **Run isolation:** Task1 and Task3 never write into a prior run directory. Every run ID is generated once, its destination must not exist, native CWD-relative outputs remain inside that run, and a model-level marker is published only after manifest verification. Failed or partial runs remain unverified and are never reused.
18. **External publication gate:** local commits may proceed only after their phase gates, but every `git push`, default-branch change, GitHub Release creation/upload, or other external publication requires a separate explicit user approval naming the remote URL, branch/tag, commit SHA, visibility, and asset list. D1/D2 define the intended destination but do not waive this final side-effect approval.

---

## 3. Scope and Non-Goals

### In scope

- The nine task entry scripts and small shared helpers.
- Setup validation and explicit grouped-gemm source selection.
- Task1 execution graph, memory, Nsight capture, summary, and provenance collection.
- Task2 isolated Echo-slowdown collection/training, predictor bundle, metrics, and markers.
- Sim-engine bf16 schedule generation with explicit output placement.
- Sim-engine overlap-aware rank0 JSON/Markdown reporter.
- Task3 fresh/prebaked validation, slowdown-assets build/use, and simulation.
- Deterministic regular-Git-versus-GitHub-Release packaging gate.
- AE README, paper-change suggestions, clean-clone rehearsal, and evidence reports.

### Out of scope

- Simulator-versus-ground-truth accuracy calibration.
- Refactoring tracer, simulator, Echo-slowdown, or communication modules beyond a confirmed chain blocker.
- Validating the paper's communication methodology; `analytical` is a weak-validation backend for this AE.
- Changing `task/ddp-overlap-comprehensive-review-20260713` or merging its work.
- Maintaining behavioral equivalence between the two `mg_scheduling` copies.
- Automatic runtime fallback of any kind.

---

## 4. Frozen Model and Workload Matrix

The wrapper exports every value explicitly; it does not depend on the source script's defaults.

| Field | GPT-175B | Qwen3-A30B | DeepSeek-V3 variant |
|-------|----------|------------|---------------------|
| Public model key | `gpt175b` | `qwen3_a30b` | `dsv3` |
| Task1 source | `examples/update_pretrain_gpt.sh` | `examples/pretrain_qwen3_30b_a3b_moe.sh` | `examples/pretrain_deepseek_v3_moe.sh` |
| `MODEL_SIZE` / `MODEL_PROFILE` | `175` | `full` | `smoke` |
| World size | 1024 | 256 | 256 |
| Task1 `capture_runtime.fake_gpus_per_node` / Task3 `simulation_topology.local_size` | 8 / 8 | 256 / 8 | 256 / 8 |
| PP / TP / DP / EXP | 8 / 8 / 16 / 1 | 4 / 8 / 8 / 8 | 4 / 8 / 8 / 8 |
| Layers / hidden | 96 / 12288 | 48 / 2048 | 32 / 2048 |
| Attention | 96 heads | 32 heads, 4 query groups | MLA, 64 heads |
| Experts / top-k | 1 / not applicable (dense) | 128 / 8 | 32 / 2 |
| Sequence length | 2048 | 256 | 256 |
| Micro-batch size | 1 | 1 | 1 |
| Micro-batches per step | 48 (`6 × PP`) | 16 (`4 × PP`) | 16 (`4 × PP`) |
| Global batch size | 768 (`48 × 1 × DP`) | 128 (`16 × 1 × DP`) | 128 (`16 × 1 × DP`) |
| Precision / data | bf16 / mock-data | bf16 / mock-data | bf16 / mock-data |
| DDP overlap | enabled | enabled | enabled |
| Full Task1 ranks | `0,128,256,384,512,640,768,896` | `0..255` | `0..255` |
| `QUICK=1` ranks | same 8 stage representatives | `0,64,128,192` | `0,64,128,192` |

### Rank-scope rules

- GPT uses one rank per PP stage, fixing TP rank 0 and DP rank 0.
- MoE defaults remain full 256 ranks until I1 is proven with a complete Task3 run. `QUICK=1` is only a smoke path unless that proof succeeds.
- Task3 always models 8 GPUs per node. GPT Task1 currently passes `--fake-gpus-per-node 8`; the two MoE Task1 source scripts pass their full fake world size. These are separate manifest fields: `simulation_topology.local_size=8` describes Task3, while `capture_runtime.fake_gpus_per_node` records the exact Task1 argv without rewriting history. Independent StepCode Claude review adjudicated I16: trace files do not serialize tracer `server_id`/`local_rank`, while sim-engine reconstructs those values from Task3 `--local-size`; therefore no MoE source-script change is planned. Gate B still verifies this consumed-artifact path at runtime.
- Release qualification times fake rank 0 separately for each MoE model and records:

```text
estimated_full_seconds = single_rank_elapsed_seconds × 256
fresh_capture_gate = estimated_full_seconds <= 7200
```

- The estimate basis is recorded as `estimate_basis_rank=0`. Because PP stages may differ, this formula is a user-selected gate rather than a claim of exact wall-clock prediction; measured full-capture time replaces the estimate when a full capture is performed.
- If the estimate exceeds 7200 seconds, the release path does not attempt a full fresh Nsight capture and Task3 validation uses a complete prebaked bundle. This is a release-time decision recorded in the manifest/README, not a runtime fallback.

---

## 5. Provenance and Artifact Contracts

### 5.1 Atomic fresh provenance

A fresh slowdown bundle is valid only when both provenance identities below are explicit and verified:

1. One `capture_id` binds the Task1 trace directory, Task1 `.nsys-rep`, SQLite exported from that `.nsys-rep`, and the trace-specific slowdown assets generated from them.
2. One independent `predictor_run_id` binds Task2 `kernel_metric_output.csv`, `xgb_model.json`, and `standard_scaler.json`; those files are referenced by exact SHA256 values and are not relabeled as if they came from the Task1 Nsight capture.

The slowdown assets generated from the two verified input sets are:
   - `manifest.json`
   - `kernel_features.json`
   - `backward_kernel_blueprints.json`

The Task3 outer manifest records both `capture_id` and `predictor_run_id`. A capture mismatch, predictor-run mismatch, or checksum mismatch is a hard error. Only trace-derived files share `capture_id`; Task2 predictor artifacts retain their own run identity.

Each Task1 invocation generates `capture_id` exactly once and requires a new path at `output/<model>/task1/runs/<capture_id>/`. The model source script executes with `runs/<capture_id>/runtime/` as its working directory, so native relative outputs such as `profiler_log/`, `memory_traces_scaling/`, and the replay cache cannot mix with older captures. The runner inventories only that run root, verifies its manifest, and then writes `output/<model>/task1/capture_marker.json`. An existing destination, a marker that points outside `runs/`, or any unverified/partial run is a hard error; no run directory is deleted or overwritten.

The Nsight capture boundary is one `nsys profile` invocation around the selected-rank loop:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --wait=all \
  --trace-fork-before-exec=true \
  --force-overwrite true \
  --output "${NSYS_BASE}" \
  bash "${SELECTED_RANK_LOOP}"

nsys export \
  -t sqlite \
  --force-overwrite true \
  -o "${NSYS_BASE}.sqlite" \
  "${NSYS_BASE}.nsys-rep"
```

A single-rank timing probe has its own capture ID and is never mixed into the full bundle.

### 5.2 Portable outer manifest

`SC26-AE/tools/artifact_manifest.py` owns schema `sc26-ae-artifact-manifest-v1`:

```json
{
  "schema_version": "sc26-ae-artifact-manifest-v1",
  "model": "qwen3_a30b",
  "task": "task1",
  "artifact_source": "fresh",
  "capture_id": "qwen3_a30b-<UTC timestamp>",
  "source_commits": {
    "megatron_lm": "<40-hex commit>",
    "echo_slowdown": "<40-hex commit>",
    "megatron_sim_engine": "<40-hex commit>"
  },
  "simulation_topology": {"world_size": 256, "local_size": 8, "pp": 4, "tp": 8, "dp": 8, "exp": 8},
  "capture_runtime": {"physical_gpu_count": 1, "fake_gpus_per_node": 256},
  "profile": "full",
  "precision": "bf16",
  "mock_data": true,
  "ddp_overlap": true,
  "files": [
    {"path": "runtime/profiler_log/example.txt", "size_bytes": 123, "sha256": "<64-hex digest>"}
  ]
}
```

Angle-bracket values above describe generated data, not unresolved design choices. Runtime manifests contain concrete values only.

Task1 manifests require `capture_id` and omit `predictor_run_id`. Shared Task2 manifests require `predictor_run_id` and omit `capture_id`. Task3 and complete prebaked model manifests require both fields, thereby preserving the independent origin of trace-specific and predictor-specific inputs.

Commit validation is source-aware rather than circular:

- A `fresh` Task3 run requires the selected Task1 marker and manifest to come from the same current producer checkout and verifies its recorded main-repository and sim-engine gitlink commits against that checkout. Release-qualified fresh captures additionally require recorded clean source states.
- A `prebaked` Task3 run does **not** require the historical producer commit to equal the consumer repository's final `HEAD`; committing the prebaked payload necessarily changes that `HEAD`. Instead, `distribution_manifest.json` records each producer commit, nested manifest SHA256, payload SHA256, model/profile/topology, `capture_id`, and `predictor_run_id`, and verification checks those records for internal consistency. The pinned sim-engine/Echo commits must still match the consumer's declared compatible commits. No field attempts to predict or hash the future publication commit.

Paths are POSIX-style and relative to the declared bundle root. Absolute paths and `..` components are rejected. The builder's internal source-path fields remain informational and are never used to locate runtime model/scaler files. Task3 always passes explicit bundle-resolved `--slowdown-model-path` and `--slowdown-scaler-path`.

The `files` array lists payload files only. `artifact_manifest.json` is the one permitted unlisted regular file at the bundle root because a manifest cannot hash itself; any other undeclared regular file is rejected. `metadata.json` and the newline-delimited creation file list are generated under the versioned `_work/` directory, outside the bundle root, and retained as provenance rather than deleted.

### 5.3 Manifest Python interface

```python
def sha256_file(path: pathlib.Path) -> str: ...
def create_manifest(root: pathlib.Path, metadata: dict, relative_files: list[str]) -> dict: ...
def verify_manifest(root: pathlib.Path, manifest: dict) -> None: ...
def evaluate_distribution_gate(
    distribution_root: pathlib.Path,
    per_file_limit_bytes: int = 52_428_800,
    bundle_limit_bytes: int = 524_288_000,
) -> str: ...
```

`evaluate_distribution_gate()` recursively measures every regular file in the staged regular-Git candidate, including all payloads, nested `artifact_manifest.json` files, and `distribution_manifest.json`. Symlinks and special files fail. It returns exactly `regular_git` or `github_release`. A file of 52,428,799 bytes passes the per-file gate; 52,428,800 bytes fails it. A total of 524,288,000 bytes passes the bundle gate; 524,288,001 bytes fails it.

CLI:

```bash
python SC26-AE/tools/artifact_manifest.py create \
  --root <bundle-root> \
  --metadata-json <metadata.json> \
  --file-list <newline-delimited-relative-files.txt> \
  --output <bundle-root>/artifact_manifest.json

python SC26-AE/tools/artifact_manifest.py verify \
  --root <bundle-root> \
  --manifest <bundle-root>/artifact_manifest.json

python SC26-AE/tools/artifact_manifest.py size-gate \
  --root <staged-regular-git-distribution-root>
```

The actual implementation command substitutes concrete paths produced by the running wrapper. Exit code 0 means verified; invalid schema, missing/extra file, size mismatch, checksum mismatch, unsafe path, simulation-topology/capture-runtime/profile/source mismatch, or partial bundle returns nonzero with an explicit error.

### 5.4 Output tree

```text
SC26-AE/
├── README.md
├── setup.sh
├── lib/
│   ├── common.sh
│   ├── task1_trace.sh
│   ├── task2_echo.sh
│   └── task3_simulation.sh
├── tools/
│   ├── artifact_manifest.py
│   └── echo_metrics.py
├── task1_gpt175b.sh
├── task1_qwen3_a30b.sh
├── task1_dsv3.sh
├── task2_gpt175b.sh
├── task2_qwen3_a30b.sh
├── task2_dsv3.sh
├── task3_gpt175b.sh
├── task3_qwen3_a30b.sh
├── task3_dsv3.sh
├── prebaked/
│   ├── distribution_manifest.json
│   ├── gpt175b/
│   ├── qwen3_a30b/
│   ├── dsv3/
│   └── shared_task2/
└── output/
    ├── _work/
    ├── _downloads/<distribution_id>/prebaked/
    ├── _shared/task2/runs/<run_id>/
    ├── gpt175b/
    │   ├── task1/
    │   │   ├── capture_marker.json
    │   │   └── runs/<capture_id>/
    │   │       ├── runtime/profiler_log/
    │   │       ├── runtime/memory_traces_scaling/
    │   │       ├── nsys/
    │   │       ├── logs/summary.log
    │   │       └── artifact_manifest.json
    │   ├── task2/predictor_marker.json
    │   └── task3/
    │       ├── run_marker.json
    │       └── runs/<simulation_run_id>/
    │           ├── schedule/
    │           ├── slowdown_assets/
    │           ├── report.json
    │           ├── report.md
    │           ├── logs/
    │           └── artifact_manifest.json
    ├── qwen3_a30b/{task1,task2,task3}/
    └── dsv3/{task1,task2,task3}/
```

Versioned Task1, Task2, and Task3 run directories are never deleted or reused. Markers contain a relative run path plus the verified manifest SHA256 and are updated only after the new bundle passes all checks. `REBUILD=1` applies only to Task2 and creates a new predictor run. Task3 creates a new `simulation_run_id` on every explicit invocation, so fresh and prebaked results cannot overwrite or contaminate each other.

---

## 6. Exact File Map

### Main repository — modify

| File | Responsibility of future change |
|------|---------------------------------|
| `.gitignore` | Ignore runtime `SC26-AE/output/`; allow the selected prebaked distribution manifest and either regular-Git bundle files or Release locator files after D21 gate |
| `examples/update_pretrain_gpt.sh` | Add explicit env-controlled bf16/fp16 selection, DDP overlap, trace-memory, kernel-ground-truth phase, train/trace knobs, and AE-safe paths while preserving existing non-AE defaults |
| `tools/ae/setup_grouped_gemm_v1.sh` | Replace automatic VCS→archive recovery with required `GROUPED_GEMM_SOURCE=vcs|archive` selection |
| `docs/ae/grouped_gemm_v1_setup.md` | Replace automatic-recovery instructions/evidence wording with the explicit selected-source contract and commands |
| `tests/unit/test_setup_grouped_gemm_v1.sh` | Replace recovery expectations with explicit-source success/failure cases |
| `tests/integration/test_gpt_example_mock_mode.sh` | Cover GPT-175B AE arguments, representative ranks, bf16, mock-data, DDP overlap, trace memory, and output controls |
| `megatron-sim-engine` gitlink | Point to the publicly reachable sim-engine `sc26-ae` commit after sim-engine tests pass |

### Main repository — create

```text
SC26-AE/README.md
SC26-AE/setup.sh
SC26-AE/lib/common.sh
SC26-AE/lib/task1_trace.sh
SC26-AE/lib/task2_echo.sh
SC26-AE/lib/task3_simulation.sh
SC26-AE/tools/artifact_manifest.py
SC26-AE/tools/echo_metrics.py
SC26-AE/task1_gpt175b.sh
SC26-AE/task1_qwen3_a30b.sh
SC26-AE/task1_dsv3.sh
SC26-AE/task2_gpt175b.sh
SC26-AE/task2_qwen3_a30b.sh
SC26-AE/task2_dsv3.sh
SC26-AE/task3_gpt175b.sh
SC26-AE/task3_qwen3_a30b.sh
SC26-AE/task3_dsv3.sh
tests/unit/test_sc26_ae_common.sh
tests/unit/test_sc26_ae_artifact_manifest.py
tests/unit/test_sc26_ae_echo_metrics.py
tests/unit/test_sc26_ae_task2_snapshot.sh
tests/unit/test_sc26_ae_task3_contracts.sh
tests/integration/test_sc26_ae_setup.sh
tests/integration/test_sc26_ae_task1_contracts.sh
tests/integration/test_sc26_ae_task2_contract.sh
tests/integration/test_sc26_ae_task3_contract.sh
tests/e2e/test_sc26_ae_task1_smoke.sh
tests/e2e/test_sc26_ae_task2_smoke.sh
tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh
tests/e2e/test_sc26_ae_fresh_chain.sh
task_memory/task_2026-07-15_sc26_ae_workflow/tex_change_suggestions.md
task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-15_sc26_ae_workflow.md
```

`SC26-AE/lib/fetch_prebaked.sh`, `tests/unit/test_sc26_ae_fetch_prebaked.sh`, and Release locator metadata are created only when the measured D21 result is `github_release`; regular-Git packaging does not carry an unused downloader.

### Sim-engine submodule — modify/create

| File | Action |
|------|--------|
| `megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py` | Modify: mutually exclusive `--fp16`/`--bf16`; add explicit `--output-dir` |
| `megatron-sim-engine/src/scheduler/mg_scheduling/mg_scheduling_plan.py` | Modify: bf16 pipeline dtype, deterministic AE output filenames, explicit output directory |
| `megatron-sim-engine/simu_main.py` | Modify: report CLI fields, pure rank0 aggregation/validation, JSON/Markdown write, extended return object |
| `megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py` | Create: three-model schedule contract |
| `megatron-sim-engine/tests/unit/test_rank0_report.py` | Create: report calculations and fail-fast branches |
| `megatron-sim-engine/tests/integration/test_rank0_report_integration.py` | Create: tiny full simulation produces consistent JSON/Markdown |

No change is planned for `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`: the outer portable manifest and explicit model/scaler CLI overrides solve I13 without expanding builder scope.

---

## 7. Shared Public Interfaces

### 7.1 Entry-script environment

All nine entries support:

```text
AE_OUTPUT_ROOT     default: <repo>/SC26-AE/output
QUICK              allowed: 0|1, default: 0
```

Task1 additionally supports:

```text
SCALE_GPU          required unless exactly one visible GPU is available
CAPTURE_NSYS       allowed: 0|1; release default fixed by D16 qualification evidence
```

The release default is a literal per-model value committed after Task 2.4 evidence is reviewed; the script never probes for artifacts and changes `CAPTURE_NSYS` automatically. Authors may set the opposite explicit value during qualification, while the README publishes only the qualified reviewer path.

Task2 additionally supports:

```text
CUDA_VISIBLE_DEVICES  exactly two comma-separated device IDs
REBUILD               allowed: 0|1, default: 0
```

Task3 additionally requires:

```text
ARTIFACT_SOURCE    required: fresh|prebaked
SIMULATOR_HARDWARE_TYPE  required on CPU-only machines
```

When the D21 result is `regular_git`, `PREBAKED_ROOT` defaults to the verified in-repository `SC26-AE/prebaked/` root. When the result is `github_release`, the explicit fetch command writes a new verified root under `AE_OUTPUT_ROOT/_downloads/<distribution_id>/prebaked/`, and the reviewer passes that exact path as `PREBAKED_ROOT`; Task3 never downloads it automatically. `PREBAKED_ROOT` is rejected when `ARTIFACT_SOURCE=fresh`.

Task3 does not accept public node-size or database-directory overrides: the wrapper sets `LOCAL_SIZE=8` from the frozen §4 matrix and binds `DATABASE_DIR="${TRACE_DIR}"` after resolving the selected manifest. It passes both literal resolved values to the scheduler/simulator, and tests assert path equality rather than only flag presence.

### 7.2 Common Bash interface

`SC26-AE/lib/common.sh` exports only shared validation/path functions:

```bash
ae_die "message"
ae_require_command command_name
ae_require_file /absolute/or/repo/path
ae_require_dir /absolute/or/repo/path
ae_require_enum VARIABLE_NAME value allowed_value_1 allowed_value_2
ae_require_positive_int VARIABLE_NAME value
ae_repo_root
ae_model_output_dir model_key task_key
ae_gitlink_commit submodule_path
ae_assert_submodule_clean submodule_path
```

Each failure writes `[ERROR]` to stderr and returns nonzero. Paths are quoted; model keys are restricted to `gpt175b|qwen3_a30b|dsv3`; task keys are restricted to `task1|task2|task3`.

### 7.3 Task2 metrics schema

`SC26-AE/tools/echo_metrics.py` writes `metrics.json` and `metrics.md` with:

```text
schema_version = sc26-ae-echo-metrics-v1
dataset_row_count
validation_mse_by_fold[5]
average_validation_mse
test_mse
model_reload_max_abs_prediction_delta
scaler_feature_count
scaler_mean_count
scaler_scale_count
scaler_nonzero_scale_count
prediction_sample.original_execution_time
prediction_sample.predicted_execution_time
prediction_sample.predicted_execution_time_clipped
prediction_sample.predicted_slowdown_factor
prediction_sample.predicted_slowdown_factor_clipped
```

Acceptance: row count > 0; exactly five finite nonnegative fold MSE values; average equals their arithmetic mean within `1e-12`; test MSE finite and nonnegative; model reload delta is `0.0` within `1e-12`; feature/mean/scale counts are equal and > 0; every scale is nonzero; every prediction field is finite; clipped slowdown is nonnegative.

### 7.4 Scheduler CLI contract

This section specifies the **target CLI after Task 4.1**, not the current scheduler interface: current `mg_test.py` has neither `--bf16` nor `--output-dir`.

For every model, `LOCAL_SIZE=8`; scheduler and simulator argv tests assert the literal resolved value rather than merely checking that the flag exists. For the canonical `analytical` backend, simulation also fails fast unless the runtime invariant below holds; it does not mutate the backend global to conceal a mismatch:

```text
config.local_size == LOCAL_SIZE == nccl_comm.GPUS_PER_MACHINE == 8
```

`--model-size` is an identifier, not an architecture selector. Current `mg_test.py` accepts any string, and `mg_scheduling_plan.py` consumes it only in the legacy output-directory label. Task3 passes the exact AE keys `gpt175b`, `qwen3_a30b`, and `dsv3`; no numeric/model-family mapping is introduced. Hidden size, sequence length, topology, expert count, and batch dimensions remain explicit arguments and Task 4.1 tests all three keys.

Each Task3 wrapper calls:

```bash
python megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size "${PP}" \
  --expert-model-parallel-size "${EXP}" \
  --num-experts "${NUM_EXPERTS}" \
  --world-size "${WORLD_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --seq-length "${SEQ_LEN}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --model-size "${MODEL_KEY}" \
  --bf16 \
  --train-iters 1 \
  --trace-start 0 \
  --output-dir "${SCHEDULE_DIR}"
```

Qwen and DeepSeek pass `--untie-embeddings-and-output-weights`; GPT does not. With `--output-dir`, files are deterministic: `stage0_scheduling_plan.txt` through `stage${PP_MINUS_ONE}_scheduling_plan.txt`.

### 7.5 Rank0 report schema

`simu_main.py` receives:

```text
--report-output-dir <task3-run-dir>
--report-model gpt175b|qwen3_a30b|dsv3
--artifact-source fresh|prebaked
```

All three must be present together or all absent. Task3 always supplies all three.

`report.json` schema:

```json
{
  "schema_version": "sc26-ae-rank0-report-v1",
  "model": "qwen3_a30b",
  "artifact_source": "fresh",
  "rank_id": 0,
  "rank0_step_time_ms": 1.0,
  "rank0_forward_step_duration_sum_ms": 1.0,
  "rank0_backward_step_duration_sum_ms": 1.0,
  "rank0_optimizer_step_duration_sum_ms": 1.0,
  "rank0_comp_plus_comm_diagnostic_ms": 1.0,
  "simulator_load_time_s": 1.0,
  "simulator_execution_time_s": 1.0,
  "simulator_wall_clock_s": 2.0
}
```

Metric definitions:

```text
rank0_step_time_ms =
    max(op.finish_time for op in rank0.final_merge_timeline)
  - min(op.join_time for op in rank0.final_merge_timeline)

rank0_<operation>_duration_sum_ms =
    sum(op.finish_time - op.join_time
        for op in rank0.comp_timeline
        if op.name == <exact operation name>)

rank0_comp_plus_comm_diagnostic_ms =
    sum(validated durations in rank0.comp_timeline and rank0.comm_timeline)

simulator_load_time_s = round(raw_load_time_s, 6)
simulator_execution_time_s = round(raw_execution_time_s, 6)
simulator_wall_clock_s = round(simulator_load_time_s + simulator_execution_time_s, 6)
```

Exact operation names are `forward_step`, `backward_step`, and `optimizer_step`; rank0 must contain at least one occurrence of each exact name. A missing operation raises `ValueError` rather than serializing a misleading zero-duration sum. Each serialized numeric field is normalized once to six decimal places before JSON and Markdown serialization; wall-clock is then derived from the two normalized component fields as shown above so readers can reproduce the exact serialized equality. All values are finite and nonnegative; `rank0_step_time_ms` is strictly positive. Missing rank0, empty final timeline, missing target operation, missing timestamps, non-finite timestamps, `finish_time < join_time`, or nonpositive span raises `ValueError`. Diagnostic `comp+comm` never fills the main value.

---

## 8. Gate A — Plan Approval (Current Phase)

### Task A1: Requirements and decision capture

**Files:**
- Modify: `task_memory/task_2026-07-15_sc26_ae_workflow/requirements.md`
- Modify: `task_memory/task_2026-07-15_sc26_ae_workflow/notes.md`
- Modify: `task_memory/task_2026-07-15_sc26_ae_workflow/issues.md`
- Modify: `task_memory/task_2026-07-15_sc26_ae_workflow/progress.md`

- [x] Capture R1–R15 and D1–D23, with every raw item marked `[Original Request]`.
- [x] Resolve source selection as explicit-only (D23 supersedes D22).
- [x] Record I13 portability, I14 setup source strictness, I15 overlap-mode requirements, and I16 node-topology contract risk.

### Task A2: Author plan rewrite and self-review

**Files:**
- Modify: `task_memory/task_2026-07-15_sc26_ae_workflow/plan.md`
- Create: `task_memory/task_2026-07-15_sc26_ae_workflow/review.md`

- [x] Map exact files, interfaces, phases, tests, gates, numeric criteria, decisions, and risks.
- [x] Run author self-review for spec coverage, placeholder-free content, field consistency, minimal scope, and `LOCAL_SIZE=8` consistency.
- [x] Record findings and remediation in `review.md` using the mandated review fields.

### Task A3: Independent StepCode Claude review

Run:

```bash
omx ask claude "Review task_memory/task_2026-07-15_sc26_ae_workflow/plan.md as an independent SC'26 AE plan reviewer. Read requirements.md, notes.md, issues.md, progress.md, the referenced model scripts, tracer output paths, sim-engine scheduler/simulator/slowdown interfaces, existing slowdown e2e, grouped-gemm setup documentation, and Echo-slowdown interfaces. Do not implement or edit code. Return exactly one verdict: APPROVE, WATCH, or BLOCK; then list requirement gaps, hidden fallback paths, provenance mismatches, stale-output risks, unsafe repository or publication operations, test gaps, reporter/scheduler semantic errors, packaging/download risks, ungrounded resource claims, and the smallest concrete plan corrections. Treat D1-D23 and the current no-implementation gate as binding. Check that Task1/Task3 versioned runs and post-verification markers prevent cross-run contamination; DATABASE_DIR is exactly TRACE_DIR; missing forward/backward/optimizer operations fail; fresh commit checks bind to the producer checkout while prebaked checks avoid final-HEAD circularity; the D21 byte gate counts distribution metadata; GitHub Release fetch is explicit and verified; and external publication requires exact-target approval. Explicitly adjudicate I16: Task3 is frozen to LOCAL_SIZE=8, while GPT Task1 passes fake-gpus-per-node=8 and both MoE Task1 sources pass fake-gpus-per-node=fake-world-size; determine whether any consumed trace field requires a minimal MoE source-script change or whether recording capture_runtime.fake_gpus_per_node separately and testing Task3's independent simulation_topology.local_size is sufficient."
```

Expected backend contract:

```bash
stepcode claude --model 'claude-opus-4-6[1m]' --effort max -p -- "<review prompt>"
```

- [x] Record the generated `.omx/artifacts/ask-claude-*.md` path and raw verdict in `review.md`.
- [ ] For `APPROVE`, continue Gate A validation.
- [x] For `WATCH`, amend the plan and record each watched risk plus verification gate.
- [ ] For `BLOCK`, stop immediately and request user adjudication; do not self-override.

### Task A4: Gate A document validation

Run read-only validation:

```bash
python - <<'PY'
import re
from pathlib import Path
root = Path("task_memory/task_2026-07-15_sc26_ae_workflow")
required = ["requirements.md", "notes.md", "issues.md", "progress.md", "plan.md", "review.md"]
for name in required:
    path = root / name
    assert path.is_file(), path
    text = path.read_text(encoding="utf-8")
    assert "## Modification History" in text, path
plan = (root / "plan.md").read_text(encoding="utf-8")
requirements = (root / "requirements.md").read_text(encoding="utf-8")
for token in ["T" + "BD", "T" + "ODO", "implement" + " later", "appropriate" + " error handling", "similar" + " to Task"]:
    assert token not in plan, token
for index in range(1, 16):
    assert f"R{index}" in plan
    assert re.search(rf"^## R{index}\\..*\\n\\[Original Request\\]", requirements, re.MULTILINE), f"R{index}"
for index in range(1, 24):
    assert f"D{index}" in plan
    assert re.search(rf"^### D{index}\\..*\\n\\[Original Request\\]", requirements, re.MULTILINE), f"D{index}"
for literal in [
    'ARTIFACT_SOURCE=fresh',
    'ARTIFACT_SOURCE=prebaked',
    'DATABASE_DIR="${TRACE_DIR}"',
    'LOCAL_SIZE=8',
    'capture_runtime.fake_gpus_per_node',
    'simulation_topology.local_size',
    'capture_marker.json',
    'run_marker.json',
]:
    assert literal in plan, literal
entries = set(re.findall(r'`(SC26-AE/task[123]_(?:gpt175b|qwen3_a30b|dsv3)\\.sh)`', plan))
assert len(entries) == 9, entries
print("PASS: task documents, histories, placeholder scan, Original Request tags, contracts, and R/D references")
PY

git status --short --untracked-files=all
git diff --name-only
git diff --submodule=short
git diff -- task_memory/task_2026-07-15_sc26_ae_workflow
```

Expected:
- Python exit code 0 with the PASS line and nine unique entry paths.
- This session changed only Markdown files under the active task directory. Because that directory is currently untracked, an empty path-specific `git diff` is expected and is not treated as proof by itself; the full status/diff inventory is compared with the recorded Gate A baseline.
- No implementation file, submodule gitlink, branch, or commit changed.

- [x] Fresh Gate A document validation passed after independent WATCH remediation: 6 documents, 15 tagged requirements, 23 tagged decisions, 9 unique public entries, 38 balanced fence pairs, 4 WATCH gates, and 1 advisor artifact; exit code 0.
- [x] Safety inventory still reports only the 3 pre-existing tracked diff files and the same 3 recursive submodule status entries; no Gate A implementation or gitlink change was introduced.

**Gate A exit:** satisfied by explicit user approval on 2026-07-16. Phase 0 is now open; Gate B and all implementation phases retain their own entry conditions.

---

## 9. Phase 0 — Safety Baseline, Branches, and Worktree

### Task 0.1: Reconfirm and protect the existing workspace

**Files inspected only:** main repo status, both submodule statuses, and `/data/ycfeng/Megatron-LM-ddp-overlap-review-20260713` branch identity.

- [x] Run `git status --short --branch`, `git diff --stat`, `git submodule status --recursive`, and `git worktree list --porcelain`.
- [x] Confirm the pre-existing tracked edits and untracked AE assets match the Gate A inventory.
- [x] Confirm no file from `task_memory/task_2026-07-13_ddp_overlap_comprehensive_review/` or `.omc/` is staged.
- [x] Stop if any new unknown modification appears; classify ownership before proceeding. No new path was found; the pre-existing empty untracked `=10.1` file remains excluded from D3.

### Task 0.2: Commit the user-approved D3 baseline on `overlap-tracing`

Stage only the explicit D3 inventory:

```bash
git add -- \
  examples/realistic_run_gpt.sh \
  examples/update_pretrain_gpt.sh \
  tests/e2e/run_ddp_slowdown_compare.py \
  tools/ae/setup_grouped_gemm_v1.sh \
  docs/ae/grouped_gemm_v1_setup.md \
  tests/integration/fixtures/mock_torchrun_bin/torchrun \
  tests/integration/test_gpt_example_mock_mode.sh \
  tests/integration/test_grouped_gemm_v1_runtime.py \
  tests/unit/test_setup_grouped_gemm_v1.sh \
  task_memory/task_2026-07-15_sc26_ae_workflow
```

Verify with `git diff --cached --name-only`; any additional path is a hard failure. Run the existing relevant tests before committing. Use a Lore-formatted commit message that records the baseline purpose, protected branch constraint, confidence, scope risk, tested commands, and remaining GPU gap.

- [x] Staged exactly the 15 D3 paths; protected/excluded staged path count was zero.
- [x] Ran the relevant baseline tests before commit with `30/30`, `22/22`, and `16/16` passing cases.
- [x] Created Lore baseline commit `0ad3cb4eda2248f4e09908a80e5693cffa6e0c1e` on `overlap-tracing`.

### Task 0.3: Create isolated feature branches/worktree

- [x] Load and follow `using-git-worktrees`.
- [x] Create main-repo branch `sc26-ae` from the approved D3 commit in isolated worktree `/data/ycfeng/Megatron-LM-sc26-ae`; do not repurpose the active overlap-review worktree.
- [x] In sim-engine, create branch `sc26-ae` from `2044cccc8fff222172b7f91571a617886841001f`.
- [x] In Echo-slowdown, create only a local convenience branch from `1390b4416ded08bc1b9cd0620d329d81d4470bf9`; introduce no AE-required commit.
- [x] Verify current pinned public reachability; repeat later for the new sim-engine commit.

### Task 0.4: Baseline test gate

Run in the documented environment:

```bash
bash tests/unit/test_setup_grouped_gemm_v1.sh
bash tests/integration/test_gpt_example_mock_mode.sh
python -m pytest megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py -q
python -m pytest megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py -q
```

Expected: all commands exit 0. Any failure is diagnosed as baseline/environment/code before feature work; no feature patch is started on a failing baseline.

- [x] Isolated-worktree baseline gate passed: grouped-gemm `30/30`, GPT mock integration `22/22`, and sim-engine pytest `16/16`; all commands exited `0`.

---

## 9.5. Gate B — Existing Three-Task Runtime Reconnaissance

Gate B satisfies R9 before any `SC26-AE/` infrastructure, source-script adaptation, scheduler/reporter change, or task wrapper is implemented. It runs existing code only, writes runtime artifacts under the designated test/output locations, and records facts in `notes.md`, `issues.md`, and `progress.md`. It does not claim the final nine-entry workflow is already valid.

### Task B1: AE-image environment and baseline interface capture

- [ ] Read `/data/ycfeng/stepfun-env-handbook/guidence.md` and `task_memory/env_handbook.md`; run the documented 1-GPU and 2-GPU `rlaunch --predict-only` checks before live allocation.
- [ ] Launch the pinned AE image with the repository mounted. Record Python, torch, CUDA, `nsys`, `ncu`, XGBoost, and grouped-gemm versions plus exact executable paths.
- [ ] Do not exercise the known automatic VCS→archive recovery in `tools/ae/setup_grouped_gemm_v1.sh`. If the image cannot run the existing baseline without that unsafe branch, record the environment blocker in `issues.md` and stop Gate B; Phase 1 remains blocked until the user-approved plan addresses the root cause.

### Task B2: Run existing Task1 and inspect real outputs

On one H800, run the existing Qwen source in a one-rank smoke configuration without any AE wrapper:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
CAPTURE_ROOT="${REPO_ROOT}/SC26-AE/output/_work/recon-task1-qwen-${RUN_ID}"
if [[ -e "${CAPTURE_ROOT}" ]]; then
  printf '[ERROR] Reconnaissance capture path already exists: %s\n' "${CAPTURE_ROOT}" >&2
  exit 1
fi
mkdir -p "${CAPTURE_ROOT}"
cd "${CAPTURE_ROOT}"
MODE=scaling \
MODEL_PROFILE=full \
FAKE_WORLD_SIZE=256 FAKE_PP=4 FAKE_TP=8 FAKE_DP=8 FAKE_EXP=8 \
FAKE_RANK_ORDER=0 \
SCALE_GPU=0 TRACE_MEMORY=1 OVERLAP_GRAD_REDUCE=1 \
TRAIN_ITERS=3 TRACE_START=2 \
bash "${REPO_ROOT}/examples/pretrain_qwen3_30b_a3b_moe.sh"
```

- [ ] Record `RUN_ID` once, exit code, elapsed seconds, peak memory, exact trace directory/file, memory JSON path, op names, comm metadata, DDP-overlap events, and all source-script defaults that were not safely overridable.
- [ ] Confirm every CWD-relative `profiler_log/`, `memory_traces_scaling/`, and replay-cache file is contained by this new `CAPTURE_ROOT`; any output outside it is an interface discrepancy to record before Phase 1.
- [ ] Confirm this is evidence gathering only. Do not copy, rename, normalize, or patch outputs during Gate B.

### Task B3: Run existing Task2 from an isolated pinned snapshot

On exactly two H800 GPUs, create a new versioned reconnaissance snapshot without modifying the Echo submodule:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
SNAPSHOT_ROOT="${REPO_ROOT}/SC26-AE/output/_work/recon-task2-${RUN_ID}"
if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  printf '[ERROR] Reconnaissance snapshot path already exists: %s\n' "${SNAPSHOT_ROOT}" >&2
  exit 1
fi
mkdir -p "${SNAPSHOT_ROOT}/source"
ECHO_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD:Echo-slowdown)
git -C "${REPO_ROOT}/Echo-slowdown" archive "${ECHO_COMMIT}" \
  | tar -x -C "${SNAPSHOT_ROOT}/source"
cd "${SNAPSHOT_ROOT}/source"
CUDA_VISIBLE_DEVICES=0,1 python update_configs.py
CUDA_VISIBLE_DEVICES=0,1 bash run_all.sh
```

`RUN_ID` is generated exactly once in UTC and recorded in `progress.md`; every Task B3 path in that run derives from the same value.

- [ ] Record module-by-module elapsed time, dataset row count, actual output paths, fold/test metrics already emitted, model/scaler reload behavior, NCU CSV schema, and every tracked file the upstream run would have overwritten without snapshot isolation.
- [ ] From any current directory, verify `git -C "${REPO_ROOT}/Echo-slowdown" status --short` is byte-for-byte identical before and after the run and `ECHO_COMMIT` equals the main-repository gitlink. Any change or mismatch is a hard failure.

### Task B4: Run the existing slowdown-enabled Task3 chain

Using one H800 and the current committed tiny baseline, run:

```bash
SLOWDOWN_E2E_SCALE_GPU=0 bash tests/e2e/test_ddp_slowdown_simulate_smoke.sh
```

- [ ] Record trace/SQLite/assets/schedule inputs, simulator argv, processed `cmd_uid` counts, slowdown-off/on backward durations, delayed/shared communication counts, output paths, and total wall-clock time.
- [ ] Inspect the in-memory simulator ownership path (`SimulatorEngine.timeline_manager.stages_timeline_process_dict`) and confirm the proposed rank0 reporter can read rank0 timelines without parsing stdout or visualization files.
- [ ] Record exact-name counts and validated durations for rank0 `forward_step`, `backward_step`, and `optimizer_step` in `comp_timeline`. The canonical path is `simu_main.py` -> `src/core/simu_engine.py`, whose direct mapping includes `optimizer_step`, and the PP=1 smoke schedule explicitly contains it; any missing target operation is an interface blocker, not an accepted PP=1 gap.
- [ ] Do not treat this tiny PP=1 smoke as proof for the final 1024/256-rank matrices; it proves only the current cross-module interface and strict blueprint chain.

### Task B5: Reconcile evidence before feature work

- [ ] Update `notes.md` with facts, `issues.md` with blockers/root causes, and `progress.md` with exact commands and numeric results.
- [ ] Re-read this plan against the observed interfaces. Apply only plan-document corrections; any newly required critical product-logic or scope decision is grilled one question at a time and reviewed independently.
- [ ] Gate B exits only when the existing three tasks have run successfully, no unexplained interface gap remains, and any plan delta has user approval when it materially changes scope. No Phase 1 RED test or implementation edit starts earlier.

---

## 10. Phase 1 — Shared AE Infrastructure

### Task 1.1: Explicit grouped-gemm source and setup entry

**Files:**
- Modify: `tools/ae/setup_grouped_gemm_v1.sh`
- Modify: `docs/ae/grouped_gemm_v1_setup.md`
- Modify: `tests/unit/test_setup_grouped_gemm_v1.sh`
- Create: `SC26-AE/setup.sh`
- Create: `tests/integration/test_sc26_ae_setup.sh`

**Interfaces:**
- Consumes: `GROUPED_GEMM_SOURCE=vcs|archive`, existing pinned URLs/commits/SHA256 values.
- Produces: verified grouped-gemm installation and setup manifest; never switches source.

- [ ] Add RED cases proving: missing source fails before pip/curl; invalid value fails; `vcs` failure does not call archive tools; `archive` does not call VCS pip; archive hash mismatch fails; both successful modes preserve current verification and idempotency.
- [ ] Run `bash tests/unit/test_setup_grouped_gemm_v1.sh`; observe the new cases fail against the automatic-recovery implementation.
- [ ] Replace the recovery branch with an explicit `case "${GROUPED_GEMM_SOURCE:?}" in vcs|archive)` path. Keep current pinned integrity checks and environment validation.
- [ ] Update the existing `docs/ae/grouped_gemm_v1_setup.md` in place: document both explicit commands, remove automatic-recovery instructions, retain historical measured evidence with a clear pre-change label, and update its modification history.
- [ ] Implement `SC26-AE/setup.sh` as a thin wrapper that verifies Python, `nsys`, and `ncu`, then invokes the installer with the selected source.
- [ ] Run unit and setup integration tests; expect all cases to pass and no selected-source cross-call in fake command logs.

### Task 1.2: Common shell contracts

**Files:**
- Create: `SC26-AE/lib/common.sh`
- Create: `tests/unit/test_sc26_ae_common.sh`

- [ ] Write RED tests for enum rejection, zero/negative integer rejection, unknown model/task keys, missing file/dir/command, dirty submodule, and safe output path construction.
- [ ] Run the shell unit test; observe missing-function failures.
- [ ] Implement only the functions in §7.2 with quoted paths and stderr errors.
- [ ] Re-run; expect a numeric PASS count and exit 0.

### Task 1.3: Portable manifest helper

**Files:**
- Create: `SC26-AE/tools/artifact_manifest.py`
- Create: `tests/unit/test_sc26_ae_artifact_manifest.py`

- [ ] Write RED pytest cases for deterministic SHA256, relative-path serialization, absolute/parent traversal rejection, missing and extra files, size/hash mismatch, wrong schema/model/simulation-topology/capture-runtime/profile/source, strict 50 MiB boundary, inclusive 500 MiB total boundary, inclusion of nested artifact/distribution manifest bytes, symlink/special-file rejection, and both distribution outcomes.
- [ ] Run `python -m pytest tests/unit/test_sc26_ae_artifact_manifest.py -q`; observe import/file failure.
- [ ] Implement the four functions and three subcommands from §5.3 with stable sorted JSON output.
- [ ] Re-run targeted tests; expect all cases to pass.

### Phase 1 commit gate

Run shell syntax checks, targeted unit/integration tests, `python -m compileall SC26-AE/tools`, and a fresh code-review pass. Commit only after evidence is recorded in `progress.md`.

---

## 11. Phase 2 — Task1 Wrappers and Atomic Capture

### Task 2.1: GPT source-script AE controls

**Files:**
- Modify: `examples/update_pretrain_gpt.sh`
- Modify: `tests/integration/test_gpt_example_mock_mode.sh`

**Interfaces:**
- Consumes: explicit `MODEL_SIZE=175`, fake topology, mock data, precision, trace, rank list, and output settings.
- Produces: one physical-GPU process per representative fake rank with bf16, memory trace, kernel-ground-truth labels, and DDP-overlap metadata.

- [ ] Add RED mock-torchrun assertions for exact topology 1024/8/8/16; rank list `0,128,...,896`; `--bf16` present and `--fp16` absent; `--mock-data`; `--overlap-grad-reduce`; trace-memory and kernel-ground-truth flags; env-controlled train/trace values; and no legacy hard-coded data path in mock mode.
- [ ] Run the targeted mock integration test and observe failures for currently missing controls.
- [ ] Add minimal env knobs while preserving existing defaults for non-AE callers. Do not refactor unrelated script structure.
- [ ] Re-run; expect captured argv to satisfy every assertion.

### Task 2.2: Shared Task1 runner and three public entries

**Files:**
- Create: `SC26-AE/lib/task1_trace.sh`
- Create: three `SC26-AE/task1_*.sh` entries
- Create: `tests/integration/test_sc26_ae_task1_contracts.sh`

**Interfaces:**
- Each entry is a three-line public wrapper: strict shell mode, source common/task1 library, call `ae_run_task1 <model_key>`.
- `ae_run_task1` exports the exact §4 matrix, forces `MODE=scaling`, `TRACE_MEMORY=1`, `TRACE_KERNEL_GROUND_TRUTH=1`, `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`, and `OVERLAP_GRAD_REDUCE=1`.
- It generates one `capture_id`, requires a nonexistent `task1/runs/<capture_id>/` root, and invokes the model script from that run's `runtime/` CWD. It publishes `capture_marker.json` only after trace/memory/Nsight inventory and manifest verification succeed.
- It records the source script's actual `capture_runtime.fake_gpus_per_node` separately from Task3's frozen `simulation_topology.local_size=8`; it does not silently rewrite the MoE value while I16 remains under independent review.

- [ ] Write RED contract tests with fake `torchrun`, `nsys`, and `nsys export`: validate all three resolved configurations, full/QUICK rank sets, one physical GPU, CWD-local trace/memory directories, capture boundary around the whole rank loop, nonexistent-run enforcement, stale sibling-run exclusion, and marker publication only after successful verification.
- [ ] Run the integration test; observe missing entry/library failures.
- [ ] Implement the minimal shared runner and entries. Do not call wall-clock scan scripts as runtime sources.
- [ ] Re-run; expect exactly nine public entry filenames across all tasks and three valid Task1 invocations.

### Task 2.3: Task1 summaries and manifests

The runner records in `logs/summary.log` and manifest metadata:

```text
model
selected_rank_ids
selected_rank_count
trace_file_count
memory_json_count
capture_runtime_fake_gpus_per_node
simulation_topology_local_size = 8
per_rank_peak_allocated_mb
maximum_peak_allocated_mb
capture_id
capture_elapsed_seconds
single_rank_elapsed_seconds
estimated_full_seconds
fresh_capture_gate_threshold_seconds = 7200
fresh_capture_gate_result = pass|prebaked_required
nsys_rep_path/sqlite_path when CAPTURE_NSYS=1
file sizes and SHA256 values
```

- [ ] Add RED cases for trace count mismatch, duplicate/missing rank, empty memory data, non-finite/negative memory values, absent `.nsys-rep`/SQLite when requested, mismatched capture ID, wrong marker manifest SHA256, marker traversal, and stale files outside the selected run.
- [ ] Implement summary extraction and call the manifest helper.
- [ ] Re-run; all invalid fixtures fail and the valid fixture records actual numeric values.

### Task 2.4: GPU smoke and D16 qualification

**Files:**
- Create: `tests/e2e/test_sc26_ae_task1_smoke.sh`

- [ ] Run `QUICK=1 CAPTURE_NSYS=1` for each model on one H800 after setup.
- [ ] For each MoE model, separately time rank 0 and compute `rank0_seconds × 256`.
- [ ] If estimate `<=7200`, run one complete atomic selected-rank capture and record measured elapsed time. If `>7200`, record `prebaked_required` and do not start the full Nsight capture.
- [ ] The `>7200` result disables the reviewer-facing fresh full-capture path; it does not waive provenance. Phase 6 must still receive a complete author-prepared atomic capture or stop release packaging.
- [ ] Verify trace count equals selected-rank count; memory JSON count equals selected-rank count; peak memory values are finite and >0; `.nsys-rep` and SQLite are nonempty when captured; every manifest checksum verifies.
- [ ] Resolve I1 by attempting Task3 with the QUICK set. Only a complete 256-world-size simulation with valid report and slowdown blueprint coverage can justify changing the default MoE rank scope.

---

## 12. Phase 3 — Task2 Isolated Echo-slowdown Workflow

### Task 3.1: Snapshot isolation and shared run identity

**Files:**
- Create: `SC26-AE/lib/task2_echo.sh`
- Create: `tests/unit/test_sc26_ae_task2_snapshot.sh`

**Interfaces:**
- Consumes: main-repo gitlink commit for `Echo-slowdown`, exactly two visible GPU IDs, `REBUILD=0|1`.
- Produces: a unique `_shared/task2/runs/<predictor_run_id>/` bundle and per-model marker; pinned submodule remains clean.

- [ ] Write RED fake-git tests proving the command is `git -C Echo-slowdown archive <gitlink_commit>`; snapshot is outside the submodule; dirty-before and dirty-after states fail; source commit mismatch fails; an existing verified marker is reused only with `REBUILD=0`; partial/corrupt existing bundle fails; `REBUILD=1` creates a new run ID without deleting the old run.
- [ ] Run the shell unit test; observe missing library failure.
- [ ] Implement archive extraction into a new `SC26-AE/output/_work/task2.<predictor_run_id>/source` and validate the extracted files before execution.
- [ ] Re-run; expect every isolation and idempotency assertion to pass.

### Task 3.2: Echo run, canonical bundle, and metrics

**Files:**
- Create: `SC26-AE/tools/echo_metrics.py`
- Create: `tests/unit/test_sc26_ae_echo_metrics.py`
- Create: `tests/integration/test_sc26_ae_task2_contract.sh`

Canonical archived files:

```text
training_testing/output/train_dataset.csv
training_testing/output/xgb_model.json
training_testing/output/standard_scaler.json
training_testing/output/prediction/*
merge/input/kernel_metric_output.csv
logs/run_all.log
metrics.json
metrics.md
artifact_manifest.json
```

- [ ] Write RED unit fixtures for all §7.3 metric invariants and negative scaler/model/log cases.
- [ ] Write RED integration fixtures proving `update_configs.py` and `run_all.sh` execute only inside the snapshot and outputs copy into a unique shared run.
- [ ] Run both tests and observe failures.
- [ ] Implement metrics extraction/validation, canonical copying, and manifest creation; write the same concrete `predictor_run_id` into the shared manifest and all three model markers.
- [ ] Re-run; expect numeric metrics and checksums to pass.

### Task 3.3: Three public Task2 entries

**Files:**
- Create: `SC26-AE/task2_gpt175b.sh`
- Create: `SC26-AE/task2_qwen3_a30b.sh`
- Create: `SC26-AE/task2_dsv3.sh`
- Create: `tests/e2e/test_sc26_ae_task2_smoke.sh`

- [ ] Add RED assertions that all entries call the same shared core and each writes `output/<model>/task2/predictor_marker.json` pointing to the same verified run/checksums.
- [ ] Implement entries and marker validation.
- [ ] On two H800 GPUs, run one build plus two reuse entries; record dataset rows, five fold MSE values, average/test MSE, reload delta, scaler counts, nonzero scales, and the five prediction sample values.
- [ ] Verify `git -C Echo-slowdown status --short` is empty before and after every entry.

---

## 13. Phase 4 — Canonical Scheduler and Rank0 Reporter

### Task 4.1: bf16 scheduler and explicit output directory

**Files:**
- Modify: `megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py`
- Modify: `megatron-sim-engine/src/scheduler/mg_scheduling/mg_scheduling_plan.py`
- Create: `megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py`

- [ ] Write RED tests for mutually exclusive `--fp16`/`--bf16`, `torch.bfloat16` PP tensor dtype, exact explicit output directory, deterministic stage filenames, direct acceptance of the exact model-size labels `gpt175b|qwen3_a30b|dsv3`, and the three matrices below.
- [ ] Run `python -m pytest megatron-sim-engine/tests/unit/test_mg_scheduling_ae_contract.py -q`; observe failures because bf16/output-dir are absent.
- [ ] Implement the minimal parser/dtype/writer changes. Legacy calls without `--output-dir` retain their current location; AE calls never omit it.
- [ ] Re-run; expect:

| Model | Stage files | Forward/stage | Backward/stage | Optimizer/stage | PP shape/dtype |
|-------|-------------|---------------|----------------|-----------------|----------------|
| GPT-175B | 8 | 48 | 48 | 1 | `[2048, 1, 12288]`, `torch.bfloat16` |
| Qwen3-A30B | 4 | 16 | 16 | 1 | `[256, 1, 2048]`, `torch.bfloat16` |
| DSV3 | 4 | 16 | 16 | 1 | `[256, 1, 2048]`, `torch.bfloat16` |

Every stage also has one `dp_allreduce`; GPT first/last stages have embedding allreduce with tied weights; Qwen/DSV3 pass untied weights and do not add it. Root `mg_scheduling/` is not edited or tested for equivalence.

The three AE labels remain opaque strings in output metadata/paths; Task 4.1 must not add a second model-configuration table or infer architecture fields from `--model-size`.

### Task 4.2: Pure rank0 report aggregation

**Files:**
- Modify: `megatron-sim-engine/simu_main.py`
- Create: `megatron-sim-engine/tests/unit/test_rank0_report.py`

**Interfaces:**

```python
def build_rank0_report(
    simulator_engine: SimulatorEngine,
    model: str,
    artifact_source: str,
    load_time_s: float,
    execution_time_s: float,
) -> dict: ...

def write_rank0_report(report: dict, output_dir: pathlib.Path) -> None: ...
```

- [ ] Write RED unit tests with in-memory fake operations for overlapping timelines, exact-name filtering, one separately missing `forward_step`/`backward_step`/`optimizer_step`, nonzero timeline origin, diagnostic sum, six-decimal normalization, JSON/Markdown parity, and every fail-fast branch in §7.5.
- [ ] Run the unit test and observe import/attribute failure.
- [ ] Implement the two pure helpers adjacent to `run_simulation()`; do not parse stdout, call visualization for data, or reread CWD logs.
- [ ] Extend `run_simulation()` to build/write the report from the same engine instance and return the report fields plus `world_size`.
- [ ] Re-run; expect all calculations and error branches to pass.

### Task 4.3: Reporter integration

**Files:**
- Create: `megatron-sim-engine/tests/integration/test_rank0_report_integration.py`
- Modify: `megatron-sim-engine/simu_main.py`

- [ ] Add a RED canonical PP=2 integration fixture with `WORLD_SIZE=8`, `LOCAL_SIZE=8`, `PP=2`, `TP=1`, `EXP=1`, derived `DP=4`, `MBS=1`, and `GBS=8`. Generate it through the Task 4.1 scheduler rather than a hand-written schedule, supply all three report CLI fields, and verify `report.json`/`report.md`. Assert rank0 contains positive-duration `forward_step` and `backward_step`, exactly one positive-duration `optimizer_step`, and a positive final timeline span before the Phase 4 commit gate.
- [ ] Add negative invocations with incomplete report flags, missing rank0, each missing exact target operation, empty/invalid timeline, and an unwritable output target.
- [ ] Add RED topology invocations proving `--cc-backend analytical --local-size 8` satisfies `config.local_size == nccl_comm.GPUS_PER_MACHINE`, while a non-8 local size fails before engine construction with an explicit mismatch error. Implement only this fail-fast comparison; do not call `set_gpus_per_machine()` or silently rewrite either value.
- [ ] Run the integration test; observe failure before CLI support is complete, then PASS after Task 4.2.
- [ ] Run the existing sim-engine unit/integration suites affected by `simu_main.py` and scheduler changes.

### Phase 4 review gate

Request a separate code-review lane for scheduler/reporter semantics. A `BLOCK` stops integration; a `WATCH` is recorded with a targeted verification. The gate cannot pass until the canonical PP=2 fixture proves `optimizer_step` reaches rank0 `comp_timeline` and the analytical topology test proves `config.local_size == nccl_comm.GPUS_PER_MACHINE == 8`. Prepare and record the reviewed sim-engine `sc26-ae` commit SHA locally. Do not push it or update a public default branch until the §2 external publication gate receives explicit user approval for the exact remote/ref/SHA; after approval, verify public fetch before updating the main-repo gitlink.

---

## 14. Phase 5 — Task3 End-to-End Simulation

### Task 5.1: Explicit source and input resolver

**Files:**
- Create: `SC26-AE/lib/task3_simulation.sh`
- Create: `tests/unit/test_sc26_ae_task3_contracts.sh`

**Interfaces:**
- `ARTIFACT_SOURCE=fresh`: resolve the verified Task1 `capture_marker.json` and shared Task2 marker; require compatible capture/simulation-topology/capture-runtime/profile plus an exact predictor run/checksum set, verify fresh producer commits against the current producer checkout, then build slowdown assets into a new model Task3 run.
- `ARTIFACT_SOURCE=prebaked`: resolve only `PREBAKED_ROOT`; verify the complete model bundle, shared predictor manifest, and distribution manifest, including both `capture_id` and `predictor_run_id`; consume it without probing fresh output or requiring its historical main-repository producer commit to equal current consumer `HEAD`.
- Both branches generate one new `simulation_run_id`, require a nonexistent `task3/runs/<simulation_run_id>/`, and publish `run_marker.json` only after schedule, assets, report, and outer-manifest verification succeed.

- [ ] Write RED cases for missing source, invalid source, selected bundle absent, partial files, checksum mismatch, wrong model/topology/profile/capture-runtime/capture ID, unsafe manifest path, fresh/current-producer commit mismatch, prebaked internal producer/distribution mismatch, permitted prebaked producer-vs-consumer-HEAD difference, existing Task3 run destination, stale sibling run, marker traversal, and fresh/prebaked cross-mixing.
- [ ] Run the shell test; observe missing resolver failure.
- [ ] Implement two explicit `case` branches with no shared-source retry path.
- [ ] Re-run; expect valid fixtures to resolve exact paths and every invalid fixture to stop.

### Task 5.2: Schedule, slowdown-assets, simulator, and report command

For fresh source, run:

```bash
python megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py \
  --trace-dir "${TRACE_DIR}" \
  --nsys-sqlite "${NSYS_SQLITE}" \
  --ncu-metrics-csv "${NCU_METRICS_CSV}" \
  --label-prefix cmd_trace \
  --output-dir "${SLOWDOWN_ASSETS_DIR}" \
  --model-path "${MODEL_PATH}" \
  --scaler-path "${SCALER_PATH}"
```

Then both sources run the canonical scheduler (§7.4) and:

```bash
DATABASE_DIR="${TRACE_DIR}"
```

The trace directory is deliberately the same single-rank operation database used by the existing slowdown e2e. Neither source branch accepts an independent `DATABASE_DIR`; a non-identical resolved path is a contract failure.

```bash
python megatron-sim-engine/simu_main.py \
  --framework megatron-lm \
  --mode simulate \
  --trace-dir "${TRACE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --pp-size "${PP}" \
  --tp-size "${TP}" \
  --exp-size "${EXP}" \
  --strategy 1F1B-none_interleaved \
  --cc-backend analytical \
  --enable-slowdown \
  --overlap-mode on \
  --slowdown-assets-dir "${SLOWDOWN_ASSETS_DIR}" \
  --slowdown-model-path "${MODEL_PATH}" \
  --slowdown-scaler-path "${SCALER_PATH}" \
  --no-visualize \
  --report-output-dir "${TASK3_DIR}" \
  --report-model "${MODEL_KEY}" \
  --artifact-source "${ARTIFACT_SOURCE}"
```

- [ ] Add integration assertions that the argv contains every explicit flag above and no default backend/source is relied upon.
- [ ] Assert `DATABASE_DIR` and `TRACE_DIR` are the same canonical resolved directory for fresh and prebaked fixtures; an independent or merely textually similar path fails.
- [ ] Assert the scheduler and simulator both receive resolved `--local-size 8`; a missing, different, or default-derived value fails the contract test.
- [ ] Verify builder strict failures propagate unchanged; no wrapper catches them and continues.
- [ ] Verify report schema/invariants and write a Task3 outer manifest containing schedule, assets, report, source manifests, `simulation_run_id`, and checksums; publish `run_marker.json` only afterward.

### Task 5.3: Three public Task3 entries and CPU prebaked e2e

**Files:**
- Create: three `SC26-AE/task3_*.sh` entries
- Create: `tests/integration/test_sc26_ae_task3_contract.sh`
- Create: `tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh`
- Create: `tests/e2e/test_sc26_ae_fresh_chain.sh`

- [ ] Add RED entry-contract assertions and CPU fixture.
- [ ] Implement the three thin entries.
- [ ] Run all three with explicit `ARTIFACT_SOURCE=prebaked` on CPU with `SIMULATOR_HARDWARE_TYPE` set; expect positive step time, all three exact operations present, finite component sums, positive wall-clock, and JSON/Markdown parity. Capture peak RSS for each run with the qualified environment's `/usr/bin/time -v` (or an equivalently documented platform counter) and record both KiB and GiB values.
- [ ] Run one full fresh chain with matching Task1 capture and Task2 predictor; expect assets builder coverage for every required backward `cmd_uid` and a valid report.
- [ ] If I1 QUICK compatibility failed, assert fresh Task3 rejects QUICK as incomplete rather than extrapolating silently.

---

## 15. Phase 6 — Prebaked Packaging and Distribution

### Task 6.1: Build complete three-model prebaked bundles

- [ ] Use only verified outputs from Phases 2–5.
- [ ] Treat the 7200-second D16 threshold as an AE reviewer-runtime gate, not an artifact-provenance exemption. If a model is marked `prebaked_required`, authors must supply or run one complete atomic author-side capture using the §5.1 boundary, even if that preparation takes longer than two hours; record its actual elapsed time and checksums. If no such complete capture exists, Phase 6 is BLOCKED and no release bundle is published.
- [ ] Include per-model trace, SQLite, slowdown assets, schedule/config metadata, and shared model/scaler/NCU metrics.
- [ ] Record a distinct `capture_id` for each model bundle and the shared `predictor_run_id`; never synthesize equality between these independent identities.
- [ ] Generate/verify per-bundle manifests plus `distribution_manifest.json`; record every payload and metadata file's bytes, MiB, SHA256, producer commits, nested manifest SHA256, and compatibility commits. Do not require a producer main-repository commit to equal the later payload/publication commit.
- [ ] Confirm bundle-relative runtime paths and explicit model/scaler overrides work after copying the bundle to a different absolute directory.

### Task 6.2: Apply D21 exactly once

Construct the complete regular-Git candidate under a new versioned staging root, including three model bundles, shared Task2, every nested `artifact_manifest.json`, and the final `distribution_manifest.json`. Run `artifact_manifest.py size-gate --root <staging-root>` only after that candidate is complete.

- `regular_git` only if every file is `<52,428,800` bytes and total is `<=524,288,000` bytes.
- Otherwise `github_release`.

For `regular_git`, whitelist and commit the exact verified candidate files under `SC26-AE/prebaked/`. Re-run the size gate on the committed-content staging tree; adding any unmeasured distribution file invalidates the gate.

For `github_release`, create one deterministic archive from the verified candidate and create `SC26-AE/lib/fetch_prebaked.sh` plus its unit test. Repository content contains only the distribution manifest, immutable Release URL/tag/asset name, archive SHA256, expected archive bytes, and verifier. Before extraction, the fetcher rejects absolute paths, `..` components, links, devices, and other non-regular archive members. It downloads exactly the declared asset once into a new `AE_OUTPUT_ROOT/_downloads/<distribution_id>/` path, verifies bytes and SHA256, extracts into that versioned path, verifies all nested manifests/payload hashes, and prints the resolved `prebaked/` root. Existing destinations, network errors, archive errors, or verification errors fail; it never searches tags, retries another source, overwrites a destination, or invokes Task3.

The README's Release branch contains two separate commands: first run the explicit fetcher, then pass its verified root through `PREBAKED_ROOT` together with `ARTIFACT_SOURCE=prebaked`. Task3 never downloads or changes artifact source on the reviewer's behalf.

README documents one canonical path matching the recorded gate result. Git LFS is not introduced because D21 selected GitHub Release for the over-limit branch.

Before any bundle push, default-branch update, Release creation, or asset upload, stop at the §2 external publication gate and request user approval for the exact remote/ref/SHA/tag/asset inventory. Packaging verification may complete locally before that approval; publication may not.

---

## 16. Phase 7 — Documentation and Paper Suggestions

### Task 7.1: AE README

**File:** `SC26-AE/README.md` (English)

Required sections:

1. System/repository/image identifiers and pinned commits.
2. Hardware and virtual topology: Task1 one GPU; Task2 exactly two GPUs; Task3 prebaked CPU-only; H800 is the qualified GPU; all simulated models use 8 GPUs per virtual node (`LOCAL_SIZE=8`). The Task3 CPU-memory statement is filled only after Phase 8 records per-model peak RSS and a successfully tested host-memory allocation; the plan does not predeclare an unmeasured 32 GiB minimum.
3. Setup command, recommending:

```bash
GROUPED_GEMM_SOURCE=archive bash SC26-AE/setup.sh
```

4. All nine commands, including explicit Task3 examples:

```bash
ARTIFACT_SOURCE=fresh bash SC26-AE/task3_qwen3_a30b.sh
ARTIFACT_SOURCE=prebaked bash SC26-AE/task3_qwen3_a30b.sh
```

5. `QUICK=1` semantics and whether QUICK is Task3-compatible based on I1 evidence.
6. D16 runtime estimate, measured times, and whether fresh full capture is release-supported per model.
7. Output tree, manifest verification, report field definitions, and why component sums need not equal step span.
8. Task2 numeric evidence interpretation.
9. Canonical prebaked distribution path from D21.
   - For `regular_git`, Task3 uses the verified in-repository root.
   - For `github_release`, show the explicit fetch command, its versioned output root, manifest/hash verification evidence, and the separate `PREBAKED_ROOT=<verified-root> ARTIFACT_SOURCE=prebaked ...` Task3 command. Do not describe download as automatic.
10. Fail-fast troubleshooting by root cause; no alternate-source instructions presented as automatic recovery.
11. Communication weak-validation statement: `analytical` is canonical; collective-sim is optional background only if public dependency verification remains valid.

### Task 7.2: Tex change suggestions

**File:** `task_memory/task_2026-07-15_sc26_ae_workflow/tex_change_suggestions.md`

Each entry contains exact old wording, proposed new wording, evidence path, and reason. At minimum:

- Single `taskN.sh --model` entries → nine per-model entries.
- Task2 one GPU → two GPUs.
- Task3 output → rank0 step span, forward/backward/optimizer scheduled-duration sums, and simulator wall-clock.
- Workflow source → explicit `ARTIFACT_SOURCE`.
- Task1 outputs → execution graphs, memory JSON, Nsight artifacts when enabled, and summary.
- Communication validation → `analytical` weak-validation scope.
- Paper duration estimates → measured setup/task runtimes.

Do not modify `sc26-ad.tex`; the user applies suggestions.

---

## 17. Phase 8 — GPU Dry-Run and Clean-Clone Rehearsal

### Task 8.1: Environment qualification

Before launch, reread `/data/ycfeng/stepfun-env-handbook/guidence.md` and `task_memory/env_handbook.md`.

One-GPU predict/live templates:

```bash
rlaunch --predict-only \
  --charged-group=codesign \
  --private-machine=group \
  --positive-tags=h800 \
  --gpu=1 --cpu=4 --memory=8192 \
  --predict-node-num=10 \
  --backoff-limit=1 \
  -- bash -lc 'true'

rlaunch \
  --charged-group=codesign \
  --private-machine=group \
  --positive-tags=h800 \
  --gpu=1 --cpu=4 --memory=8192 \
  --backoff-limit=1 \
  --image hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef \
  --volume /data:/data \
  --workdir /data/ycfeng/Megatron-LM \
  -- bash
```

Task2 uses the same single-node recipe with `--gpu=2`; before live allocation, use the matching `--predict-only` command. Record actual CPU/memory needs from the first qualified run and update README evidence; do not guess a smaller requirement.

Inside the image verify exact Python, torch, CUDA, grouped-gemm, `nsys`, and `ncu` versions. Never override platform-injected `NCCL_*` variables.

For CPU-only prebaked Task3 qualification, record the host CPU model/count, total available memory, per-model peak RSS, and the explicit memory allocation under which all three runs complete. Report measured values first; publish a README minimum only when that allocation was actually tested successfully. Do not infer the requirement from the earlier unmeasured 32 GiB assumption.

### Task 8.2: Nine-entry rehearsal

- [ ] Setup from a clean recursive clone with explicit grouped-gemm source.
- [ ] Run Task1 QUICK for all models and the D16-qualified full captures.
- [ ] Run one Task2 build and two verified reuses.
- [ ] Run Task3 prebaked for all models and every release-supported fresh path.
- [ ] Record exit codes, elapsed seconds, peak GPU memory, trace/memory counts, Nsight sizes, Task2 numeric metrics, Task3 report values, artifact sizes, and checksums.
- [ ] For each CPU prebaked Task3 run, record peak RSS in KiB and GiB, host memory allocation, and the maximum/minimum observed across the three models.
- [ ] No numerical accuracy-vs-ground-truth threshold is applied; structural validity, finiteness, positivity, provenance, and workflow completion are the criteria.

### Task 8.3: Public clean-clone gate

After the §2 external publication gate has approved and published the exact remote/default-branch/SHA/assets, use a new temporary path:

```bash
git clone --recursive https://github.com/fwyc0573/sc26-reproduce.git
cd sc26-reproduce
git submodule status --recursive
```

Verify the checked-out main commit is the approved publication SHA, the sim-engine `sc26-ae` gitlink, Echo pin, and nested collective-sim commit are publicly fetchable, and any Release locator resolves the approved immutable asset. Execute setup and the release command matrix without editing repository files. Preserve the rehearsal directory until evidence is archived; do not delete it automatically.

---

## 18. Phase 9 — Final Validation, Review, and Archive

### Task 9.1: Comprehensive regression

Run targeted unit tests first, then integration, e2e, shell syntax, Python compile/static checks, existing affected suites, and the nine-entry rehearsal. Test report path:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-15_sc26_ae_workflow.md
```

The report records environment, exact reproducible commands, PASS/FAIL counts, exit codes, logs, and actual numeric comparisons. For Task3 it records each model's step span, three component sums, diagnostic sum, load, execution, wall-clock, peak CPU RSS in KiB/GiB, and tested host-memory allocation. For Task2 it records rows, all five folds, average/test MSE, reload delta, scaler counts, and prediction samples. For Task1 it records rank counts, file counts, per-rank/maximum memory, probe/full elapsed time, estimate, gate result, and artifact sizes.

Any failure stops finalization, receives root-cause analysis in `issues.md`, is fixed through RED/GREEN evidence, and reruns all affected tests.

### Task 9.2: Independent code review and verification

Record each checkpoint in `review.md` with:

```text
Target Component/Phase
Reviewer Agent Identity
Inspected Artifacts
Identified Issues/Anomalies
Remediation/Verification Code Actions Taken
```

Use separate author/reviewer lanes for setup, Task1, Task2, scheduler/reporter, Task3, packaging/docs, and final evidence. Important `BLOCK` verdicts go to the user.

### Task 9.3: Final task archive

- Update `progress.md` and close/resolution status in `issues.md`.
- Write `summary.md` in English with required sections: `Task Overview`, `Deliverables Inventory` with exact paths and SHA256 hashes, `Validation Status` matrices, and `Open Items/Future Extensions`.
- Write `lessons.md` only with verified reusable lessons.
- Confirm no pending required work, no known errors, tests pass, public artifacts resolve, and evidence is complete before claiming completion.

---

## 19. Issue Disposition Matrix

| Issue | Plan disposition | Resolution gate |
|-------|------------------|-----------------|
| I1 MoE subset compatibility | QUICK remains smoke-only by default | Full Task3 structural/provenance run in Task 2.4/5.3 |
| I2 slowdown assets chain | Builder wired to atomic trace capture plus independently identified predictor run; author packaging remains mandatory when fresh AE runtime is gated off | Fresh-chain e2e in Task 5.3 and complete Phase 6 bundle |
| I3 tex drift | Exact suggestion list | Task 7.2 |
| I4 DSV3 memory/runtime | Existing smoke profile reused; runtime evidence required | Task 2.4 |
| I5 scheduler source | Sim-engine built-in only | Task 4.1 |
| I6 GPT adaptation | Minimal env knobs on existing 175B support | Task 2.1 |
| I7 `ncu`/`nsys` availability | Setup/environment gate | Task 1.1 and 8.1 |
| I8 public gitlinks | Current pins verified; new sim-engine pin rechecked | Task 0.3 and 8.3 |
| I9 report semantics | Timeline span + exact-name sums + diagnostic-only | Task 4.2/4.3 |
| I10 distribution size | Strict D21 byte gate | Phase 6 |
| R-I11 Echo checkout pollution | Isolated `git archive` snapshot | Task 3.1 |
| I12 automatic fallback conflict | D23 explicit-source-only | Task 5.1 |
| I13 builder path portability | Outer relative manifest + explicit model/scaler CLI | Tasks 1.3, 5.2, 6.1 |
| I14 setup automatic recovery | Required explicit `GROUPED_GEMM_SOURCE` | Task 1.1 |
| I15 overlap auto mode | Explicit `--overlap-mode on` | Tasks 5.2/5.3 |
| I16 Task3/MoE fake-node-size contract | Task3 fixed to `LOCAL_SIZE=8`; MoE Task1 change only if independent review proves a consumed-field requirement | Gate A review, then Task 2.2/5.2 contract tests |
| I17 stale/mixed runtime outputs | Immutable Task1/Task3 run roots plus post-verification markers | Tasks 2.2/2.3 and 5.1/5.2 |
| I18 prebaked provenance/distribution gap | Internal producer consistency, full metadata byte gate, explicit Release fetch, and separate publication approval | Tasks 5.1, 6.1/6.2, 8.3 |
| I19 unmeasured Task3 CPU memory | Remove 32 GiB assumption; publish only measured RSS and tested allocation | Tasks 5.3, 7.1, 8.1/8.2 |

---

## 20. Requirement and Decision Traceability

### Requirements R1–R15

| ID | Covered by | Verification evidence |
|----|------------|-----------------------|
| R1 | Goal, output architecture, Phases 1–8 | Clean-clone nine-entry rehearsal |
| R2 | §5 output tree, §7.3/7.5 schemas | Task1/2/3 manifests and reports |
| R3 | Gate A hard stop | Docs-only diff and user approval gate |
| R4 | Scope/non-goals, analytical backend | Task3 argv contract and README |
| R5 | Model keys and EXP terminology | Common enum/config tests |
| R6 | Nine entries and README | Entry count test and reviewer commands |
| R7 | Task1 source scripts and atomic capture | Task1 contract/GPU evidence |
| R8 | Task2 shared bundle and Task3 consumption | Fresh-chain e2e |
| R9 | Phase 0 branches/worktree plus Gate B existing three-task reconnaissance before feature edits | Git/reachability evidence and recorded Task1/Task2/Task3 runtime outputs |
| R10 | Scheduler contract and Task1 distinction | Scheduler unit tests + README |
| R11 | In-scope orchestration, non-goals | End-to-end completion; no calibration patch |
| R12 | §4 model and 8-GPU/node matrix | Static argv/schedule tests and GPU logs |
| R13 | Thin entries/shared focused helpers | File map/code review |
| R14 | Setup entry and AE image | Setup integration/environment report |
| R15 | Safety constraints/Phase 0 | Worktree status evidence |

### Decisions D1–D23

| ID | Covered by | Verification evidence |
|----|------------|-----------------------|
| D1 | Phase 8 public repo clone | Default-branch clean clone |
| D2 | Echo no-commit policy, sim-engine branch | Submodule status and public fetch |
| D3 | Task 0.2 explicit baseline list | Cached name list + Lore commit |
| D4 | Global constraint #1 | Exactly nine executable entries |
| D5 | Phase 3 shared Task2 core | One build, two verified reuses |
| D6 | §4 MoE topology | Config/scheduler assertions |
| D7 | Existing DSV3 smoke profile | Static and GPU validation |
| D8 | Rank-scope rules | QUICK/full evidence and I1 gate |
| D9 | Explicit analytical backend | Task3 argv test |
| D10 | Built-in sim-engine reporter | Reporter unit/integration output |
| D11 | Complete prebaked bundles | Phase 6 manifests/distribution |
| D12 | §5 output tree and markers | Path/marker tests |
| D13 | Task 7.2 only | Tex suggestion file; tex unchanged |
| D14 | Phase 8 image/rlaunch flow | GPU test report |
| D15 | Scheduler automation, precision, overlap, setup, report | Phase 1/2/4/5 tests |
| D16 | Atomic capture and 7200-second gate | Probe/full timing fields |
| D17 | `git archive` Task2 snapshot | Snapshot unit test + clean status |
| D18 | Sim-engine scheduler only | File-diff scope + scheduler tests |
| D19 | Rank0 metric formulas | Reporter unit fixtures |
| D20 | Diagnostic-only, fail-fast | Negative reporter tests |
| D21 | Strict size gate then regular Git/Release | Byte-boundary tests + distribution manifest |
| D22 | Superseded preference retained only in requirements history | No automatic-selection code/path |
| D23 | Explicit source only | Task3 source negative tests |

---

## 21. Final Acceptance Criteria

1. Exactly nine public task entry scripts exist and are executable; no dispatcher exists.
2. All scripts run without reviewer edits from a public recursive clone and the documented image.
3. Task1 uses one physical GPU, produces one trace and one memory JSON per selected rank inside a new immutable capture root, records actual trace counts, per-rank/maximum peak memory, elapsed time, capture ID, actual `capture_runtime.fake_gpus_per_node`, sizes, and checksums, and publishes a marker only after verification.
4. Fresh Task1 Nsight artifacts come from one capture around the complete selected-rank loop; trace/SQLite `cmd_uid` provenance is validated by successful strict asset building. Task2 artifacts keep a separate `predictor_run_id`; Task3/prebaked manifests record both identities.
5. The rank0×256 estimate and 7200-second release decision are recorded for both MoE models; no runtime source switching occurs.
6. Task2 uses exactly two GPUs, leaves the Echo submodule clean, archives the pinned gitlink commit, produces one shared predictor bundle, and records all §7.3 numeric metrics.
7. Scheduler output has 8/4/4 stage files, 48/16/16 forward and backward operations per stage, one optimizer per stage, correct PP shapes, and bf16 dtype.
8. Task3 always passes `analytical`, slowdown enabled, `--overlap-mode on`, `--local-size 8`, `--database-dir` equal to the canonical resolved `--trace-dir`, explicit model/scaler paths, explicit source, and explicit report output; outer manifests record `simulation_topology.local_size=8`.
9. Each model's Task3 report has the exact schema, all three exact target operations present, all finite nonnegative values, strictly positive step span, and `wall_clock = round(round(load, 6) + round(execution, 6), 6)`. Markdown equals JSON numerically.
10. Invalid rank0 timelines or any missing target operation fail; diagnostic `comp+comm` never replaces the main metric.
11. Fresh/prebaked bundles are never mixed. Missing/partial/corrupt/wrong-provenance selected bundles fail immediately.
12. The outer manifest uses relative paths, verifies every SHA256/size, and remains usable after relocating the bundle.
13. D21 chooses exactly one canonical distribution path based on every staged regular file, including metadata. Release distribution, when selected, requires an explicit verified fetch before Task3; README matches the selected path.
14. Unit, integration, e2e, affected regression, GPU smoke, and clean-clone rehearsals pass with evidence in the required test report.
15. `review.md` contains author and independent reviews; any WATCH is tied to a test gate; no unresolved BLOCK remains.
16. Prebaked verification accepts a historical producer main-repository commit that differs from consumer `HEAD` only when the distribution/nested manifests and payload hashes are internally consistent; fresh verification still binds to the current producer checkout.
17. Task3 publishes only versioned verified run markers, records measured CPU peak RSS and tested memory allocation, and never overwrites a fresh run with a prebaked run or vice versa.
18. No push, default-branch update, Release creation, or asset upload occurs without explicit approval of the exact external target and immutable identifiers.
19. Root/source code changes remain minimal and traceable to R1–R15 or D1–D23; root legacy scheduler and protected overlap-review branch are untouched.

---

## 22. Execution Handoff

Gate A ends by delivering this reviewed plan to the user. Implementation remains blocked until the user explicitly approves moving to Phase 0. After approval, the recommended execution mode is low-concurrency `subagent-driven-development` with separate ownership for main-repo orchestration, sim-engine scheduler/reporter, and verification, while final integration and evidence remain serial.
