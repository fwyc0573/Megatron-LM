# Suggested `sc26-ad.tex` Changes for the SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Bound every suggestion to verbatim wording from the reviewed paper snapshot and added the explicit source hash |
| 2026-07-19 | Added evidence-bound wording suggestions for the nine-entry AE workflow and its measured artifact contracts |

This file is a suggestion list only. It does not modify
`2026-SC-first-submission/sc25-ad-ae/for-paper-authors/sc26-ad.tex`; the paper author
applies the final wording after real qualification evidence is available.

The suggestions below were checked against this exact draft snapshot:

```text
SC26_AD_SOURCE_PATH=2026-SC-first-submission/sc25-ad-ae/for-paper-authors/sc26-ad.tex
SC26_AD_SOURCE_BYTES=8693
SC26_AD_SOURCE_SHA256=31d2053436e8a058bc3ee2a4d32868625d737c8260a058c3a4bd07b3c57adf17
```

If the author changes that file, compare the new draft against these copied
passages before applying the suggestions; the hash above is a review binding, not
a claim that the paper source is frozen.

## 1. Replace the single-entry task wording

**Current wording (copied verbatim from the reviewed draft):**

> For $T_1$ (Workload Tracing, validating $C_2$), the entry is \texttt{SC26-AE/task1.sh}.
> It invokes the ex-situ tracer on a single GPU, sequentially emulating each rank to produce per-rank execution graphs containing computation/communication operators and GPU memory footprints.
> Users specify the model configuration and parallelism setting via command-line arguments (e.g., \texttt{--model gpt-175b --tp 8 --pp 16 --dp 2}).

**Suggested replacement:**

The public surface contains one explicit entry for each model and task. The Task1
entries are `SC26-AE/task1_gpt175b.sh`, `SC26-AE/task1_qwen3_a30b.sh`, and
`SC26-AE/task1_dsv3.sh`; the corresponding Task2 and Task3 entries use the same
model suffixes. The evaluator selects a model by choosing the documented entry,
not by editing a generic dispatcher or reconstructing parallelism flags.

**Evidence path:** `SC26-AE/README.md`, the nine executable
`SC26-AE/task{1,2,3}_*.sh` entries, and `tests/integration/test_sc26_ae_task1_contracts.sh`.

**Reason:** The AE contract forbids a model-selecting dispatcher and requires an
auditor to identify the exact command used for every model/task pair.

## 2. Clarify Task1's single-device tracing contract

**Current wording (copied verbatim from the reviewed draft):**

> For $T_1$ (Workload Tracing, validating $C_2$), the entry is \texttt{SC26-AE/task1.sh}.
> It invokes the ex-situ tracer on a single GPU, sequentially emulating each rank to produce per-rank execution graphs containing computation/communication operators and GPU memory footprints.
> Users specify the model configuration and parallelism setting via command-line arguments (e.g., \texttt{--model gpt-175b --tp 8 --pp 16 --dp 2}).

**Suggested replacement:**

Task1 runs the selected fake ranks sequentially on one physical GPU. Compute is
executed, while target multi-GPU communication is recorded as metadata or shape
simulation. Each invocation receives a fresh `capture_id` and writes its trace,
memory JSON, optional Nsight artifacts, summary, and SHA256 manifest below an
immutable run directory. A completion marker is published only after semantic and
provenance validation succeeds. The summary records selected-rank count, trace
file count, memory-file count, elapsed time, peak-memory values, and the explicit
model topology.

**Evidence path:** `tests/integration/test_sc26_ae_task1_contracts.sh`,
`tests/e2e/test_sc26_ae_task1_smoke.sh`, and
`task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-18_task1_local_contracts.md`.

**Reason:** These checks bind rank inventory, memory JSON, capture identity,
manifest checksums, and optional Nsight artifacts without turning a synthetic
fixture into a real qualification claim.

## 3. Make the Task2 hardware requirement explicit

**Current wording (copied verbatim from the reviewed draft):**

> For $T_2$ (Slowdown Dataset Collection and Training, validating $C_4$), the entry is \texttt{SC26-AE/task2.sh}. It collects per-kernel performance features under overlapping conditions on a single GPU, trains the XGBoost predictor on the collected dataset, and reports per-kernel slowdown predictions, validating that the model produces reasonable adjustments under overlap-heavy settings.

**Suggested replacement:**

For Task2, the three model-labelled entries share one isolated Echo-slowdown
collection/training implementation and require exactly two visible GPUs for a
real run. The pinned Echo checkout remains clean and read-only; collection,
processing, training, save, reload, and prediction execute in a fresh filtered
snapshot. The output records one `predictor_run_id`, the model/scaler pair,
dataset row count, all five validation-fold MSE values, average and test MSE,
reload delta, scaler dimensions, and a deterministic prediction sample. A
synthetic CPU fixture is available for contract testing only and is labelled as
non-qualification evidence.

**Evidence path:** `tests/integration/test_sc26_ae_task2_contract.sh`,
`tests/unit/test_sc26_ae_task2_interpreter_contract.sh`, and
`task_memory/task_2026-07-15_sc26_ae_workflow/test_report_2026-07-19_task2_shared_pointer_verified.md`.

**Reason:** Real Echo collection needs two distinct GPUs and a pinned Python
environment; the separate synthetic fixture must remain visibly non-qualified.

## 4. Define Task3's primary metric and overlap semantics

**Current wording (copied verbatim from the reviewed draft):**

> The output is the predicted training step time with a per-stage breakdown.

**Suggested replacement:**

Task3 reports the final rank0 timeline span as
`rank0_step_time_ms = max(finish_time) - min(join_time)`. It also reports exact
scheduled-duration sums for `forward_step`, `backward_step`, and
`optimizer_step`, plus simulator load time, execution time, and wall-clock time.
Because overlap is enabled, the three operation sums are diagnostics and are not
required to add up to the final timeline span. Any unavailable or invalid rank0
timeline fails rather than being replaced by a `comp+comm` diagnostic.

**Evidence path:** `tests/integration/test_sc26_ae_task3_contract.sh`,
`tests/unit/test_sc26_ae_task3_contracts.sh`, and
`megatron-sim-engine/tests/unit/test_rank0_report.py`.

**Reason:** The rank0 timeline span is the overlap-aware primary metric; exact
operation sums and `comp+comm` are diagnostic values and must not replace it.

## 5. State the explicit artifact-source contract

**Current wording (copied verbatim from the reviewed draft):**

> Workflow: $T_1, T_2 \rightarrow T_3$ ($T_3$ can be run independently using pre-traced workloads; $T_1, T_2$ can be run independently on a single GPU). Detailed usage instructions are provided in \texttt{SC26-AE/README.md}.

**Suggested replacement:**

Task3 accepts exactly one explicit source selection:

```text
ARTIFACT_SOURCE=fresh
ARTIFACT_SOURCE=prebaked
```

`fresh` validates the current Task1 capture, the independent Task2 predictor run,
and the source commits before building slowdown assets. `prebaked` validates the
complete portable distribution and consumes only the explicitly supplied verified
root. Missing, partial, stale, mixed, or checksum-inconsistent inputs fail fast;
Task3 never changes source, downloads an artifact, or retries through the other
branch.

**Evidence path:** `SC26-AE/lib/task3_simulation.sh`,
`tests/integration/test_sc26_ae_task3_contract.sh`, and
`tests/integration/test_sc26_ae_task3_portability.sh`.

**Reason:** Explicit source selection prevents stale or partial inputs from being
silently substituted, preserving checksum and provenance auditability.

## 6. Update the workflow graph and hardware summary

**Current wording (copied verbatim from the reviewed draft):**

> Workflow: $T_1, T_2 \rightarrow T_3$ ($T_3$ can be run independently using pre-traced workloads; $T_1, T_2$ can be run independently on a single GPU). Detailed usage instructions are provided in \texttt{SC26-AE/README.md}.
> A single NVIDIA GPU (H800, A800, A100, etc.) for workload tracing and kernel slowdown dataset generation.
> For simulation-only mode (using the pre-traced dataset we provide), a CPU-only environment with $\geq$32 GB RAM is usually sufficient.

**Suggested replacement:**

The workflow is `Task1 + Task2 -> Task3` for a fresh run. Task3 can instead consume
the explicitly verified prebaked distribution on a CPU-only host. Task1 uses one
physical GPU; real Task2 uses exactly two GPUs; Task3's analytical simulator uses
`LOCAL_SIZE=8` as the virtual node size. The README records measured setup and
execution times, peak memory, artifact sizes, and checksums rather than relying on
unmeasured duration or RAM claims.

**Evidence path:** `SC26-AE/README.md`, `plan.md` sections 16--18, and the
Task3 prebaked CPU e2e report.

**Reason:** The hardware summary must match the actual AE launch topology while
keeping measured resource requirements separate from pending qualification data.

## 7. Expand the output list with provenance

**Current wording (copied verbatim from the reviewed draft):**

> The artifact produces the following outputs for each task:
> \begin{itemize}
> \item $T_1$: Per-rank execution graphs, GPU memory usage reports, and a summary log confirming successful tracing for all ranks.
> \item $T_2$: Trained slowdown predictor model, per-kernel slowdown predictions, and aggregate statistics verifying the predictor functions correctly.
> \item $T_3$: Predicted training step time (ms) with breakdown into exclusive computation, exclusive communication, pipeline bubble, and overlap time.
> \end{itemize}

**Suggested replacement:**

For Task1, list trace files, non-empty memory JSON, optional Nsight/SQLite files,
summary, `capture_id`, and `artifact_manifest.json`. For Task2, list the shared
predictor bundle, `predictor_run_id`, metrics JSON/Markdown, and source snapshot
identity. For Task3, list the resolved schedule, slowdown assets, report JSON and
Markdown, `simulation_run_id`, and the outer manifest. Every manifest uses
root-relative paths and records file size plus SHA256. The paper should distinguish
local/synthetic contract evidence from real H800 qualification evidence.

**Evidence path:** `SC26-AE/tools/artifact_manifest.py`,
`tests/unit/test_sc26_ae_artifact_manifest.py`, and the Task3 provenance report.

**Reason:** Root-relative paths, byte counts, SHA256 values, and separate capture
and predictor identities are required for a portable reusable pre-dataset.

## 8. Narrow the communication-validation claim

**Current wording (copied verbatim from the reviewed draft):**

> At least 32 GPUs are needed to run the collective communication benchmarks, so we provide pre-profiled datasets and optional communication simulators as backends.

**Suggested replacement:**

The canonical AE path uses the analytical communication backend with explicit
`--overlap-mode on`. Collective-simulation support may be described as background
infrastructure only when its public dependency and commit are independently
verified; it is not an implicit runtime fallback and is not required to support the
primary paper result.

**Evidence path:** `SC26-AE/lib/task3_simulation.sh` command logs,
`tests/integration/test_sc26_ae_task3_contract.sh`, and plan decision D9.

**Reason:** `analytical` is the intentional weak-validation backend; describing
collective-sim as an implicit fallback would overstate what the AE verifies.

## 9. Replace unmeasured duration claims with evidence placeholders

**Current wording (copied verbatim from the reviewed draft):**

> The expected execution time for all validation tasks mainly depends on the target cluster's scale. For a dense model in a 1k+ GPU cluster, it may take about 30 minutes for workload tracing and 10 minutes for E2E simulation. For a MoE model, it usually takes 5--10$\times$ longer than a dense model due to the need for a more fine-grained simulation graph.

**Suggested replacement:**

Report the measured values from the qualified rehearsal: setup elapsed time,
Task1 single-rank probe and full-capture elapsed time (or the documented
prebaked-required decision), Task2 elapsed time, and each Task3 simulator wall-clock
time. Keep any estimate clearly labelled as an estimate and bind it to the recorded
configuration; do not present a local synthetic run as a cluster qualification.

**Evidence path:** Phase 8/9 test reports under
`task_memory/task_2026-07-15_sc26_ae_workflow/`, including the Task1, Task2,
and Task3 local reports.

**Reason:** Measured elapsed times and an explicit prebaked-required decision are
auditable; unmeasured cluster estimates can mislead AE operators and reviewers.

## 10. Hardware wording that must remain evidence-bound

**Current wording (copied verbatim from the reviewed draft):**

> A single NVIDIA GPU (H800, A800, A100, etc.) for workload tracing and kernel slowdown dataset generation.
> For simulation-only mode (using the pre-traced dataset we provide), a CPU-only environment with $\geq$32 GB RAM is usually sufficient.

**Suggested replacement:**

Say that the qualified Task1 path uses one H800 (or the exact hardware recorded by
the final qualification), real Task2 uses two H800 GPUs, and prebaked Task3 is
CPU-only. Fill the CPU memory minimum only after the three model runs record peak
RSS and a successful explicit host-memory allocation. Until then, use a measured
placeholder rather than retaining an unverified `32 GB` guarantee.

**Evidence path:** `plan.md` Phase 8 Task 8.1--8.2 and
`tests/e2e/test_sc26_ae_task3_prebaked_cpu.sh`.

**Reason:** A memory minimum is a qualification result, not a default assumption;
it must be backed by peak RSS and a successful tested allocation for all models.
