# Design — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Added the strict D16 orchestration design: independent MoE rank-0 preflight, observation-only QUICK behavior, fail-fast full gate, and no selected-loop artifacts above threshold |
| 2026-07-20 | Added the model-aware I53/D16 timing contract: GPT representative timing is diagnostic with no D16 gate, while MoE retains rank0×256 pending independent preflight |
| 2026-07-20 | Reclassified I55 from pending implementation to completed local wrapper-only semantic hardening; approved digest and worker qualification remain pending |
| 2026-07-20 | Recorded the I55 nested-interpreter RED reproduction and a pending wrapper-only fixed-chain design; no runtime implementation or gate promotion |
| 2026-07-19 | Added D30 latest-user overlay: test/validation/rehearsal-exposed AE defects may be repaired autonomously while the acceptance target and evidence classes stay unchanged |
| 2026-07-19 | Recorded the current three-task pipeline, artifact/provenance boundaries, D29 control-plane overlay, and incomplete qualification state |

## 1. Design objective

The workflow is designed for an AE user who should be able to clone the repository, prepare the
approved runtime, and run one model-specific shell entry per task without hand-editing module code.
The deliverable is a reproducible chain, not merely three independent demos:

```text
Task1 workload capture  ->  Task2 slowdown predictor  ->  Task3 e2e simulation/report
       traces + memory       model + scaler + metrics       rank0 report + manifests
```

The current design remains **INCOMPLETE** until the real qualification and release pre-dataset
gates are closed.

## 2. Public entry surface

The main repository owns exactly nine model-specific entries under `SC26-AE/`:

- Task1: `task1_gpt175b.sh`, `task1_qwen3_a30b.sh`, `task1_dsv3.sh`;
- Task2: `task2_gpt175b.sh`, `task2_qwen3_a30b.sh`, `task2_dsv3.sh`;
- Task3: `task3_gpt175b.sh`, `task3_qwen3_a30b.sh`, `task3_dsv3.sh`.

Shared libraries hold common validation and artifact handling. Public entries select the model and
delegate to the shared implementation; they do not silently select another model, source, or
runtime.

## 3. Pipeline interfaces

### Task1 — workload/tracer capture

Task1 invokes the instrumented Megatron workload in the requested topology and records traces,
memory JSON, optional Nsight/SQLite evidence, source identity, effective arguments, and a verified
manifest. The run root is versioned and immutable. Warmup/profile settings are explicit and must
be reflected in the manifest.

### Task2 — slowdown dataset and predictor

Task2 runs the pinned Echo-slowdown logic from an isolated snapshot rather than mutating the pinned
submodule checkout. It produces a predictor, scaler, dataset/validation metrics, and source-bound
provenance. A local synthetic predictor is useful for wiring tests, but it is not a real slowdown
dataset.

### Task3 — simulator composition

Task3 consumes either a caller-selected `fresh` bundle or a caller-selected `prebaked` bundle. It
passes explicit trace/database, scheduler, overlap, slowdown model, scaler, and topology arguments
to the canonical sim-engine. It validates the rank0 report and all nested/outer manifests before
publishing a verified marker. Missing fields, checksum errors, existing run roots, and builder
failures stop the selected path immediately.

## 4. Artifact and provenance model

Each run has a versioned root and a manifest containing:

- relative artifact paths, byte sizes, and SHA256 values;
- producer and compatibility commits, including nested submodule identity;
- image/runtime identity and effective topology;
- capture/run identity and explicit artifact source;
- semantic report values and validation status.

The outer repository gitlink is not sufficient to describe a nested worktree. If
`megatron-sim-engine` is dirty, the workflow must either make that state a legitimate recorded
source revision or fail before publishing a qualified marker. It must never present dirty code as
the clean pinned commit.

## 5. Validation layers

The design separates four layers so that a green local test cannot accidentally open a real gate:

1. **Shell/schema layer** — syntax, enums, required arguments, and fail-fast branches;
2. **Synthetic integration layer** — deterministic fixtures and manifest/marker ordering;
3. **Real qualification layer** — approved image/digest, H800/NVML/CUDA, actual workload and tools;
4. **Release layer** — complete real pre-dataset, relocation, checksum, data quality, and clean-clone
   reproducibility.

The 2026-07-19 Task3 report passed layers 1–2 for its covered local fixtures and the directly
related sim-engine tests (`45/45`), but explicitly did not pass layers 3–4.

## 6. D29 governance overlay

D29 changes the control-plane response, not the product acceptance target. A test, audit, schema,
validator, documentation, or orchestration defect may be repaired in the same session when it
serves the AE chain. The repair is still RED→root cause→minimal fix→GREEN→regression, and it must
preserve assertions, provenance, checksums, no-fallback behavior, and evidence labels.

Real GPU/image/quota/scheduler/product/runtime/workload and real pre-dataset failures remain outside
the autonomous lane.

## 6.1 D30 latest test-failure overlay

D30 supersedes that narrow D29 scope interpretation for the current execution. Any defect exposed by
a test, validation, rehearsal, audit, or qualification check can be diagnosed, decided, and repaired
autonomously when the change directly serves the one-click AE shell entries or reusable pre-dataset,
including a task-scoped implementation fix proven by the check. The repair still follows
RED→root-cause→minimal fix→GREEN→regression and cannot weaken acceptance thresholds, provenance,
checksums, evidence classes, no-fallback behavior, or data-quality requirements. A failed real check
stays closed until it passes; external authority/resource problems, destructive or irreversible
actions, external publication, and materially scope-changing refactors remain outside this lane.

## 7. Current design state

Local Task1/Task3 control-plane paths and sim-engine contracts have executable evidence, but the
following are still open:

- the sim-engine submodule worktree is dirty while Task3 provenance currently records only the outer
  gitlink;
- the current `v1.2-ae` image digest and clean runtime qualification are not established;
- exact-two-H800 Echo/integrated B1 qualification is blocked;
- fresh real traces, predictor assets, simulator outputs, and final pre-dataset quality/checksum
  evidence are absent.

Therefore the design is ready for bounded local repair and later qualification, but not for a final
`AE-ready` claim.

## 7. I53/D16 timing contract (local observation, qualification pending)

Task1 records one wall-clock interval per selected fake rank in a run-owned timing log. The
rank-0 value is useful for audit and diagnostics, but the current implementation observes it inside
the selected-rank capture and therefore cannot make the preflight decision required by D16.

The metadata contract is model-aware:

| Model | `d16_gate_applicable` | `estimate_rank_count` | 7200-second fields |
|---|---:|---:|---|
| GPT-175B | `false` | `8` representative ranks | omitted |
| Qwen3-A30B | `true` | `256` | required |
| DeepSeek-V3 | `true` | `256` | required |

The MoE formula remains `estimated_full_seconds = single_rank_elapsed_seconds × 256`, with
`pass|prebaked_required` derived against `7200`. GPT's eight-rank value is a selected
representative-capture estimate and must not be described as a 1024-rank estimate or a D16 gate.
Any missing, non-boolean, model-inconsistent, or extra gate field fails before marker publication.

Closing I53 still requires an independent rank-0-only preflight capture, real SQLite/NVTX
semantics, canonical Nsight identity, and producer provenance. Local synthetic tests prove only
the schema and fail-fast control plane; they do not promote Gate B1 or either pre-dataset.

## 8. I55 nested-interpreter chain (local hardening implemented; qualification pending)

### 8.1 Observed root cause

The outer Task2 wrapper invokes the fixed Echo interpreter by absolute path, but the pinned
`Echo-slowdown/update_configs.py` discovers `python` through `which`. The generated
`kernel_metric`, `merge`, `slowdown_collection`, and `training_testing` configurations then carry
that PATH-dependent value into subordinate shell scripts. Those scripts use both bare `python` and
the configured `${python_path}`, so an outer fixed invocation does not by itself bind the nested
producer chain.

The durable CPU-only reproduction is recorded in
`logs/i55-nested-interpreter-red-20260720.log` (1,431 bytes,
SHA256=`0551da75f50e9801e875ba55ab036ce31b42c54d187c2239766da2050ac551f4`). It observed:

| Observation | Value |
|-------------|-------:|
| `PROBE_RC` | `1` (the fixture intentionally stops at the later mocked metrics boundary) |
| `EXTERNAL_INVOCATIONS` | `1` |
| `NESTED_SENTINEL` | `created` |
| Configs carrying the external path | `4/4` |

This is a control-plane audit only. It did not run GPU/RJob/Docker, did not alter pinned Echo, and
is not a qualification result.

### 8.2 Implemented wrapper-only semantic contract

The candidate seam was implemented only in the wrapper-owned library and tests; pinned
`Echo-slowdown` source remains unchanged. The implementation now:

1. Binds the fixed absolute interpreter directory at the front of `PATH` and checks requested,
   canonical, and observed SHA256 identity before nested work.
2. Validates all four generated configs before and after `run_all.sh`, requiring regular files,
   valid object JSON, safe absolute `python_path`, lexical equality to the fixed requested path,
   and canonical executable/hash equality.
3. Archives post-update config bytes and writes an exact-schema `interpreter_binding.json` with
   `status=bound` and `automatic_fallback=false`; provenance and the generic manifest reference
   the sidecar and all four archives.
4. Uses duplicate-key rejection for sidecar, archive, provenance, manifest, and reuse-evidence
   reads, and rejects non-canonical artifact-only path spellings.
5. Applies artifact semantic verification to real-evidence bundles even when the consumer is
   synthetic; only live executable comparison is mode-specific. A missing run-root argument fails
   with a controlled non-zero diagnostic.

The final local contract is backed by parser negatives=`11`, duplicate-key negatives=`4`, sidecar
tamper negatives=`9`, interpreter `PASS_COUNT=12`, evidence-mode `PASS_COUNT=5`, and a full
affected regression with `94 passed` in the artifact/sealer/package pytest subset. The formal
numbers and byte identities are in
`test_report_2026-07-20_i55_interpreter_chain.md`.

### 8.3 Authority and qualification boundary

The implementation does **not** create an approved interpreter digest. The recorded executable
SHA256 is an observed synthetic-fixture value and proves only the local byte identity checked by
the test. A canonical v1.2-ae worker, immutable image authority, external issuer, producer source
snapshot, frozen inputs, trusted-root/TOCTOU closure, and Nsight-tool binding are separate pending
inputs. Therefore this local GREEN does not close I55, I51, I53, I54, I56, I57, or I58 and does not
promote Gate B1, either pre-dataset, or `AE-ready`.

### 8.4 Independent review and authority boundary

The independent read-only reviewer `/root/audit_i54_i55` confirmed the propagation chain and the
semantic test boundary. The local wrapper-only implementation is now verified by the final regression,
but the setup contract still lacks an authority-approved immutable interpreter digest and canonical
worker evidence. Accordingly, this checkpoint is **local GREEN with WATCH**, not I55 closure. I55
remains `OPEN / HIGH / BLOCK`, and the global release state remains unchanged.

## 9. D16 preflight orchestration design (local controller)

For the two 256-rank MoE models, Task1 now has an explicit preflight boundary before a full
selected-rank capture:

1. Allocate an isolated rank-0 preflight root and run exactly one rank-0 controller invocation.
2. Validate its trace, memory, provenance copy, and strict timing report before making a gate
   decision.
3. Compute `estimated_full_seconds = rank0_elapsed_seconds × 256` with the frozen threshold `7200`.
4. In `QUICK=1`, record the result as observation (`d16_gate_enforced=false` and
   `gate_decision_applied=false`) and continue only the four-rank smoke capture.
5. In full mode, enforce the result before creating the full run root. A pass starts exactly 256
   selected-rank calls; an above-threshold result exits `2` with no selected-loop call, full root,
   marker, or automatic source switch.

The GPT-175B path remains diagnostic: its eight representative ranks have no MoE D16 gate fields.
The preflight report is copied into the final full manifest only as a strict provenance artifact;
raw preflight trace/memory/Nsight files remain isolated from the selected-capture manifest.
This design is a local control-plane contract and does not claim real timing accuracy or H800
qualification.
