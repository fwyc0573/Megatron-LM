# Review — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-16 | Recorded explicit user Gate A approval and execution boundary |
| 2026-07-15 | Recorded independent StepCode Claude WATCH verdict, evidence reconciliation, and plan remediation gates |
| 2026-07-15 | Added Gate A author self-review and remediation record |

## Gate A Author Self-Review

**Target Component/Phase**

Gate A enhanced plan documents; implementation, GPU execution, Git publication, and submodule changes remain out of scope.

**Reviewer Agent Identity**

Codex primary author, self-review lane (`/root`), 2026-07-15. This is not the independent approval lane.

**Inspected Artifacts**

- `task_memory/task_2026-07-15_sc26_ae_workflow/{requirements,plan,notes,issues,progress}.md`
- `examples/update_pretrain_gpt.sh`
- `examples/pretrain_qwen3_30b_a3b_moe.sh`
- `examples/pretrain_deepseek_v3_moe.sh`
- `megatron/training/arguments.py`
- `megatron/training/training.py`
- `megatron/profiler/{cmd,trace_memory,rank_manager}.py`
- `tests/e2e/test_ddp_slowdown_simulate_smoke.sh`
- `megatron-sim-engine/simu_main.py`
- `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`
- `tools/ae/setup_grouped_gemm_v1.sh`
- `docs/ae/grouped_gemm_v1_setup.md`
- Current main-repository status and the protected-worktree boundary recorded in `progress.md`

**Identified Issues/Anomalies**

| ID | Severity | Finding |
|----|----------|---------|
| A-SR-01 | High | Task1 native trace, replay, and memory files use CWD-relative fixed directories, while the prior output contract allowed stale files from older captures to enter a new manifest. |
| A-SR-02 | High | Task3 referenced an undefined independent `DATABASE_DIR`, although the verified slowdown e2e uses the selected trace directory as both trace and operation database. |
| A-SR-03 | High | Requiring a prebaked producer main-repository commit to equal final consumer `HEAD` would be circular because publishing the payload changes `HEAD`. |
| A-SR-04 | High | The GitHub Release branch lacked an explicit download, archive-safety, verification, destination, and `PREBAKED_ROOT` handoff contract. |
| A-SR-05 | Medium | The D21 calculation covered manifest-declared payloads but did not explicitly count nested manifests and `distribution_manifest.json` as final distributed bytes. |
| A-SR-06 | High | Exact-name report aggregation could serialize `0` when `forward_step`, `backward_step`, or `optimizer_step` was absent, hiding an incomplete timeline. |
| A-SR-07 | High | Future sim-engine pushes, default-branch changes, and Release publication lacked a separate exact-target approval gate. |
| A-SR-08 | Medium | The README plan stated a `32 GiB` Task3 CPU minimum without measured peak RSS or a tested memory allocation. |
| A-SR-09 | Medium | Gate B Task2 used relative repository paths after `cd`; Gate B Task1 did not isolate CWD-relative outputs in a new reconnaissance directory. |
| A-SR-10 | Medium | `notes.md` incorrectly stated that both MoE scripts use `SCALING_FAKE_RANK_ORDER`; Qwen uses `FAKE_RANK_ORDER`. |
| A-SR-11 | Medium | The future file map omitted the existing grouped-gemm setup document that must change with the explicit-source installer contract. |
| A-SR-12 | Watch | Task3 requires `LOCAL_SIZE=8`, while Task1 records GPT `fake_gpus_per_node=8` and MoE `fake_gpus_per_node=world_size`; static inspection found only tracer `RankZoo.local_rank/server_id` effects, not a confirmed consumed trace field. Independent adjudication remains required. |

**Remediation/Verification Code Actions Taken**

- No code action was taken. Only task-plan Markdown was changed.
- Added immutable Task1/Task3 run roots, nonexistent-destination checks, runtime CWD isolation, and post-verification markers with stale/partial/marker-traversal tests.
- Bound Task3 `DATABASE_DIR` to the canonical resolved `TRACE_DIR` and added equality assertions for both sources.
- Split `capture_runtime.fake_gpus_per_node` from `simulation_topology.local_size`; the plan records actual source argv instead of silently rewriting MoE history.
- Defined source-aware commit verification: fresh binds to the producer checkout; prebaked verifies distribution/nested manifests, producer/compatibility commits, and payload hashes without final-HEAD equality.
- Changed D21 to scan every regular file in a complete staged regular-Git candidate, including metadata.
- Added explicit single-asset Release fetch, archive-member validation, versioned extraction, nested verification, and separate `PREBAKED_ROOT` handoff; no automatic Task3 download.
- Added the exact-target external publication gate.
- Added fail-fast checks and tests for each missing target operation.
- Replaced the unmeasured `32 GiB` statement with Phase 5/8 peak-RSS and tested-allocation evidence.
- Corrected Gate B absolute repository paths, new-path checks, and versioned Task1 reconnaissance CWD.
- Corrected rank-selector notes and added `docs/ae/grouped_gemm_v1_setup.md` to the future modification map.

**Author Self-Review Result**

Ready for independent StepCode Claude review. This result does not approve Gate A and does not authorize implementation.

## Independent StepCode Claude Review

**Target Component/Phase**

Gate A enhanced plan and its pre-implementation safety/provenance/test gates. No implementation, GPU execution, Git mutation, or submodule mutation was authorized.

**Reviewer Agent Identity**

Independent StepCode Claude advisor via `omx ask claude`, backend `stepcode claude --model 'claude-opus-4-6[1m]' --effort max`, 2026-07-15.

**Inspected Artifacts**

- Advisor artifact: `.omx/artifacts/claude-review-task-memory-task-2026-07-15-sc26-ae-workflow-plan-md--2026-07-15T09-25-53-479Z.md`
- Raw verdict: `WATCH`
- Task documents plus referenced model scripts, tracer output code, canonical scheduler/simulator, slowdown builder/e2e, grouped-gemm setup, and Echo-slowdown interfaces named in the review prompt.

**Identified Issues/Anomalies**

| ID | Disposition | Finding |
|----|-------------|---------|
| A-CR-01 | Resolved by evidence | I16 requires no MoE source-script change: tracer `server_id/local_rank` are not serialized, and sim-engine independently reconstructs topology from `--local-size`. |
| A-CR-02 | Resolved in plan | `--model-size` needed an explicit value-domain contract. Static reconciliation found it is an arbitrary string used only as a legacy path label, so AE keys are accepted directly and Task 4.1 tests them. |
| A-CR-03 | WATCH with Gate B + Task 4.3 gates | Rank0 must prove `optimizer_step` presence. The advisor cited non-canonical `simu_engine1.py`; canonical `simu_main.py` imports `src/core/simu_engine.py`, where direct mapping includes `optimizer_step`, and the PP=1 smoke schedule contains it. Therefore PP=1 absence is a blocker, while a scheduler-generated PP=2 integration test prevents false confidence from the manual schedule. |
| A-CR-04 | WATCH with Task 4.3 gate | `analytical` communication uses fixed `nccl_comm.GPUS_PER_MACHINE=8`; simulation must fail fast if `config.local_size` differs instead of mutating the global. |
| A-CR-05 | Resolved in plan | §7.4 now labels `--bf16`/`--output-dir` as the Task 4.1 target interface rather than current behavior. |

**Remediation/Verification Code Actions Taken**

- No implementation or code action was taken; only active task Markdown was amended.
- Recorded the I16 no-source-change decision and retained separate capture/simulation topology fields.
- Defined direct AE model labels and prohibited a redundant model mapping/implicit architecture lookup.
- Added Gate B exact-op evidence, a canonical scheduler-generated PP=2 reporter integration fixture, and a Phase 4 commit blocker for missing `optimizer_step`.
- Added the analytical topology invariant `config.local_size == LOCAL_SIZE == nccl_comm.GPUS_PER_MACHINE == 8`, with mismatch fail-fast tests and no setter-based auto-alignment.
- Marked §7.4 as a post-Task-4.1 target CLI.

**Independent Review Result**

Raw verdict remains `WATCH`; no `BLOCK` was issued. Every WATCH is now bound to a concrete Gate B or Phase 4 verification. This review does not approve Gate A and does not authorize implementation; explicit user approval remains the only Gate A exit.

## Gate A User Approval

**Target Component/Phase**

Gate A exit and authorization to begin Phase 0 of the reviewed SC'26 AE workflow plan.

**Reviewer Agent Identity**

User/boss YC, explicit approval received 2026-07-16 through the active thread goal.

**Inspected Artifacts**

- `task_memory/task_2026-07-15_sc26_ae_workflow/plan.md`
- Gate A author and independent StepCode Claude review records above
- Final Gate A validation evidence in `progress.md`

**Identified Issues/Anomalies**

- No new plan `BLOCK` was introduced by the approval.
- I20 and I21 remain execution-time `WATCH` items and retain their Gate B/Task 4.3 closure requirements.
- External push, Release creation, asset upload, or default-branch mutation still requires the later exact-target approval defined by the plan; Gate A approval does not pre-authorize those operations.

**Remediation/Verification Code Actions Taken**

- Recorded the raw approval in `requirements.md` as `[Original Request]`.
- Marked Gate A approved and Phase 0 in progress in `plan.md`/`progress.md`.
- No feature implementation, GPU workload, branch, worktree, stage, commit, push, Release, or submodule mutation was performed as part of this approval record.

**Approval Result**

Gate A approved. Phase 0 may proceed; downstream gates remain binding.
