## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-11 | Created review plan for cross-repo DDP overlap tracing/replay audit |

# Plan

## Goal
- Audit the completed `task_2026-03-08_ddp_overlap_tracing` and `megatron-sim-engine/task_2026-03-08_ddp_overlap_replay` tasks.
- Fix correctness gaps or missing implementation paths.
- Re-run targeted validation so the overlap path remains correct from Megatron tracing to sim-engine replay output.

## Scope
- `megatron/profiler/cmd.py`
- `megatron/core/distributed/*`
- `megatron/training/training.py`
- `tests/unit_tests/*ddp_overlap*`
- `tests/integration/test_ddp_overlap_trace_smoke.sh`
- `megatron-sim-engine/src/core/simu_engine.py`
- `megatron-sim-engine/simu_main.py`
- `megatron-sim-engine/tests/*ddp_overlap*`

## Acceptance Criteria
- Root repo targeted unit tests pass.
- Root repo distributed + scaling smoke trace test passes.
- Fresh scaling traces include `op_semantics=metadata_placeholder` and explicit `finalize_base_duration_ms` on top-level `dp_allreduce`.
- Sim-engine overlap CLI e2e remains green.
- Sim-engine canonical acceptance validator remains green.

## Steps
1. Review both task directories and compare claims against code/tests.
2. Reproduce failures with the most targeted regression suite first.
3. Apply the minimal fix for any missing or incorrect implementation.
4. Re-run unit, integration, and replay acceptance validation.
5. Record outcomes, residual risks, and follow-up recommendations.
