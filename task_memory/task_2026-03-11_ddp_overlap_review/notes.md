## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-11 | Recorded audit notes and discovered gaps |

# Notes

## High-Level Objective Under Review
- Megatron tracing must capture DDP overlap semantics in both `distributed` and `scaling` modes.
- `megatron-sim-engine` must replay those semantics in both `profile` and `simulate` modes and expose overlap output.

## Main Finding
- The tracing task documentation claimed that `megatron/training/training.py` already contained:
  - `_validate_trace_ddp_grad_overlap_runtime_args(...)`
  - `_emit_scaling_dp_allreduce_placeholder(...)`
- In the actual code, those helpers were missing, and the scaling path still used a zero-duration `cmd.no_trace_update(0, 0)` placeholder.

## Consequence
- The review-targeted unit tests failed immediately.
- The missing helper meant the explicit finalize-base tracing contract described in `task_memory/task_2026-03-08_ddp_overlap_tracing/progress.md` was not truly implemented in `training.py`.

## Replay-Side Review Result
- No new replay-path code bug was found in `megatron-sim-engine` during this audit.
- Existing replay regression tests and the canonical acceptance validator remained green after the tracing-side fix.

## Constraint
- Keep the fix surgical.
- Do not change replay semantics unless a real replay bug is reproduced.
