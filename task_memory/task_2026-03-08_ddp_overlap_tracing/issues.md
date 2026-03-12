## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-08 | Created issue tracker for DDP grad overlap tracing task |
| 2026-03-08 | Updated issue tracker after implementation and validation |
| 2026-03-09 | Closed explicit scaling finalize-base trace gap and documented remaining historical-trace caveat |
| 2026-03-11 | Recorded follow-up audit repair of the missing `training.py` helpers |

# Issues

## Open
- Historical scaling traces collected before this change still rely on replay-side `neighbor_gap` inference because they do not carry explicit `finalize_base_duration_ms`.


## Resolved
- The checked-out working tree no longer misses `_emit_scaling_dp_allreduce_placeholder(...)` or `_validate_trace_ddp_grad_overlap_runtime_args(...)`; both helpers were restored in `megatron/training/training.py` and revalidated by unit + smoke tests on 2026-03-11.
- Kept legacy top-level trace parsers working by appending new top-level fields while preserving trailing `sub_operations=`.
- Added a clean scaling-mode metadata-only path that records intended DDP launch without executing a real collective.
- Distinguished `param_hook` launch from delayed `grad_sync_func` launch in the bucket lifecycle record.
- Fixed scaling warmup leakage so `ddp_grad_comm` records only appear for traced iterations.
- Fixed sub-op tracing regression caused by assuming every active CMD object implements `phase_range()`.
- Scaling traces now emit explicit `finalize_base_duration_ms` for top-level `dp_allreduce(metadata_placeholder)` in fresh overlap runs, removing replay-side dependence on neighbor-gap inference for the new captures.
- `_emit_scaling_dp_allreduce_placeholder(...)` now fail-fast reports missing `cmd.stage_operations_trace_dict` instead of surfacing a less actionable `AttributeError`.
- `_emit_scaling_dp_allreduce_placeholder(...)` now fail-fast rejects semantic misuse (`name_cmd != dp_allreduce` or `op_semantics != metadata_placeholder`) instead of silently emitting malformed placeholders.
