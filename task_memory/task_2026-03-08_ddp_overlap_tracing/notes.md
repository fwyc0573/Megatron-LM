## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-08 | Added implementation notes and constraints for DDP grad overlap tracing |
| 2026-03-08 | Added validation notes for distributed and scaling smoke coverage |

# Notes: DDP Grad Overlap Tracing MVP

## Constraints
- Only DDP grad overlap is in scope.
- `megatron-sim-engine/` remains unchanged.
- Scaling mode must not execute real DP collectives when overlap tracing is enabled.
- Fail fast on unsupported flag combinations.

## Key Code Paths
- `megatron/profiler/cmd.py`
- `megatron/core/distributed/param_and_grad_buffer.py`
- `megatron/core/distributed/distributed_data_parallel.py`
- `megatron/core/distributed/finalize_model_grads.py`
- `megatron/training/training.py`
- `megatron/training/arguments.py`

## Trace Design Summary
- Add `cmd_uid` and `op_semantics` to top-level CMD lines.
- Add standalone `ddp_grad_comm(...)` event lines in the same trace file.
- Record trigger, launch, completion, and wait timestamps when available.
- Record metadata-only intended schedule in scaling mode.

## Validation Notes
- Unit tests cover schema serialization, bucket launch/completion/wait lifecycle, scaling metadata-only behavior, warmup gating, and runtime/CLI fail-fast checks.
- Integration smoke covers both distributed and scaling modes with real `torchrun` execution and trace-file assertions.
