## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initialized issue tracking for reverse-groundtruth task |
| 2026-03-04 | Recorded and resolved PROFILE parsing/runtime issues during validation |
| 2026-03-04 | Recorded simulate-mode non-smooth bubble response under comm scaling |
| 2026-03-04 | Recorded and resolved schedule mismatch in reconstructed global_ranks_profile |

# Issues

## Open
- Under strict `comm_execute`/`bubble` split and fixed `comp error=-1.75%`, E2E tuning shows step-like behavior:
  - Nearby plateaus observed around `E2E error=-10.4863%` and `-10.0250%`.
  - Exact `-10.3%` was not hit in tested comm-scale points.
  - Suspected cause: duration discretization/rounding (`2-digit`-level in parsing/execution path) and synchronization-driven bubble jumps.

## Resolved
- Reconstructed `global_ranks_profile` did not follow stage `schedule` sequence.
  - Root cause:
    - Previous reconstruction method generated simplified 5-op traces per rank instead of schedule-driven per-stage op sequence.
  - Resolution:
    - Rewrote `tests/performance/reconstruct_ws256_groundtruth_profile.py` to parse and replay `schedule/stage*.txt` line-by-line for each rank.
    - Verified `(op_name, batch_id, mg_state)` sequence matches for `256/256` ranks.
    - Regenerated:
      - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile`

- `PROFILE` run failed with `TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'`.
  - Root cause:
    - In `simu_engine.py`, parsed `duration=0` becomes `None` because of `duration = round(float(duration),2) if duration else None`.
    - Reconstructed traces initially used `duration=0` for `dp_allreduce/ep_allreduce`.
  - Resolution:
    - Updated reconstruction script to emit `duration=0.001` for `dp_allreduce/ep_allreduce`.
    - Regenerated traces and reran PROFILE verification successfully.
