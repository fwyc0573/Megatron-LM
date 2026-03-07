## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial reverse-groundtruth reconstruction plan |

# Plan: Reverse Ground Truth Timeline and Profile Reconstruction

## Scope
- Compute target Ground Truth metrics from known errors (`E2E=-10.3%`, `Comp=-1.75%`).
- Reconstruct `global_ranks_profile`-style traces under `task_memory/task_2026-03-04_reverse_groundtruth/reconstructed_traces/`.
- Validate by running `megatron-sim-engine` in `PROFILE` mode.
- Produce reproducible test report with commands, logs, and pass/fail checks.

## Execution Steps
1. Extract simulation baseline (`iteration/comp/comm`) from existing WS256 simulation artifacts.
2. Solve Ground Truth values and required comm error mathematically.
3. Generate reconstructed trace files with per-op duration scaling:
   - Comp scale fixed by `-1.75%`.
   - Comm scale solved from `-10.3%` E2E target.
4. Run `simu_main.py --mode profile` with reconstructed traces.
5. Parse verification metrics and compare against required errors.

## Acceptance Criteria
- `reconstructed_traces/` generated with valid trace format.
- PROFILE run exits with code `0`.
- Verified errors:
  - `Total comp error = -1.75%`
  - `Total comm error = solved value (<0)`
  - `E2E error = -10.3%`
