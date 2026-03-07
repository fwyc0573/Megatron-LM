## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Logged blocking/observed issues during WS256 simulation execution |

# Issues and Resolutions

## Issue 1: Excessive runtime with verbose simulation logs
- Symptom: default execution produced massive per-operation debug output and long wall-clock.
- Impact: practical completion risk for WS256 run in interactive session.
- Resolution: rerun using `collective-sim` with backend option:
  - `{"collective-sim": {"placement_mode": "group_size"}}`
- Result: simulation completed and produced usable timing outputs.

## Issue 2: No distributed ground-truth folder for direct measured comparison
- Symptom: workspace does not contain matching `global_ranks_profile` for this WS256 dense case.
- Impact: cannot run distributed-vs-sim measured comparison in this task.
- Resolution: explicitly mark measured comparison as skipped:
  - `N/A (skipped by task decision)`.

## Issue 3: tensor metadata warning on some communication ops
- Symptom: warning lines like `tensor_shape=None or tensor_dtype=None` for some comm ops.
- Impact: selected comm ops fallback to default payload handling in backend path.
- Resolution: no code mutation in this task; keep evidence in logs and report as residual risk.
