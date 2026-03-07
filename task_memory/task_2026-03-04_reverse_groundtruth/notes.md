## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Added assumptions and implementation notes for reverse-groundtruth task |
| 2026-03-04 | Added Step-1 solved Ground Truth metrics and scaling factors |
| 2026-03-04 | Added reconstruction/verification scripts and runtime caveat notes |
| 2026-03-04 | Added comm-vs-bubble decomposition and simulate-mode feasibility analysis |
| 2026-03-04 | Added requested E2E -10.486301268860625% reconstruction target and output location |
| 2026-03-04 | Corrected reconstruction method to be schedule-driven (stage schedule to rank trace) |

# Notes

## Inputs
- Baseline metrics file:
  - `task_memory/task_2026-03-04_ws256_dense_simulation_phase/logs/ws256_dense_simulation_metrics_20260304_140456.json`
- Baseline simulation profile database:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/database_profile/`

## Locked Assumptions
- Error definition: `(simulation - ground_truth) / ground_truth`.
- `Total comp` / `Total comm` are derived from the critical rank decomposition where `sum=iteration`.
- This task reconstructs trace/profile only; no modifications to simulator core logic.

## Step-1 Solved Metrics
- Baseline critical rank: `wrank=16`
- Baseline values:
  - `sim_e2e = 3587.99 ms`
  - `sim_comp = 948.47 ms`
  - `sim_comm = 2639.52 ms`
- Given errors:
  - `e2e_error = -10.3%`
  - `comp_error = -1.75%`

### Formula
- Error definition:
  - `err = (sim - gt) / gt`
- Therefore:
  - `gt = sim / (1 + err)`

### Result
- `gt_e2e = 3587.99 / 0.897 = 3999.988851727982 ms`
- `gt_comp = 948.47 / 0.9825 = 965.3638676844784 ms`
- `gt_comm = gt_e2e - gt_comp = 3034.6249840435034 ms`
- `comm_error = (sim_comm - gt_comm) / gt_comm = -13.01989491686856%`

### Scaling Factors (GT / SIM)
- `comp_scale = 1.0178117048346056`
- `comm_scale = 1.1496881948397828`

### Artifact
- Computation record:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution.json`

## Environment
- Project root:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`
- Simulation engine:
  - `megatron-sim-engine/simu_main.py`

## Scripts Added
- Reconstruction:
  - `tests/performance/reconstruct_ws256_groundtruth_profile.py`
- Verification:
  - `tests/performance/verify_reverse_groundtruth_profile.py`

## Runtime Caveat
- In current engine parsing logic, `duration=0` in trace can be normalized to `None` for some ops.
- For `dp_allreduce/ep_allreduce`, reconstruction uses `duration=0.001` to avoid PROFILE runtime type errors.

## Comm/Bubble Decomposition (SIMULATE MODE Baseline)
- Baseline source:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_baseline.json`
- Critical rank (`wrank=16`) decomposition:
  - `comp_execute_ms = 948.47`
  - `comm_execute_ms = 1974.732074`
  - `bubble_ms = 664.787926`
  - `comm_total_ms = 2639.52` (`comm_execute + bubble`)
  - `sum_ms = 3587.99`

## If Bubble Is Fixed (Analytical)
- Target constraints:
  - `E2E error = -10.3%`
  - `Comp error = -1.75%`
- Solved:
  - `gt_e2e_ms = 3999.988851727982`
  - `gt_comp_ms = 965.3638676844784`
  - `required_noncomp_ms = 3034.6249840435034`
  - `required_comm_exec_ms_if_bubble_fixed = 2369.8370580435035`
  - `implied_comm_exec_error_pct_if_bubble_fixed = -16.672242621173936%`
- Artifact:
  - `logs/sim_decomp_target_if_bubble_fixed.json`

## Dynamic Bubble Reality (Timeline Simulation)
- In simulate mode with injected scales, bubble changes with synchronization and shows step-like jumps.
- Therefore, fixing comp error to `-1.75%` and scaling comm execute does not yield a smooth single-point solution for exact `E2E=-10.3%`.

## Requested Reconstruction Target (2026-03-04)
- Target:
  - `overall_error_pct = -10.486301268860625`
  - `comp_error_pct = -1.75`
- Solved critical-rank GT values from reconstruction script:
  - `gt_e2e_ms = 4008.3138680000006`
  - `gt_comp_ms = 965.3638676844784`
  - `gt_comm_ms = 3042.950000315522`
- Derived scale factors (GT / SIM):
  - `comp_scale = 1.0178117048346056`
  - `comm_scale = 1.1528421835468274`
- Output directory:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile`
- Output file count:
  - `256`

## Schedule-Driven Reconstruction Correction
- Problem identified:
  - Previous reconstruction generated simplified 5-op traces (`forward/backward/dp/ep/optimizer`) and did not follow stage scheduling plan operation sequence.
- Correct method:
  - Build each rank trace directly from corresponding stage schedule file:
    - `stage_id = rank // (tp * dp)`
    - rank trace operation sequence = schedule sequence for that stage
  - Fill durations by stage-level comp/comm targets and operation weighting.
  - Keep split-op sub_operations for:
    - `forward_step`, `backward_step`, `loss_func`
- Current output:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile`
  - Fully schedule-aligned for all 256 ranks.
