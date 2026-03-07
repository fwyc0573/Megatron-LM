## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Started reverse-groundtruth execution tracking |
| 2026-03-04 | Completed Step-1 metric solving and recorded derived comm error |
| 2026-03-04 | Completed trace reconstruction and PROFILE-mode verification |
| 2026-03-04 | Added final markdown test report |
| 2026-03-04 | Completed comm/execute-vs-bubble reanalysis in simulate mode |
| 2026-03-04 | Generated requested global_ranks_profile for E2E error -10.486301268860625% under simulation_inputs |
| 2026-03-04 | Fixed reconstruction to be schedule-driven and regenerated schedule-aligned global_ranks_profile |
| 2026-03-04 | Executed variant v1/v2 E2E simulations and recorded comp/comm_execute/bubble breakdown |
| 2026-03-04 | Executed variant v3 simulation and generated consolidated CSV for groundtruth/ours/v1/v2/v3 |
| 2026-03-04 | Updated v2/v3 error targets, reran simulations, and refreshed consolidated CSV |
| 2026-03-05 | Added new-v1/new-v2 simulations with updated target settings and generated new summary table/doc |
| 2026-03-05 | Computed 4-component overlap decomposition (pure_comp, pure_comm, overlap, bubble) for all variants |
| 2026-03-05 | Inferred v3 overlap and n from ours->v3 delta and persisted overlap decomposition report/artifacts |
| 2026-03-05 | Computed overlap decomposition for new-v1/new-v2 assumptions and added dedicated report/artifacts |

# Progress Log

## 2026-03-04

### Completed
- [x] Created task directory and artifact folders.
- [x] Step 1 solved Ground Truth targets from known errors:
  - `GT e2e = 3999.9889 ms`
  - `GT comp = 965.3639 ms`
  - `GT comm = 3034.6250 ms`
  - `Derived comm error = -13.0199%`
  - Saved: `logs/step1_groundtruth_solution.json`
- [x] Added reconstruction script:
  - `tests/performance/reconstruct_ws256_groundtruth_profile.py`
- [x] Generated reconstructed traces:
  - `task_memory/task_2026-03-04_reverse_groundtruth/reconstructed_traces/`
  - `256` files (rank0..rank255)
- [x] Added verification script:
  - `tests/performance/verify_reverse_groundtruth_profile.py`
- [x] Completed PROFILE-mode validation and saved:
  - `logs/profile_verification.json`
  - `logs/profile_verification_run.log`
  - `megatron-sim-engine/log/mg_scheduling/ws256_reverse_groundtruth_profile_20260304_164451.log`
- [x] Achieved errors (critical rank decomposition):
  - `E2E error = -10.300025750064377%`
  - `Comp error = -1.7506241130343783%`
  - `Comm error = -13.019752061213593%`
- [x] Added SIMULATE-mode decomposition script:
  - `tests/performance/simulate_ws256_comp_comm_bubble_scaled.py`
- [x] Collected baseline `comp_execute/comm_execute/bubble` split:
  - `logs/sim_decomp_baseline.json`
- [x] Ran comm-scale scenarios with fixed comp target:
  - `logs/sim_decomp_scaled_*.json`
  - `logs/sim_decomp_scaled_summary.json`
- [x] Baseline critical rank split:
  - `comp=948.47`, `comm_execute=1974.732074`, `bubble=664.787926`, `sum=3587.99`
- [x] Under fixed comp target (`-1.75%`), observed E2E outcomes with dynamic bubble:
  - Best nearby plateau A: `E2E error ≈ -10.4863%`
  - Best nearby plateau B: `E2E error ≈ -10.0250%`
  - Exact `-10.3%` was not hit in tested discrete step interval due bubble jumps.
- [x] Generated `global_ranks_profile` for requested target:
  - `overall_error = -10.486301268860625%`
  - `comp_error = -1.75%`
  - Output dir:
    - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile`
  - Output files: `256`
  - Solution JSON:
    - `logs/step1_groundtruth_solution_e2e_minus10_486301268860625.json`
  - Verification JSON:
    - `logs/profile_verification_e2e_minus10_486301268860625.json`
- [x] Fixed reconstruction logic to follow schedule exactly:
  - Updated script:
    - `tests/performance/reconstruct_ws256_groundtruth_profile.py`
  - New behavior:
    - Parse `schedule/stage*.txt` and generate rank traces line-by-line from stage schedule.
    - Keep stage mapping by `stage = rank // (tp*dp)`.
    - Preserve split-op sub_operations templates for `forward_step/backward_step/loss_func`.
- [x] Schedule consistency verification (all ranks):
  - `256/256` ranks matched their stage schedule exactly in `(op_name, batch_id, mg_state)` sequence.
  - Stage line counts match:
    - stage0 `323==323`, stage1..14 `386==386`, stage15 `387==387`.
- [x] PROFILE re-verification after schedule-driven reconstruction:
  - `E2E error = -10.476387707186115%`
  - `Comp error = -1.8004679767254042%`
  - `Comm error = -13.231054467276577%`
  - Note: slight deviation from previous target is expected after switching from simplified 5-op profile to full schedule + split-op parsing.
- [x] Executed requested variants (simulate mode, critical rank decomposition):
  - v1 target:
    - `comp error = +89.32%`
    - `comm_execute error = -13.257573267957964%`
  - v1 achieved:
    - `e2e_ms = 5148.936152`
    - `comp_execute_ms = 1828.566152`
    - `comm_execute_ms = 1974.732074`
    - `bubble_ms = 1345.637926`
    - `e2e_error = +28.470637820088985%`
  - v2 target:
    - `comp error = -1.8004679767254042%`
    - `comm_execute error = -23.55%`
  - v2 achieved:
    - `e2e_ms = 3343.02`
    - `comp_execute_ms = 948.47`
    - `comm_execute_ms = 1740.420147`
    - `bubble_ms = 654.129853`
    - `e2e_error = -16.588611906074796%`
  - Result summary artifact:
    - `logs/variant_v1_v2_e2e_comp_comm_bubble_summary.json`
- [x] Executed requested variant v3 (simulate mode, critical rank decomposition):
  - v3 target:
    - `comp error = -5.91%`
    - `comm_execute error = -14.75%`
  - v3 achieved:
    - `e2e_ms = 3494.467674`
    - `comp_execute_ms = 908.777674`
    - `comm_execute_ms = 1940.756279`
    - `bubble_ms = 644.933721`
    - `e2e_error = -12.809854760758208%`
  - summary:
    - `logs/variant_v1_v2_v3_e2e_comp_comm_bubble_summary.json`
  - consolidated csv:
    - `logs/variant_groundtruth_ours_v1_v2_v3_e2e_comp_comm_bubble.csv`
- [x] Executed updated-condition variants `new-v1/new-v2` (simulate mode):
  - baseline errors used:
    - `E2E=-10.476387707186115%`
    - `Comp=-1.8004679767254042%`
    - `Comm_execute=-13.231054467276577%`
  - new-v1 targets:
    - `Comp=+89.32%`
    - `Comm_execute=-11.364%`
  - new-v1 results:
    - `e2e_ms = 5186.796152`
    - `comp_execute_ms = 1828.566152`
    - `comm_execute_ms = 2017.223455`
    - `bubble_ms = 1341.006545`
    - `e2e_error = +29.41527923809904%`
  - new-v2 targets:
    - `Comp=-6.34%`
    - `Comm_execute=-23.55%`
  - new-v2 results:
    - `e2e_ms = 3264.884476`
    - `comp_execute_ms = 904.624476`
    - `comm_execute_ms = 1739.88823`
    - `bubble_ms = 620.37177`
    - `e2e_error = -18.53816426181487%`
  - new artifacts:
    - `logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble.csv`
    - `logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble.md`
    - `logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble_summary.json`
- [x] Inferred v3 overlap and `-n%` from existing ours/v3 metrics and persisted docs:
  - Inference rule:
    - Preserve `(pure_comp + pure_comm_execute)` total from `ours` to `v3`.
    - `overlap_v3 = overlap_ours - ((comp_ours + comm_ours) - (comp_v3 + comm_v3))`
  - Key result:
    - `overlap_v3 = 151.9142407695 ms`
    - `n = 31.949759239405626%` (in `-n%` form relative to `groundtruth overlap`)
  - New artifacts:
    - `logs/variant_groundtruth_ours_v1_v2_v3_pure_comp_pure_comm_bubble_overlap.csv`
    - `logs/variant_groundtruth_ours_v1_v2_v3_overlap_inference_v3_from_ours.json`
    - `test_report_2026-03-05_overlap_decomposition_v3_inferred.md`
- [x] Computed overlap decomposition for `new-v1/new-v2` under updated overlap-error assumptions:
  - Assumptions:
    - `groundtruth overlap = 5.57% * groundtruth_e2e`
    - `ours overlap error vs gt = +1.05%`
    - `new-v1 overlap error vs gt = +21.05%`
    - `new-v2 overlap error vs gt = -6.75%`
  - Key results:
    - `new-v1: pure_comp=1693.45113521525, pure_comm_execute=1882.10843821525, bubble=1341.006545, overlap=270.2300335695`
    - `new-v2: pure_comp=800.53959111625, pure_comm_execute=1635.80334511625, bubble=620.37177, overlap=208.1697697675`
  - New artifacts:
    - `logs/variant_groundtruth_ours_new_v1_new_v2_pure_comp_pure_comm_bubble_overlap.csv`
    - `logs/variant_groundtruth_ours_new_v1_new_v2_pure_comp_pure_comm_bubble_overlap_summary.json`
    - `test_report_2026-03-05_overlap_decomposition_new_v1_new_v2.md`
    - `logs/overlap_decomp_new_v1_new_v2_verification.log`

### In Progress
- [x] Step 2: Trace reconstruction and scaling.
- [x] Step 3: PROFILE-mode verification and report.

- [x] Computed 4-component overlap decomposition for all variants:
  - Script: `tests/performance/compute_overlap_decomposition.py`
  - v3 overlap error n = 31.949759% (solved via sum constraint)
  - CSV: `logs/variant_overlap_decomposition.csv`
  - JSON: `logs/variant_overlap_decomposition.json`
  - Report: `test_report_2026-03-05_overlap_decomposition.md`

### Pending
- [ ] None.
