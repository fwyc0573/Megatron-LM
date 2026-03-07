## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial test report for reverse ground-truth reconstruction and PROFILE validation |
| 2026-03-04 | Added simulate-mode comp/comm_execute/bubble decomposition and feasibility analysis |
| 2026-03-04 | Added reconstruction run for requested E2E error -10.486301268860625% and verification evidence |
| 2026-03-04 | Fixed schedule mismatch by switching to schedule-driven reconstruction and added consistency tests |
| 2026-03-04 | Added WS256 variant v1/v2 E2E simulation runs and metric summary |
| 2026-03-04 | Added WS256 variant v3 run and consolidated CSV across groundtruth/ours/v1/v2/v3 |
| 2026-03-04 | Updated v2/v3 target errors and refreshed simulation outputs and consolidated CSV |
| 2026-03-05 | Added new-v1/new-v2 rerun with updated conditions and generated dedicated summary CSV/doc |

# Test Report: Reverse Ground Truth Reconstruction (WS256 Dense H800)

**Date**: 2026-03-04

## 1. Test Script Information

### Scripts
- Reconstruction script:
  - `tests/performance/reconstruct_ws256_groundtruth_profile.py`
- Verification script:
  - `tests/performance/verify_reverse_groundtruth_profile.py`

### Reproducible Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM

python tests/performance/reconstruct_ws256_groundtruth_profile.py --allow-overwrite

python tests/performance/verify_reverse_groundtruth_profile.py
```

### Environment
- Python: `3.9.18`
- `SIMULATOR_HARDWARE_TYPE=H800_SXM` (set in verification script runtime)
- `PYTHONPATH=/research/d1/gds/ytyang/yichengfeng/trace_test_proj::/research/d1/gds/ytyang/yichengfeng/Megatron-LM/`

## 2. Validation Criteria

### Required Targets
1. `Total comp error = -1.75%`
2. `Total comm error = Step-1 solved value (<0)`
3. `E2E total error = -10.3%`

### Error Definition
- `error = (simulation - ground_truth) / ground_truth`

### Step-1 Solved Ground Truth Targets (Critical Rank)
- Baseline critical rank: `wrank=16`
- `sim_e2e = 3587.99 ms`
- `sim_comp = 948.47 ms`
- `sim_comm = 2639.52 ms`
- `gt_e2e = 3999.988851727982 ms`
- `gt_comp = 965.3638676844784 ms`
- `gt_comm = 3034.6249840435034 ms`
- Derived `target_comm_error = -13.01989491686856%`

## 3. Test Results and Evidence

### Result Summary
| Check | Target | Achieved | Status |
|------|--------|----------|--------|
| E2E error | `-10.3%` | `-10.300025750064377%` | PASS |
| Comp error | `-1.75%` | `-1.7506241130343783%` | PASS |
| Comm error | `-13.01989491686856%` | `-13.019752061213593%` | PASS |

### Critical Rank Evidence
- Baseline critical:
  - `wrank=16, stage=1, sim_e2e=3587.99 ms, sim_comp=948.47 ms, sim_comm=2639.52 ms`
- Reconstructed PROFILE critical:
  - `wrank=16, stage=1, gt_e2e=3999.99 ms, gt_comp=965.37 ms, gt_comm=3034.62 ms`

### Per-rank Breakdown (PROFILE reconstructed)
Top entries:
- `wrank=16 stage=1 comp=965.37 comm=3034.62 sum=3999.99`
- `wrank=32 stage=2 comp=968.06 comm=3003.86 sum=3971.92`
- `wrank=48 stage=3 comp=972.60 comm=2969.00 sum=3941.60`

### Per-stage Breakdown (PROFILE reconstructed)
- `stage0: sum=3898.44 ms`
- `stage1: sum=3999.99 ms` (critical)
- `stage2: sum=3971.92 ms`
- `...`
- `stage15: sum=3579.75 ms`

### Key Artifacts
- Step-1 solved metrics:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution.json`
- Reconstructed traces directory:
  - `task_memory/task_2026-03-04_reverse_groundtruth/reconstructed_traces/`
- Verification output JSON:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/profile_verification.json`
- Verification stdout/stderr log:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/profile_verification_run.log`
- Direct PROFILE run log via CLI:
  - `megatron-sim-engine/log/mg_scheduling/ws256_reverse_groundtruth_profile_20260304_164451.log`

## 4. PASS/FAIL Conclusion

- Overall status: **PASS**
- Reconstructed profile traces satisfy the required target errors within numerical rounding tolerance.

## 5. Reanalysis: `comm_execute` Excluding Bubble

### Additional Script
- `tests/performance/simulate_ws256_comp_comm_bubble_scaled.py`

### Additional Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM

# Baseline decomposition
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 1.0 \
  --comm-scale 1.0 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_baseline.json

# Fixed comp target (-1.75%), scan comm execute scale
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 1.0178117048346056 \
  --comm-scale 1.1718 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_scaled_final4.json
```

### Baseline Critical Rank Split (`wrank=16`)
- `comp_execute_ms = 948.47`
- `comm_execute_ms = 1974.732074`
- `bubble_ms = 664.787926`
- `comm_total_ms = 2639.52`
- `sum_ms = 3587.99`

### Analytical Result (If Bubble Fixed)
- Required `comm_execute` error to satisfy both targets:
  - `-16.672242621173936%`
- Artifact:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_target_if_bubble_fixed.json`

### Timeline Simulation Result (Dynamic Bubble)
- With fixed comp target (`-1.75%`) and comm scale tuning, bubble changes non-linearly.
- Tested results (critical rank):
  - Plateau A:
    - `sum_ms = 4008.313868`
    - `E2E error = -10.486301268860625%`
  - Plateau B:
    - `sum_ms = 3987.763868`
    - `E2E error = -10.025013547266541%`
- Exact `E2E error = -10.3%` was not hit in tested comm-scale points while keeping comp error fixed at `-1.75%`.

### Evidence Files
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_baseline.json`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_scaled_summary.json`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_scaled_try1.json`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_scaled_try2.json`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_scaled_try3.json`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_scaled_final*.json`

## 6. Requested Build: `E2E error = -10.486301268860625%`

### Command
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM

python tests/performance/reconstruct_ws256_groundtruth_profile.py \
  --overall-error-pct -10.486301268860625 \
  --comp-error-pct -1.75 \
  --output-trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile \
  --solution-json task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution_e2e_minus10_486301268860625.json \
  --allow-overwrite
```

### File Output Check
- Output directory:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile`
- File count: `256` (rank0..rank255)

### Verification Command
```bash
python tests/performance/verify_reverse_groundtruth_profile.py \
  --trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile \
  --solution-json task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution_e2e_minus10_486301268860625.json \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/profile_verification_e2e_minus10_486301268860625.json
```

### Verified Result (Critical Rank)
- Target:
  - `E2E error = -10.486301268860625%`
  - `Comp error = -1.75%`
- Achieved:
  - `E2E error = -10.486214888569002%`
  - `Comp error = -1.7506241130343783%`
  - `Comm error = -13.257573267957964%`

### Additional Artifacts
- Reconstruction log:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/reconstruct_run_e2e_minus10_486301268860625.log`
- Verification run log:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/profile_verification_e2e_minus10_486301268860625_run.log`

## 7. Schedule Alignment Fix and Validation

### Root Cause
- The previous reconstruction generated reduced traces and did not fully follow:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/schedule/stage*.txt`

### Fix
- Updated:
  - `tests/performance/reconstruct_ws256_groundtruth_profile.py`
- New reconstruction logic:
  - Read stage schedules and generate rank traces line-by-line from schedule.
  - Apply stage mapping:
    - `stage = rank // (tp * dp)` with `tp=8`, `dp=2`.
  - Preserve split-op sub_operations templates for:
    - `forward_step`, `backward_step`, `loss_func`

### Reproducible Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM

python tests/performance/reconstruct_ws256_groundtruth_profile.py \
  --overall-error-pct -10.486301268860625 \
  --comp-error-pct -1.75 \
  --schedule-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/schedule \
  --output-trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile \
  --solution-json task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution_e2e_minus10_486301268860625.json \
  --allow-overwrite

python tests/performance/verify_reverse_groundtruth_profile.py \
  --trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/global_ranks_profile \
  --solution-json task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution_e2e_minus10_486301268860625.json \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/profile_verification_e2e_minus10_486301268860625.json
```

### Consistency Validation Criteria
- For every rank `r`:
  - Compare trace sequence against stage schedule sequence using:
    - `(op_name, batch_id, mg_state)`
  - Expected:
    - exact match between rank trace and corresponding stage schedule.

### Consistency Validation Result
- PASS:
  - `256/256` ranks matched their stage schedule sequence exactly.
  - No mismatch ranks.
- Evidence:
  - Stage line count parity recorded in:
    - `task_memory/task_2026-03-04_reverse_groundtruth/logs/step1_groundtruth_solution_e2e_minus10_486301268860625.json`
    - `stage_schedule_line_counts == output_stage_line_counts`
  - Full rank-level consistency check report:
    - `task_memory/task_2026-03-04_reverse_groundtruth/logs/schedule_consistency_check_e2e_minus10_486301268860625.json`

### PROFILE Re-Verification Result (after schedule-driven fix)
- Achieved (critical rank):
  - `E2E error = -10.476387707186115%`
  - `Comp error = -1.8004679767254042%`
  - `Comm error = -13.231054467276577%`
- Note:
  - Compared with simplified reconstruction, full schedule + split-op parsing introduces small numerical drift, but trace structure is now schedule-correct.

## 8. Variant v1/v2 E2E Simulations

### Objective
- Run two what-if variants by scaling comp/comm operations and report:
  - `e2e`, `comp_execute`, `comm_execute`, `bubble`
  - `e2e error` relative to ground truth

### Baseline and Targets (User-Specified)
- Baseline ours errors (relative to GT):
  - `E2E error = -10.476387707186115%`
  - `Comp error = -1.8004679767254042%`
  - `Comm error (comm_execute baseline for variant calc) = -13.257573267957964%`
- v1 target:
  - `comp error = +89.32%`
  - `comm_execute error = -13.257573267957964%` (unchanged)
- v2 target:
  - `comp error = -1.8004679767254042%` (unchanged)
  - `comm_execute error = -23.55%`

### Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM

# Baseline decomposition
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 1.0 \
  --comm-scale 1.0 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_baseline_for_variants.json

# v1 scales
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 1.9279114278785832 \
  --comm-scale 1.0 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v1.json

# v2 scales
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 1.0 \
  --comm-scale 0.8813449528702189 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v2.json
```

### Results (Critical Rank)
| Variant | wrank | stage | e2e_ms | comp_execute_ms | comm_execute_ms | bubble_ms | e2e_error_pct |
|--------|------:|------:|-------:|----------------:|----------------:|----------:|--------------:|
| v1 | 16 | 1 | 5148.936152 | 1828.566152 | 1974.732074 | 1345.637926 | +28.470637820088985 |
| v2 | 16 | 1 | 3343.02 | 948.47 | 1740.420147 | 654.129853 | -16.588611906074796 |

### Artifacts
- Baseline:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_baseline_for_variants.json`
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_baseline_for_variants_run.log`
- v1:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v1.json`
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v1_run.log`
- v2:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v2.json`
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v2_run.log`
- Consolidated summary:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_v1_v2_e2e_comp_comm_bubble_summary.json`

## 9. Variant v3 and Consolidated CSV

### v3 Target
- `Comp error = -5.91%`
- `Comm error (comm_execute) = -14.75%`

### v3 Result (Critical Rank)
- `e2e_ms = 3494.467674`
- `comp_execute_ms = 908.777674`
- `comm_execute_ms = 1940.756279`
- `bubble_ms = 644.933721`
- `e2e_error_pct = -12.809854760758208`

### Consolidated CSV
- File:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_v1_v2_v3_e2e_comp_comm_bubble.csv`
- Included rows:
  - `groundtruth`, `ours`, `v1`, `v2`, `v3`
- Included columns:
  - `variant`, `e2e_ms`, `comp_execute_ms`, `comm_execute_ms`, `bubble_ms`, `e2e_error_pct`

### v3 Artifacts
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v3.json`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_v3_run.log`
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_v1_v2_v3_e2e_comp_comm_bubble_summary.json`

## 10. Updated-Condition Rerun: `new-v1/new-v2`

### Methodology Review (same simulate approach as prior v1/v2)
- Baseline decomposition source:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_v1_v2_v3_e2e_comp_comm_bubble.csv`
- Solve scale factors from target errors:
  - `comp_scale = (1 + target_comp_error) / (1 + ours_comp_error)`
  - `comm_scale = (1 + target_comm_error) / (1 + ours_comm_error)`
- Run:
  - `tests/performance/simulate_ws256_comp_comm_bubble_scaled.py`
- Read critical rank decomposition:
  - `e2e_ms`, `comp_execute_ms`, `comm_execute_ms`, `bubble_ms`

### Baseline Errors Used (User-Specified for this rerun)
- `E2E error = -10.476387707186115%`
- `Comp error = -1.8004679767254042%`
- `Comm error (comm execute) = -13.231054467276577%`

### New Targets
- `new-v1`:
  - `Comp error = +89.32%`
  - `Comm error (comm execute) = -11.364%`
- `new-v2`:
  - `Comp error = -6.34%`
  - `Comm error (comm execute) = -23.55%`

### Scales and Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM

# new-v1 scales
# comp_scale=1.9279114278785832, comm_scale=1.0215175424319574
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 1.9279114278785832 \
  --comm-scale 1.0215175424319574 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_new_v1.json

# new-v2 scales
# comp_scale=0.9537723660210655, comm_scale=0.8810755913954051
python tests/performance/simulate_ws256_comp_comm_bubble_scaled.py \
  --comp-scale 0.9537723660210655 \
  --comm-scale 0.8810755913954051 \
  --output-json task_memory/task_2026-03-04_reverse_groundtruth/logs/sim_decomp_variant_new_v2.json
```

### Results (Critical Rank)
| Variant | wrank | stage | e2e_ms | comp_execute_ms | comm_execute_ms | bubble_ms | e2e_error_pct |
|--------|------:|------:|-------:|----------------:|----------------:|----------:|--------------:|
| new-v1 | 16 | 1 | 5186.796152 | 1828.566152 | 2017.223455 | 1341.006545 | +29.41527923809904 |
| new-v2 | 16 | 1 | 3264.884476 | 904.624476 | 1739.888230 | 620.371770 | -18.53816426181487 |

### New Summary Artifacts
- CSV (same schema as historical variant table):
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble.csv`
- JSON:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble_summary.json`
- Markdown doc:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble.md`
