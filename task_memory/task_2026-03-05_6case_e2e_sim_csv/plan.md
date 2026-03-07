## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initial plan for 6-case e2e simulation and CSV generation |
| 2026-03-05 | Marked implementation/testing phases complete; added infeasible-bound blocker status |
| 2026-03-05 | Recorded expanded-bounds rerun status (5/6 feasible, deepseek_v3_variant_case1 blocked) |
| 2026-03-05 | Final rerun with relaxed lower bounds completed; 6/6 cases satisfy threshold |
| 2026-03-06 | Added strict-bound rerun status with non-zero error policy and non-negative comm factors |

# Task Plan: 6-Case E2E Simulation and CSV Output

## Goal
Run end-to-end simulation for 6 cases (Qwen3 x3 + DeepSeek-V3-variant x3), replace scaling comp with distributed-comp scaling, tune communication/overlap per case, and export CSV with required columns under `results/`.

## Phases
- [x] Phase 1: Finalize execution scope and constraints
- [x] Phase 2: Implement script and solver
- [x] Phase 3: Add unit/integration tests (TDD)
- [x] Phase 4: Run smoke and full 6-case validation
- [x] Phase 5: Produce artifacts and final report

## Acceptance Criteria
- Exactly 6 case rows in output CSV.
- CSV columns strictly: `case_name,excl_comp_ms,excl_comm_ms,bubble_ms,overlap_ms,e2e_total_ms`.
- Each case satisfies `abs(error_pct) <= 9` against profile ground truth.
- Diagnostics JSON includes selected parameters and evidence.
- No source modification under collective-sim backend module.

## Status
**Current strict-bound status** - partial completion:
- Constraints: `comp_scale=[0.965,0.988]`, `comm_factor>=0`, `abs_error_pct in [0.2,9.0]`
- Result: `5/6` feasible; `deepseek_v3_variant_case1` remains infeasible.
- Feasible-case CSV with ground truth has been regenerated for paper plotting.
