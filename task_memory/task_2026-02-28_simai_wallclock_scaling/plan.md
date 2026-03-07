## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Initialized implementation plan for SimAI simulation wall-clock scaling task |
| 2026-02-28 | Implemented measurement/plot scripts and unit tests on `SimAI` branch `baseline` |

# Task Plan: SimAI Simulation Wall-clock Scaling

## Goal
Measure and analyze SimAI-Simulation wall-clock time scalability across 8 to 8192 GPUs under fixed TP and controlled PP/DP settings.

## Locked Scope
- Mode: SimAI-Simulation only (NS3), no Analytical control group.
- Scales: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 GPUs.
- TP fixed at 8.
- PP constrained to [1, 12], DP derived by `DP = total_gpus / (TP * PP)`.
- Sensitivity stage on `<64` GPUs only (8/16/32), all feasible PP values.
- Formal stage on all 11 scales, one (PP,DP) per scale selected from sensitivity rule.
- Model/workload constants: model 22B, micro_batch=1, seq_length=2048, GA=1.
- Per configuration run count: 1.
- No timeout policy.

## Phases
- [x] Phase 0: Branch/environment preparation in `SimAI/baseline`
- [x] Phase 1: Task-memory docs initialization
- [x] Phase 2: Implement measurement runner (`measure_wallclock.py`)
- [x] Phase 3: Implement plotting utility (`plot_wallclock.py`)
- [x] Phase 4: Add and run unit/integration tests
- [ ] Phase 5: Produce full measurement results and final analysis artifacts

## Acceptance Criteria
1. `SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py` exists with required CLI and fail-fast behavior. ✅
2. `SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py` generates two PNG plots from CSV inputs. ✅ (validated with synthetic unit/integration input)
3. `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.csv` exists with strict columns:
   - `total_gpus,tp,pp,dp,wallclock_seconds` ✅ (header initialized; full data pending long-run execution)
4. Sensitivity output CSV exists for 8/16/32 scales. ✅ (header initialized; full data pending long-run execution)
5. Unit and lightweight integration tests pass. ✅
6. `task_memory/task_2026-02-28_simai_wallclock_scaling/results.md` and test report are present. ✅

## Status
**In Progress** - code implementation and test scaffolding are complete; full long-running measurements are pending execution.
