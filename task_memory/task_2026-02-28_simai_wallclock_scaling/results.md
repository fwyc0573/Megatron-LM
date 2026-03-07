## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Initialized results report template for SimAI wall-clock scaling |
| 2026-02-28 | Added implementation-complete status and pending measurement notes |
| 2026-03-01 | Added measured sensitivity and formal (up to 4096) results with policy decision evidence |
| 2026-03-02 | Updated with detached 8192 rerun status and auto-plot pipeline command |
| 2026-03-03 | Finalized 11-point formal results, plots, and conclusions |
| 2026-03-03 | Added isolated rerun setup for new PP=16 plan on >=1024 scales |
| 2026-03-04 | Finalized new-plan PP=16 (>=1024) run results and baseline comparison |

# Results: SimAI Wall-clock Scaling

## Experiment Configuration
- Mode: SimAI-Simulation (NS3)
- TP: 8 (fixed)
- PP range: [1, 12]
- Small-scale sensitivity: 8/16/32 GPUs
- Formal scales: 8 to 8192 GPUs (11 points)
- Model: 22B
- micro_batch: 1
- seq_length: 2048
- GA policy: 1 (`global_batch = dp * micro_batch`)

## Current Data Status
- Sensitivity CSV is complete (6 rows for 8/16/32 all feasible PP).
- Formal CSV is complete (11 rows for 8..8192).
- Output artifacts:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_sensitivity_small_scale.csv`
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.csv`
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_sensitivity.png`
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.png`

## Reproducible Execution Commands

```bash
python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase all
python3 SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py
# Detached continuation for missing 8192 point and auto-plot:
setsid -f bash -lc "python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase formal --topology-template AlibabaHPN && python3 SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py" \
  > SimAI/tests/performance/simai_wallclock_scaling/results/runtime/formal_8192_and_plot.log 2>&1
```

## Sensitivity Analysis (<64 GPUs)
- Measured values:
  - 8 GPUs: `(PP=1,DP=1)=958.434263`
  - 16 GPUs: `(PP=1,DP=2)=1099.413891`, `(PP=2,DP=1)=591.806766`
  - 32 GPUs: `(PP=1,DP=4)=1202.906457`, `(PP=2,DP=2)=657.470807`, `(PP=4,DP=1)=374.402373`
- Policy decision (`threshold=0.20`):
  - `ratio_16 = 1099.413891 / 591.806766 = 1.857724`
  - `ratio_32 = 1202.906457 / 374.402373 = 3.212871`
  - Conclusion: **severe** (`ratio > 1.20`), formal stage uses `max_feasible_pp`.

## Formal Scaling (11 Points)
- Measured rows (11/11):
  - `8,8,1,1,933.412342`
  - `16,8,2,1,590.764510`
  - `32,8,4,1,371.486758`
  - `64,8,8,1,334.270983`
  - `128,8,8,2,621.125627`
  - `256,8,8,4,1160.331067`
  - `512,8,8,8,2353.625036`
  - `1024,8,8,16,4786.794728`
  - `2048,8,8,32,8410.008989`
  - `4096,8,8,64,19932.021011`
  - `8192,8,8,128,46642.760256`
- 8192 completion evidence:
  - runtime log: `SimAI/tests/performance/simai_wallclock_scaling/results/runtime/formal_8192_and_plot.log`
  - key line: `[done] total_gpus=8192 tp=8 pp=8 dp=128 wallclock_seconds=46642.760256`
  - plotting lines:
    - `[plot] scaling png: .../wallclock_scaling.png`
    - `[plot] sensitivity png: .../wallclock_sensitivity.png`

## Key Findings
- For 16/32 scales, PP/DP choice materially changes simulator wall-clock (ratio >> 1.2), validating sensitivity concern.
- Policy result is **severe**, so formal phase used `max_feasible_pp` at each scale (`PP=1,2,4,8,8,...`).
- With selected formal policy, scaling is clearly super-linear in large scales:
  - `8 -> 8192` wall-clock grows from `933.41s` to `46642.76s` (**49.97x**).
  - Doubling factors in high scale are typically `~1.76x to ~2.37x` (`1024->2048->4096->8192`).
- Bottleneck attribution remains consistent with prior assumption:
  - NS3 event processing
  - event-driven scheduling/synchronization overhead
  - excludes external profiling collection in this measurement path

## Limitations
- Single-run per configuration (no repeated trials / confidence interval).
- Topology for 8192 uses approved override `AlibabaHPN`; 8..4096 points were generated under `Spectrum-X`.
- Therefore, `4096 -> 8192` increment includes both scale effect and topology-template effect; interpret this step with caution.

## New Plan Rerun Setup
- Requested update:
  - Reuse completed data for `<1024`.
  - Rerun `1024/2048/4096/8192` with `PP=16`.
- Isolation strategy:
  - Old records are untouched.
  - New outputs are written to:
    - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/`
- New run artifacts (working files):
  - `wallclock_scaling_newplan_pp16_ge1024.csv`
  - `wallclock_sensitivity_small_scale_reused.csv`
  - `run.log`
  - `run_meta.txt`
  - `run_summary.json` (written after measurement completes)
- Topology strategy for rerun:
  - 1024/2048/4096: `Spectrum-X`
  - 8192: `AlibabaHPN` (approved override to bypass Spectrum-X capacity limit)
- Note on DP:
  - Measurement code enforces `DP = total_gpus / (TP * PP)`.
  - For `8192, TP=8, PP=16`, computed `DP=64`.

## New Plan Rerun (Completed)
- New-plan formal CSV (isolated):
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/wallclock_scaling_newplan_pp16_ge1024.csv`
- New-plan plot outputs:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/wallclock_scaling_newplan_pp16_ge1024.png`
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/wallclock_sensitivity_reused.png`
- New-plan measured rows for `>=1024`:
  - `1024,8,16,8,3459.977473`
  - `2048,8,16,16,7196.982990`
  - `4096,8,16,32,14806.385846`
  - `8192,8,16,64,34282.179494`

## Baseline vs New Plan (`>=1024`)
- Baseline (PP=8 at >=1024) came from:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.csv`
- New plan (PP=16 at >=1024) came from isolated new CSV above.
- Wall-clock comparison:

| Total GPUs | Baseline (PP=8) | New Plan (PP=16) | Speedup (Baseline/New) | Change |
|------------|------------------|------------------|-------------------------|--------|
| 1024       | 4786.794728      | 3459.977473      | 1.383x                  | -27.72% |
| 2048       | 8410.008989      | 7196.982990      | 1.169x                  | -14.42% |
| 4096       | 19932.021011     | 14806.385846     | 1.346x                  | -25.72% |
| 8192       | 46642.760256     | 34282.179494     | 1.361x                  | -26.50% |

## Updated Conclusion for New Plan
- Under current test settings, switching `>=1024` to `PP=16` reduces simulator wall-clock across all four measured scales.
- Improvement is largest at 1024/4096/8192 (about 25-28%), and smaller at 2048 (about 14%).
- This confirms PP configuration materially affects SimAI simulation runtime in large-scale settings.
