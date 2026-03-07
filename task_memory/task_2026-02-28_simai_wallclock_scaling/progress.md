## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Initialized progress tracker for SimAI wall-clock scaling task |
| 2026-02-28 | Completed implementation scripts and unit/light integration tests |
| 2026-02-28 | Started real long-running measurement; completed first sensitivity point (8 GPUs, PP=1, DP=1) |
| 2026-02-28 | Added background queue shell and validated start/status/log command paths |
| 2026-03-01 | Confirmed long run completion status and identified 8192-GPU topology generation blocker |
| 2026-03-01 | Probed topology templates for 8192 GPUs and confirmed only AlibabaHPN succeeds with current generator |
| 2026-03-01 | Fixed topology filename resolution for AlibabaHPN and started 8192 formal rerun |
| 2026-03-02 | Relaunched 8192 formal run in detached mode and chained auto-plot on completion |
| 2026-03-03 | Completed 8192 backfill, generated plots, and finalized result analysis |
| 2026-03-03 | Started new-plan rerun for >=1024 with PP=16 in isolated output directory |
| 2026-03-03 | Patched new-plan runner log-print typo and added watcher to guarantee post-run plotting |
| 2026-03-04 | Completed isolated new-plan rerun and generated new-plan plots/results |

# Progress Log: SimAI Wall-clock Scaling

## 2026-02-28
- [x] Created `task_memory/task_2026-02-28_simai_wallclock_scaling/`.
- [x] Created initial planning docs (`plan.md`, `notes.md`, `progress.md`, `issues.md`).
- [x] Stashed existing dirty state in `SimAI` and `SimAI/aicb`.
- [x] Created and switched to `SimAI` branch `baseline`.
- [x] Implemented `measure_wallclock.py`.
- [x] Implemented `plot_wallclock.py`.
- [x] Implemented `run_wallclock_queue.sh` for background sequential execution.
- [x] Added unit/integration tests in `test_simai_wallclock_scaling.py`.
- [x] Ran unit test suite and dry-run CLI validation.
- [x] Executed full sensitivity + formal long-running measurements.
- [x] Generated final plots from real measured CSV.
- [x] Completed empirical analysis in `results.md`.

## Implementation Outputs
- Added: `SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py`
- Added: `SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py`
- Added: `SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh`
- Added: `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.csv` (header initialized)
- Added: `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_sensitivity_small_scale.csv` (header initialized)
- Added: `SimAI/tests/unit/test_simai_wallclock_scaling.py`

## Validation Snapshot
- PASS: `python3 -m unittest discover -s SimAI/tests/unit -p 'test_simai_wallclock_scaling.py' -v`
- PASS: `python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase sensitivity --dry-run`
- Expected FAIL-FAST: `python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase formal --dry-run`
  - Error: sensitivity CSV must contain measured rows before formal phase.
- PASS: `bash -n SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh`
- PASS: `SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh start` (detected existing measure process and skipped duplicate queue launch)
- PASS: `SimAI/tests/performance/simai_wallclock_scaling/run_wallclock_queue.sh status`
- FAIL-FAST (expected by current topology limits): `python3 SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py --simai-root SimAI --phase formal`
  - Error: `ValueError: Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)` from `gen_Topo_Template.py` on `-g 8192`.
- PASS: `python3 -m unittest discover -s SimAI/tests/unit -p 'test_simai_wallclock_scaling.py' -v`
  - Added test: `test_ensure_topology_file_accepts_template_with_extra_tokens`
- PASS: `setsid -f bash -lc "python3 ... --phase formal --topology-template AlibabaHPN && python3 .../plot_wallclock.py" > .../formal_8192_and_plot.log 2>&1`
  - Completed 8192 row: `8192,8,8,128,46642.760256`
  - Plot artifacts generated:
    - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_scaling.png`
    - `SimAI/tests/performance/simai_wallclock_scaling/results/wallclock_sensitivity.png`

## Current Step
New-plan rerun and plotting are completed; old records remain unchanged.

## 2026-03-03 (New Plan: PP=16 for >=1024)
- [x] Confirmed new plan table in `parallel_configs.md`.
- [x] Created isolated run directory:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/`
- [x] Seeded new formal CSV with reused baseline rows (`total_gpus < 1024`) from existing completed formal CSV.
- [x] Launched detached background run for new points (1024, 2048, 4096, 8192) without touching old CSV.
- [x] Patched runner typo (`run_summary.json` print) and added detached watcher to rerun-with-resume if first run exits before plot.
- [x] Waited for 4 new points to finish and verified `wallclock_scaling_newplan_pp16_ge1024.csv` has 11 rows.
- [x] Verified new plot outputs:
  - `wallclock_scaling_newplan_pp16_ge1024.png`
  - `wallclock_sensitivity_reused.png`

## New-plan Runtime Evidence
- Detached script:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/run_new_plan.sh`
- Log:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/run.log`
- Meta:
  - `SimAI/tests/performance/simai_wallclock_scaling/results/new_plan_pp16_ge1024_2026-03-03/run_meta.txt`
- Watcher:
  - `bash -lc while ps -p 1221712 ...; run_new_plan.sh >> run.log`
- Completion evidence:
  - `run_summary.json` generated
  - `run.log` contains `[done]` rows for `1024/2048/4096/8192`
