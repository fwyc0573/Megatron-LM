## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Initialized test report template for GPT-175B scaling wall-clock task |
| 2026-03-02 | Filled commands, results, and evidence after implementation and full run |
| 2026-03-04 | Confirmed and documented that all timing values are in **seconds (s)** |
| 2026-03-04 | Changed extrapolation multiplier from ×15 to ×5; updated script, CSV, and report |

## Test Report: GPT-175B Scaling Wall-clock Scan

**Date**: 2026-03-02

### 1. Test Script Information
- Script(s):
  - `examples/gpt175b_scaling_wallclock_scan.sh`
  - `tests/unit/test_gpt175b_scaling_wallclock_config.sh`
  - `tests/integration/test_gpt175b_scaling_wallclock_dryrun.sh`
- Commands:
  ```bash
  # Static check
  bash -n examples/gpt175b_scaling_wallclock_scan.sh
  bash -n tests/unit/test_gpt175b_scaling_wallclock_config.sh
  bash -n tests/integration/test_gpt175b_scaling_wallclock_dryrun.sh

  # Unit test
  bash tests/unit/test_gpt175b_scaling_wallclock_config.sh

  # Integration dry-run
  bash tests/integration/test_gpt175b_scaling_wallclock_dryrun.sh

  # Real run
  SCALE_GPU=1 bash examples/gpt175b_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-02_gpt175b_scaling_wallclock/logs/run_gpt175b_wallclock_20260302_115651.log
  ```
- Environment:
  - Conda env: `myenv_yc`
  - Python: `3.9.18`
  - Python path: `/opt/anaconda/envs/myenv_yc/bin/python`

### 2. Validation Criteria
- Unit checks for config arithmetic/rank mapping/global-batch formula pass.
- Dry-run generates CSV with exact header/4 rows/counts/derived metric relation.
- Real measurement run attempts all required ranks per config in fail-fast mode.

### 3. Test Results and Evidence
| Suite | Result | Evidence |
|------|--------|----------|
| Static syntax check | PASS | `syntax_ok` |
| Unit test | PASS | `[PASS] GPT-175B scaling wall-clock config checks passed.` |
| Integration dry-run | PASS | `[PASS] Dry-run integration checks passed.` |
| Real measurement run | PASS | 4 configs completed with per-config timing lines in run log |

#### Key real-run evidence
- Log file: `task_memory/task_2026-03-02_gpt175b_scaling_wallclock/logs/run_gpt175b_wallclock_20260302_115651.log`
- Extracted completion lines:
  - `Config ws256_pp16_tp8_dp2 done: single_iter_wallclock_seconds=269.702444`
  - `Config ws1024_pp16_tp8_dp8 done: single_iter_wallclock_seconds=270.106722`
  - `Config ws4096_pp32_tp8_dp16 done: single_iter_wallclock_seconds=537.983920`
  - `Config ws8192_pp32_tp8_dp32 done: single_iter_wallclock_seconds=699.711289`
  - `Completed all configurations. CSV saved to .../docs/data/gpt175b_scaling_wallclock_timing.csv`

#### Timing unit clarification
- **所有时间值的单位为秒 (seconds, s)**。
- 脚本通过 `date +%s%N`（纳秒时间戳）计差后除以 `1000000000` 得到秒值（保留 6 位小数）。
- CSV 列名 `single_iter_wallclock_seconds` 和 `estimated_5_iters_seconds` 也明确标记了单位。

#### CSV output
- File: `docs/data/gpt175b_scaling_wallclock_timing.csv`
- Final rows:
  - `256,16,8,2,16,269.702444,1348.512220`
  - `1024,16,8,8,16,270.106722,1350.533610`
  - `4096,32,8,16,32,537.983920,2689.919600`
  - `8192,32,8,32,32,699.711289,3498.556445`

#### Failure and resolution
- During the first end-to-end run, CSV content was observed with only the final row.
- Resolution:
  1. Updated `examples/gpt175b_scaling_wallclock_scan.sh` to accumulate rows in-memory and write CSV once at the end.
  2. Reconstructed the expected four rows from the successful run log and rewrote `docs/data/gpt175b_scaling_wallclock_timing.csv`.
  3. Re-ran unit + integration dry-run tests to validate script behavior after the fix.
