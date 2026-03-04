## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initialized test report template for Qwen3-A3B MoE scaling wall-clock task |
| 2026-03-04 | Filled static, unit, and integration dry-run results with evidence |

## Test Report: Qwen3-A3B MoE Scaling Wall-clock Scan

**Date**: 2026-03-04

### 1. Test Script Information
- Script(s):
  - `examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`
  - `tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh`
  - `tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh`
- Commands:
  ```bash
  # Static check
  bash -n examples/qwen3_a3b_moe_scaling_wallclock_scan.sh
  bash -n tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh
  bash -n tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh

  # Unit test
  bash tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh

  # Integration dry-run
  bash tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh
  ```
- Environment:
  - Conda env: N/A (shell-only validation)
  - Python: N/A (shell-only validation)

### 2. Validation Criteria
- Config arithmetic and MoE constraints are correct for all four fixed configurations.
- Representative rank mapping follows `rank = pp_stage * tp * dp + exp_rank * tp`.
- Dry-run generates CSV with exact header, exact field count, and expected `measured_ranks_count`.
- `estimated_5_iters_seconds` equals `single_iter_wallclock_seconds * 5` for each CSV row.

### 3. Test Results and Evidence
| Suite | Result | Evidence |
|------|--------|----------|
| Static syntax check | PASS | `syntax_ok_script`, `syntax_ok_unit`, `syntax_ok_integration` |
| Unit test | PASS | `[PASS] Qwen3-A3B MoE scaling wall-clock config checks passed.` |
| Integration dry-run | PASS | `[PASS] Qwen3-A3B MoE dry-run integration checks passed.` |

#### Key dry-run evidence
- Artifact root: `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_084535_2712073`
- CSV path:
  `tests/integration/artifacts/dryrun_qwen3_a3b_moe_20260304_084535_2712073/qwen3_a3b_moe_scaling_wallclock_timing.csv`
- Expected `measured_ranks_count` by row:
  - `32`
  - `128`
  - `512`
  - `1024`

### 4. Real Measurement Status
- Status: **Pending**
- Planned command:
  ```bash
  SCALE_GPU=<idle_gpu_id> \
  bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
    2>&1 | tee task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_YYYYMMDD_HHMMSS.log
  ```
