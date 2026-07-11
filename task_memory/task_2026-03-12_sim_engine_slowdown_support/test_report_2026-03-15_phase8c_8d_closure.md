## Test Report: Phase 8c/8d Workflow Closure

**Date**: 2026-03-15
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### Test Script Information
- Scripts / files:
  - `megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py`
  - `megatron-sim-engine/tools/data_prep/slowdown/prepare_case_kernel_metrics.py`
  - `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`
  - `megatron-sim-engine/tests/unit/test_build_trace_shaped_pp_schedule.py`
  - `megatron-sim-engine/tests/unit/test_build_trace_shaped_pp2_schedule.py`
  - `megatron-sim-engine/tests/unit/test_prepare_case_kernel_metrics.py`
  - `megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py`
  - `megatron-sim-engine/tests/unit/test_slowdown_predictor.py`
  - `tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`
  - `megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`
- Commands:
  ```bash
  export PYTHONPATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine:/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM:$PYTHONPATH

  pytest -q \
    megatron-sim-engine/tests/unit/test_build_trace_shaped_pp_schedule.py \
    megatron-sim-engine/tests/unit/test_build_trace_shaped_pp2_schedule.py \
    megatron-sim-engine/tests/unit/test_prepare_case_kernel_metrics.py \
    megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py \
    megatron-sim-engine/tests/unit/test_slowdown_predictor.py \
    tests/unit/test_gpt67b_ddp_slowdown_lightweight_helpers.py \
    megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py \
    megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py

  python megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py --help
  python megatron-sim-engine/tools/data_prep/slowdown/prepare_case_kernel_metrics.py --help
  bash -n tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh
  ```

### Validation Criteria
- Phase 8c auto schedule must be available as a standalone CLI entrypoint and covered by both function-level and CLI-level tests.
- Phase 8d case-local kernel metrics preparation must be available as a standalone CLI entrypoint and covered by both function-level and CLI-level tests.
- The self-contained GPT-6.7B lightweight workflow script must still parse cleanly and remain the canonical Phase 8d orchestrator.
- Workflow closure must be documented as “functionally complete” even though fresh targeted `NCU` recollection remains expensive on this machine.

### Test Results

| Check | Result | Details |
|-------|--------|---------|
| New CLI regression coverage | PASS | `5 passed` |
| Expanded Phase 8c/8d regression | PASS | `31 passed` |
| CLI help entrypoints | PASS | both scripts returned usage/help successfully |
| Lightweight E2E shell syntax | PASS | `bash -n` completed successfully |

### Evidence
- New CLI regression run:
  - `.....                                                                    [100%]`
  - `5 passed in 0.47s`
- Expanded Phase 8c/8d regression run:
  - `...............................                                          [100%]`
  - `31 passed in 0.60s`
- CLI help evidence captured during validation:
  - `build_trace_shaped_pp_schedule.py --help` printed the expected usage banner.
  - `prepare_case_kernel_metrics.py --help` printed the expected usage banner.
- Canonical functional workflow evidence remains frozen in:
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`
- Documentation/entrypoint index updated in:
  - `task_memory/task_2026-03-12_sim_engine_slowdown_support/notes.md`
  - `megatron-sim-engine/tools/README.md`
  - `megatron-sim-engine/tests/README.md`

### Failure Handling Notes
- Phase 8c/8d are now treated as closed from a workflow-completeness standpoint.
- Remaining targeted `NCU` wall-clock cost is recorded as a non-blocking practicality limitation, not as a functional blocker.
