## Test Report: Strict Bool Parsing and Qwen MoE Overlap Entry

**Date**: 2026-03-15
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### Test Script Information
- Scripts / files:
  - `megatron/training/arguments.py`
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - `examples/pretrain_qwen3_30b_a3b_moe_ddp_overlap_trace.sh`
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py`
  - `tests/unit_tests/test_strict_bool_argument_parsing.py`
  - `tests/unit/test_qwen3_a3b_moe_overlap_script.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`
  - `megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace.py`
  - `tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py`
  - `tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py`
  - `tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py`
  - `tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py`
- Commands:
  ```bash
  export PYTHONPATH=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine:/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM:$PYTHONPATH

  pytest -q \
    tests/unit_tests/test_strict_bool_argument_parsing.py \
    tests/unit_tests/test_trace_ddp_grad_overlap_args.py \
    tests/unit/test_qwen3_a3b_moe_overlap_script.py \
    megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py \
    megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py \
    tests/unit_tests/distributed/test_ddp_overlap_trace.py \
    tests/unit_tests/distributed/test_ddp_overlap_trace_scaling.py \
    tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
    tests/unit_tests/profiler/test_cmd_ddp_overlap_schema.py \
    tests/unit_tests/test_training_scaling_ddp_overlap_finalize_base.py

  python -m py_compile \
    megatron/training/arguments.py \
    mg_scheduling/arguments.py \
    tools/retro/sft/sft_retro.py \
    megatron-sim-engine/src/scheduler/mg_scheduling/arguments.py

  bash -n examples/pretrain_qwen3_30b_a3b_moe.sh
  bash -n examples/pretrain_qwen3_30b_a3b_moe_ddp_overlap_trace.sh
  ```

### Validation Criteria
- All active former `type=bool` CLI flags must parse strict booleans instead of Python truthiness.
- `--do-trace False` and `--do-trace 0` must parse to `False`.
- Invalid `--do-trace` bool tokens must be rejected at argument parsing time.
- `trace_ddp_grad_overlap` must still auto-enable when `overlap_grad_reduce=True` and tracing stays on.
- `examples/pretrain_qwen3_30b_a3b_moe.sh` must accept `OVERLAP_GRAD_REDUCE=1` and forward overlap arguments in both:
  - `MODE=distributed` tracing flow,
  - `MODE=scaling` simulate flow.
- The Qwen script must not explicitly pass `--trace-ddp-grad-overlap`; Megatron should auto-enable it from runtime args.
- `examples/pretrain_qwen3_30b_a3b_moe_ddp_overlap_trace.sh` must stay one-for-one aligned with the DeepSeek wrapper style while defaulting overlap tracing on.
- Archived `backup/` and `legacy/` files may still contain historical `type=bool` code, but active runtime entry points must not.

### Test Results

| Check | Result | Details |
|-------|--------|---------|
| Strict-bool + script unit tests | PASS | `23 passed` |
| Expanded overlap/slowdown regression | PASS | `55 passed` |
| Python syntax validation | PASS | `py_compile` completed successfully |
| Qwen script shell syntax | PASS | `bash -n` completed successfully |

### Evidence
- Targeted parser/script run:
  - `.......................                                                  [100%]`
  - `23 passed, 3 warnings in 8.00s`
- Expanded regression run:
  - `.......................................................                  [100%]`
  - `55 passed, 3 warnings in 8.88s`
- Behavioral evidence captured by tests:
  - `tests/unit_tests/test_trace_ddp_grad_overlap_args.py` verifies `--do-trace False`, `--do-trace 0`, invalid-token rejection, and overlap auto-enable.
  - `tests/unit/test_qwen3_a3b_moe_overlap_script.py` stubs `torchrun` and verifies that both distributed and scaling paths forward `--overlap-grad-reduce`, `--ddp-bucket-size`, and `--do-trace True` without explicitly injecting `--trace-ddp-grad-overlap`.

- Additional validation uncovered and fixed a latent issue in `tools/retro/sft/sft_retro.py`:
  - `_argparse_bool()` now relies on `argparse.ArgumentTypeError`, so the module also needed an explicit `import argparse`; this is now covered by the strict-bool unit test.

### Failure Handling Notes
- Root cause of the parser bug was `argparse` using `type=bool`, which treats any non-empty string as truthy.
- The active-scope fix now covers all remaining runtime `type=bool` CLI entry points in the main repo plus the active sim-engine scheduler copy. Historical `backup/` / `legacy/` snapshots remain untouched on purpose.
