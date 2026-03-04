## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Backported B1 strict-grad-replay guard to mainline (default-off) and completed unit/static validation |

## Test Report: B1 strict-grad-replay backport to mainline (default-off)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Code Changes

- `megatron/training/arguments.py`
  - added `--scaling-strict-grad-replay` (default-off).
- `megatron/training/training.py`
  - added `_is_scaling_profile_window(...)` and `_should_fail_on_missing_scaling_grad_replay(...)`.
  - in `_build_scaling_output_tensor_grad(...)`, strict mode now fail-fast on missing replay grad cache in profiled window.
- `examples/pretrain_deepseek_v3_moe.sh`
  - added `SCALING_STRICT_GRAD_REPLAY` env knob, `0/1` validation, and scaling-arg passthrough.
- `tests/unit_tests/test_training_optimizer_microphase.py`
  - added parser/helper tests for strict-grad-replay.

### 2) Validation Criteria

- Strict mode must be parseable and default-off.
- Helper behavior must only activate fail-fast in scaling profiled window.
- Modified Python modules must pass syntax checks.
- DeepSeek stage-2 script must pass shell syntax check.

### 3) Commands

```bash
pytest -q tests/unit_tests/test_training_optimizer_microphase.py
python -m py_compile megatron/training/training.py megatron/training/arguments.py tests/unit_tests/test_training_optimizer_microphase.py
bash -n examples/pretrain_deepseek_v3_moe.sh
```

### 4) Results

- `pytest` result: **15 passed**.
- `py_compile`: PASS.
- `bash -n`: PASS.

### 5) Evidence

- log:
  - `logs/stage2_b1_mainline_test_training_strict_grad_replay.log`

### 6) Conclusion

- B1 strict replay guard is now available on mainline as a **default-off integrity control**.
- This backport does not change default runtime behavior and is safe for diagnostic activation when replay-cache completeness needs to be enforced.
