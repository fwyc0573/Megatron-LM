## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added Round6-8-baseline B1 strict-grad-replay implementation and validation (single-pass fail-fast probe + 3-run two-pass repeated pairing evidence) |

## Test Report: Stage-2 B1 on Round6-8 Baseline (`SCALING_STRICT_GRAD_REPLAY`)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Execution workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_o2`  
**Archive workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Code Changes (Round6-8 baseline worktree)

- `megatron/training/arguments.py`
  - added `--scaling-strict-grad-replay` (default-off).
- `megatron/training/training.py`
  - added `_is_scaling_profile_window(...)` and `_should_fail_on_missing_scaling_grad_replay(...)` helpers.
  - in `_build_scaling_output_tensor_grad(...)`, strict mode now raises `FileNotFoundError` when profiled backward cannot load grad replay cache.
- `examples/pretrain_deepseek_v3_moe.sh`
  - added `SCALING_STRICT_GRAD_REPLAY` env knob (`0/1`, fail-fast validation + argument passthrough).
- `tests/unit_tests/test_training_optimizer_microphase.py`
  - added parser tests for strict flag and helper behavior tests for profile-window gating.

### 2) Validation Criteria

- B1 hypothesis: random grad fallback in scaling backward should be removed from profiled window, and missing replay cache should fail fast.
- Required checks:
  1. strict mode single-pass should explicitly fail when replay cache is missing (no silent fallback).
  2. strict mode should run successfully when cache is pre-populated (two-pass replay setup).
  3. under fixed 3-run repeated pairing protocol, compare fidelity metrics (`forward_step/backward_step/optimizer_step`) should be evaluated via `op_rank_median_aux_summary`.

### 3) Executed Commands

```bash
# unit/static checks
pytest -q tests/unit_tests/test_training_optimizer_microphase.py
python -m py_compile megatron/training/training.py megatron/training/arguments.py tests/unit_tests/test_training_optimizer_microphase.py
bash -n examples/pretrain_deepseek_v3_moe.sh

# strict single-pass probe (expected fail-fast)
MODE=scaling ... TRACE_OPTIMIZER_MICROPHASES=1 \
SCALING_STRICT_GRAD_REPLAY=1 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_REPLAY_CACHE_TAG=stage2_round68_b1_strict_probe_singlepass \
bash examples/pretrain_deepseek_v3_moe.sh

# two-pass strict replay (3 runs)
# runN: distributed -> scaling warm pass (strict=0) -> scaling strict pass (strict=1) -> compare
MODE=distributed ... TRACE_OPTIMIZER_MICROPHASES=1 ...
MODE=scaling ... TRACE_OPTIMIZER_MICROPHASES=1 SCALING_STRICT_GRAD_REPLAY=0 ... SCALING_REPLAY_CACHE_TAG=stage2_round68_b1_strict_runN ...
MODE=scaling ... TRACE_OPTIMIZER_MICROPHASES=1 SCALING_STRICT_GRAD_REPLAY=1 ... SCALING_REPLAY_CACHE_TAG=stage2_round68_b1_strict_runN ...
python tests/performance/compare_qwen_trace_comp.py ... --distributed-subtract-comm --repeat-report .../deepseek_v3_stage2_repeat_round68_b1_strict_subtract.jsonl
```

### 4) Test Results and Evidence

#### 4.1 Unit/static validation

- `pytest` result: **9 passed**.
- evidence log:
  - `logs/stage2_b1_round68_test_training_strict_grad_replay.log`

#### 4.2 Single-pass strict probe (expected fail-fast)

- result: **FAIL as expected** at first profiled backward on rank0.
- key error (strict mode):
  - missing `grad_to_rank0_iter3.pt` / `grad_to_rank0.pt` under current cache tag.
- evidence:
  - `logs/deepseek_v3_stage2_scaling_round68_b1_strict_probe_singlepass.log`

#### 4.3 Two-pass strict replay, 3-run repeated pairing

Protocol fixed for all runs:
- `TRACE_START=4`, `TRAIN_ITERS=6`
- `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
- `TRACE_OPTIMIZER_MICROPHASES=1`
- rank order: `0,4,1,5,2,6,3,7`
- warm pass: `SCALING_STRICT_GRAD_REPLAY=0`
- strict pass: `SCALING_STRICT_GRAD_REPLAY=1`

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step |
|-----|----------------|-------------:|--------------:|---------------:|
| run1 | 20260228081605 | 5.11% | 10.02% | 7.93% |
| run2 | 20260228082131 | 5.84% | 9.51% | 9.85% |
| run3 | 20260228082618 | 10.25% | 9.20% | 9.29% |

Median-of-runs:
- `forward_step = 5.84%`
- `backward_step = 9.51%`
- `optimizer_step = 9.29%`
- `mean_3ops = 8.21%`
- `max_3ops = 9.51%`

Artifacts:
- distributed:
  - `logs/deepseek_v3_stage2_dist_round68_b1_strict_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_b1_strict_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_b1_strict_run3.log`
- scaling warm pass:
  - `logs/deepseek_v3_stage2_scaling_round68_b1_warmup_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_b1_warmup_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_b1_warmup_run3.log`
- scaling strict pass:
  - `logs/deepseek_v3_stage2_scaling_round68_b1_strict_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_b1_strict_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_b1_strict_run3.log`
- compare:
  - `logs/deepseek_v3_stage2_compare_round68_b1_strict_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_b1_strict_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_b1_strict_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_b1_strict_run1.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_b1_strict_run2.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_b1_strict_run3.stdout.log`
- repeat jsonl:
  - `logs/deepseek_v3_stage2_repeat_round68_b1_strict_subtract.jsonl`

### 5) Comparison vs previous O2-best baseline

Reference (Round6-8 O2 report, `TRACE_OPTIMIZER_MICROPHASES=1`, no strict mode):
- `forward=5.71%`, `backward=9.34%`, `optimizer=8.36%`
- `mean_3ops=7.80%`, `max_3ops=9.34%`

B1 strict two-pass median-of-runs delta (B1 - O2-best):
- `forward: +0.13%`
- `backward: +0.17%`
- `optimizer: +0.93%`
- `mean_3ops: +0.41%`
- `max_3ops: +0.17%`

### 6) Conclusion

- B1 strict replay logic is successfully implemented and verified:
  - single-pass missing cache now fail-fast (no silent random fallback in profiled backward window).
  - two-pass pre-populated cache can run strict mode end-to-end.
- On Round6-8 baseline under fixed repeated protocol, B1 strict two-pass **does not improve** fidelity metrics relative to O2-best baseline; overall it regresses slightly (notably optimizer).
- Practical implication:
  - strict replay is still valuable as an integrity guard/diagnostic mode;
  - it should remain **default-off diagnostic control**, not be promoted to current primary gate configuration.
