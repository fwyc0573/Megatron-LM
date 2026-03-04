## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added Round6-8-baseline O1 A/B report for pre-CMD optimizer drain (`torch.cuda.synchronize()`), with fixed rank7-cap repeated pairing x5 and noise-floor-aware verdict |

## Test Report: Stage-2 O1 A/B on Round6-8 Baseline (`TRACE_OPTIMIZER_PRE_CMD_DRAIN`)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Execution workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_noise`  
**Archive workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) O1 Frozen Mechanism (single variable)

- Added default-off trace knob: `--trace-optimizer-pre-cmd-drain`.
- Behavior when enabled:
  - call `torch.cuda.synchronize()` **immediately before entering top-level `optimizer_step` CMD**.
- Applied symmetrically in both paths:
  - distributed `train_step(...)`
  - scaling profile loop optimizer branch.
- Explicitly excluded from O1 variable:
  - no `dist.barrier()`;
  - no CMD start/end boundary relocation;
  - no optimizer algorithm or microphase semantics changes.

### 2) Code Changes

- `megatron/training/arguments.py`
  - added `--trace-optimizer-pre-cmd-drain` (default-off).
- `megatron/training/training.py`
  - added `_maybe_optimizer_pre_cmd_drain(args)` helper.
  - called it right before entering `optimizer_step` CMD in distributed/scaling paths.
- `examples/pretrain_deepseek_v3_moe.sh`
  - added env knob `TRACE_OPTIMIZER_PRE_CMD_DRAIN` (`0/1`) and trace-arg passthrough.
- `tests/unit_tests/test_training_optimizer_microphase.py`
  - parser default/on tests for O1 flag.
  - helper behavior test for synchronize-call gating.

### 3) Validation Commands

```bash
pytest -q tests/unit_tests/test_training_optimizer_microphase.py
python -m py_compile megatron/training/training.py megatron/training/arguments.py tests/unit_tests/test_training_optimizer_microphase.py
bash -n examples/pretrain_deepseek_v3_moe.sh
```

Validation result:
- `9 passed` + py_compile PASS + script syntax PASS.
- evidence:
  - `logs/stage2_o1_round68_test_training_pre_cmd_drain.log`

### 4) Runtime Protocol (A/B, repeated pairing x5)

Common fixed protocol:
- `TRACE_START=4`, `TRAIN_ITERS=6`
- `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
- `TRACE_OPTIMIZER_MICROPHASES=1`
- scaling rank order: `0,4,1,5,2,6,3,7`
- `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`
- compare: `--distributed-subtract-comm`, `op_rank_median_aux_summary`
- pairing cap policy: **rank7 end-of-run timestamp**

A arm:
- `TRACE_OPTIMIZER_PRE_CMD_DRAIN=0`

B arm:
- `TRACE_OPTIMIZER_PRE_CMD_DRAIN=1`

### 5) A/B Results (single-run and median-of-runs)

#### 5.1 A arm (`drain=0`)

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step |
|-----|----------------|-------------:|--------------:|---------------:|
| run1 | 20260228141333 | 12.84% | 7.65% | 5.92% |
| run2 | 20260228141607 | 14.50% | 10.06% | 8.23% |
| run3 | 20260228141843 | 4.55% | 11.77% | 9.67% |
| run4 | 20260228142117 | 16.53% | 17.87% | 5.63% |
| run5 | 20260228142352 | 8.10% | 6.21% | 10.90% |

Median-of-runs:
- `forward_step=12.84%`
- `backward_step=10.06%`
- `optimizer_step=8.23%`
- `mean_3ops=8.80%`
- `max_3ops=12.84%`

#### 5.2 B arm (`drain=1`)

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step |
|-----|----------------|-------------:|--------------:|---------------:|
| run1 | 20260228142648 | 11.55% | 14.63% | 11.05% |
| run2 | 20260228142923 | 13.19% | 10.45% | 12.67% |
| run3 | 20260228143302 | 5.36% | 19.87% | 6.19% |
| run4 | 20260228143537 | 20.68% | 22.19% | 10.00% |
| run5 | 20260228143812 | 6.51% | 12.41% | 7.86% |

Median-of-runs:
- `forward_step=11.55%`
- `backward_step=14.63%`
- `optimizer_step=10.00%`
- `mean_3ops=12.10%`
- `max_3ops=14.63%`

#### 5.3 Delta (B - A, median-of-runs)

- `forward_step: -1.29%`
- `backward_step: +4.57%`
- `optimizer_step: +1.77%`
- `mean_3ops: +3.30%`
- `max_3ops: +1.79%`

### 6) Noise-floor-aware evaluation

Using the previously measured pure-noise baseline (`repeat x5`, no code change):
- median was `forward=7.16%`, `backward=11.67%`, `optimizer=8.28%`.
- per-op run range was about `5%~6%`.

O1 acceptance criteria requested:
- effect should exceed noise floor,
- and spread metrics (`range` / `IQR`) should decrease consistently.

Observed spread comparison (B vs A):
- range:
  - `forward +3.34%`, `backward +0.08%`, `optimizer +1.21%`
  - `mean_3ops +3.76%`, `max_3ops +2.81%`
- IQR:
  - `forward +0.28%`, `backward +3.34%`, `optimizer -0.56%`
  - `mean_3ops -0.33%`, `max_3ops +3.95%`

Verdict:
- O1 (`pre-CMD drain=1`) **does not beat noise floor** and **does not show synchronous spread reduction**.
- On this baseline/protocol, O1 causes clear regression in `backward_step`, `optimizer_step`, and composite metrics.

### 7) Conclusion

- O1 mechanism implementation is correct and strictly single-variable.
- Runtime evidence rejects O1 as a beneficial gate strategy under current protocol.
- Recommendation:
  - keep `TRACE_OPTIMIZER_PRE_CMD_DRAIN` as default-off diagnostic knob only;
  - do not promote it to primary fidelity gate path.

### 8) Evidence

- repeat artifacts:
  - `logs/deepseek_v3_stage2_repeat_round68_o1_drain0_subtract.jsonl`
  - `logs/deepseek_v3_stage2_repeat_round68_o1_drain1_subtract.jsonl`
- A arm logs:
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain0_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain0_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain0_run3.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain0_run4.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain0_run5.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain0_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain0_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain0_run3.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain0_run4.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain0_run5.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run4.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run5.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run1.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run2.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run3.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run4.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain0_run5.stdout.log`
- B arm logs:
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain1_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain1_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain1_run3.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain1_run4.log`
  - `logs/deepseek_v3_stage2_dist_round68_o1_drain1_run5.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain1_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain1_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain1_run3.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain1_run4.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o1_drain1_run5.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run4.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run5.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run1.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run2.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run3.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run4.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_o1_drain1_run5.stdout.log`
