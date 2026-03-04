## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added Round6-8-baseline pure-noise quantification report (no code changes, 5-run repeated pairing) and noise-floor verdict |

## Test Report: Stage-2 Round6-8 Baseline Noise Floor (microphase=1, repeated pairing x5)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Execution workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_noise`  
**Archive workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Goal

Quantify run-to-run noise floor **without any code changes** before O1 implementation.

### 2) Baseline and Protocol

- Baseline code state:
  - worktree from commit `3a50265d` (`round68_noise_floor`) 
  - no new patches applied in this worktree during this test.
- Fixed protocol (all 5 runs):
  - `TRACE_START=4`, `TRAIN_ITERS=6`
  - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
  - `TRACE_OPTIMIZER_MICROPHASES=1`
  - scaling rank order: `0,4,1,5,2,6,3,7`
  - `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`
  - compare: `--distributed-subtract-comm`, `op_rank_median_aux_summary`

### 3) Commands

```bash
# 5x repeated paired runs, no code changes
MODE=distributed ... TRACE_OPTIMIZER_MICROPHASES=1 ...
MODE=scaling ... TRACE_OPTIMIZER_MICROPHASES=1 ...
python tests/performance/compare_qwen_trace_comp.py ... --distributed-subtract-comm --repeat-report ...
```

### 4) Pairing Note (important)

- Initial compare attempt used `rank0` timestamp cap and could under-cover later ranks in sequential scaling run.
- Final noise report uses **rank7 timestamps** (end-of-run cap) for strict same-batch pairing:
  - run1 `20260228134044`
  - run2 `20260228134319`
  - run3 `20260228134554`
  - run4 `20260228134829`
  - run5 `20260228135103`

### 5) Results (single-run op-rank-median)

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step | mean_3ops | max_3ops |
|-----|----------------|-------------:|--------------:|---------------:|----------:|---------:|
| run1 | 20260228134044 | 5.64% | 14.66% | 9.09% | 9.80% | 14.66% |
| run2 | 20260228134319 | 10.36% | 10.54% | 8.28% | 9.73% | 10.54% |
| run3 | 20260228134554 | 7.16% | 11.67% | 11.13% | 9.99% | 11.67% |
| run4 | 20260228134829 | 8.37% | 15.85% | 4.89% | 9.70% | 15.85% |
| run5 | 20260228135103 | 4.57% | 10.55% | 6.12% | 7.08% | 10.55% |

Median-of-runs:
- `forward_step = 7.16%`
- `backward_step = 11.67%`
- `optimizer_step = 8.28%`
- `mean_3ops = 9.73%`
- `max_3ops = 11.67%`

Spread (single-run range):
- `forward_step`: min `4.57%`, max `10.36%`, range `5.79%`
- `backward_step`: min `10.54%`, max `15.85%`, range `5.31%`
- `optimizer_step`: min `4.89%`, max `11.13%`, range `6.24%`

### 6) Noise-Floor Verdict

- Under this smoke protocol and measurement setup, noise floor is high enough that:
  - all three op medians stay above 5% (`7.16 / 11.67 / 8.28`),
  - run-to-run range is about `5%~6%` for each op.
- Therefore, before interpreting small code-level A/B deltas (<~1-2%) as real gains, we must either:
  1. increase measurement stability (longer profile window / stronger pairing control / more repeats), or
  2. require larger effect size for O1 acceptance.

### 7) Evidence

- repeat aggregate:
  - `logs/deepseek_v3_stage2_repeat_round68_noise_floor_micro1_subtract.jsonl`
- per-run logs:
  - `logs/deepseek_v3_stage2_dist_round68_noise_floor_micro1_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_noise_floor_micro1_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_noise_floor_micro1_run3.log`
  - `logs/deepseek_v3_stage2_dist_round68_noise_floor_micro1_run4.log`
  - `logs/deepseek_v3_stage2_dist_round68_noise_floor_micro1_run5.log`
  - `logs/deepseek_v3_stage2_scaling_round68_noise_floor_micro1_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_noise_floor_micro1_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_noise_floor_micro1_run3.log`
  - `logs/deepseek_v3_stage2_scaling_round68_noise_floor_micro1_run4.log`
  - `logs/deepseek_v3_stage2_scaling_round68_noise_floor_micro1_run5.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run4.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run5.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run1.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run2.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run3.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run4.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_noise_floor_micro1_run5.stdout.log`
