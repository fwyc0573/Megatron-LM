## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added Round6-8-baseline O2 A/B experiment (`TRACE_OPTIMIZER_MICROPHASES=0` vs `1`) with 3-run repeated pairing evidence |

## Test Report: Stage-2 O2 A/B on Round6-8 Baseline (`TRACE_OPTIMIZER_MICROPHASES=0` vs `1`)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Workspace (execution)**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_o2`  
**Workspace (report/log archive)**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

- Baseline setup:
  - worktree created from Round6-8 baseline commit `a3158883`
  - branch: `round68_o2_exp`
  - cherry-picked diagnostic commit: `3a50265d` (enables microphase flag and traces)
- Training script:
  - `examples/pretrain_deepseek_v3_moe.sh`
- Compare script:
  - `tests/performance/compare_qwen_trace_comp.py`

### 2) Validation Criteria

- Objective (O2 hypothesis): determine whether enabling optimizer microphase instrumentation perturbs top-level gate metrics.
- Fixed protocol for both A/B arms:
  - `TRACE_START=4`, `TRAIN_ITERS=6`
  - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
  - scaling fixed rank order: `0,4,1,5,2,6,3,7`
  - scaling replay knobs kept constant (no round11/12 semantic knobs used in this baseline runset)
  - repeated pairing: 3 paired runs per arm
- Compare metrics:
  - `op_rank_median_aux_summary` on `forward_step/backward_step/optimizer_step`
  - median-of-runs summary over three runs.

### 3) Executed Commands

```bash
# env check
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /opt/anaconda/envs/myenv_yc
python --version
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits

# A arm: TRACE_OPTIMIZER_MICROPHASES=0, run1/2/3
MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 \
TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
TRACE_OPTIMIZER_MICROPHASES=0 MASTER_PORT=<11060|12060|13060> \
bash examples/pretrain_deepseek_v3_moe.sh > .../deepseek_v3_stage2_dist_round68_o2_micro0_run<RUN>.log 2>&1

MODE=scaling MODEL_PROFILE=smoke FAKE_WORLD_SIZE=8 FAKE_PP=2 FAKE_TP=1 FAKE_EXP=2 \
TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
TRACE_OPTIMIZER_MICROPHASES=0 SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 \
SCALE_GPU=2 MASTER_PORT=<11150|12150|13150> \
SCALING_REPLAY_CACHE_TAG=stage2_round68_o2_micro0_run<RUN> \
bash examples/pretrain_deepseek_v3_moe.sh > .../deepseek_v3_stage2_scaling_round68_o2_micro0_run<RUN>.log 2>&1

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 --pair-timestamp <run pair ts> --distributed-subtract-comm \
  --repeat-report .../deepseek_v3_stage2_repeat_round68_o2_micro0_subtract.jsonl \
  --report-path .../deepseek_v3_stage2_compare_round68_o2_micro0_run<RUN>.log

# B arm: TRACE_OPTIMIZER_MICROPHASES=1, run1/2/3
# Same commands as above with TRACE_OPTIMIZER_MICROPHASES=1
# Ports used:
#   distributed: 14060 / 15060 / 16060
#   scaling base: 14150 / 15150 / 16150
# repeat-report:
#   .../deepseek_v3_stage2_repeat_round68_o2_micro1_subtract.jsonl
```

### 4) Test Results and Evidence

#### 4.1 A arm (`TRACE_OPTIMIZER_MICROPHASES=0`)

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step |
|-----|----------------|-------------:|--------------:|---------------:|
| run1 | 20260228074837 | 4.79% | 12.07% | 7.00% |
| run2 | 20260228075141 | 8.37% | 7.13% | 8.61% |
| run3 | 20260228075450 | 13.00% | 10.47% | 10.42% |

Median-of-runs:

- `forward_step = 8.37%`
- `backward_step = 10.47%`
- `optimizer_step = 8.61%`
- `mean_3ops = 9.15%`
- `max_3ops = 10.47%`

#### 4.2 B arm (`TRACE_OPTIMIZER_MICROPHASES=1`)

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step |
|-----|----------------|-------------:|--------------:|---------------:|
| run1 | 20260228075756 | 10.93% | 9.34% | 12.90% |
| run2 | 20260228080102 | 4.79% | 8.09% | 6.61% |
| run3 | 20260228080406 | 5.71% | 11.82% | 8.36% |

Median-of-runs:

- `forward_step = 5.71%`
- `backward_step = 9.34%`
- `optimizer_step = 8.36%`
- `mean_3ops = 7.80%`
- `max_3ops = 9.34%`

#### 4.3 A/B delta (B - A, median-of-runs)

- `forward_step: -2.66%`
- `backward_step: -1.13%`
- `optimizer_step: -0.25%`
- `mean_3ops: -1.35%`
- `max_3ops: -1.13%`

### 5) Conclusion (O2 hypothesis verdict)

- Under this Round6-8-baseline A/B experiment, enabling microphase instrumentation (`TRACE_OPTIMIZER_MICROPHASES=1`) did **not** worsen top-level op-rank-median; it improved median-of-runs on all three target ops.
- Therefore, O2 as originally phrased (“microphase instrumentation likely perturbs hot path and should reduce fidelity”) is **not supported** by this experiment on this baseline/protocol.
- However, both A and B remain above `<=5%` gate, and run-to-run spread is still material, so next optimization should continue focusing on main residual sources rather than attributing primary blame to microphase toggle itself.

### 6) Evidence Locations

- Unit test log:
  - `logs/stage2_o2_round68_test_training_optimizer_microphase.log`
- A arm logs:
  - `logs/deepseek_v3_stage2_dist_round68_o2_micro0_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_o2_micro0_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_o2_micro0_run3.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o2_micro0_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o2_micro0_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o2_micro0_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_o2_micro0_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_o2_micro0_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_o2_micro0_run3.log`
  - `logs/deepseek_v3_stage2_repeat_round68_o2_micro0_subtract.jsonl`
- B arm logs:
  - `logs/deepseek_v3_stage2_dist_round68_o2_micro1_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_o2_micro1_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_o2_micro1_run3.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o2_micro1_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o2_micro1_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_o2_micro1_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_o2_micro1_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_o2_micro1_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_o2_micro1_run3.log`
  - `logs/deepseek_v3_stage2_repeat_round68_o2_micro1_subtract.jsonl`
