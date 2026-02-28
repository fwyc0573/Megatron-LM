## Test Report: DeepSeek-V3 Stage-2 Round12 (`post_optimizer` Replay Write, Repeated Pairing)

**Date**: 2026-02-28  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

- Training script:
  - `examples/pretrain_deepseek_v3_moe.sh`
- Compare script:
  - `tests/performance/compare_qwen_trace_comp.py`
- Aggregation input:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_microphase_round12_postwrite_subtract.jsonl`

- Reproducible command pattern (run1/run2/run3 use different `MASTER_PORT` and log name):

```bash
# Optional pre-run GPU check (pick SM-utilization 0 GPU for scaling)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits

# Distributed
MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 \
TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
TRACE_OPTIMIZER_MICROPHASES=1 MASTER_PORT=<DIST_PORT> \
bash examples/pretrain_deepseek_v3_moe.sh \
  > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_round12_postwrite_run<RUN>.log 2>&1

# Scaling (semantic-touching flag default-off in code path, only this run explicitly enables it)
MODE=scaling MODEL_PROFILE=smoke FAKE_WORLD_SIZE=8 FAKE_PP=2 FAKE_TP=1 FAKE_EXP=2 \
TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
TRACE_OPTIMIZER_MICROPHASES=1 SCALING_REPLAY_WRITE_PHASE=post_optimizer \
SCALING_ALIGN_SCHEDULER_INCREMENT=0 SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALE_GPU=<IDLE_GPU> MASTER_PORT=<SCALING_BASE_PORT> \
bash examples/pretrain_deepseek_v3_moe.sh \
  > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_round12_postwrite_run<RUN>.log 2>&1

# Compare (append repeat report)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step,optimizer_main_update,optimizer_state_update,optimizer_post_update \
  --threshold-pct 5 \
  --pair-timestamp <PAIR_TS> \
  --distributed-subtract-comm \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_microphase_round12_postwrite_subtract.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run<RUN>.log
```

- This round used pair timestamps:
  - run1: `20260228051919`
  - run2: `20260228052156`
  - run3: `20260228052432`

### 2) Validation Criteria

- Primary gate: `op_rank_median_aux_summary` for each op should satisfy `<= 5%`.
- Required ops:
  - `forward_step`
  - `backward_step`
  - `optimizer_step`
  - `optimizer_main_update` (diagnostic key op)
- Secondary evidence:
  - 3-run repeated pairing with median-of-runs (anti-outlier view).
- Consistency checks:
  - distributed/scaling both use `TRACE_START=4`, `TRAIN_ITERS=6`, `TRACE_OPTIMIZER_MICROPHASES=1`.
  - scaling uses fixed rank-order and fixed high port segment.

### 3) Test Results and Evidence

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step | optimizer_main_update | Gate |
|-----|----------------|--------------|---------------|----------------|-----------------------|------|
| run1 | 20260228051919 | 5.69% | 8.24% | 12.44% | 12.03% | FAIL |
| run2 | 20260228052156 | 8.23% | 12.65% | 7.76% | 7.70% | FAIL |
| run3 | 20260228052432 | 8.13% | 13.03% | 6.11% | 6.14% | FAIL |

**Median-of-runs (3 runs)**

- `forward_step = 8.13%`
- `backward_step = 12.65%`
- `optimizer_step = 7.76%`
- `optimizer_main_update = 7.70%`

**Result Summary**: **FAIL**（本轮未达到 `<=5%` 目标）

### 4) Evidence Locations

- Compare reports:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run1.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run2.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run3.log`
- Repeat aggregate JSONL:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_microphase_round12_postwrite_subtract.jsonl`
- Runtime logs:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_round12_postwrite_run1.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_round12_postwrite_run1.log`
  - run2/run3 files follow the same naming pattern.

### 5) Failure Notes and Next Action

- `post_optimizer` replay-write semantic-touching knob shows improvement relative to older baselines, but cannot push `backward_step`/`optimizer_step` below 5% in repeated protocol.
- Residual remains concentrated in `optimizer_main_update` + stage1 ranks; run-to-run drift remains material.
- Next suggested direction: keep default path unchanged, add finer-grained `optimizer_main_update` diagnostics before proposing new semantic change.
