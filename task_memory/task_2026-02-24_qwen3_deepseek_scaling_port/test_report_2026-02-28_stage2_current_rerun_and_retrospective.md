## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added current-latest 3-run rerun evidence (Round12 protocol), cross-round comparison (Round4/Round6-8), best-round decision, retrospective matrix, and forward optimization plan |

## Test Report: DeepSeek-V3 Stage-2 Current Rerun and Retrospective

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

- Training script:
  - `examples/pretrain_deepseek_v3_moe.sh`
- Compare script:
  - `tests/performance/compare_qwen_trace_comp.py`
- Repeat aggregate artifact:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_current_subtract.jsonl`

#### Executed Commands (Round12 protocol, run1/run2/run3)

```bash
# Pre-check
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /opt/anaconda/envs/myenv_yc
python --version
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits

# run1 distributed
MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 \
TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
TRACE_OPTIMIZER_MICROPHASES=1 MASTER_PORT=9760 \
bash examples/pretrain_deepseek_v3_moe.sh \
  > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run1.log 2>&1

# run1 scaling
MODE=scaling MODEL_PROFILE=smoke FAKE_WORLD_SIZE=8 FAKE_PP=2 FAKE_TP=1 FAKE_EXP=2 \
TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
TRACE_OPTIMIZER_MICROPHASES=1 SCALING_REPLAY_WRITE_PHASE=post_optimizer \
SCALING_ALIGN_SCHEDULER_INCREMENT=0 SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALE_GPU=2 MASTER_PORT=9750 SCALING_REPLAY_CACHE_TAG=stage2_current_run1 \
bash examples/pretrain_deepseek_v3_moe.sh \
  > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run1.log 2>&1

# run1 compare
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step,optimizer_main_update,optimizer_state_update,optimizer_post_update \
  --threshold-pct 5 \
  --pair-timestamp 20260228072907 \
  --distributed-subtract-comm \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_current_subtract.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_current_run1.log \
  > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_current_run1.stdout.log 2>&1

# run2/run3 follow the same protocol with isolated ports/cache tags:
# run2: dist=9860, scale_base=9850, pair_timestamp=20260228073218
# run3: dist=9960, scale_base=9950, pair_timestamp=20260228073523
```

### 2) Validation Criteria

- Protocol consistency:
  - `TRACE_START=4`, `TRAIN_ITERS=6`
  - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
  - `TRACE_OPTIMIZER_MICROPHASES=1`
  - scaling fixed knobs:
    - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`
    - `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
    - `SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7`
- Primary metric:
  - `op_rank_median_aux_summary` for `forward_step/backward_step/optimizer_step`
- Repeat metric:
  - median-of-runs over three paired runs.

### 3) Test Results and Evidence

#### 3.1 Runtime outcome

| Run | Distributed | Scaling | Compare | Result |
|-----|-------------|---------|---------|--------|
| run1 | PASS | PASS | report generated | FAIL (threshold) |
| run2 | PASS | PASS | report generated | FAIL (threshold) |
| run3 | PASS | PASS | report generated | FAIL (threshold) |

- Runtime evidence logs:
  - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run1.log`
  - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run2.log`
  - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run3.log`
  - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run1.log`
  - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run2.log`
  - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run3.log`

#### 3.2 Current rerun metrics (single-run)

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step | optimizer_main_update |
|-----|----------------|-------------:|--------------:|---------------:|----------------------:|
| run1 | 20260228072907 | 5.21% | 12.03% | 11.61% | 11.28% |
| run2 | 20260228073218 | 9.35% | 17.87% | 9.43% | 9.37% |
| run3 | 20260228073523 | 6.14% | 15.11% | 7.84% | 7.56% |

- Compare evidence logs:
  - `logs/deepseek_v3_stage2_compare_trace4_iter6_current_run1.log`
  - `logs/deepseek_v3_stage2_compare_trace4_iter6_current_run2.log`
  - `logs/deepseek_v3_stage2_compare_trace4_iter6_current_run3.log`

#### 3.3 Current rerun metrics (median-of-runs)

From `logs/deepseek_v3_stage2_repeat_current_subtract.jsonl`:

- `forward_step = 6.14%`
- `backward_step = 15.11%`
- `optimizer_step = 9.43%`

Composite:

- `mean_3ops = (6.14 + 15.11 + 9.43) / 3 = 10.23%`
- `max_3ops = 15.11%`

#### 3.4 Cross-round comparison (current vs Round4 vs Round6-8)

| Round | forward | backward | optimizer | mean_3ops | max_3ops |
|-------|--------:|---------:|----------:|----------:|---------:|
| Current (median-of-runs) | 6.14% | 15.11% | 9.43% | 10.23% | 15.11% |
| Round4 | 4.02% | 5.11% | 7.68% | 5.60% | 7.68% |
| Round6-8 (best single-run) | 3.06% | 7.51% | 5.97% | 5.51% | 7.51% |

Delta (current median-of-runs - baseline):

- vs Round4: `+2.12 / +10.00 / +1.75` (forward/backward/optimizer)
- vs Round6-8: `+3.08 / +7.60 / +3.46` (forward/backward/optimizer)

**Best-performing round (rule: mean first, max tie-break):**
- **Round6-8** (`mean_3ops=5.51%`, `max_3ops=7.51%`) is best overall.

### 4) Deep Retrospective (because current is not best)

#### Q1. Why Round9+ failed to improve over best round?

- Round9/10 introduced optimizer microphase instrumentation and then validated with microphase-enabled protocol; metrics degraded to around:
  - `forward ~7.79%`, `backward ~8.17%`, `optimizer ~13.22%` (single-run evidence family)
  - and median-of-runs remained high.
- Round11/12 semantic-touching knobs (`post_optimizer` replay write, scheduler increment align) improved some local patterns but did not close main residual:
  - `optimizer_main_update` still around `~6%` to `~12%` in different runs.
- Dominant residual remained concentrated in `optimizer_main_update` and stage1 ranks, while backward drift stayed large.

**Conclusion**: Round9+ changes improved observability and some stability, but they did not hit the dominant fidelity bottleneck strongly enough.

#### Q2. Were fixes conceptually incorrect?

- **Partially yes** for using microphase instrumentation as an implicit fidelity-improvement path:
  - microphase is excellent for diagnosis, but adding extra trace boundaries in optimizer hot path can perturb timing and should not be treated as a primary “improvement” path.
- **Partially no** for replay-write-phase:
  - this is a valid hypothesis-driven adjustment and showed localized benefits, but insufficient to pass the global gate.

**Conclusion**: not all fixes were wrong; some were “right tool for diagnosis, wrong tool for direct optimization.”

#### Q3. Were root causes misdiagnosed?

- Early emphasis on `optimizer_state_update` / `optimizer_post_update` was not supported by phase-aware evidence.
- Phase-aware compare repeatedly showed `optimizer_main_update` tracks the top-level `optimizer_step` residual most closely.

**Conclusion**: there was partial mis-attribution in early hypotheses; the main residual driver is still `optimizer_main_update`, not short-tail post/scheduler phases.

#### Q4. Did protocol/measurement noise mask regressions?

- Yes, noise is material:
  - Round6-8 showed large gap between best single-run and repeat median-of-runs.
  - Current rerun also has wide run-to-run spread (`forward 5.21%~9.35%`, `backward 12.03%~17.87%`, `optimizer 7.84%~11.61%`).
- But noise does **not** fully explain current underperformance versus Round4/Round6-8 because the gap is too large (especially backward).

**Conclusion**: noise amplifies uncertainty on near-threshold judgments, but it does not overturn the “current < best historical” result.

#### Q5. Which later fixes are still valid and should be preserved?

Preserve (recommended):

1. `--trace-optimizer-microphases` diagnostic path (default-off) + corresponding tests.
2. `--scaling-replay-write-phase` diagnostic path (default-off) + corresponding tests.
3. Existing Round3/Round4 timing-boundary parity improvements in scaling optimizer pre-work.

Do not use as primary gate path by default:

1. `TRACE_OPTIMIZER_MICROPHASES=1` as mandatory sampling mode.
2. `SCALING_ALIGN_SCHEDULER_INCREMENT=1` (insufficient gain + regression risk observed).

### 5) Forward Plan from Best Round Code State

Recommended baseline:

- Start from **Round6-8 best code state** (`a3158883`, code-equivalent `d572d935`), then cherry-pick only validated default-off diagnostics if needed.

#### Priority P0 — Optimizer residual (highest)

- **Hypothesis O1**: `optimizer_main_update` is still contaminated by profile-iteration replay/cache I/O and queue-boundary effects.
  - Change: add optional strict boundary experiment around scaling optimizer CMD timing (`training.py`), mirrored in distributed path for measurement symmetry.
  - Validation: fixed 3-run repeated pairing.
  - Expected impact: `optimizer_step -0.8% ~ -2.0%`, `optimizer_main_update -0.6% ~ -1.5%`.

- **Hypothesis O2**: microphase instrumentation perturbs hot path; primary gate should use microphase disabled.
  - Experiment: A/B with identical protocol except `TRACE_OPTIMIZER_MICROPHASES=0/1`.
  - Expected impact when gate uses `=0`: `optimizer_step -0.5% ~ -1.2%`.

#### Priority P1 — Backward residual

- **Hypothesis B1**: scaling grad replay fallback randomness introduces non-negligible drift.
  - Change: convert profile-window fallback in `_build_scaling_output_tensor_grad` to strict fail-fast when replay cache is missing.
  - Validation: replay cache integrity check before runs + 3-run compare.
  - Expected impact: `backward_step -1.0% ~ -3.0%` (if fallback path was active), plus reduced variance.

- **Hypothesis B2**: stage1 steady-state subtraction semantics remain biased.
  - Experiment: add stage-bucket diagnostic output (no gate change).
  - Expected impact: no direct metric drop, but better attribution and faster iteration convergence.

#### Priority P2 — Forward stability guard

- **Hypothesis F1**: replay activation H2D transfer semantics (`non_blocking`) affect warmup/steady consistency.
  - Change: add default-off A/B knob in `megatron/profiler/utils.py`.
  - Expected impact: `forward_step -0.3% ~ -1.0%` on stage1-heavy ranks.

### 6) Final Verdict

- Current latest code rerun (strict Round12 protocol) is **worse** than both Round4 and Round6-8 baselines on the 3-op gate.
- Overall best round remains **Round6-8** under the agreed decision rule.
- Next optimization should be anchored on best-round baseline + selective default-off diagnostics, with one-variable-at-a-time 3-run repeated validation.
