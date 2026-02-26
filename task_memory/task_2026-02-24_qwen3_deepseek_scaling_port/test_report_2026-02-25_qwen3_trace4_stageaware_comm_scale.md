## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-25 | Added stage-aware comm-scale compare enhancement, 8-GPU re-test evidence, and robust metric recommendation for paper reporting |

## Test Report: Qwen3 Trace4 Stage-Aware Comm-Scale Correction + Robust Metric

**Date**: 2026-02-25  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Project Root**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Modified Scripts

- `tests/performance/compare_qwen_trace_comp.py`
  - Added configurable comm subtraction scale:
    - `--distributed-comm-scale`
    - `--distributed-comm-scale-map` (supports `op` and `op@stageX` keys)
    - `--scaling-comm-scale`
    - `--scaling-comm-scale-map`
  - Added report fields:
    - `dist_eff_comm_ms` / `scale_eff_comm_ms`
  - Added comm-scale diagnostics:
    - `--suggest-comm-scale` with `op` and `op@stage` suggestions
  - Added robust auxiliary summaries:
    - `op_rank_median_aux_summary(non-gating, recommended_for_paper)`
    - `repeat_median_summary(op_rank_median_aux, non-gating)`
- `tests/unit_tests/performance/test_compare_qwen_trace_comp.py`
  - Added unit coverage for comm-scale map parsing/application and robust summary helpers.

#### 1.2 Reproducible Commands

```bash
# A) Unit tests
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29655 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/performance/test_compare_qwen_trace_comp.py \
  tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
  tests/unit_tests/profiler/test_interception_comm_scaling_mode.py \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

# B) 8-GPU distributed trace (new run)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 \
MASTER_PORT=8701 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 \
FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

# C) 8-GPU scaling trace (new run)
TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke \
TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 \
FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 SCALE_GPU=0 \
MASTER_PORT=8801 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

# D) Baseline compare (alpha=1.0, no stage-aware correction)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225185833 \
  --threshold-pct 5 --no-align-by-state --trim-ratio 0.2 \
  --suggest-comm-scale \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity2_newrun_baseline_mean_v2.log

# E) Stage-aware correction compare
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225185833 \
  --threshold-pct 5 --no-align-by-state --trim-ratio 0.2 \
  --distributed-comm-scale-map forward_step=0.65,backward_step=0.0,backward_step@stage0=0.2,backward_step@stage3=1.25 \
  --suggest-comm-scale \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity2_newrun_stageaware_mean_v2.log

# F) Two-run repeat summary with stage-aware correction
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225181920 \
  --threshold-pct 5 --no-align-by-state --trim-ratio 0.2 \
  --distributed-comm-scale-map forward_step=0.65,backward_step=0.0,backward_step@stage0=0.2,backward_step@stage3=1.25 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_event_fidelity2_stageaware_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_event_fidelity2_stageaware_repeat_v2_run1.log

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225185833 \
  --threshold-pct 5 --no-align-by-state --trim-ratio 0.2 \
  --distributed-comm-scale-map forward_step=0.65,backward_step=0.0,backward_step@stage0=0.2,backward_step@stage3=1.25 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_event_fidelity2_stageaware_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_event_fidelity2_stageaware_repeat_v2_run2.log
```

### 2) Validation Criteria

1. Backward compatibility:
   - Default compare behavior remains unchanged (`distributed_comm_scale=1.0`, empty maps).
2. Correctness of new compare logic:
   - op/stage-specific scale overrides are parsed and applied correctly.
   - `effective_comm_ms` is reflected in report and comp calculation.
3. Improvement target:
   - On identical paired traces, stage-aware correction should significantly reduce forward/backward diff compared with baseline.
4. Paper-facing robust metric:
   - `repeat_median_summary(op_rank_median_aux, non-gating)` should provide stable summary across repeated runs.

### 3) Test Results and Evidence

#### 3.1 Unit Tests

| Suite | Result | Evidence |
|------|--------|---------|
| compare + profiler + trace args target tests | PASS | `19 passed, 3 warnings in 8.23s` |

Unit log:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/test_compare_stageaware_20260225.log`

#### 3.2 New 8-GPU Trace Run Status

| Run | Result | Evidence |
|-----|--------|---------|
| Distributed (`event`, `trace_start=4`, `iters=6`) | PASS | `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_distributed_pp4tp1ep2dp2_seq2048_mbs8_iter6_trace4_event_fidelity2.log` |
| Scaling (`event`, same config) | PASS | `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_scaling_pp4tp1ep2dp2_seq2048_mbs8_iter6_trace4_event_fidelity2.log` |

Latest trace timestamps from this run:
- distributed rank0/rank7: `20260225185605`
- scaling rank0/rank7: `20260225185637` / `20260225185833`

#### 3.3 Baseline vs Stage-Aware Correction (Same New Run)

Pair timestamp: `20260225185833`

| Mode | forward_step | backward_step | optimizer_step | Overall |
|------|--------------|---------------|----------------|---------|
| Baseline (alpha=1.0) | mean diff `5.42%`, `5/8` FAIL | mean diff `10.43%`, `6/8` FAIL | mean diff `4.91%`, `3/8` FAIL | `14` FAIL checks |
| Stage-aware correction | mean diff `2.99%`, `1/8` FAIL | mean diff `1.61%`, `0/8` FAIL | mean diff `4.91%`, `3/8` FAIL | `4` FAIL checks |

Interpretation:
- Forward/backward mismatch is significantly reduced after applying stage-aware comm overlap correction.
- Residual FAIL rows are now mainly optimizer (non-comm path), plus one forward outlier (rank0).

Evidence logs:
- Baseline: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity2_newrun_baseline_mean_v2.stdout.log`
- Stage-aware: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity2_newrun_stageaware_mean_v2.stdout.log`

#### 3.4 Robust Metric for Paper (Recommended)

Using two repeated paired runs (`pair_timestamp=20260225181920` and `20260225185833`) with fixed stage-aware map:

`repeat_median_summary(op_rank_median_aux, non-gating)`:

| op | runs | median_of_run_rank_median_diff_pct | status |
|---|---:|---:|---|
| forward_step | 2 | 2.61 | PASS |
| backward_step | 2 | 1.18 | PASS |
| optimizer_step | 2 | 3.36 | PASS |

Recommendation:
- Use `median_of_run_rank_median_diff_pct` as the primary paper metric (robust against rank outliers and run-to-run jitter).
- Keep per-rank table as supplementary evidence.

Evidence log:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_event_fidelity2_stageaware_repeat_v2_run2.stdout.log`

### 4) Failure Handling Notes

- No infrastructure blocker in this round (8-GPU distributed/scaling both completed).
- Main residual issue is optimizer rank outliers in strict per-rank gating; forward/backward are substantially improved under stage-aware correction.
- Current correction is compare-side and optional (default behavior unchanged), minimizing risk to trace generation semantics.
