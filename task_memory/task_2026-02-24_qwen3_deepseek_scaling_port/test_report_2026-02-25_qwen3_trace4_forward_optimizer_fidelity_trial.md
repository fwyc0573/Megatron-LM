## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-25 | Added commit checkpoint, forward/optimizer decomposition, compare trimmed-mean extension, and 8-GPU forward fidelity trial results |

## Test Report: Qwen3 Trace4 Forward/Optimizer Decomposition + Fidelity Trial

**Date**: 2026-02-25  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Project Root**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Commit Checkpoint (Step 1)

- Commit completed before this round implementation:
  - `76911f4f` (`Stabilize scaling trace comparison and document 8-GPU analyses`)

#### 1.2 Code Changes in This Round

- Added compare auxiliary robust reporting (`trimmed-mean` + `median-of-runs` dual report):
  - `tests/performance/compare_qwen_trace_comp.py`
  - `tests/unit_tests/performance/test_compare_qwen_trace_comp.py`
- Added forward/optimizer decomposition tool:
  - `tests/performance/analyze_qwen_forward_optimizer_breakdown.py`
- Minimal forward fidelity boundary trial (single boundary change):
  - `megatron/profiler/utils.py`
  - change: scaling replay tensor H2D copy in `sim_forward_step` from `non_blocking=True` to `non_blocking=False`.

#### 1.3 Reproducible Commands

```bash
# A) Unit tests for compare enhancements
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29635 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/performance/test_compare_qwen_trace_comp.py \
  tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
  tests/unit_tests/profiler/test_interception_comm_scaling_mode.py \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

# B) Forward/optimizer decomposition (pre-fidelity baseline)
python tests/performance/analyze_qwen_forward_optimizer_breakdown.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --pair-timestamp 20260225173310 \
  --ranks 0,1,2,3,4,5,6,7 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_forward_optimizer_breakdown_pp4tp1_8gpu_trace4_event_prefidelity.log

# C) Compare dual-report validation (trimmed-mean + median-of-runs)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225172610 \
  --threshold-pct 5 \
  --no-align-by-state \
  --trim-ratio 0.2 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_trace4_event_trimmed_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_trace4_event_trimmed_repeat_v2_run1.log

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225173310 \
  --threshold-pct 5 \
  --no-align-by-state \
  --trim-ratio 0.2 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_trace4_event_trimmed_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_trace4_event_trimmed_repeat_v2_run2.log

# D) 8-GPU regression after minimal forward fidelity boundary change
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 MASTER_PORT=8501 GPUS_PER_NODE=8 \
PP=4 TP=1 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=6 TRACE_START=4 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
SCALE_GPU=0 MASTER_PORT=8601 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225181920 \
  --threshold-pct 5 \
  --no-align-by-state \
  --trim-ratio 0.2 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity1_mean.log

python tests/performance/analyze_qwen_forward_optimizer_breakdown.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --pair-timestamp 20260225181920 \
  --ranks 0,1,2,3,4,5,6,7 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_forward_optimizer_breakdown_pp4tp1_8gpu_trace4_event_postfidelity.log
```

### 2) Validation Criteria

1. `compare` 主口径保持不变（mean-based）并继续 gate `<=5%`。
2. `compare` 需新增辅助稳健性视图：
   - 单次运行 `trimmed_mean_aux_summary`
   - 多次运行 `repeat_median_summary`（主口径）+ `repeat_median_summary(trimmed_mean_aux)`
3. 8-GPU fidelity 试验仅允许最小改动（单个边界）。
4. forward/optimizer 分解需输出每 rank：
   - `dist_total/comm/comp`
   - `scale_total/comm/comp`
   - forward sub-op 分类耗时（`trace_src_func` / `comm_func`）。

### 3) Test Results and Evidence

#### 3.1 Unit Tests

| Suite | Result | Evidence |
|------|--------|---------|
| compare + trace-mode targeted tests | PASS | `15 passed, 3 warnings in 8.70s` |
| compare-only tests | PASS | `5 passed in 0.03s` |

Logs:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/test_compare_trimmed_and_trace_mode_20260225.log`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/test_compare_trimmed_only_20260225.log`

#### 3.2 Forward/Optimizer Decomposition (Step 2)

Baseline decomposition (`pair_timestamp=20260225173310`) shows:

- forward:
  - distributed forward has measurable comm sub-op duration (`all_to_all` + `allgather`)
  - scaling comm sub-ops are metadata-only (`0.0 ms`) by design
- optimizer:
  - no sub-op entries in either mode; compare is total/comp direct difference

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_forward_optimizer_breakdown_pp4tp1_8gpu_trace4_event_prefidelity.log`

#### 3.3 Compare Dual Report (Step 3)

Implementation result:

- Main report (unchanged gate): mean-based per-rank/op diff
- Added non-gating auxiliary sections:
  - `trimmed_mean_aux_summary(trim_ratio=...)`
  - `repeat_median_summary(trimmed_mean_aux, non-gating)`

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_trace4_event_trimmed_repeat_v2_run2.stdout.log`

#### 3.4 Minimal Forward Fidelity Boundary Trial (Step 4)

Change applied:

- In scaling replay path, H2D copy for input activation changed to blocking (`non_blocking=False`), so replay-copy completion stays outside `forward_step` CMD timing window.

8-GPU regression result (`pair_timestamp=20260225181920`):

- `forward_step`: avg diff `7.96%` (7/8 FAIL)
- `backward_step`: avg diff `9.86%` (5/8 FAIL)
- `optimizer_step`: avg diff `5.24%` (4/8 FAIL)

Compared to pre-trial baseline (`pair_timestamp=20260225173310`):

- `forward_step`: `8.17% -> 7.96%` (small improvement, rank-level mixed)
- `backward_step`: `10.93% -> 9.86%` (small improvement)
- `optimizer_step`: `6.86% -> 5.24%` (improved average, but still unstable by rank)

Conclusion for trial:

- This single boundary tweak has limited net benefit and does not achieve <=5% gate.
- Residual mismatch remains systematic across middle pipeline ranks.

Evidence:
- compare: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity1_mean.stdout.log`
- decomposition (post-trial): `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_forward_optimizer_breakdown_pp4tp1_8gpu_trace4_event_postfidelity.log`
- repeat summary (baseline + trial): `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_event_fidelitytrial_repeat_run2.stdout.log`

### 4) Failure Handling Notes

- No blocking infra failure occurred in this round’s smoke reruns.
- Validation gate remains FAIL due residual comp mismatch; changes were kept minimal and evidence-complete as required.

