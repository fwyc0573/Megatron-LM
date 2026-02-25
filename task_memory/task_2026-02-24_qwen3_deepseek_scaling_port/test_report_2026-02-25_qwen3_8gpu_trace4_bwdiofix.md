## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-25 | Added 8-GPU Qwen3 trace_start=4 retest (event/global), single-vs-avg robustness analysis, and backward I/O fix impact verification |

## Test Report: Qwen3 8-GPU Trace4 (Scaling vs Realistic, bwd I/O fix)

**Date**: 2026-02-25  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Project Root**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Code Scope (This Round)

- Trace timing sync policy and metadata-only comm path:
  - `megatron/profiler/cmd.py`
- Scaling backward profiling boundary (move grad-cache save out of backward CMD region):
  - `megatron/training/training.py`
- Compare utility (timestamp pairing + repeat median):
  - `tests/performance/compare_qwen_trace_comp.py`

#### 1.2 Reproducible Commands

```bash
# A) Unit tests (trace sync mode / compare / interception / scaling config)
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29634 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/transformer/test_transformer_config_scaling_mode.py \
  tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
  tests/unit_tests/profiler/test_interception_comm_scaling_mode.py \
  tests/unit_tests/performance/test_compare_qwen_trace_comp.py \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

# B) 8-GPU event-mode retest (Qwen3 smoke, seq=2048, mbs=8)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 MASTER_PORT=7601 GPUS_PER_NODE=8 \
PP=4 TP=1 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=6 TRACE_START=4 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
SCALE_GPU=0 MASTER_PORT=7701 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225173310 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_bwdiofix_rerun2_mean.log

# C) 8-GPU global-mode control run (same config)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=global MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 MASTER_PORT=7801 GPUS_PER_NODE=8 \
PP=4 TP=1 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

TRACE_SUBOP_SYNC_MODE=global MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=6 TRACE_START=4 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 \
SCALE_GPU=0 MASTER_PORT=7901 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn32_dp2_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn32_dp2_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225173820 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_global_bwdiofix_mean.log

# D) Robustness diagnostics (single-vs-avg, op/sub-op coverage, event-vs-global overhead)
# E) Model-size escalation check (full profile)
# E1) mbs=4 trial (expected OOM in this environment)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=full TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=2048 MICRO_BATCH_SIZE=4 MASTER_PORT=8101 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 bash examples/pretrain_qwen3_30b_a3b_moe.sh

# E2) mbs=1 fallback (full profile, event mode)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=full TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=1 MASTER_PORT=8301 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 bash examples/pretrain_qwen3_30b_a3b_moe.sh

TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=full TRAIN_ITERS=6 TRACE_START=4 SEQ_LEN=2048 MICRO_BATCH_SIZE=1 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 SCALE_GPU=0 MASTER_PORT=8401 GPUS_PER_NODE=8 PP=4 TP=1 EP=2 bash examples/pretrain_qwen3_30b_a3b_moe.sh

python tests/performance/compare_qwen_trace_comp.py   --distributed-dir realistic_trace/pp4_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048   --scaling-dir profiler_log/pp4_tp1_ep2_expn128_dp2_nl48_hs2048_sl2048   --ranks 0,1,2,3,4,5,6,7   --ops forward_step,backward_step,optimizer_step   --pair-timestamp 20260225175130   --threshold-pct 5   --no-align-by-state   --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs1_iter6_trace4_event_full_mean.log
# Outputs are in task_memory/.../logs/*.log listed below.
```

### 2) Validation Criteria

1. **Comp definition correctness**
   - Realistic run must use `comp_ms = total_ms - comm_ms`.
   - Scaling run keeps `comp_ms = total_ms` (comm is metadata-only, no real collective).
2. **Sampling policy clarity**
   - Verify scaling trace has one profiled record per `forward_step/backward_step/optimizer_step`.
   - Verify realistic trace has multiple records (3 in this config).
3. **Single-vs-avg comparability**
   - Check realistic in-file variance (CV) for each op.
   - Compare scaling single sample against realistic mean/median and evaluate sensitivity.
4. **Measurement overhead check**
   - Compare `event` vs `global` sync mode under identical run config.
5. **Acceptance gate**
   - Per-rank per-op diff threshold: `<= 5%`.

### 3) Test Results and Evidence

#### 3.1 Unit Tests

| Suite | Result | Evidence |
|------|--------|---------|
| trace/compare/scaling-path unit matrix | PASS | `16 passed, 3 warnings in 8.27s` |

- Log: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/test_suite_trace_compare_qwen3_rerun_20260225.log`

#### 3.2 Event-Mode 8-GPU Retest (pair timestamp `20260225173310`)

- Compare result: **FAIL (14 checks > 5%)**
- Forward avg diff: **8.17%**
- Backward avg diff: **10.93%**
- Optimizer avg diff: **6.86%**

Evidence:
- stdout/report
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_bwdiofix_rerun2_mean.stdout.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_bwdiofix_rerun2_mean.log`

#### 3.3 Backward I/O Fix Impact (Key Improvement)

Using same compare protocol before/after moving scaling `torch.save(grad...)` outside `backward_step` CMD region:

- Pre-fix (`trace4_event_mean`):
  - backward `rank2-7` mean diff: **125.56%**
- Post-fix (`trace4_event_bwdiofix_mean`):
  - backward `rank2-7` mean diff: **11.18%**

Conclusion: the backward timing region pollution by grad-cache I/O was a major root cause of extreme backward mismatch.

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_mean.stdout.log`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_bwdiofix_mean.stdout.log`

#### 3.4 Single-Op (Scaling) vs Avg/Median (Realistic) Assessment

- Scaling indeed has **1** sample/op; realistic has **3** samples/op in this setup.
- Realistic in-file comp variance is low after `TRACE_START=4`:
  - avg CV: `forward 0.86%`, `backward 1.01%`, `optimizer 4.81%`
- Scaling single sample compared to realistic mean vs median:
  - mean and median gaps are very close (for forward/backward nearly equivalent).

Interpretation:
- Under current stable window (`TRACE_START=4`), using realistic **avg** is acceptable and not materially different from median.
- For acceptance, repeated paired runs with median-of-runs remains necessary to suppress run-level outlier effects.

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_pp4tp1_8gpu_seq2048_mbs8_trace4_event_bwdiofix_rerun2_single_vs_avg_analysis.log`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_trace4_event_bwdiofix_repeat_run2.stdout.log`

#### 3.5 Op/Sub-op Consistency Check

- Compared ops (`forward_step`, `backward_step`) have aligned sub-op composition in this TP1 profile:
  - forward comm breakdown per record: `all_to_all x6`, `allgather x3`
  - backward comm breakdown per record: `all_to_all x6`
- Additional mismatch exists only in non-compared transport ops (e.g., `send_forward`, `recv_backward`, `recv_forward`, `send_backward`) because scaling loop does not trace real PP transport ops.

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_pp4tp1_8gpu_seq2048_mbs8_trace4_event_bwdiofix_rerun2_subop_coverage.log`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_pp4tp1_8gpu_seq2048_mbs8_trace4_event_bwdiofix_rerun2_op_coverage.log`

#### 3.6 Measurement Overhead (event vs global)

- Event/global mode changes measured comp by a small amount on average:
  - distributed comp delta (`global - event`):
    - forward `+2.02%`, backward `+0.20%`, optimizer `+0.65%`
  - scaling comp delta (`global - event`):
    - forward `-1.79%`, backward `-0.99%`, optimizer `-2.20%`

Interpretation:
- Sub-op sync mode affects measured values, but current residual mismatch is dominated by cross-mode workload/path differences rather than sync policy alone.

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_pp4tp1_8gpu_seq2048_mbs8_trace4_event_vs_global_overhead.log`
- Global compare report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_global_bwdiofix_mean.stdout.log`

#### 3.7 Model Size Escalation Check (Qwen3 full profile)

- Attempted larger model (`MODEL_PROFILE=full`, 48L/2048H/128 experts):
  1. `mbs=4`, `seq=2048`, distributed (`event`) -> **OOM** on GPU1.
  2. fallback `mbs=1`, `seq=2048`, distributed+scaling (`event`, `TRAIN_ITERS=6`, `TRACE_START=4`) -> both runs succeed.

- Full-profile compare result (`mbs=1`):
  - `forward_step` avg diff: **15.31%** (8/8 FAIL)
  - `backward_step` avg diff: **3.17%** (0/8 FAIL)
  - `optimizer_step` avg diff: **12.60%** (8/8 FAIL)

- Interpretation:
  - increasing model size (under memory-safe `mbs=1`) significantly improves backward alignment;
  - but does not improve forward/optimizer alignment in this setup, indicating residual mismatch is not purely “small model overhead ratio”.

Evidence:
- OOM log:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_distributed_pp4tp1ep2dp2_seq2048_mbs4_iter2_trace2_event_full_try.log`
- Full-profile successful compare:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs1_iter6_trace4_event_full_mean.stdout.log`
- Full-profile single-vs-avg analysis:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_pp4tp1_8gpu_seq2048_mbs1_trace4_event_full_single_vs_avg_analysis.log`

### 4) Failures and Resolution Chain

1. **Failure (historical)**: backward diff exploded (`>100%`) for ranks 2-7.
   - **Root cause**: scaling `backward_step` trace window included grad cache `torch.save(...)` I/O.
   - **Fix**: move grad save outside backward CMD scope.
   - **Result**: backward diff dropped to around `~11%` average for ranks 2-7.

2. **Current status**: still above `<=5%` target for many mid ranks.
   - **Diagnosis**: residual mismatch is systematic (not only outlier sensitivity, not only sync mode).

### 5) Conclusion (This Round)

- **I agree conditionally** with “scaling single-op vs realistic avg”:
  - In this stable setting (`TRACE_START=4`), realistic in-file variance is low; avg and median are effectively equivalent.
  - Therefore, single scaling sample vs realistic avg is a valid per-run comparator.
- **I do not agree** with relying on one run + mean only for acceptance:
  - run-to-run drift still exists; acceptance should use repeated paired runs and median-of-runs.
- After backward I/O fix, major backward inflation is resolved, but core comp alignment is still not within 5% for all ranks/ops.
