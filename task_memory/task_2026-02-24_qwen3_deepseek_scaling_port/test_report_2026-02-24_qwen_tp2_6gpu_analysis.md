## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added 6-GPU TP2/DP3/EP1/PP1 qwen-moe scaling-vs-distributed consistency analysis, fixes, and revalidation |

## Test Report: Qwen-MoE 6-GPU (TP2/DP3/EP1/PP1) Scaling vs Distributed Consistency

**Date**: 2026-02-24  
**Environment**: `/opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Project Root**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Modified Files in This Round

- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron/core/transformer/moe/router.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/performance/compare_qwen_trace_comp.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/examples/pretrain_qwen3_30b_a3b_moe.sh`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/examples/pretrain_deepseek_v3_proxy_moe.sh`

#### 1.2 Reproducible Commands

```bash
# Syntax sanity
python -m py_compile \
  megatron/core/transformer/moe/router.py \
  megatron/core/transformer/moe/token_dispatcher.py \
  megatron/core/pipeline_parallel/schedules.py \
  tests/performance/compare_qwen_trace_comp.py

# Unit sanity for existing trace sync behaviors
pytest -q tests/unit_tests/profiler/test_cmd_subop_sync_mode.py
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29630 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 PYTHONPATH=$(pwd) \
pytest -q tests/unit_tests/transformer/moe/test_routers.py::TestTop2Router::test_aux_loss

# 6-GPU distributed (GPU 2-7), qwen smoke, TP2/DP3/EP1/PP1, event mode
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
  TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=256 MASTER_PORT=7002 GPUS_PER_NODE=6 PP=1 TP=2 EP=1 \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh

# scaling counterpart (single GPU loop fake ranks 0..5)
TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 \
  SEQ_LEN=256 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 SCALE_GPU=2 MASTER_PORT=7101 \
  GPUS_PER_NODE=6 PP=1 TP=2 EP=1 bash examples/pretrain_qwen3_30b_a3b_moe.sh

# compare (rank0/rank5, op-level, no mg_state alignment)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl256 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl256 \
  --ranks 0,5 \
  --pair-timestamp 20260224174724 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_run2.log

# global sync mode control pair
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=global MODE=distributed MODEL_PROFILE=smoke \
  TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=256 MASTER_PORT=7003 GPUS_PER_NODE=6 PP=1 TP=2 EP=1 \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh
TRACE_SUBOP_SYNC_MODE=global MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 \
  SEQ_LEN=256 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 SCALE_GPU=2 MASTER_PORT=7102 \
  GPUS_PER_NODE=6 PP=1 TP=2 EP=1 bash examples/pretrain_qwen3_30b_a3b_moe.sh
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl256 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl256 \
  --ranks 0,5 \
  --pair-timestamp 20260224175252 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_global_run1.log

# higher-load run (SEQ_LEN=1024)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
  TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=1024 MASTER_PORT=7004 GPUS_PER_NODE=6 PP=1 TP=2 EP=1 \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh
TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 \
  SEQ_LEN=1024 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 SCALE_GPU=2 MASTER_PORT=7103 \
  GPUS_PER_NODE=6 PP=1 TP=2 EP=1 bash examples/pretrain_qwen3_30b_a3b_moe.sh
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl1024 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl1024 \
  --ranks 0,5 \
  --pair-timestamp 20260224175619 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq1024_run1.log
```

#### 1.3 Logs / Evidence Files

- Distributed event success: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_distributed_smoke_tp2dp3ep1pp1_seq256_event_run2b.log`
- Scaling event success: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_scaling_smoke_tp2dp3ep1pp1_seq256_event_run2.log`
- Event compare: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_run2.log`
- Global compare: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_global_run1.log`
- SEQ1024 compare: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq1024_run1.log`
- Consistency breakdown (sub-op categories): `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_tp2_6gpu_event_consistency_analysis.log`
- Higher-load summary: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_tp2_6gpu_seq1024_event_analysis.log`

### 2) Validation Criteria

1. Use 6 GPUs (`2-7`) for distributed tests.
2. Confirm forward/backward `duration` includes comm contributions, and `comp_ms = total_ms - comm_ms` is computed consistently.
3. Ensure stage/op measurement coverage is aligned (no missing key ops in one mode only for target compare set).
4. Assess measurement overhead sensitivity (`event` vs `global`) and whether mode-specific overhead bias is significant.
5. Increase workload (`SEQ_LEN=1024`) and check whether comp diff shrinks.

### 3) Results

#### 3.1 Parallel Strategy Execution

- Requested primary strategy (`TP=3, DP=2, EP=2, PP=1`) is blocked by model divisibility constraints in smoke profile (`num_attention_heads=16` incompatible with `TP=3`), not by OOM.
- Effective fallback used for this round: `TP=2, DP=3, EP=1, PP=1` on GPUs `2-7`.

#### 3.2 Forward/Backward Duration Contains Comm (Confirmed)

From event pair (`rank0/rank5`, seq256):

- `forward_step`: `dist total=131.84ms, comm=19.01ms, comp=112.83ms` (rank0)
- `backward_step`: `dist total=86.69ms, comm=9.99ms, comp=76.70ms` (rank0)

=> `total` clearly includes sub-op comm time, and subtraction is applied.

#### 3.3 Stage/Op Consistency Findings

- Found and fixed a real inconsistency in no-pipeline distributed path: `backward_step` was not traced before this round.
  - After fix, distributed op set becomes: `get_batch/forward_step/backward_step/dp_allreduce/optimizer_step`.
- Remaining structural mismatch persists:
  - Scaling has `loss_func` op while distributed does not emit standalone `loss_func` in this path.
  - More importantly, sub-op composition differs:
    - Distributed forward/backward lack per-layer `allreduce` entries that appear in scaling (`allreduce x12`), indicating comm attribution mismatch across modes.

#### 3.4 Measurement Overhead / Bias

- Event mode compare (`seq256`, no state alignment):
  - rank0: `forward 57.96%`, `backward 48.16%`, `optimizer 4.80%`
  - rank5: `forward 63.77%`, `backward 52.77%`, `optimizer 16.35%`
- Global mode compare (`seq256`, same config):
  - rank0: `forward 31.43%`, `backward 47.10%`, `optimizer 4.53%`
  - rank5: `forward 47.26%`, `backward 61.52%`, `optimizer 12.96%`

结论：`event` 不能单独消除偏差；`global/event` 切换对结果有影响，但在当前路径差异下不是决定性因素。

#### 3.5 Higher Load (SEQ_LEN=1024)

- `seq1024` compare still shows large comp gaps:
  - rank0: `forward 50.61%`, `backward 48.64%`, `optimizer 0.15%`
  - rank5: `forward 48.52%`, `backward 65.86%`, `optimizer 21.07%`

结论：仅提高 workload（这里通过更长序列）并未显著收敛 forward/backward comp gap。

### 4) Failure -> Fix -> Re-run Chain

1. **Scaling TP>1 blocked by router assertion**
   - Failure: `AssertionError: SP in scaling mode is not supported (tp should be 1)`
   - Fix: made router TP gating fake-rank aware and removed scaling-blocking assertions in token dispatcher TP comm path.
   - Re-run: scaling TP2 path passes.

2. **Scaling EP1 blocked by missing precomputed dispatch cache**
   - Failure: `AttributeError: TransformerConfig has no attribute per_rank_dispatching_results`
   - Fix: in dispatcher preprocess, when scaling EP1 has no precomputed dispatch cache, use runtime token histogram path.
   - Re-run: scaling TP2/EP1 run passes.

3. **Distributed no-pipeline backward op missing from trace**
   - Failure symptom: distributed trace had no `backward_step` lines for PP1.
   - Fix: added optional CMD wrapper for backward in `forward_backward_no_pipelining`.
   - Re-run: distributed now emits backward trace entries correctly.

4. **Local transformer implementation attempt blocked**
   - Failure: `AssertionError: (RMSNorm) is not supported in FusedLayerNorm`
   - Impact: unable to use `TRANSFORMER_IMPL=local` as alternate path under current qwen config.

### 5) Final Conclusion for This Round

- 已完成 6 卡实验、comm/comp 口径核对、阶段一致性修复（backward tracing）、开销对比和增载复测。
- `comp_ms` 当前仍不能在 scaling vs distributed 间稳定对齐（forward/backward 远超 5%）。
- 主要残余根因不是单一 sync mode，而是**跨模式路径与comm归因不一致**（尤其是 sub-op composition / traced-comm coverage 不一致）。
