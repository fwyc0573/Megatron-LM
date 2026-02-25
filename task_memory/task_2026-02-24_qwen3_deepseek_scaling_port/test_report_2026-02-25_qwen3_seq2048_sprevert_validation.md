## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-25 | Added Qwen3 seq2048 (GPU2-7) validation results after scaling/distributed parity probes, including TP3 fallback and latest compare evidence |

## Test Report: Qwen3 seq2048 scaling vs realistic (GPU2-7)

**Date**: 2026-02-25  
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)  
**Working Directory**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

- **Unit tests**:
  - `tests/unit_tests/transformer/test_transformer_config_scaling_mode.py`
  - `tests/unit_tests/profiler/test_cmd_subop_sync_mode.py`
  - `tests/unit_tests/profiler/test_interception_comm_scaling_mode.py`
  - `tests/unit_tests/performance/test_compare_qwen_trace_comp.py`
  - `tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global`
  - `tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event`
  - `tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value`
- **Integration scripts**:
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - `tests/performance/compare_qwen_trace_comp.py`

#### Reproducible Commands

```bash
# unit
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29632 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/transformer/test_transformer_config_scaling_mode.py \
  tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
  tests/unit_tests/profiler/test_interception_comm_scaling_mode.py \
  tests/unit_tests/performance/test_compare_qwen_trace_comp.py \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

# attempted target parallelism (expected fail-fast)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=6 \
TP=3 PP=1 EP=2 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=3 FAKE_DP=2 FAKE_EXP=2 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 TRAIN_ITERS=3 TRACE_START=1 TRACE_SUBOP_SYNC_MODE=event MASTER_PORT=6140 \
PYTHONPATH=$(pwd) bash examples/pretrain_qwen3_30b_a3b_moe.sh

# fallback parallelism (TP=2,DP=3,EP=1,PP=1), mbs=8
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=6 \
TP=2 PP=1 EP=1 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_DP=3 FAKE_EXP=1 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 TRAIN_ITERS=3 TRACE_START=1 TRACE_SUBOP_SYNC_MODE=event MASTER_PORT=6190 \
PYTHONPATH=$(pwd) bash examples/pretrain_qwen3_30b_a3b_moe.sh

CUDA_VISIBLE_DEVICES=2 MODE=scaling MODEL_PROFILE=smoke \
TP=2 PP=1 EP=1 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_DP=3 FAKE_EXP=1 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 TRAIN_ITERS=3 TRACE_START=1 TRACE_SUBOP_SYNC_MODE=event \
SCALE_GPU=2 MASTER_PORT=6290 PYTHONPATH=$(pwd) bash examples/pretrain_qwen3_30b_a3b_moe.sh

PYTHONPATH=$(pwd) python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225163400 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs8_event_sprevert.log

# fallback parallelism (TP=2,DP=3,EP=1,PP=1), mbs=4
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=6 \
TP=2 PP=1 EP=1 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_DP=3 FAKE_EXP=1 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=4 TRAIN_ITERS=3 TRACE_START=1 TRACE_SUBOP_SYNC_MODE=event MASTER_PORT=6200 \
PYTHONPATH=$(pwd) bash examples/pretrain_qwen3_30b_a3b_moe.sh

CUDA_VISIBLE_DEVICES=2 MODE=scaling MODEL_PROFILE=smoke \
TP=2 PP=1 EP=1 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_DP=3 FAKE_EXP=1 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=4 TRAIN_ITERS=3 TRACE_START=1 TRACE_SUBOP_SYNC_MODE=event \
SCALE_GPU=2 MASTER_PORT=6300 PYTHONPATH=$(pwd) bash examples/pretrain_qwen3_30b_a3b_moe.sh

PYTHONPATH=$(pwd) python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225163850 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs4_event_sprevert.log
```

### 2) Validation Criteria

- `trace` compare gate: rank `0-5`, op `{forward_step, backward_step, optimizer_step}`
- comparison metric: `diff_pct = |scale_comp_ms - dist_comp_ms| / dist_comp_ms`
- acceptance threshold: `diff_pct <= 5%`
- comp definition:
  - distributed: `comp = total - comm`
  - scaling: `comp = total` (metadata-only comm)

### 3) Test Results and Evidence

#### 3.1 Unit tests

- **PASS**: `16 passed`
- Evidence: terminal output with `16 passed, 3 warnings`

#### 3.2 Parallel strategy gating

- **TP=3,DP=2,EP=2,PP=1**: **FAIL (expected fail-fast)**
  - Error: `num_attention_heads (16) must be a multiple of tensor_model_parallel_size (3)`
  - Log: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_attempt_tp3_dp2_ep2_pp1_seq2048_mbs8_spfix.log`

- **Fallback TP=2,DP=3,EP=1,PP=1**: distributed/scaling both can run to completion.

#### 3.3 Compare outcomes (latest stable reruns)

- mbs=8 report: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs8_event_sprevert.log`
  - `forward_step`: mean diff `63.49%`, FAIL `6/6`
  - `backward_step`: mean diff `42.80%`, FAIL `6/6`
  - `optimizer_step`: mean diff `14.79%`, FAIL `6/6`

- mbs=4 report: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs4_event_sprevert.log`
  - `forward_step`: mean diff `79.23%`, FAIL `6/6`
  - `backward_step`: mean diff `63.18%`, FAIL `6/6`
  - `optimizer_step`: mean diff `14.66%`, FAIL `4/6`

#### 3.4 Additional diagnostics

- Trace entry count mismatch (rank0): distributed has 3 entries/op vs scaling 1 entry/op.
  - Evidence: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_seq2048_trace_entry_count_sprevert.log`
- Distributed per-iteration variance is high (same run, same rank, large comp spread).
  - Example extraction was performed from latest rank0 traces and showed strong outliers.

### 4) Failure Analysis and Resolution Status

- Status: **NOT RESOLVED** (still above 5% gate).
- Confirmed factors in this round:
  1. scaling mode currently profiles a single final step, while distributed trace files contain multiple profiled steps, causing sample mismatch;
  2. distributed per-step comp variance is large, and a few outliers dominate averages;
  3. timestamp-only pairing is fragile when scaling rank-by-rank timestamps span long windows.

- Code-level safety fixes added in this round are validated by unit tests, but they do not by themselves close the comp gap.
