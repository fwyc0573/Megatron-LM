## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added trace sub-op sync mode (`global/event`) implementation and verification report |

## Test Report: Trace Sub-op Sync Mode (Qwen3 / DeepSeek)

**Date**: 2026-02-24  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Project Root**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Changed Files Covered

- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron/profiler/cmd.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron/training/arguments.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/performance/compare_qwen_trace_comp.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/unit_tests/profiler/test_cmd_subop_sync_mode.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/unit_tests/test_training.py`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/examples/pretrain_qwen3_30b_a3b_moe.sh`
- `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/examples/pretrain_deepseek_v3_proxy_moe.sh`

#### 1.2 Commands (Reproducible)

```bash
# Syntax validation
python -m py_compile \
  megatron/profiler/cmd.py \
  megatron/training/arguments.py \
  tests/performance/compare_qwen_trace_comp.py \
  tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
  tests/unit_tests/test_training.py

# Unit tests: sub-op sync mode behavior
pytest -q tests/unit_tests/profiler/test_cmd_subop_sync_mode.py

# Unit tests: argument parsing (default/event/invalid)
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

# Integration smoke: Qwen (event mode) [existing logs reused]
TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=128 MASTER_PORT=... \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh
TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=128 FAKE_WORLD_SIZE=8 MASTER_PORT=... \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh

# Integration smoke: DeepSeek scaling (event mode)
TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=128 FAKE_WORLD_SIZE=8 MASTER_PORT=6670 \
  bash examples/pretrain_deepseek_v3_proxy_moe.sh

# Integration smoke: DeepSeek distributed (event mode) - ws8 / ws4 attempts (blocked)
TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=128 MASTER_PORT=6690 \
  bash examples/pretrain_deepseek_v3_proxy_moe.sh

TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=128 GPUS_PER_NODE=4 PP=2 EP=1 MASTER_PORT=6700 \
  bash examples/pretrain_deepseek_v3_proxy_moe.sh

# Compare with timestamp pairing + repeat median summary (5 runs total)
python tests/performance/compare_qwen_trace_comp.py \
  --pair-timestamp 20260224162610 \
  --threshold-pct 5 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_run3.log

python tests/performance/compare_qwen_trace_comp.py \
  --pair-timestamp 20260224143304 \
  --threshold-pct 5 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_run4.log

python tests/performance/compare_qwen_trace_comp.py \
  --pair-timestamp 20260224143112 \
  --threshold-pct 5 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_repeat_v2.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_run5.log

# Compare argument fail-fast check
python tests/performance/compare_qwen_trace_comp.py \
  --pair-timestamp 2026BAD \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_invalid_timestamp.log
```

#### 1.3 Logs

- Unit:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/test_cmd_subop_sync_mode.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/test_training_trace_subop_sync_mode.log`
- Integration:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_distributed_smoke_syncmode_event.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_scaling_smoke_syncmode_event.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_scaling_smoke_syncmode_event.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_distributed_smoke_syncmode_event_ws8_fail_stderr.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_distributed_smoke_syncmode_event_ws4_fail_stderr.log`
- Compare:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_run3.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_run4.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_run5.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_invalid_timestamp.stdout.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_syncmode_repeat_v2.jsonl`

### 2) Validation Criteria

1. `--trace-subop-sync-mode` parser behavior is correct:
   - default = `global`
   - valid values accept `event`
   - invalid value fails fast.
2. `CMD` trace decorator and `async_end_trace` both follow the same sync policy.
3. Compare script supports:
   - timestamp-capped pairing (`--pair-timestamp`)
   - repeated-run median summary (`--repeat-report`)
   - detailed decomposition fields (`total/comm/comp/sub_op_count`).
   - invalid timestamp input fails fast with non-zero exit.
4. Smoke sanity:
   - Qwen event-mode distributed/scaling traces both runnable.
   - DeepSeek event-mode scaling runnable end-to-end.
5. Acceptance gate (multi-run median):
   - rank0/rank7 for `forward_step/backward_step/optimizer_step`, median diff <= 5%.

### 3) Test Results and Evidence

| Suite | Result | Evidence |
|------|--------|----------|
| `py_compile` syntax check | PASS | no error output, exit code 0 |
| Unit: `test_cmd_subop_sync_mode.py` | PASS | log shows `3 passed` |
| Unit: `test_training.py` trace sync args | PASS | log shows `3 passed, 3 warnings` |
| Integration: Qwen distributed (event) | PASS | log contains `[after training is done]` |
| Integration: Qwen scaling (event) | PASS | log contains `fake_current_rank_id=7/8` and rank7 optimizer finish |
| Integration: DeepSeek scaling (event) | PASS | log contains `fake_current_rank_id=7/8` and rank7 optimizer finish |
| Integration: DeepSeek distributed ws8 (event) | FAIL (env blocker) | CUDA OOM on GPU0 + `DEEPSEEK_WS8_EXIT=1` |
| Integration: DeepSeek distributed ws4 fallback (event) | FAIL (env blocker) | NCCL internal/socket recv error + `DEEPSEEK_WS4_EXIT=1` |
| Compare invalid timestamp argument | PASS | exits with code `2` and prints invalid timestamp error |
| Compare (5-run median gate) | FAIL | repeated median for forward/backward still >> 5% |

#### 3.1 Key Evidence Excerpts

- Unit pass evidence:
  - `test_cmd_subop_sync_mode.log`: `3 passed in 1.24s`
  - `test_training_trace_subop_sync_mode.log`: `3 passed, 3 warnings in 8.94s`
- Qwen event smoke evidence:
  - `qwen_distributed_smoke_syncmode_event.log`: `[after training is done] datetime: 2026-02-24 16:23:46`
  - `qwen_scaling_smoke_syncmode_event.log`: `fake_current_rank_id=7/8`, `rank:7, finish optimizer.step profile ...`
- DeepSeek scaling event smoke evidence:
  - `deepseek_scaling_smoke_syncmode_event.log`: `fake_current_rank_id=7/8`, `rank:7, finish optimizer.step profile ...`
- DeepSeek distributed blockers:
  - `deepseek_distributed_smoke_syncmode_event_ws8_fail_stderr.log`:
    - `torch.cuda.OutOfMemoryError: CUDA out of memory ... GPU 0 ...`
    - `DEEPSEEK_WS8_EXIT=1`
  - `deepseek_distributed_smoke_syncmode_event_ws4_fail_stderr.log`:
    - `ncclInternalError: Internal check failed.`
    - `DEEPSEEK_WS4_EXIT=1`
- Compare median (5 runs):
  - `qwen_trace_compare_syncmode_run5.log` median summary:
    - rank0 `forward_step`: `45.20%` (FAIL)
    - rank0 `backward_step`: `40.13%` (FAIL)
    - rank7 `forward_step`: `31.42%` (FAIL)
    - rank7 `backward_step`: `30.25%` (FAIL)
    - optimizer medians remain PASS (`3.56%` / `2.51%`)
- Compare fail-fast evidence:
  - `qwen_trace_compare_invalid_timestamp.stdout.log`: `[ERROR] Invalid --pair-timestamp: 2026BAD`, `COMPARE_INVALID_TS_EXIT=2`

#### 3.2 Failure -> Fix -> Re-run Chain

1. **Failure A (unit test import cycle)**
   - Symptom: new profiler unit test initially failed during collection with circular import (`megatron.profiler`).
   - Fix: updated test loader to import `cmd.py` via `importlib.util.spec_from_file_location` and patched module-local `torch.cuda` symbols directly.
   - Re-run: `tests/unit_tests/profiler/test_cmd_subop_sync_mode.py` passed.

2. **Failure B (`test_training.py` env prerequisites)**
   - Symptom: `KeyError: LOCAL_RANK` during test module import.
   - Fix: rerun with explicit single-GPU distributed env vars (`LOCAL_RANK/RANK/WORLD_SIZE/...`).
   - Re-run: target parser tests passed.

3. **Failure C (DeepSeek distributed integration in shared environment)**
   - Symptom 1: ws8 run hit CUDA OOM due external GPU0 memory occupancy.
   - Symptom 2: ws4 fallback run hit NCCL internal/socket recv error.
   - Status: unresolved environment blocker in this window; code-level sync-mode changes unaffected in unit/Qwen/DeepSeek-scaling validation.

### 4) Conclusion

- Code changes for trace sync policy + compare robustness are implemented and validated at unit level.
- Event-mode smoke is validated for Qwen distributed/scaling and DeepSeek scaling.
- Acceptance gate (`median <= 5%` for rank0/rank7 forward/backward/optimizer) is **not met** yet:
  - event sync reduces timing intrusiveness but does not eliminate residual comp bias.
- Next phase should focus on workload/path fidelity alignment rather than additional sync-only tuning.
