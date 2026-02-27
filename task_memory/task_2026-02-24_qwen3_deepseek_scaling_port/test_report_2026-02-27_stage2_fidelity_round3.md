## Test Report: Stage-2 Fidelity Alignment Round3

**Date**: 2026-02-27  
**Environment**: `conda activate myenv_yc` (Python 3.9), CUDA GPUs (8x A100 80GB), repo `Megatron-LM`

### Test Script Information

- Code changes under test:
  - `megatron/training/training.py`
  - `examples/pretrain_deepseek_v3_moe.sh`
- Validation commands:

```bash
# Unit tests (arg parser sync-mode coverage)
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 PYTHONPATH=$(pwd) \
pytest tests/unit_tests/test_training.py::TestTraining::test_trace_cmd_sync_mode_default_global \
       tests/unit_tests/test_training.py::TestTraining::test_trace_cmd_sync_mode_event \
       tests/unit_tests/test_training.py::TestTraining::test_trace_cmd_sync_mode_invalid_value \
       tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
       tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
       tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value -q

# Scaling two-pass replay with fixed fake-rank order
MODE=scaling MODEL_PROFILE=smoke SCALE_GPU=4 TRACE_START=4 TRAIN_ITERS=6 \
SCALING_MIN_WARMUP_ITERS=3 SCALING_PROFILE_ITERS=3 \
SCALING_REPLAY_CACHE_TAG=stage2_fidelityfix3 SCALING_FAKE_RANK_ORDER=0,1,2,3,4,5,6,7 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
bash examples/pretrain_deepseek_v3_moe.sh \
> task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_smoke_trace4_iter6_profile3_fidelityfix3_pass1.log 2>&1

MODE=scaling MODEL_PROFILE=smoke SCALE_GPU=4 TRACE_START=4 TRAIN_ITERS=6 \
SCALING_MIN_WARMUP_ITERS=3 SCALING_PROFILE_ITERS=3 \
SCALING_REPLAY_CACHE_TAG=stage2_fidelityfix3 SCALING_FAKE_RANK_ORDER=0,1,2,3,4,5,6,7 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
bash examples/pretrain_deepseek_v3_moe.sh \
> task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_smoke_trace4_iter6_profile3_fidelityfix3_pass2.log 2>&1

# Distributed paired run
MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
bash examples/pretrain_deepseek_v3_moe.sh \
> task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_smoke_trace4_iter6_profile3_fidelityfix3.log 2>&1

# Compare (primary)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp 20260227141611 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_fidelityfix3_20260227141611_sub.log

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp 20260227141950 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_fidelityfix3_20260227141950_sub.log
```

### Validation Criteria

- Parser and sync-mode argument behavior remains correct (no regression): unit tests pass.
- Stage-2 target script runs complete in both scaling and distributed modes with trace output files for ranks `0..7`.
- Fidelity metric target check (non-gating in this stage): compare reports for `forward_step`, `backward_step`, `optimizer_step` with threshold `5%`.

### Test Results

| Test Suite | Result | Details |
|------------|--------|---------|
| Unit tests (`test_training.py` sync-mode subset) | PASS | `6 passed` |
| Scaling pass1 (`fidelityfix3`) | PASS | exit code `0`; rank `0..7` all executed |
| Scaling pass2 (`fidelityfix3`) | PASS | exit code `0`; rank `0..7` all executed |
| Distributed run (`fidelityfix3`) | PASS | exit code `0`; rank `0..7` traces written |
| Compare @ `pair=20260227141611` | FAIL (threshold gate) | forward `3.83%` PASS; backward `11.09%` FAIL; optimizer `7.84%` FAIL |
| Compare @ `pair=20260227141950` | FAIL (threshold gate) | forward `7.58%` FAIL; backward `9.86%` FAIL; optimizer `10.54%` FAIL |

### Evidence

- Unit output excerpt:
  - `6 passed, 3 warnings in 8.61s`
- Scaling logs:
  - `.../deepseek_v3_stage2_scaling_smoke_trace4_iter6_profile3_fidelityfix3_pass1.log`
  - `.../deepseek_v3_stage2_scaling_smoke_trace4_iter6_profile3_fidelityfix3_pass2.log`
  - Both contain fake-rank sequence `0..7` (`Processing Fake Rank X` entries present for all ranks).
- Distributed log:
  - `.../deepseek_v3_stage2_dist_smoke_trace4_iter6_profile3_fidelityfix3.log` (exit code `0`).
- Compare reports:
  - `.../deepseek_v3_stage2_compare_fidelityfix3_20260227141611_sub.log`
  - `.../deepseek_v3_stage2_compare_fidelityfix3_20260227141950_sub.log`

### Failure Diagnosis and Resolution Status

- Current round confirms the dominant residual is no longer a crash/runability blocker.
- Remaining fidelity failures are concentrated in:
  - backward compute decomposition instability (comm subtraction sensitivity is high);
  - optimizer systematic scaling-side positive bias (~`+8%` to `+12%` median in latest stable pairs).
- No additional semantic behavior change was applied in this round beyond timing-boundary alignment and script default sync rollback.
