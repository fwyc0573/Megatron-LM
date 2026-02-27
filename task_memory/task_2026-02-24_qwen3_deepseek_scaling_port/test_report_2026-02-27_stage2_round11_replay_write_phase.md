## Test Report: Stage-2 Round11 Replay-Write-Phase / Scheduler-Increment Experiments

**Date**: 2026-02-27  
**Environment**: conda env `myenv_yc` (Python 3.9), CUDA GPUs: NVIDIA A800-SXM4-80GB

### 1) Test Script Information

- Modified code:
  - `megatron/training/training.py`
  - `megatron/training/arguments.py`
  - `examples/pretrain_deepseek_v3_moe.sh`
  - `tests/unit_tests/test_training_optimizer_microphase.py`
- Reproducible commands:
  ```bash
  pytest -q tests/unit_tests/test_training_optimizer_microphase.py
  python -m py_compile megatron/training/training.py megatron/training/arguments.py tests/unit_tests/test_training_optimizer_microphase.py
  bash -n examples/pretrain_deepseek_v3_moe.sh

  # Distributed baseline
  MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=4 TRAIN_ITERS=6 \
  TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
  MASTER_PORT=9660 bash examples/pretrain_deepseek_v3_moe.sh

  # Scaling run A: post-optimizer replay write
  MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global \
  TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
  SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 SCALING_REPLAY_WRITE_PHASE=post_optimizer \
  SCALING_REPLAY_CACHE_TAG=stage2_replayphase_postopt_run1 SCALE_GPU=1 MASTER_PORT=9650 \
  bash examples/pretrain_deepseek_v3_moe.sh

  # Scaling run B: pre-optimizer replay write (control)
  MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global \
  TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
  SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 SCALING_REPLAY_WRITE_PHASE=pre_optimizer \
  SCALING_REPLAY_CACHE_TAG=stage2_replayphase_preopt_run1 SCALE_GPU=1 MASTER_PORT=9670 \
  bash examples/pretrain_deepseek_v3_moe.sh

  # Scaling run C: post-optimizer write + scheduler increment alignment
  MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=global \
  TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
  SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 SCALING_REPLAY_WRITE_PHASE=post_optimizer \
  SCALING_ALIGN_SCHEDULER_INCREMENT=1 \
  SCALING_REPLAY_CACHE_TAG=stage2_replayphase_postopt_aligninc_run1 SCALE_GPU=1 MASTER_PORT=9680 \
  bash examples/pretrain_deepseek_v3_moe.sh

  # Compare reports
  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
    --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step,optimizer_main_update,optimizer_state_update,optimizer_post_update \
    --threshold-pct 5 \
    --pair-timestamp <cap> \
    --report-path <report.log> \
    --repeat-report <repeat.jsonl>
  ```

### 2) Validation Criteria

- Unit/syntax checks:
  - Modified parser/training helper paths must pass unit tests.
  - Python compile + shell syntax checks must pass.
- Fidelity checks (trace compare):
  - Primary observation metrics:
    - `forward_step`
    - `backward_step`
    - `optimizer_step`
    - `optimizer_main_update`
  - Compare different scaling semantic knobs on the same distributed baseline.

### 3) Test Results and Evidence

| Test Item | Result | Evidence |
|---|---|---|
| Unit tests (`test_training_optimizer_microphase.py`) | PASS | `12 passed` |
| Python syntax check | PASS | `py_compile` exit code `0` |
| Script syntax check | PASS | `bash -n` exit code `0` |
| Distributed baseline run | PASS | `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_replayphasepost_baseline.log` |
| Scaling run A (post write) | PASS | `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_replayphasepost_run1.log` |
| Scaling run B (pre write) | PASS | `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_replayphasepre_run1.log` |
| Scaling run C (post + align increment) | PASS | `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_replayphasepost_aligninc_run1.log` |

#### Key Compare Summaries (op-rank-median)

- Post write (`...replayphasepost_run1.log`):
  - `forward_step=10.74%`
  - `backward_step=12.71%`
  - `optimizer_step=6.56%`
  - `optimizer_main_update=6.21%`

- Pre write control (`...replayphasepre_run1.log`):
  - `forward_step=14.09%`
  - `backward_step=19.52%`
  - `optimizer_step=6.47%`
  - `optimizer_main_update=6.04%`

- Post write + align increment (`...replayphasepost_aligninc_run1.log`):
  - `forward_step=8.10%`
  - `backward_step=10.92%`
  - `optimizer_step=7.16%`
  - `optimizer_main_update=6.50%`

### 4) Failure/Gap Analysis

- No execution failures in unit/integration commands.
- Fidelity gate (`<=5%`) is still not met on target ops in this round.
- `optimizer_main_update` remains around `~6%` (best in this round), indicating residual gap persists.

### 5) Conclusion

- The new scaling semantic knobs are implemented and validated as default-off options.
- `scaling_replay_write_phase=post_optimizer` improves forward/backward stability relative to pre-write control.
- The additional `scaling_align_scheduler_increment` switch did not improve optimizer gate in this round.
