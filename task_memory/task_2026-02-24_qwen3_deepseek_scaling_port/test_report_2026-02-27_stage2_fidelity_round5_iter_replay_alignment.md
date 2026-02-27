## Test Report: Stage-2 Fidelity Round5 (Iteration-Indexed Replay Cache Alignment)

**Date**: 2026-02-27  
**Environment**: conda `myenv_yc` (Python 3.9.18)

### 1) Test Script Information

- Modified code:
  - `megatron/training/training.py`
  - `megatron/profiler/utils.py`
  - `tests/unit_tests/profiler/test_scaling_replay_cache_paths.py`
- Reproducible commands:
  ```bash
  # Unit tests (new replay-path selector + existing trace-arg parser checks)
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 PYTHONPATH=$(pwd) \
  pytest -q \
    tests/unit_tests/profiler/test_scaling_replay_cache_paths.py \
    tests/unit_tests/test_training.py::TestTraining::test_trace_cmd_sync_mode_default_global \
    tests/unit_tests/test_training.py::TestTraining::test_trace_cmd_sync_mode_event \
    tests/unit_tests/test_training.py::TestTraining::test_trace_cmd_sync_mode_invalid_value \
    tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
    tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
    tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

  # Scaling smoke probe for iter-indexed cache handoff (rank3 -> rank7)
  MODE=scaling MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=4 TRAIN_ITERS=6 \
  SCALING_MIN_WARMUP_ITERS=3 SCALING_PROFILE_ITERS=3 \
  SCALING_REPLAY_CACHE_TAG=itercacheprobe SCALING_FAKE_RANK_ORDER=3,7 \
  bash examples/pretrain_deepseek_v3_moe.sh

  # Full scaling pass (0..7) with iter-indexed cache
  MODE=scaling MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=4 TRAIN_ITERS=6 \
  SCALING_MIN_WARMUP_ITERS=3 SCALING_PROFILE_ITERS=3 \
  SCALING_REPLAY_CACHE_TAG=itercachefullA SCALING_FAKE_RANK_ORDER=0,1,2,3,4,5,6,7 \
  bash examples/pretrain_deepseek_v3_moe.sh

  # Fresh distributed baseline
  MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=4 TRAIN_ITERS=6 \
  TRACE_CMD_SYNC_MODE=global TRACE_SUBOP_SYNC_MODE=global \
  bash examples/pretrain_deepseek_v3_moe.sh

  # Compare (trace-based)
  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
    --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step \
    --pair-timestamp 20260227145502 \
    --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_itercacheA_pair45502_sub.log

  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
    --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step \
    --no-distributed-subtract-comm \
    --pair-timestamp 20260227145502 \
    --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_itercacheA_pair45502_nosub.log
  ```

### 2) Validation Criteria

- Functional criteria:
  - scaling replay cache must support iteration-indexed activation/grad handoff (`*_iter{current_iter}.pt`).
  - consumer rank should load iter-specific cache first; legacy path remains compatibility fallback.
- Correctness criteria:
  - new unit tests for replay-path resolution all PASS.
  - scaling smoke (`rank3,7`) can finish with iter-indexed cache files generated.
- Fidelity criteria (paper gate):
  - `forward_step/backward_step/optimizer_step` rank-median relative error <= 5%.

### 3) Test Results and Evidence

- Unit tests: **PASS** (`10 passed`)
  - includes new file: `tests/unit_tests/profiler/test_scaling_replay_cache_paths.py`.
- Iter-indexed cache generation: **PASS**
  - cache directory: `profiler_log/scaling_replay_cache/wd8_tp1_pp2_exp2_expNum16_numl8_bs1_sl256_hs1024_itercacheprobe`
  - generated files include:
    - `activation_to_rank7_iter1.pt ... activation_to_rank7_iter5.pt`
    - `grad_to_rank3_iter1.pt ... grad_to_rank3_iter5.pt`
- Full scaling + distributed rerun: **PASS** (process-level)
  - scaling trace latest ranks 0..7 timestamps: `20260227145212..20260227145409`
  - distributed trace timestamp (all ranks): `20260227145502`
- Fidelity compare outcome: **FAIL (threshold not met yet)**
  - subtract-comm (`...pair45502_sub.log`):
    - `forward_step` rank median `4.23%` (**PASS**)
    - `backward_step` rank median `14.18%` (**FAIL**)
    - `optimizer_step` rank median `7.57%` (**FAIL**)
  - no-subtract (`...pair45502_nosub.log`):
    - `forward_step` rank median `14.80%` (**FAIL**)
    - `backward_step` rank median `17.53%` (**FAIL**)
    - `optimizer_step` rank median `7.57%` (**FAIL**)

### 4) Failure Notes

- This round fixed a concrete replay-alignment defect (single-file cache collapsing multiple iterations into one replay sample), but the 5% fidelity gate is still not reached.
- Remaining dominant gaps stay in `backward_step` and `optimizer_step`, and run-to-run variance of distributed baseline remains large.
