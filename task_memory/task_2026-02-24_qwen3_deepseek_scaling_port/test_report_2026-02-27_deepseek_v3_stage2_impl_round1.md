## Test Report: DeepSeek-V3 Stage-2 Implementation (Round 1)

**Date**: 2026-02-27  
**Environment**: `conda activate myenv_yc` (Python 3.9.18), `PYTHONPATH=$(pwd)`

### 1) Test Script Information

- **Code paths under test**
  - `megatron/training/arguments.py`
  - `megatron/core/transformer/transformer_config.py`
  - `megatron/core/transformer/moe/shared_experts.py`
  - `examples/pretrain_deepseek_v3_moe.sh`
  - stage-2 related tests under `tests/unit_tests/transformer/` and `tests/unit_tests/test_deepseek_v3_args.py`

- **Executed commands**
  ```bash
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 \
  PYTHONPATH=$(pwd) \
  pytest -q \
    tests/unit_tests/test_deepseek_v3_args.py \
    tests/unit_tests/transformer/test_deepseek_v3_config_validation.py \
    tests/unit_tests/transformer/test_yarn_rotary_embedding.py \
    tests/unit_tests/transformer/test_multi_latent_attention.py \
    tests/unit_tests/transformer/test_tenorm_dtype_cast.py \
    tests/unit_tests/transformer/moe/test_routers.py \
    tests/unit_tests/transformer/moe/test_shared_experts.py \
    tests/unit_tests/transformer/moe/test_expert_bias_update.py
  ```

  ```bash
  MODE=scaling MODEL_PROFILE=smoke TRACE_START=1 TRAIN_ITERS=3 \
  bash examples/pretrain_deepseek_v3_moe.sh \
    > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_smoke_iter3_gate.log 2>&1
  ```

  ```bash
  MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3 \
  bash examples/pretrain_deepseek_v3_moe.sh \
    > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_smoke_iter3_gate.log 2>&1
  ```

  ```bash
  MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3 PP=1 EP=1 \
  bash examples/pretrain_deepseek_v3_moe.sh \
    > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_smoke_pp1_ep1_iter3_gate.log 2>&1
  ```

  ```bash
  MODE=scaling MODEL_PROFILE=smoke TRACE_START=1 TRAIN_ITERS=3 PP=1 EP=1 \
  bash examples/pretrain_deepseek_v3_moe.sh \
    > task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_smoke_pp1_ep1_iter3_gate.log 2>&1
  ```

### 2) Validation Criteria

- Stage-2 related unit tests pass (new args/config/MLA/YaRN/router/shared experts/expert bias/TENorm path).
- Scaling smoke (`PP=2,TP=1,EP=2`) completes fake ranks `0..7` and writes trace files.
- Distributed smoke (`PP=2,TP=1,EP=2`) should run without NaN and write trace files for ranks `0..7`.
- No change introduced to legacy model scripts; new behavior scoped to stage-2 code paths and new script knobs.

### 3) Test Results and Evidence

| Suite | Result | Evidence |
|---|---|---|
| Stage-2 unit tests (8 files) | PASS | `19 passed` |
| Scaling smoke (`PP=2,EP=2`) | PASS | `profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256` with rank `0..7` |
| Distributed smoke (`PP=2,EP=2`) | FAIL | NaN in forward loss on ranks `7/6/5/4` |
| Distributed smoke diagnostic (`PP=1,EP=1`) | PASS | `realistic_trace/pp1_tp1_exp1_expn16_dp8_nl8_hs1024_sl256` rank `0..7` |
| Scaling smoke diagnostic (`PP=1,EP=1`) | PASS | `profiler_log/pp1_tp1_ep1_expn16_dp8_nl8_hs1024_sl256` rank `0..7` |

#### Key log excerpts

- Unit tests:
  - `19 passed, 3 warnings in 8.68s`
- Distributed blocker:
  - `AssertionError: Rank 7: found NaN in local forward loss calculation`
  - `AssertionError: Rank 6: found NaN in local forward loss calculation`
  - `AssertionError: Rank 5: found NaN in local forward loss calculation`
  - `AssertionError: Rank 4: found NaN in local forward loss calculation`
  - source log: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_smoke_iter3_gate.log`

### 4) Failure Handling (Root Cause Status)

- **Failure type**: distributed stage-2 smoke NaN under `PP=2,EP=2,bf16`.
- **Current status**: unresolved; tracked in `issues.md` (new item `#31`).
- **What was verified**
  - NaN persists even with `MOE_SHARED_EXPERT_GATE=0` (not a gate-only issue).
  - `PP=1,EP=1` passes in both modes, so blocker is tied to `PP>1` architecture-standard path in current environment.

### 5) Conclusion

- Stage-2 code implementation and unit-level verification are in place.
- Scaling-mode architecture-standard smoke is runnable with complete trace output.
- Distributed architecture-standard smoke remains blocked by NaN on pipeline last-stage ranks and requires focused runtime debugging before Gate A closure.
