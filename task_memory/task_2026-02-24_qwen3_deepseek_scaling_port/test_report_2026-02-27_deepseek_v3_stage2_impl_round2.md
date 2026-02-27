## Test Report: DeepSeek-V3 Stage-2 Implementation Round2

**Date**: 2026-02-27  
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)

### 1) Test Script Information

- **Code paths validated**:
  - `megatron/core/pipeline_parallel/p2p_communication.py`
  - `megatron/core/transformer/moe/moe_utils.py`
  - `megatron/core/transformer/moe/router.py`
  - `megatron/core/transformer/multi_latent_attention.py`
  - `tests/unit_tests/pipeline_parallel/test_p2p_dtype_alignment.py`
  - `tests/unit_tests/transformer/moe/test_routers.py`
  - `tests/unit_tests/transformer/test_multi_latent_attention.py`

- **Commands (reproducible)**:

```bash
# Stage-2 unit suite (round2)
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
  tests/unit_tests/transformer/moe/test_expert_bias_update.py \
  tests/unit_tests/pipeline_parallel/test_p2p_dtype_alignment.py

# Distributed smoke (target stage-2 architecture-standard profile)
MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3 \
  bash examples/pretrain_deepseek_v3_moe.sh

# Scaling smoke (target stage-2 architecture-standard profile)
MODE=scaling MODEL_PROFILE=smoke TRACE_START=1 TRAIN_ITERS=3 \
  bash examples/pretrain_deepseek_v3_moe.sh

# Compare (non-gating output generation)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp 20260227111506 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_pp2_ep2_after_fix.log
```

### 2) Validation Criteria

- **Smoke stability criteria**
  1. Target distributed smoke exits with code `0` and has no `AssertionError: ... found NaN in local forward loss calculation`.
  2. Target scaling smoke exits with code `0` and executes fake ranks `0..7`.
- **Trace completeness criteria**
  1. Distributed run config dir contains latest rank traces for `0..7`.
  2. Scaling run config dir contains latest rank traces for `0..7`.
- **Regression-safety criteria (for existing model paths)**
  1. p2p dtype alignment is constrained to `multi_latent_attention=True` only.
  2. Unit test proves non-MLA path remains no-op for p2p dtype alignment.
- **Numerical hardening criteria**
  1. Sigmoid router normalization remains finite on bf16 extreme logits.

### 3) Test Results and Evidence

| Test Item | Result | Evidence |
|---|---|---|
| Stage-2 unit suite (round2) | PASS | `25 passed` |
| Distributed smoke (`PP=2,EP=2,bf16`) | PASS | log: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_dist_smoke_iter3_after_fix.log` |
| Scaling smoke (`PP=2,EP=2`) | PASS | log: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_scaling_smoke_iter3_after_fix.log` |
| Distributed trace rank coverage | PASS | latest ranks `0..7` in `realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256` |
| Scaling trace rank coverage | PASS | latest ranks `0..7` in `profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256` |
| Compare report generation | PASS (script), threshold gate FAIL | `logs/deepseek_v3_stage2_compare_pp2_ep2_after_fix.log` generated; diff still >5% |

**Key output excerpts**

- Unit suite:
  - `25 passed, 3 warnings in 8.76s`
- Distributed smoke:
  - `done with setup ...`
  - `training ...`
  - `[after training is done] datetime: 2026-02-27 11:12:05`
- Scaling smoke:
  - `[Scaling Mode] fake_current_rank_id=0/8` ... `[Scaling Mode] fake_current_rank_id=7/8`

### 4) Failure Diagnosis and Resolution (required)

- **Observed failure before fix**
  - Command (distributed smoke, stage-2 target) failed with:
    - `AssertionError: Rank 4/5/6/7: found NaN in local forward loss calculation`
  - Additional debug evidence:
    - `RuntimeError: Detected non-finite transformer input (rank=4, pre_process=False, post_process=True, dtype=torch.bfloat16)`
    - `RuntimeError: Detected non-finite tensor in pipeline p2p recv_forward (rank=6, shape=(256, 1, 1024), dtype=torch.bfloat16, non_finite_count=551)`

- **Root cause (evidence-based)**
  - Non-finite activations were already present at pipeline receive boundary of last PP stage under MLA + bf16 path.
  - This is consistent with PP>1 failure + PP=1 success + fp32 success isolation outcomes.

- **Fixes applied**
  1. `p2p_communication.py`: align forward send tensor dtype to `pipeline_dtype` in `send_forward*` paths.
     - Guarded by `config.multi_latent_attention=True` to avoid changing existing non-MLA behavior.
  2. `moe_utils.py` + `router.py`: fp32-safe sigmoid normalization with denominator clamp to avoid bf16 `0/0` edge-case.

- **Post-fix outcome**
  - Distributed/scaling smoke both pass on target stage-2 profile.
  - Stage-2 blocker (NaN crash) removed.
  - Compare fidelity gap remains (non-gating item for this stage).

### 5) Residual Risk

- `compare_qwen_trace_comp.py` remains above 5% for target ops on this profile (see compare report path above).
- This does not block stage-2 “architecture-standard runnability + trace drop” gate, but still blocks paper-facing fidelity gate.
