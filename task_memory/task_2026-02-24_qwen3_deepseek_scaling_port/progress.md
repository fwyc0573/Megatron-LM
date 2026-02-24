## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Recorded stage-1 implementation progress and checkpoints |
| 2026-02-24 | Completed stage-1 validation matrix and captured trace-alignment evidence |
| 2026-02-24 | Completed scaling NaN timing-impact assessment and restored router unit tests to green |
| 2026-02-24 | Added 32-rank scaling validation rerun and distributed-vs-scaling rank0/rank7 comp-timing investigation with fixes |

# Progress

## 2026-02-24

### Completed

- Added new CLI arguments:
  - `--moe-layer-freq`
  - `--moe-ffn-hidden-size`
  - `--rotary-base`
- Extended `TransformerConfig` with new fields and validations.
- Added dense/MoE mixed layer pattern generation in GPT layer specs.
- Wired `pretrain_llama.py` to:
  - use block spec for MoE models
  - pass `rotary_base` into `GPTModel`
  - build routing hidden state shape using correct TP dimension
- Updated expert MLP sizing:
  - expert path uses `moe_ffn_hidden_size`
  - dense path remains on `ffn_hidden_size`
- Added scripts:
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - `examples/pretrain_deepseek_v3_proxy_moe.sh`
- Added/updated unit tests for parser/config/layer-pattern/expert-MLP behavior.
- Fixed stage-1 regressions in MoE/unit-test paths:
  - `megatron/core/transformer/moe/moe_layer.py`: guard `pre_fixed_routing_results` with `getattr(...)` fallback.
  - `megatron/core/transformer/moe/token_dispatcher.py`: replaced undefined `moe_gather/moe_scatter` path with gather/scatter-add helpers.
  - `megatron/core/tensor_parallel/layers.py`: fallback when `training args` are not initialized in standalone unit tests.
- Ran integration smoke matrix (2 iters, mock data, trace enabled):
  1. Qwen3 distributed
  2. Qwen3 scaling (sequential fake rank 0..7)
  3. DeepSeek-V3-Proxy distributed
  4. DeepSeek-V3-Proxy scaling (sequential fake rank 0..7)
- Completed distributed vs scaling trace structure alignment check for both models.
- Completed scaling NaN timing-impact assessment:
  - routing indices/tokens-per-expert patterns remain stable per rank;
  - trace op structure remains aligned with distributed;
  - no additional code fix applied for NaN specifically in stage-1.
- Router-related targeted unit tests restored to green (`test_aux_loss` included).
- Ran Qwen3 scaling-mode 32-rank rerun (single-GPU sequential fake ranks `0..31`) and validated:
  - trace output completeness (32 files, rank coverage完整)
  - per-line trace format correctness
  - stage-specific op sequence correctness (stage0/1/2/3 pattern)
  - duration sanity (non-negative, no extreme outlier)
- Completed non-scaling 8-GPU vs scaling-mode (8 fake ranks) comp-timing comparison for rank0/rank7:
  - target compare scope: `forward_step`, `backward_step`, `optimizer_step` (plus `loss_func/get_batch` if present)
  - evidence log: `qwen_trace_rank0_rank7_compare_syncfix.log`
- Root-cause investigation and fixes applied:
  - Removed hot-path debug prints (`tolist()` + large tensor string formatting) in MoE forward/dispatcher to avoid trace-time sync perturbation.
  - Added fixed-routing numeric-stability guard (`nan_to_num`) in `moe_layer.py` to prevent router score NaN cascade in scaling/debug path.
  - Added auto idle-GPU selection for scaling scripts (`pretrain_qwen3_30b_a3b_moe.sh`, `pretrain_deepseek_v3_proxy_moe.sh`) to avoid contention bias from busy default GPU.
- Captured latest validation logs:
  - `qwen_scaling_32cards_smoke_idlegpu.log`
  - `qwen_scaling_32cards_validation_idlegpu.log`
  - `qwen_distributed_smoke_compare_idlegpu.log`
  - `qwen_scaling_smoke_compare_idlegpu.log`

### In Progress

- Residual rank0/rank7 comp timing gap analysis (>5% on部分操作) still open as stage-1 limitation characterization.

### Pending

- Decide whether to introduce stage-1.5 calibration for scaling-mode comp timing (simulation-dispatch overhead accounting).
