## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Recorded stage-1 implementation progress and checkpoints |
| 2026-02-24 | Completed stage-1 validation matrix and captured trace-alignment evidence |
| 2026-02-24 | Completed scaling NaN timing-impact assessment and restored router unit tests to green |

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

### In Progress

- None.

### Pending

- None in stage-1 scope.
