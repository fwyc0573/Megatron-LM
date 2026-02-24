## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Created stage-1 implementation plan for Qwen3-MoE + DeepSeek-V3-Proxy scaling-mode port |

# Stage-1 Plan: Qwen3-MoE + DeepSeek-V3-Proxy Port

## Scope

- Port Qwen3-30B-A3B (stage-1 subset) and DeepSeek-V3-Proxy (MHA simplified) into current tracing/scaling Megatron-LM.
- Keep existing tracing/scaling framework behavior unchanged as first priority.
- Deliver dual-mode scripts:
  - Distributed tracing mode
  - Scaling mode (single GPU, sequential per-rank)

## Implementation Steps

1. Extend public CLI/config surface
   - Add `--moe-layer-freq`, `--moe-ffn-hidden-size`, `--rotary-base`
   - Thread fields into `TransformerConfig` and validation logic
2. Activate MoE layer pattern + expert FFN size
   - Support mixed dense/MoE decoder layer patterns by `moe_layer_freq`
   - Use `moe_ffn_hidden_size` on expert MLP paths only
3. Integrate model entrypoint changes
   - Reuse `pretrain_llama.py`
   - Select block spec generation for MoE models
   - Keep routing pre-generation active in distributed + scaling when EP>1
4. Add runnable scripts
   - `examples/pretrain_qwen3_30b_a3b_moe.sh`
   - `examples/pretrain_deepseek_v3_proxy_moe.sh`
5. Add tests and execute validation
   - Unit tests for parser/config/layer pattern/expert MLP
   - Integration smoke runs: 2 models × 2 modes
6. Document simplifications and known limitations

## Acceptance Criteria

- New args parse correctly and are visible in `TransformerConfig`.
- DeepSeek proxy layer pattern (`[0]*3 + [1]*11`) builds dense+MoE mixed block specs.
- Expert MLPs honor `moe_ffn_hidden_size`; dense MLP remains on `ffn_hidden_size`.
- Qwen3 and DeepSeek scripts can run in distributed and scaling modes with trace enabled.
- Stage-1 limitations are explicitly documented.
