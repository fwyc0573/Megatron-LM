## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added architecture comparison, gap analysis, stage-1 simplifications, and stage-2 backlog |

# Architecture Notes (Mixtral vs Qwen3 vs DeepSeek-V3-Proxy)

## 1) Mixtral vs Qwen3 vs DeepSeek-V3-Proxy

- **Mixtral (current baseline)**
  - All transformer layers are MoE in current usage.
  - Router path: Mixtral-style top-k + aux_loss (existing code path).
  - Standard MHA/GQA + RMSNorm + SwiGLU.
- **Qwen3-30B-A3B (stage-1 target)**
  - MoE model requiring explicit `moe_ffn_hidden_size` and `rotary_base` support.
  - Router can be kept in Mixtral-compatible top-k + aux_loss in stage-1.
  - All layers can be treated as MoE (`moe_layer_freq=1`) for stage-1.
- **DeepSeek-V3-Proxy (stage-1 target)**
  - Mixed dense/MoE layer schedule (proxy: 3 dense + 11 MoE).
  - Full upstream stack includes MLA, seq_aux_loss routing variants, shared experts, and optional MTP.
  - Stage-1 keeps MHA simplification and Mixtral-style router semantics for tracing correctness.

## 2) Missing modules / params in legacy codebase

### Missing or not yet ported modules (from latest reference)

- `multi_latent_attention.py` (MLA)
- `moe/shared_experts.py`
- `moe/router_replay.py`
- `moe/fused_a2a.py`
- `multi_token_prediction.py`
- DeepSeek-specific router variants (`seq_aux_loss`, score/group/bias semantics)
- `flex` dispatcher + `deepep` integration

### Missing first-class config/CLI support before stage-1 patch

- `--moe-layer-freq`
- `--moe-ffn-hidden-size`
- `--rotary-base`
- `TransformerConfig.moe_layer_freq`
- `TransformerConfig.moe_ffn_hidden_size`
- `TransformerConfig.rotary_base`

### Baseline layer-spec gap

- Old `gpt_layer_specs.py` only produced a single layer spec repeated by block.
- No native dense/MoE mixed pattern generation by layer index.

## 3) Stage-1 simplifications and impact

- **DeepSeek attention simplification**: MLA -> MHA
  - Impact: architecture fidelity reduced for DeepSeek-specific attention path.
  - Benefit: keeps tracing/scaling integration stable and runnable in current environment.
- **Router semantics simplification**: keep Mixtral-style top-k + aux_loss
  - Impact: no DeepSeek-specific router behavior (seq_aux_loss/group-topk/bias).
  - Benefit: distributed/scaling route-control path remains deterministic and debuggable.
- **Checkpoint path omitted**: random/mock initialization only
  - Impact: no direct training continuation from official checkpoints.
  - Benefit: avoids blocker dependencies and allows fast tracing verification.
- **Smoke profiles in scripts**
  - Impact: reduced model size defaults for local validation throughput.
  - Benefit: practical stage-1 verification with mock data and short iterations.

## 4) Stage-2 backlog

1. Add full DeepSeek MLA path (`multi_latent_attention` stack).
2. Add DeepSeek router semantics (`seq_aux_loss`, sigmoid/group-topk/expert-bias).
3. Add shared experts support.
4. Add optional MTP branch.
5. Add `flex`/`deepep` dispatcher integration.
6. Implement checkpoint conversion/loading (HF -> Megatron).

## Environment constraints identified

- `deepep` is not installed in current environment.
- `transformer-engine` / `triton` versions are older than latest DeepSeek advanced feature expectations.
