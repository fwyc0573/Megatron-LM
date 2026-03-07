# Scaling Mode Rank-Skipping Optimization Analysis

## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-03-02 | Initial analysis for dense and MoE models   |
| 2026-03-02 | Added engine MoE simulating debug execution notes and linked verification artifacts |

---

## Executive Summary

| Model Type | DP Redundant? | TP Redundant? | EP Redundant? | Minimal Rank Set |
|------------|:---:|:---:|:---:|---|
| **Dense** | ✅ Yes | ✅ Yes | N/A | `PP` ranks (one per PP stage) |
| **MoE** | ✅ Yes | ✅ Yes | ⚠️ Partially | `PP × EP` ranks (all EP ranks per PP stage, one TP+DP representative) |

---

## Part 1: Dense Model Analysis

### 1.1 Agreement Assessment

**I fully agree** with the proposed dense model measurement optimization. For dense models in Scaling Mode, **measuring only one representative rank per PP stage** (specifically `tp_rank=0, dp_rank=0, pp_rank=0..PP-1`) is sufficient to capture all unique compute workloads.

### 1.2 DP Dimension Redundancy — ✅ CONFIRMED

**Conclusion**: All DP ranks within the same (PP stage, TP rank) group have **identical compute workloads** in Scaling Mode.

**Evidence**:

1. **Model partitioning is independent of DP rank**:
   - Layer assignment (`get_num_layers_to_build` in `megatron/core/transformer/transformer_block.py:30-66`) depends only on `pp_rank` and `fake_pp` — DP rank does not affect which layers are built.
   - TP sharding (`ColumnParallelLinear`, `RowParallelLinear` in `megatron/core/tensor_parallel/layers.py`) divides weights by `fake_tp` — DP rank has zero influence on weight shapes.

2. **Data does not affect compute timing**:
   - In Scaling Mode, `sim_get_batch()` (`megatron/profiler/utils.py:374-377`) feeds tokens to the model, but the tensor shapes are identical across all DP ranks: `[micro_batch_size, seq_length]`.
   - Since we're timing CUDA kernels, the actual data values don't affect compute duration — only tensor shapes and dtypes matter.
   - The `get_batch_on_this_tp_rank()` function generates random data for non-tp_rank=0 ranks (`megatron/profiler/utils.py:333-369`), further confirming that data content is irrelevant.

3. **DP only affects communication (allreduce), not compute**:
   - In `pretrain()` (`megatron/training/training.py`), the `dp_allreduce` CMD is created with `cmd.no_trace_update(0,0)` — it records metadata but doesn't actually execute the comm, so DP group size only affects metadata, not timing.

### 1.3 TP Dimension Redundancy — ✅ CONFIRMED

**Conclusion**: All TP ranks at the same PP stage have **identical compute workloads** (tensor shapes, FLOP count, kernel calls) in Scaling Mode.

**Evidence**:

1. **ColumnParallelLinear** (`megatron/core/tensor_parallel/layers.py:719-867`):
   - Weight shape: `[output_size / fake_tp, input_size]` — every TP rank gets the **same** partition size.
   - `self.output_size_per_partition = divide(output_size, world_size)` with `world_size = config.fake_tp` in scaling mode.
   - Forward: `output = input @ weight.T` — same shape matmul on every TP rank.

2. **RowParallelLinear** (`megatron/core/tensor_parallel/layers.py:988-1093`):
   - Weight shape: `[output_size, input_size / world_size]` — same partition size per TP rank.
   - Note: `RowParallelLinear.__init__` uses `get_tensor_model_parallel_world_size()` directly (returns 1 in scaling mode since real world_size=1), **but the TE wrapper overrides this** — TE layers use `config.tensor_model_parallel_size` which equals `fake_tp`.

3. **TELinear / TELayerNormColumnParallelLinear / TEDotProductAttention** (`megatron/core/transformer/custom_layers/transformer_engine.py`):
   - All three use `actual_tp_size = self.config.tensor_model_parallel_size` in scaling mode.
   - `tp_group=None` (no real comm group) — TE internally partitions by `tp_size` but each rank computes the **same** partition size.

4. **Attention** (`megatron/core/transformer/attention.py:56-113`):
   - `self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)` with `world_size = config.fake_tp`.
   - `self.num_query_groups_per_partition = divide(num_query_groups, world_size)`.
   - Every TP rank computes the same number of heads → same FlashAttention kernel shape.

5. **VocabParallelEmbedding** (`megatron/core/tensor_parallel/layers.py:167-228`):
   - `num_embeddings_per_partition = vocab_end_index - vocab_start_index`.
   - Each TP rank gets `vocab_size / fake_tp` embeddings — same compute per rank.
   - Note: the `vocab_start_index` and `vocab_end_index` differ per TP rank, meaning different rows of the embedding table are accessed, but the **kernel shape** and **compute amount** are identical.

6. **Communication in forward/backward** — In scaling mode:
   - All-reduce/reduce-scatter/all-gather ops are either intercepted (metadata only) or bypassed.
   - These comm ops don't affect actual compute kernel timing.

**Key insight**: TP partitioning in Megatron-LM is **uniform**. Every dimension partitioned by TP (`hidden_size`, `num_attention_heads`, `ffn_hidden_size`, `vocab_size`) is divided evenly among TP ranks. No TP rank gets a "larger" or "smaller" shard.

### 1.4 PP Dimension — ❌ NOT REDUNDANT (must measure all PP stages)

**Conclusion**: Different PP stages have **different compute workloads** and **must** be measured individually.

**Evidence**:

1. **Layer assignment** (`get_gpt_decoder_block_spec` in `megatron/core/models/gpt/gpt_layer_specs.py:190-208`):
   ```
   offset = pp_rank * num_layers_to_build
   local_layer_specs = layer_specs[offset : offset + num_layers_to_build]
   ```
   While each PP stage gets the **same number** of transformer layers, the first and last stages have additional components.

2. **First stage** (`is_pre_process=True`):
   - Runs embedding layer (VocabParallelEmbedding + position embedding).
   - Additional compute for `get_batch` data loading and broadcast.

3. **Last stage** (`is_post_process=True`):
   - Runs output layer (linear projection from hidden to vocab).
   - Runs `loss_func` (cross-entropy + DP allreduce for loss).
   - Additional backward through the output layer.

4. **However**, if `PP > 2`, intermediate stages (pp_rank=1..PP-2) have **identical compute** (same number of transformer layers, no embedding or output layer). This is a potential further optimization.

**Potential sub-optimization**: For dense models with PP > 2:
- Measure pp_rank=0 (first stage with embedding)
- Measure pp_rank=PP-1 (last stage with output layer)
- Measure pp_rank=1 once (representative for all intermediate stages 1..PP-2)
- Minimal set: **min(PP, 3)** ranks instead of PP ranks

### 1.5 Recommended Minimal Rank Set for Dense Models

| Scenario | Representative Ranks | Count |
|----------|---------------------|-------|
| **PP=1** | rank 0 only | **1** |
| **PP=2** | pp_rank ∈ {0, 1} | **2** |
| **PP>2** | pp_rank ∈ {0, 1, PP-1} | **3** |
| **General (conservative)** | pp_rank ∈ {0, 1, ..., PP-1}, tp_rank=0, dp_rank=0 | **PP** |

**Speedup**: From `fake_world_size = PP × TP × DP` ranks down to `PP` ranks (or even `min(PP, 3)` with the intermediate-stage optimization).

Example: For a 540B model with `PP=8, TP=4, DP=256` → `fake_world_size=8192`, measuring only **8 ranks** (or **3** with the sub-optimization) instead of 8192 — a **1024×** (or **2731×**) speedup.

---

## Part 2: MoE Model Analysis

### 2.1 EP Dimension Impact — ⚠️ NOT FULLY REDUNDANT

**Conclusion**: EP ranks are **NOT** duplicates like DP ranks. Different EP ranks hold different experts and process **different numbers of tokens** per expert, leading to **different compute workloads**.

**Evidence**:

1. **Different expert assignments per EP rank** (`BaseMoELayer.__init__` in `megatron/core/transformer/moe/moe_layer.py:26-105`):
   ```python
   self.num_local_experts = config.num_moe_experts // self.expert_parallel_size
   local_expert_indices_offset = exp_rank * self.num_local_experts
   self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
   ```
   - EP rank 0 gets experts `[0, 1, ..., num_local-1]`
   - EP rank 1 gets experts `[num_local, ..., 2*num_local-1]`
   - Each EP rank's `GroupedMLP` has the **same weight shapes** (`num_local_experts × hidden × ffn`), but processes **different numbers of tokens** per expert.

2. **Token routing variability** (`sim_routing` in `megatron/profiler/moe/sim_routing.py:45-73`):
   - The router uses `torch.manual_seed(args.seed + ep_rank)` — different EP ranks get different random hidden states.
   - `topk_routing_with_score_function` produces **token-to-expert assignments** that are data-dependent.
   - After all-to-all dispatching, each EP rank receives a **different number of total tokens** to process.

3. **GroupedMLP compute is token-count-dependent** (`GroupedMLP/forward` in `megatron/core/transformer/moe/experts.py:143-181`):
   ```python
   fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False)
   ```
   - The `gmm` (Grouped Matrix Multiply) kernel's compute time **varies with `tokens_per_expert`**.
   - If EP rank 0's experts receive 100 tokens total but EP rank 1's experts receive 150 tokens, the GEMM timings will differ.

4. **Dispatching simulation** (`sim_dispatching` in `megatron/profiler/moe/sim_dispatching.py:27-55`):
   - `num_local_tokens_per_expert` is computed per EP rank via `torch.histc`.
   - The `num_global_tokens_per_expert` gathered table shows how tokens are distributed globally — and different EP ranks end up with different local token counts.

### 2.2 Token Routing Variability Analysis

**Question**: Does token routing introduce compute differences within a (same PP stage, same EP rank) group across TP or DP?

**Answer**: **No, TP and DP within the same EP rank DO NOT create compute differences for MoE**.

**Evidence**:

1. **TP dimension within MoE**: The `GroupedMLP` weight is partitioned by TP (`fc1_output_size_per_partition = divide(fc1_output_size, tp_size)`), but the **number of tokens** processed is the same across all TP ranks — the token dispatch is EP-level, and TP sharding only splits the hidden dimension.

2. **DP dimension within MoE**: DP ranks process different data batches, but in scaling mode with `pre_fixed_routing_results`, the routing is pre-computed per EP rank. All DP ranks sharing the same EP rank use the **same routing results** and therefore the same `tokens_per_expert` distribution.

3. **Shared experts** (`SharedExpertMLP` in `moe_layer.py:150`): If enabled, the shared expert processes all tokens identically regardless of TP/DP/EP rank — same hidden_size, same token count.

### 2.3 Can We Skip Some EP Ranks?

**Short answer**: It depends on whether routing is balanced.

**Analysis**:

- With **perfect load balancing** (e.g., uniform token distribution across experts), all EP ranks would process approximately the same number of total tokens → **EP ranks would be near-identical** → could skip EP ranks.
- With **imbalanced routing** (typical in practice), different EP ranks process significantly different token counts → **cannot skip EP ranks** without losing accuracy.
- The current `sim_routing` uses random data with `torch.manual_seed(args.seed + ep_rank)` — this **does produce different routing patterns** per EP rank by design.

**However**, there is a subtle point: in practice, for the purpose of compute timing:
- All EP ranks have the **same weight shapes** (same `num_local_experts`, same `hidden_size`, same `ffn_hidden_size`).
- The only difference is `tokens_per_expert` — the token count distribution.
- If we can predict or measure the range of `tokens_per_expert` distributions, we could **parametrize** the MoE compute timing as a function of token count, instead of measuring every EP rank.

### 2.4 Dense Layers in MoE Models

MoE models typically interleave dense and MoE layers (controlled by `moe_layer_freq`). For **dense layers within a MoE model**, the same analysis from Part 1 applies:
- Dense layers have no EP dependency.
- TP/DP are redundant.
- Only PP stage matters.

The MoE layers add EP as an additional dimension that matters.

### 2.5 Recommended Minimal Rank Set for MoE Models

| Component | Dimension | Must Measure? | Reason |
|-----------|-----------|:---:|--------|
| Dense layers | PP | ✅ | Different layer slices per PP stage |
| Dense layers | TP | ❌ | Uniform sharding, identical kernels |
| Dense layers | DP | ❌ | Same model shard, different data only |
| MoE layers | PP | ✅ | Different layer slices may have different MoE/dense patterns |
| MoE layers | TP | ❌ | MoE GMM partitioned uniformly by TP |
| MoE layers | EP | ✅ | Different expert sets, different token counts |
| MoE layers | DP | ❌ | Same routing results shared within EP rank |

**Conservative minimal rank set for MoE**:
```
For each pp_rank in 0..PP-1:
  For each exp_rank in 0..EP-1:
    Measure rank with (pp_rank, exp_rank, tp_rank=0, dp_rank=0)
```

**Count**: `PP × EP` ranks.

**Example**: Qwen3-30B-A3B with `PP=4, TP=2, EP=2, DP=1` → `fake_world_size=16`, measure only `PP × EP = 8` ranks — **2× speedup**.

**Example**: DeepSeek-V3-Proxy with `PP=4, TP=2, EP=4, DP=1` → `fake_world_size=32`, measure only `PP × EP = 16` ranks — **2× speedup**.

**Aggressive optimization for MoE** (with PP > 2):
- `min(PP, 3) × EP` ranks, applying the intermediate-PP-stage optimization from Part 1.

---

## Part 3: Edge Cases and Caveats

### 3.1 RowParallelLinear Inconsistency (Minor)

In `megatron/core/tensor_parallel/layers.py:1010`:
```python
world_size = get_tensor_model_parallel_world_size()  # returns 1 in scaling mode!
self.input_size_per_partition = divide(input_size, world_size)
```

This uses the **real** world size (1) instead of `fake_tp`. However, when using Transformer Engine (the default for scaling mode), `TELinear` wraps this with the correct `tp_size=config.tensor_model_parallel_size`. So this is a non-issue for TE-based models, but could be problematic if someone runs scaling mode with `--no-transformer-engine` (non-TE local spec).

### 3.2 Random Seed Sensitivity

- `sim_get_batch()` generates random data for non-tp_rank=0 ranks — different data values but same shapes.
- `_build_scaling_output_tensor_grad()` uses a rank-dependent seed for fake gradient generation.
- These affect **numerical values** (hence gradient magnitudes), but NOT **kernel shapes or timing**.
- **Caveat**: If gradient magnitudes differ wildly between ranks (e.g., gradient underflow/overflow), this could affect optimizer step timing on different ranks. In practice, this is negligible.

### 3.3 Embedding Group Specialization

The `ep_allreduce` in `pretrain()` (training.py) is only recorded for ranks in the embedding group (`is_rank_in_embedding_group`). This is a **communication** operation (metadata-only in scaling mode), not compute. It doesn't affect the compute timing comparison.

### 3.4 MoE `SequentialMLP` Not Supported

`MoELayer.__init__` raises `NotImplementedError` for `SequentialMLP` in scaling mode. This is a non-issue since all scaling mode MoE uses `GroupedMLP` (`--moe-grouped-gemm` is required).

### 3.5 Activation/Grad Replay Cache Dependencies

In scaling mode, PP stages depend on each other via the activation/grad replay cache:
- Stage i saves activation to `activation_to_rank{next_rank}.pt`
- Stage i+1 loads it as input tensor

**Impact on rank skipping**: If we skip measuring intermediate DP/TP duplicates, we must ensure the replay cache from the representative rank (tp_rank=0, dp_rank=0) is used consistently. The current implementation uses `fake_current_rank_id` in cache file names, so we'd need to either:
1. Map skipped ranks to their representative's cache files, OR
2. Only run the representative ranks and accept that only those traces are produced.

Option 2 is simpler and sufficient for profiling purposes.

### 3.6 MoE Layer Pattern and PP Stage Heterogeneity

For MoE models, different PP stages may have **different mixes of dense and MoE layers** depending on `moe_layer_freq`. For example, with `num_layers=24, PP=4, moe_layer_freq=2`:
- Stage 0: layers 0-5 (layers 0,2,4 = MoE; layers 1,3,5 = dense)
- Stage 1: layers 6-11 (same pattern)
- Stage 2: layers 12-17 (same pattern)
- Stage 3: layers 18-23 (same pattern)

If `moe_layer_freq` is a regular integer, all PP stages get the **same dense/MoE pattern** (assuming `num_layers / PP` is a multiple of `moe_layer_freq`). But with custom `moe_layer_freq` lists, stages could differ.

**Recommendation**: Always measure all PP stages for MoE models unless you can verify the layer pattern is uniform.

---

## Part 4: Implementation Recommendations

### 4.1 How to Map World Rank to (pp, tp, dp, ep)

The `ParallelGroupManager` + `RankManager` already provide this mapping. The key function chain:
```
ParallelGroupManager._initialize_groups() → sim_initialize_model_parallel()
→ RankManager._create_rank_zoos() → RankZoo instances
```

To select representative ranks, filter by:
```python
representative_ranks = [
    rank_id for rank_id, zoo in rank_instances.items()
    if zoo._get_tp_local_rank() == 0 and zoo._get_dp_local_rank() == 0
]
```

For dense models: this gives exactly `PP` ranks.
For MoE models: this gives `PP × EP` ranks.

### 4.2 Suggested Implementation Approach

1. Add a CLI flag: `--skip-redundant-ranks` (default: False for backward compat)
2. After building `rank_instances`, compute the representative set.
3. If `--skip-redundant-ranks`:
   - Only loop through representative ranks.
   - For trace output, annotate which ranks are "representative" vs "inferred".
4. For the timeline simulator:
   - Use the representative rank's trace for all ranks in the same equivalence class.
   - Comm metadata remains rank-specific (recorded from metadata, not compute).

### 4.3 Validation Strategy

Before deploying rank-skipping:
1. Run **full** scaling mode (all ranks) once for a target config.
2. Run **skipped** scaling mode (representative ranks only).
3. Compare compute timings between equivalent ranks to verify they match within noise tolerance (< 1%).
4. Report: representative rank timing vs. average of equivalent ranks.

---

## Summary Table

| Dimension | Dense Model | MoE Model | Reason |
|-----------|:-----------:|:---------:|--------|
| **PP** | Must measure per stage | Must measure per stage | Different layers, embedding/output at boundaries |
| **TP** | Skip (all equivalent) | Skip (all equivalent) | Uniform weight sharding, same kernel shapes |
| **DP** | Skip (all equivalent) | Skip (all equivalent) | Same model shard, different data irrelevant to timing |
| **EP** | N/A | Must measure per EP rank | Different expert sets, variable token counts |

| Model Type | Current Rank Count | Optimized Rank Count | Speedup |
|------------|:-:|:-:|:-:|
| Dense PP=8 TP=4 DP=256 | 8192 | 8 (or 3) | **1024× (or 2731×)** |
| MoE PP=4 TP=2 EP=2 DP=1 | 16 | 8 | **2×** |
| MoE PP=4 TP=2 EP=4 DP=1 | 32 | 16 | **2×** |
| MoE PP=4 TP=4 EP=2 DP=2 | 64 | 8 | **8×** |
