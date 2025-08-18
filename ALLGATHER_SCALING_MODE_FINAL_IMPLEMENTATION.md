# MoEAllGatherTokenDispatcher Scaling Mode Implementation - Final

This document describes the final implementation of scaling mode support for `MoEAllGatherTokenDispatcher` in Megatron-LM.

## Overview

The `MoEAllGatherTokenDispatcher` now supports scaling mode with proper communication metadata recording and pre-computed token distribution usage, following the pattern established by `MoEAlltoAllTokenDispatcher`.

## Key Implementation Principles

### 1. Communication Metadata Recording
- **Always call communication functions** to record metadata via CMD decorators
- **Override communication results** with pre-computed data in scaling mode
- **Maintain profiling accuracy** while ensuring correct token distribution

### 2. Pre-computed Data Usage
- Use `config.per_rank_dispatching_results` for token distribution
- Use `config.pre_fixed_routing_results` for routing indices and scores
- Ensure exact consistency between scaling and real modes

### 3. Tensor Size Simulation
- Simulate AllGather tensor expansion effects
- Simulate ReduceScatter tensor reduction effects
- Maintain correct tensor shapes for downstream operations

## Implementation Details

### Communication Function Pattern

Following the MoEAlltoAllTokenDispatcher pattern:

```python
# Always call communication function to record metadata
global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
    max_ind, func="gather_from_sequence_parallel_region_to_moe"
)

if self.config.is_scaling_mode:
    # Override with pre-computed data
    pre_computed_indices = self.config.pre_fixed_routing_results[self.config.exp_rank]['indices']
    ep_group_size = tp_size * exp_size
    
    if pre_computed_indices.dim() == 1:
        global_indices = pre_computed_indices.repeat(ep_group_size)
    else:
        repeat_dims = [ep_group_size] + [1] * (pre_computed_indices.dim() - 1)
        global_indices = pre_computed_indices.repeat(*repeat_dims)
    
    global_indices = global_indices.to(max_ind.device)
```

### Token Distribution

```python
if self.config.is_scaling_mode:
    # Use pre-computed token distribution for exact consistency
    num_local_tokens_per_expert = self.config.per_rank_dispatching_results[self.config.exp_rank]['num_local_tokens_per_expert']
    if not num_local_tokens_per_expert.is_cuda:
        num_local_tokens_per_expert = num_local_tokens_per_expert.cuda()
    
    # Extract tokens for local experts only
    tokens_per_expert = num_local_tokens_per_expert[self.local_expert_indices]
    tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
```

### ReduceScatter Simulation

```python
# Always call communication function to record metadata
output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
    unpermuted_global_hidden, func="reduce_scatter_to_sequence_parallel_region_from_moe"
)

if self.config.is_scaling_mode:
    # Override with simulated reduce_scatter effect
    ep_group_size = tp_size * exp_size
    original_size = unpermuted_global_hidden.shape[0] // ep_group_size
    output_total = unpermuted_global_hidden[:original_size]
```

## Data Structure Requirements

### Pre-computed Routing Results
```python
config.pre_fixed_routing_results = {
    0: {
        'scores': tensor([...]),    # Shape: [S*B/TP, topk]
        'indices': tensor([...])    # Shape: [S*B/TP, topk]
    },
    # ... for each expert parallel rank
}
```

### Per-rank Dispatching Results
```python
config.per_rank_dispatching_results = {
    0: {
        'num_local_tokens_per_expert': tensor([...])  # Shape: [num_experts]
    },
    # ... for each expert parallel rank
}
```

## Key Benefits

### 1. Metadata Recording Preservation
- All communication operations called to record CMD metadata
- Profiling tools receive complete communication operation information
- No loss of tracing data in scaling mode

### 2. Exact Token Distribution Consistency
- Uses pre-computed token distributions from `megatron/profiler/moe`
- Ensures identical expert workloads between scaling and real modes
- Maintains load balancing characteristics

### 3. Tensor Shape Accuracy
- Correctly simulates communication operation tensor size effects
- Maintains compatibility with downstream operations
- Preserves mathematical correctness

## Testing Results

### Successful Integration
- ✅ Passes AllGather dispatcher scaling mode tests
- ✅ Successfully runs complete training simulation
- ✅ Maintains compatibility with GroupedGEMM
- ✅ Preserves tensor shape consistency

### Real Training Validation
- ✅ Successfully processes multiple expert parallel ranks
- ✅ Correct expert assignment (e.g., Rank 3 → Experts [12-15])
- ✅ Complete training step execution (warm up, FWD, BWD, optimizer)
- ✅ No tensor size or device compatibility issues

## Usage Example

```python
# Configuration
config = TransformerConfig(
    pipeline_model_parallel_size=4,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=4,
    num_moe_experts=16,
    moe_router_topk=2,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="allgather",
    is_scaling_mode=True,
    fake_tp=1,
    fake_exp=4,
    exp_rank=0,
)

# Usage
dispatcher = MoEAllGatherTokenDispatcher(
    num_local_experts=4,
    local_expert_indices=[0, 1, 2, 3],
    config=config
)

# Automatically handles scaling mode
permuted_tokens, tokens_per_expert = dispatcher.token_permutation(
    hidden_states, max_prob, max_ind
)
```

## Summary

This implementation successfully provides scaling mode support for `MoEAllGatherTokenDispatcher` by:

1. **Preserving communication metadata recording** through consistent function calls
2. **Using pre-computed token distributions** for exact consistency with real training
3. **Correctly simulating tensor size effects** of communication operations
4. **Maintaining device compatibility** across all tensor operations
5. **Following established patterns** from MoEAlltoAllTokenDispatcher implementation

The result is a robust scaling mode implementation that enables accurate single-GPU simulation of multi-GPU MoE training scenarios while preserving all profiling and tracing capabilities.
