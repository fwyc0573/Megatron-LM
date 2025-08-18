# AllGather Token Dispatcher Scaling Mode Implementation

## Overview

This document describes the implementation of scaling mode support for `MoEAllGatherTokenDispatcher` in Megatron-LM, enabling performance comparison with `MoEAlltoAllTokenDispatcher`.

## Implementation Details

### 1. Core Changes in `MoEAllGatherTokenDispatcher`

#### 1.1 Scaling Mode Support in `token_permutation()`

```python
# Get tensor parallel size for scaling mode support
tp_size = self.config.fake_tp if self.config.is_scaling_mode else parallel_state.get_tensor_model_parallel_world_size()

# Permute the tokens across the expert parallel devices.
if (tp_size > 1) or (self.config.expert_model_parallel_size > 1):
    # In scaling mode, we support TP>1 for tracing purposes
    if tp_size > 1 and not self.config.is_scaling_mode:
        # In real training, AllGather dispatcher supports TP>1 naturally
        pass
```

**Key Features:**
- Uses `config.fake_tp` in scaling mode instead of real TP world size
- Maintains compatibility with both scaling and real training modes
- Supports TP>1 configurations in scaling mode for communication tracing

#### 1.2 Token Distribution Control

```python
# Calculate tokens_per_expert with scaling mode support
if self.config.is_scaling_mode:
    # In scaling mode, use pre-computed token distribution
    if hasattr(self.config, 'per_rank_dispatching_results') and self.config.exp_rank in self.config.per_rank_dispatching_results:
        tokens_per_expert = self.config.per_rank_dispatching_results[self.config.exp_rank]['tokens_per_local_expert']
        if not tokens_per_expert.is_cuda:
            tokens_per_expert = tokens_per_expert.cuda()
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
    else:
        # Fallback to histc calculation
        tokens_per_expert = torch.histc(...)
else:
    tokens_per_expert = torch.histc(...)
```

**Key Features:**
- Uses pre-computed token distribution from `config.per_rank_dispatching_results`
- Provides fallback to dynamic calculation if pre-computed data unavailable
- Ensures compatibility with `megatron/profiler/moe` token distribution control

#### 1.3 Scaling Mode Support in `token_unpermutation()`

```python
# Get tensor parallel size for scaling mode support
tp_size = self.config.fake_tp if self.config.is_scaling_mode else parallel_state.get_tensor_model_parallel_world_size()

# Use tp_size for scaling mode support in bias processing
tp_world_size = tp_size if self.config.is_scaling_mode else parallel_state.get_tensor_model_parallel_world_size()
output_bias_total = output_bias_total / tp_world_size
```

**Key Features:**
- Consistent TP size handling across permutation and unpermutation
- Proper bias scaling in scaling mode
- Maintains numerical correctness in both modes

### 2. Communication Tracing Support

#### 2.1 CMD Decorator Addition

```python
@CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype']}, group_type='exp', comm_func='allgather')
def gather_from_sequence_parallel_region_to_moe(input_, use_global_buffer=False, func=None):
    return _GatherFromSequenceParallelRegionToMOE.apply(input_)

@CMD.get_trace_decorator(attrs={'input_': ['shape', 'dtype']}, group_type='exp', comm_func='reduce_scatter')
def reduce_scatter_to_sequence_parallel_region_from_moe(input_, func=None):
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)
```

**Key Features:**
- Captures AllGather/ReduceScatter communication operations
- Provides performance data for comparison with AlltoAll
- Uses `group_type='exp'` for expert parallel communication tracking

#### 2.2 Function Parameter Updates

```python
# Updated function calls with func parameter for tracing
global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
    max_ind, func="gather_from_sequence_parallel_region_to_moe"
)

output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
    unpermuted_global_hidden, func="reduce_scatter_to_sequence_parallel_region_from_moe"
)
```

### 3. MoE Gather/Scatter Operations

#### 3.1 Custom Autograd Functions

```python
class moe_gather(torch.autograd.Function):
    """Gather operation for MoE token dispatching."""
    
    @staticmethod
    def forward(ctx, input_, indices, output_shape=None):
        ctx.save_for_backward(indices)
        ctx.input_shape = input_.shape
        ctx.output_shape = output_shape
        return torch.gather(input_, 0, indices)
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        if ctx.output_shape is not None:
            grad_input = torch.zeros(ctx.output_shape, dtype=grad_output.dtype, device=grad_output.device)
        else:
            grad_input = torch.zeros(ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input, None, None
```

**Key Features:**
- Proper gradient handling for token gathering/scattering
- Support for custom output shapes
- Memory-efficient implementation

## Configuration Compatibility

### Target Configuration Support

The implementation supports the requested configuration:
- **Pipeline Parallelism (PP)**: >1 ✓
- **Tensor Parallelism (TP)**: 1 ✓  
- **Expert Parallelism (EP)**: >1 ✓
- **GroupedGEMM**: True ✓
- **TopK**: 2 ✓

### Scaling Mode Integration

```python
config = TransformerConfig(
    # Parallelism config
    pipeline_model_parallel_size=4,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=4,
    
    # MoE config
    num_moe_experts=8,
    moe_router_topk=2,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="allgather",
    
    # Scaling mode config
    is_scaling_mode=True,
    fake_tp=1,
    exp_rank=0,
)
```

## Performance Comparison Capability

### 1. Communication Tracing
- AllGather operations tracked with `comm_func='allgather'`
- ReduceScatter operations tracked with `comm_func='reduce_scatter'`
- Expert parallel group communication captured

### 2. Token Distribution Control
- Uses same `per_rank_dispatching_results` structure as AlltoAll
- Ensures identical token routing for fair comparison
- Maintains deterministic behavior across runs

### 3. Memory Usage Tracking
- Scaling mode enables memory profiling
- Compatible with existing memory tracing infrastructure
- Supports comparison of memory patterns between dispatchers

## Testing

A comprehensive test script `test_allgather_scaling_mode.py` is provided to verify:

1. **Basic Functionality**: Token permutation/unpermutation in scaling mode
2. **GroupedGEMM Compatibility**: Integration with GroupedGEMM configuration
3. **Shape Consistency**: Proper tensor shape handling
4. **Configuration Validation**: Support for target configuration parameters

## Usage Example

```python
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher

# Create dispatcher with scaling mode
dispatcher = MoEAllGatherTokenDispatcher(
    num_local_experts=2,
    local_expert_indices=[0, 1],
    config=config  # config with is_scaling_mode=True
)

# Use in MoE layer
permuted_tokens, tokens_per_expert = dispatcher.token_permutation(
    hidden_states, max_prob, max_ind
)

# Process through experts...

restored_tokens, restored_bias = dispatcher.token_unpermutation(expert_output)
```

## Benefits

1. **Performance Comparison**: Enables direct comparison between AllGather and AlltoAll dispatchers
2. **Debugging Support**: Scaling mode facilitates debugging and profiling
3. **Code Consistency**: Maintains same patterns as AlltoAll implementation
4. **Backward Compatibility**: No impact on existing real training workflows
5. **Comprehensive Tracing**: Full communication operation visibility for analysis
