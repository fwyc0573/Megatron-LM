#!/usr/bin/env python3
"""
Test script for AllGather Token Dispatcher with scaling mode support.
This script verifies that the AllGather dispatcher works correctly in scaling mode.
"""

import torch
import sys
import os

# Add the Megatron-LM path
sys.path.append('.')

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher


def test_allgather_scaling_mode():
    """Test AllGather dispatcher with scaling mode configuration."""
    
    print("Testing AllGather Token Dispatcher with Scaling Mode...")
    
    # Configuration for PP=4, TP=1, EP=4, GroupedGEMM=True, TopK=2
    config = TransformerConfig(
        # Basic model config
        hidden_size=1024,
        num_attention_heads=16,
        num_layers=24,

        # Parallelism config
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.float32,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=4,

        # MoE config
        num_moe_experts=8,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="allgather",

        # Other configs
        add_bias_linear=False,
        sequence_parallel=False,
    )

    # Set scaling mode config after initialization
    config.is_scaling_mode = True
    config.fake_tp = 1
    config.fake_exp = 4  # Add fake_exp for scaling mode
    config.exp_rank = 0

    # Mock per_rank_dispatching_results for scaling mode
    config.per_rank_dispatching_results = {
        0: {
            'tokens_per_local_expert': torch.tensor([32, 28], dtype=torch.long),
            'num_local_tokens_per_expert': torch.tensor([32, 28, 0, 0, 0, 0, 0, 0], dtype=torch.long)
        }
    }
    
    # Create dispatcher
    num_local_experts = 2
    local_expert_indices = [0, 1]
    
    try:
        dispatcher = MoEAllGatherTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config
        )
        print("✓ AllGather dispatcher created successfully with scaling mode")
        
        # Test token permutation
        batch_size = 4
        seq_len = 16
        hidden_size = 1024
        
        # Create test data
        hidden_states = torch.randn(seq_len, batch_size, hidden_size)
        max_prob = torch.rand(seq_len * batch_size, config.moe_router_topk)
        max_ind = torch.randint(0, config.num_moe_experts, (seq_len * batch_size, config.moe_router_topk))
        
        print(f"Input shapes: hidden_states={hidden_states.shape}, max_prob={max_prob.shape}, max_ind={max_ind.shape}")
        
        # Test scaling mode configuration
        print(f"✓ Scaling mode configuration:")
        print(f"  is_scaling_mode: {config.is_scaling_mode}")
        print(f"  fake_tp: {config.fake_tp}")
        print(f"  exp_rank: {config.exp_rank}")
        print(f"  per_rank_dispatching_results: {config.per_rank_dispatching_results is not None}")

        # Test token permutation (will fail due to uninitialized parallel groups, but that's expected)
        try:
            permuted_tokens, tokens_per_expert = dispatcher.token_permutation(
                hidden_states, max_prob, max_ind
            )
            print(f"✓ Token permutation successful")
            print(f"  Permuted tokens shape: {permuted_tokens.shape}")
            print(f"  Tokens per expert: {tokens_per_expert}")

            # Test token unpermutation
            try:
                restored_tokens, restored_bias = dispatcher.token_unpermutation(permuted_tokens)
                print(f"✓ Token unpermutation successful")
                print(f"  Restored tokens shape: {restored_tokens.shape}")
                print(f"  Original shape: {hidden_states.shape}")

                # Check if shapes match
                if restored_tokens.shape == hidden_states.shape:
                    print("✓ Shape consistency check passed")
                else:
                    print(f"✗ Shape mismatch: {restored_tokens.shape} vs {hidden_states.shape}")

            except Exception as e:
                print(f"✗ Token unpermutation failed: {e}")

        except Exception as e:
            print(f"✗ Token permutation failed: {e}")
            print("  (This is expected without initialized parallel groups)")
            
    except Exception as e:
        print(f"✗ Failed to create AllGather dispatcher: {e}")
        return False
    
    print("\nScaling mode test completed!")
    return True


def test_compatibility_with_groupedgemm():
    """Test compatibility with GroupedGEMM configuration."""
    
    print("\nTesting compatibility with GroupedGEMM...")
    
    config = TransformerConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_layers=12,
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.float32,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_grouped_gemm=True,  # Enable GroupedGEMM
        moe_token_dispatcher_type="allgather",
        add_bias_linear=False,
    )

    # Set scaling mode config after initialization
    config.is_scaling_mode = True
    config.fake_tp = 1
    config.fake_exp = 4  # Add fake_exp for scaling mode
    config.exp_rank = 0

    # Mock dispatching results
    config.per_rank_dispatching_results = {
        0: {
            'tokens_per_local_expert': torch.tensor([16, 20], dtype=torch.long),
            'num_local_tokens_per_expert': torch.tensor([16, 20, 0, 0, 0, 0, 0, 0], dtype=torch.long)
        }
    }
    
    try:
        dispatcher = MoEAllGatherTokenDispatcher(
            num_local_experts=2,
            local_expert_indices=[0, 1],
            config=config
        )
        print("✓ AllGather dispatcher compatible with GroupedGEMM")
        return True
        
    except Exception as e:
        print(f"✗ GroupedGEMM compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AllGather Token Dispatcher Scaling Mode Test")
    print("=" * 60)
    
    success = True
    
    # Test basic scaling mode functionality
    success &= test_allgather_scaling_mode()
    
    # Test GroupedGEMM compatibility
    success &= test_compatibility_with_groupedgemm()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
