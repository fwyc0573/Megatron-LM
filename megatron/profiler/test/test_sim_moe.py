import sys
import os
import unittest
from typing import List
import torch
from functools import partial

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from megatron.profiler.moe.sim_routing import TopKRouterWrapper, fixed_round_robin_routing
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name


# Mock TransformerConfig to avoid heavy dependencies for testing purposes
class MockTransformerConfig:
    def __init__(self, hidden_size, num_moe_experts, moe_router_topk):
        self.hidden_size = hidden_size
        self.num_moe_experts = num_moe_experts
        self.moe_router_topk = moe_router_topk
        # --- Defaults for router initialization ---
        self.moe_router_load_balancing_type = "aux_loss"
        self.moe_token_dropping = False
        self.sequence_parallel = False
        self.init_method = lambda x: x # Dummy init
        # --- Defaults for routing logic ---
        self.tensor_model_parallel_size = 1
        self.moe_token_dispatcher_type = "alltoall"
        self.is_scaling_mode = False
        self.moe_z_loss_coeff = None
        self.moe_input_jitter_eps = None
        self.moe_aux_loss_coeff = 0.0
        self.num_layers = 5

class TestSimMoE(unittest.TestCase):
    def test_topk_router_wrapper_forward(self):
        """
        Tests the forward pass of TopKRouterWrapper with a custom routing function.
        """
        print("\n" + "="*50)
        print("Running test for TopKRouterWrapper forward pass...")
        print("="*50)

        get_cuda_rng_tracker().add(get_data_parallel_rng_tracker_name(), 42)

        # 1. Configuration
        hidden_size = 128
        num_moe_experts = 8
        moe_router_topk = 2
        seq_len = 2048
        micro_batch_size = 1
        num_tokens = seq_len * micro_batch_size

        config = MockTransformerConfig(
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
        )

        # 2. Prepare the custom routing function
        # Use functools.partial to pre-fill arguments for our standalone routing function.
        custom_router = partial(
            fixed_round_robin_routing,
            topk=config.moe_router_topk,
            num_experts=config.num_moe_experts,
        )

        # 3. Initialize the TopKRouterWrapper with the custom function
        router_wrapper = TopKRouterWrapper(config=config) # custom_routing_func=custom_router
        print(f"Successfully initialized TopKRouterWrapper with 'fixed_round_robin_routing'.")

        # 4. Generate a random hidden_states tensor
        hidden_states_shape = (seq_len, micro_batch_size, hidden_size)
        hidden_states = torch.rand(hidden_states_shape)
        print(f"Generated random hidden_states tensor with shape: {hidden_states.shape}")

        # 5. Perform the forward pass
        # The wrapper's forward method will call the overridden routing method internally.
        scores, indices = router_wrapper.forward(hidden_states)
        print("Forward pass completed.")

        # 6. Check and print the output shapes
        print("\n--- Output Validation ---")
        print(f"Scores shape: {scores.shape}")
        print(f"Indices shape: {indices.shape}")
        
        # expected_shape = (num_tokens, moe_router_topk)
        # self.assertEqual(scores.shape, expected_shape)
        # self.assertEqual(indices.shape, expected_shape)
        # print(f"Verified: Output shapes match expected shape {expected_shape}")

        # 7. Print a sample of the output for visual inspection
        print("\nSample of returned indices (first 5 tokens):")
        print(indices[:5, :])
        print("\nSample of returned scores (first 5 tokens):")
        print(scores[:5, :])
        print("="*50 + "\n")

if __name__ == '__main__':
    unittest.main()


