import sys
import os
import unittest
from typing import List
import pprint

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from megatron.profiler.sim_parallel_state import sim_initialize_model_parallel, RankGenerator


class TestSimParallelState(unittest.TestCase):
    
    def verify_group_consistency(self, groups_info, world_size):
        """Verify all ranks in groups are within valid range and not duplicated."""
        all_ranks = set()
        for group_name, group_list in groups_info.items():
            if group_list is None:
                continue
                
            if isinstance(group_list[0], list):  # Handle nested lists
                for group in group_list:
                    for rank in group:
                        self.assertLess(rank, world_size, f"Rank {rank} in {group_name} exceeds world_size {world_size}")
                        self.assertGreaterEqual(rank, 0, f"Rank {rank} in {group_name} is negative")
            else:  # Handle flat lists
                for rank in group_list:
                    self.assertLess(rank, world_size, f"Rank {rank} in {group_name} exceeds world_size {world_size}")
                    self.assertGreaterEqual(rank, 0, f"Rank {rank} in {group_name} is negative")
    
    def verify_group_completeness(self, groups_info, expected_keys):
        """Verify all expected group keys exist in the groups_info dictionary."""
        for key in expected_keys:
            self.assertIn(key, groups_info, f"Missing expected group: {key}")
    
    def print_group_details(self, groups_info, test_name):
        """Print detailed information about each group."""
        print(f"\n{'='*80}\n{test_name} - Group Details:\n{'='*80}")
        
        # Count non-None groups
        valid_groups = {k: v for k, v in groups_info.items() if v is not None}
        print(f"Total valid groups: {len(valid_groups)}")
        
        # Print each group with nice formatting
        for group_name, group_data in valid_groups.items():
            print(f"\n{'-'*40}")
            print(f"Group: {group_name}")
            
            if isinstance(group_data[0], list):  # Nested list
                print(f"  Number of subgroups: {len(group_data)}")
                for i, subgroup in enumerate(group_data):
                    print(f"  Subgroup {i}: {subgroup}")
            else:  # Flat list
                print(f"  Ranks: {group_data}")
        
        print(f"\n{'='*80}\n")
    
    # def test_simple_config(self):
    #     """Test simple parallel configuration: 2x2 tensor-pipeline parallel."""
    #     world_size = 8
    #     groups_info = sim_initialize_model_parallel(
    #         world_size=world_size,
    #         tensor_model_parallel_size=2,
    #         pipeline_model_parallel_size=2,
    #         context_parallel_size=1,
    #         expert_model_parallel_size=1
    #     )
        
    #     self.print_group_details(groups_info, "Simple Config (2x2 TP-PP)")
        
    #     expected_keys = [
    #         'data_parallel_global_ranks',
    #         'tensor_model_parallel_global_ranks',
    #         'pipeline_global_ranks',
    #         'embedding_global_ranks',
    #         'position_embedding_global_ranks',
    #         'tensor_and_data_parallel_group',
    #     ]
        
    #     self.verify_group_completeness(groups_info, expected_keys)
    #     self.verify_group_consistency(groups_info, world_size)
        
    # def test_expert_parallel_config(self):
    #     """Test expert parallel configuration."""
    #     world_size = 8
    #     groups_info = sim_initialize_model_parallel(
    #         world_size=world_size,
    #         tensor_model_parallel_size=2,
    #         pipeline_model_parallel_size=2,
    #         context_parallel_size=1,
    #         expert_model_parallel_size=2
    #    )
        
    #     self.print_group_details(groups_info, "Expert Parallel Config (2x2 TP-PP with EP=2)")
        
    #     expected_keys = [
    #         'data_parallel_global_ranks',
    #         'tensor_model_parallel_global_ranks',
    #         'pipeline_global_ranks',
    #         'expert_model_parallel_group',
    #         'tensor_and_expert_parallel_group',
    #         'data_modulo_expert_parallel_group',
    #     ]
        
    #     self.verify_group_completeness(groups_info, expected_keys)
    #     self.verify_group_consistency(groups_info, world_size)
        
    #     # Verify expert parallel groups
    #     ep_groups = groups_info['expert_model_parallel_group']
    #     self.assertTrue(len(ep_groups) > 0)
    #     print(f"\nExpert parallel groups: {ep_groups}")
        
    #     # Verify tensor-expert parallel groups
    #     tp_ep_groups = groups_info['tensor_and_expert_parallel_group']
    #     self.assertTrue(len(tp_ep_groups) > 0)
    #     print(f"Tensor-Expert parallel groups: {tp_ep_groups}")
    
    # def test_larger_config(self):
    #     """Test a larger configuration to visualize more complex group patterns."""
    #     world_size = 8  # 2x2x2
    #     groups_info = sim_initialize_model_parallel(
    #         world_size=world_size,
    #         tensor_model_parallel_size=2,
    #         pipeline_model_parallel_size=2,
    #         context_parallel_size=2,
    #         expert_model_parallel_size=1
    #     )
        
    #     self.print_group_details(groups_info, "Larger Config (2x2x2 TP-CP-PP)")
        
    #     # Print a visual representation of the process grid
    #     print("\nProcess Grid Visualization:")
    #     print("===========================")
    #     print("Format: (TP, CP, PP, DP)")
        
    #     # Create a grid representation
    #     grid = []
    #     for pp in range(2):
    #         pp_layer = []
    #         for cp in range(2):
    #             cp_layer = []
    #             for tp in range(2):
    #                 dp = 1  # Only one DP dimension in this case
    #                 cp_layer.append(f"({tp},{cp},{pp},{dp})")
    #             pp_layer.append(cp_layer)
    #         grid.append(pp_layer)
        
    #     # Print the grid
    #     for pp_idx, pp_layer in enumerate(grid):
    #         print(f"\nPP Layer {pp_idx}:")
    #         for cp_idx, cp_layer in enumerate(pp_layer):
    #             print(f"  CP {cp_idx}: {cp_layer}")
    
    def test_rank_generator(self):
        """Test RankGenerator functionality."""
        rg = RankGenerator(
            tp=4, ep=1, dp=2, pp=1, cp=1, order="tp-cp-ep-dp-pp"
        )
        
        # Test getting DP groups
        dp_groups = rg.get_ranks('dp')
        print("\nDP Groups:", dp_groups)
        # self.assertEqual(len(dp_groups), 4)  # Should have 4 DP groups
        # for group in dp_groups:
        #     self.assertEqual(len(group), 2)  # Each DP group should have 2 ranks
        
        # Test getting TP groups
        tp_groups = rg.get_ranks('tp')
        print("\nTP Groups:", tp_groups)
        # self.assertEqual(len(tp_groups), 4)  # Should have 4 TP groups
        # for group in tp_groups:
        #     self.assertEqual(len(group), 2)  # Each TP group should have 2 ranks
        
        # # Test getting PP groups
        pp_groups = rg.get_ranks('pp')
        print("\nPP Groups:", pp_groups)
        # self.assertEqual(len(pp_groups), 4)  # Should have 4 PP groups
        # for group in pp_groups:
        #     self.assertEqual(len(group), 2)  # Each PP group should have 2 ranks
        
        # Test getting compound TP-DP groups
        tp_dp_groups = rg.get_ranks('tp-dp')
        print("\nTP-DP Groups:", tp_dp_groups)
        # self.assertEqual(len(tp_dp_groups), 2)  # Should have 2 TP-DP groups
        # for group in tp_dp_groups:
        #     self.assertEqual(len(group), 4)  # Each TP-DP group should have 4 ranks


if __name__ == "__main__":
    # Set verbosity to get more detailed output
    unittest.main(verbosity=2)