#!/usr/bin/env python3
"""
Test script for the improved communication module functionality.
Tests the enhanced interpolation, cross-machine adaptation, and caching features.
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.comm_sim.nccl_comm import (
    get_comm_op_exc_time, 
    set_cross_machine_correction_factor,
    set_gpus_per_machine,
    clear_comm_cache,
    get_cache_stats,
    validate_comm_database,
    logger
)

def test_basic_functionality():
    """Test basic communication time calculation."""
    print("=== Testing Basic Functionality ===")
    
    # Test single GPU (should return minimal time)
    time_1gpu = get_comm_op_exc_time([0], 1024, "dp_allreduce")
    print(f"1 GPU communication time: {time_1gpu} ms")
    assert time_1gpu == 0.01, "Single GPU should return 0.01 ms"
    
    # Test standard configurations
    test_cases = [
        ([0, 1], 1024, "dp_allreduce"),
        ([0, 1, 2, 3], 1048576, "tp_allreduce"),
        ([0, 1, 2, 3, 4, 5, 6, 7], 67108864, "dp_allreduce"),
    ]
    
    for comm_group, data_size, comm_func in test_cases:
        time_ms = get_comm_op_exc_time(comm_group, data_size, comm_func)
        print(f"{len(comm_group)} GPUs, {data_size} bytes, {comm_func}: {time_ms} ms")
        assert time_ms > 0, "Communication time should be positive"

def test_interpolation():
    """Test enhanced interpolation for missing data sizes."""
    print("\n=== Testing Enhanced Interpolation ===")
    
    # Test interpolation for sizes not in database
    test_sizes = [43 * 1024 * 1024, 150 * 1024 * 1024, 500 * 1024]  # 43MB, 150MB, 500KB
    comm_group = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 GPUs
    
    for data_size in test_sizes:
        time_ms = get_comm_op_exc_time(comm_group, data_size, "dp_allreduce")
        print(f"Interpolated time for {data_size} bytes: {time_ms} ms")
        assert time_ms > 0, "Interpolated time should be positive"

def test_cross_machine_adaptation():
    """Test cross-machine communication adaptation."""
    print("\n=== Testing Cross-Machine Adaptation ===")
    
    # Test 8 GPUs distributed across 2 machines (ranks 0-3 on machine 0, 4-7 on machine 1)
    cross_machine_group = [0, 1, 2, 3, 8, 9, 10, 11]  # Spans machines 0 and 1
    same_machine_group = [0, 1, 2, 3, 4, 5, 6, 7]     # All on machine 0
    
    data_size = 256 * 1024 * 1024  # 256MB
    
    cross_time = get_comm_op_exc_time(cross_machine_group, data_size, "dp_allreduce")
    same_time = get_comm_op_exc_time(same_machine_group, data_size, "dp_allreduce")
    
    print(f"Cross-machine 8 GPUs: {cross_time} ms")
    print(f"Same-machine 8 GPUs: {same_time} ms")
    
    # Cross-machine should be different due to correction factor
    print(f"Time difference ratio: {cross_time/same_time:.3f}")

def test_gpu_config_adaptation():
    """Test adaptation for GPU configurations not in database."""
    print("\n=== Testing GPU Configuration Adaptation ===")
    
    # Test configurations that need adaptation
    test_configs = [
        ([0, 1, 2, 3, 4, 5], "6 GPUs"),           # Should adapt from 8 or 4
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "10 GPUs"),  # Should adapt from 8 or 16
        ([i for i in range(12)], "12 GPUs"),      # Should adapt from 16 or 8
    ]
    
    data_size = 100 * 1024 * 1024  # 100MB
    
    for comm_group, description in test_configs:
        time_ms = get_comm_op_exc_time(comm_group, data_size, "dp_allreduce")
        print(f"{description}: {time_ms} ms")
        assert time_ms > 0, f"Time for {description} should be positive"

def test_caching():
    """Test caching functionality."""
    print("\n=== Testing Caching Functionality ===")
    
    # Clear cache and check stats
    clear_comm_cache()
    stats = get_cache_stats()
    print(f"Cache stats after clear: {stats}")
    assert stats['cache_size'] == 0, "Cache should be empty after clear"
    
    # Perform some calculations to populate cache
    comm_group = [0, 1, 2, 3]
    data_size = 1024 * 1024
    
    # First call should populate cache
    time1 = get_comm_op_exc_time(comm_group, data_size, "tp_allreduce")
    stats_after_first = get_cache_stats()
    
    # Second call should use cache
    time2 = get_comm_op_exc_time(comm_group, data_size, "tp_allreduce")
    stats_after_second = get_cache_stats()
    
    print(f"First call: {time1} ms, cache size: {stats_after_first['cache_size']}")
    print(f"Second call: {time2} ms, cache size: {stats_after_second['cache_size']}")
    
    assert time1 == time2, "Cached results should be identical"
    assert stats_after_first['cache_size'] == stats_after_second['cache_size'], "Cache size should not change on hit"

def test_configuration_options():
    """Test configuration option changes."""
    print("\n=== Testing Configuration Options ===")
    
    # Test changing correction factor
    original_factor = 0.95
    new_factor = 0.90
    
    set_cross_machine_correction_factor(new_factor)
    clear_comm_cache()  # Clear cache to see effect
    
    # Test with cross-machine group
    cross_group = [0, 1, 2, 3, 8, 9, 10, 11]
    time_with_new_factor = get_comm_op_exc_time(cross_group, 1024*1024, "dp_allreduce")
    
    # Reset to original
    set_cross_machine_correction_factor(original_factor)
    clear_comm_cache()
    
    time_with_original = get_comm_op_exc_time(cross_group, 1024*1024, "dp_allreduce")
    
    print(f"Time with factor {new_factor}: {time_with_new_factor} ms")
    print(f"Time with factor {original_factor}: {time_with_original} ms")
    
    # Test changing GPUs per machine
    set_gpus_per_machine(4)  # Change from default 8 to 4
    clear_comm_cache()
    
    time_with_4gpu_machine = get_comm_op_exc_time(cross_group, 1024*1024, "dp_allreduce")
    print(f"Time with 4 GPUs per machine: {time_with_4gpu_machine} ms")
    
    # Reset to default
    set_gpus_per_machine(8)

def test_database_validation():
    """Test database validation functionality."""
    print("\n=== Testing Database Validation ===")
    
    is_valid = validate_comm_database()
    print(f"Database validation result: {is_valid}")
    assert is_valid, "Database should be valid"

def main():
    """Run all tests."""
    print("Starting Communication Module Enhancement Tests")
    print("=" * 60)
    
    # Set logging level for detailed output
    logger.setLevel(logging.INFO)
    
    try:
        test_basic_functionality()
        test_interpolation()
        test_cross_machine_adaptation()
        test_gpu_config_adaptation()
        test_caching()
        test_configuration_options()
        test_database_validation()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        
        # Print final cache stats
        final_stats = get_cache_stats()
        print(f"Final cache stats: {final_stats}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import logging
    exit(main())
