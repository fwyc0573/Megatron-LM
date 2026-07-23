#!/usr/bin/env python3
"""
Demonstration script showing how to use the enhanced communication module
for specific scenarios mentioned in the requirements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.comm_sim.nccl_comm import (
    get_comm_op_exc_time, 
    set_cross_machine_correction_factor,
    set_gpus_per_machine,
    clear_comm_cache,
    get_cache_stats,
    logger
)

def demo_tensor_size_interpolation():
    """
    Demonstrate tensor size interpolation for missing data sizes.
    Example: Predict 8-card 43MB allreduce time based on existing data.
    """
    print("=== Tensor Size Interpolation Demo ===")
    print("Scenario: Predict 8-card 43MB allreduce time from existing data points")
    
    # 8 GPUs on same machine
    comm_group = [0, 1, 2, 3, 4, 5, 6, 7]
    target_size = 43 * 1024 * 1024  # 43MB in bytes
    
    # Get prediction for target size
    predicted_time = get_comm_op_exc_time(comm_group, target_size, "dp_allreduce")
    
    # Also get times for known data points for comparison
    known_sizes = [
        (32 * 1024 * 1024, "32MB"),
        (64 * 1024 * 1024, "64MB"),
        (100 * 1024 * 1024, "100MB")
    ]
    
    print(f"\nTarget prediction:")
    print(f"  8 GPUs, 43MB: {predicted_time:.2f} ms")
    
    print(f"\nComparison with known sizes:")
    for size_bytes, size_label in known_sizes:
        time_ms = get_comm_op_exc_time(comm_group, size_bytes, "dp_allreduce")
        print(f"  8 GPUs, {size_label}: {time_ms:.2f} ms")
    
    print(f"\nInterpolation shows reasonable progression between data points.")

def demo_cross_machine_adaptation():
    """
    Demonstrate cross-machine communication adaptation.
    Example: 8 GPUs distributed across 2 machines vs single machine.
    """
    print("\n=== Cross-Machine Communication Adaptation Demo ===")
    print("Scenario: 8 GPUs distributed across 2 machines (4+4) vs single machine")
    
    # Single machine: all GPUs 0-7 on machine 0
    single_machine_group = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Cross-machine: GPUs 0-3 on machine 0, GPUs 8-11 on machine 1
    cross_machine_group = [0, 1, 2, 3, 8, 9, 10, 11]
    
    test_size = 256 * 1024 * 1024  # 256MB
    
    single_time = get_comm_op_exc_time(single_machine_group, test_size, "dp_allreduce")
    cross_time = get_comm_op_exc_time(cross_machine_group, test_size, "dp_allreduce")
    
    print(f"\nResults for 256MB allreduce:")
    print(f"  Single machine (GPUs 0-7): {single_time:.2f} ms")
    print(f"  Cross-machine (GPUs 0-3,8-11): {cross_time:.2f} ms")
    print(f"  Cross-machine overhead: {((cross_time/single_time - 1) * 100):+.1f}%")
    
    # Test with different correction factors
    print(f"\nTesting different correction factors:")
    factors = [0.90, 0.95, 1.00, 1.05]
    
    for factor in factors:
        set_cross_machine_correction_factor(factor)
        clear_comm_cache()  # Clear cache to apply new factor
        
        time_with_factor = get_comm_op_exc_time(cross_machine_group, test_size, "dp_allreduce")
        print(f"  Factor {factor:.2f}: {time_with_factor:.2f} ms")
    
    # Reset to default
    set_cross_machine_correction_factor(0.95)

def demo_gpu_config_adaptation():
    """
    Demonstrate GPU configuration adaptation for non-standard group sizes.
    """
    print("\n=== GPU Configuration Adaptation Demo ===")
    print("Scenario: Predict times for non-standard GPU group sizes")
    
    test_configs = [
        ([0, 1, 2, 3, 4, 5], "6 GPUs (adapts from 4 or 8)"),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "10 GPUs (adapts from 8 or 16)"),
        ([i for i in range(12)], "12 GPUs (adapts from 8 or 16)"),
        ([i for i in range(20)], "20 GPUs (adapts from 16 or 32)"),
    ]
    
    test_size = 128 * 1024 * 1024  # 128MB
    
    print(f"\nPredictions for {test_size // (1024*1024)}MB allreduce:")
    
    for comm_group, description in test_configs:
        time_ms = get_comm_op_exc_time(comm_group, test_size, "dp_allreduce")
        print(f"  {description}: {time_ms:.2f} ms")

def demo_realistic_scenarios():
    """
    Demonstrate realistic training scenarios with various tensor sizes and configurations.
    """
    print("\n=== Realistic Training Scenarios Demo ===")
    
    # Common model sizes and their approximate gradient sizes
    model_scenarios = [
        ("7B model gradients", 28 * 1024 * 1024),    # ~28MB
        ("13B model gradients", 52 * 1024 * 1024),   # ~52MB  
        ("30B model gradients", 120 * 1024 * 1024),  # ~120MB
        ("70B model gradients", 280 * 1024 * 1024),  # ~280MB
    ]
    
    # Different parallelism configurations
    parallel_configs = [
        ([0, 1, 2, 3, 4, 5, 6, 7], "8 GPUs (single node)"),
        ([0, 1, 2, 3, 8, 9, 10, 11], "8 GPUs (2 nodes, 4+4)"),
        ([i for i in range(16)], "16 GPUs (2 nodes, 8+8)"),
        ([i for i in range(32)], "32 GPUs (4 nodes, 8+8+8+8)"),
    ]
    
    print("\nData Parallel AllReduce Times:")
    print("=" * 80)
    print(f"{'Model Size':<20} {'Config':<25} {'Time (ms)':<12} {'Bandwidth':<15}")
    print("-" * 80)
    
    for model_name, gradient_size in model_scenarios:
        for comm_group, config_name in parallel_configs:
            time_ms = get_comm_op_exc_time(comm_group, gradient_size, "dp_allreduce")
            
            # Calculate effective bandwidth (approximate)
            # AllReduce transfers 2*(N-1)/N * data_size
            n_gpus = len(comm_group)
            effective_data = gradient_size * 2 * (n_gpus - 1) / n_gpus
            bandwidth_gbps = (effective_data * 8) / (time_ms * 1e-3) / 1e9  # Gbps
            
            print(f"{model_name:<20} {config_name:<25} {time_ms:<12.2f} {bandwidth_gbps:<15.1f}")

def demo_caching_benefits():
    """
    Demonstrate the benefits of caching for repeated calculations.
    """
    print("\n=== Caching Benefits Demo ===")
    
    import time
    
    # Clear cache first
    clear_comm_cache()
    
    # Test scenario
    comm_group = [0, 1, 2, 3, 4, 5, 6, 7]
    data_size = 100 * 1024 * 1024
    comm_func = "dp_allreduce"
    
    # Time first calculation (cache miss)
    start_time = time.time()
    result1 = get_comm_op_exc_time(comm_group, data_size, comm_func)
    first_calc_time = time.time() - start_time
    
    # Time second calculation (cache hit)
    start_time = time.time()
    result2 = get_comm_op_exc_time(comm_group, data_size, comm_func)
    second_calc_time = time.time() - start_time
    
    print(f"First calculation (cache miss): {result1:.2f} ms, took {first_calc_time*1000:.2f} ms")
    print(f"Second calculation (cache hit): {result2:.2f} ms, took {second_calc_time*1000:.2f} ms")
    print(f"Speedup from caching: {first_calc_time/second_calc_time:.1f}x")
    
    # Show cache stats
    stats = get_cache_stats()
    print(f"Cache stats: {stats}")

def main():
    """Run all demonstration scenarios."""
    print("Communication Module Enhancement Demonstration")
    print("=" * 60)
    
    # Set logging level for detailed output
    logger.setLevel(logging.INFO)
    
    try:
        demo_tensor_size_interpolation()
        demo_cross_machine_adaptation()
        demo_gpu_config_adaptation()
        demo_realistic_scenarios()
        demo_caching_benefits()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import logging
    exit(main())
