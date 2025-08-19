#!/usr/bin/env python3
"""
Performance Test for 8K GPU Scale

This module tests the performance of CC-estimator integration at 8K GPU scale,
evaluating prediction overhead and cache effectiveness.
"""

import os
import sys
import time
import statistics
import random
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator_config import create_h800_sxm_ib_config
from cc_estimator_integration import CCEstimatorWrapper
from message_size_calculator import MessageSizeCalculator


class PerformanceProfiler:
    """Performance profiler for communication prediction"""
    
    def __init__(self):
        self.measurements = {}
    
    def measure_time(self, operation_name: str, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        
        self.measurements[operation_name].append(execution_time)
        
        return result
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if operation_name not in self.measurements:
            return {}
        
        times = self.measurements[operation_name]
        return {
            'count': len(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'total': sum(times)
        }
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING REPORT")
        print("="*80)
        
        for operation, stats in self.measurements.items():
            print(f"\nOperation: {operation}")
            print("-" * 40)
            
            stats_dict = self.get_statistics(operation)
            print(f"Count:      {stats_dict['count']:>8}")
            print(f"Mean:       {stats_dict['mean']:>8.3f} ms")
            print(f"Median:     {stats_dict['median']:>8.3f} ms")
            print(f"Min:        {stats_dict['min']:>8.3f} ms")
            print(f"Max:        {stats_dict['max']:>8.3f} ms")
            print(f"Std Dev:    {stats_dict['std']:>8.3f} ms")
            print(f"Total:      {stats_dict['total']:>8.3f} ms")


def generate_8k_communication_scenarios() -> List[Tuple[List[int], int, str]]:
    """
    Generate realistic communication scenarios for 8K GPU scale
    
    Returns:
        List of (comm_group, data_size, comm_func) tuples
    """
    scenarios = []
    
    # Typical communication patterns in 8K GPU training
    # Assume 8 GPUs per node, 1024 nodes total
    gpus_per_node = 8
    total_gpus = 8192
    num_nodes = total_gpus // gpus_per_node
    
    # Data parallel groups (within nodes and across nodes)
    dp_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Tensor parallel groups (typically within nodes)
    tp_sizes = [2, 4, 8]
    
    # Pipeline parallel groups (across nodes)
    pp_sizes = [2, 4, 8, 16, 32, 64]
    
    # Expert parallel groups (for MoE)
    ep_sizes = [2, 4, 8, 16, 32, 64]
    
    # Typical tensor sizes (in bytes)
    tensor_sizes = [
        1024 * 1024,      # 1MB
        4 * 1024 * 1024,  # 4MB
        16 * 1024 * 1024, # 16MB
        64 * 1024 * 1024, # 64MB
        256 * 1024 * 1024 # 256MB
    ]
    
    # Generate DP AllReduce scenarios
    for dp_size in dp_sizes:
        if dp_size <= total_gpus:
            for tensor_size in tensor_sizes:
                # Create random DP group
                start_rank = random.randint(0, total_gpus - dp_size)
                comm_group = list(range(start_rank, start_rank + dp_size))
                scenarios.append((comm_group, tensor_size, 'dp_allreduce'))
    
    # Generate TP AllReduce scenarios (typically smaller groups)
    for tp_size in tp_sizes:
        for tensor_size in tensor_sizes[:3]:  # Smaller tensors for TP
            # Create intra-node TP group
            node_id = random.randint(0, num_nodes - 1)
            start_rank = node_id * gpus_per_node
            comm_group = list(range(start_rank, start_rank + tp_size))
            scenarios.append((comm_group, tensor_size, 'tp_allreduce'))
    
    # Generate P2P scenarios (pipeline parallel)
    for _ in range(100):  # Generate 100 random P2P scenarios
        rank1 = random.randint(0, total_gpus - 1)
        rank2 = random.randint(0, total_gpus - 1)
        if rank1 != rank2:
            tensor_size = random.choice(tensor_sizes)
            scenarios.append(([rank1, rank2], tensor_size, 'send_forward'))
    
    # Generate MoE scenarios
    for ep_size in ep_sizes:
        if ep_size <= total_gpus:
            for tensor_size in tensor_sizes:
                start_rank = random.randint(0, total_gpus - ep_size)
                comm_group = list(range(start_rank, start_rank + ep_size))
                
                # Different MoE operations
                moe_ops = ['exp_allgather', 'exp_all_to_all', 'exp_dp_allreduce']
                for op in moe_ops:
                    scenarios.append((comm_group, tensor_size, op))
    
    return scenarios


def test_prediction_performance():
    """Test prediction performance at 8K scale"""
    print("Testing CC-estimator performance at 8K GPU scale...")
    
    # Initialize components
    config = create_h800_sxm_ib_config()
    config.communication.cache_predictions = True
    config.communication.cache_size_limit = 50000  # Large cache for 8K scale
    
    profiler = PerformanceProfiler()
    
    try:
        # Initialize estimator
        estimator = profiler.measure_time(
            "estimator_init",
            CCEstimatorWrapper,
            config
        )
        
        print(f"Estimator initialized successfully")
        
        # Generate communication scenarios
        print("Generating 8K scale communication scenarios...")
        scenarios = generate_8k_communication_scenarios()
        print(f"Generated {len(scenarios)} communication scenarios")
        
        # Test prediction performance
        print("Testing prediction performance...")
        
        # First pass: measure cold cache performance
        cold_cache_scenarios = scenarios[:1000]  # First 1000 scenarios
        
        for i, (comm_group, data_size, comm_func) in enumerate(cold_cache_scenarios):
            if i % 100 == 0:
                print(f"Cold cache progress: {i}/{len(cold_cache_scenarios)}")
            
            profiler.measure_time(
                "cold_cache_prediction",
                estimator.predict_communication_time,
                comm_group, data_size, comm_func
            )
        
        # Second pass: measure warm cache performance (repeat some scenarios)
        warm_cache_scenarios = random.sample(cold_cache_scenarios, 500)
        
        for i, (comm_group, data_size, comm_func) in enumerate(warm_cache_scenarios):
            if i % 100 == 0:
                print(f"Warm cache progress: {i}/{len(warm_cache_scenarios)}")
            
            profiler.measure_time(
                "warm_cache_prediction",
                estimator.predict_communication_time,
                comm_group, data_size, comm_func
            )
        
        # Test cache effectiveness
        cache_stats = estimator.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"Cache size: {cache_stats['cache_size']}")
        print(f"Cache limit: {cache_stats['cache_limit']}")
        print(f"Cache enabled: {cache_stats['cache_enabled']}")
        
        # Test different group sizes
        group_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        tensor_size = 16 * 1024 * 1024  # 16MB
        
        for group_size in group_sizes:
            if group_size <= 8192:
                comm_group = list(range(group_size))
                
                profiler.measure_time(
                    f"group_size_{group_size}",
                    estimator.predict_communication_time,
                    comm_group, tensor_size, 'dp_allreduce'
                )
        
        # Print performance report
        profiler.print_report()
        
        # Analyze scalability
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS")
        print("="*80)
        
        cold_stats = profiler.get_statistics("cold_cache_prediction")
        warm_stats = profiler.get_statistics("warm_cache_prediction")
        
        if cold_stats and warm_stats:
            speedup = cold_stats['mean'] / warm_stats['mean']
            print(f"Cache speedup: {speedup:.2f}x")
            print(f"Cold cache mean: {cold_stats['mean']:.3f} ms")
            print(f"Warm cache mean: {warm_stats['mean']:.3f} ms")
        
        # Estimate total simulation overhead
        total_predictions = len(scenarios)
        avg_prediction_time = cold_stats['mean'] if cold_stats else 1.0
        total_overhead = total_predictions * avg_prediction_time / 1000  # Convert to seconds
        
        print(f"\nEstimated overhead for full 8K simulation:")
        print(f"Total predictions: {total_predictions}")
        print(f"Average prediction time: {avg_prediction_time:.3f} ms")
        print(f"Total overhead: {total_overhead:.2f} seconds")
        
        # Performance recommendations
        print("\n" + "="*80)
        print("PERFORMANCE RECOMMENDATIONS")
        print("="*80)
        
        if avg_prediction_time > 1.0:
            print("⚠️  High prediction overhead detected!")
            print("   Recommendations:")
            print("   - Enable caching (already enabled)")
            print("   - Consider prediction batching")
            print("   - Profile CC-estimator internals")
        else:
            print("✅ Prediction overhead is acceptable")
        
        if cache_stats['cache_size'] >= cache_stats['cache_limit'] * 0.9:
            print("⚠️  Cache is nearly full!")
            print("   Recommendations:")
            print("   - Increase cache size limit")
            print("   - Implement cache eviction strategy")
        else:
            print("✅ Cache utilization is healthy")
        
    except ImportError as e:
        print(f"CC-estimator not available: {e}")
        print("Skipping performance test")
    
    except Exception as e:
        print(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()


def test_memory_usage():
    """Test memory usage of CC-estimator integration"""
    print("\nTesting memory usage...")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Initialize estimator
        config = create_h800_sxm_ib_config()
        estimator = CCEstimatorWrapper(config)
        
        after_init_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory after initialization: {after_init_memory:.1f} MB")
        print(f"Initialization overhead: {after_init_memory - initial_memory:.1f} MB")
        
        # Run many predictions to test memory growth
        scenarios = generate_8k_communication_scenarios()[:5000]  # 5000 scenarios
        
        for i, (comm_group, data_size, comm_func) in enumerate(scenarios):
            estimator.predict_communication_time(comm_group, data_size, comm_func)
            
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                print(f"Memory after {i} predictions: {current_memory:.1f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Total memory growth: {final_memory - initial_memory:.1f} MB")
        
        # Check for memory leaks
        if final_memory > initial_memory * 2:
            print("⚠️  Potential memory leak detected!")
        else:
            print("✅ Memory usage is stable")
    
    except ImportError:
        print("psutil not available, skipping memory test")


if __name__ == '__main__':
    print("CC-estimator 8K Scale Performance Test")
    print("="*50)
    
    # Run performance tests
    test_prediction_performance()
    test_memory_usage()
    
    print("\nPerformance testing completed!")
