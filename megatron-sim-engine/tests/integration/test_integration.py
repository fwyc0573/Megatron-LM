#!/usr/bin/env python3
"""
Integration test to ensure the improved communication module works correctly
with the existing simu_engine.
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.comm_sim.nccl_comm import get_comm_op_exc_time, validate_comm_database

def test_integration_with_simu_engine():
    """Test integration with existing simu_engine patterns."""
    print("=== Integration Test with Simu Engine ===")
    
    # Test typical simu_engine usage patterns
    test_cases = [
        # (comm_group, data_size, comm_func, description)
        ([0, 1], 1024, "send_recv", "PP communication"),
        ([0, 1, 2, 3], 4194304, "tp_allreduce", "TP allreduce"),
        ([0, 8, 16, 24], 67108864, "dp_allreduce", "DP allreduce"),
        ([0, 1, 2, 3, 4, 5, 6, 7], 134217728, "dp_allreduce", "8-GPU DP"),
        ([i for i in range(16)], 268435456, "dp_allreduce", "16-GPU DP"),
    ]
    
    print("Testing typical simu_engine communication patterns:")
    print("-" * 60)
    
    for comm_group, data_size, comm_func, description in test_cases:
        try:
            time_ms = get_comm_op_exc_time(comm_group, data_size, comm_func)
            print(f"{description:<20}: {time_ms:>8.2f} ms")
            assert time_ms >= 0, f"Time should be non-negative for {description}"
        except Exception as e:
            print(f"{description:<20}: ERROR - {e}")
            return False
    
    return True

def test_moe_scenarios():
    """Test MOE-specific communication scenarios."""
    print("\n=== MOE Communication Scenarios ===")
    
    # Test expert parallel scenarios
    moe_cases = [
        ([0, 1, 2, 3], 16777216, "ep_allreduce", "EP allreduce 4 experts"),
        ([0, 1, 2, 3, 4, 5, 6, 7], 33554432, "exp_all_to_all", "Expert all-to-all"),
        ([0, 1, 2, 3, 4, 5, 6, 7], 8388608, "exp_allgather", "Expert allgather"),
    ]
    
    print("Testing MOE communication patterns:")
    print("-" * 60)
    
    for comm_group, data_size, comm_func, description in moe_cases:
        try:
            time_ms = get_comm_op_exc_time(comm_group, data_size, comm_func)
            print(f"{description:<25}: {time_ms:>8.2f} ms")
            assert time_ms >= 0, f"Time should be non-negative for {description}"
        except Exception as e:
            print(f"{description:<25}: ERROR - {e}")
            return False
    
    return True

def test_edge_cases():
    """Test edge cases that might occur in simu_engine."""
    print("\n=== Edge Cases Test ===")
    
    edge_cases = [
        # Single GPU (should return minimal time)
        ([0], 1024, "dp_allreduce", "Single GPU"),
        # Very small data
        ([0, 1], 1, "send_recv", "1 byte transfer"),
        # Very large data
        ([0, 1, 2, 3], 1024*1024*1024, "dp_allreduce", "1GB transfer"),
        # Unusual group sizes
        ([0, 1, 2], 1048576, "tp_allreduce", "3-GPU group"),
        ([i for i in range(5)], 4194304, "dp_allreduce", "5-GPU group"),
    ]
    
    print("Testing edge cases:")
    print("-" * 60)
    
    for comm_group, data_size, comm_func, description in edge_cases:
        try:
            time_ms = get_comm_op_exc_time(comm_group, data_size, comm_func)
            print(f"{description:<20}: {time_ms:>8.2f} ms")
            
            if len(comm_group) == 1:
                assert time_ms == 0.01, "Single GPU should return 0.01 ms"
            else:
                assert time_ms > 0, f"Multi-GPU time should be positive for {description}"
                
        except Exception as e:
            print(f"{description:<20}: ERROR - {e}")
            return False
    
    return True

def test_backward_compatibility():
    """Test that existing code patterns still work."""
    print("\n=== Backward Compatibility Test ===")
    
    # Test the old-style calls that simu_engine might make
    try:
        # Original function signature should still work
        result1 = get_comm_op_exc_time([0, 1, 2, 3], 1048576, "allreduce")
        result2 = get_comm_op_exc_time([0, 1], 2048, "send_recv")
        
        print(f"4-GPU allreduce (1MB): {result1:.2f} ms")
        print(f"P2P send_recv (2KB): {result2:.2f} ms")
        
        assert result1 > 0 and result2 > 0, "Results should be positive"
        print("✓ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility failed: {e}")
        return False

def main():
    """Run integration tests."""
    print("Communication Module Integration Tests")
    print("=" * 60)
    
    # Validate database first
    if not validate_comm_database():
        print("✗ Database validation failed")
        return 1
    
    print("✓ Database validation passed")
    
    # Run all tests
    tests = [
        ("Integration with simu_engine", test_integration_with_simu_engine),
        ("MOE scenarios", test_moe_scenarios),
        ("Edge cases", test_edge_cases),
        ("Backward compatibility", test_backward_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} passed")
                passed += 1
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Integration test results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The improved communication module")
        print("   is ready for production use with simu_engine.")
        return 0
    else:
        print("❌ Some integration tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
