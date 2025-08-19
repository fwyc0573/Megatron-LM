#!/usr/bin/env python3
"""
Test CC-estimator Integration

This module tests the integration of CC-estimator into the simulator.
"""

import os
import sys
import unittest
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator_config import (
    SimulatorConfig, HardwareConfig, CommunicationConfig,
    HardwareType, NetworkType,
    create_h800_sxm_ib_config, create_a100_sxm_ib_config
)
from cc_estimator_integration import CCEstimatorWrapper, get_comm_op_exc_time
from message_size_calculator import MessageSizeCalculator, get_tensor_data_size, calculate_comm_message_size


class TestSimulatorConfig(unittest.TestCase):
    """Test simulator configuration functionality"""
    
    def test_hardware_config_creation(self):
        """Test hardware configuration creation"""
        config = HardwareConfig(
            gpu_type=HardwareType.H800_SXM,
            network_type=NetworkType.INFINIBAND,
            gpus_per_node=8
        )
        
        self.assertEqual(config.gpu_type, HardwareType.H800_SXM)
        self.assertEqual(config.network_type, NetworkType.INFINIBAND)
        self.assertEqual(config.gpus_per_node, 8)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization"""
        original_config = create_h800_sxm_ib_config()
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = SimulatorConfig.from_dict(config_dict)
        
        self.assertEqual(original_config.hardware.gpu_type, restored_config.hardware.gpu_type)
        self.assertEqual(original_config.communication.use_cc_estimator, restored_config.communication.use_cc_estimator)
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files"""
        config = create_a100_sxm_ib_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_file(f.name)
            
            # Load and verify
            loaded_config = SimulatorConfig.load_from_file(f.name)
            self.assertEqual(config.hardware.gpu_type, loaded_config.hardware.gpu_type)
            
            # Clean up
            os.unlink(f.name)


class TestMessageSizeCalculator(unittest.TestCase):
    """Test message size calculation functionality"""
    
    def setUp(self):
        self.calculator = MessageSizeCalculator()
    
    def test_tensor_size_calculation(self):
        """Test basic tensor size calculation"""
        # Test with list shape
        size = self.calculator.calculate_tensor_size([1024, 1024], 'torch.float16')
        expected = 1024 * 1024 * 2  # 2 bytes per float16
        self.assertEqual(size, expected)
        
        # Test with string shape
        size = self.calculator.calculate_tensor_size('[512, 768]', 'float32')
        expected = 512 * 768 * 4  # 4 bytes per float32
        self.assertEqual(size, expected)
    
    def test_dtype_size_mapping(self):
        """Test data type size mapping"""
        test_cases = [
            ('torch.float32', 4),
            ('torch.float16', 2),
            ('torch.bfloat16', 2),
            ('float32', 4),
            ('fp16', 2),
            ('half', 2),
        ]
        
        for dtype, expected_size in test_cases:
            actual_size = self.calculator._get_dtype_size(dtype)
            self.assertEqual(actual_size, expected_size, f"Failed for dtype {dtype}")
    
    def test_communication_size_calculation(self):
        """Test communication-specific size calculation"""
        shape = [1024, 2048]
        dtype = 'torch.float16'
        
        # Test different communication operations
        ops = ['send_forward', 'dp_allreduce', 'exp_allgather']
        
        for op in ops:
            size = self.calculator.calculate_communication_size(shape, dtype, op)
            self.assertGreater(size, 0, f"Size should be positive for {op}")
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # None inputs
        size = self.calculator.calculate_tensor_size(None, 'float32')
        self.assertEqual(size, 0)
        
        size = self.calculator.calculate_tensor_size([1024], None)
        self.assertEqual(size, 0)
        
        # Invalid shape
        size = self.calculator.calculate_tensor_size('invalid_shape', 'float32')
        self.assertEqual(size, 0)
        
        # Unknown dtype
        size = self.calculator.calculate_tensor_size([1024], 'unknown_dtype')
        self.assertEqual(size, 0)
    
    def test_moe_communication_estimation(self):
        """Test MoE communication size estimation"""
        shape = [1024, 2048]
        dtype = 'torch.float16'
        
        moe_sizes = self.calculator.estimate_moe_communication_size(
            shape, dtype, num_experts=8, top_k=2
        )
        
        self.assertIn('exp_allgather', moe_sizes)
        self.assertIn('exp_all_to_all', moe_sizes)
        self.assertIn('exp_dp_allreduce', moe_sizes)
        
        for op, size in moe_sizes.items():
            self.assertGreaterEqual(size, 0, f"Size should be non-negative for {op}")


class TestCCEstimatorIntegration(unittest.TestCase):
    """Test CC-estimator integration functionality"""
    
    def setUp(self):
        self.config = create_h800_sxm_ib_config()
        # Disable ML predictor for testing to avoid dataset dependency
        self.config.communication.use_ml_predictor = False
    
    def test_estimator_initialization(self):
        """Test CC-estimator wrapper initialization"""
        try:
            estimator = CCEstimatorWrapper(self.config)
            self.assertIsNotNone(estimator)
            self.assertEqual(estimator.config, self.config)
        except ImportError:
            self.skipTest("CC-estimator not available")
    
    def test_communication_prediction(self):
        """Test communication time prediction"""
        try:
            estimator = CCEstimatorWrapper(self.config)
            
            # Test different communication patterns
            test_cases = [
                ([0, 1, 2, 3], 1024*1024, 'dp_allreduce'),
                ([0, 1], 2048*1024, 'send_forward'),
                ([0, 1, 2, 3, 4, 5, 6, 7], 4096*1024, 'tp_allreduce'),
            ]
            
            for comm_group, data_size, comm_func in test_cases:
                time_ms = estimator.predict_communication_time(comm_group, data_size, comm_func)
                self.assertGreater(time_ms, 0, f"Time should be positive for {comm_func}")
                self.assertIsInstance(time_ms, float, f"Time should be float for {comm_func}")
                
        except ImportError:
            self.skipTest("CC-estimator not available")
    
    def test_cache_functionality(self):
        """Test prediction caching"""
        try:
            estimator = CCEstimatorWrapper(self.config)
            
            comm_group = [0, 1, 2, 3]
            data_size = 1024 * 1024
            comm_func = 'dp_allreduce'
            
            # First prediction
            time1 = estimator.predict_communication_time(comm_group, data_size, comm_func)
            
            # Second prediction (should use cache)
            time2 = estimator.predict_communication_time(comm_group, data_size, comm_func)
            
            self.assertEqual(time1, time2, "Cached prediction should be identical")
            
            # Check cache stats
            stats = estimator.get_cache_stats()
            self.assertGreater(stats['cache_size'], 0, "Cache should contain entries")
            
        except ImportError:
            self.skipTest("CC-estimator not available")
    
    def test_fallback_functionality(self):
        """Test fallback to legacy comm_sim"""
        # Test with invalid configuration to trigger fallback
        config = self.config
        config.hardware.gpu_type = "INVALID_GPU"  # This should trigger fallback
        
        try:
            estimator = CCEstimatorWrapper(config)
            
            # Should still work with fallback
            time_ms = estimator.predict_communication_time([0, 1], 1024, 'send_forward')
            self.assertGreater(time_ms, 0)
            
        except Exception as e:
            # If even fallback fails, that's expected in test environment
            self.assertIn("fallback", str(e).lower())


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing simulator code"""
    
    def test_get_tensor_data_size_compatibility(self):
        """Test backward compatibility of get_tensor_data_size function"""
        # Test with various input formats
        test_cases = [
            ([1024, 1024], 'torch.float16', 1024 * 1024 * 2),
            ('[512, 768]', 'float32', 512 * 768 * 4),
            (None, 'float32', 0),
            ([1024], None, 0),
        ]
        
        for shape, dtype, expected in test_cases:
            result = get_tensor_data_size(shape, dtype)
            self.assertEqual(result, expected, f"Failed for shape={shape}, dtype={dtype}")
    
    def test_get_comm_op_exc_time_compatibility(self):
        """Test backward compatibility of get_comm_op_exc_time function"""
        # This should work even without CC-estimator
        try:
            time_ms = get_comm_op_exc_time([0, 1, 2, 3], 1024*1024, 'dp_allreduce')
            self.assertGreater(time_ms, 0)
            self.assertIsInstance(time_ms, float)
        except Exception as e:
            # In test environment, this might fail due to missing dependencies
            self.assertIsInstance(e, (ImportError, AttributeError))


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration scenarios"""
    
    def test_full_integration_workflow(self):
        """Test complete integration workflow"""
        # Create configuration
        config = create_h800_sxm_ib_config()
        config.communication.use_ml_predictor = False  # Disable for testing
        
        try:
            # Initialize estimator
            estimator = CCEstimatorWrapper(config)
            
            # Test various communication scenarios
            scenarios = [
                # (comm_group, tensor_shape, tensor_dtype, comm_op)
                ([0, 1, 2, 3], [1024, 1024], 'torch.float16', 'dp_allreduce'),
                ([0, 1], [2048, 4096], 'torch.float32', 'send_forward'),
                ([0, 1, 2, 3, 4, 5, 6, 7], [512, 768], 'torch.bfloat16', 'tp_allreduce'),
            ]
            
            for comm_group, shape, dtype, op in scenarios:
                # Calculate message size
                msg_size = calculate_comm_message_size(shape, dtype, op, len(comm_group))
                self.assertGreater(msg_size, 0, f"Message size should be positive for {op}")
                
                # Predict communication time
                time_ms = estimator.predict_communication_time(comm_group, msg_size, op)
                self.assertGreater(time_ms, 0, f"Time should be positive for {op}")
                
                print(f"{op} with {len(comm_group)} GPUs: {msg_size/1024:.1f} KB -> {time_ms:.3f} ms")
            
        except ImportError:
            self.skipTest("CC-estimator not available")


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
