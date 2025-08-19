#!/usr/bin/env python3
"""
End-to-End Integration Test

This module tests the complete integration of CC-estimator with the simulator,
including configuration, message size calculation, and communication prediction.
"""

import os
import sys
import unittest
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator_config import (
    create_h800_sxm_ib_config, create_a100_sxm_ib_config,
    load_config_from_env, get_default_config
)
from cc_estimator_integration import CCEstimatorWrapper, initialize_cc_estimator
from message_size_calculator import MessageSizeCalculator, calculate_comm_message_size
from ml_sendrecv_predictor import MLSendRecvPredictor


class TestEndToEndIntegration(unittest.TestCase):
    """Test complete end-to-end integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_configuration_workflow(self):
        """Test complete configuration workflow"""
        # Test default configuration
        default_config = get_default_config()
        self.assertIsNotNone(default_config)
        
        # Test H800 configuration
        h800_config = create_h800_sxm_ib_config()
        self.assertEqual(h800_config.hardware.gpu_type.value, "H800-SXM")
        self.assertTrue(h800_config.communication.use_cc_estimator)
        self.assertTrue(h800_config.communication.use_ml_predictor)
        
        # Test A100 configuration
        a100_config = create_a100_sxm_ib_config()
        self.assertEqual(a100_config.hardware.gpu_type.value, "A100-SXM")
        self.assertTrue(a100_config.communication.use_cc_estimator)
        self.assertFalse(a100_config.communication.use_ml_predictor)
        
        # Test configuration file operations
        config_file = os.path.join(self.temp_dir, 'test_config.json')
        h800_config.save_to_file(config_file)
        
        loaded_config = type(h800_config).load_from_file(config_file)
        self.assertEqual(h800_config.hardware.gpu_type, loaded_config.hardware.gpu_type)
        
        # Test environment-based configuration loading
        os.environ['SIMULATOR_CONFIG_PATH'] = config_file
        env_config = load_config_from_env()
        self.assertEqual(env_config.hardware.gpu_type.value, "H800-SXM")
    
    def test_message_size_calculation_integration(self):
        """Test message size calculation integration"""
        calculator = MessageSizeCalculator()
        
        # Test various tensor configurations
        test_cases = [
            # (shape, dtype, expected_size_range)
            ([1024, 1024], 'torch.float16', (2*1024*1024 - 100, 2*1024*1024 + 100)),
            ([2048, 4096], 'torch.float32', (4*2048*4096 - 100, 4*2048*4096 + 100)),
            ('[512, 768, 1024]', 'torch.bfloat16', (2*512*768*1024 - 100, 2*512*768*1024 + 100)),
        ]
        
        for shape, dtype, (min_size, max_size) in test_cases:
            size = calculator.calculate_tensor_size(shape, dtype)
            self.assertGreaterEqual(size, min_size, f"Size too small for {shape}, {dtype}")
            self.assertLessEqual(size, max_size, f"Size too large for {shape}, {dtype}")
        
        # Test communication-specific size calculation
        comm_ops = ['send_forward', 'dp_allreduce', 'tp_allreduce', 'exp_allgather']
        
        for op in comm_ops:
            size = calculate_comm_message_size([1024, 2048], 'torch.float16', op, 8)
            self.assertGreater(size, 0, f"Communication size should be positive for {op}")
        
        # Test MoE communication estimation
        moe_sizes = calculator.estimate_moe_communication_size(
            [1024, 2048], 'torch.float16', num_experts=8, top_k=2
        )
        
        for op, size in moe_sizes.items():
            self.assertGreaterEqual(size, 0, f"MoE size should be non-negative for {op}")
    
    def test_cc_estimator_integration(self):
        """Test CC-estimator integration"""
        try:
            # Test with H800 configuration
            h800_config = create_h800_sxm_ib_config()
            h800_config.communication.use_ml_predictor = False  # Disable for testing
            
            estimator = CCEstimatorWrapper(h800_config)
            
            # Test various communication scenarios
            scenarios = [
                # (comm_group, tensor_shape, tensor_dtype, comm_op)
                ([0, 1, 2, 3], [1024, 1024], 'torch.float16', 'dp_allreduce'),
                ([0, 1], [2048, 4096], 'torch.float32', 'send_forward'),
                ([0, 1, 2, 3, 4, 5, 6, 7], [512, 768], 'torch.bfloat16', 'tp_allreduce'),
                ([0, 1, 2, 3, 8, 9, 10, 11], [256, 512], 'torch.float16', 'exp_allgather'),
            ]
            
            for comm_group, shape, dtype, op in scenarios:
                # Calculate message size
                msg_size = calculate_comm_message_size(shape, dtype, op, len(comm_group))
                
                # Predict communication time
                time_ms = estimator.predict_communication_time(comm_group, msg_size, op)
                
                self.assertGreater(time_ms, 0, f"Time should be positive for {op}")
                self.assertIsInstance(time_ms, float, f"Time should be float for {op}")
                
                print(f"{op} with {len(comm_group)} GPUs: "
                      f"{msg_size/1024:.1f} KB -> {time_ms:.3f} ms")
            
            # Test caching
            cache_stats = estimator.get_cache_stats()
            self.assertGreater(cache_stats['cache_size'], 0, "Cache should contain entries")
            
        except ImportError:
            self.skipTest("CC-estimator not available")
    
    def test_ml_predictor_integration(self):
        """Test ML predictor integration"""
        # Create test dataset
        test_data_dir = os.path.join(self.temp_dir, 'sendrecv_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Create minimal test dataset
        import pandas as pd
        import numpy as np
        
        sizes = [2048, 4096, 8192, 16384, 32768]
        
        # Intra-node data
        intra_data = []
        for size in sizes:
            time_us = 50 + size / (600 * 1e9 / 8) * 1e6  # Simple model
            intra_data.append({
                'size_bytes': size,
                'time_us': time_us,
                'algbw_gbps': size / (time_us * 1e-6) / 1e9,
                'busbw_gbps': size / (time_us * 1e-6) / 1e9,
                'errors': 0
            })
        
        pd.DataFrame(intra_data).to_csv(
            os.path.join(test_data_dir, 'single_node_sendrecv_test.csv'), 
            index=False
        )
        
        # Inter-node data
        inter_data = []
        for size in sizes:
            time_us = 100 + size / (200 * 1e9 / 8) * 1e6  # Slower model
            inter_data.append({
                'size_bytes': size,
                'time_us': time_us,
                'algbw_gbps': size / (time_us * 1e-6) / 1e9,
                'busbw_gbps': size / (time_us * 1e-6) / 1e9,
                'errors': 0
            })
        
        pd.DataFrame(inter_data).to_csv(
            os.path.join(test_data_dir, 'multi_node_sendrecv_test.csv'), 
            index=False
        )
        
        # Test ML predictor
        predictor = MLSendRecvPredictor(test_data_dir)
        self.assertTrue(predictor.model_trained, "ML predictor should be trained")
        
        # Test predictions
        test_sizes = [4096, 16384, 65536]
        
        for size in test_sizes:
            intra_time = predictor.predict(size, is_inter_node=False)
            inter_time = predictor.predict(size, is_inter_node=True)
            
            self.assertGreater(intra_time, 0, f"Intra-node time should be positive for {size}")
            self.assertGreater(inter_time, 0, f"Inter-node time should be positive for {size}")
        
        # Test integration with CC-estimator
        config = create_h800_sxm_ib_config()
        config.communication.use_ml_predictor = True
        config.communication.sendrecv_dataset_path = test_data_dir
        
        try:
            estimator = CCEstimatorWrapper(config)
            
            # Test send/recv prediction with ML
            send_time = estimator.predict_communication_time([0, 1], 16384, 'send_forward')
            self.assertGreater(send_time, 0, "Send time should be positive")
            
        except ImportError:
            self.skipTest("CC-estimator not available for ML integration test")
    
    def test_simulator_engine_integration(self):
        """Test integration with SimulatorEngine"""
        try:
            # This would test the actual simulator integration
            # For now, we'll test the key components that would be used
            
            # Test global initialization
            config = create_h800_sxm_ib_config()
            estimator = initialize_cc_estimator(config)
            
            self.assertIsNotNone(estimator, "Global estimator should be initialized")
            
            # Test backward compatibility function
            from cc_estimator_integration import get_comm_op_exc_time
            
            time_ms = get_comm_op_exc_time([0, 1, 2, 3], 1024*1024, 'dp_allreduce')
            self.assertGreater(time_ms, 0, "Backward compatibility function should work")
            
        except ImportError:
            self.skipTest("CC-estimator not available for simulator integration test")
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        # Test with invalid configuration
        config = create_h800_sxm_ib_config()
        config.hardware.gpu_type = "INVALID_GPU"
        
        # Should not crash, should use fallback
        try:
            estimator = CCEstimatorWrapper(config)
            time_ms = estimator.predict_communication_time([0, 1], 1024, 'send_forward')
            self.assertGreater(time_ms, 0, "Fallback should still work")
        except Exception as e:
            # Some failures are expected in test environment
            self.assertIn("fallback", str(e).lower())
        
        # Test with missing dataset path for ML predictor
        config = create_h800_sxm_ib_config()
        config.communication.use_ml_predictor = True
        config.communication.sendrecv_dataset_path = "/non/existent/path"
        
        # Should not crash, should disable ML predictor
        estimator = CCEstimatorWrapper(config)
        self.assertIsNone(estimator.ml_predictor, "ML predictor should be None for invalid path")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the integration"""
        try:
            config = create_h800_sxm_ib_config()
            config.communication.cache_predictions = True
            
            estimator = CCEstimatorWrapper(config)
            
            # Test prediction time
            import time
            
            comm_group = [0, 1, 2, 3]
            data_size = 1024 * 1024
            comm_func = 'dp_allreduce'
            
            # Measure cold cache performance
            start_time = time.perf_counter()
            time_ms = estimator.predict_communication_time(comm_group, data_size, comm_func)
            cold_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Measure warm cache performance
            start_time = time.perf_counter()
            time_ms2 = estimator.predict_communication_time(comm_group, data_size, comm_func)
            warm_time = (time.perf_counter() - start_time) * 1000  # ms
            
            self.assertEqual(time_ms, time_ms2, "Cached result should be identical")
            self.assertLess(warm_time, cold_time, "Cached prediction should be faster")
            
            print(f"Cold cache: {cold_time:.3f} ms, Warm cache: {warm_time:.3f} ms")
            print(f"Cache speedup: {cold_time/warm_time:.1f}x")
            
        except ImportError:
            self.skipTest("CC-estimator not available for performance test")


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Running End-to-End Integration Tests")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)
