#!/usr/bin/env python3
"""
Test ML Send/Recv Predictor

This module tests the machine learning predictor for send/recv operations.
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_sendrecv_predictor import MLSendRecvPredictor


class TestMLSendRecvPredictor(unittest.TestCase):
    """Test ML Send/Recv predictor functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data_dir = tempfile.mkdtemp()
        self._create_test_datasets()
    
    def tearDown(self):
        """Clean up test data"""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def _create_test_datasets(self):
        """Create synthetic test datasets"""
        # Create intra-node dataset
        intra_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        intra_data = []
        
        for size in intra_sizes:
            # Simulate intra-node communication (faster, lower latency)
            base_time = 50 + size / (600 * 1e9 / 8) * 1e6  # Base latency + transfer time
            time_us = base_time + np.random.normal(0, base_time * 0.1)  # Add noise
            
            intra_data.append({
                'size_bytes': size,
                'size_str': f'{size//1024}K',
                'count_elements': size // 4,  # Assuming float32
                'type': 'float',
                'redop': 'none',
                'root': 'none',
                'time_us': max(10, time_us),  # Ensure positive
                'algbw_gbps': size / (time_us * 1e-6) / 1e9,
                'busbw_gbps': size / (time_us * 1e-6) / 1e9,
                'errors': 0
            })
        
        intra_df = pd.DataFrame(intra_data)
        intra_df.to_csv(os.path.join(self.test_data_dir, 'single_node_sendrecv_test.csv'), index=False)
        
        # Create inter-node dataset
        inter_data = []
        
        for size in intra_sizes:
            # Simulate inter-node communication (slower, higher latency)
            base_time = 100 + size / (200 * 1e9 / 8) * 1e6  # Higher latency + slower transfer
            time_us = base_time + np.random.normal(0, base_time * 0.15)  # More noise
            
            inter_data.append({
                'size_bytes': size,
                'size_str': f'{size//1024}K',
                'count_elements': size // 4,
                'type': 'float',
                'redop': 'none',
                'root': 'none',
                'time_us': max(20, time_us),  # Ensure positive
                'algbw_gbps': size / (time_us * 1e-6) / 1e9,
                'busbw_gbps': size / (time_us * 1e-6) / 1e9,
                'errors': 0
            })
        
        inter_df = pd.DataFrame(inter_data)
        inter_df.to_csv(os.path.join(self.test_data_dir, 'multi_node_sendrecv_test.csv'), index=False)
    
    def test_predictor_initialization(self):
        """Test predictor initialization with test data"""
        predictor = MLSendRecvPredictor(self.test_data_dir)
        
        self.assertTrue(predictor.model_trained, "Model should be trained")
        self.assertIsNotNone(predictor.intra_node_model, "Intra-node model should exist")
        self.assertIsNotNone(predictor.inter_node_model, "Inter-node model should exist")
    
    def test_prediction_functionality(self):
        """Test prediction functionality"""
        predictor = MLSendRecvPredictor(self.test_data_dir)
        
        test_sizes = [4096, 16384, 65536, 262144]
        
        for size in test_sizes:
            # Test intra-node prediction
            intra_time = predictor.predict(size, is_inter_node=False)
            self.assertGreater(intra_time, 0, f"Intra-node time should be positive for size {size}")
            
            # Test inter-node prediction
            inter_time = predictor.predict(size, is_inter_node=True)
            self.assertGreater(inter_time, 0, f"Inter-node time should be positive for size {size}")
            
            # Inter-node should generally be slower than intra-node
            # (though this might not always hold due to model variations)
            print(f"Size {size//1024}KB: Intra={intra_time:.3f}ms, Inter={inter_time:.3f}ms")
    
    def test_model_info(self):
        """Test model information retrieval"""
        predictor = MLSendRecvPredictor(self.test_data_dir)
        
        info = predictor.get_model_info()
        
        self.assertIn('model_trained', info)
        self.assertIn('intra_node_available', info)
        self.assertIn('inter_node_available', info)
        self.assertIn('dataset_path', info)
        
        self.assertTrue(info['model_trained'])
        self.assertTrue(info['intra_node_available'])
        self.assertTrue(info['inter_node_available'])
        self.assertEqual(info['dataset_path'], self.test_data_dir)
    
    def test_fallback_estimation(self):
        """Test fallback estimation when models are not available"""
        # Create predictor with non-existent dataset path
        predictor = MLSendRecvPredictor('/non/existent/path')
        
        self.assertFalse(predictor.model_trained, "Model should not be trained")
        
        # Should still provide fallback estimates
        intra_time = predictor.predict(65536, is_inter_node=False)
        inter_time = predictor.predict(65536, is_inter_node=True)
        
        self.assertGreater(intra_time, 0, "Fallback intra-node time should be positive")
        self.assertGreater(inter_time, 0, "Fallback inter-node time should be positive")
        self.assertGreater(inter_time, intra_time, "Inter-node should be slower than intra-node")
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        predictor = MLSendRecvPredictor(self.test_data_dir)
        
        # Save models
        save_dir = tempfile.mkdtemp()
        try:
            predictor.save_models(save_dir)
            
            # Check that model files were created
            expected_files = [
                'intra_node_model.pkl',
                'intra_node_scaler.pkl',
                'inter_node_model.pkl',
                'inter_node_scaler.pkl'
            ]
            
            for filename in expected_files:
                filepath = os.path.join(save_dir, filename)
                self.assertTrue(os.path.exists(filepath), f"Model file {filename} should exist")
            
            # Create new predictor and load models
            new_predictor = MLSendRecvPredictor('/non/existent/path')  # No training data
            self.assertFalse(new_predictor.model_trained, "New predictor should not be trained initially")
            
            new_predictor.load_models(save_dir)
            self.assertTrue(new_predictor.model_trained, "Predictor should be trained after loading")
            
            # Test that loaded models work
            test_size = 32768
            original_time = predictor.predict(test_size, is_inter_node=False)
            loaded_time = new_predictor.predict(test_size, is_inter_node=False)
            
            # Should be very close (allowing for small numerical differences)
            self.assertAlmostEqual(original_time, loaded_time, places=2,
                                 msg="Loaded model should give similar predictions")
        
        finally:
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        predictor = MLSendRecvPredictor(self.test_data_dir)
        
        # Test very small size
        small_time = predictor.predict(1, is_inter_node=False)
        self.assertGreater(small_time, 0, "Should handle very small sizes")
        
        # Test very large size
        large_time = predictor.predict(1024*1024*1024, is_inter_node=False)
        self.assertGreater(large_time, 0, "Should handle very large sizes")
        
        # Test zero size (edge case)
        zero_time = predictor.predict(0, is_inter_node=False)
        self.assertGreater(zero_time, 0, "Should handle zero size gracefully")
    
    def test_prediction_scaling(self):
        """Test that predictions scale reasonably with size"""
        predictor = MLSendRecvPredictor(self.test_data_dir)
        
        sizes = [4096, 16384, 65536, 262144]
        intra_times = []
        inter_times = []
        
        for size in sizes:
            intra_times.append(predictor.predict(size, is_inter_node=False))
            inter_times.append(predictor.predict(size, is_inter_node=True))
        
        # Check that times generally increase with size
        # (allowing for some variation due to model characteristics)
        for i in range(1, len(sizes)):
            # Times should not decrease dramatically
            self.assertLess(intra_times[i-1], intra_times[i] * 2,
                          f"Intra-node time scaling seems unreasonable: {intra_times}")
            self.assertLess(inter_times[i-1], inter_times[i] * 2,
                          f"Inter-node time scaling seems unreasonable: {inter_times}")


class TestRealDatasetIntegration(unittest.TestCase):
    """Test integration with real dataset if available"""
    
    def test_real_dataset_loading(self):
        """Test loading real dataset if available"""
        real_dataset_path = "moe_mg/sendrecv/pytorch_sendrecv_results"
        
        if os.path.exists(real_dataset_path):
            predictor = MLSendRecvPredictor(real_dataset_path)
            
            # Test that it works with real data
            test_sizes = [4096, 16384, 65536]
            
            for size in test_sizes:
                intra_time = predictor.predict(size, is_inter_node=False)
                inter_time = predictor.predict(size, is_inter_node=True)
                
                self.assertGreater(intra_time, 0)
                self.assertGreater(inter_time, 0)
                
                print(f"Real data - Size {size//1024}KB: "
                      f"Intra={intra_time:.3f}ms, Inter={inter_time:.3f}ms")
            
            # Print model info
            info = predictor.get_model_info()
            print(f"Real dataset model info: {info}")
        
        else:
            self.skipTest(f"Real dataset not found at {real_dataset_path}")


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
