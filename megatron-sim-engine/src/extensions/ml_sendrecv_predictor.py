#!/usr/bin/env python3
import os
import logging

logger = logging.getLogger(__name__)

class MLSendRecvPredictor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.intra_node_model = None
        self.inter_node_model = None
        self.model_trained = False
        
        try:
            self._load_and_train_models()
        except Exception as e:
            logger.warning(f"Failed to train ML models: {e}")
            self.model_trained = False
    
    def _load_and_train_models(self):
        try:
            import pandas as pd
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # Load datasets
            intra_file = os.path.join(self.dataset_path, 'single_node_sendrecv_20250818_151722.csv')
            inter_file = os.path.join(self.dataset_path, 'multi_node_sendrecv_20250818_150945.csv')
            
            if os.path.exists(intra_file):
                intra_data = pd.read_csv(intra_file)
                self._train_model(intra_data, 'intra_node')
            
            if os.path.exists(inter_file):
                inter_data = pd.read_csv(inter_file)
                self._train_model(inter_data, 'inter_node')
            
            self.model_trained = (self.intra_node_model is not None or 
                                self.inter_node_model is not None)
            
        except ImportError:
            logger.warning("Required ML libraries not available")
            self.model_trained = False
    
    def _train_model(self, data, model_type):
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            X = np.log10(data['size_bytes'].values + 1).reshape(-1, 1)
            y = data['time_us'].values
            
            valid_mask = (X.flatten() > 0) & (y > 0) & np.isfinite(X.flatten()) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 2:
                return
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            if model_type == 'intra_node':
                self.intra_node_model = model
                self.intra_node_scaler = scaler
            else:
                self.inter_node_model = model
                self.inter_node_scaler = scaler
                
        except Exception as e:
            logger.warning(f"Failed to train {model_type} model: {e}")
    
    def predict(self, data_size_bytes: int, is_inter_node: bool = False) -> float:
        if not self.model_trained:
            return self._fallback_estimate(data_size_bytes, is_inter_node)
        
        try:
            if is_inter_node and self.inter_node_model is not None:
                model = self.inter_node_model
                scaler = self.inter_node_scaler
            elif self.intra_node_model is not None:
                model = self.intra_node_model
                scaler = self.intra_node_scaler
            else:
                return self._fallback_estimate(data_size_bytes, is_inter_node)
            
            import numpy as np
            log_size = np.log10(data_size_bytes + 1)
            X = np.array([[log_size]])
            X_scaled = scaler.transform(X)
            
            time_us = model.predict(X_scaled)[0]
            time_ms = max(0.001, time_us / 1000.0)
            
            return time_ms
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return self._fallback_estimate(data_size_bytes, is_inter_node)
    
    def _fallback_estimate(self, data_size_bytes: int, is_inter_node: bool) -> float:
        if is_inter_node:
            bandwidth_gbps = 200.0
            latency_us = 10.0
        else:
            bandwidth_gbps = 600.0
            latency_us = 2.0
        
        bandwidth_bytes_per_us = bandwidth_gbps * 1e9 / 8 / 1e6
        transfer_time_us = data_size_bytes / bandwidth_bytes_per_us
        total_time_us = latency_us + transfer_time_us
        
        return total_time_us / 1000.0
    
    def get_model_info(self):
        return {
            'model_trained': self.model_trained,
            'intra_node_available': self.intra_node_model is not None,
            'inter_node_available': self.inter_node_model is not None,
            'dataset_path': self.dataset_path
        }
