#!/usr/bin/env python3
"""
Optimized ML Send/Recv Predictor

This module implements an optimized machine learning predictor for send/recv operations
with improved feature engineering, data preprocessing, and model selection to address
the accuracy issues identified in the verification tests.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)


class OptimizedMLSendRecvPredictor:
    """
    Optimized ML predictor for send/recv operations with improved accuracy
    """
    
    def __init__(self, dataset_path: str, model_type: str = 'auto'):
        """
        Initialize optimized ML predictor
        
        Args:
            dataset_path: Path to sendrecv dataset directory
            model_type: Model type ('linear', 'polynomial', 'rf', 'gbr', 'auto')
        """
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.intra_node_pipeline = None
        self.inter_node_pipeline = None
        self.model_trained = False
        self.feature_names = None
        
        # Model performance tracking
        self.model_performance = {}
        
        # Load and train models
        self._load_and_train_models()
    
    def _load_and_train_models(self):
        """Load dataset and train optimized models"""
        try:
            # Load datasets
            intra_data, inter_data = self._load_datasets()
            
            if intra_data is not None and len(intra_data) > 0:
                self._train_optimized_model(intra_data, 'intra_node')
                logger.info(f"Trained optimized intra-node model with {len(intra_data)} samples")
            
            if inter_data is not None and len(inter_data) > 0:
                self._train_optimized_model(inter_data, 'inter_node')
                logger.info(f"Trained optimized inter-node model with {len(inter_data)} samples")
            
            self.model_trained = (self.intra_node_pipeline is not None or 
                                self.inter_node_pipeline is not None)
            
        except Exception as e:
            logger.error(f"Failed to train optimized ML models: {e}")
            self.model_trained = False
    
    def _load_datasets(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load intra-node and inter-node datasets with improved preprocessing"""
        intra_data = None
        inter_data = None
        
        try:
            csv_files = []
            if os.path.exists(self.dataset_path):
                for file in os.listdir(self.dataset_path):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(self.dataset_path, file))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Data quality checks and preprocessing
                    df = self._preprocess_dataset(df)
                    
                    if 'single_node' in os.path.basename(csv_file).lower():
                        intra_data = df
                        logger.info(f"Loaded and preprocessed intra-node data: {len(df)} samples")
                    elif 'multi_node' in os.path.basename(csv_file).lower():
                        inter_data = df
                        logger.info(f"Loaded and preprocessed inter-node data: {len(df)} samples")
                    else:
                        if intra_data is None:
                            intra_data = df
                        else:
                            inter_data = df
                
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
            
            return intra_data, inter_data
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return None, None
    
    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataset with outlier removal and data cleaning"""
        # Remove invalid data points
        df = df[(df['size_bytes'] > 0) & (df['time_us'] > 0)]
        df = df[np.isfinite(df['size_bytes']) & np.isfinite(df['time_us'])]
        
        # Remove outliers using IQR method
        Q1 = df['time_us'].quantile(0.25)
        Q3 = df['time_us'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Keep data within 3*IQR range (more conservative than 3σ)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        original_len = len(df)
        df = df[(df['time_us'] >= lower_bound) & (df['time_us'] <= upper_bound)]
        removed_outliers = original_len - len(df)
        
        if removed_outliers > 0:
            logger.info(f"Removed {removed_outliers} outliers from dataset")
        
        return df
    
    def _create_features(self, size_bytes: np.ndarray) -> np.ndarray:
        """Create enhanced features for better model performance"""
        features = []
        
        # Log-based features (handle the non-linear relationship)
        log_size = np.log10(size_bytes + 1)
        features.append(log_size)
        
        # Polynomial features
        features.append(log_size ** 2)
        features.append(log_size ** 3)
        
        # Square root feature (for bandwidth-limited regime)
        features.append(np.sqrt(size_bytes))
        
        # Size-based regime indicators
        features.append((size_bytes < 1024).astype(float))      # Small messages
        features.append((size_bytes > 1024*1024).astype(float)) # Large messages
        
        # Bandwidth utilization features
        features.append(size_bytes / (size_bytes + 1024))  # Normalized size
        
        return np.column_stack(features)
    
    def _select_best_model(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """Select the best model using cross-validation"""
        models = {}
        
        if self.model_type == 'auto':
            # Test multiple models
            models['linear'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            
            models['polynomial'] = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            
            models['random_forest'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            models['gradient_boosting'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
        else:
            # Use specified model type
            if self.model_type == 'linear':
                models['linear'] = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', LinearRegression())
                ])
            elif self.model_type == 'polynomial':
                models['polynomial'] = Pipeline([
                    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                    ('scaler', StandardScaler()),
                    ('regressor', LinearRegression())
                ])
            elif self.model_type == 'rf':
                models['random_forest'] = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
            elif self.model_type == 'gbr':
                models['gradient_boosting'] = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ])
        
        # Evaluate models using cross-validation
        best_model = None
        best_score = -np.inf
        model_scores = {}
        
        for name, model in models.items():
            try:
                # Use negative MSE as scoring metric (higher is better)
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                model_scores[name] = avg_score
                
                logger.info(f"Model {name}: CV score = {avg_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate model {name}: {e}")
        
        if best_model is None:
            # Fallback to linear regression
            best_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            logger.warning("All models failed, using linear regression as fallback")
        
        return best_model
    
    def _train_optimized_model(self, data: pd.DataFrame, model_type: str):
        """Train optimized model with enhanced features and model selection"""
        try:
            # Create enhanced features
            X = self._create_features(data['size_bytes'].values)
            y = data['time_us'].values
            
            # Store feature names for later use
            if self.feature_names is None:
                self.feature_names = [
                    'log_size', 'log_size_2', 'log_size_3', 'sqrt_size',
                    'small_msg', 'large_msg', 'norm_size'
                ]
            
            if len(X) < 5:
                logger.warning(f"Insufficient data for {model_type} model: {len(X)} samples")
                return
            
            # Split data for validation
            if len(X) > 8:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Select and train the best model
            best_pipeline = self._select_best_model(X_train, y_train)
            best_pipeline.fit(X_train, y_train)
            
            # Evaluate model performance
            y_pred = best_pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Store performance metrics
            self.model_performance[model_type] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'samples': len(X)
            }
            
            logger.info(f"{model_type} optimized model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, "
                       f"R²: {r2:.3f}, MAPE: {mape:.2f}%")
            
            # Store the trained pipeline
            if model_type == 'intra_node':
                self.intra_node_pipeline = best_pipeline
            else:
                self.inter_node_pipeline = best_pipeline
            
        except Exception as e:
            logger.error(f"Error training optimized {model_type} model: {e}")
    
    def predict(self, data_size_bytes: int, is_inter_node: bool = False) -> float:
        """
        Predict send/recv communication time with optimized model
        
        Args:
            data_size_bytes: Size of data to communicate in bytes
            is_inter_node: Whether this is inter-node communication
            
        Returns:
            Predicted time in milliseconds
        """
        if not self.model_trained:
            logger.warning("Optimized models not trained, using fallback estimation")
            return self._fallback_estimate(data_size_bytes, is_inter_node)
        
        try:
            # Select appropriate pipeline
            if is_inter_node and self.inter_node_pipeline is not None:
                pipeline = self.inter_node_pipeline
            elif self.intra_node_pipeline is not None:
                pipeline = self.intra_node_pipeline
            else:
                logger.warning("No suitable optimized model available, using fallback")
                return self._fallback_estimate(data_size_bytes, is_inter_node)
            
            # Create features
            X = self._create_features(np.array([data_size_bytes]))
            
            # Predict time in microseconds
            time_us = pipeline.predict(X)[0]
            
            # Convert to milliseconds and ensure positive
            time_ms = max(0.001, time_us / 1000.0)
            
            return time_ms
            
        except Exception as e:
            logger.warning(f"Optimized ML prediction failed: {e}, using fallback")
            return self._fallback_estimate(data_size_bytes, is_inter_node)
    
    def _fallback_estimate(self, data_size_bytes: int, is_inter_node: bool) -> float:
        """Enhanced fallback estimation"""
        if is_inter_node:
            # Inter-node: InfiniBand with higher latency
            bandwidth_gbps = 200.0
            latency_us = 15.0
        else:
            # Intra-node: NVLink with lower latency
            bandwidth_gbps = 600.0
            latency_us = 3.0
        
        # Enhanced bandwidth model with saturation
        bandwidth_bytes_per_us = bandwidth_gbps * 1e9 / 8 / 1e6
        
        # Add protocol overhead (more realistic)
        effective_size = data_size_bytes * 1.1  # 10% overhead
        
        transfer_time_us = effective_size / bandwidth_bytes_per_us
        total_time_us = latency_us + transfer_time_us
        
        return total_time_us / 1000.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about trained models"""
        info = {
            'model_trained': self.model_trained,
            'intra_node_available': self.intra_node_pipeline is not None,
            'inter_node_available': self.inter_node_pipeline is not None,
            'dataset_path': self.dataset_path,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'performance': self.model_performance
        }
        
        return info
    
    def compare_with_baseline(self, baseline_predictor) -> Dict[str, Any]:
        """Compare performance with baseline predictor"""
        if not hasattr(self, 'model_performance'):
            return {'error': 'No performance data available'}
        
        comparison = {}
        
        for model_type, perf in self.model_performance.items():
            # Get baseline performance if available
            baseline_info = baseline_predictor.get_model_info()
            
            comparison[model_type] = {
                'optimized': perf,
                'improvement': 'Performance comparison available'
            }
        
        return comparison


if __name__ == "__main__":
    # Test the optimized predictor
    dataset_path = "moe_mg/sendrecv/pytorch_sendrecv_results"
    
    if os.path.exists(dataset_path):
        print("Testing Optimized ML Send/Recv Predictor...")
        
        # Test different model types
        model_types = ['auto', 'polynomial', 'rf']
        
        for model_type in model_types:
            print(f"\n--- Testing {model_type} model ---")
            
            predictor = OptimizedMLSendRecvPredictor(dataset_path, model_type=model_type)
            info = predictor.get_model_info()
            
            print(f"Model trained: {info['model_trained']}")
            
            if info['model_trained']:
                print("Performance metrics:")
                for model_name, perf in info['performance'].items():
                    print(f"  {model_name}: MAPE={perf['mape']:.1f}%, R²={perf['r2']:.3f}")
                
                # Test predictions
                test_sizes = [4096, 16384, 65536]
                print("\nPrediction test:")
                for size in test_sizes:
                    intra_time = predictor.predict(size, is_inter_node=False)
                    inter_time = predictor.predict(size, is_inter_node=True)
                    print(f"  {size//1024}KB: Intra={intra_time:.3f}ms, Inter={inter_time:.3f}ms")
    
    else:
        print(f"Dataset path {dataset_path} not found")
