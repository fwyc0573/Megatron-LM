#!/usr/bin/env python3
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CCEstimatorWrapper:
    def __init__(self, config):
        self.config = config
        self.predictor = None
        self.ml_predictor = None
        self.cache = {}
        self._initialize_cc_estimator()
    
    def _initialize_cc_estimator(self):
        try:
            # Add CC-estimator to Python path
            import sys
            import os
            cc_estimator_path = os.path.join(os.path.dirname(__file__), 'CC-estimator')
            if cc_estimator_path not in sys.path:
                sys.path.insert(0, cc_estimator_path)

            # Try to import CC-estimator
            from nccl_predictor import create_h800_sxm_ib_predictor, create_a100_sxm_ib_predictor
            
            hardware_type = self.config.hardware.gpu_type.value
            if hardware_type == 'H800-SXM':
                self.predictor = create_h800_sxm_ib_predictor()
            elif hardware_type == 'A100-SXM':
                self.predictor = create_a100_sxm_ib_predictor()
            else:
                self.predictor = create_a100_sxm_ib_predictor()
                
        except ImportError as e:
            logger.warning(f"CC-estimator not available: {e}")
            self.predictor = None
    
    def predict_communication_time(self, comm_group: List[int], data_size: int, comm_func: str) -> float:
        cache_key = (tuple(sorted(comm_group)), data_size, comm_func)
        
        if self.config.communication.cache_predictions and cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.predictor is not None:
            try:
                if comm_func in ['dp_allreduce', 'tp_allreduce', 'ep_allreduce', 'exp_dp_allreduce']:
                    time_ms = self.predictor.predict_allreduce(comm_group, data_size)
                elif comm_func in ['exp_allgather']:
                    time_ms = self.predictor.predict_allgather(comm_group, data_size)
                elif comm_func in ['send_forward', 'send_backward', 'recv_forward', 'recv_backward']:
                    time_ms = self.predictor.predict_p2p_send(comm_group, data_size)
                else:
                    time_ms = self.predictor.predict_allreduce(comm_group, data_size)
            except Exception as e:
                logger.warning(f"CC-estimator prediction failed: {e}")
                time_ms = self._fallback_estimate(comm_group, data_size, comm_func)
        else:
            time_ms = self._fallback_estimate(comm_group, data_size, comm_func)
        
        if self.config.communication.cache_predictions:
            if len(self.cache) < self.config.communication.cache_size_limit:
                self.cache[cache_key] = time_ms
        
        return time_ms
    
    def _fallback_estimate(self, comm_group: List[int], data_size: int, comm_func: str) -> float:
        try:
            from src.core.comm_sim.nccl_comm import get_comm_op_exc_time
            return get_comm_op_exc_time(comm_group, data_size, comm_func)
        except Exception:
            # Simple bandwidth-based fallback
            if len(comm_group) <= 2:
                bandwidth_gbps = 600.0  # NVLink
                latency_us = 2.0
            else:
                bandwidth_gbps = 200.0  # InfiniBand
                latency_us = 10.0
            
            bandwidth_bytes_per_us = bandwidth_gbps * 1e9 / 8 / 1e6
            transfer_time_us = data_size / bandwidth_bytes_per_us
            total_time_us = latency_us + transfer_time_us
            
            return total_time_us / 1000.0  # Convert to milliseconds
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            'cache_size': len(self.cache),
            'cache_limit': self.config.communication.cache_size_limit,
            'cache_enabled': self.config.communication.cache_predictions
        }

_global_estimator: Optional[CCEstimatorWrapper] = None

def initialize_cc_estimator(config) -> CCEstimatorWrapper:
    global _global_estimator
    _global_estimator = CCEstimatorWrapper(config)
    return _global_estimator

def get_comm_op_exc_time(comm_group: List[int], data_size: int, comm_func: str) -> float:
    if _global_estimator is None:
        from src.core.simulator_config import get_default_config
        initialize_cc_estimator(get_default_config())
    
    return _global_estimator.predict_communication_time(comm_group, data_size, comm_func)
