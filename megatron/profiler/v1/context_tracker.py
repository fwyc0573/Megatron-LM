"""
Pipeline context tracker for accurate stage detection.
"""
import threading
from typing import Optional
from enum import Enum

class PipelineStage(Enum):
    WARMUP = "warmup"
    STEADY = "steady" 
    COOLDOWN = "cooldown"
    FINALIZE = "finalize"
    HELP = "help"
    UNKNOWN = "unknown"

class ContextTracker:
    """Track current pipeline execution context."""
    
    def __init__(self):
        self.current_stage = PipelineStage.UNKNOWN
        self.current_iteration = 0
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize context tracker."""
        self.current_stage = PipelineStage.UNKNOWN
        self.current_iteration = 0
        
    def set_stage(self, stage: PipelineStage):
        """Set current pipeline stage."""
        with self.lock:
            self.current_stage = stage
            
    def set_iteration(self, iteration: int):
        """Set current iteration."""
        with self.lock:
            self.current_iteration = iteration
            
    def get_current_stage(self) -> str:
        """Get current stage as string."""
        with self.lock:
            return self.current_stage.value
            
    def get_current_iteration(self) -> int:
        """Get current iteration."""
        with self.lock:
            return self.current_iteration

# Global context tracker
context_tracker = ContextTracker() 