"""
Megatron Profiler Package
Provides unified, non-invasive profiling capabilities.
"""

from .v1.profiler_manager import ProfilerManager, profiler
from .v1.communication_hooks import CommunicationHook, comm_hook  
from .v1.decorators import trace_operation as trace_decorator
from .v1.context_tracker import ContextTracker, context_tracker
from .comm_utils.interception_comm import allreduce_wrapper, broadcast_wrapper, reduce_wrapper

# Export main interfaces
__all__ = [
    'profiler',           # Global profiler manager instance
    'comm_hook',          # Global communication hook instance  
    'trace_decorator',    # Decorator for manual tracing
    'context_tracker',   # Pipeline context tracking
    'allreduce_wrapper', # Interception allreduce wrapper
    'broadcast_wrapper', # Interception broadcast wrapper
    'reduce_wrapper', # Interception reduce wrapper
]

def initialize_profiling(rank_id: str, stage_id: int, enabled: bool = True):
    """Convenience function to initialize all profiling components."""
    profiler.initialize(rank_id, stage_id, enabled)
    context_tracker.initialize()
    
def install_hooks():
    """Install all automatic hooks."""
    if profiler.enabled:
        comm_hook.install()
        
def uninstall_hooks():
    """Uninstall all hooks and export traces."""
    comm_hook.uninstall()
    profiler.export_traces() 