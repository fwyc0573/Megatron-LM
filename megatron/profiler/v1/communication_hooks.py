"""
Non-invasive communication operation interceptors.
Automatically intercepts torch.distributed calls without modifying external libraries.
"""
import functools
import torch
import torch.distributed as dist
from typing import Optional, Any
from megatron.core import parallel_state
from .profiler_manager import profiler
from .context_tracker import context_tracker


class CommunicationHook:
    """Hooks into torch.distributed operations for automatic profiling."""
    
    def __init__(self):
        self.original_functions = {}
        self.installed = False
        
    def install(self):
        """Install hooks for communication operations."""
        if self.installed:
            return
            
        # Hook all_reduce operations
        self._hook_function(dist, 'all_reduce', self._trace_all_reduce)
        self._hook_function(dist, 'send', self._trace_send)
        self._hook_function(dist, 'recv', self._trace_recv)
        self._hook_function(dist, '_all_gather_base', self._trace_all_gather)
        self._hook_function(dist, '_reduce_scatter_base', self._trace_reduce_scatter)
        
        self.installed = True
        
    def uninstall(self):
        """Restore original functions."""
        if not self.installed:
            return
            
        for module, func_name in self.original_functions:
            setattr(module, func_name, self.original_functions[(module, func_name)])
            
        self.original_functions.clear()
        self.installed = False
        
    def _hook_function(self, module: Any, func_name: str, wrapper_func):
        """Replace module function with wrapped version."""
        original_func = getattr(module, func_name)
        self.original_functions[(module, func_name)] = original_func
        
        @functools.wraps(original_func)
        def wrapped(*args, **kwargs):
            return wrapper_func(original_func, *args, **kwargs)
            
        setattr(module, func_name, wrapped)
        
    def _trace_all_reduce(self, original_func, tensor, op=dist.ReduceOp.SUM, 
                         group=None, async_op=False):
        """Trace all_reduce operations."""
        group_kind = self._infer_group_kind(group)
        
        with profiler.trace_operation(
            op_name="allreduce",
            group_kind=group_kind,
            tensor_shape=list(tensor.shape),
            tensor_dtype=tensor.dtype,
            description=f"all_reduce_{group_kind}"
        ):
            return original_func(tensor, op, group, async_op)
            
    def _trace_send(self, original_func, tensor, dst, group=None, tag=0):
        """Trace send operations."""
        with profiler.trace_operation(
            op_name="send_forward" if self._is_forward_context() else "send_backward",
            group_kind="pp",
            tensor_shape=list(tensor.shape),
            tensor_dtype=tensor.dtype,
            description=f"send_to_rank_{dst}"
        ):
            return original_func(tensor, dst, group, tag)
            
    def _trace_recv(self, original_func, tensor, src=None, group=None, tag=0):
        """Trace recv operations."""
        with profiler.trace_operation(
            op_name="recv_forward" if self._is_forward_context() else "recv_backward", 
            group_kind="pp",
            tensor_shape=list(tensor.shape),
            tensor_dtype=tensor.dtype,
            description=f"recv_from_rank_{src}"
        ):
            return original_func(tensor, src, group, tag)
            
    def _trace_all_gather(self, original_func, output_tensor, input_tensor, group=None, async_op=False):
        """Trace all_gather operations."""
        group_kind = self._infer_group_kind(group)
        
        with profiler.trace_operation(
            op_name="all_gather",
            group_kind=group_kind,
            tensor_shape=list(input_tensor.shape),
            tensor_dtype=input_tensor.dtype,
            description=f"all_gather_{group_kind}"
        ):
            return original_func(output_tensor, input_tensor, group, async_op)
            
    def _trace_reduce_scatter(self, original_func, output, input_list, group=None, async_op=False):
        """Trace reduce_scatter operations."""
        group_kind = self._infer_group_kind(group)
        
        with profiler.trace_operation(
            op_name="reduce_scatter",
            group_kind=group_kind,
            tensor_shape=list(input_list.shape) if hasattr(input_list, 'shape') else None,
            tensor_dtype=input_list.dtype if hasattr(input_list, 'dtype') else None,
            description=f"reduce_scatter_{group_kind}"
        ):
            return original_func(output, input_list, group, async_op)
            
    def _infer_group_kind(self, group) -> str:
        """Infer group type from torch.distributed group."""
        if group is None:
            return "world"
            
        # Try to map to known Megatron groups
        
        try:
            if group == parallel_state.get_tensor_model_parallel_group():
                return "tp"
            elif group == parallel_state.get_pipeline_model_parallel_group():
                return "pp"
            elif group == parallel_state.get_data_parallel_group():
                return "dp"
            elif group == parallel_state.get_embedding_group():
                return "ep"
        except:
            pass
            
        return "unknown"
        
    def _is_forward_context(self) -> bool:
        """Determine if we're in forward pass context using improved heuristics."""
        # Use context tracker first
        current_stage = context_tracker.get_current_stage()
        if current_stage != "unknown":
            return True  # During pipeline stages, determine by call stack
            
        # Fallback to call stack inspection
        import inspect
        frame = inspect.currentframe()
        try:
            while frame:
                func_name = frame.f_code.co_name.lower()
                if 'forward' in func_name:
                    return True
                if 'backward' in func_name:
                    return False
                frame = frame.f_back
        finally:
            del frame
        return True
        
    def _get_current_stage(self) -> str:
        """Get current pipeline stage."""
        return context_tracker.get_current_stage()


# Global communication hook instance
comm_hook = CommunicationHook() 