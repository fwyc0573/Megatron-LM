"""
Simple decorators for manual operation tracing.
"""
import functools
from typing import Optional, Dict, Any
from .profiler_manager import profiler
import torch

def trace_operation(op_name: str, 
                   stage: str = "unknown",
                   group_kind: str = "compute", 
                   description: str = None):
    """
    Decorator for tracing function execution.
    
    Args:
        op_name: Operation name
        stage: Pipeline stage
        group_kind: Operation group kind
        description: Additional description
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract tensor information from arguments if available
            tensor_shape, tensor_dtype = _extract_tensor_info(args, kwargs)
            
            with profiler.trace_operation(
                op_name=op_name,
                stage=stage, 
                group_kind=group_kind,
                tensor_shape=tensor_shape,
                tensor_dtype=tensor_dtype,
                description=description or func.__name__
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def _extract_tensor_info(args, kwargs):
    """Extract tensor shape and dtype from function arguments."""
    # Look for tensor in args
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return list(arg.shape), arg.dtype
            
    # Look for tensor in kwargs
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            return list(value.shape), value.dtype
            
    return None, None 