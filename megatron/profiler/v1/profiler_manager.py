"""
Unified profiling manager for Megatron operations.
Provides clean, non-invasive profiling capabilities.
"""
import time
import torch
import threading
import inspect
import functools
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from collections import defaultdict


class ProfilerManager:
    """Central profiling manager with minimal code invasion."""
    
    def __init__(self):
        self.enabled = False
        self.traces = defaultdict(list)
        self.batch_counters = defaultdict(int)
        self.lock = threading.Lock()
        self.current_iteration = 0
        self.rank_id = None
        self.stage_id = None
        
        # 新增：支持嵌套子操作追踪
        self.current_operation = None
        self.temp_async_records = {}
        
    def initialize(self, rank_id: str, stage_id: int, enabled: bool = True):
        """Initialize profiler for current rank."""
        self.rank_id = rank_id
        self.stage_id = stage_id
        self.enabled = enabled
        
    def reset_iteration(self, iteration: int):
        """Reset counters for new iteration."""
        with self.lock:
            self.current_iteration = iteration
            self.batch_counters.clear()
            
    def set_current_operation(self, operation):
        """Set current operation for nested sub-operation tracking."""
        with self.lock:
            self.current_operation = operation

    def get_current_operation(self):
        """Get current operation for nested sub-operation tracking."""
        with self.lock:
            return self.current_operation

    def reset_current_operation(self):
        """Reset current operation."""
        with self.lock:
            self.current_operation = None
            
    @contextmanager
    def trace_operation(self, 
                       op_name: str, 
                       stage: str = "unknown",
                       group_kind: str = "unknown",
                       tensor_shape: Optional[List[int]] = None,
                       tensor_dtype: Optional[torch.dtype] = None,
                       description: str = None,
                       track_sub_ops: bool = False):
        """
        Universal operation tracer using context manager.
        
        Args:
            op_name: Operation name (e.g., "send_forward", "backward_step")
            stage: Pipeline stage (e.g., "warmup", "steady", "cooldown")
            group_kind: Communication group (e.g., "pp", "tp", "dp")
            tensor_shape: Tensor shape for communication ops
            tensor_dtype: Tensor data type
            description: Additional description
            track_sub_ops: Whether to track sub-operations for this operation
        """
        if not self.enabled:
            yield
            return
            
        # 创建操作对象
        operation = OperationTracker(
            op_name=op_name,
            stage=stage,
            group_kind=group_kind,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            description=description,
            rank_id=self.rank_id,
            stage_id=self.stage_id
        )
        
        # Increment batch counter
        with self.lock:
            self.batch_counters[op_name] += 1
            operation.batch_id = self.batch_counters[op_name]
            
        # 如果需要追踪子操作，设置为当前操作
        if track_sub_ops:
            self.set_current_operation(operation)
            
        # Start timing
        operation.start_timing()
            
        try:
            yield operation
        finally:
            # End timing
            operation.end_timing()
            
            # Record trace
            trace_entry = operation.create_trace_entry()
            
            with self.lock:
                self.traces[self.rank_id].append(trace_entry)
                
            # 重置当前操作
            if track_sub_ops:
                self.reset_current_operation()

    def add_sub_operation(self, operation_name: str, duration: float, attr_info: Dict[str, Any]):
        """Add sub-operation to current operation."""
        current_op = self.get_current_operation()
        if current_op is not None:
            timestamp = round(time.perf_counter() * 1000, 2)
            sub_operation = f"trace_src_func={operation_name},duration={duration},timestamp={timestamp}"
            if attr_info:
                sub_operation += "," + ",".join(f"{key}={value}" for key, value in attr_info.items())
            current_op.sub_operations.append(sub_operation)

    def get_trace_decorator(self, attrs: Optional[Dict[str, List[str]]] = None, 
                           group_type: Optional[str] = None, 
                           comm_func: Optional[str] = None):
        """
        Get decorator for automatic operation tracing.
        
        Args:
            attrs: Dictionary mapping variable names to attributes to extract
            group_type: Communication group type
            comm_func: Communication function name
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                current_op = self.get_current_operation()
                if current_op is not None:
                    # Start timing
                    if torch.cuda.is_available():
                        start_event = torch.cuda.Event(enable_timing=True)
                        stop_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        result = func(*args, **kwargs)
                        stop_event.record()
                        torch.cuda.synchronize()
                        duration = start_event.elapsed_time(stop_event)
                    else:
                        start_time = time.perf_counter()
                        result = func(*args, **kwargs)
                        torch.cuda.synchronize()
                        end_time = time.perf_counter()
                        duration = (end_time - start_time) * 1000

                    # Extract attributes
                    attr_info = {}
                    if attrs:
                        bound_args = inspect.signature(func).bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        for var_name, attributes in attrs.items():
                            variable = bound_args.arguments.get(var_name)
                            if variable is not None:
                                for attr in attributes:
                                    if hasattr(variable, attr):
                                        value = getattr(variable, attr)
                                        if attr == 'shape':
                                            attr_info[f"{var_name}_{attr}"] = list(value)
                                        else:
                                            attr_info[f"{var_name}_{attr}"] = value
                                    else:
                                        attr_info[f"{var_name}_{attr}"] = variable
                        if group_type:
                            attr_info['group'] = group_type
                        if comm_func:
                            attr_info['comm_func'] = comm_func

                    self.add_sub_operation(func.__name__, round(duration, 2), attr_info)
                else:
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator

    def async_start_trace(self, operation_name: str, var_dict: Dict[str, Any], 
                         attrs: Dict[str, List[str]], group_type: Optional[str] = None, 
                         comm_func: Optional[str] = None):
        """Start async operation tracing."""
        current_op = self.get_current_operation()
        if current_op is None:
            return

        attr_info = {}
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.temp_async_records[operation_name] = {'attr_info': attr_info, 'start_event': start_event}
        else:
            start_time = time.perf_counter()
            self.temp_async_records[operation_name] = {'attr_info': attr_info, 'start_time': start_time}

        # Extract attributes
        for var_name, variables in var_dict.items():
            if var_name in attrs:
                for attr in attrs[var_name]:
                    if hasattr(variables, attr):
                        value = getattr(variables, attr)
                        attr_info[f"{var_name}_{attr}"] = list(value) if attr == 'shape' else value
                    else:
                        attr_info[f"{var_name}_{attr}"] = variables

        if group_type:
            attr_info['group'] = group_type
        if comm_func:
            attr_info['comm_func'] = comm_func

    def async_end_trace(self, operation_name: str):
        """End async operation tracing."""
        current_op = self.get_current_operation()
        if current_op is None or operation_name not in self.temp_async_records:
            return

        record = self.temp_async_records.pop(operation_name)
        duration = "Async operation, duration not measured"
        
        if torch.cuda.is_available():
            start_event = record.get('start_event', None)
            if start_event is not None:
                stop_event = torch.cuda.Event(enable_timing=True)
                stop_event.record()
                torch.cuda.synchronize()
                duration = start_event.elapsed_time(stop_event)
        else:
            start_time = record.get('start_time', None)
            if start_time is not None:
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000

        attr_info = record.get('attr_info', {})
        self.add_sub_operation(operation_name, round(duration, 2), attr_info)
                
    def export_traces(self, output_path: str = "profiler_log", 
                     name_prefix: str = "trace"):
        """Export traces to file."""
        if not self.enabled or self.rank_id not in self.traces:
            return
            
        import os
        os.makedirs(output_path, exist_ok=True)
        
        filename = f"{output_path}/{name_prefix}_rank{self.rank_id}_{time.strftime('%Y%m%d%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            for trace in self.traces[self.rank_id]:
                f.write(f"{trace}\n")
                
        print(f"Rank {self.rank_id}: Exported traces to {filename}")


class OperationTracker:
    """Individual operation tracker with sub-operation support."""
    
    def __init__(self, op_name: str, stage: str, group_kind: str,
                 tensor_shape: Optional[List[int]], tensor_dtype: Optional[torch.dtype],
                 description: str, rank_id: str, stage_id: int):
        self.op_name = op_name
        self.stage = stage
        self.group_kind = group_kind
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.description = description
        self.rank_id = rank_id
        self.stage_id = stage_id
        self.batch_id = None
        self.duration = None
        self.timestamp = None
        self.sub_operations = []  # 新增：子操作列表
        
        self.start_event = None
        self.stop_event = None
        self.start_time = None
        
    def start_timing(self):
        """Start timing the operation."""
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
            
    def end_timing(self):
        """End timing the operation."""
        if torch.cuda.is_available():
            self.stop_event.record()
            torch.cuda.synchronize()
            self.duration = self.start_event.elapsed_time(self.stop_event)
        else:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            self.duration = (end_time - self.start_time) * 1000
            
        self.duration = round(self.duration, 2)
        self.timestamp = round(time.perf_counter() * 1000, 2)
        
    def create_trace_entry(self) -> str:
        """Create formatted trace entry."""
        shape_str = list(self.tensor_shape) if self.tensor_shape else None
        dtype_str = str(self.tensor_dtype) if self.tensor_dtype else None
        
        return (f"rank:{self.rank_id}:{self.op_name}"
                f"(stage_id={self.stage_id},batch_id={self.batch_id},"
                f"stage={self.stage},duration={self.duration:.2f},"
                f"group_kind={self.group_kind},input__shape={shape_str},"
                f"input__dtype={dtype_str},timestamp={self.timestamp},"
                f"description={self.description},sub_operations={self.sub_operations})")


# Global profiler instance
profiler = ProfilerManager() 