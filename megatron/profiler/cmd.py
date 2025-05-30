""" 
一般用法：
        # trace send_backward的duration和一系列属性, 并记录到stage_operations_trace
        cmd = CMD(
            rank_id=args.simu_rank,
            mg_state=args.simu_state,
            name_cmd="send_backward",
            use_cuda=True,
            stage_operations_trace_dict=args.stage_operations_trace,
            micro_batch_ids_dict=args.simu_micro_batch_ids,
            stage_id=args.simu_stage_id,
            simu_start=args.simu_start,
            trace_start=args.trace_start,
            group_kind="pp",
            current_iter=args.current_iter,
            args=args
        )
        with cmd: 
            send_backward(input_tensor_grad, recv_tensor_shapes, config)
        
            
        # 当添加CMD.set_current_cmd(cmd)时, 设定了全局cmd, 用于记录当前cmd调用的子cmd(不过嵌套追踪的条件是func有装饰器修饰,如果是额外些的func,则不会记录,参考dp_allreduce和ep...)
        cmd = CMD(
            rank_id=args.simu_rank,
            mg_state=args.simu_state,
            name_cmd="backward_step",
            use_cuda=True,
            stage_operations_trace_dict=args.stage_operations_trace,
            micro_batch_ids_dict=args.simu_micro_batch_ids,
            stage_id=args.simu_stage_id,
            simu_start=args.simu_start,
            trace_start=args.trace_start,
            current_iter=args.current_iter,
            args=args
        )
        CMD.set_current_cmd(cmd)
        with cmd: 
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
        print(f"rank:{args.simu_rank}, bwd_subop num: {len(cmd.sub_operations)}, bwd_subop: {cmd.sub_operations}")

"""
import time
import torch
import os
import uuid
import threading
from typing import Optional, Dict, List
from threading import Lock
import inspect

# Global variables with thread safety
cmd_var_lock = Lock()
async_records_lock = Lock()
current_cmd_var = None
single_gpu_profile = False

class CMD:
    # Thread-safe async records with proper cleanup
    temp_async_records = {}
    _async_record_timeout = 300  # 5 minutes timeout for async operations

    def __init__(self, rank_id, mg_state, name_cmd, use_cuda=True, stage_operations_trace_dict=None, 
                 micro_batch_ids_dict=None, stage_id=None, duration=None, time_stamp=None, description=None, \
                    group_kind=None, input__shape=None, input__dtype=None, simu_start=None, trace_start=None, current_iter=None, \
                        args=None):
        self.rank_id = rank_id
        self.stage_id = stage_id
        self.mg_state = mg_state
        self.name_cmd = name_cmd
        self.duration = duration
        self.time_stamp = time_stamp
        self.description = description
        self.group_kind = group_kind
        
        # Normalize input shape to list format
        if isinstance(input__shape, torch.Size):
            self.input__shape = list(input__shape)
        elif isinstance(input__shape, tuple):
            self.input__shape = list(input__shape)
        elif isinstance(input__shape, list) or input__shape == None:
            self.input__shape = input__shape
        else:
            raise ValueError(f"input__shape type error: {input__shape}")

        self.input__dtype = input__dtype

        self.use_cuda = use_cuda  # Fixed: use the parameter instead of hardcoded True

        self.start_event = None
        self.stop_event = None
        self.start_time = None
        self.stop_time = None
        self.simu_start = simu_start
        self.trace_start = trace_start
        self.current_iter = current_iter
        self.stage_operations_trace_dict = stage_operations_trace_dict
        self.micro_batch_ids_dict = micro_batch_ids_dict
        self.args = args
        self.sub_operations = []
        
        # Track if this CMD is the current global command
        self._is_current_cmd = False

    def set_tensor_shape_and_dtype(self, input__shape, input__dtype):
        """Set tensor shape and dtype with proper type validation"""
        if isinstance(input__shape, torch.Size):
            self.input__shape = list(input__shape)
        elif isinstance(input__shape, tuple):
            self.input__shape = list(input__shape)
        elif isinstance(input__shape, list) or input__shape == None:
            self.input__shape = input__shape
        else:
            raise ValueError(f"input__shape type error: {input__shape}")

        self.input__dtype = input__dtype

    def no_trace_update(self, duration, timestamp):
        """Update CMD with duration and timestamp without tracing"""
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return

        if self.micro_batch_ids_dict is not None:
            self.micro_batch_ids_dict[self.name_cmd] += 1
            self.batch_id = self.micro_batch_ids_dict[self.name_cmd]

        self.duration = duration
        self.time_stamp = timestamp
        if self.rank_id not in self.stage_operations_trace_dict:
            self.stage_operations_trace_dict[self.rank_id] = []
        self.stage_operations_trace_dict[self.rank_id].append(str(self))

    def __enter__(self):
        """Context manager entry with proper error handling"""
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return self

        if self.micro_batch_ids_dict is not None:
            self.micro_batch_ids_dict[self.name_cmd] += 1
            self.batch_id = self.micro_batch_ids_dict[self.name_cmd]

        try:
            if self.use_cuda and torch.cuda.is_available():
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.stop_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
            else:
                self.start_time = time.perf_counter()
        except Exception as e:
            print(f"Warning: Failed to start timing for {self.name_cmd}: {e}")
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with guaranteed cleanup"""
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return

        try:
            if self.use_cuda and torch.cuda.is_available():
                if self.stop_event is not None:
                    self.stop_event.record()
                    torch.cuda.synchronize()
                    self.stop_time = time.perf_counter()
                    if self.start_event is not None:
                        self.duration = self.start_event.elapsed_time(self.stop_event)
                    else:
                        self.duration = 0.0
                else:
                    self.duration = 0.0
            else:
                torch.cuda.synchronize()
                self.stop_time = time.perf_counter()
                if self.start_time is not None:
                    self.duration = (self.stop_time - self.start_time) * 1000
                else:
                    self.duration = 0.0
            
            self.duration = round(self.duration, 2)
            self.time_stamp = round(self.stop_time * 1000, 2)

            if self.rank_id not in self.stage_operations_trace_dict:
                self.stage_operations_trace_dict[self.rank_id] = []
            self.stage_operations_trace_dict[self.rank_id].append(str(self))
            
        except Exception as e:
            print(f"Warning: Failed to stop timing for {self.name_cmd}: {e}")
        finally:
            # Ensure cleanup always happens
            if self._is_current_cmd:
                self.reset_current_cmd()

    def __str__(self):
        """String representation of the CMD object"""
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return f"rank:{self.rank_id}:{self.name_cmd}(stage_id={self.stage_id},batch_id=None,mg_state={self.mg_state},duration=None,description={self.description},group_kind={self.group_kind})"
        return f"rank:{self.rank_id}:{self.name_cmd}(stage_id={self.stage_id},batch_id={self.batch_id},mg_state={self.mg_state},duration={self.duration},description={self.description},group_kind={self.group_kind},input__shape={self.input__shape},input__dtype={self.input__dtype},timestamp={self.time_stamp},sub_operations={self.sub_operations})"

    def add_sub_operation(self, operation_name, duration, attr_info):
        """Add a sub-operation to this CMD"""
        timestamp = round(time.perf_counter() * 1000, 2)
        sub_operation = f"trace_src_func={operation_name},duration={duration},timestamp={timestamp}"
        if attr_info:
            sub_operation += "," + ",".join(f"{key}={value}" for key, value in attr_info.items())
        self.sub_operations.append(sub_operation)

    @staticmethod
    def set_current_cmd(cmd):
        """Set the global current CMD instance with thread safety"""
        global current_cmd_var
        with cmd_var_lock:
            current_cmd_var = cmd
            if cmd is not None:
                cmd._is_current_cmd = True

    @staticmethod
    def reset_current_cmd():
        """Reset the global current CMD instance with thread safety"""
        global current_cmd_var
        with cmd_var_lock:
            if current_cmd_var is not None:
                current_cmd_var._is_current_cmd = False
            current_cmd_var = None

    @staticmethod
    def get_current_cmd():
        """Get the global current CMD instance with thread safety"""
        global current_cmd_var
        with cmd_var_lock:
            return current_cmd_var
        
    @staticmethod
    def set_current_profile_sign(setting_sign):
        """Set the single GPU profile mode flag"""
        global single_gpu_profile
        with cmd_var_lock:
            single_gpu_profile = setting_sign

    @staticmethod
    def get_current_profile_sign():
        """Get the single GPU profile mode flag"""
        global single_gpu_profile
        with cmd_var_lock:
            return single_gpu_profile

    @staticmethod
    def _cleanup_expired_async_records():
        """Clean up expired async records to prevent memory leaks"""
        current_time = time.time()
        with async_records_lock:
            expired_keys = []
            for key, record in CMD.temp_async_records.items():
                if current_time - record.get('created_time', 0) > CMD._async_record_timeout:
                    expired_keys.append(key)
            
            for key in expired_keys:
                print(f"Warning: Async operation {key} expired and was cleaned up")
                CMD.temp_async_records.pop(key, None)

    @staticmethod
    def get_trace_decorator(attrs: Optional[Dict[str, List[str]]] = None, group_type: Optional[str] = None, comm_func: Optional[str] = None, overlap_op: Optional[str] = None):
        """Get a decorator for tracing function calls"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Fix: Correct condition check for async operations
                # TODO: I think in async comm. op, we should not use sync() which break the oringinal running mode (async and paralell)
                # 这2类async comm在real running mode的下，直接返回：因为无法记录comm（还会导致并行并发被sync破坏
                # scaling mode模式下，还是会测量耗时（单纯用于模拟cpu侧？）
                if (func.__name__ in ["allreduce", "_reduce"]) and kwargs.get('async_op', False) and CMD.get_current_profile_sign() is False:
                    return func(*args, **kwargs)
                
                current_cmd = CMD.get_current_cmd()
                if current_cmd is not None:
                    try:
                        if current_cmd.use_cuda and torch.cuda.is_available():
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
                            duration = (end_time - start_time) * 1000  # convert to ms

                        attr_info = {}
                        if attrs:
                            try:
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
                            except Exception as e:
                                print(f"Warning: Failed to extract attributes for {func.__name__}: {e}")
                                
                        if group_type:
                            attr_info['group'] = group_type
                        if comm_func:
                            attr_info['comm_func'] = comm_func
                        # if overlap_op:
                        #     attr_info['overlap_op'] = overlap_op

                        current_cmd.add_sub_operation(func.__name__, round(duration, 2), attr_info)
                    except Exception as e:
                        print(f"Warning: Failed to trace {func.__name__}: {e}")
                        result = func(*args, **kwargs)
                else:
                    # Handle the case when current_cmd is None - just execute without tracing
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator

    @staticmethod
    def async_start_trace(operation_name: str, var_dict: Dict[str, any], attrs: Dict[str, List[str]], group_type: Optional[str] = None, comm_func: Optional[str] = None):
        """Start tracing an async operation with thread safety and unique keys"""
        
        # Only scaling mode will invoke CMD.get_current_profile_sign()=true, i.e, the big async op will not be traced (only real running mode will trace the big async op)
        if CMD.get_current_profile_sign() is True:
            return None

        current_cmd = CMD.get_current_cmd()
        if current_cmd is None:
            print("Warning: current_cmd is None, skip async start trace.")
            return None

        # Clean up expired records periodically
        CMD._cleanup_expired_async_records()

        # raise 0

        # Generate unique key to avoid conflicts
        unique_key = f"{operation_name}_{uuid.uuid4().hex[:8]}_{time.time()}"
        
        attr_info = {}
        
        # Extract attributes first
        try:
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
        except Exception as e:
            raise 0
            print(f"Warning: Failed to extract attributes for async {operation_name}: {e}")

        # Create timing record
        try:
            with async_records_lock:
                if current_cmd.use_cuda and torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    CMD.temp_async_records[unique_key] = {
                        'attr_info': attr_info, 
                        'start_event': start_event,
                        'created_time': time.time(),
                        'operation_name': operation_name
                    }
                else:
                    start_time = time.perf_counter()
                    CMD.temp_async_records[unique_key] = {
                        'attr_info': attr_info, 
                        'start_time': start_time,
                        'created_time': time.time(),
                        'operation_name': operation_name
                    }
        except Exception as e:
            print(f"Warning: Failed to start async trace for {operation_name}: {e}")
            return None

        return unique_key

    @staticmethod
    def async_end_trace(unique_key: str):
        """End tracing an async operation using the unique key"""
        # single gpu profile模式下让装饰器控制即可
        if CMD.get_current_profile_sign() is True:
            return

        if unique_key is None:
            return

        current_cmd = CMD.get_current_cmd()
        if current_cmd is None:
            print("Warning: current_cmd is None, skip async end trace.")
            return 
        raise 0
        try:
            with async_records_lock:
                if unique_key in CMD.temp_async_records:
                    record = CMD.temp_async_records.pop(unique_key)
                    operation_name = record.get('operation_name', 'unknown')
                    duration = "Async operation, duration not measured"
                    
                    if current_cmd.use_cuda and torch.cuda.is_available():
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
                            duration = (end_time - start_time) * 1000  # convert to ms
                    
                    attr_info = record.get('attr_info', {})
                    current_cmd.add_sub_operation(operation_name, round(duration, 2), attr_info)
                else:
                    print(f"Warning: Async operation {unique_key} not found in records")
        except Exception as e:
            print(f"Warning: Failed to end async trace for {unique_key}: {e}")

    @staticmethod
    def _write_to_file(content):
        """Write content to log file"""
        filename = "./cmd_operations_log.txt"
        with open(filename, 'a') as f:
            f.write(f"{content}\n")

def write_list_to_file(stage_or_rank_id, list_to_write, file_path=None, name_args=None):
    """Write a list to file with proper naming and directory creation"""
    log_file_name = f"{name_args}_rank{stage_or_rank_id}_{time.strftime('%Y%m%d%H%M%S')}.txt"

    if file_path is None:
        os.makedirs("./realistic_trace", exist_ok=True)
        filename = f"./realistic_trace/{log_file_name}"
    else:
        os.makedirs(file_path, exist_ok=True)
        filename = f"{file_path}/{log_file_name}"

    with open(filename, 'w') as f:
        for item in list_to_write:
            f.write(f"{item}\n")




