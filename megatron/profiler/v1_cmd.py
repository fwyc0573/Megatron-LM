import time
import torch
import os
import contextvars
from typing import Optional, Tuple, Callable, Dict, List
# from functools import partial
import inspect

from threading import Lock
cmd_var_lock = Lock()

# current_cmd_var = contextvars.ContextVar('current_cmd')
current_cmd_var = None

class CMD:
    temp_async_records = {}
    
    def __init__(self, rank_id, mg_state, name_cmd, use_cuda=True, stage_operations_trace_dict=None, 
                 micro_batch_ids_dict=None, stage_id=None, duration=None, time_stamp=None, description=None, group_kind=None, simu_start=None, trace_start=None, current_iter=None, args=None):
        # print(f"simu_start:{simu_start}, current_iter:{current_iter}, trace_start:{trace_start}")
        assert simu_start and current_iter is not None and trace_start is not None, "stage_operations_trace and micro_batch_ids_dict must be provided"
        self.rank_id = rank_id
        self.stage_id = stage_id
        self.mg_state = mg_state
        self.name_cmd = name_cmd
        self.duration = duration
        self.time_stamp = time_stamp
        self.description = description
        self.group_kind = group_kind
        self.use_cuda = use_cuda
        self.start_event = None
        self.stop_event = None
        self.start_time = None
        self.stop_time = None
        self.simu_start = simu_start
        self.trace_start = trace_start # 从第几次开始记录
        self.current_iter = current_iter
        self.stage_operations_trace_dict = stage_operations_trace_dict
        self.micro_batch_ids_dict = micro_batch_ids_dict
        self.args = args
        self.sub_operations = []  # 用于记录子操作的信息
        
    def no_trace_update(self, duration):
        if not self.simu_start or (self.current_iter < self.trace_start-1):
            return
        
        if self.micro_batch_ids_dict is not None:
            self.micro_batch_ids_dict[self.name_cmd] += 1
            self.batch_id = self.micro_batch_ids_dict[self.name_cmd]

        self.duration = duration
        if self.rank_id not in self.stage_operations_trace_dict:
            self.stage_operations_trace_dict[self.rank_id] = []
        self.stage_operations_trace_dict[self.rank_id].append(str(self))

    def __enter__(self):
        if not self.simu_start or (self.current_iter < self.trace_start-1): # (self.current_iter == self.args.train_iters-1)
            return
        
        if self.micro_batch_ids_dict is not None:
            self.micro_batch_ids_dict[self.name_cmd] += 1
            self.batch_id = self.micro_batch_ids_dict[self.name_cmd]

        if self.use_cuda and torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.simu_start or (self.current_iter < self.trace_start-1):
            return
        
        if self.use_cuda and torch.cuda.is_available():
            self.stop_event.record()
            torch.cuda.synchronize()
            self.duration = self.start_event.elapsed_time(self.stop_event)
        else:
            self.stop_time = time.perf_counter()
            self.duration = (self.stop_time - self.start_time) * 1000  # cover to ms
        self.duration = round(self.duration,2)

        # 获取操作完成时的时间戳
        self.time_stamp = round(time.perf_counter() * 1000, 2) # covert to ms

        if self.rank_id not in self.stage_operations_trace_dict:
            self.stage_operations_trace_dict[self.rank_id] = []
        self.stage_operations_trace_dict[self.rank_id].append(str(self))

    def __call__(self, use_cuda=None):
        if use_cuda is not None:
            self.use_cuda = use_cuda
        return self

    def __str__(self):
        if not self.simu_start or (self.current_iter < self.trace_start-1):
            return f"rank:{self.rank_id}:{self.name_cmd}(stage_id={self.stage_id},batch_id=None,mg_state={self.mg_state},duration=None,description={self.description},group_kind={self.group_kind})"
        return f"rank:{self.rank_id}:{self.name_cmd}(stage_id={self.stage_id},batch_id={self.batch_id},mg_state={self.mg_state},duration={self.duration},description={self.description},group_kind={self.group_kind},time_stamp={self.time_stamp})"

    def add_sub_operation(self, operation_name, duration, attr_info):
        timestamp = round(time.perf_counter() * 1000, 2)
        sub_operation = f"{operation_name}: duration={duration}, timestamp={timestamp}"
        if attr_info:
            sub_operation += ", " + ", ".join(f"{key}: {value}" for key, value in attr_info.items())
        self.sub_operations.append(sub_operation)

    def add_variable_info(self, func_name, var_name, attr_info):
        if self.sub_operations and self.sub_operations[-1].startswith(f"{func_name}: duration"):
            # Append attribute information to the latest entry if it matches the func_name
            self.sub_operations[-1] += f", {attr_info}"
        else:
            info = f"{func_name}: {var_name} attributes: {attr_info}"
            self.sub_operations.append(info)

    # @staticmethod
    # def get_trace_decorator(attrs: Optional[Dict[str, List[str]]] = None, group_type: Optional[str] = None):
    #     def decorator(func):
    #         def wrapper(*args, **kwargs):
    #             current_cmd = current_cmd_var.get(None)
    #             if current_cmd is not None:
    #                 if current_cmd.use_cuda and torch.cuda.is_available():
    #                     start_event = torch.cuda.Event(enable_timing=True)
    #                     stop_event = torch.cuda.Event(enable_timing=True)
    #                     start_event.record()
    #                     result = func(*args, **kwargs)
    #                     stop_event.record()
    #                     torch.cuda.synchronize()
    #                     duration = start_event.elapsed_time(stop_event)
                        
    #                 else:
    #                     start_time = time.perf_counter()
    #                     result = func(*args, **kwargs)
    #                     end_time = time.perf_counter()
    #                     duration = (end_time - start_time) * 1000  # convert to ms

    #                 attr_info = {}
    #                 if attrs:
    #                     bound_args = inspect.signature(func).bind(*args, **kwargs)
    #                     bound_args.apply_defaults()
    #                     for var_name, attributes in attrs.items():
    #                         variable = bound_args.arguments.get(var_name)
    #                         if variable == "embedding_bwd_async":
    #                             raise 0

    #                         if variable is not None:
    #                             for attr in attributes:
    #                                 if hasattr(variable, attr):
    #                                     value = getattr(variable, attr)
    #                                     if attr == 'shape':
    #                                         attr_info[f"{var_name}.{attr}"] = list(value)
    #                                     else:
    #                                         attr_info[f"{var_name}.{attr}"] = value
    #                                 else:
    #                                     # attr_info[f"{var_name}.{attr}"] = 'Attribute not found'
    #                                     attr_info[f"{var_name}.{attr}"] = variable
    #                     if group_type:
    #                         attr_info['group.type'] = group_type

    #                 current_cmd.add_sub_operation(func.__name__, round(duration, 2), attr_info)
    #                 return result
    #             else:
    #                 return func(*args, **kwargs)
    #         return wrapper
    #     return decorator


    @staticmethod
    def set_current_cmd(cmd):
        global current_cmd_var
        with cmd_var_lock:
            current_cmd_var = cmd

    @staticmethod
    def get_current_cmd():
        global current_cmd_var
        with cmd_var_lock:
            return current_cmd_var

    @staticmethod
    def get_trace_decorator(attrs: Optional[Dict[str, List[str]]] = None, group_type: Optional[str] = None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                # current_cmd = current_cmd_var.get(None)
                current_cmd = current_cmd_var
                if current_cmd is not None:
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
                        end_time = time.perf_counter()
                        duration = (end_time - start_time) * 1000  # convert to ms

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
                                            attr_info[f"{var_name}.{attr}"] = list(value)
                                        else:
                                            attr_info[f"{var_name}.{attr}"] = value
                                    else:
                                        # attr_info[f"{var_name}.{attr}"] = 'Attribute not found'
                                        attr_info[f"{var_name}.{attr}"] = variable
                        if group_type:
                            attr_info['group'] = group_type

                    current_cmd.add_sub_operation(func.__name__, round(duration, 2), attr_info)
                else:
                    print("current_cmd is none, so write to txt..")
                    raise ValueError("current_cmd should not be None")
                    # # Handle the case when current_cmd is None
                    # start_time = time.perf_counter()
                    # result = func(*args, **kwargs)
                    # end_time = time.perf_counter()
                    # duration = (end_time - start_time) * 1000  # convert to ms
                    # attr_info = {}
                    # if attrs:
                    #     bound_args = inspect.signature(func).bind(*args, **kwargs)
                    #     bound_args.apply_defaults()
                    #     for var_name, attributes in attrs.items():
                    #         variable = bound_args.arguments.get(var_name)
                    #         if variable is not None:
                    #             for attr in attributes:
                    #                 if hasattr(variable, attr):
                    #                     value = getattr(variable, attr)
                    #                     if attr == 'shape':
                    #                         attr_info[f"{var_name}.{attr}"] = list(value)
                    #                     else:
                    #                         attr_info[f"{var_name}.{attr}"] = value
                    #                 else:
                    #                     # attr_info[f"{var_name}.{attr}"] = 'Attribute not found'
                    #                     attr_info[f"{var_name}.{attr}"] = variable
                    #     if group_type:
                    #         attr_info['group'] = group_type
                    # # Directly write to file when current_cmd is None
                    # print(f"txt_info: {func.__name__}: duration={round(duration, 2)}, attributes={attr_info}")
                    # CMD._write_to_file(f"{func.__name__}: duration={round(duration, 2)}, attributes={attr_info}")
                return result
            return wrapper
        return decorator

    @staticmethod
    def async_start_trace(operation_name: str, var_dict: Dict[str, any], attrs: Dict[str, List[str]], group_type: Optional[str] = None):
        # current_cmd = current_cmd_var.get(None)
        current_cmd = current_cmd_var
        attr_info = {}
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        for var_name, variables in var_dict.items():
            if var_name in attrs:
                for attr in attrs[var_name]:
                    if hasattr(variables, attr):
                        value = getattr(variables, attr)
                        if attr == 'shape':
                            attr_info[f"{var_name}.{attr}"] = list(value)
                        else:
                            attr_info[f"{var_name}.{attr}"] = value
                    else:
                        attr_info[f"{var_name}.{attr}"] = 'Attribute not found'
                        # attr_info[f"{var_name}.{attr}"] = variable

        if group_type:
            attr_info['group'] = group_type

        if current_cmd is not None:
            current_cmd.start_event = start_event
            current_cmd.sub_operations.append({'operation_name': operation_name, 'attr_info': attr_info, 'start_event': start_event})
        # else:
        #     # 临时存储异步操作的信息
        #     CMD.temp_async_records[operation_name] = {'attr_info': attr_info, 'start_event': start_event}
        #     print(f"async:{attr_info}")


    @staticmethod
    def async_end_trace(operation_name: str):
        # current_cmd = current_cmd_var.get(None)
        current_cmd = current_cmd_var
        if current_cmd is not None:
            for op in current_cmd.sub_operations:
                if op.get('operation_name') == operation_name:
                    start_event = op.get('start_event')
                    duration = "Async operation, duration not measured"
                    if start_event is not None:
                        stop_event = torch.cuda.Event(enable_timing=True)
                        stop_event.record()
                        stop_event.synchronize()
                        duration = start_event.elapsed_time(stop_event)
                    attr_info = op.get('attr_info', {})
                    current_cmd.add_sub_operation(operation_name, duration, attr_info)
                    break
        # else:
        #     # 当 current_cmd 为 None 时，从临时变量中取出信息
        #     if operation_name in CMD.temp_async_records:
        #         record = CMD.temp_async_records.pop(operation_name)
        #         start_event = record['start_event']
        #         duration = "Async operation, duration not measured"
        #         if start_event is not None:
        #             stop_event = torch.cuda.Event(enable_timing=True)
        #             stop_event.record()
        #             stop_event.synchronize()
        #             duration = start_event.elapsed_time(stop_event)
        #         attr_info = record.get('attr_info', {})
        #         CMD._write_to_file(f"{operation_name}: duration={duration}, attributes={attr_info}")
        #         print(f"async:{operation_name}")

        
    @staticmethod
    def _write_to_file(content):
        # Write content to a log file, ensure this function is thread-safe if needed
        filename = "./cmd_operations_log.txt"
        with open(filename, 'a') as f:
            f.write(f"{content}\n")






def write_list_to_file(stage_id, list_to_write):
    os.makedirs("./mg_scheduling_plan_log", exist_ok=True)
    filename = f"./mg_scheduling_plan_log/0713mg_rank{stage_id}_scheduling_plan.txt"
    with open(filename, 'w') as f:
        for item in list_to_write:
            f.write(f"{item}\n")