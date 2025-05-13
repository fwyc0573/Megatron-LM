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
from typing import Optional, Dict, List
from threading import Lock
import inspect

cmd_var_lock = Lock()
current_cmd_var = None
single_gpu_profile = False

class CMD:
    temp_async_records = {}

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
        if isinstance(input__shape, torch.Size):
            self.input__shape = list(input__shape)
        elif isinstance(input__shape, tuple):
            self.input__shape = list(input__shape)
        elif isinstance(input__shape, list) or input__shape == None:
            self.input__shape = input__shape
        else:
            raise ValueError(f"input__shape type error: {input__shape}")

        self.input__dtype = input__dtype

        self.use_cuda = True # True False # use_cuda

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


    def set_tensor_shape_and_dtype(self, input__shape, input__dtype):
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
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return
        # self.set_current_cmd(self.cur_cmd)

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
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return

        if self.use_cuda and torch.cuda.is_available():
            self.stop_event.record()
            torch.cuda.synchronize()
            self.stop_time = time.perf_counter()
            self.duration = self.start_event.elapsed_time(self.stop_event)
        else:
            torch.cuda.synchronize()
            self.stop_time = time.perf_counter()
            self.duration = (self.stop_time - self.start_time) * 1000
        self.duration = round(self.duration, 2)

        self.time_stamp = round(self.stop_time * 1000, 2)

        if self.rank_id not in self.stage_operations_trace_dict:
            self.stage_operations_trace_dict[self.rank_id] = []
        self.stage_operations_trace_dict[self.rank_id].append(str(self))
        self.reset_current_cmd()

    # def __call__(self, use_cuda=True):
    #     self.use_cuda = use_cuda
    #     self.use_cuda = False
    #     return self

    def __str__(self):
        if not self.simu_start or (self.current_iter < self.trace_start - 1):
            return f"rank:{self.rank_id}:{self.name_cmd}(stage_id={self.stage_id},batch_id=None,mg_state={self.mg_state},duration=None,description={self.description},group_kind={self.group_kind})"
        return f"rank:{self.rank_id}:{self.name_cmd}(stage_id={self.stage_id},batch_id={self.batch_id},mg_state={self.mg_state},duration={self.duration},description={self.description},group_kind={self.group_kind},input__shape={self.input__shape},input__dtype={self.input__dtype},timestamp={self.time_stamp},sub_operations={self.sub_operations})"

    def add_sub_operation(self, operation_name, duration, attr_info):
        timestamp = round(time.perf_counter() * 1000, 2)
        sub_operation = f"trace_src_func={operation_name},duration={duration},timestamp={timestamp}"
        if attr_info:
            sub_operation += "," + ",".join(f"{key}={value}" for key, value in attr_info.items())
        self.sub_operations.append(sub_operation)

    @staticmethod
    def set_current_cmd(cmd):
        global current_cmd_var
        with cmd_var_lock:
            current_cmd_var = cmd
            # print(f"current_cmd_var:{current_cmd_var}")

    @staticmethod
    def reset_current_cmd():
        global current_cmd_var
        with cmd_var_lock:
            current_cmd_var = None

    @staticmethod
    def get_current_cmd():
        global current_cmd_var
        with cmd_var_lock:
            return current_cmd_var
        
    @staticmethod
    def set_current_profile_sign(setting_sign):
        global single_gpu_profile
        with cmd_var_lock:
            single_gpu_profile = setting_sign

    @staticmethod
    def get_current_profile_sign():
        global single_gpu_profile
        with cmd_var_lock:
            return single_gpu_profile


    @staticmethod
    def get_trace_decorator(attrs: Optional[Dict[str, List[str]]] = None, group_type: Optional[str] = None, comm_func: Optional[str] = None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 对于非profile模式下的异步操作由专门的async负责trace和record,装饰器不参与
                # 对于profile模式下，则由装饰器处理（因为profile的GPU为1，handle必为None，部分模块无法使用async的逻辑记录）
                # TODO：现在只有allreduce和_reduce被加了装饰器
                
                if (func.__name__ == "allreduce" or "_reduce") and kwargs.get('async_op', False) and CMD.get_current_profile_sign() is False:
                    return func(*args, **kwargs)
                
                current_cmd = CMD.get_current_cmd()
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
                        torch.cuda.synchronize()
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
                                            attr_info[f"{var_name}_{attr}"] = list(value)
                                        else:
                                            attr_info[f"{var_name}_{attr}"] = value
                                    else:
                                        # attr_info[f"{var_name}.{attr}"] = 'Attribute not found'
                                        attr_info[f"{var_name}_{attr}"] = variable
                        if group_type:
                            attr_info['group'] = group_type
                        if comm_func:
                            attr_info['comm_func'] = comm_func

                    current_cmd.add_sub_operation(func.__name__, round(duration, 2), attr_info)
                else:
                    # Handle the case when current_cmd is None, e.g., log directly
                    print("current_cmd is none, skip record operation.")
                    # raise ValueError("current_cmd should not be None")
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator

    # @staticmethod
    # def async_start_trace(operation_name: str, var_dict: Dict[str, any], attrs: Dict[str, List[str]], group_type: Optional[str] = None, comm_func: Optional[str] = None):
    #     current_cmd = CMD.get_current_cmd()
    #     if current_cmd is None:
    #         raise ValueError("current_cmd should not be None")

    #     attr_info = {}
    #     start_event = torch.cuda.Event(enable_timing=True)
    #     start_event.record()

    #     for var_name, variables in var_dict.items():
    #         if var_name in attrs:
    #             for attr in attrs[var_name]:
    #                 if hasattr(variables, attr):
    #                     value = getattr(variables, attr)
    #                     attr_info[f"{var_name}.{attr}"] = list(value) if attr == 'shape' else value
    #                 else:
    #                     # attr_info[f"{var_name}.{attr}"] = 'Attribute not found'
    #                     attr_info[f"{var_name}.{attr}"] = variables

    #     if group_type:
    #         attr_info['group'] = group_type
    #     if comm_func:
    #         attr_info['comm_func'] = comm_func
    #     CMD.temp_async_records[operation_name] = {'attr_info': attr_info, 'start_event': start_event}

    # @staticmethod
    # def async_end_trace(operation_name: str):
    #     current_cmd = CMD.get_current_cmd()
    #     if current_cmd is not None:
    #         if operation_name in CMD.temp_async_records:
    #             record = CMD.temp_async_records.pop(operation_name)
    #             start_event = record['start_event']
    #             duration = "Async operation, duration not measured"
    #             if start_event is not None:
    #                 stop_event = torch.cuda.Event(enable_timing=True)
    #                 stop_event.record()
    #                 stop_event.synchronize()
    #                 duration = start_event.elapsed_time(stop_event)
    #             attr_info = record.get('attr_info', {})
    #             current_cmd.add_sub_operation(operation_name, round(duration, 2), attr_info)
    #     else:
    #         raise ValueError("current_cmd should not be None")


    @staticmethod
    def async_start_trace(operation_name: str, var_dict: Dict[str, any], attrs: Dict[str, List[str]], group_type: Optional[str] = None, comm_func: Optional[str] = None):
        
        # single gpu profile模式下让装饰器控制即可
        if CMD.get_current_profile_sign() is True:
            return

        current_cmd = CMD.get_current_cmd()
        if current_cmd is None:
            print("current_cmd is none, skip record operation.")
            return 
            # raise ValueError("current_cmd should not be None")

        attr_info = {}
        if current_cmd.use_cuda and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            CMD.temp_async_records[operation_name] = {'attr_info': attr_info, 'start_event': start_event}
        else:
            start_time = time.perf_counter()
            CMD.temp_async_records[operation_name] = {'attr_info': attr_info, 'start_time': start_time}

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

    @staticmethod
    def async_end_trace(operation_name: str):
        # single gpu profile模式下让装饰器控制即可
        if CMD.get_current_profile_sign() is True:
            return

        current_cmd = CMD.get_current_cmd()
        if current_cmd is None:
            print("current_cmd is none, skip record operation.")
            return 
        
        if current_cmd is not None:
            if operation_name in CMD.temp_async_records:
                record = CMD.temp_async_records.pop(operation_name)
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
            raise ValueError("current_cmd should not be None")


    @staticmethod
    def _write_to_file(content):
        filename = "./cmd_operations_log.txt"
        with open(filename, 'a') as f:
            f.write(f"{content}\n")

def write_list_to_file(stage_or_rank_id, list_to_write, file_path=None, name_args=None):
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




