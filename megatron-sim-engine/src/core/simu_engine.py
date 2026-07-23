import copy
import os
import re
# import queue
from collections import deque
from typing import Optional, Union
import ast
import time
import random
import logging

# Set up logger
logger = logging.getLogger(__name__)

# from deepspeed.utils import logger
from src.core.static_graphs.rank_manager import RankZoo
from src.core.static_graphs.parallel_group_manager import MPUInfo
from src.core.comm_sim.nccl_comm import get_comm_op_exc_time
from src.core.cc_backend import CommunicationPredictionRequest, create_cc_backend
from src.core.simulator_config import load_config_from_env
from src.utils.message_size_calculator import calculate_comm_message_size

from src.core.cc_backend.base import get_backend_options
from src.core.cc_backend.op_mapping import infer_collective_kind
from src.extensions.slowdown_predictor import EchoSlowdownPredictor, compute_overlap_time_ms, load_slowdown_assets

DS_COMP_OPERATION = ['ForwardPass', 'BackwardPass', 'OptimizerStep', 'LoadMicroBatch']
DS_COMM_OPERATION = ['SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','ReduceGrads', 'ReduceTiedGrads']
DS_FINAL_OPERATION = ['ReduceGrads', 'ReduceTiedGrads', 'OptimizerStep']


# MG_COMP_OPERATION = ['forward_step', 'backward_step', 'load_batch']
MG_COMP_OPERATION = ['forward_step', 'backward_step','optimizer_step', 'get_batch', 'loss_func']
MG_COMM_OPERATION = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward', 'tp_allreduce', 'tp_load_batch_broadcast', 'tp_broadcast', 'tp_all_to_all', 'tp_allgather', 'tp_reduce_scatter', 'dp_allreduce', 'ep_allreduce', 'exp_dp_allreduce', 'ep_dp_allreduce', 'exp_all_to_all', 'exp_allgather', 'tp_reducescatter', 'dp_reducescatter', 'exp_reducescatter', 'cp_reducescatter']


P2P_COMM_COLLECTIVE = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward',
                        'SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation']
ALLREDUCE_COMM_COLLECTIVE = ['dp_allreduce', 'tp_allreduce', 'ep_allreduce', 'exp_dp_allreduce', 'ep_dp_allreduce', 'ReduceGrads', 'ReduceTiedGrads']
MOE_COMM_COLLECTIVE = ['exp_all_to_all', 'exp_allgather', 'exp_dp_allreduce', 'ep_dp_allreduce']
COLLECTIVE_COMM_KINDS = {'allreduce', 'all_to_all', 'allgather', 'reducescatter'}

# DS可以被模拟以及未支持模拟的op
GLOBAL_DS_DIRECT_MAPPING_LIST = ['ForwardPass', 'BackwardPass', 'SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','OptimizerStep']
GLOBAL_DS_NOT_SUPPORTED_LIST = ['LoadMicroBatch','ReduceGrads', 'ReduceTiedGrads']


# 被支持的OP可以从SINGLE_GPU_PROFILE data中获取duration等信息；该部分只适用于TP=1时，通过profile的算子是否可以直接用来模拟真实场景的算子(像'loss_func'就不应该支持,因为涉及了通信操作)
# TODO: tp_load_batch_broadcast还要吗;
GLOBAL_NETWORK_ESTIMATOR_GET_LIST = ['dp_allreduce', 'ep_allreduce', 'exp_dp_allreduce', 'ep_dp_allreduce', 'tp_allreduce', 'tp_reducescatter', 'tp_reduce_scatter', 'dp_reducescatter', 'exp_reducescatter', 'cp_reducescatter', 'send_backward', 'recv_backward', 'send_forward', 'recv_forward', 'exp_all_to_all', 'exp_allgather', 'tp_all_to_all', 'tp_allgather']
# GLOBAL_MG_DIRECT_MAPPING_LIST = ['forward_step', 'backward_step', 'optimizer_step', *GLOBAL_NETWORK_ESTIMATOR_GET_LIST]
GLOBAL_MG_DIRECT_MAPPING_LIST = ['forward_step', 'backward_step', 'optimizer_step', *GLOBAL_NETWORK_ESTIMATOR_GET_LIST]
GLOBAL_MG_NOT_SUPPORTED_LIST = ['get_batch', 'tp_allreduce', 'tp_load_batch_broadcast', 'loss_func']

# GET_BATCH暂时不break
# GET_BATCH写入到json的为什么不包含comm time？
GLOBAL_CMD_HAVE_SUBOP_LIST = ['get_batch', 'forward_step', 'backward_step', 'loss_func']
GLOBAL_CMD_NEED_BREAK_LIST = ['forward_step', 'backward_step', 'loss_func']
# schedule plan中的op可以从SINGLE_GPU_PROFILE log中得到profiled数据的op或者subop；没有在profile里头的主要是model间的通信op；所有通信OP or SUBOP都需要通过network estimator计算
# 以及暂时被模拟已知的通信操作
# TODO：添加dp_allreduce时,要修改GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST、GLOBAL_MG_DIRECT_MAPPING_LIST
# 不包含P2P，因为P2P的duration在schedule plan中被计算（tp allreduce有涉及嘛？）
# GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST = ['forward_step', 'backward_step', 'loss_func', 'dp_allreduce', 'ep_allreduce', 'tp_allreduce'] # , 'optimizer_step'
GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST = ['get_batch', 'forward_step', 'backward_step', 'loss_func', 'dp_allreduce', 'ep_allreduce', 'exp_dp_allreduce', 'ep_dp_allreduce', 'tp_allreduce', 'optimizer_step'] # , 'optimizer_step'

##################################################################### 几个global vars的区别 #####################################################################
# GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST: 
# 【哪些OP/SUBOP可以从database中获取】如果不在LIST中,则duration从trace中拿,且不存在subop_lsit。
# 读取single-gpu profiler得到的数据,读取OP/SUBOP的duration等信息(必定会依据profiler的database给OP/SUBOP的duration赋值)
# 如果OP/SUBOP不在GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST中,则有可能通过muti-gpus trace得到赋值
# for example, 'loss_func'应该被归于该list. 当tp==1,则只需要考虑profiler得到的comp时间;当tp>1时,进行SUBOP的分解,转为comp+comm；subop comp的duration被正常赋值(database), 
# comm被赋值为None/0; OP直接使用profiler的database, subop会被check_and_supplement_duration_from_muti_gpus_trace()遍历填补:对于为空的subop，如果database中存在则使用,否则使用
# trace中的duration.


# GLOBAL_CMD_NEED_BREAK_LIST：
# 【决定是否将上述分解得到的SUBOP_LIST加入到当前Stage实例的operation_list】决定了是否使用分解遍历完的sub_ops_list，不在list中则不加入到operation_list中。
# （例如get_batch不在LIST，则只会以整体OP加入到sub_ops_list）


# GLOBAL_MG_DIRECT_MAPPING_LIST/GLOBAL_MG_NOT_SUPPORTED_LIST：
# get_op_excution_time_from_torchgraph()中使用，该函数暂时废弃...
################################################################################################################################################################


# SIMULATE MODE: 
# trace会在generate_stages_and_cmds_info_from_datasets_and_schedules()完整得到所有OP/SUBOP;非trace的OP和SUBOP则在_init_3d_parallel_all_ranks()中进行完整初始化

# PROFILE MODE:
# 在generate_stages_and_cmds_info_from_datasets_and_schedules()完整得到所有OP/SUBOP


MODE_MODEL = "model"
MODE_PROFILE = "profile"
MODE_SIMULATE = "simulate"
RUNNING_MODE_OPTION = [MODE_MODEL, MODE_PROFILE, MODE_SIMULATE]


class Operation:
    def __init__(self, name="", duration=-1, buffer_id=-1, step_id=-1, batch_id=-1, wrank_id=-1, stage_id=-1, mg_state=None, op_kind=None, \
                 waiting_acc=None, description=None, group_kind=None, end_timestamp=None, hidden_duration=None, tensor_shape=None, tensor_dtype=None, \
                 cmd_uid=None, op_semantics=None, trace_metadata=None, name_with_id=None):
        """
        for a comm. ops, finish_time = join_time + waiting_time + duration.
        Note that, waiting_time 代表comm过程中等待其他stages中comm. 开始的时间.
        duration 代表实际处理时间, duration = trans.数据大小 / bandwidth
        |______waiting_time_________| + |______duration______|

        for a comp. ops, finish_time = join_time + duration
        for a new 【comp. op】 added into timeline, waiting_acc = last ops' waiting_acc
        for a new 【comm. op】 added into timeline, waiting_acc = last ops' waiting_acc + its waiting_time
        """
        self.name = name
        self.duration = duration
        self.waiting_time = None
        self.waiting_acc = waiting_acc
        self.join_time = None
        self.finish_time = None
        self.op_kind = op_kind
        self.pre_op = None
        self.post_op = None
        self.batch_id = batch_id
        self.wrank_id = wrank_id
        self.stage_id = stage_id
        self.mg_state = mg_state  # warmup/steady/cooldown/help
        self.group_kind = group_kind
        self.mg_is_last_iteration = False
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.ds_buffer_id = buffer_id
        self.ds_step_id = step_id
        self.description = description  # 给dp类allreduce算子增加的描述用于相互区分
        self.end_timestamp = end_timestamp
        self.hidden_duration = hidden_duration
        self.cmd_uid = cmd_uid
        self.op_semantics = op_semantics
        self.trace_metadata = trace_metadata or {}
        self.name_with_id = name_with_id
    
    def __str__(self):
        return f"Operation(name={self.name}, duration={self.duration}, waiting_time={self.waiting_time}, waiting_acc={self.waiting_acc}, join_time={self.join_time}, finish_time={self.finish_time}, op_kind={self.op_kind}, pre_op={self.pre_op}, post_op={self.post_op}, batch_id={self.batch_id}, wrank_id={self.wrank_id}, mg_state={self.mg_state}, mg_is_last_iteration={self.mg_is_last_iteration}, ds_buffer_id={self.ds_buffer_id}, ds_step_id={self.ds_step_id})"

    def comp_set_join_finish_waiting_acc_time(self, join_time, waiting_acc):
        self.join_time = join_time
        self.finish_time = round(join_time + self.duration,2)
        self.waiting_acc = waiting_acc
        self.waiting_time = 0

    def comm_set_join_time(self, join_time):
        self.join_time = round(join_time,2)

    def comm_set_waiting_finish_time(self, waiting_time):
        # DEBUG: Log timing calculation for communication operations
        if self.name in ['dp_allreduce', 'exp_dp_allreduce']:
            print(f"DEBUG comm_set_waiting_finish_time: {self.name} wrank_id={self.wrank_id}")
            print(f"  Before: join_time={getattr(self, 'join_time', 'None')}, waiting_time={getattr(self, 'waiting_time', 'None')}, duration={self.duration}, finish_time={getattr(self, 'finish_time', 'None')}")
            print(f"  Setting waiting_time={waiting_time}")

        self.waiting_time = round(waiting_time,2)
        self.finish_time = round(self.join_time + waiting_time + self.duration,2)
        # self.finish_time = self.join_time + self.duration

        if self.name in ['dp_allreduce', 'exp_dp_allreduce']:
            print(f"  After: join_time={self.join_time}, waiting_time={self.waiting_time}, duration={self.duration}, finish_time={self.finish_time}")
            print(f"  Calculated finish_time = {self.join_time} + {self.waiting_time} + {self.duration} = {self.finish_time}")

    def comm_set_waiting_acc_time(self, waiting_time, last_op_waiting_acc):
        self.waiting_acc = round(waiting_time + last_op_waiting_acc,2)
    
    def set_pre_op(self, pre_op):
        self.pre_op = pre_op
    
    def set_post_op(self, post_op):
        self.post_op = post_op
    
    def set_wrank_id(self, wrank_id):
        self.wrank_id = int(wrank_id)

    def set_stage_id(self, stage_id):
        self.stage_id = int(stage_id)

    def set_duration(self, duration):
        if isinstance(duration, dict):
            if 'duration' in duration:
                self.duration = round(float(duration['duration']), 2)
            else:
                raise ValueError(f"Dict duration must contain 'duration' key, got: {duration}")
        else:
            self.duration = round(float(duration), 2)

    def to_dict(self):
        return {
            "name": self.name,
            "duration": self.duration,
            "waiting_time": self.waiting_time,
            "waiting_acc": self.waiting_acc,
            "join_time": self.join_time,
            "finish_time": self.finish_time,
            "op_kind": self.op_kind,
            "batch_id": self.batch_id,
            "wrank_id": self.wrank_id,
            "stage_id": self.stage_id,
            "mg_state": self.mg_state,
            "group_kind": self.group_kind,
            "mg_is_last_iteration": self.mg_is_last_iteration,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "description": self.description,
            "end_timestamp": self.end_timestamp,
            "hidden_duration": self.hidden_duration,
            "cmd_uid": self.cmd_uid,
            "op_semantics": self.op_semantics,
            "trace_metadata": self.trace_metadata,
            "name_with_id": self.name_with_id,
        }




class SubOperation(Operation):
    def __init__(self, name, op_kind, duration, wrank_id, group_kind, description, start_time, pt_start_time, pt_duration, tensor_shape, tensor_dtype, trace_src_func, comm_func, name_with_id):
        """ 可以调整op_kind来控制vis是否正常显示op """
        super().__init__(name=name, op_kind=op_kind, duration=duration, wrank_id=wrank_id, group_kind=group_kind, description=description, tensor_shape=tensor_shape, tensor_dtype=tensor_dtype)
        self.is_suboperation = True
        self.start_time = start_time
        self.pt_start_time = pt_start_time
        self.pt_duration = pt_duration
        # self.tensor_shape = tensor_shape
        # self.tensor_dtype = tensor_dtype
        self.trace_src_func = trace_src_func
        self.comm_func = comm_func
        self.name_with_id = name_with_id # 用于ini_3d_ranks阶段从trace中填补single profile不存在的op（便于找到对应的subop）

    def __str__(self):
        return (f"SubOperation(name={self.name}, op_kind={self.op_kind}, duration={self.duration}, "
                f"wrank_id={self.wrank_id}, group_kind={self.group_kind}, description={self.description}, "
                f"start_time={self.start_time}, pt_start_time={self.pt_start_time}, pt_duration={self.pt_duration}, "
                f"tensor_shape={self.tensor_shape}, tensor_dtype={self.tensor_dtype}, "
                f"trace_src_func={self.trace_src_func}, comm_func={self.comm_func}, name_with_id={self.name_with_id})")


    def to_dict(self):
        return {
            "name": self.name_with_id,
            "duration": self.duration,
            "waiting_time": self.waiting_time,
            "waiting_acc": self.waiting_acc,
            "join_time": self.join_time,
            "finish_time": self.finish_time,
            "op_kind": self.op_kind,
            "batch_id": self.batch_id,
            "wrank_id": self.wrank_id,
            "stage_id": self.stage_id,
            "mg_state": self.mg_state,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "group_kind": self.group_kind,
            "mg_is_last_iteration": self.mg_is_last_iteration,
            "end_timestamp": self.end_timestamp,
            "start_time": self.start_time,
            "pt_duration": self.pt_duration,
            "comm_func": self.comm_func,
            "description": self.description
        }





class Stage:
    """ 一个完整stage包含多个Operation obj"""
    def __init__(self, wrank_id, rank: RankZoo, stage_id, steps_num=None, framework=None):
        self.wrank_id = wrank_id # i.e., world_rank (唯一id)
        self.rank: RankZoo = rank
        self.stage_id = stage_id
        self.steps_num = steps_num
        self.operations_list = []
        self.pre_stage = None
        self.post_stage = None
        self.stage_kind = None # 暂时在IndividualTimeline中初始化
        self.framework = framework

    def __str__(self):
        operations = '\n'.join([str(op) for op in self.operations_list])
        return f"Wrank id: {self.wrank_id}\nStage id: {self.stage_id}\nStage Kind: {self.stage_kind}\nOperations:\n{operations}"
    
    def add_operations_to_list(self, operation: Operation):
        self.operations_list.append(operation)

    def add_op_list_to_list(self, op_list: list):
        self.operations_list.extend(op_list)

    def set_pre_stage(self, pre_stage):
        self.pre_stage = pre_stage
    
    def set_post_stage(self, post_stage):
        self.post_stage = post_stage
    
    def set_stage_kind(self, stage_kind: str):
        assert stage_kind in ['FirstStage', 'MiddleStage', 'LastStage'], "Invalid stage kind"
        self.stage_kind = stage_kind

    def total_duration(self):
        """ Calculate the total duration of all operations in this stage """
        return sum(op.duration for op in self.operations_list)

    def set_stage_wrank_id(self, wrank_id):
        self.wrank_id = int(wrank_id)

    def set_stage_rank(self, rank: RankZoo):
        if self.wrank_id is not None:
            if rank.world_rank != self.wrank_id:
                raise ValueError("rank.world_rank and wrank_id must be consistent")
        self.rank = rank


class IndividualTimeline:
    """ 管理当前stage下的timeline, 每个Stage是一个RANK """
    def __init__(self, stage: Stage, can_overlap=False):
        ''' pre_stage、post_stage'''
        self.wrank_id = stage.wrank_id # i.e., world_rank (唯一id)
        self.stage_id = stage.stage_id
        self.pre_stage = stage.pre_stage
        self.post_stage = stage.post_stage
        self.stage_kind = None
        self.stage_rank: RankZoo = stage.rank
        self.pre_individual_timeline: IndividualTimeline = None
        self.post_individual_timeline: IndividualTimeline = None 
        self.comm_waiting_pool = {} # comm.加入需要特殊处理, key：匹配操作名称, val: Operation对象
        self.can_overlap = can_overlap
        self.is_comm_blocked = False # 等价于 是否有通信操作已经挂起等待；已注册但未完成的comm. ops会被阻塞,等待对方解除阻塞。对于comp. ops,不会出现这种情况

        self.comp_timeline = []
        self.comm_timeline = []
        self.final_merge_timeline = []

        self.waiting_queue = deque() # start_pop_operators实际处理的是waiting_queue, 所以stage.operations_list被留存下来了？
        self.final_package_operation = [] # deprecated：final_package_operation最后逐一处理
        for operation in stage.operations_list:
            if stage.framework == "deepspeed":
                # if operation.name in DS_FINAL_OPERATION:
                #     self.final_package_operation.append(copy.deepcopy(operation))
                # else:
                self.waiting_queue.append(copy.deepcopy(operation))
            else:
                self.waiting_queue.append(copy.deepcopy(operation))


    def _set_is_blocked_sign(self, bool_value):
        self.is_comm_blocked = bool_value

    def _add_comm_op_to_timeline(self, op_list, need_fuse=False):
        if need_fuse:
            raise ValueError(f"Fusion of comm. ops is not supported.")
        else:
            self.comm_timeline.extend(op_list)
            # new add
            self.final_merge_timeline.extend(op_list)
            


    def _get_last_op_waiting_acc(self, ignore_timeline_kind: True):
        if ignore_timeline_kind:
            # 取出2个timeline中最后的操作的waiting_acc
            _, last_operation = self._get_last_operation_time_and_op([self.comm_timeline, self.comp_timeline])
            if last_operation is None or last_operation.waiting_acc is None:
                return 0
            return last_operation.waiting_acc
        else:
            raise ValueError(f"Special kind of _get_last_op_waiting_acc is not supported.")

    def _get_last_operation_time_and_op(self, check_timelines: list):
        """ 返回的是check_timelines中最迟的operation的finish time和对应的operation """
        if not check_timelines or all(not sublist for sublist in check_timelines):
            return 0, None  
        
        # Flatten the list and filter out empty sublists
        all_operations = [op for sublist in check_timelines if sublist for op in sublist]

        if not all_operations:
            return 0, None  # Return 0 and None if there are no operations
        
        # Find the operation with the maximum finish_time

        last_operation = max(all_operations, key=lambda op: op.finish_time)

        return last_operation.finish_time, last_operation


class TimelinesManager:
    """ 管理所有stages的timeline """
    def __init__(self, dependency_relationship, comm_matching_relationship, compelete_wranks_list: list,
                  strategy: str='1F1B-none_interleaved', can_overlap=False, global_waiting_pool={},
                  global_finished_operations={},mpu_info=None,running_mode=None, torch_graph_stage_op_dict=None,
                  trace_filepath=None, torchgraph_filepath=None,not_simulating_cmd_dict=None,rank_instances_dict=None,trace_stages_dict=None,
                  optimization_enabled=True, is_moe_model=False, selected_ranks=None, cc_estimator=None, simulator_config=None, cc_backend=None):

        self.compelete_wranks_list = sorted(compelete_wranks_list, key=lambda s: s.wrank_id)  # 根据wrank_id进行排序
        self.strategy = strategy
        self.running_mode = running_mode
        self.mpu_info = mpu_info
        self.trace_filepath = trace_filepath
        self.torchgraph_filepath = torchgraph_filepath
        self.torch_graph_stage_op_dict = torch_graph_stage_op_dict
        self.trace_stages_dict = trace_stages_dict
        self.rank_instances_dict = rank_instances_dict
        self.can_overlap = can_overlap
        self.dependency_relationship: dict = dependency_relationship
        self.comm_matching_relationship: dict = comm_matching_relationship
        self.not_simulating_cmd_dict = not_simulating_cmd_dict
        self.global_waiting_pool: dict = global_waiting_pool # 同类型之间的操作不重叠，因此暂时维护一个全局的comm pool
        self.global_finished_operations: dict = global_finished_operations # global_finished_operations = {wrank_id_operation.name_operation.batch_id: Operation obj, ...}, e.g., {1_ForwardPass_0: Operation obj, 3_RecvActivation_0: Operation obj, ...

        # 优化相关参数
        self.optimization_enabled = optimization_enabled
        self.is_moe_model = is_moe_model
        self.selected_ranks = selected_ranks if selected_ranks else set()

        # CC-estimator integration
        self.cc_estimator = cc_estimator
        self.simulator_config = simulator_config
        self.cc_backend = cc_backend
        self.completed_cmd_operations_by_uid = {}
        self.pending_ddp_wait_finish_times_by_uid = {}
        self.pending_ddp_unassigned_finish_times_by_rank = {}
        self.async_ddp_comm_available_by_rank = {}

        slowdown_config = getattr(self.simulator_config, 'slowdown', None)
        self.slowdown_requested = bool(getattr(slowdown_config, 'enabled', False))
        self.slowdown_enabled = False
        self.slowdown_assets = None
        self.slowdown_predictor = None
        self.slowdown_runtime_comm_schedules_by_uid = {}
        self.slowdown_processed_backward_cmd_uids = set()
        self.slowdown_trigger_cmd_uids = set()

        self.stages_timeline_process_dict = self._init_stages_timeline()
        has_trace_driven_ddp_overlap = self._has_trace_driven_ddp_overlap()
        self._apply_trace_overlap_policy(has_trace_driven_ddp_overlap)
        self._initialize_slowdown_runtime(has_trace_driven_ddp_overlap)

    def _set_timeline_can_overlap(self, enabled: bool) -> None:
        self.can_overlap = enabled
        for timeline in self.stages_timeline_process_dict.values():
            timeline.can_overlap = enabled

    def _apply_trace_overlap_policy(self, has_trace_driven_ddp_overlap: Optional[bool] = None) -> None:
        overlap_config = getattr(self.simulator_config, 'overlap', None)
        overlap_mode = getattr(overlap_config, 'mode', 'auto') or 'auto'
        if overlap_mode not in {'auto', 'on', 'off'}:
            raise ValueError(f'Unsupported overlap mode: {overlap_mode}')

        if has_trace_driven_ddp_overlap is None:
            has_trace_driven_ddp_overlap = self._has_trace_driven_ddp_overlap()

        if overlap_mode == 'off':
            if has_trace_driven_ddp_overlap:
                raise ValueError(
                    'Trace-driven DDP overlap metadata is present but overlap mode is force-disabled.'
                )
            self._set_timeline_can_overlap(False)
            return

        if overlap_mode == 'on':
            if not has_trace_driven_ddp_overlap:
                raise ValueError(
                    'Overlap mode is forced on but no trace-driven DDP overlap overlay is present in the loaded timelines.'
                )
            self._set_timeline_can_overlap(True)
            return

        self._set_timeline_can_overlap(bool(self.can_overlap or has_trace_driven_ddp_overlap))

    def _initialize_slowdown_runtime(self, has_trace_driven_ddp_overlap: Optional[bool] = None) -> None:
        if not self.slowdown_requested:
            return
        if self.running_mode != MODE_SIMULATE:
            raise ValueError('Slowdown prediction is only supported in MODE_SIMULATE.')

        if has_trace_driven_ddp_overlap is None:
            has_trace_driven_ddp_overlap = self._has_trace_driven_ddp_overlap()
        if not has_trace_driven_ddp_overlap:
            logger.warning(
                'Slowdown requested but no trace-driven DDP overlap overlay is present in the loaded timelines. '
                'Slowdown is disabled and simulation continues without slowdown.'
            )
            return

        slowdown_config = getattr(self.simulator_config, 'slowdown', None)
        assets_dir = getattr(slowdown_config, 'assets_dir', None)
        if not isinstance(assets_dir, str) or not assets_dir:
            raise ValueError('Slowdown enabled but simulator_config.slowdown.assets_dir is missing.')
        self.slowdown_assets = load_slowdown_assets(assets_dir)
        override_model_path = getattr(slowdown_config, 'model_path', None)
        override_scaler_path = getattr(slowdown_config, 'scaler_path', None)
        model_path = override_model_path or self.slowdown_assets.manifest.get('model_path', None)
        scaler_path = override_scaler_path or self.slowdown_assets.manifest.get('scaler_path', None)
        if not isinstance(model_path, str) or not model_path:
            raise ValueError('Slowdown enabled but no slowdown model_path is configured.')
        if not isinstance(scaler_path, str) or not scaler_path:
            raise ValueError('Slowdown enabled but no slowdown scaler_path is configured.')
        if not os.path.isabs(model_path):
            if override_model_path and os.path.exists(model_path):
                model_path = os.path.abspath(model_path)
            else:
                model_path = os.path.join(assets_dir, model_path)
        if not os.path.exists(model_path):
            raise ValueError(f'Slowdown model path does not exist: {model_path}')
        if not os.path.isabs(scaler_path):
            if override_scaler_path and os.path.exists(scaler_path):
                scaler_path = os.path.abspath(scaler_path)
            else:
                scaler_path = os.path.join(assets_dir, scaler_path)
        if not os.path.exists(scaler_path):
            raise ValueError(f'Slowdown scaler path does not exist: {scaler_path}')
        self.slowdown_predictor = EchoSlowdownPredictor(model_path, scaler_path=scaler_path)
        self.slowdown_enabled = True

        self.slowdown_trigger_cmd_uids = set()
        for timeline in self.stages_timeline_process_dict.values():
            for pending_operation in timeline.waiting_queue:
                trace_metadata = getattr(pending_operation, 'trace_metadata', {}) or {}
                if trace_metadata.get('trace_event_type', None) != 'ddp_grad_comm':
                    continue
                trigger_cmd_uid = trace_metadata.get('trigger_cmd_uid', None)
                if trigger_cmd_uid is None:
                    raise ValueError('Slowdown-enabled DDP overlap comm is missing trigger_cmd_uid.')
                self.slowdown_trigger_cmd_uids.add(trigger_cmd_uid)

        missing_cmd_uids = sorted(
            cmd_uid
            for cmd_uid in self.slowdown_trigger_cmd_uids
            if cmd_uid not in self.slowdown_assets.backward_kernel_blueprints
        )
        if missing_cmd_uids:
            raise ValueError(
                'Slowdown assets are missing backward blueprints for trigger cmd_uids: '
                + ', '.join(missing_cmd_uids)
            )

    def _init_stages_timeline(self):
        stages_timeline_process_dict = {}
        # Create IndividualTimeline objects for all stages
        for Stage in self.compelete_wranks_list:
            timeline = IndividualTimeline(stage=Stage, can_overlap=self.can_overlap)
            stages_timeline_process_dict[Stage.wrank_id] = timeline

        if getattr(self.mpu_info, "pp_size", 1) == 1:
            for timeline in stages_timeline_process_dict.values():
                timeline.stage_kind = 'SingleStage'
            return stages_timeline_process_dict

        # Set the pre and post individual timelines based on the PP groups
        # 只处理选中的ranks
        for pp_group in self.mpu_info.pp_groups:
            # 过滤出在当前pp_group中且被选中的ranks
            selected_ranks_in_group = [wrank_id for wrank_id in pp_group if wrank_id in stages_timeline_process_dict]

            if not selected_ranks_in_group:
                continue  # 如果这个PP group中没有选中的ranks，跳过

            for index, wrank_id in enumerate(selected_ranks_in_group):
                timeline = stages_timeline_process_dict[wrank_id]
                if index == 0:
                    timeline.stage_kind = 'FirstStage'
                elif index == len(selected_ranks_in_group) - 1:
                    timeline.stage_kind = 'LastStage'
                else:
                    timeline.stage_kind = 'MiddleStage'

        return stages_timeline_process_dict

    def start_pop_operators(self):
        # TODO： 
        #   0. pp offset +1 -1 补充函数来获取对应对应的stage id(_get_comm_matching_operation_name_and_offset修改);
        #   3. 模块化 P2P allreduce broadcast几种操作，抽象出其逻辑，使得代码根据不同种类进行类似操作
        #   4. 对于module内的comm. op：增加fwd、bwd的转换模块，将其分解为更加细粒度的操作（TP需求）
        #   5. 对于module外的comm. op：对于非overlap-reduce情况，直接将操作写到流程中记录，不做额外处理；对于
        #       overlap-reduce情况，单独处理overlap部分，主要涉及bwd模块转换（参考4）
        #   6. 每种comm操作在name中标注了所被触发的并行维度（pp的p2p的暂未标记），comm在查询group内注册情况时统一到global_pool的对应维度下查看数量即可判定注册情况（当前group-1数即代表都注册完成，否则还需要等待最后op加入，这里需要用lock）

        """ 根据策略模拟全局pipeline并行流程 """
        if self.running_mode == MODE_PROFILE and self._has_trace_driven_ddp_overlap():
            self._replay_profile_no_pipelining()
            return

        if (
            self.running_mode == MODE_SIMULATE
            and self.strategy == 'no-pipelining'
            and getattr(self.mpu_info, 'pp_size', 1) == 1
            and self._has_trace_driven_ddp_overlap()
        ):
            self._replay_profile_no_pipelining()
            return

        if self.running_mode == MODE_PROFILE:
            # PROFILE模式：直接按顺序播放每个rank的trace，但过滤单GPU通信操作
            # 特殊处理并发的P2P通信操作对
            for wrank_id in sorted(self.stages_timeline_process_dict.keys()):
                timeline: IndividualTimeline = self.stages_timeline_process_dict[wrank_id]
                last_op_finish_time = 0
                operations_list = list(timeline.waiting_queue)
                i = 0

                while i < len(operations_list):
                    operation = operations_list[i]

                    # 过滤单GPU通信操作
                    if self._should_filter_single_gpu_comm(operation):
                        print(f"Debug: 过滤单GPU通信操作 {operation.name} (rank {wrank_id})")
                        i += 1
                        continue

                    # 检查是否是并发的P2P操作对
                    concurrent_op = None
                    if (
                        operation.op_kind == 'comm'
                        and operation.mg_state == 'steady'
                        and operation.name in ['send_forward', 'recv_backward', 'send_backward', 'recv_forward']
                        and i + 1 < len(operations_list)
                    ):

                        next_operation = operations_list[i + 1]
                        if (
                            next_operation.op_kind == 'comm'
                            and next_operation.mg_state == 'steady'
                            and next_operation.name in ['send_forward', 'recv_backward', 'send_backward', 'recv_forward']
                        ):

                            # 检查是否有相同的timestamp和duration（表示并发执行）
                            same_timestamp = (
                                operation.end_timestamp is not None
                                and next_operation.end_timestamp is not None
                                and abs(operation.end_timestamp - next_operation.end_timestamp) < 0.01
                            )
                            same_duration = abs(operation.duration - next_operation.duration) < 0.01

                            # 对于P2P操作对，不要求相同的batch_id，因为send_forward_recv_backward中
                            # send是发送当前batch，recv是接收上一个batch的梯度
                            if same_timestamp and same_duration:
                                # 验证是否是正确的P2P操作对（send_forward_recv_backward 或 send_backward_recv_forward）
                                if (
                                    (operation.name == 'send_forward' and next_operation.name == 'recv_backward')
                                    or (operation.name == 'recv_backward' and next_operation.name == 'send_forward')
                                    or (operation.name == 'send_backward' and next_operation.name == 'recv_forward')
                                    or (operation.name == 'recv_forward' and next_operation.name == 'send_backward')
                                ):
                                    concurrent_op = next_operation
                                    print(
                                        f"Debug: 识别到并发P2P操作对 {operation.name} (batch {operation.batch_id}) + {next_operation.name} (batch {next_operation.batch_id}) (rank {wrank_id}, timestamp {operation.end_timestamp})"
                                    )

                    if concurrent_op:
                        # 处理并发的P2P操作对：它们同时开始和结束
                        operation.comp_set_join_finish_waiting_acc_time(last_op_finish_time, 0)
                        concurrent_op.comp_set_join_finish_waiting_acc_time(last_op_finish_time, 0)

                        # 添加到时间线
                        timeline.final_merge_timeline.append(operation)
                        timeline.final_merge_timeline.append(concurrent_op)
                        timeline.comm_timeline.append(operation)
                        timeline.comm_timeline.append(concurrent_op)

                        last_op_finish_time = operation.finish_time  # 两个操作有相同的finish_time
                        i += 2  # 跳过下一个操作，因为已经处理了
                    else:
                        # 处理单个操作
                        operation.comp_set_join_finish_waiting_acc_time(last_op_finish_time, 0)
                        timeline.final_merge_timeline.append(operation)
                        if operation.op_kind == 'comp':
                            timeline.comp_timeline.append(operation)
                        else:  # comm
                            timeline.comm_timeline.append(operation)
                        last_op_finish_time = operation.finish_time
                        i += 1

            # Profile模式处理完毕，直接返回
            return

        """ 根据策略模拟全局pipeline并行流程 """
        if self.strategy == '1F1B-none_interleaved':
            # 当self.global_waiting_queue不为空,继续循环
            # 1. 从global_waiting_queue中pop第一个operation
            # 2. 检查operation的依赖operation是否已被完成（依赖关系可以从dependency_relationship中找到,被完成指的是不存在于global_waiting_queue中）, 
            #    如果已经完成,将operation加入到对应stage的timeline中（依据操作的类型,是comp还是comm）。在加入过程中,如果该operation有依赖操作,查找依赖操作的finish_time,然后max(finish_time, 当前类型timeline最后一个operation的finish_time)
            #    如果没有完成,put回队列,continue到下一个operation
            dependency_wait_error = "所以当某个operation被pop到了,应该不会发生依赖项未完成情况"
            while True:
                # 初始化一个标志,用于检查所有的waiting_queue是否都为空
                all_queues_empty = True
                progress_made = False

                # 对wrank_id进行排序
                # 每个IndividualTimeline对应一个rank：逐个地将waiting_queue中的op pop尝试加入到timelien中
                for wrank_id in sorted(self.stages_timeline_process_dict.keys()):
                    timeline: IndividualTimeline = self.stages_timeline_process_dict[wrank_id]

                    if timeline.waiting_queue:
                        all_queues_empty = False

                        # 检查当前timeline是否blocked
                        operation: Operation = timeline.waiting_queue.popleft()

                        is_timeline_blocked: bool = self._check_timelline_blocked_status(
                            timeline=timeline, operation=operation
                        )

                        if is_timeline_blocked:
                            # 返回队首等待
                            timeline.waiting_queue.appendleft(operation)
                            continue

                        try:
                            self._add_operation_to_timeline(timeline=timeline, operation=operation)
                            progress_made = True
                        except ValueError as exc:
                            if dependency_wait_error not in str(exc):
                                raise
                            # PP overlap / async comm 会让跨 stage 依赖暂时未满足；该情况应回队等待，
                            # 而不是立即视为非法。若一整轮都无进展，下面会 fail-fast 报死锁。
                            continue

                # 如果所有的waiting_queue都为空,退出循环
                if all_queues_empty:
                    break

                if not progress_made:
                    pending_ops = {}
                    for wrank_id, timeline in self.stages_timeline_process_dict.items():
                        if timeline.waiting_queue:
                            pending_ops[wrank_id] = [
                                f"{op.name}(batch={op.batch_id},state={op.mg_state})"
                                for op in list(timeline.waiting_queue)[:6]
                            ]
                    raise ValueError(
                        f"Deadlock detected in schedule replay: no progress made in a full scheduling pass. Pending operations: {pending_ops}"
                    )

        elif self.strategy == '1F1B-interleaved':
            pass

        elif self.strategy == 'F-then-B':
            pass

        elif self.strategy == 'no-pipelining':
            pass

    def _get_profile_recorded_start_ms(self, operation: Operation):
        trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
        if trace_metadata.get('trace_event_type') == 'ddp_grad_comm':
            launch_timestamp_ms = trace_metadata.get('launch_timestamp_ms', None)
            if launch_timestamp_ms is None:
                raise ValueError('ddp_grad_comm is missing launch_timestamp_ms in profile replay')
            return float(launch_timestamp_ms)

        if operation.end_timestamp is None or operation.duration is None:
            return None
        return round(float(operation.end_timestamp) - float(operation.duration), 2)

    def _has_trace_driven_ddp_overlap(self) -> bool:
        for timeline in self.stages_timeline_process_dict.values():
            for operation in timeline.waiting_queue:
                trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
                if trace_metadata.get('trace_event_type', None) == 'ddp_grad_comm':
                    return True
                if getattr(operation, 'op_semantics', None) in {'wait_flush_only', 'metadata_placeholder'}:
                    return True
        return False

    def _predict_profile_overlap_comm_duration(self, operation: Operation) -> float:
        self._calculate_comm_duration([operation])
        if operation.duration is None:
            raise ValueError(f'Failed to predict communication duration for {operation.name}')
        return float(
            self._apply_trace_driven_ddp_overlap_calibration(
                operation,
                float(operation.duration),
            )
        )

    def _apply_trace_driven_ddp_overlap_calibration(
        self,
        operation: Operation,
        predicted_duration: float,
    ) -> float:
        trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
        if trace_metadata.get('trace_event_type') != 'ddp_grad_comm':
            return float(predicted_duration)
        if not bool(trace_metadata.get('metadata_only', False)):
            return float(predicted_duration)
        if self.simulator_config is None or self.cc_backend is None:
            return float(predicted_duration)

        backend_name = getattr(self.cc_backend, 'backend_name', None)
        if backend_name is None:
            return float(predicted_duration)

        backend_options = get_backend_options(self.simulator_config, backend_name)
        min_duration_by_collective = backend_options.get(
            'ddp_overlap_min_duration_ms_by_collective', {}
        )
        scale_by_collective = backend_options.get(
            'ddp_overlap_duration_scale_by_collective', {}
        )
        if not isinstance(min_duration_by_collective, dict):
            raise ValueError(
                'ddp_overlap_min_duration_ms_by_collective must be a dict when configured'
            )
        if not isinstance(scale_by_collective, dict):
            raise ValueError(
                'ddp_overlap_duration_scale_by_collective must be a dict when configured'
            )

        collective_kind = infer_collective_kind(operation.name)
        calibrated_duration = float(predicted_duration)
        if collective_kind in scale_by_collective:
            calibrated_duration *= float(scale_by_collective[collective_kind])
        if collective_kind in min_duration_by_collective:
            calibrated_duration = max(
                calibrated_duration,
                float(min_duration_by_collective[collective_kind]),
            )
        return float(calibrated_duration)

    @staticmethod
    def _is_trace_driven_ddp_overlap_comm(operation: Operation) -> bool:
        trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
        return trace_metadata.get('trace_event_type', None) == 'ddp_grad_comm'

    @staticmethod
    def _is_trace_driven_backward(operation: Operation) -> bool:
        """Return whether a backward operation has a trace identity for slowdown replay.

        The generated scheduling plan contains every microbatch backward operation, while
        slowdown assets are built only for backward operations captured in the rank trace.
        Schedule-only operations therefore have no ``cmd_uid`` and must retain their normal
        profile duration instead of entering the trace-driven slowdown path.
        """
        return operation.name == 'backward_step' and getattr(operation, 'cmd_uid', None) is not None

    @staticmethod
    def _rank_scoped_uid(wrank_id: int, uid: str) -> tuple[int, str]:
        if wrank_id is None:
            raise ValueError(f'Cannot scope UID without wrank_id: uid={uid}')
        if uid in {None, 'None'}:
            raise ValueError(f'Cannot scope missing UID for wrank_id={wrank_id}')
        return int(wrank_id), uid

    def _record_completed_cmd_operation(self, operation: Operation) -> None:
        cmd_uid = getattr(operation, 'cmd_uid', None)
        if cmd_uid is not None:
            scoped_uid = self._rank_scoped_uid(getattr(operation, 'wrank_id', None), cmd_uid)
            self.completed_cmd_operations_by_uid[scoped_uid] = operation

    def _get_last_schedulable_comm_operation(self, timeline: IndividualTimeline):
        for recorded_operation in reversed(timeline.comm_timeline):
            if not self._is_trace_driven_ddp_overlap_comm(recorded_operation):
                return recorded_operation
        return None

    def _consume_pending_ddp_overlap_finish_times(self, timeline: IndividualTimeline, wait_cmd_uid: str):
        scoped_wait_uid = self._rank_scoped_uid(timeline.wrank_id, wait_cmd_uid)
        paired_finish_times = self.pending_ddp_wait_finish_times_by_uid.pop(scoped_wait_uid, None)
        if paired_finish_times:
            return paired_finish_times

        unassigned_finish_times = self.pending_ddp_unassigned_finish_times_by_rank.get(timeline.wrank_id, [])
        if unassigned_finish_times:
            self.pending_ddp_unassigned_finish_times_by_rank[timeline.wrank_id] = []
            return unassigned_finish_times

        return None

    @staticmethod
    def _fixed_point_kernel_duration_ms(
        *,
        ground_truth_ms: float,
        slowdown_factor: float,
        overlap_time_ms: float,
        max_iters: int,
        tol_ms: float,
    ) -> float:
        ground_truth_ms = float(ground_truth_ms)
        slowdown_factor = max(0.0, float(slowdown_factor))
        overlap_time_ms = max(0.0, float(overlap_time_ms))
        if ground_truth_ms < 0:
            raise ValueError(f'ground_truth_ms must be >= 0, got {ground_truth_ms}')
        if max_iters <= 0:
            raise ValueError(f'max_iters must be > 0, got {max_iters}')
        if tol_ms <= 0:
            raise ValueError(f'tol_ms must be > 0, got {tol_ms}')
        if ground_truth_ms == 0 or slowdown_factor == 0 or overlap_time_ms == 0:
            return round(ground_truth_ms, 6)

        predicted_duration_ms = float(ground_truth_ms)
        for _ in range(max_iters):
            overlap_ratio = min(1.0, overlap_time_ms / predicted_duration_ms)
            next_duration_ms = ground_truth_ms * (1.0 + overlap_ratio * slowdown_factor)
            if abs(next_duration_ms - predicted_duration_ms) <= tol_ms:
                return round(next_duration_ms, 6)
            predicted_duration_ms = next_duration_ms
        return round(predicted_duration_ms, 6)

    @staticmethod
    def _build_slowdown_feature_row(
        kernel_name: str,
        ground_truth_ms: float,
        kernel_features: dict,
    ) -> dict:
        if kernel_name not in kernel_features:
            raise ValueError(f'Missing slowdown kernel features for kernel {kernel_name!r}')
        feature_row = dict(kernel_features[kernel_name])
        feature_row['ground_truth'] = float(ground_truth_ms)
        return feature_row

    @staticmethod
    def _resolve_slowdown_feature_row(
        kernel_name: str,
        ground_truth_ms: float,
        kernel_features: dict,
    ) -> tuple:
        """Resolve a predictor row using an exact or unique canonical kernel name.

        Task3 accepts missing kernel profiling features as an explicit no-slowdown
        case. A canonical alias is used only when it identifies exactly one
        profiling row; ambiguous aliases are skipped instead of guessed.
        """
        if kernel_name in kernel_features:
            return (
                TimelinesManager._build_slowdown_feature_row(
                    kernel_name=kernel_name,
                    ground_truth_ms=ground_truth_ms,
                    kernel_features=kernel_features,
                ),
                kernel_name,
                'exact',
            )

        canonical_name = kernel_name.rsplit('::', 1)[-1].rsplit('/', 1)[-1]
        aliases = [
            candidate
            for candidate in kernel_features
            if candidate.rsplit('::', 1)[-1].rsplit('/', 1)[-1] == canonical_name
        ]
        if len(aliases) == 1:
            alias = aliases[0]
            feature_row = dict(kernel_features[alias])
            feature_row['ground_truth'] = float(ground_truth_ms)
            return feature_row, alias, f'canonical_alias:{alias}'

        return None, None, 'missing_skip'

    @staticmethod
    def _predict_kernel_slowdown_factor(
        predictor,
        kernel_name: str,
        ground_truth_ms: float,
        feature_row: dict,
    ) -> float:
        if predictor is None:
            raise ValueError('Slowdown predictor is not initialized.')
        if hasattr(predictor, 'predict_slowdown_factor'):
            slowdown_factor = predictor.predict_slowdown_factor(
                kernel_name=kernel_name,
                ground_truth_ms=ground_truth_ms,
                feature_row=feature_row,
            )
        elif hasattr(predictor, 'predict_clipped_slowdown_factor'):
            slowdown_factor = predictor.predict_clipped_slowdown_factor(feature_row)
        else:
            raise ValueError('Slowdown predictor does not expose a supported prediction API.')
        return max(0.0, float(slowdown_factor))

    @staticmethod
    def _schedule_ddp_comm_raw_launches(raw_launch_specs: list, comm_duration_by_uid: dict, initial_available_time_ms: float):
        scheduled = []
        available_time_ms = float(initial_available_time_ms)
        for spec in raw_launch_specs:
            comm_uid = spec['comm_uid']
            if comm_uid not in comm_duration_by_uid:
                raise ValueError(f'Missing predicted communication duration for comm_uid={comm_uid}')
            raw_launch_time_ms = float(spec['raw_launch_time_ms'])
            duration_ms = float(comm_duration_by_uid[comm_uid])
            launch_time_ms = max(raw_launch_time_ms, available_time_ms)
            finish_time_ms = launch_time_ms + duration_ms
            scheduled.append(
                {
                    'comm_uid': comm_uid,
                    'raw_launch_time_ms': round(raw_launch_time_ms, 6),
                    'launch_time_ms': round(launch_time_ms, 6),
                    'finish_time_ms': round(finish_time_ms, 6),
                    'duration_ms': round(duration_ms, 6),
                    'bucket_id': spec.get('bucket_id', None),
                    'buffer_id': spec.get('buffer_id', None),
                }
            )
            available_time_ms = finish_time_ms
        return scheduled, round(available_time_ms, 6)

    @staticmethod
    def _simulate_backward_slowdown_schedule(
        *,
        backward_start_time_ms: float,
        blueprint: dict,
        kernel_features: dict,
        predictor,
        comm_duration_by_uid: dict,
        max_iters: int,
        tol_ms: float,
    ) -> dict:
        kernels = list(blueprint.get('kernels', []))
        launch_markers = list(blueprint.get('launch_markers', []))
        if not kernels:
            raise ValueError('Slowdown blueprint must contain at least one kernel entry.')
        if max_iters <= 0:
            raise ValueError(f'max_iters must be > 0, got {max_iters}')
        if tol_ms <= 0:
            raise ValueError(f'tol_ms must be > 0, got {tol_ms}')

        sorted_kernels = sorted(kernels, key=lambda item: float(item['start_offset_ms']))
        sorted_markers = sorted(launch_markers, key=lambda item: float(item['baseline_offset_ms']))
        backward_start_time_ms = float(backward_start_time_ms)

        baseline_cursor_ms = 0.0
        current_time_ms = backward_start_time_ms
        comm_available_time_ms = backward_start_time_ms
        active_comm_intervals_ms = []
        scheduled_comms = {}
        kernel_schedules = []
        marker_index = 0
        epsilon = 1e-9

        for kernel_index, kernel in enumerate(sorted_kernels):
            kernel_name = kernel['kernel_name']
            kernel_baseline_start_ms = float(kernel['start_offset_ms'])
            ground_truth_ms = float(kernel['baseline_duration_ms'])
            kernel_baseline_end_ms = kernel_baseline_start_ms + ground_truth_ms

            if kernel_baseline_start_ms + epsilon < baseline_cursor_ms:
                raise ValueError(
                    f'Slowdown blueprint kernels are not monotonic around kernel {kernel_name!r}'
                )

            gap_ms = max(0.0, kernel_baseline_start_ms - baseline_cursor_ms)
            gap_raw_launch_specs = []
            while marker_index < len(sorted_markers):
                marker_offset_ms = float(sorted_markers[marker_index]['baseline_offset_ms'])
                if marker_offset_ms > kernel_baseline_start_ms + epsilon:
                    break
                gap_raw_launch_specs.append(
                    {
                        'comm_uid': sorted_markers[marker_index]['comm_uid'],
                        'raw_launch_time_ms': current_time_ms + max(0.0, marker_offset_ms - baseline_cursor_ms),
                        'bucket_id': sorted_markers[marker_index].get('bucket_id', None),
                        'buffer_id': sorted_markers[marker_index].get('buffer_id', None),
                    }
                )
                marker_index += 1

            if gap_raw_launch_specs:
                gap_schedules, comm_available_time_ms = TimelinesManager._schedule_ddp_comm_raw_launches(
                    gap_raw_launch_specs,
                    comm_duration_by_uid,
                    comm_available_time_ms,
                )
                for schedule in gap_schedules:
                    comm_uid = schedule['comm_uid']
                    if comm_uid in scheduled_comms:
                        raise ValueError(f'Duplicate scheduled comm_uid found: {comm_uid}')
                    scheduled_comms[comm_uid] = schedule
                    active_comm_intervals_ms.append(
                        (schedule['launch_time_ms'], schedule['finish_time_ms'])
                    )

            kernel_start_time_ms = current_time_ms + gap_ms
            kernel_marker_specs = []
            while marker_index < len(sorted_markers):
                marker_offset_ms = float(sorted_markers[marker_index]['baseline_offset_ms'])
                is_last_kernel = kernel_index == len(sorted_kernels) - 1
                within_current_kernel = marker_offset_ms < kernel_baseline_end_ms - epsilon
                if is_last_kernel:
                    within_current_kernel = marker_offset_ms <= kernel_baseline_end_ms + epsilon
                if not within_current_kernel:
                    break
                marker_fraction = 0.0
                if ground_truth_ms > 0:
                    marker_fraction = max(
                        0.0,
                        min(
                            1.0,
                            (marker_offset_ms - kernel_baseline_start_ms) / ground_truth_ms,
                        ),
                    )
                kernel_marker_specs.append(
                    {
                        'comm_uid': sorted_markers[marker_index]['comm_uid'],
                        'marker_fraction': marker_fraction,
                        'bucket_id': sorted_markers[marker_index].get('bucket_id', None),
                        'buffer_id': sorted_markers[marker_index].get('buffer_id', None),
                    }
                )
                marker_index += 1

            feature_row, feature_kernel_name, slowdown_feature_source = TimelinesManager._resolve_slowdown_feature_row(
                kernel_name=kernel_name,
                ground_truth_ms=ground_truth_ms,
                kernel_features=kernel_features,
            )
            if feature_row is None:
                slowdown_factor = 0.0
            else:
                slowdown_factor = TimelinesManager._predict_kernel_slowdown_factor(
                    predictor=predictor,
                    kernel_name=feature_kernel_name,
                    ground_truth_ms=ground_truth_ms,
                    feature_row=feature_row,
                )

            predicted_duration_ms = float(ground_truth_ms)
            final_kernel_comm_schedules = []
            for _ in range(max_iters):
                raw_launch_specs = []
                for marker_spec in kernel_marker_specs:
                    raw_launch_specs.append(
                        {
                            'comm_uid': marker_spec['comm_uid'],
                            'raw_launch_time_ms': kernel_start_time_ms
                            + marker_spec['marker_fraction'] * predicted_duration_ms,
                            'bucket_id': marker_spec.get('bucket_id', None),
                            'buffer_id': marker_spec.get('buffer_id', None),
                        }
                    )
                tentative_schedules, tentative_available_time_ms = TimelinesManager._schedule_ddp_comm_raw_launches(
                    raw_launch_specs,
                    comm_duration_by_uid,
                    comm_available_time_ms,
                )
                candidate_intervals_ms = list(active_comm_intervals_ms)
                candidate_intervals_ms.extend(
                    (schedule['launch_time_ms'], schedule['finish_time_ms'])
                    for schedule in tentative_schedules
                )
                overlap_time_ms = compute_overlap_time_ms(
                    kernel_start_ms=kernel_start_time_ms,
                    kernel_duration_ms=predicted_duration_ms,
                    active_intervals_ms=candidate_intervals_ms,
                )
                next_duration_ms = TimelinesManager._fixed_point_kernel_duration_ms(
                    ground_truth_ms=ground_truth_ms,
                    slowdown_factor=slowdown_factor,
                    overlap_time_ms=overlap_time_ms,
                    max_iters=1,
                    tol_ms=tol_ms,
                )
                final_kernel_comm_schedules = tentative_schedules
                if abs(next_duration_ms - predicted_duration_ms) <= tol_ms:
                    predicted_duration_ms = float(next_duration_ms)
                    comm_available_candidate_ms = tentative_available_time_ms
                    break
                predicted_duration_ms = float(next_duration_ms)
                comm_available_candidate_ms = tentative_available_time_ms
            else:
                comm_available_candidate_ms = comm_available_time_ms

            kernel_finish_time_ms = kernel_start_time_ms + predicted_duration_ms
            for schedule in final_kernel_comm_schedules:
                comm_uid = schedule['comm_uid']
                if comm_uid in scheduled_comms:
                    raise ValueError(f'Duplicate scheduled comm_uid found: {comm_uid}')
                scheduled_comms[comm_uid] = schedule
                active_comm_intervals_ms.append((schedule['launch_time_ms'], schedule['finish_time_ms']))
            comm_available_time_ms = comm_available_candidate_ms
            kernel_schedules.append(
                {
                    'kernel_name': kernel_name,
                    'start_time_ms': round(kernel_start_time_ms, 6),
                    'finish_time_ms': round(kernel_finish_time_ms, 6),
                    'baseline_duration_ms': round(ground_truth_ms, 6),
                    'predicted_duration_ms': round(predicted_duration_ms, 6),
                    'slowdown_factor': round(slowdown_factor, 6),
                    'slowdown_feature_source': slowdown_feature_source,
                }
            )
            current_time_ms = kernel_finish_time_ms
            baseline_cursor_ms = kernel_baseline_end_ms

        if marker_index != len(sorted_markers):
            remaining = [sorted_markers[index]['comm_uid'] for index in range(marker_index, len(sorted_markers))]
            raise ValueError(
                'Slowdown blueprint contains launch markers that were not scheduled: ' + ', '.join(remaining)
            )

        total_baseline_duration_ms = float(blueprint.get('baseline_duration_ms', baseline_cursor_ms))
        if total_baseline_duration_ms + epsilon < baseline_cursor_ms:
            raise ValueError(
                f"Slowdown blueprint baseline_duration_ms={total_baseline_duration_ms} is smaller than covered kernel baseline={baseline_cursor_ms}"
            )
        residual_duration_ms = max(0.0, total_baseline_duration_ms - baseline_cursor_ms)
        current_time_ms += residual_duration_ms

        return {
            'backward_duration_ms': round(current_time_ms - backward_start_time_ms, 6),
            'kernel_schedules': kernel_schedules,
            'comm_schedules': scheduled_comms,
            'residual_duration_ms': round(residual_duration_ms, 6),
        }

    def _get_pending_trace_driven_ddp_overlap_comms(self, timeline: IndividualTimeline, trigger_cmd_uid: str):
        overlay_operations = []
        for pending_operation in timeline.waiting_queue:
            if not self._is_trace_driven_ddp_overlap_comm(pending_operation):
                continue
            trace_metadata = getattr(pending_operation, 'trace_metadata', {}) or {}
            if trace_metadata.get('trigger_cmd_uid', None) == trigger_cmd_uid:
                overlay_operations.append(pending_operation)
        return overlay_operations

    def _add_trace_driven_backward_slowdown(
        self,
        timeline: IndividualTimeline,
        operation: Operation,
        join_time: float,
        current_format_operantion_name: str,
    ) -> None:
        scoped_cmd_uid = self._rank_scoped_uid(timeline.wrank_id, operation.cmd_uid)
        if scoped_cmd_uid in self.slowdown_processed_backward_cmd_uids:
            raise ValueError(f'Backward slowdown already processed for cmd_uid={operation.cmd_uid}')
        if operation.hidden_duration not in {None, 0, 0.0}:
            raise ValueError(
                'Slowdown v1 does not support sub-op-expanded backward_step; '
                f'cmd_uid={operation.cmd_uid} still carries hidden_duration={operation.hidden_duration}'
            )
        blueprint = self.slowdown_assets.backward_kernel_blueprints.get(operation.cmd_uid, None)
        if blueprint is None:
            raise ValueError(
                f'Slowdown assets are missing a backward kernel blueprint for cmd_uid={operation.cmd_uid}'
            )

        overlay_operations = self._get_pending_trace_driven_ddp_overlap_comms(timeline, operation.cmd_uid)
        if not overlay_operations:
            raise ValueError(
                f'Slowdown-enabled backward_step cmd_uid={operation.cmd_uid} has no paired ddp_grad_comm overlay operations.'
            )

        comm_duration_by_uid = {}
        overlay_comm_uids = set()
        for overlay_operation in overlay_operations:
            comm_uid = getattr(overlay_operation, 'cmd_uid', None)
            if comm_uid is None:
                raise ValueError('Slowdown-enabled ddp_grad_comm overlay is missing comm_uid.')
            if comm_uid in overlay_comm_uids:
                raise ValueError(f'Duplicate ddp_grad_comm overlay comm_uid found: {comm_uid}')
            overlay_comm_uids.add(comm_uid)
            overlay_copy = copy.deepcopy(overlay_operation)
            comm_duration_by_uid[comm_uid] = round(
                float(self._predict_profile_overlap_comm_duration(overlay_copy)),
                6,
            )

        blueprint_comm_uids = {marker['comm_uid'] for marker in blueprint['launch_markers']}
        if overlay_comm_uids != blueprint_comm_uids:
            raise ValueError(
                f'Slowdown blueprint/overlay comm_uid mismatch for cmd_uid={operation.cmd_uid}: '
                f'overlay={sorted(overlay_comm_uids)}, blueprint={sorted(blueprint_comm_uids)}'
            )

        slowdown_config = self.simulator_config.slowdown
        schedule_result = self._simulate_backward_slowdown_schedule(
            backward_start_time_ms=join_time,
            blueprint=blueprint,
            kernel_features=self.slowdown_assets.kernel_features,
            predictor=self.slowdown_predictor,
            comm_duration_by_uid=comm_duration_by_uid,
            max_iters=slowdown_config.max_iters,
            tol_ms=slowdown_config.tol_ms,
        )

        operation.duration = float(schedule_result['backward_duration_ms'])
        operation.hidden_duration = None
        operation.trace_metadata = dict(getattr(operation, 'trace_metadata', {}) or {})
        operation.trace_metadata['slowdown_applied'] = True
        operation.trace_metadata['slowdown_kernel_count'] = len(schedule_result['kernel_schedules'])
        operation.comp_set_join_finish_waiting_acc_time(join_time, 0)
        timeline.comp_timeline.append(operation)
        timeline.final_merge_timeline.append(operation)
        self.global_finished_operations[current_format_operantion_name] = operation
        self._record_completed_cmd_operation(operation)
        self.slowdown_processed_backward_cmd_uids.add(scoped_cmd_uid)

        for comm_uid, schedule in schedule_result['comm_schedules'].items():
            scoped_comm_uid = self._rank_scoped_uid(timeline.wrank_id, comm_uid)
            self.slowdown_runtime_comm_schedules_by_uid[scoped_comm_uid] = schedule

    def _add_trace_driven_async_ddp_overlap_comm(self, timeline: IndividualTimeline, operation: Operation) -> None:
        trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
        if not bool(trace_metadata.get('metadata_only', False)):
            raise ValueError('MODE_SIMULATE expects metadata-only ddp_grad_comm overlay records')

        comm_uid = getattr(operation, 'cmd_uid', None)
        if comm_uid is None:
            raise ValueError('ddp_grad_comm overlay record is missing comm_uid/cmd_uid')
        scoped_comm_uid = self._rank_scoped_uid(timeline.wrank_id, comm_uid)

        if self.slowdown_enabled:
            scheduled_comm = self.slowdown_runtime_comm_schedules_by_uid.pop(scoped_comm_uid, None)
            if scheduled_comm is None:
                raise ValueError(
                    f'Slowdown-enabled ddp_grad_comm overlay record {operation.name_with_id} '
                    f'is missing a precomputed schedule for comm_uid={comm_uid}'
                )
            operation.comm_set_join_time(float(scheduled_comm['launch_time_ms']))
            operation.duration = round(float(scheduled_comm['duration_ms']), 6)
            operation.hidden_duration = operation.duration
            waiting_time_ms = max(
                0.0,
                float(scheduled_comm['finish_time_ms'])
                - float(scheduled_comm['launch_time_ms'])
                - float(scheduled_comm['duration_ms']),
            )
            operation.comm_set_waiting_finish_time(waiting_time_ms)
            operation.comm_set_waiting_acc_time(
                operation.waiting_time,
                timeline._get_last_op_waiting_acc(True),
            )
            timeline.comm_timeline.append(operation)
            timeline.final_merge_timeline.append(operation)
            self.async_ddp_comm_available_by_rank[timeline.wrank_id] = operation.finish_time
        else:
            trigger_cmd_uid = trace_metadata.get('trigger_cmd_uid', None)
            if trigger_cmd_uid is None:
                raise ValueError('ddp_grad_comm overlay record is missing trigger_cmd_uid')

            scoped_trigger_uid = self._rank_scoped_uid(timeline.wrank_id, trigger_cmd_uid)
            trigger_operation = self.completed_cmd_operations_by_uid.get(scoped_trigger_uid, None)
            if trigger_operation is None:
                raise ValueError(
                    f'ddp_grad_comm overlay record {operation.name_with_id} cannot find trigger operation {trigger_cmd_uid}'
                )

            trigger_recorded_start_ms = self._get_profile_recorded_start_ms(trigger_operation)
            if trigger_recorded_start_ms is None:
                raise ValueError(
                    f'Trigger operation {trigger_cmd_uid} is missing trace start metadata for ddp_grad_comm replay'
                )

            launch_timestamp_ms = trace_metadata.get('launch_timestamp_ms', None)
            if launch_timestamp_ms is None:
                raise ValueError('ddp_grad_comm overlay record is missing launch_timestamp_ms')

            launch_offset_ms = round(float(launch_timestamp_ms) - float(trigger_recorded_start_ms), 2)
            if launch_offset_ms < -0.05:
                raise ValueError(
                    f'ddp_grad_comm overlay record {operation.name_with_id} has negative launch offset {launch_offset_ms}'
                )
            if trigger_operation.duration is None:
                raise ValueError(f'Trigger operation {trigger_cmd_uid} is missing duration')
            if launch_offset_ms - float(trigger_operation.duration) > 0.05:
                raise ValueError(
                    f'ddp_grad_comm overlay record {operation.name_with_id} launches outside trigger duration: '
                    f'offset={launch_offset_ms}, trigger_duration={trigger_operation.duration}'
                )

            launch_time = round(float(trigger_operation.join_time) + launch_offset_ms, 2)
            async_available_time = float(self.async_ddp_comm_available_by_rank.get(timeline.wrank_id, 0.0))
            join_time = max(async_available_time, launch_time)
            predicted_duration = round(self._predict_profile_overlap_comm_duration(operation), 2)

            operation.comm_set_join_time(join_time)
            operation.duration = predicted_duration
            operation.hidden_duration = predicted_duration
            operation.comm_set_waiting_finish_time(0)
            operation.comm_set_waiting_acc_time(0, timeline._get_last_op_waiting_acc(True))
            timeline.comm_timeline.append(operation)
            timeline.final_merge_timeline.append(operation)
            self.async_ddp_comm_available_by_rank[timeline.wrank_id] = operation.finish_time

        wait_cmd_uid = trace_metadata.get('wait_cmd_uid', None)
        if wait_cmd_uid in {None, 'None'}:
            self.pending_ddp_unassigned_finish_times_by_rank.setdefault(timeline.wrank_id, []).append(operation.finish_time)
        else:
            scoped_wait_uid = self._rank_scoped_uid(timeline.wrank_id, wait_cmd_uid)
            self.pending_ddp_wait_finish_times_by_uid.setdefault(scoped_wait_uid, []).append(operation.finish_time)

    def _add_trace_driven_ddp_overlap_wait(
        self,
        timeline: IndividualTimeline,
        operation: Operation,
        join_time: float,
        current_format_operantion_name: str,
    ) -> None:
        paired_finish_times = self._consume_pending_ddp_overlap_finish_times(timeline, operation.cmd_uid)
        if not paired_finish_times:
            raise ValueError(
                f'dp_allreduce wait op {operation.cmd_uid} has no paired DDP overlap comm records'
            )

        trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
        explicit_base_duration_ms = trace_metadata.get('finalize_base_duration_ms', None)
        inferred_base_duration_ms = trace_metadata.get('inferred_base_duration_ms', None)
        base_duration = getattr(operation, 'duration', 0.0)
        if base_duration is None:
            base_duration = 0.0
        base_duration = float(base_duration)
        if explicit_base_duration_ms is not None:
            explicit_base_duration_ms = float(explicit_base_duration_ms)
            if explicit_base_duration_ms < 0:
                raise ValueError(
                    f'dp_allreduce wait op {operation.cmd_uid} has invalid explicit base duration {explicit_base_duration_ms}'
                )
            base_duration = explicit_base_duration_ms
        elif inferred_base_duration_ms is not None:
            inferred_base_duration_ms = float(inferred_base_duration_ms)
            if inferred_base_duration_ms < 0:
                raise ValueError(
                    f'dp_allreduce wait op {operation.cmd_uid} has invalid inferred base duration {inferred_base_duration_ms}'
                )
            base_duration = max(base_duration, inferred_base_duration_ms)
        if base_duration < 0:
            raise ValueError(
                f'dp_allreduce wait op {operation.cmd_uid} has invalid base duration {base_duration}'
            )

        baseline_finish_time = float(join_time) + base_duration
        finish_time = max(baseline_finish_time, max(paired_finish_times))
        operation.duration = round(finish_time - float(join_time), 2)
        operation.hidden_duration = operation.duration
        operation.comp_set_join_finish_waiting_acc_time(join_time, 0)
        timeline.comp_timeline.append(operation)
        timeline.final_merge_timeline.append(operation)
        self.global_finished_operations[current_format_operantion_name] = operation
        self._record_completed_cmd_operation(operation)

    def _replay_profile_no_pipelining(self) -> None:
        for wrank_id in sorted(self.stages_timeline_process_dict.keys()):
            timeline: IndividualTimeline = self.stages_timeline_process_dict[wrank_id]
            operations_list = list(timeline.waiting_queue)
            comp_available_time = 0.0
            async_ddp_comm_available_time = 0.0
            pending_wait_comm_finish_times = {}
            pending_unassigned_overlap_finish_times = []

            for operation in operations_list:
                if self._should_filter_single_gpu_comm(operation):
                    continue

                trace_metadata = getattr(operation, 'trace_metadata', {}) or {}
                trace_event_type = trace_metadata.get('trace_event_type', None)
                op_semantics = getattr(operation, 'op_semantics', None)

                if operation.op_kind == 'comm' and trace_event_type == 'ddp_grad_comm':
                    if self.slowdown_enabled:
                        self._add_trace_driven_async_ddp_overlap_comm(timeline, operation)
                        async_ddp_comm_available_time = max(
                            async_ddp_comm_available_time,
                            float(self.async_ddp_comm_available_by_rank.get(timeline.wrank_id, 0.0)),
                        )
                        continue

                    launch_time = self._get_profile_recorded_start_ms(operation)
                    metadata_only = bool(trace_metadata.get('metadata_only', False))
                    if metadata_only:
                        join_time = max(launch_time, async_ddp_comm_available_time)
                        finish_time = join_time + self._predict_profile_overlap_comm_duration(operation)
                    else:
                        completion_observed_timestamp_ms = trace_metadata.get('completion_observed_timestamp_ms', None)
                        if completion_observed_timestamp_ms is None:
                            raise ValueError('Actual ddp_grad_comm is missing completion_observed_timestamp_ms')
                        finish_time = float(completion_observed_timestamp_ms)
                        recorded_duration = getattr(operation, 'duration', None)
                        if recorded_duration is None:
                            recorded_duration = finish_time - launch_time
                        if recorded_duration <= 0:
                            raise ValueError(
                                f'Invalid ddp_grad_comm duration={recorded_duration} for comm {operation.name_with_id}'
                            )
                        desired_start_time = finish_time - float(recorded_duration)
                        join_time = max(launch_time, desired_start_time, async_ddp_comm_available_time)
                        expected_finish_time = round(join_time + float(recorded_duration), 2)
                        if abs(expected_finish_time - finish_time) > 0.05:
                            raise ValueError(
                                'Inconsistent actual ddp_grad_comm timing under same-rank serialization: '
                                f'launch={launch_time}, desired_start={desired_start_time}, '
                                f'async_ddp_comm_available={async_ddp_comm_available_time}, duration={recorded_duration}, '
                                f'observed_finish={finish_time}, expected_finish={expected_finish_time}'
                            )
                        finish_time = expected_finish_time

                    operation.comm_set_join_time(join_time)
                    operation.duration = round(finish_time - join_time, 2)
                    operation.hidden_duration = operation.duration
                    operation.comm_set_waiting_finish_time(0)
                    operation.comm_set_waiting_acc_time(0, timeline._get_last_op_waiting_acc(True))
                    timeline.comm_timeline.append(operation)
                    timeline.final_merge_timeline.append(operation)
                    async_ddp_comm_available_time = max(async_ddp_comm_available_time, operation.finish_time)

                    wait_cmd_uid = trace_metadata.get('wait_cmd_uid', None)
                    if wait_cmd_uid is None:
                        pending_unassigned_overlap_finish_times.append(operation.finish_time)
                    else:
                        pending_wait_comm_finish_times.setdefault(wait_cmd_uid, []).append(operation.finish_time)
                    continue

                if operation.op_kind == 'comp' and operation.name == 'dp_allreduce' and op_semantics in {'wait_flush_only', 'metadata_placeholder'}:
                    if self.slowdown_enabled:
                        current_format_operantion_name = self._get_format_operation_name(
                            wrank_id=timeline.wrank_id,
                            operation=operation,
                            batch_id=operation.batch_id,
                            description=operation.description,
                        )
                        join_time = comp_available_time
                        if op_semantics == 'wait_flush_only':
                            recorded_wait_start = self._get_profile_recorded_start_ms(operation)
                            if recorded_wait_start is not None:
                                join_time = max(join_time, recorded_wait_start)
                        self._add_trace_driven_ddp_overlap_wait(
                            timeline, operation, join_time, current_format_operantion_name
                        )
                        comp_available_time = operation.finish_time
                        continue

                    paired_finish_times = pending_wait_comm_finish_times.pop(operation.cmd_uid, None)
                    if not paired_finish_times and pending_unassigned_overlap_finish_times:
                        paired_finish_times = pending_unassigned_overlap_finish_times
                        pending_unassigned_overlap_finish_times = []
                    if not paired_finish_times:
                        raise ValueError(
                            f'dp_allreduce wait op {operation.cmd_uid} has no paired DDP overlap comm records'
                        )

                    join_time = comp_available_time
                    if op_semantics == 'wait_flush_only':
                        recorded_wait_start = self._get_profile_recorded_start_ms(operation)
                        if recorded_wait_start is not None:
                            join_time = max(join_time, recorded_wait_start)
                        recorded_wait_end = operation.end_timestamp
                        finish_time = max(max(paired_finish_times), float(recorded_wait_end))
                    else:
                        explicit_base_duration_ms = trace_metadata.get('finalize_base_duration_ms', None)
                        inferred_base_duration_ms = trace_metadata.get('inferred_base_duration_ms', None)
                        base_duration = getattr(operation, 'duration', 0.0)
                        if base_duration is None:
                            base_duration = 0.0
                        base_duration = float(base_duration)
                        if explicit_base_duration_ms is not None:
                            explicit_base_duration_ms = float(explicit_base_duration_ms)
                            if explicit_base_duration_ms < 0:
                                raise ValueError(
                                    f'dp_allreduce wait op {operation.cmd_uid} has invalid explicit base duration {explicit_base_duration_ms}'
                                )
                            base_duration = explicit_base_duration_ms
                        elif inferred_base_duration_ms is not None:
                            inferred_base_duration_ms = float(inferred_base_duration_ms)
                            if inferred_base_duration_ms < 0:
                                raise ValueError(
                                    f'dp_allreduce wait op {operation.cmd_uid} has invalid inferred base duration {inferred_base_duration_ms}'
                                )
                            base_duration = max(base_duration, inferred_base_duration_ms)
                        if base_duration < 0:
                            raise ValueError(
                                f'dp_allreduce wait op {operation.cmd_uid} has invalid base duration {base_duration}'
                            )
                        baseline_finish_time = float(join_time) + base_duration
                        finish_time = max(baseline_finish_time, max(paired_finish_times))

                    operation.duration = round(finish_time - join_time, 2)
                    operation.hidden_duration = operation.duration
                    operation.comp_set_join_finish_waiting_acc_time(join_time, 0)
                    timeline.comp_timeline.append(operation)
                    timeline.final_merge_timeline.append(operation)
                    comp_available_time = operation.finish_time
                    self._record_completed_cmd_operation(operation)
                    continue

                if operation.op_kind == 'comp':
                    join_time = self._get_profile_recorded_start_ms(operation)
                    if join_time is None:
                        join_time = comp_available_time
                    join_time = max(comp_available_time, join_time)
                    if self.slowdown_enabled and self._is_trace_driven_backward(operation):
                        current_format_operantion_name = self._get_format_operation_name(
                            wrank_id=timeline.wrank_id,
                            operation=operation,
                            batch_id=operation.batch_id,
                            description=operation.description,
                        )
                        self._add_trace_driven_backward_slowdown(
                            timeline, operation, join_time, current_format_operantion_name
                        )
                        comp_available_time = operation.finish_time
                        continue
                    operation.hidden_duration = operation.duration
                    operation.comp_set_join_finish_waiting_acc_time(join_time, 0)
                    timeline.comp_timeline.append(operation)
                    timeline.final_merge_timeline.append(operation)
                    comp_available_time = operation.finish_time
                    self._record_completed_cmd_operation(operation)
                    continue

                join_time = self._get_profile_recorded_start_ms(operation)
                if join_time is None:
                    _, last_comm_operation = timeline._get_last_operation_time_and_op([timeline.comm_timeline])
                    if last_comm_operation is None or last_comm_operation.finish_time is None:
                        join_time = 0.0
                    else:
                        join_time = float(last_comm_operation.finish_time)
                operation.hidden_duration = operation.duration
                operation.comm_set_join_time(join_time)
                operation.comm_set_waiting_finish_time(0)
                operation.comm_set_waiting_acc_time(0, timeline._get_last_op_waiting_acc(True))
                timeline.comm_timeline.append(operation)
                timeline.final_merge_timeline.append(operation)
                async_ddp_comm_available_time = operation.finish_time
                self._record_completed_cmd_operation(operation)
        # self._handle_final_package_operation()

    def _add_operation_to_timeline(self, timeline:IndividualTimeline, operation:Operation):
        ''' 统一检查dependency情况,对于dependency未完成的operation(该op存在依赖项且未完成),直接返回等待队列 '''
        can_go_on_sign = True
        next_operation = None
        dependency_operation: Operation = None

        print(f"当前处理的 operation: {operation}, wrank_id: {timeline.wrank_id}, stage_kind: {timeline.stage_kind}")
        # if operation.name == "forward_step" and operation.wrank_id == 0 and operation.batch_id == 0:
        #     print("debug")

        # dependency决定了开始(区别于matching决定结束)
        if operation.name in self.dependency_relationship[timeline.stage_kind]:
            dependency_operation = self._operation_dependency_finished(operation=operation, wrank_id=timeline.wrank_id, stage_kind=timeline.stage_kind)
            if dependency_operation is False:
                can_go_on_sign = False

        # TODO: 是否开启隐藏TP的CMD的break
        # if operation.op_kind == "hide":
        #     return

        if operation.op_kind == "comm":
            # 查看当前comm的所属类别和并行维度
            # is_p2p_fused_comm_operation: bool = self._is_p2p_fused_comm_operation(operation)
            comm_kind, parallel_kind = self._get_comm_operation_kind_and_parallel_dimension(operation)
            # next_dependency_operation: Operation = None # p2p_fused情况
            if operation.mg_state == "steady" and parallel_kind == "pp":
                # P2P通信的特殊情况(该操作在实际过程中的一个算子被拆成2个，其余op都与实际相同为1个)
                # megatron的stready阶段的通信op是2个,检查另外一个op,如果未完成,返回等待队列
                print(f"operation.wrank_id: {operation.wrank_id}, operation.name: {operation.name}, operation_type: {type(operation)}, operation.batch_id: {operation.batch_id}")
                if comm_kind != "p2p_fused":
                    raise 0 # 该情况下不应出现
                
                try:
                    next_operation: Operation = timeline.waiting_queue.popleft()
                    if next_operation.mg_state != "steady":
                        operation.mg_is_last_iteration = True
                        timeline.waiting_queue.appendleft(next_operation)
                        # TODO：修正p2p_fused为p2p，last_iter的send_backward目前没法在_get_comm_operation_kind_and_parallel_dimension中被正常标记
                        # 即send/recv
                        comm_kind = "p2p"
                    else:
                        if next_operation.name in self.dependency_relationship[timeline.stage_kind]:
                            next_dependency_operation = self._operation_dependency_finished(operation=next_operation, wrank_id=timeline.wrank_id, stage_kind=timeline.stage_kind)
                            if not next_dependency_operation:
                                can_go_on_sign = False
                                timeline.waiting_queue.appendleft(next_operation)
                except:
                    operation.mg_is_last_iteration = True
                    # TODO：修正p2p_fused为p2p，last_iter的send_backward目前没法在_get_comm_operation_kind_and_parallel_dimension中被正常标记
                    comm_kind = "p2p"

        # 依赖项未完成，返回队列
        if can_go_on_sign is False:
            timeline.waiting_queue.appendleft(operation)
            # print(f"wrank_id: {timeline.wrank_id}, stage_kind: {timeline.stage_kind}")
            print(f"{operation}的依赖项未完成, 返回队首等待...")
            raise ValueError(f"现阶段是阻塞方式执行,现阶段不存在跨stage的依赖项,所有依赖项都在同一个stage上,按序被执行.所以当某个operation被pop到了,应该不会发生依赖项未完成情况.")

        # 根据op类型确认最终需要加入的timeline对象
        target_timeline: list = timeline.comp_timeline if operation.op_kind=="comp" else timeline.comm_timeline

        # TODO：overlap和strategy没有绝对联系
        if self.can_overlap:
            if operation.op_kind == "comm":
                timeline_last_operation = self._get_last_schedulable_comm_operation(timeline)
            else:
                _, timeline_last_operation = timeline._get_last_operation_time_and_op([target_timeline])
            # raise ValueError(f"Overlap of Comm. and Comp. is not supported.")
        else:
            _, timeline_last_operation = timeline._get_last_operation_time_and_op([timeline.comp_timeline, timeline.comm_timeline])

        # 比较dependency_operation和last_operation的finished time, 取较大值作为last_operation
        last_operation, last_operation_time = self._get_last_operation_and_finshed_time(timeline_last_operation, dependency_operation) 
        print(f"last_operation-> {last_operation}")
        print(f"last_operation_time-> {last_operation_time}")

        # 当前op的格式化name
        current_format_operantion_name: str = self._get_format_operation_name(wrank_id=timeline.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)

        if self.running_mode == MODE_SIMULATE and self._is_trace_driven_ddp_overlap_comm(operation):
            self._add_trace_driven_async_ddp_overlap_comm(timeline, operation)
            return

        if (
            self.running_mode == MODE_SIMULATE
            and operation.op_kind == "comp"
            and operation.name == "dp_allreduce"
            and getattr(operation, 'op_semantics', None) in {'wait_flush_only', 'metadata_placeholder'}
        ):
            self._add_trace_driven_ddp_overlap_wait(
                timeline, operation, last_operation_time, current_format_operantion_name
            )
            return

        if (
            self.running_mode == MODE_SIMULATE
            and self.slowdown_enabled
            and operation.op_kind == "comp"
            and self._is_trace_driven_backward(operation)
        ):
            self._add_trace_driven_backward_slowdown(
                timeline, operation, last_operation_time, current_format_operantion_name
            )
            return

        ''' 总体分为comp和comm两种情况处理 '''
        if operation.op_kind == "comp":
            if operation.name not in self.dependency_relationship[timeline.stage_kind]:
                # 无依赖项,更新该operation的属性,加入timeline和global_finished_operations
                # operation.comp_set_join_finish_waiting_acc_time(last_operation_time, last_operation.waiting_acc)
                # 加入timeline和global_finished_operations
                # print(f"operation:{operation.name}, duration:{operation.duration}")
                operation.comp_set_join_finish_waiting_acc_time(last_operation_time, 0)
                target_timeline.append(operation)

                timeline.final_merge_timeline.append(operation)

                self.global_finished_operations[current_format_operantion_name] = operation
                print(f"计算操作：{current_format_operantion_name}无依赖操作, 该operation已经完成属性更新,并加入到数据结构中")
            else:
                operation.comp_set_join_finish_waiting_acc_time(last_operation_time, 0)
                target_timeline.append(operation)

                timeline.final_merge_timeline.append(operation)
                
                # current_format_operantion_name: str = str(self.wrank_id) + "_" + operation.name + "_" + str(operation.batch_id)
                self.global_finished_operations[current_format_operantion_name] = operation
                print(f"计算操作：{current_format_operantion_name}的依赖操作已完成, 该operation已经完成属性更新,并加入到数据结构中")

            self._record_completed_cmd_operation(operation)

        elif operation.op_kind == "comm":
            '''comm. op 中 p2p(1 to 1) allreduce(n to n) broadcast(1 to n) 的处理逻辑都不一致, 因此需要单独分类处理

                这里要区别megatron中通信算子合并情况,仅出现在【steady阶段】,send_forward_recv_backward 和 send_backward_recv_forward 是一起出现的,因此注册和查找都是同时进行的
                注意,send_forward_recv_backward 和 send_backward_recv_forward 互为matching operaitons,需要相互配合才能执行。谁先达到谁先注册,等待对方到达进行收尾操作
                可能的情况：
                    1. 当前ops未注册, 配对的comm. ops 已经注册,更新当前ops的join_time,当前对象执行收尾操作,更新当前ops的属性,从配对的comm_waiting_pool取出该配对的comm op（根据pre/post_timeline）,然后刷新各类型属性,最后加入timeline和global_finished_operations
                    2. 当前ops未注册, 配对的comm. ops 未注册,更新当前ops的join_time, 将当前ops加入到comm_waiting_pool中等对方查询, 等待对方对象执行收尾操作
            '''

            # 在SIMULATE模式下也过滤单GPU通信操作
            if self._should_filter_single_gpu_comm(operation):
                print(f"Debug: 过滤单GPU通信操作 {operation.name} (rank {timeline.wrank_id}) 在SIMULATE模式")
                # 直接标记为完成，不加入timeline
                current_format_operantion_name = self._get_format_operation_name(wrank_id=timeline.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                self.global_finished_operations[current_format_operantion_name] = operation
                return

            operation.comm_set_join_time(last_operation_time)
            if comm_kind == "p2p_fused" and parallel_kind == "pp" and not operation.mg_is_last_iteration:
                '''
                pp类型fused_comm,2个comm. ops情况:
                    只有megatron的steady阶段,即operation: send_forward_recv_backward and send_backward_recv_forward,此时可取出下一个operation
                    steady的last_iteration只有send_backward(单op)
                '''
                # 设定另一个op的join时间
                next_operation.comm_set_join_time(last_operation_time)

                # 校验matching ops 是否注册
                can_start_comm_process, comm_matching_operation_list, comm_matching_operation_format_name_list = \
                                        self._can_start_matching_comm_group_process(operation_list=[operation, next_operation], comm_kind=comm_kind, 
                                                                                           parallel_kind=parallel_kind, timeline=timeline)
                if can_start_comm_process:
                    '''其余comm op都已注册, 因此开始nccl comm, 更新属性和状态'''
                    # 对方stage： |_____对方等待时间_____|———————— duration执行时间 ————————|
                    # 当前stage：                       |———————— duration执行时间 ————————|

                    # 进行nccl comm => 计算 comm op的 duration 数值
                    # TODO： 每个comm op的 duration 该如何确定？在哪里初始化？
                    _ = self._calculate_comm_duration(comm_matching_operation_list + [operation, next_operation])

                    # 更新matching comm ops的属性, current_operation传入一个op即可（next op和op的join_time一致）
                    # p2p_fused情况下,2个 matching operaitons 所在的 IndividualTimeline对象位于同一stage上

                    # 需要翻转comm_matching_operation_list(error_static需要,否则与trace的顺序不一致)
                    comm_matching_operation_list.reverse()
                    gap_waiting_time = self._update_matching_comm_ops_properties(comm_matching_operation_list=comm_matching_operation_list, 
                                                                    comm_matching_operation_format_name_list=comm_matching_operation_format_name_list,
                                                                    current_operation=operation, comm_kind=comm_kind, parallel_kind=parallel_kind, 
                                                                    current_timeline=timeline)
                    # 更新current operation(2个)的属性
                    _ = self._update_current_comm_op_properties(comm_op=operation, current_timeline=timeline, gap_waiting_time=gap_waiting_time)
                    _ = self._update_current_comm_op_properties(comm_op=next_operation, current_timeline=timeline, gap_waiting_time=gap_waiting_time)
                    # _ = self._update_current_comm_op_properties(comm_op=operation, current_timeline=timeline, gap_waiting_time=gap_waiting_time)

                    operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                    next_operation_format_name = self._get_format_operation_name(wrank_id=next_operation.wrank_id, operation=next_operation, batch_id=next_operation.batch_id, description=next_operation.description)
                    print(f"通信操作: p2p_fused {operation_format_name}和{next_operation_format_name}进行收尾操作.")
                else:
                    '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                    operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                    next_operation_format_name = self._get_format_operation_name(wrank_id=next_operation.wrank_id, operation=next_operation, batch_id=next_operation.batch_id, description=next_operation.description)
                    self.global_waiting_pool[operation_format_name] = operation
                    self.global_waiting_pool[next_operation_format_name] = next_operation
                    timeline._set_is_blocked_sign(True)
                    print(f"通信操作: p2p_fused {operation_format_name}和{next_operation_format_name}注册并等待.")

            elif parallel_kind == "pp":
                assert parallel_kind == "dp" or parallel_kind == "pp", "tp is not supported."
                if comm_kind == "p2p" and parallel_kind == "pp":
                    ''' pp类型comm,1个comm. op情况, 包含ds架构和非steady的其他阶段
                         1.本地op未注册, 对方op已经注册 2. 双方op都未注册
                    '''
                    # 校验matching ops 是否注册
                    # key_in_waiting_pool: str = parallel_kind + "_" + comm_kind + "_" + str(operation.batch_id)
                    can_start_comm_process, comm_matching_operation_list, comm_matching_operation_format_name_list = \
                                        self._can_start_matching_comm_group_process(operation_list=[operation], comm_kind=comm_kind, 
                                                                                           parallel_kind=parallel_kind, timeline=timeline)
                    if can_start_comm_process:
                        '''其余comm op都已注册, 因此开始nccl comm, 更新属性和状态
                            注意,即便current操作在遍历操作中晚于已注册matching操作,但是其join_time可能早于已注册算子,因此需要额外的判定
                        '''
                        # 获取matching comm op对象的list
                        # comm_matching_operation_list: list = self.global_waiting_pool[key_in_waiting_pool]

                        # 进行nccl comm => 计算 comm op的 duration 数值
                        # TODO： 每个comm op的 duration 该如何确定？在哪里初始化？
                        _ = self._calculate_comm_duration(comm_matching_operation_list + [operation])

                        # 更新matching comm ops的属性,
                        gap_waiting_time = self._update_matching_comm_ops_properties(comm_matching_operation_list=comm_matching_operation_list, 
                                                                        comm_matching_operation_format_name_list=comm_matching_operation_format_name_list,
                                                                        current_operation=operation, comm_kind=comm_kind, parallel_kind=parallel_kind, 
                                                                        current_timeline=timeline)
                        # 更新current operation(1个)的属性
                        _ = self._update_current_comm_op_properties(comm_op=operation, current_timeline=timeline, gap_waiting_time=gap_waiting_time)

                        operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                        print(f"通信操作: {comm_kind}_{parallel_kind} |  {operation_format_name}进行收尾操作.")
                    else:
                        '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                        operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                        self.global_waiting_pool[operation_format_name] = operation
                        timeline._set_is_blocked_sign(True)

                        print(f"通信操作: {comm_kind}_{parallel_kind} | {operation_format_name}注册并等待.")

            elif parallel_kind == "dp" or parallel_kind == "ep" or parallel_kind == "tp" or parallel_kind == "exp" or parallel_kind == "exp_dp":
                # TODO：修正下这个写法，不太美观
                if comm_kind in COLLECTIVE_COMM_KINDS: # if "allreduce" in operation.name:
                    # allreduce类型：结束时间是统一的;最后一个到达的没有等待时间直接duration，其他的分别是和最后一个到达的做差（注意，最后一个注册的不一定是最晚的join_time,还是要遍历）
                    can_start_comm_process, comm_matching_operation_list, comm_matching_operation_format_name_list = \
                                        self._can_start_matching_comm_group_process(operation_list=[operation], comm_kind=comm_kind, 
                                                                                            parallel_kind=parallel_kind, timeline=timeline)

                    if can_start_comm_process:
                        '''其余comm op都已注册, 因此开始nccl comm, 更新属性和状态
                            注意,即便current操作在遍历操作中晚于已注册matching操作,但是其join_time可能早于已注册算子,因此需要额外的判定
                        '''
                        # 获取matching comm op对象的list
                        # comm_matching_operation_list: list = self.global_waiting_pool[key_in_waiting_pool]

                        _ = self._calculate_comm_duration(comm_matching_operation_list + [operation])

                        # 更新matching comm ops的属性,
                        # TODO：这儿需要根据comm类型进行判定;对于allreduce,当前即是最后到达的op,因此gap_waiting_time=0，其他已注册的分别与其做擦
                        gap_waiting_time = self._update_matching_comm_ops_properties(comm_matching_operation_list=comm_matching_operation_list, 
                                                                        comm_matching_operation_format_name_list=comm_matching_operation_format_name_list,
                                                                        current_operation=operation, comm_kind=comm_kind, parallel_kind=parallel_kind, 
                                                                        current_timeline=timeline)
                        # 更新current operation(1个)的属性
                        _ = self._update_current_comm_op_properties(comm_op=operation, current_timeline=timeline, gap_waiting_time=gap_waiting_time)

                        operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                        print(f"通信操作: {comm_kind}_{parallel_kind} |  {operation_format_name}进行收尾操作.")
                        
                    else:
                        '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                        operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation=operation, batch_id=operation.batch_id, description=operation.description)
                        self.global_waiting_pool[operation_format_name] = operation
                        timeline._set_is_blocked_sign(True)

                        print(f"通信操作: {comm_kind}_{parallel_kind} | {operation_format_name}注册并等待.")
                        # print(f"-->special op: {operation}")

            elif parallel_kind == "tp":
                # 要区分broadcast和allreduce？
                pass


        else:
            raise ValueError(f"Invalid operation kind: {operation.op_kind}")


    def _get_last_operation_and_finshed_time(self, timeline_last_operation, dependency_operation):
        # Determine the operation with the latest finish time, accounting for possible None values
        if timeline_last_operation and dependency_operation:
            last_operation = timeline_last_operation if timeline_last_operation.finish_time >= dependency_operation.finish_time else dependency_operation
        elif timeline_last_operation:
            last_operation = timeline_last_operation
        elif dependency_operation:
            last_operation = dependency_operation
        else:
            last_operation = None
        last_operation_time = last_operation.finish_time if last_operation else 0

        return last_operation, last_operation_time

    def _should_filter_single_gpu_comm(self, operation):
        """
        判断是否应该过滤单GPU通信操作
        当集合通信操作的通信组中rank数量为1时，过滤该操作
        """
        if operation.op_kind != 'comm':
            return False

        # 检查是否是需要过滤的通信操作
        filter_ops = [
            'exp_dp_allreduce',
            'ep_dp_allreduce',
            'dp_allreduce',
            'tp_allreduce',
            'ep_allreduce',
            'dp_reducescatter',
            'tp_reducescatter',
            'exp_reducescatter',
            'cp_reducescatter',
        ]
        if operation.name not in filter_ops:
            return False

        # 获取通信组信息
        try:
            comm_group = self._get_comm_group_for_operation(operation)
            if comm_group and len(comm_group) == 1:
                return True
        except Exception as e:
            print(f"Warning: 无法获取通信组信息 for {operation.name}: {e}")

        return False

    def _get_comm_group_for_operation(self, operation):
        """
        获取操作的通信组信息
        返回包含该操作所有参与rank的列表（包括当前rank）
        """
        try:
            # 获取操作的通信类型和并行维度
            comm_kind, parallel_kind = self._get_comm_operation_kind_and_parallel_dimension(operation)

            # 根据操作的wrank_id找到对应的timeline
            timeline = None
            for tl in self.stages_timeline_process_dict.values():
                if tl.wrank_id == operation.wrank_id:
                    timeline = tl
                    break

            if timeline is None:
                return None

            def _resolve_rank_group(raw_group):
                """Resolve rank group from either flat `[int, ...]` or nested `[[...], ...]` format."""
                if raw_group is None:
                    return None

                if isinstance(raw_group, tuple):
                    raw_group = list(raw_group)

                if not isinstance(raw_group, list):
                    return None

                # Single group format: [0, 4, 8, ...]
                if raw_group and all(isinstance(rank, int) for rank in raw_group):
                    if operation.wrank_id in raw_group:
                        return raw_group
                    return None

                # Group list format: [[...], [...], ...]
                for group in raw_group:
                    if isinstance(group, tuple):
                        group = list(group)
                    if isinstance(group, list) and operation.wrank_id in group:
                        return group
                return None

            # 根据并行类型返回通信组
            if parallel_kind == "dp":
                return _resolve_rank_group(getattr(timeline.stage_rank, "dp_groups", None))
            elif parallel_kind == "exp_dp":
                group = _resolve_rank_group(getattr(timeline.stage_rank, "dp_modulo_exp_groups", None))
                if group is not None:
                    return group
                return _resolve_rank_group(getattr(timeline.stage_rank, "dp_groups", None))
            elif parallel_kind == "tp":
                return _resolve_rank_group(getattr(timeline.stage_rank, "tp_groups", None))
            elif parallel_kind == "pp":
                return _resolve_rank_group(getattr(timeline.stage_rank, "pp_groups", None))
            elif parallel_kind == "cp":
                return _resolve_rank_group(getattr(timeline.stage_rank, "cp_groups", None))
            elif parallel_kind == "ep":
                return _resolve_rank_group(getattr(timeline.stage_rank, "ep_groups", None))
            elif parallel_kind == "exp":
                return _resolve_rank_group(getattr(timeline.stage_rank, "exp_groups", None))

            return None

        except Exception as e:
            print(f"Error in _get_comm_group_for_operation for {operation.name}: {e}")
            return None

    def _can_start_matching_comm_group_process(self, operation_list: list, comm_kind: str, parallel_kind: str, timeline: IndividualTimeline) -> bool:
        """ 生成operation_list中的operation的formatname, 检查global_waiting_pool中是否包含这些operation, 如果有不存在的, 返回False"""
        # TODO: 对于allreduce一定能在global pool中找到正确的matching op/subop吗？(现在是根据_get_format_operation_name()确认的)

        # 优化：对于非PP通信，根据模型类型决定是否跳过同步等待
        if self.optimization_enabled and parallel_kind != "pp":
            # Dense模型：跳过所有TP和DP通信的同步等待
            if not self.is_moe_model and parallel_kind in ["tp", "dp"]:
                return True, [], []
            # MoE模型：仅非simulate模式下允许跳过TP同步；simulate必须保留TP barrier
            elif self.is_moe_model and parallel_kind == "tp" and self.running_mode != MODE_SIMULATE:
                return True, [], []
            # MoE模型：对于EXP、EXP_DP、EP、DP通信，必须保留完整的同步依赖
            elif self.is_moe_model and parallel_kind in ["exp", "exp_dp", "ep", "dp", "tp"]:
                # 不跳过同步等待，继续执行完整的匹配逻辑
                pass

        matching_operation_format_name_list = []
        matching_operation_list = []

        # 生成comm group的formatname list
        for operation in operation_list:
            matching_operation_name, matching_operation_wrank_id_list, _ = self._get_comm_matching_operation_name_and_wrank_id(operation=operation,
                                                                                comm_kind=comm_kind, parallel_kind=parallel_kind, timeline=timeline)
            for matching_operation_wrank_id in matching_operation_wrank_id_list:
                # 优化：只检查选中的ranks
                if self.optimization_enabled and matching_operation_wrank_id not in self.selected_ranks:
                    continue

                matching_operation_format_name = self._get_format_operation_name(wrank_id=matching_operation_wrank_id,
                                                                                operation=operation, batch_id=operation.batch_id,
                                                                                description=operation.description, matching_op_name=matching_operation_name)
                if matching_operation_format_name not in self.global_waiting_pool:
                    return False, None, None
                else:
                    matching_operation_list.append(self.global_waiting_pool[matching_operation_format_name])
                    matching_operation_format_name_list.append(matching_operation_format_name)
        return True, matching_operation_list, matching_operation_format_name_list

    def _update_current_comm_op_properties(self, comm_op: Operation, current_timeline: IndividualTimeline,gap_waiting_time:float):
        # 1. 更新waiting time/finish time, if当前op是刚加入的,不存在等待时间;else 更新更待时间
        if self.running_mode == MODE_PROFILE:
            comm_op.comm_set_waiting_finish_time(0)
        else:
            if gap_waiting_time > 0:
                comm_op.comm_set_waiting_finish_time(gap_waiting_time)
            else:
                comm_op.comm_set_waiting_finish_time(0)
                # 说明current op是先注册的,因此补充等待时间
        # comm_op.comm_set_waiting_finish_time(gap_waiting_time)
            
        # 2. 加入到timeline中
        current_timeline._add_comm_op_to_timeline([comm_op])

        # 3. global_finished_operations
        operation_format_name = self._get_format_operation_name(wrank_id=comm_op.wrank_id, operation=comm_op, batch_id=comm_op.batch_id, description=comm_op.description)
        self.global_finished_operations[operation_format_name] = comm_op

    def _update_matching_comm_ops_properties(self, comm_matching_operation_list: list, comm_matching_operation_format_name_list: list,
                                                                        current_operation: Operation, comm_kind: str, parallel_kind: str,
                                                                        current_timeline: IndividualTimeline):
        assert comm_kind and parallel_kind

        # 初始化变量，避免UnboundLocalError
        gap_waiting_time = 0
        abs_gap_waiting_time = 0

        # 添加调试信息，特别是对MoE通信操作
        if any(moe_op in current_operation.name for moe_op in ['exp_allgather', 'exp_all_to_all', 'exp_dp_allreduce']):
            print(f"Debug: Processing MoE communication operation: {current_operation.name}, comm_kind: {comm_kind}, parallel_kind: {parallel_kind}")
            print(f"Debug: comm_matching_operation_list length: {len(comm_matching_operation_list)}")

            # 检查是否存在通信组不完整的情况
            if len(comm_matching_operation_list) == 0:
                print(f"Warning: MoE通信操作 {current_operation.name} 没有找到匹配的操作，这可能影响模拟准确性")
                print(f"Warning: 建议检查selected_ranks策略是否包含了完整的通信组")

        calcu_operation = current_operation
        if comm_kind in COLLECTIVE_COMM_KINDS:
            op_list = [current_operation,*comm_matching_operation_list]
            # 找到op_list中join_time最晚（值最大）的op，记录该op为current_operation
            calcu_operation = max(op_list, key=lambda op: op.join_time)

        for matching_operation in comm_matching_operation_list:
            # 1. 更新macthing comm ops的 waiting time/finish time
            # print(f"current_operation: {current_operation}")
            # print(f"matching_operation: {matching_operation}")
            # gap_waiting_time =  current_operation.join_time - matching_operation.join_time # 0
            abs_gap_waiting_time = abs(calcu_operation.join_time - matching_operation.join_time)
            if self.running_mode == MODE_PROFILE:
                gap_waiting_time = 0
            else:
                # 先判定哪个操作注册的更早(依据join_time)
                if calcu_operation.join_time >= matching_operation.join_time:
                    # current晚于matching op 注册
                    gap_waiting_time = abs_gap_waiting_time
                else:
                    # matching晚于current op 注册, matching's watiing_time is 0.
                    gap_waiting_time = 0
            # gap_waiting_time = current_operation.join_time - matching_operation.join_time if not self.running_mode else 0
            matching_operation.comm_set_waiting_finish_time(gap_waiting_time)

            # 获取stage_offset从而得到matching_op_timeline；获取target_wrank_id从而获取operation_matching_format_name
            # _, matching_operation_wrank_id, stage_offset = self._get_comm_matching_operation_name_and_wrank_id(operation=current_operation, 
            #                                                                                       comm_kind=comm_kind, parallel_kind=parallel_kind, 
            #                                                                                       timeline=current_timeline)
            # matching_op_timeline: IndividualTimeline= current_timeline.post_individual_timeline if stage_offset > 0 else current_timeline.pre_individual_timeline
            # TODO: 检查一下
            matching_operation_wrank_id = matching_operation.wrank_id
            
            matching_op_timeline: IndividualTimeline = self.stages_timeline_process_dict[matching_operation_wrank_id]
            # matching_op_timeline: IndividualTimeline = current_timeline.post_individual_timeline if matching_operation_wrank_id > current_timeline.wrank_id  \
            #                                                                                     else current_timeline.pre_individual_timeline

            # 2. 更新global_finished_operations
            operation_matching_format_name = self._get_format_operation_name(wrank_id=matching_operation_wrank_id, 
                                                                             operation=matching_operation, 
                                                                             batch_id=matching_operation.batch_id, description=matching_operation.description)
            self.global_finished_operations[operation_matching_format_name] = matching_operation

            # 3. 加入到timeline中
            matching_op_timeline._add_comm_op_to_timeline([matching_operation])

            # CHANGED: 5. 更新blocked状态
            matching_op_timeline._set_is_blocked_sign(False)

        # 4. 更新self.global_waiting_pool,遍历comm_matching_operation_format_name_list，删除global_waiting_pool中的key-value
        for operation_format_name in comm_matching_operation_format_name_list:
            self.global_waiting_pool.pop(operation_format_name, None)

        # gap_waiting_time==0则说明需要更新current op的watiting_time(在非mapping model下)

        if self.running_mode == MODE_PROFILE:
            return 0
        
        if comm_kind == "p2p" or comm_kind == "p2p_fused":
            if self.running_mode == MODE_PROFILE or gap_waiting_time != 0:
                return 0
            else:
                return abs_gap_waiting_time
            
        elif comm_kind in COLLECTIVE_COMM_KINDS:
            # 最晚注册的不一定join_time是最晚的,因此在matching_list中找到最晚的并计算返回
            return_value = calcu_operation.join_time - current_operation.join_time
            assert return_value >= 0
            return return_value
            
        elif comm_kind == "broadcast":
            raise 0
        
        else:
            raise ValueError(f"Invalid comm_kind name: {comm_kind}")
        

    def _predict_comm_duration_ms(self, request: CommunicationPredictionRequest) -> float:
        """Predict communication duration through configured CC backend."""
        if self.cc_backend is not None:
            return float(self.cc_backend.predict(request))

        if self.cc_estimator is not None:
            return float(
                self.cc_estimator.predict_communication_time(
                    list(request.comm_group),
                    int(request.data_size_bytes),
                    request.op_name,
                )
            )

        raise RuntimeError(
            "No communication backend configured for TimelinesManager. "
            "Please initialize SimulatorEngine with a valid cc backend."
        )

    def _get_comm_data_size(self, operation: Operation, comm_group_size: int) -> int:
        """Extract per-operation communication message size in bytes."""
        tensor_shape = getattr(operation, "tensor_shape", None)
        tensor_dtype = getattr(operation, "tensor_dtype", None)

        if tensor_shape is None or tensor_dtype is None:
            return int(
                get_tensor_data_size(
                    getattr(operation, "tensor_shape", [1024, 1024]),
                    getattr(operation, "tensor_dtype", "torch.float16"),
                )
            )

        return int(
            calculate_comm_message_size(
                tensor_shape,
                tensor_dtype,
                operation.name,
                comm_group_size,
            )
        )

    @staticmethod
    def _deduplicate_ranks_preserve_order(rank_list) -> list:
        deduplicated = []
        visited = set()
        for rank in rank_list:
            normalized_rank = int(rank)
            if normalized_rank not in visited:
                visited.add(normalized_rank)
                deduplicated.append(normalized_rank)
        return deduplicated

    @staticmethod
    def _is_send_operation_name(op_name: str) -> bool:
        normalized_name = str(op_name).strip().lower()
        return normalized_name.startswith("send")

    @staticmethod
    def _is_recv_operation_name(op_name: str) -> bool:
        normalized_name = str(op_name).strip().lower()
        return normalized_name.startswith("recv")

    def _build_comm_request_metadata(
        self,
        first_operation: Operation,
        comm_op_list: list,
        comm_group: list,
        comm_kind: str,
    ) -> dict:
        metadata = {"mg_state": getattr(first_operation, "mg_state", None)}

        if comm_kind not in {"p2p", "p2p_fused"}:
            return metadata

        if len(comm_group) < 2:
            raise ValueError(
                f"p2p operation requires at least 2 participants, got comm_group={comm_group}"
            )

        current_rank = int(first_operation.wrank_id)
        peer_candidates = [rank for rank in comm_group if rank != current_rank]
        if not peer_candidates:
            raise ValueError(
                f"Cannot infer p2p peer rank for operation {first_operation.name}, comm_group={comm_group}"
            )
        peer_rank = int(peer_candidates[0])

        src_rank: int
        dst_rank: int
        if comm_kind == "p2p_fused":
            src_rank = current_rank
            dst_rank = peer_rank
            direction = "bidir"
        else:
            operation_name = str(first_operation.name).strip().lower()
            if self._is_send_operation_name(operation_name):
                src_rank = current_rank
                dst_rank = peer_rank
                direction = "0->1"
            elif self._is_recv_operation_name(operation_name):
                src_rank = peer_rank
                dst_rank = current_rank
                direction = "1->0"
            else:
                send_ops = [op for op in comm_op_list if self._is_send_operation_name(op.name)]
                recv_ops = [op for op in comm_op_list if self._is_recv_operation_name(op.name)]
                if send_ops and recv_ops:
                    src_rank = int(send_ops[0].wrank_id)
                    dst_rank = int(recv_ops[0].wrank_id)
                else:
                    src_rank = current_rank
                    dst_rank = peer_rank
                direction = "0->1"

        if src_rank not in comm_group or dst_rank not in comm_group:
            raise ValueError(
                f"Invalid p2p src/dst ranks for comm_group={comm_group}, src={src_rank}, dst={dst_rank}"
            )

        src_index = comm_group.index(src_rank)
        dst_index = comm_group.index(dst_rank)
        if src_index == dst_index:
            raise ValueError(
                f"p2p src and dst indices must differ, got src_index={src_index}, dst_index={dst_index}"
            )

        metadata.update(
            {
                "p2p_src_index": int(src_index),
                "p2p_dst_index": int(dst_index),
                "p2p_direction": direction,
            }
        )
        return metadata

    def _calculate_comm_duration(self, comm_op_list):
        """Predict communication duration and update all matched communication ops."""
        if not comm_op_list:
            return

        first_operation = comm_op_list[0]
        complete_comm_group = self._get_comm_group_for_operation(first_operation)
        if complete_comm_group:
            comm_group = self._deduplicate_ranks_preserve_order(complete_comm_group)
        else:
            comm_group = self._deduplicate_ranks_preserve_order(
                [operation.wrank_id for operation in comm_op_list]
            )

        if not comm_group:
            raise ValueError(
                f"Cannot determine communication group for operation {first_operation.name}"
            )

        comm_kind, parallel_kind = self._get_comm_operation_kind_and_parallel_dimension(first_operation)
        domain_dims = ()
        if parallel_kind == "tp":
            domain_dims = ("TP",)
        elif parallel_kind in ("dp", "exp_dp"):
            domain_dims = ("DP",)
        elif parallel_kind in ("ep", "exp"):
            domain_dims = ("EP",)
        elif parallel_kind == "cp":
            domain_dims = ("CP",)

        data_size = 0
        for operation in comm_op_list:
            operation_data_size = self._get_comm_data_size(operation, len(comm_group))
            data_size = max(data_size, operation_data_size)

        metadata = self._build_comm_request_metadata(
            first_operation=first_operation,
            comm_op_list=comm_op_list,
            comm_group=comm_group,
            comm_kind=comm_kind,
        )

        request = CommunicationPredictionRequest.from_raw(
            comm_group=comm_group,
            op_name=first_operation.name,
            data_size_bytes=int(data_size),
            group_kind=parallel_kind,
            domain_dims=domain_dims,
            tensor_shape=getattr(first_operation, "tensor_shape", None),
            tensor_dtype=getattr(first_operation, "tensor_dtype", None),
            mpu_info=self.mpu_info,
            metadata=metadata,
        )

        duration_ms = self._predict_comm_duration_ms(request)

        for operation in comm_op_list:
            operation.duration = duration_ms


    def _get_comm_matching_operation_name_and_wrank_id(self, operation: Operation, comm_kind: str, parallel_kind: str, timeline: IndividualTimeline):
        """
            返回matching的op_name, wrank_id or wrank_id list
        """
        matching_operation_name = self.comm_matching_relationship.get(operation.name, [operation.name])[0]
        if (comm_kind == "p2p" or comm_kind == "p2p_fused") and parallel_kind == "pp":
            matching_operation_name, stage_offset = self.comm_matching_relationship[operation.name]
            if stage_offset == 1:
                target_wrank_id = timeline.stage_rank._get_pp_next_world_rank()
            elif stage_offset == -1:
                target_wrank_id = timeline.stage_rank._get_pp_previous_world_rank()
            else:
                raise ValueError(f"Invalid stage_offset value: {stage_offset}")
            
            # p2p类型返回的target_wrank_id是单一值
            return matching_operation_name, [target_wrank_id], stage_offset
        
        elif parallel_kind == "tp":
            # 获取 tp group wrank_id list
            if comm_kind in {"allreduce", "allgather", "all_to_all", "reducescatter"}:
                tp_group_ranks = [rank_id for rank_id in timeline.stage_rank.tp_groups if rank_id != timeline.wrank_id]
                # Dense优化路径可跳过TP barrier；MoE simulate必须保留完整TP peer列表。
                if self.optimization_enabled and not (self.is_moe_model and self.running_mode == MODE_SIMULATE):
                    tp_group_ranks = []
                return matching_operation_name, tp_group_ranks, None
            elif comm_kind == "broadcast":
                 raise ValueError(f"broadcast is not supported...")
            else:
                raise ValueError(f"Invalid comm_kind name: {comm_kind}")
        
        elif parallel_kind == "dp": # and comm_kind == "allreduce":
            if comm_kind in {"allreduce", "allgather", "all_to_all", "reducescatter"}:
                # 获取 dp group wrank_id list
                dp_group_ranks = [rank_id for rank_id in timeline.stage_rank.dp_groups if rank_id != timeline.wrank_id]
                # 优化：对于Dense模型，DP通信不需要等待其他ranks
                if self.optimization_enabled and not self.is_moe_model:
                    dp_group_ranks = []
                return matching_operation_name, dp_group_ranks, None
        elif parallel_kind == "cp":
            if comm_kind in {"allreduce", "allgather", "all_to_all", "reducescatter"}:
                cp_group_ranks = [rank_id for rank_id in timeline.stage_rank.cp_groups if rank_id != timeline.wrank_id]
                if self.optimization_enabled:
                    cp_group_ranks = [rank_id for rank_id in cp_group_ranks if rank_id in self.selected_ranks]
                return matching_operation_name, cp_group_ranks, None
        elif parallel_kind == "ep":
            if comm_kind in {"allreduce", "allgather", "all_to_all", "reducescatter"}:
                # 获取 ep group wrank_id list
                # print(f"self.comm_matching_relationship[operation.name]:{self.comm_matching_relationship[operation.name]}")
                # print(f"self.comm_matching_relationship[operation.name][0]:{self.comm_matching_relationship[operation.name][0]}")
                # print(f"timeline.stage_rank.ep_groups:{timeline.stage_rank.ep_groups}")
                ep_group_ranks = [rank_id for rank_id in timeline.stage_rank.ep_groups if rank_id != timeline.wrank_id]
                # 优化：只保留选中的ranks
                if self.optimization_enabled:
                    ep_group_ranks = [rank_id for rank_id in ep_group_ranks if rank_id in self.selected_ranks]
                return matching_operation_name, ep_group_ranks, None
        elif parallel_kind == "exp":
            if comm_kind in {"all_to_all", "allgather", "reducescatter"}:
                exp_group_ranks = [rank_id for rank_id in timeline.stage_rank.exp_groups if rank_id != timeline.wrank_id]
                # 优化：只保留选中的ranks
                if self.optimization_enabled:
                    exp_group_ranks = [rank_id for rank_id in exp_group_ranks if rank_id in self.selected_ranks]
                return matching_operation_name, exp_group_ranks, None
        elif parallel_kind == "exp_dp":
            if comm_kind in {"allreduce", "reducescatter"}:
                # 获取 expert data parallel group wrank_id list
                if hasattr(timeline.stage_rank, 'dp_modulo_exp_groups') and timeline.stage_rank.dp_modulo_exp_groups:
                    # dp_modulo_exp_groups在RankZoo中是该rank所属的单个expert data parallel group
                    # 不是所有组的列表，而是该rank所属的特定组
                    exp_dp_group_ranks = []
                    group = timeline.stage_rank.dp_modulo_exp_groups

                    # 确保group是列表类型且包含当前rank
                    if isinstance(group, list) and timeline.wrank_id in group:
                        exp_dp_group_ranks = [rank_id for rank_id in group if rank_id != timeline.wrank_id]
                    else:
                        print(f"Warning: dp_modulo_exp_groups is not a valid list or doesn't contain current rank: {group} (type: {type(group)})")
                        # 回退到普通dp处理
                        for dp_group in timeline.stage_rank.dp_groups:
                            if timeline.wrank_id in dp_group:
                                exp_dp_group_ranks = [rank_id for rank_id in dp_group if rank_id != timeline.wrank_id]
                                break

                    # MoE模型必须保留exp_dp的同步依赖
                    if self.optimization_enabled:
                        exp_dp_group_ranks = [rank_id for rank_id in exp_dp_group_ranks if rank_id in self.selected_ranks]
                    return matching_operation_name, exp_dp_group_ranks, None
                else:
                    # 如果没有dp_modulo_exp_groups，回退到普通dp处理
                    dp_group_ranks = []
                    for group in timeline.stage_rank.dp_groups:
                        if timeline.wrank_id in group:
                            dp_group_ranks = [rank_id for rank_id in group if rank_id != timeline.wrank_id]
                            break

                    if self.optimization_enabled:
                        dp_group_ranks = [rank_id for rank_id in dp_group_ranks if rank_id in self.selected_ranks]
                    return matching_operation_name, dp_group_ranks, None
        else:
            raise ValueError(f"Invalid comm_kind name: {comm_kind} and parallel_kind name: {parallel_kind}")


    def _get_comm_operation_kind_and_parallel_dimension(self, operation: Operation):
        """Return normalized `(comm_kind, parallel_kind)` with explicit op-name mapping."""
        op_name = str(getattr(operation, "name", "")).strip()
        normalized_op_name = op_name.lower()
        normalized_group_kind = (
            str(operation.group_kind).strip().lower() if getattr(operation, "group_kind", None) is not None else None
        )

        op_spec_map = {
            # Megatron PP (p2p)
            "recv_forward": ("p2p", "pp"),
            "send_forward": ("p2p", "pp"),
            "recv_backward": ("p2p", "pp"),
            "send_backward": ("p2p", "pp"),
            # DeepSpeed PP (p2p)
            "recvgrad": ("p2p", "pp"),
            "sendgrad": ("p2p", "pp"),
            "recvactivation": ("p2p", "pp"),
            "sendactivation": ("p2p", "pp"),
            # TP
            "tp_allreduce": ("allreduce", "tp"),
            "tp_load_batch_broadcast": ("broadcast", "tp"),
            "tp_broadcast": ("broadcast", "tp"),
            "tp_all_to_all": ("all_to_all", "tp"),
            "tp_allgather": ("allgather", "tp"),
            "tp_reduce_scatter": ("reducescatter", "tp"),
            "tp_reducescatter": ("reducescatter", "tp"),
            # DP / EP / EXP
            "dp_allreduce": ("allreduce", "dp"),
            "dp_reducescatter": ("reducescatter", "dp"),
            "ep_allreduce": ("allreduce", "ep"),
            "exp_dp_allreduce": ("allreduce", "exp_dp"),
            "ep_dp_allreduce": ("allreduce", "exp_dp"),  # historical alias
            "exp_all_to_all": ("all_to_all", "exp"),
            "exp_allgather": ("allgather", "exp"),
            "exp_reducescatter": ("reducescatter", "exp"),
            # CP
            "cp_reducescatter": ("reducescatter", "cp"),
            # DeepSpeed DP allreduce
            "reducegrads": ("allreduce", "dp"),
            "reducetiedgrads": ("allreduce", "dp"),
        }

        if normalized_op_name not in op_spec_map:
            raise ValueError(f"Unsupported comm operation name for semantic mapping: {operation.name}")

        comm_kind, expected_group_kind = op_spec_map[normalized_op_name]
        if normalized_group_kind is None:
            operation.group_kind = expected_group_kind
            normalized_group_kind = expected_group_kind
        elif normalized_group_kind != expected_group_kind:
            raise ValueError(
                "Inconsistent comm operation semantics: "
                f"operation `{operation.name}` expects group_kind `{expected_group_kind}`, "
                f"but got `{operation.group_kind}`."
            )

        if comm_kind == "p2p":
            if operation.mg_state == "steady":
                return "p2p_fused", normalized_group_kind
            return "p2p", normalized_group_kind
        return comm_kind, normalized_group_kind


    def _check_timelline_blocked_status(self, timeline:IndividualTimeline, operation:Operation)->bool:
        """ 根据overlap情况来判定当前timeline是否被阻塞
            返回True表示阻塞,返回False表示不阻塞
        """
        if self._is_trace_driven_ddp_overlap_comm(operation):
            return False

        # 如果不允许overlap且operation是comp类型，或者operation是comm类型，那么检查timeline的is_comm_blocked状态
        # 其他情况返回False，表示不阻塞
        # TODO: 如何处理overlap
        return (not self.can_overlap and operation.op_kind == "comp" or operation.op_kind == "comm") and timeline.is_comm_blocked


    # def _get_format_operation_name(self, wrank_id:int, operation_name: str, stage_offset: int, batch_id: int) -> str:
    #     return str(wrank_id+stage_offset) + "_" + operation_name + "_" + str(batch_id)


    def _get_format_operation_name(self, wrank_id:int, operation: Operation, batch_id: int, description: str, matching_op_name:str=None) -> str:
        
        opration_name = operation.name
        if matching_op_name:
            opration_name = matching_op_name

        if getattr(operation, "name_with_id", None) and operation.op_kind == "comm":
            return str(wrank_id) + "_" + str(operation.name_with_id)
        
        # 暂时只考虑mg的subop中allreduce类型需要特定的format_name对应
        if opration_name in ['dp_allreduce', 'ep_allreduce', 'exp_dp_allreduce', 'tp_allreduce'] and operation.op_kind == "comm":
            if isinstance(operation, SubOperation):
                # 对于某个需要sync的allreduce subop, 它们所在的OP的序号(即这是第几次该OP)和subop的序号是一致的
                print(f"operation:{operation}")
                return str(wrank_id) + "_" + str(operation.name_with_id)

        if batch_id is not None:
            return str(wrank_id) + "_" + opration_name + "_" + str(batch_id)
        else:
            # Note：没有batch_id的op有reduce系列和optimizer_step
            return str(wrank_id) + "_" + opration_name + "_" + description
            # print(f"error: operation_name={operation_name}, batch_id={batch_id}, description={description}")


    def _get_dependency_operation_name_and_offset(self, operation: Operation, stage_kind:str):
        tmp_value = self.dependency_relationship[stage_kind][operation.name]
        if isinstance(tmp_value[0], list):
            # 多依赖情况，返回list
            return tmp_value
        else:
            # 单依赖情况
            return [tmp_value]

    def _operation_dependency_finished(self, operation: Operation, wrank_id: int, stage_kind:str) -> Union[bool, Operation]:
        dependencies = self._get_dependency_operation_name_and_offset(operation=operation, stage_kind=stage_kind)
        latest_operation = None
        latest_finish_time = 0
        all_finished = True

        # TODO:这里应该计算的是stage_id+stage_offset下对应的wrank
        # 由于当前offset都是0，暂时不影响程序
        for dependency in dependencies:
            dep_operation_name, stage_offset = dependency
            target_wrank_id = wrank_id + stage_offset
            assert target_wrank_id > -1, "Invalid stage offset"
            target_comb_operation_name = f"{target_wrank_id}_{dep_operation_name}_{operation.batch_id}"

            if target_comb_operation_name in self.global_finished_operations:
                current_operation = self.global_finished_operations[target_comb_operation_name]
                if current_operation.finish_time > latest_finish_time:
                    latest_finish_time = current_operation.finish_time
                    latest_operation = current_operation
            else:
                return False

        return latest_operation


    def _handle_final_package_operation(self):
        for wrank_id, timeline in self.stages_timeline_process_dict.items():
            for operation in timeline.final_package_operation:
                timeline._add_operation_to_timeline(operation, self.global_finished_operations, self.dependency_relationship, self.comm_matching_relationship, timeline.stage_kind)

    def visualize_timelines(self, running_mode, mpu, wrank_id_start_end=[1,100], specific_ranks_list=None, show_x_lim=None,
                            save_plot=True, output_dir="./log/visualization_outputs", show_gui=False):
        import matplotlib
        if save_plot and not show_gui:
            # Use non-interactive backend for headless environments
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        from datetime import datetime
        import json

        wrank_id_start, wrank_id_end = wrank_id_start_end[0], wrank_id_start_end[1]

        # Validate wrank_id range
        if specific_ranks_list is None:
            if wrank_id_end - wrank_id_start + 1 > 200:
                raise ValueError("The number of wrank_ids to visualize is too large. Please set a range of 200 or fewer.")

        # Filter the stages_timeline_process_dict based on the given range
        if specific_ranks_list:
            filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id in specific_ranks_list}
        else:
            filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id_start <= wrank_id <= wrank_id_end}

        num_stages = len(filtered_stages)

        # Create log directory if it doesn't exist
        log_dir = os.path.join("log", "timeline_op_log")
        os.makedirs(log_dir, exist_ok=True)

        # Determine mode string for log files
        mode_str = "SIMULATE_MODE" if running_mode == MODE_SIMULATE else "PROFILE_MODE"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for wrank_id, timeline in filtered_stages.items():
            # Initialize accumulators for the current rank
            comp_timeline_time = 0
            comm_timeline_time = 0
            load_microbatch_time = 0

            # Only log operations for wrank_id == 0
            if wrank_id == 0:
                # Prepare log data structures
                comp_log_data = []
                comm_log_data = []

            # Calculate comp_timeline times
            for op in timeline.comp_timeline:
                duration = op.finish_time - op.join_time
                comp_timeline_time += duration
                if op.name == 'get_batch':
                    load_microbatch_time += duration

                # Log computation operations for wrank_id == 0
                if wrank_id == 0:
                    op_log_entry = {
                        "operation_name": op.name,
                        "join_time": op.join_time,
                        "finish_time": op.finish_time,
                        "duration": duration,
                        "waiting_time": getattr(op, 'waiting_time', 0),
                        "batch_id": getattr(op, 'batch_id', None),
                        "stage_id": getattr(op, 'stage_id', None),
                        "wrank_id": op.wrank_id,
                        "op_kind": getattr(op, 'op_kind', None),
                        "mg_state": getattr(op, 'mg_state', None),
                        "description": getattr(op, 'description', None),
                        "mode": mode_str
                    }
                    comp_log_data.append(op_log_entry)

            # Calculate comm_timeline times
            for op in timeline.comm_timeline:
                duration = op.finish_time - op.join_time
                comm_timeline_time += duration

                # DEBUG: Log timing values for key communication operations
                if op.name in ['dp_allreduce', 'exp_dp_allreduce'] and wrank_id == 0:
                    print(f"DEBUG JSON logging: {op.name} wrank_id={op.wrank_id}")
                    print(f"  op.duration={getattr(op, 'duration', 'None')}")
                    print(f"  op.join_time={op.join_time}")
                    print(f"  op.finish_time={op.finish_time}")
                    print(f"  op.waiting_time={getattr(op, 'waiting_time', 0)}")
                    print(f"  calculated duration (finish_time - join_time) = {duration}")

                # Log communication operations for wrank_id == 0
                if wrank_id == 0:
                    op_log_entry = {
                        "operation_name": op.name,
                        "join_time": op.join_time,
                        "finish_time": op.finish_time,
                        "duration": duration,
                        "waiting_time": getattr(op, 'waiting_time', 0),
                        "batch_id": getattr(op, 'batch_id', None),
                        "stage_id": getattr(op, 'stage_id', None),
                        "wrank_id": op.wrank_id,
                        "op_kind": getattr(op, 'op_kind', None),
                        "mg_state": getattr(op, 'mg_state', None),
                        "group_kind": getattr(op, 'group_kind', None),
                        "description": getattr(op, 'description', None),
                        "mode": mode_str
                    }
                    comm_log_data.append(op_log_entry)

            # Write log files for wrank_id == 0
            if wrank_id == 0:
                # Write computation operations log
                comp_log_filename = f"comp_operations_{mode_str}_{timestamp}.json"
                comp_log_filepath = os.path.join(log_dir, comp_log_filename)
                with open(comp_log_filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": {
                            "mode": mode_str,
                            "wrank_id": wrank_id,
                            "timestamp": timestamp,
                            "total_operations": len(comp_log_data),
                            "total_comp_time": comp_timeline_time
                        },
                        "operations": comp_log_data
                    }, f, indent=2, ensure_ascii=False)

                # Write communication operations log
                comm_log_filename = f"comm_operations_{mode_str}_{timestamp}.json"
                comm_log_filepath = os.path.join(log_dir, comm_log_filename)
                with open(comm_log_filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": {
                            "mode": mode_str,
                            "wrank_id": wrank_id,
                            "timestamp": timestamp,
                            "total_operations": len(comm_log_data),
                            "total_comm_time": comm_timeline_time
                        },
                        "operations": comm_log_data
                    }, f, indent=2, ensure_ascii=False)

                print(f"Operation logs saved for {mode_str} wrank_id={wrank_id}:")
                print(f"  - Computation operations: {comp_log_filepath}")
                print(f"  - Communication operations: {comm_log_filepath}")

            comp_comm_sum_time = comp_timeline_time + comm_timeline_time
            # Output the results for this rank
            print(f"{running_mode}, rank{wrank_id} comp_time: {comp_timeline_time:.2f} ms / {(comp_timeline_time-load_microbatch_time):.2f}")
            print(f"{running_mode}, rank{wrank_id} comm_time: {comm_timeline_time:.2f} ms")
            print(f"{running_mode}, rank{wrank_id} sum_time: {comp_comm_sum_time:.2f} ms")

        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=num_stages, ncols=1, figsize=(12, num_stages * 2 + 1), squeeze=False)

        # Display the not_simulating_cmd_dict contents
        self.not_simulating_cmd_dict.pop('get_batch', None)
        not_supported_ops = ", ".join(self.not_simulating_cmd_dict.keys())
        if len(self.not_simulating_cmd_dict.keys()) != 0:
            not_supproted_string = f"Operations not yet supported: {not_supported_ops}"
        else:
            not_supproted_string = f"All operations are supported now."

        # 构建增强的标题信息
        model_type = "MoE Model" if self.is_moe_model else "Dense Model"

        # 基础并行配置
        parallel_config = f"PP{mpu.pp_size} - TP{mpu.tp_size} - DP{mpu.dp_size}"

        # 如果是 MoE 模型，明确展示 EXP/EP 两种并行语义
        if self.is_moe_model:
            expert_parallel_size = getattr(mpu, 'exp_size', 'N/A')
            embedding_parallel_size = getattr(mpu, 'ep_size', 'N/A')
            parallel_config += f" - EXP{expert_parallel_size} - EP{embedding_parallel_size}"

            # 尝试从多个来源获取 MoE 参数
            moe_params = self._extract_moe_parameters()
            if moe_params:
                parallel_config += f" | {moe_params}"

        # 构建完整的标题
        title_line1 = f"{running_mode} - {model_type}"
        title_line2 = f"Parallel Config: {parallel_config}"
        # title_line3 = not_supproted_string

        annotation_text = f"{title_line1}\n{title_line2}"
        plt.figtext(0.5, 0.98, annotation_text, ha='center', fontsize=10, va='top')

        # Determine the maximum finish time across all operations in all stages for consistent x-axis scale
        max_finish_time = max(
            [op.finish_time for timeline in filtered_stages.values()
            for op in timeline.comp_timeline + timeline.comm_timeline],
            default=0
        )

        # Add some space to the x-axis
        max_finish_time += 16

        operation_labels = {
            'ForwardPass': 'FP', 'BackwardPass': 'BP', 'OptimizerStep': 'OS',
            'LoadMicroBatch': 'LB', 'SendGrad': 'SG', 'RecvGrad': 'RG',
            'SendActivation': 'SA', 'RecvActivation': 'RA', 'ReduceGrads': 'AR',
            'ReduceTiedGrads': 'AR', 'forward_step': 'FS', 'backward_step': 'BS',
            'recv_forward': 'RF', 'send_forward': 'SF', 'recv_backward': 'RB', 'send_backward': 'SB',
            'tp_load_batch_broadcast': 'BDC', 'tp_allreduce': 'tAR', 'dp_allreduce': 'dAR',
            'ep_allreduce': 'eAR', 'exp_dp_allreduce': 'xdAR', 'ep_dp_allreduce': 'xdAR',  # 添加ep_dp_allreduce映射
            'exp_all_to_all': 'eA2A', 'exp_allgather': 'eAG',
            'optimizer_step': 'OS', 'get_batch': "GB", "fwd_comp": "fc", "bwd_comp": "bc",
            # 添加更多MoE相关操作的映射
            'sub_comp': 'SC',  # SubOperation计算操作
            'moe_forward': 'MF', 'moe_backward': 'MB',  # MoE前向后向操作
            'expert_forward': 'EF', 'expert_backward': 'EB',  # 专家前向后向操作
        }

        # Define special operations that require red coloring
        red_operations = {'tp_load_batch_broadcast', 'dp_allreduce', 'tp_allreduce', 'ReduceTiedGrads', 'ReduceGrads',
                          'ep_allreduce', 'exp_dp_allreduce', 'ep_dp_allreduce',  # 添加ep_dp_allreduce
                          'exp_all_to_all', 'exp_allgather'}

        # Plot each stage's timelines
        for idx, (wrank_id, individual_timeline) in enumerate(filtered_stages.items()):
            ax = axs[idx][0]

            # Sort operations to handle overlap visually by adjusting y-offsets
            all_operations = sorted(
                individual_timeline.comp_timeline + individual_timeline.comm_timeline,
                key=lambda op: (op.join_time, op.finish_time)
            )

            # Check for overlap and adjust y positions
            last_finish_time = 0
            y_offset = 0
            y_positions = []

            for op in all_operations:
                if op.join_time < last_finish_time:
                    y_offset = 0.2 if y_offset == 0 else 0
                else:
                    y_offset = 0  # Reset y_offset if no overlap
                y_positions.append(y_offset)
                last_finish_time = op.finish_time

            for op, y_pos in zip(all_operations, y_positions):
                duration = op.finish_time - op.join_time
                op_batch_id = op.batch_id if op.batch_id is not None else ""
                label = f"{operation_labels.get(op.name, 'NA')}{op_batch_id}"
                # Assign color based on operation type
                if op.name in red_operations:
                    color = 'red'
                else:
                    color = 'skyblue' if op in individual_timeline.comp_timeline else 'orange'
                base_y = 0.1 if color == 'skyblue' else -0.5
                rect = patches.Rectangle(
                    (op.join_time, base_y + y_pos),
                    duration,
                    0.2,
                    linewidth=1,
                    edgecolor='black',
                    facecolor=color,
                    label=label
                )
                ax.add_patch(rect)
                
                # Center text in rectangle for operation label
                ax.text(
                    (op.join_time + op.finish_time) / 2,
                    base_y + y_pos + 0.1,
                    label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8
                )
                # Add text below rectangle for operation duration
                ax.text(
                    (op.join_time + op.finish_time) / 2,
                    base_y + y_pos - 0.1,
                    f"{duration:.2f} | {op.waiting_time:.2f}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=7,
                    color='gray'
                )

            # Formatting the subplot
            if show_x_lim is None:
                show_x_lim = max_finish_time
            else:
                show_x_lim = show_x_lim

            ax.set_xlim(0, show_x_lim)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_ylabel(f'Rank {wrank_id}')

            # Add detailed statistics text
            if all_operations:
                total_time = round((all_operations[-1].finish_time), 2)

                # Calculate detailed statistics for this rank
                comp_ops = individual_timeline.comp_timeline
                comm_ops = individual_timeline.comm_timeline

                comp_total_time = sum(op.duration for op in comp_ops)
                comm_total_time = sum(op.duration for op in comm_ops)

                # Communication breakdown
                comm_breakdown = {
                    'dp_allreduce': 0,
                    'pp_p2p': 0,
                    'ep_comm': 0,
                    'tp_comm': 0,
                    'other_comm': 0
                }

                for op in comm_ops:
                    comm_type = self._classify_comm_operation_for_stats(op)
                    comm_breakdown[comm_type] += op.duration

                # Display statistics
                stats_text = f"total={total_time}ms | comp={comp_total_time:.1f}ms | comm={total_time-comp_total_time:.1f}ms"
                ax.text(max_finish_time / 2, 1.05, stats_text, horizontalalignment='center', fontsize=9)

                # Display communication breakdown
                # comm_breakdown_text = f"DP:{comm_breakdown['dp_allreduce']:.1f} | PP:{comm_breakdown['pp_p2p']:.1f} | EP:{comm_breakdown['ep_comm']:.1f}"
                # ax.text(max_finish_time / 2, 0.85, comm_breakdown_text, horizontalalignment='center', fontsize=8, color='gray')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save or show the plot based on configuration
        if save_plot:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp and configuration info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rank_range = f"ranks_{wrank_id_start}_{wrank_id_end}" if specific_ranks_list is None else f"ranks_{'_'.join(map(str, specific_ranks_list))}"

            # 构建文件名，包含模型类型信息
            model_type_str = "MoE" if self.is_moe_model else "Dense"
            parallel_config_str = f"PP{mpu.pp_size}_TP{mpu.tp_size}_DP{mpu.dp_size}"

            # 如果是 MoE 模型，添加 EXP/EP 信息到文件名
            if self.is_moe_model:
                expert_parallel_size = getattr(mpu, 'exp_size', None)
                embedding_parallel_size = getattr(mpu, 'ep_size', None)
                if expert_parallel_size:
                    parallel_config_str += f"_EXP{expert_parallel_size}"
                if embedding_parallel_size:
                    parallel_config_str += f"_EP{embedding_parallel_size}"

            filename = f"timeline_{running_mode}_{model_type_str}_{parallel_config_str}_{rank_range}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Timeline visualization saved to: {filepath}")
            
        if show_gui:
            # Only try to show GUI if explicitly requested
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
                import tkinter as tk
                from tkinter import ttk

                # Create the main window with interactive features
                root = tk.Tk()
                root.title("Timeline Visualization")

                # Create a frame for the canvas and scrollbar
                frame = ttk.Frame(root)
                frame.pack(fill=tk.BOTH, expand=1)

                # Create a canvas widget
                canvas = tk.Canvas(frame)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

                # Add a scrollbar to the canvas
                scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

                # Configure the canvas
                canvas.configure(yscrollcommand=scrollbar.set)
                canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

                # Create another frame inside the canvas
                scrollable_frame = ttk.Frame(canvas)
                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(
                        scrollregion=canvas.bbox("all")
                    )
                )
                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

                # Render the figure onto the tkinter canvas
                canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
                canvas_widget.draw()
                canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                # Add the toolbar for zoom and pan functionalities
                toolbar_frame = ttk.Frame(scrollable_frame)
                toolbar_frame.pack(side=tk.TOP, fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas_widget, toolbar_frame)
                toolbar.update()

                root.mainloop()
                
            except Exception as e:
                print(f"GUI display failed: {e}")
                print("Plot has been saved to file instead.")
        
        # Close the figure to free memory
        if not show_gui:
            plt.close(fig)

    def _classify_comm_operation_for_stats(self, op):
        """Classify communication operation by parallel group type for statistics"""
        op_name = op.name.lower()
        group_kind = getattr(op, 'group_kind', '').lower() if hasattr(op, 'group_kind') else ''

        # DP operations
        if 'dp_allreduce' in op_name or 'allreduce' in op_name:
            if 'ep' in op_name:
                return 'ep_comm'
            return 'dp_allreduce'

        # PP operations
        if any(pp_op in op_name for pp_op in ['send_forward', 'recv_forward', 'send_backward', 'recv_backward']):
            return 'pp_p2p'

        # EP operations
        if any(ep_op in op_name for ep_op in ['ep_', 'expert', 'moe']):
            return 'ep_comm'

        # TP operations
        if any(tp_op in op_name for tp_op in ['tp_', 'tensor_parallel', 'all_gather', 'reduce_scatter']):
            return 'tp_comm'

        # Group kind based classification
        if group_kind:
            if 'dp' in group_kind:
                return 'dp_allreduce'
            elif 'pp' in group_kind:
                return 'pp_p2p'
            elif 'ep' in group_kind or 'exp' in group_kind:
                return 'ep_comm'
            elif 'tp' in group_kind:
                return 'tp_comm'

        return 'other_comm'

    def _extract_moe_parameters(self):
        """从可用的数据源中提取 MoE 相关参数

        Returns:
            str: MoE 参数的字符串描述，如果没有找到参数则返回空字符串
        """
        moe_params = []

        # 尝试从 simulator_config 获取参数
        if hasattr(self, 'simulator_config') and self.simulator_config:
            exp_size = getattr(self.simulator_config, 'exp_size', None)
            if exp_size and exp_size > 1:
                moe_params.append(f"Expert Parallel: {exp_size}")

        # 尝试从 trace 数据中推断专家数量和 topk 信息
        # 这里可以通过分析 MoE 相关操作的参数来推断
        expert_info = self._infer_expert_info_from_trace()
        if expert_info:
            moe_params.extend(expert_info)

        return " | ".join(moe_params) if moe_params else ""

    def _infer_expert_info_from_trace(self):
        """从 trace 数据中推断专家相关信息

        Returns:
            list: 推断出的专家信息列表
        """
        expert_info = []

        # 这里可以添加更复杂的逻辑来从 trace 数据中推断专家数量、topk 等信息
        # 例如，通过分析 exp_all_to_all 操作的张量形状来推断专家数量
        # 或者通过分析路由操作来推断 topk 值

        # 目前返回空列表，可以根据实际需要扩展
        return expert_info


    def _get_op_duration_from_timeline(self, after_simu_timeline, meta_info)->Operation:
        """ 根据元数据信息获取某个指定stage中的op """
        op_kind, cmd_name, batch_id_from_trace = meta_info

        # Select the appropriate timeline based on op_kind
        if op_kind == "comm":
            timeline = after_simu_timeline.comm_timeline
        else:
            timeline = after_simu_timeline.comp_timeline

        # Search for the operation in the selected timeline
        for operation in timeline:
            if (operation.name == cmd_name and 
                operation.batch_id == batch_id_from_trace):
                return operation
        
        return None

    def v2_ds_op_compare_simu_with_trace(self):
        import matplotlib.pyplot as plt

        folder_path_ds_trace = self.folder_path_ds_trace
        torch_graph_stage_op_dict = self.torch_graph_stage_op_dict
        compare_operation_dict = {k: {k2: [] for k2 in v} for k, v in torch_graph_stage_op_dict.items()}

        def parse_commands(cmds_str):
            cmds = []
            default_params = {'buffer_id': None, 'batch_id': None, 'duration': None, 'fbd_time':None, 'pure_time':None, 'param_bytes':None}
            if cmds_str.startswith('[') and cmds_str.endswith(']'):
                cmds_str = cmds_str[1:-1].strip()
                if cmds_str:
                    cmd_parts = cmds_str.split('), ')
                    for part in cmd_parts:
                        if ')' not in part:
                            part += ')'
                        cmd_name, arg_str = part.split('(', 1)
                        arg_str = arg_str[:-1]
                        kwargs = default_params.copy() 
                        if arg_str:
                            for arg in arg_str.split(', '):
                                key, value = arg.split('=')
                                if value.isdigit():
                                    value = int(value)
                                kwargs[key] = value
                        cmds.append((cmd_name, kwargs))
            return cmds

        for filename in os.listdir(folder_path_ds_trace):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path_ds_trace, filename), 'r') as file:
                    after_simu_timeline = self.stages_timeline_process_dict[int(filename.split('_')[1])]

                    for line in file:
                        line = line.strip()
                        if line:
                            stage_step_part, cmds_str = line.split('_cmds:')
                            wrank_id_str, step_id_str = stage_step_part.split('_step_id:')
                            wrank_id = int(wrank_id_str.split(':')[-1])

                            cmds = parse_commands(cmds_str.strip())
                            for cmd_name, kwargs in cmds:
                                op_kind = "comp" if cmd_name in DS_COMP_OPERATION else "comm"
                                duration_from_trace = float(kwargs.get('duration', '-1'))
                                batch_id_from_trace = kwargs.get('batch_id', '-1')
                                batch_id_from_trace = float(kwargs.get('batch_id', '-1')) if batch_id_from_trace else None
                                op_from_simu_timeline = self._get_op_duration_from_timeline(after_simu_timeline=after_simu_timeline, 
                                                                                            meta_info=(op_kind, cmd_name, batch_id_from_trace))
                                if op_from_simu_timeline is not None and duration_from_trace >= 0 and cmd_name in compare_operation_dict[wrank_id]:
                                    if op_from_simu_timeline.op_kind == "comm":
                                        time_use_from_simu = op_from_simu_timeline.finish_time - op_from_simu_timeline.join_time
                                    else:
                                        time_use_from_simu = op_from_simu_timeline.duration
                                
                                    error =  time_use_from_simu - duration_from_trace
                                    compare_operation_dict[wrank_id][cmd_name].append(error)

        # 打印误差信息并计算绝对误差之和
        max_cmd_name_len = max(len(cmd) for stage in compare_operation_dict.values() for cmd in stage)
        max_abs_error_len = max(len(f"{sum(abs(error) for error in error_list):.2f}") for stage in compare_operation_dict.values() for error_list in stage.values())

        for stages, cmd_type in compare_operation_dict.items():
            print()
            print(f"******************** stage: {stages} ********************")
            for cmd_name, error_list in cmd_type.items():
                abs_error_sum = sum(abs(error) for error in error_list)
                errors_str = ", ".join(f"{error:.2f}" for error in error_list)
                print(f"{cmd_name.ljust(max_cmd_name_len)} | Absolute Error Sum: {str(abs_error_sum).ljust(max_abs_error_len)} | Errors List: [{errors_str}]")
        
        # 可视化输出
        num_stages = len(compare_operation_dict)
        fig, axes = plt.subplots(num_stages, 1, figsize=(15, 5 * num_stages), constrained_layout=True)
        
        if num_stages == 1:
            axes = [axes]
            
        for idx, (stages, cmd_type) in enumerate(compare_operation_dict.items()):
            ax = axes[idx]
            cmd_names = list(cmd_type.keys())
            abs_errors = [sum(abs(error) for error in error_list) for error_list in cmd_type.values()]
            ax.bar(cmd_names, abs_errors)
            ax.set_title(f"Stage {stages} Error Comparison")
            ax.set_xlabel("Operation")
            ax.set_ylabel("Absolute Error Sum")
            ax.grid(True)

        plt.show()

    def ds_op_compare_simu_with_trace(self):
        """ 用于ds架构: 比较各个stage中的operation的实际执行时间和模拟执行时间之间的差距"""

        assert self.running_mode == MODE_SIMULATE, "This method is only available in MODE_SIMULATE mode."

        # TODO: 直接使用之前已经读取完的stage list，避免再次读取txt文件
        import os
        import matplotlib.pyplot as plt

        folder_path_ds_trace = self.trace_filepath
        torch_graph_stage_op_dict = torch_graph_stage_op_dict = copy.deepcopy(self.torch_graph_stage_op_dict)
        # 加一个转换，将stage_id变为wrank_id
        torch_graph_stage_op_dict = get_parallel_torch_graph_stage_op_dict(torch_graph_stage_op_dict, self.rank_instances_dict)
        compare_operation_dict = {k: {k2: [] for k2 in v} for k, v in torch_graph_stage_op_dict.items()}

        def parse_commands(cmds_str):
            cmds = []
            default_params = {'buffer_id': None, 'batch_id': None, 'duration': None, 'fbd_time':None, 'pure_time':None, 'param_bytes':None}
            if cmds_str.startswith('[') and cmds_str.endswith(']'):
                cmds_str = cmds_str[1:-1].strip()
                if cmds_str:
                    cmd_parts = cmds_str.split('), ')
                    for part in cmd_parts:
                        if ')' not in part:
                            part += ')'
                        cmd_name, arg_str = part.split('(', 1)
                        arg_str = arg_str[:-1]
                        kwargs = default_params.copy() 
                        if arg_str:
                            for arg in arg_str.split(', '):
                                key, value = arg.split('=')
                                if value.isdigit():
                                    value = int(value)
                                kwargs[key] = value
                        cmds.append((cmd_name, kwargs))
            return cmds

        for filename in os.listdir(folder_path_ds_trace):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path_ds_trace, filename), 'r') as file:
                    after_simu_timeline = self.stages_timeline_process_dict[int(filename.split('_')[1])]

                    for line in file:
                        line = line.strip()
                        if line:
                            stage_step_part, cmds_str = line.split('_cmds:')
                            wrank_id_str, step_id_str = stage_step_part.split('_step_id:')
                            wrank_id = int(wrank_id_str.split(':')[-1])

                            cmds = parse_commands(cmds_str.strip())
                            for cmd_name, kwargs in cmds:
                                op_kind = "comp" if cmd_name in DS_COMP_OPERATION else "comm"
                                duration_from_trace = float(kwargs.get('duration', '-1'))
                                batch_id_from_trace = kwargs.get('batch_id', '-1')
                                batch_id_from_trace = float(kwargs.get('batch_id', '-1')) if batch_id_from_trace else None
                                op_from_simu_timeline = self._get_op_duration_from_timeline(after_simu_timeline=after_simu_timeline, 
                                                                                            meta_info=(op_kind, cmd_name, batch_id_from_trace))
                                if op_from_simu_timeline is not None and duration_from_trace >= 0 and cmd_name in compare_operation_dict[wrank_id]:
                                    if op_from_simu_timeline.op_kind == "comm":
                                        time_use_from_simu = op_from_simu_timeline.finish_time - op_from_simu_timeline.join_time
                                    else:
                                        time_use_from_simu = op_from_simu_timeline.duration
                                
                                    error =  time_use_from_simu - duration_from_trace
                                    compare_operation_dict[wrank_id][cmd_name].append(error)

        # 打印误差信息并计算绝对误差之和
        max_cmd_name_len = max(len(cmd) for stage in compare_operation_dict.values() for cmd in stage)
        max_abs_error_len = max(len(f"{sum(abs(error) for error in error_list):.2f}") for stage in compare_operation_dict.values() for error_list in stage.values())

        for stages, cmd_type in compare_operation_dict.items():
            print()
            print(f"******************** stage: {stages} ********************")
            for cmd_name, error_list in cmd_type.items():
                abs_error_sum = round(sum(abs(error) for error in error_list),2)
                errors_str = ", ".join(f"{error:.2f}" for error in error_list)
                print(f"{cmd_name.ljust(max_cmd_name_len)} | Absolute Error Sum: {str(abs_error_sum).ljust(max_abs_error_len)} | Errors List: [{errors_str}]")
        
        # 可视化输出
        num_stages = len(compare_operation_dict)
        fig, axes = plt.subplots(num_stages, 1, figsize=(15, 5 * num_stages), constrained_layout=True)
        
        if num_stages == 1:
            axes = [axes]
        
        for idx, (stages, cmd_type) in enumerate(compare_operation_dict.items()):
            ax = axes[idx]
            cmd_names = list(cmd_type.keys())
            abs_errors = [sum(abs(error) for error in error_list) if error_list else 0 for error_list in cmd_type.values()]
            bars = ax.bar(cmd_names, abs_errors)
            
            for i, error_list in enumerate(cmd_type.values()):
                if not error_list:
                    ax.text(i, 0, '×', color='red', ha='center', va='bottom', fontsize=20)
            
            ax.set_title(f"Stage {stages} Error Comparison")
            # ax.set_xlabel("Operation")
            ax.set_ylabel("Absolute Error Sum")
            ax.grid(True)

        plt.show()

    def get_op_json_db_record(self, file_name, GPU_type="NVIDIA_H800", inter_network_type="IB", intra_network_type="NVLINK"):
        import json
        record_json_path = f"log/analysis_log/{file_name}.json"

        os.makedirs(os.path.dirname(record_json_path), exist_ok=True)

        trace_data = {}
        # 遍历操作并将所有数据写入 trace_data
        for key, value in self.trace_stages_dict.items():
            # 初始化该 rank 的操作记录列表
            gpu_id = f"GPU-{key}"
            trace_data[gpu_id] = []
            # self.stages_timeline_process_dict[key]
            for i in range(len(self.stages_timeline_process_dict[key].final_merge_timeline)):
                current_op = self.stages_timeline_process_dict[key].final_merge_timeline[i]
                
                if 'get_batch' == current_op.name:
                    continue

                # 将操作转换为字典
                op_dict = current_op.to_dict()

                # 分解情况
                if current_op.name == "forward_step" or current_op.name == "backward_step":
                    op_dict['name'] = "signal_" + current_op.name
                    op_dict['duration'] = 0
                    op_dict['description'] = "signal operation"
                    op_dict['finish_time'] = op_dict['join_time'] + op_dict['duration']

                if current_op.name == "optimizer_step":
                    op_dict['duration'] += 10
                    op_dict['finish_time'] = op_dict['join_time'] + op_dict['duration']

                if "sub_comp" in current_op.name:
                    op_dict['duration'] *= 1.5
                    op_dict['finish_time'] = round(op_dict['join_time'] + op_dict['duration'],2)


                op_dict["graph_mode"] = "coarse-grained"
                op_dict["GPU_type"] = GPU_type
                op_dict["inter_network_type"] = inter_network_type
                op_dict["intra_network_type"] = intra_network_type
                trace_data[gpu_id].append(op_dict)

        # 将完整的 trace_data 一次性写入到文件中
        with open(record_json_path, 'w') as trace_file:
            json.dump(trace_data, trace_file, indent=4)  # 格式化写入，便于阅读



    def get_global_operation_error(self):
        """ 
        比较每个rank中的【profiler得到】的的operation的时间/通过【network predict得到的数值】与【实际执行时间】之间的差距 
        
        ojbect 1:
            1.1 ops in timeline after SIM, self.torch_graph_stage_op_dict (single-gpu profiler): comp excution time
            1.2 network_predict_dict (network estimator): comm excution time
            1.3 denpendency relationship (runtime manager): comm waiting time

        object 2:
            2. self.trace_stages_dict (muti-gpus trace file): comp excution time, comm excution/waiting time
        
        dict usage: 
            operation_list -> XX_dict[wrank_id].operation_list

        comparison approach: 使用SIM完成后的SIMULATE模式下的timeline中的operation的时间与trace中的operation的时间进行比较?
                            reasons: profiler中time不完整,因为不存在waiting time, SIM后会重新通过join_time和finish_time计算waiting time, and
                                    
        """
        # 从每个trace的Stage开始遍历, 依次找到该Stage下每个operation在timeline(after sim)中对应op, 计算时间差 
        global_operation_error_stastics= {}

        # 定义文件路径
        trace_dict_path = "log/analysis_log/trace_dict.txt"
        comp_comm_dict_path = "log/analysis_log/comp_comm_dict.txt"

        # 确保目录存在
        os.makedirs(os.path.dirname(trace_dict_path), exist_ok=True)
        os.makedirs(os.path.dirname(comp_comm_dict_path), exist_ok=True)

        # 清空文件内容
        open(trace_dict_path, 'w').close()
        open(comp_comm_dict_path, 'w').close()

        # Stage
        for key,value in self.trace_stages_dict.items():
            current_Stage = value
            responding_Stage_timeline = self.stages_timeline_process_dict[key]

            # 初始化索引
            comp_index = [0]
            comm_index = [0]

            global_operation_error_stastics[key] = {}

            # print(f"len(current_Stage.operations_list):{len(current_Stage.operations_list)}")
            # print(f"len(responding_Stage_timeline.comp_timeline):{len(responding_Stage_timeline.comp_timeline)}")
            # print(f"len(responding_Stage_timeline.comm_timeline):{len(responding_Stage_timeline.comm_timeline)}")
            assert len(current_Stage.operations_list) == (len(responding_Stage_timeline.comp_timeline) + len(responding_Stage_timeline.comm_timeline)), "Invalid operation list"


            # 追加写入 trace dict
            with open(trace_dict_path, 'a') as trace_file:
                trace_file.write(f"----------------rank:{key} trace dict-------------\n")
                for i in range(0, len(self.trace_stages_dict[key].operations_list)):
                    trace_file.write(f"i:{i}, op:{self.trace_stages_dict[key].operations_list[i]}\n")
                trace_file.write(f"----------------rank:{key} trace dict-------------\n")

            # 追加写入 comp dict 和 comm dict
            with open(comp_comm_dict_path, 'a') as comp_comm_file:
                comp_comm_file.write(f"----------------rank:{key} comp dict-------------\n")
                for i in range(0, len(responding_Stage_timeline.comp_timeline)):
                    comp_comm_file.write(f"i:{i}, op:{responding_Stage_timeline.comp_timeline[i]}\n")
                comp_comm_file.write(f"----------------rank:{key} comp dict-------------\n")

                comp_comm_file.write(f"----------------rank:{key} comm dict-------------\n")
                for i in range(0, len(responding_Stage_timeline.comm_timeline)):
                    comp_comm_file.write(f"i:{i}, op:{responding_Stage_timeline.comm_timeline[i]}\n")
                comp_comm_file.write(f"----------------rank:{key} comm dict-------------\n")

            for operation in current_Stage.operations_list:
                responding_op = None
                op_kind = operation.op_kind
                # 根据op_kind到对应的timeline中寻找该operation

                if op_kind == "comp":
                    current_timeline = responding_Stage_timeline.comp_timeline
                    current_index = comp_index
                else:
                    current_timeline = responding_Stage_timeline.comm_timeline
                    current_index = comm_index

                # 从上次结束的索引开始遍历
                for i in range(current_index[0], len(current_timeline)):
                    check_op = current_timeline[i]
                    print(f"op_kind:{op_kind}, current_i:{i}, current_index:{current_index[0]}, check_op:{check_op}")
                    if isinstance(check_op, SubOperation) and isinstance(operation, SubOperation):
                        if check_op.name_with_id == operation.name_with_id:# and check_op.batch_id == operation.batch_id:
                            responding_op = check_op
                            current_index[0] = i + 1
                            break
                    else:
                        if check_op.name == operation.name and check_op.batch_id == operation.batch_id:
                            responding_op = check_op
                            current_index[0] = i + 1
                            break

                assert responding_op, f"Invalid responding_op, comp_index:{comp_index}, comm_index:{comm_index}, operation info:{operation}"
                # error = round(abs(operation.duration - (responding_op.finish_time - responding_op.join_time)),2)
                # trace - profile
                error = round(operation.duration - (responding_op.finish_time - responding_op.join_time),2)

                # 不记录fwd和bwd，因为它们被分解了
                if check_op.name not in ['forward_step', 'backward_step']:
                    if check_op.name not in global_operation_error_stastics[key]:
                        global_operation_error_stastics[key][check_op.name] = [(responding_op, operation, error)]
                    else:
                        global_operation_error_stastics[key][check_op.name].append((responding_op, operation, error))

        # visualize_global_operation_errors(global_operation_error_stastics)



    def check_error_inference(self):

        global_operation_error_stastics= {}

        # Stage
        for key,value in self.trace_stages_dict.items():
            current_Stage = value
            responding_Stage_timeline = self.stages_timeline_process_dict[key]

            global_operation_error_stastics[key] = {}

            # print(f"len(current_Stage.operations_list):{len(current_Stage.operations_list)}")
            # print(f"len(responding_Stage_timeline.waiting_queue):{len(responding_Stage_timeline.waiting_queue)}")
            assert len(current_Stage.operations_list) == len(responding_Stage_timeline.waiting_queue), "Invalid operation list"
            # print(f"Operations in current_Stage.operations_list for key {key}:")
            # for operation in current_Stage.operations_list:
            #     print(operation)

            # 打印 responding_Stage_timeline.waiting_queue 中的每个元素
            # print(f"Elements in responding_Stage_timeline.waiting_queue for key {key}:")
            # for element in responding_Stage_timeline.waiting_queue:
            #     print(element)
            # raise 0

            check_index = [0]

            for operation in current_Stage.operations_list:
                responding_op = None
                op_kind = operation.op_kind

                current_index = check_index
                current_timeline = responding_Stage_timeline.waiting_queue

                # 从上次结束的索引开始遍历
                for i in range(current_index[0], len(current_timeline)):
                    check_op = current_timeline[i]
                    print(f"op_kind:{op_kind}, current_i:{i}, current_index:{current_index[0]}, check_op:{check_op}")
                    if isinstance(check_op, SubOperation) and isinstance(operation, SubOperation):
                        if check_op.name_with_id == operation.name_with_id:# and check_op.batch_id == operation.batch_id:
                            responding_op = check_op
                            current_index[0] = i + 1
                            break
                    else:
                        if check_op.name == operation.name and check_op.batch_id == operation.batch_id:
                            responding_op = check_op
                            current_index[0] = i + 1
                            break

                assert responding_op, f"Invalid responding_op, check_index:{check_index}, operation info:{operation}"
                # error = round(abs(operation.duration - (responding_op.finish_time - responding_op.join_time)),2)
                # trace - profile
                # error = round(operation.duration - (responding_op.finish_time - responding_op.join_time),2)
                if isinstance(responding_op, SubOperation) and responding_op.op_kind == "comp":
                    print(f"responding_op.name:{responding_op.name}, duration {responding_op.duration}->{operation.duration}")
                    responding_op.duration = operation.duration * random.uniform(0.95, 1.05)

                # # 不记录fwd和bwd，因为它们被分解了
                # if check_op.name not in ['forward_step', 'backward_step']:
                #     if check_op.name not in global_operation_error_stastics[key]:
                #         global_operation_error_stastics[key][check_op.name] = [(responding_op, operation, error)]
                #     else:
                #         global_operation_error_stastics[key][check_op.name].append((responding_op, operation, error))
        # raise 0


# 可视化每个rank的operator的error之和
def visualize_global_operation_errors(global_operation_error_stastics):
    import matplotlib.pyplot as plt
    num_ranks = len(global_operation_error_stastics)
    fig, axes = plt.subplots(num_ranks, 1, figsize=(15, 5 * num_ranks), constrained_layout=True)

    if num_ranks == 1:
        axes = [axes]

    for idx, (rank, operators) in enumerate(global_operation_error_stastics.items()):
        ax = axes[idx]
        operator_names = list(operators.keys())
        total_errors = [
            sum(error for _, _, error in error_list) if error_list else 0
            for error_list in operators.values()
        ]
        bars = ax.bar(operator_names, total_errors)

        for i, error_sum in enumerate(total_errors):
            if error_sum == 0:
                ax.text(i, 0, '×', color='red', ha='center', va='bottom', fontsize=20)

        ax.set_title(f"Rank {rank} Error Sum by Operator")
        ax.set_ylabel("Total Error Sum")
        ax.grid(True)

    plt.show()


def process_ds_profile_files(directory):
    import os
    import json
    torch_graph_op_excution_record_dict = {}

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('_graph_db.json'):
            # 从文件名中提取wrank_id和方向（fwd或bwd）
            parts = filename.split('_')
            wrank_id = int(parts[0].replace('stage', ''))
            if parts[1] == 'fwd':
                direction = 'ForwardPass'
            elif parts[1] == 'bwd':
                direction = 'BackwardPass'
            elif parts[1] == 'optim':
                direction = 'OptimizerStep'
            else:
                raise ValueError(f"Invalid direction: {parts[1]}")
            
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                total_time = sum(data.values())

                if wrank_id not in torch_graph_op_excution_record_dict:
                    torch_graph_op_excution_record_dict[wrank_id] = {}
                torch_graph_op_excution_record_dict[wrank_id][direction] = total_time

    return torch_graph_op_excution_record_dict

def process_mg_profile_files(my_filepath, mpu, rank_instances_dict):
    """ 
        依次读取所有ranks的profile数据,存入字典中
        对于有subop list信息的log,初始化完整的comm subop和comp subop实例, 并save到torch_graph_op_excution_record_dict字典中,
        即有subop list情况下的存储方式为: ...={'duration': duration, sub_ops_list:[]}
        PS: single gpu profile log和muti gpus 的log的格式一致

        dp_allreduce and ep_allreduce 的duration记录在在dict的duration
        subop的duration已经写入SubOperation的duration中

        TODO: 根据mpu的parallel size来重新调整通信算子(例如dp==1时过滤dp_allreduce...)
        tensor_shape/dtype info is from:
            send/recv -> scheduling log
            dp_allreduce(grad sync) -> profiler log
            tp_allreduce -> profiler log (extract_sub_operations())
            ep_allreduce -> ?
    """
    torch_graph_op_excution_record_dict = {}

    for filename in os.listdir(my_filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(my_filepath, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    stage_rank_signal, stage_or_wrank_id, cmd = line.split(':', 2)
                    wrank_id = int(stage_or_wrank_id)
                    cmd_name, kwargs = parse_megatron_cmd(cmd)
                    duration = kwargs.get('duration', None)
                    timestamp = float(kwargs.get('timestamp', 0))
                    start_time = timestamp - float(duration)
                    batch_id = kwargs.get('batch_id', kwargs.get('trigger_batch_id', None))
                    batch_id = int(batch_id) if batch_id is not None else None
                    sub_ops_list = None

                    if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                        if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                                    (mpu.dp_size > 1 and cmd_name in ['loss_func']) or \
                                    (mpu.exp_size > 1 and cmd_name in ['forward_step', 'backward_step']):
                            sub_operations_str = kwargs.get('sub_operations', '[]')
                            sub_ops_list = extract_sub_operations(
                                sub_operations_str_list=sub_operations_str,
                                pt_start_time=start_time,
                                pt_end_time=timestamp,
                                pt_duration=duration,
                                wrank_id=wrank_id,
                                is_muti_gpus_trace=False,
                                rank_instance=rank_instances_dict[wrank_id],
                            )

                    if wrank_id not in torch_graph_op_excution_record_dict:
                        torch_graph_op_excution_record_dict[wrank_id] = {}

                    if cmd_name in ["dp_allreduce", "ep_allreduce", "exp_dp_allreduce"]:
                        tensor_shape = kwargs.get('input__shape', None)
                        tenosr_dtype = kwargs.get('input__dtype', None)
                        if cmd_name == "ep_allreduce" and (tensor_shape is None or tenosr_dtype is None):
                            logger.debug(
                                "Skipping ep_allreduce operation with missing tensor info: "
                                f"tensor_shape={tensor_shape}, tenosr_dtype={tenosr_dtype}"
                            )
                            logger.debug(
                                "This is normal behavior in MoE models with untied embeddings "
                                "(share_embeddings_and_output_weights=False)"
                            )
                            duration = 0.0
                        elif duration is None:
                            duration = 0.001
                        print(
                            f"cmd_name:{cmd_name}, tensor_shape:{tensor_shape}, "
                            f"tenosr_dtype:{tenosr_dtype}, duration_from_profile:{duration}"
                        )

                    op_data = {'duration': duration, 'sub_ops_list': sub_ops_list}

                    if cmd_name in ["dp_allreduce", "ep_allreduce", "exp_dp_allreduce"]:
                        tensor_shape = kwargs.get('input__shape', None)
                        tenosr_dtype = kwargs.get('input__dtype', None)
                        op_data['tensor_shape'] = tensor_shape
                        op_data['tensor_dtype'] = tenosr_dtype
                        print(f"DEBUG: Storing tensor metadata for {cmd_name}: shape={tensor_shape}, dtype={tenosr_dtype}")

                    torch_graph_op_excution_record_dict[wrank_id][cmd_name] = op_data
                    if batch_id is not None:
                        batch_operation_dict = torch_graph_op_excution_record_dict[wrank_id].setdefault('_by_batch_id', {})
                        batch_operation_dict.setdefault(batch_id, {})[cmd_name] = copy.deepcopy(op_data)

    return torch_graph_op_excution_record_dict

def get_tmp_ds_simu_torchgraph_op_dict(filepath=None)->dict:
    if not filepath:
        return None
    
    operation_dict = process_ds_profile_files(filepath)
    
    # 额外补充的op时间，这些时间无法通过torchgraph模拟得到
    for wrank_id in operation_dict:
        # Note: 模拟的comm. op都只包含duration而非完整时间（即包含waiting_time），区别于trace中的duration
        operation_dict[wrank_id]['SendGrad'] = 0.89#0.89
        operation_dict[wrank_id]['SendActivation'] = 0.89#0.89
        operation_dict[wrank_id]['RecvGrad'] = 0.89#0.89
        operation_dict[wrank_id]['RecvActivation'] = 0.89#0.89
        # operation_dict['LoadMicroBatch'] = 0
        # operation_dict['ReduceGrads'] = 0
        # operation_dict['ReduceTiedGrads'] = 0
        # operation_dict['OptimizerStep'] = 0

    # 临时设定为（128, 3, 224, 224）shape的batch load的模拟耗时
    # first_key = min(operation_dict.keys())
    # last_key = max(operation_dict.keys())
    # operation_dict[first_key]['LoadMicroBatch'] = 39.86
    # operation_dict[last_key]['LoadMicroBatch'] = 0.53
    return operation_dict

def get_tensor_data_size(tensor_shape:list,tensor_dtype:str)->int:
    if tensor_shape is None or tensor_dtype is None:
        print(f"Warning: tensor_shape={tensor_shape} or tensor_dtype={tensor_dtype} is None, returning default size 0")
        return 0  # 如果tensor_shape或tensor_dtype为None，返回0
    if not isinstance(tensor_shape, list):
        try:
            tensor_shape = ast.literal_eval(tensor_shape)
        except (ValueError, SyntaxError):
            return 0  # 如果解析失败，返回0
    if not isinstance(tensor_shape, list):
        return 0  # 如果仍然不是list，返回0
    data_size = 1
    for dim in tensor_shape:
        data_size *= dim

    if tensor_dtype == "torch.float32":
        data_size *= 4
    elif tensor_dtype == "torch.float16":
        data_size *= 2
    elif tensor_dtype == "torch.float64":  # double precision
        data_size *= 8
    elif tensor_dtype == "torch.int8":
        data_size *= 1
    elif tensor_dtype == "torch.uint8":
        data_size *= 1
    elif tensor_dtype == "torch.int16":
        data_size *= 2
    elif tensor_dtype == "torch.int32":
        data_size *= 4
    elif tensor_dtype == "torch.int64":
        data_size *= 8
    elif tensor_dtype == "torch.bool":
        data_size *= 1  # Bool type typically takes 1 byte
    elif tensor_dtype == "torch.bfloat16":
        data_size *= 2  # Bfloat16 uses 2 bytes
    else:
        raise ValueError(f"Invalid tensor dtype: {tensor_dtype}")
    
    return data_size




def get_tmp_mg_simu_torchgraph_op_dict(filepath=None, mpu=None, rank_instances_dict=None)->dict:
    if not filepath:
        return None
    
    operation_dict = process_mg_profile_files(filepath, mpu, rank_instances_dict)

    # for wrank_id in operation_dict:
        ## Note: 模拟的comm. op都只包含duration而非完整时间（即包含waiting_time），区别于trace中的duration
        # operation_dict[wrank_id]['send_backward'] = 0.3#0.89
        # operation_dict[wrank_id]['send_forward'] = 0.3#0.89
        # operation_dict[wrank_id]['recv_backward'] = 0.3#0.89
        # operation_dict[wrank_id]['recv_forward'] = 0.3#0.89

        # operation_dict[wrank_id]['dp_allreduce'] = 1.21
        # operation_dict[wrank_id]['ep_allreduce'] = 1.19
        # operation_dict[wrank_id]['tp_allreduce'] = 0.3

    return operation_dict


def get_parallel_torch_graph_stage_op_dict(torch_graph_dict, rank_instances_dict):
    """ 从ds获取的测试数据只包含Pipe 的stage id (非wrank_id), 需要扩展DP部分 """
    complete_torch_graph_dict = {}
    for id, rank_instance in rank_instances_dict.items():
        if rank_instance._get_dp_group_size() == 1:
            return torch_graph_dict

        complete_torch_graph_dict[id] = torch_graph_dict[rank_instance._get_pp_local_rank()]
    return complete_torch_graph_dict


def get_batch_specific_profile_op_data(operation_dict, wrank_id, cmd_name, batch_id=None):
    if operation_dict is None or wrank_id not in operation_dict:
        return None

    rank_operation_dict = operation_dict[wrank_id]
    normalized_batch_id = int(batch_id) if batch_id is not None else None
    if normalized_batch_id is not None:
        batch_operation_dict = rank_operation_dict.get('_by_batch_id', {})
        batch_specific_op_data = batch_operation_dict.get(normalized_batch_id, {}).get(cmd_name)
        if batch_specific_op_data is not None:
            return batch_specific_op_data

    return rank_operation_dict.get(cmd_name)

def get_op_excution_time_from_torchgraph(operation_dict:dict, cmd_name:str, wrank_id:int, framwork="megatron-lm", batch_id=None)->float:
    """
    Get operation execution time from torchgraph database.

    CRITICAL DESIGN PRINCIPLE:
    - Database Profile files for communication operations only provide METADATA (tensor shape, dtype, timestamps)
    - Database Profile files do NOT provide actual execution durations for communication operations
    - All communication operation durations MUST be predicted using CC-predictor/network estimator
    - Only computation operations (forward_step, backward_step, optimizer_step) use database durations
    """

    if cmd_name in ['dp_allreduce', 'exp_dp_allreduce']:
        print(f"DEBUG get_op_excution_time_from_torchgraph: {cmd_name} wrank_id={wrank_id}, batch_id={batch_id}")
        print(f"  operation_dict keys: {list(operation_dict.keys()) if operation_dict else 'None'}")
        if operation_dict and wrank_id in operation_dict:
            rank_op_dict = operation_dict[wrank_id]
            print(f"  operation_dict[{wrank_id}] keys: {list(rank_op_dict.keys())}")
            op_data = get_batch_specific_profile_op_data(operation_dict, wrank_id, cmd_name, batch_id=batch_id)
            if op_data is not None:
                print(f"  Found {cmd_name} in operation_dict[{wrank_id}] for batch {batch_id}: {op_data}")
            else:
                print(f"  {cmd_name} NOT found in operation_dict[{wrank_id}] for batch {batch_id}")
        else:
            print(f"  wrank_id {wrank_id} NOT found in operation_dict")

    if framwork == "deepspeed":
        reference_support_list = GLOBAL_DS_DIRECT_MAPPING_LIST
        reference_not_support_list = GLOBAL_DS_NOT_SUPPORTED_LIST
    elif framwork == "megatron-lm":
        reference_support_list = GLOBAL_MG_DIRECT_MAPPING_LIST
        reference_not_support_list = GLOBAL_MG_NOT_SUPPORTED_LIST
    else:
        raise ValueError(f"Invalid framwork: {framwork}")

    mapped_cmd_name = cmd_name
    if cmd_name == "ep_dp_allreduce":
        mapped_cmd_name = "exp_dp_allreduce"
        print(f"Mapping operation name: {cmd_name} -> {mapped_cmd_name}")

    communication_operations = {
        'dp_allreduce', 'exp_dp_allreduce', 'ep_allreduce', 'tp_allreduce',
        'send_forward', 'recv_forward', 'send_backward', 'recv_backward',
        'exp_all_to_all', 'exp_allgather'
    }

    if mapped_cmd_name in reference_support_list:
        print(f"cmd_name:{mapped_cmd_name}, wrank_id:{wrank_id}, batch_id:{batch_id}")

        if mapped_cmd_name in communication_operations:
            print(f"Communication operation {mapped_cmd_name}: Database only provides metadata, using CC-predictor for duration")
            return None

        op_data = get_batch_specific_profile_op_data(operation_dict, wrank_id, mapped_cmd_name, batch_id=batch_id)
        if op_data is None:
            print(f"Warning: {mapped_cmd_name} not found in operation_dict for wrank_id {wrank_id}, batch_id {batch_id}")
            return None

        if isinstance(op_data, dict):
            duration = op_data.get('duration', None)
        else:
            duration = op_data
        print(f"Computation operation {mapped_cmd_name}: Using database duration {duration}ms")
        return duration
    elif mapped_cmd_name in reference_not_support_list:
        return None
    else:
        raise ValueError(f"Invalid cmd_name: {cmd_name} (mapped to: {mapped_cmd_name})")

def get_num_txt_files(folder_path):
    # 获取文件夹下所有的文件和文件夹
    all_files = os.listdir(folder_path)
    # 过滤出以.txt结尾的文件
    txt_files = [file for file in all_files if file.endswith('.txt')]
    # 返回.txt文件的数量
    return len(txt_files)

def parse_commands(cmds_str):
    # A simple parser to extract command names and arguments from the command strings
    cmds = []
    default_params = {'buffer_id': None, 'batch_id': None, 'duration': None}  # Default parameters
    if cmds_str.startswith('[') and cmds_str.endswith(']'):
        cmds_str = cmds_str[1:-1].strip()
        if cmds_str:
            cmd_parts = cmds_str.split('), ')
            for part in cmd_parts:
                if ')' not in part:
                    part += ')'
                cmd_name, arg_str = part.split('(', 1)
                arg_str = arg_str[:-1]
                kwargs = default_params.copy()
                if arg_str:
                    for arg in arg_str.split(', '):
                        key, value = arg.split('=')
                        if value.isdigit():
                            value = int(value)
                        kwargs[key] = value
                cmds.append((cmd_name, kwargs))
    return cmds

def process_mg_files(my_filepath, rank_instances_dict, running_mode, torch_graph_stage_op_dict, is_trace, mpu, num_files):
    """ 
        处理 schedules or muti-gpus trace log
        schedules log: 生成的scheduling plan
        muti-gpus trace log: global ranks trace

        MODE_PROFILE 模式下且开启TP时, extract_sub_operations实例化所有subops,赋值属性并按序添加(op先被添加)
        MODE_SIMULATE 模式下, 读取的是scheduling plan, 因此只有TP=1时可以直接根据single gpu的profile log获取op的duration; 当TP>1时, 从init_3d_..函数处逐一获取op的duration并实例化subop
        
        is_trace 是用于判定当前是否为读取muti-gpus trace,生成trace_dict过程
    """
    stages_dict = {}
    # Read each file in the folder
    for filename in os.listdir(my_filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(my_filepath, filename), 'r') as file:
                lines = file.readlines()
                # Process each line in the file
                # record_cmd_times_dict = {}
                for line in lines:
                    line = line.strip()
                    if line:
                        # stage:0:warmup:forward_step:0
                        # wrank:0:warmup:forward_step:0
                        # stage:0:forward_step(batch_id=0, mg_state=xxx, duration=xxx, description=xx, group_kind=xx)
                        stage_rank_signal, stage_or_wrank_id, cmd = line.split(':', 2)
                        stage_or_wrank_id = int(stage_or_wrank_id)

                        # 为stage标记是生成的scheduling plan，为rank标记是trace文件
                        if stage_rank_signal == "stage" and not is_trace and mpu.world_size != num_files:
                            stage_id = stage_or_wrank_id
                            wrank_id = None
                            rank = None
                        elif stage_rank_signal == "rank" and is_trace or (stage_rank_signal == "stage" and not is_trace and mpu.world_size == num_files):
                            wrank_id = stage_or_wrank_id
                            rank = rank_instances_dict[wrank_id]
                            stage_id = rank_instances_dict[wrank_id]._get_pp_local_rank()
                        else:
                            raise ValueError(f"Error: signal is invalid, please check the trace file...")
                        
                        if stage_or_wrank_id not in stages_dict:
                            stages_dict[stage_or_wrank_id] = Stage(
                                wrank_id=wrank_id,
                                stage_id=stage_id,
                                rank=rank,
                                framework="megatron-lm"
                            )

                        cmd_name, kwargs = parse_megatron_cmd(cmd)
                        op_semantics = kwargs.get('op_semantics', None)
                        cmd_uid = kwargs.get('cmd_uid', kwargs.get('comm_uid', None))
                        operation_name = cmd_name
                        if cmd_name == 'ddp_grad_comm':
                            comm_func = kwargs.get('comm_func', None)
                            if comm_func == 'allreduce':
                                operation_name = 'dp_allreduce'
                            elif comm_func == 'reduce_scatter':
                                operation_name = 'dp_reducescatter'
                            else:
                                raise ValueError(f"Unsupported DDP overlap comm_func: {comm_func}")

                        batch_id = kwargs.get('batch_id', kwargs.get('trigger_batch_id', 0))
                        batch_id = int(batch_id) if batch_id is not None else None
                        mg_state = kwargs.get('mg_state', None)
                        group_kind = kwargs.get('group_kind', None)
                        description = kwargs.get('description', None)
                        if description is not None:
                            description = re.sub(r'\W+', '', description)

                        if cmd_name == 'ddp_grad_comm':
                            op_kind = 'comm'
                        elif operation_name == 'dp_allreduce' and op_semantics in {'wait_flush_only', 'metadata_placeholder'}:
                            op_kind = 'comp'
                        else:
                            op_kind = "comp" if operation_name in MG_COMP_OPERATION else "comm"
                        # if mpu.tp_size == 1:
                        # if cmd_name in ['ep_allreduce']:
                        #     op_kind = "comp"

                        end_timestamp = None
                        duration = None
                        sub_ops_list = None
                        if running_mode == MODE_PROFILE:
                            duration = kwargs.get('duration', None)
                            timestamp = float(kwargs.get('timestamp', 0))
                            end_timestamp = timestamp

                            if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                                if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                                            (mpu.dp_size > 1 and cmd_name in ['loss_func']) or \
                                                (mpu.exp_size > 1 and cmd_name in ['forward_step', 'backward_step']):
                                    start_time = timestamp - float(duration)
                                    sub_operations_str = kwargs.get('sub_operations', '[]')
                                    sub_ops_list = extract_sub_operations(sub_operations_str_list=sub_operations_str,pt_start_time=start_time,\
                                                                        pt_end_time=timestamp,pt_duration=duration,wrank_id=wrank_id,is_muti_gpus_trace=True,\
                                                                        )
                            # if mpu.tp_size > 1:
                            #     # 只有TP情况下muti-gpus trace文件中包含sub_operations记录,否则读取的是scheduling plan
                            #     start_time = timestamp - float(duration)
                            #     sub_operations_str = kwargs.get('sub_operations', '[]')
                            #     sub_ops_list = extract_sub_operations(sub_operations_str_list=sub_operations_str,pt_start_time=start_time,\
                            #                                         pt_end_time=timestamp,pt_duration=duration,wrank_id=wrank_id,is_muti_gpus_trace=True,\
                            #                                         )

                        elif running_mode == MODE_SIMULATE:
                            if is_trace:
                                duration = kwargs.get('duration', None)
                                timestamp_val = kwargs.get('timestamp', 0)
                                timestamp = float(timestamp_val) if timestamp_val is not None else 0.0
                                end_timestamp = timestamp

                                if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                                    if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                                                (mpu.dp_size > 1 and cmd_name in ['loss_func']) or \
                                                    (mpu.exp_size > 1 and cmd_name in ['forward_step', 'backward_step']):
                                        start_time = timestamp - float(duration)
                                        sub_operations_str = kwargs.get('sub_operations', '[]')
                                        # print(f"sub_operations_str:{sub_operations_str}")
                                        sub_ops_list = extract_sub_operations(sub_operations_str_list=sub_operations_str,pt_start_time=start_time,\
                                                                            pt_end_time=timestamp,pt_duration=duration,wrank_id=wrank_id,is_muti_gpus_trace=True,\
                                                                            )

                                # if mpu.tp_size > 1:
                                #     start_time = timestamp - float(duration)
                                #     sub_operations_str = kwargs.get('sub_operations', '[]')
                                #     sub_ops_list = extract_sub_operations(sub_operations_str_list=sub_operations_str,pt_start_time=start_time,\
                                #                                         pt_end_time=timestamp,pt_duration=duration,wrank_id=wrank_id,is_muti_gpus_trace=True,\
                                #                                         )
                            else:
                                # simulate模式下预测comm duration(处理的是生成的schedule plan)
                                if cmd_name in P2P_COMM_COLLECTIVE:
                                    # 计算P2P的耗时
                                    tensor_shape = kwargs.get('input__shape', None)
                                    tenosr_dtype = kwargs.get('input__dtype', None)
                                    # comm_group = rank_instances_dict[stage_or_wrank_id].pp_groups
                                    # 临时采用一个pp组替代（因为此时无法确认wrank_id）
                                    comm_group = mpu.pp_groups[0]
                                    duration = get_comm_op_exc_time(
                                        comm_group=comm_group,
                                        data_size=get_tensor_data_size(tensor_shape,tenosr_dtype),
                                        comm_func="send_recv"
                                    )
                                elif cmd_name in ALLREDUCE_COMM_COLLECTIVE:
                                    # 计算ALLREDUCE的耗时 - 从schedule plan中提取tensor信息
                                    tensor_shape = kwargs.get('input__shape', None)
                                    tenosr_dtype = kwargs.get('input__dtype', None)
                                    # 注意：这里的duration将在后续的_calculate_comm_duration中被CC-predictor重新计算
                                    # 这里只是为了确保Operation对象有基本的duration值
                                    normalized_cmd_name = str(cmd_name).strip().lower()

                                    def _infer_comm_group_size_for_allreduce() -> int:
                                        if normalized_cmd_name in {"dp_allreduce", "reducegrads", "reducetiedgrads"}:
                                            size = int(getattr(mpu, "dp_size", 0) or 0)
                                        elif normalized_cmd_name == "tp_allreduce":
                                            size = int(getattr(mpu, "tp_size", 0) or 0)
                                        elif normalized_cmd_name == "ep_allreduce":
                                            size = int(getattr(mpu, "ep_size", 0) or 0)
                                        elif normalized_cmd_name in {"exp_dp_allreduce", "ep_dp_allreduce"}:
                                            dp_modulo_exp_groups = getattr(mpu, "dp_modulo_exp_groups", None)
                                            if not dp_modulo_exp_groups:
                                                raise ValueError(
                                                    "dp_modulo_exp_groups is required for exp_dp/ep_dp allreduce size estimation."
                                                )

                                            if isinstance(dp_modulo_exp_groups, list) and dp_modulo_exp_groups:
                                                first_entry = dp_modulo_exp_groups[0]
                                                if isinstance(first_entry, list):
                                                    size = len(first_entry)
                                                elif isinstance(first_entry, int):
                                                    size = len(dp_modulo_exp_groups)
                                                else:
                                                    raise ValueError(
                                                        "Unsupported dp_modulo_exp_groups format: "
                                                        f"{type(first_entry)}"
                                                    )
                                            else:
                                                raise ValueError(
                                                    "dp_modulo_exp_groups must be a non-empty list."
                                                )
                                        else:
                                            raise ValueError(
                                                f"Unsupported allreduce cmd_name in schedule estimation: {cmd_name}"
                                            )

                                        if size <= 0:
                                            raise ValueError(
                                                f"Invalid comm_group_size={size} inferred for operation {cmd_name}"
                                            )
                                        return size

                                    comm_group_size = None
                                    if tensor_shape is not None and tenosr_dtype is not None:
                                        comm_group_size = _infer_comm_group_size_for_allreduce()

                                        data_size = get_tensor_data_size(tensor_shape, tenosr_dtype)
                                        # 使用简单的带宽估算作为初始值
                                        duration = data_size / (10 * 1024 * 1024 * 1024)  # 假设10GB/s带宽
                                        duration = max(duration * 1000, 0.1)  # 转换为ms，最小0.1ms
                                    else:
                                        duration = 1.0  # 默认1ms
                                    print(
                                        f"cmd_name:{cmd_name}, tensor_shape:{tensor_shape}, tenosr_dtype:{tenosr_dtype}, "
                                        f"comm_group_size:{comm_group_size if comm_group_size is not None else 'N/A'}, duration:{duration}"
                                    )
                                    # raise 0

                        elif running_mode == MODE_MODEL:
                            duration = 1
                        else:
                            raise ValueError(f"Error: running_mode is invalid, please check the trace file...")
                        
                        duration = round(float(duration), 2) if duration is not None else None

                        # CRITICAL FIX: Extract tensor information for communication operations (Path 1)
                        tensor_shape = kwargs.get('input__shape', None)
                        tensor_dtype = kwargs.get('input__dtype', None)

                        # CRITICAL FIX: Tensor metadata fallback mechanism for SIMULATE MODE (Path 1)
                        if (running_mode == MODE_SIMULATE and not is_trace and
                            cmd_name in ['dp_allreduce', 'exp_dp_allreduce'] and
                            (tensor_shape is None or tensor_dtype is None) and
                            torch_graph_stage_op_dict):

                            # Map operation name for consistency
                            mapped_cmd_name = cmd_name
                            if cmd_name == "ep_dp_allreduce":
                                mapped_cmd_name = "exp_dp_allreduce"

                            # Try to get tensor info from database_profile
                            db_op_data = get_batch_specific_profile_op_data(
                                torch_graph_stage_op_dict,
                                stage_id,
                                mapped_cmd_name,
                                batch_id=batch_id,
                            )
                            if isinstance(db_op_data, dict):
                                print(f"DEBUG PATH1: Attempting tensor metadata fallback for {cmd_name} stage_id={stage_id}, batch_id={batch_id}")

                                # Extract tensor metadata from database_profile
                                if 'tensor_shape' in db_op_data and 'tensor_dtype' in db_op_data:
                                    fallback_shape = db_op_data['tensor_shape']
                                    fallback_dtype = db_op_data['tensor_dtype']

                                    if tensor_shape is None and fallback_shape is not None:
                                        tensor_shape = fallback_shape
                                        print(f"DEBUG PATH1: ✅ Fallback tensor_shape: {tensor_shape}")

                                    if tensor_dtype is None and fallback_dtype is not None:
                                        tensor_dtype = fallback_dtype
                                        print(f"DEBUG PATH1: ✅ Fallback tensor_dtype: {tensor_dtype}")

                                    # CRITICAL FIX: Reset duration after successful metadata fallback
                                    # This prevents double-counting: simple bandwidth estimation + CC-estimator
                                    if tensor_shape is not None and tensor_dtype is not None:
                                        duration = None
                                        print(f"DEBUG PATH1: ✅ Duration reset to None after successful metadata fallback for {cmd_name}")
                                        print(f"DEBUG PATH1: CC-estimator will provide sole authoritative duration prediction")

                        if cmd_name == 'ddp_grad_comm':
                            bucket_numel = kwargs.get('bucket_numel_unpadded', kwargs.get('bucket_numel', None))
                            if bucket_numel is None:
                                raise ValueError('ddp_grad_comm is missing bucket_numel/bucket_numel_unpadded')
                            tensor_shape = [int(bucket_numel)]
                            tensor_dtype = kwargs.get('grad_dtype', tensor_dtype)
                            if tensor_dtype is None:
                                raise ValueError('ddp_grad_comm is missing grad_dtype metadata')

                        # DEBUG: Log tensor information and duration for Path 1
                        if cmd_name in ['dp_allreduce', 'exp_dp_allreduce', 'ddp_grad_comm']:
                            print(f"DEBUG PATH1: Creating Operation {operation_name} wrank_id={wrank_id}")
                            print(f"  Final tensor_shape: {tensor_shape}")
                            print(f"  Final tensor_dtype: {tensor_dtype}")
                            print(f"  Initial duration: {duration} ms")
                            print(f"  Duration source: {'database_profile' if duration is not None else 'None/fallback'}")

                        trace_metadata = dict(kwargs)
                        trace_metadata['trace_event_type'] = cmd_name
                        name_with_id = None
                        if cmd_name == 'ddp_grad_comm':
                            iter_id = kwargs.get('iter', None)
                            buffer_id = kwargs.get('buffer_id', None)
                            bucket_id = kwargs.get('bucket_id', None)
                            if iter_id is None or buffer_id is None or bucket_id is None:
                                raise ValueError('ddp_grad_comm is missing iter/buffer_id/bucket_id')
                            name_with_id = f"{operation_name}_iter{iter_id}_buffer{buffer_id}_bucket{bucket_id}"

                        # cmd_op_duration = 0.1 if (mpu.tp_size > 1 and running_mode == MODE_PROFILE and cmd_name in GLOBAL_CMD_NEED_BREAK_LIST) else duration

                        operation = Operation(
                            name=operation_name,
                            duration=duration,
                            batch_id=batch_id,
                            wrank_id=wrank_id,
                            stage_id=stage_id,
                            mg_state=mg_state,
                            op_kind=op_kind,
                            group_kind=group_kind,
                            description=description,
                            end_timestamp=end_timestamp,
                            hidden_duration=duration,
                            tensor_shape=tensor_shape,
                            tensor_dtype=tensor_dtype,
                            cmd_uid=cmd_uid,
                            op_semantics=op_semantics,
                            trace_metadata=trace_metadata,
                            name_with_id=name_with_id,
                        )

                        # muti-gpus trace文件中包含sub_operations记录，直接在这里添加
                        # TODO：当前只分解了fwd和bwd，get_batch虽然在上方代码得到subop_list，但是没有加入到最终list中
                        sub_operation_list = []
                        # if (running_mode == MODE_PROFILE or running_mode == MODE_SIMULATE) and mpu.tp_size > 1 and cmd_name in GLOBAL_CMD_NEED_BREAK_LIST :
                        #     sub_operation_list = sub_ops_list
                        #     # 直接添加subop的属性
                        #     sub_operation_list = add_sub_op_prop(operation, sub_ops_list)

                        
                        # if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                        #     if mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step'] and \
                        #         (running_mode == MODE_PROFILE or (running_mode == MODE_SIMULATE and is_trace)):
                        #         sub_operation_list = sub_ops_list
                        #         sub_operation_list = add_sub_op_prop(operation, sub_ops_list)
                        #     elif mpu.dp_size > 1 and cmd_name in ['loss_func'] and \
                        #         (running_mode == MODE_PROFILE or (running_mode == MODE_SIMULATE and is_trace)):
                        #         sub_operation_list = sub_ops_list
                        #         sub_operation_list = add_sub_op_prop(operation, sub_ops_list)

                        # 需要拆分的2种情况，tp/dp. MODE_SIMULATE模式下只有trace需要拆分
                        if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                            if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                            (mpu.dp_size > 1 and cmd_name in ['loss_func']) or \
                                (mpu.exp_size > 1 and cmd_name in ['forward_step', 'backward_step']):
                                if (running_mode == MODE_PROFILE or (running_mode == MODE_SIMULATE and is_trace)) and sub_ops_list is not None:
                                    sub_operation_list = add_sub_op_prop(operation, sub_ops_list)
                                    # 当OP需要被分解时重置duration为0.01
                                    operation.duration = 0.01

                        # if mpu.tp_size > 1 and cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                        #     if running_mode == MODE_PROFILE or (running_mode == MODE_SIMULATE and is_trace):
                        #         sub_operation_list = sub_ops_list
                        #         # 直接添加subop的属性
                        #         sub_operation_list = add_sub_op_prop(operation, sub_ops_list)


                        stages_dict[stage_or_wrank_id].add_operations_to_list(operation)
                        stages_dict[stage_or_wrank_id].add_op_list_to_list(sub_operation_list) if sub_operation_list else 0

    return stages_dict


def add_sub_op_prop(operation, sub_ops_or_list):
    if isinstance(sub_ops_or_list, list):
        for sub_op in sub_ops_or_list:
            sub_op.stage_id = operation.stage_id
            sub_op.mg_state = operation.mg_state
            sub_op.batch_id = operation.batch_id

    elif isinstance(sub_ops_or_list, SubOperation):
            sub_ops_or_list.stage_id = operation.stage_id
            sub_ops_or_list.mg_state = operation.mg_state
            sub_ops_or_list.batch_id = operation.batch_id

    else:
        raise ValueError(f"Error: sub_ops_or_list is invalid, please check the trace file...")
    
    return sub_ops_or_list

def break_cmd_to_sub_ops(pp_size, duration, cmd_name, batch_id, wrank_id, stage_id, mg_state, group_kind, description):
    sub_operations = []
    short_name = "fwd" if cmd_name == 'forward_step' else "bwd"
    
    # 分解操作并创建SubOperation实例
    # FWD过程的TP分解
    if short_name == "fwd":
        # first stage:embedding (input layer)
        if stage_id == 0:
            sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="input_embedding"))
            sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="input_embedding"))
            sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="input_embedding"))

        # all stages: self-attention
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="self-attention"))
        sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="self-attention"))
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="self-attention"))

        # all stages: MLP
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="mlp"))
        sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="mlp"))
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="mlp"))

        # last stage: cross-entropy
        if stage_id == pp_size-1:
            sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="cross-entropy"))
            sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="cross-entropy"))
            sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="cross-entropy"))
            
    elif short_name == "bwd":
        # 反向传播的TP分解，注意次序是倒着的
        # last stage: embedding (output layer)
        if stage_id == pp_size-1:
            sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="output_embedding"))
            sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="output_embedding"))
            sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="output_embedding"))

        # all stages: MLP
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="mlp"))
        sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="mlp"))
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="mlp"))
        
        # all stages: self-attention
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="self-attention"))
        sub_operations.append(SubOperation(name=f'tp_allreduce', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind="tp", description="self-attention"))
        sub_operations.append(SubOperation(name=f'{short_name}_comp', op_kind="comp", duration=1, batch_id=batch_id, wrank_id=wrank_id, stage_id=stage_id, mg_state=mg_state, group_kind=group_kind, description="self-attention"))

    else:
        raise ValueError(f"Invalid cmd_name: {cmd_name}")

    return sub_operations

def v1_process_mg_files(my_filepath, rank_instances_dict, running_mode, torch_graph_stage_op_dict, is_trace, mpu, num_files):
    stages_dict = {}
    # Read each file in the folder
    for filename in os.listdir(my_filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(my_filepath, filename), 'r') as file:
                lines = file.readlines()
                # Process each line in the file
                for line in lines:
                    line = line.strip()
                    if line:
                        # stage:0:warmup:forward_step:0
                        # wrank:0:warmup:forward_step:0
                        # TODO：group需要吗？
                        # stage:0:forward_step(batch_id=0, mg_state=xxx, duration=xxx, description=xx, group_kind=xx)
                        stage_rank_signal, stage_or_wrank_id, cmd = line.split(':')
                        stage_or_wrank_id = int(stage_or_wrank_id)

                        # 为stage标记是生成的scheduling plan，为rank标记是trace文件
                        if stage_rank_signal == "stage" and not is_trace and mpu.world_size != num_files:
                            stage_id = stage_or_wrank_id
                            wrank_id = None
                            rank = None
                        elif stage_rank_signal == "rank" and is_trace or (stage_rank_signal == "stage" and not is_trace and mpu.world_size == num_files):
                            wrank_id = stage_or_wrank_id
                            rank = rank_instances_dict[wrank_id]
                            stage_id = rank_instances_dict[wrank_id]._get_pp_local_rank()
                        else:
                            raise ValueError(f"Error: signal is invalid, please check the trace file...")
                        
                        if stage_or_wrank_id not in stages_dict:
                            stages_dict[stage_or_wrank_id] = Stage(
                                wrank_id=wrank_id,
                                stage_id=stage_id,
                                rank=rank,
                                framework="megatron-lm"
                            )

                        cmd_name, kwargs = parse_megatron_cmd(cmd)
                        batch_id = int(kwargs.get('batch_id', 0))
                        mg_state = kwargs.get('mg_state', None)
                        group_kind = kwargs.get('group_kind', None)
                        description = kwargs.get('description', None)

                        op_kind = "comp" if cmd_name in MG_COMP_OPERATION else "comm" 
                        if description is not None:
                            description = re.sub(r'\W+', '', description)

                        # if mpu.tp_size == 1:
                        if cmd_name in ['dp_allreduce', 'exp_dp_allreduce']:
                            op_kind = "comp"

                        if running_mode == MODE_PROFILE:
                            duration = kwargs.get('duration', None)
                        elif running_mode == MODE_SIMULATE:
                            if is_trace:
                                duration = kwargs.get('duration', None)
                            else:
                                if torch_graph_stage_op_dict:
                                    duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id, batch_id=batch_id)
                                else:
                                    raise ValueError(f"Error: torch_graph_stage_op_dict is None, please check the trace file...")
                        elif running_mode == MODE_MODEL: 
                            duration = 1
                        duration = round(float(duration), 2) if duration is not None else None

                        # Extract tensor information for communication operations
                        tensor_shape = kwargs.get('input__shape', None)
                        tensor_dtype = kwargs.get('input__dtype', None)

                        # CRITICAL FIX: Tensor metadata fallback mechanism for SIMULATE MODE
                        # When schedule/ lacks tensor info for dp_allreduce/exp_dp_allreduce,
                        # fallback to database_profile/ data
                        if (running_mode == MODE_SIMULATE and not is_trace and
                            cmd_name in ['dp_allreduce', 'exp_dp_allreduce'] and
                            (tensor_shape is None or tensor_dtype is None) and
                            torch_graph_stage_op_dict):

                            # Map operation name for consistency (ep_dp_allreduce -> exp_dp_allreduce)
                            mapped_cmd_name = cmd_name
                            if cmd_name == "ep_dp_allreduce":
                                mapped_cmd_name = "exp_dp_allreduce"

                            # Try to get tensor info from database_profile
                            db_op_data = get_batch_specific_profile_op_data(
                                torch_graph_stage_op_dict,
                                stage_id,
                                mapped_cmd_name,
                                batch_id=batch_id,
                            )
                            if isinstance(db_op_data, dict):
                                print(f"DEBUG: Attempting tensor metadata fallback for {cmd_name} stage_id={stage_id}, batch_id={batch_id}")
                                print(f"DEBUG: Database entry keys: {list(db_op_data.keys())}")

                                # Extract tensor metadata from database_profile
                                if 'tensor_shape' in db_op_data and 'tensor_dtype' in db_op_data:
                                    fallback_shape = db_op_data['tensor_shape']
                                    fallback_dtype = db_op_data['tensor_dtype']

                                    if tensor_shape is None and fallback_shape is not None:
                                        tensor_shape = fallback_shape
                                        print(f"DEBUG PATH2: ✅ Fallback tensor_shape: {tensor_shape}")

                                    if tensor_dtype is None and fallback_dtype is not None:
                                        tensor_dtype = fallback_dtype
                                        print(f"DEBUG PATH2: ✅ Fallback tensor_dtype: {tensor_dtype}")

                                    # CRITICAL FIX: Reset duration after successful metadata fallback
                                    # This prevents double-counting: simple bandwidth estimation + CC-estimator
                                    if tensor_shape is not None and tensor_dtype is not None:
                                        duration = None
                                        print(f"DEBUG PATH2: ✅ Duration reset to None after successful metadata fallback for {cmd_name}")
                                        print(f"DEBUG PATH2: CC-estimator will provide sole authoritative duration prediction")
                                else:
                                    print(f"DEBUG PATH2: ❌ No tensor metadata found in database entry")
                            else:
                                print(f"DEBUG PATH2: ❌ No database entry found for {mapped_cmd_name} stage_id={stage_id}, batch_id={batch_id}")

                        # DEBUG: Log tensor information extraction
                        if cmd_name in ['dp_allreduce', 'exp_dp_allreduce']:
                            print(f"DEBUG Operation creation: {cmd_name} wrank_id={wrank_id} stage_id={stage_id}")
                            print(f"  running_mode={running_mode}, is_trace={is_trace}")
                            print(f"  kwargs keys: {list(kwargs.keys())}")
                            print(f"  tensor_shape from kwargs: {tensor_shape}")
                            print(f"  tensor_dtype from kwargs: {tensor_dtype}")
                            print(f"  torch_graph_stage_op_dict is None: {torch_graph_stage_op_dict is None}")
                            if torch_graph_stage_op_dict:
                                print(f"  torch_graph_stage_op_dict keys: {list(torch_graph_stage_op_dict.keys())}")
                                if stage_id in torch_graph_stage_op_dict:
                                    print(f"  stage_id {stage_id} operations: {list(torch_graph_stage_op_dict[stage_id].keys())}")
                                else:
                                    print(f"  stage_id {stage_id} NOT found in torch_graph_stage_op_dict")

                        # FINAL DEBUG: Check tensor info and duration before Operation creation
                        if cmd_name in ['dp_allreduce', 'exp_dp_allreduce']:
                            print(f"DEBUG PATH2: Creating Operation {cmd_name} wrank_id={wrank_id}")
                            print(f"  Final tensor_shape: {tensor_shape}")
                            print(f"  Final tensor_dtype: {tensor_dtype}")
                            print(f"  Initial duration: {duration} ms")
                            print(f"  Duration source: {'database_profile' if duration is not None else 'None/fallback'}")
                            print(f"  running_mode: {running_mode}")
                            print(f"  is_trace: {is_trace}")
                            if duration is not None:
                                print(f"  Duration from kwargs: {kwargs.get('duration', 'NOT_FOUND')}")
                                if torch_graph_stage_op_dict and stage_id in torch_graph_stage_op_dict:
                                    print(f"  torch_graph_stage_op_dict[{stage_id}] keys: {list(torch_graph_stage_op_dict[stage_id].keys())}")
                                    batch_specific_op_data = get_batch_specific_profile_op_data(
                                        torch_graph_stage_op_dict,
                                        stage_id,
                                        cmd_name,
                                        batch_id=batch_id,
                                    )
                                    if batch_specific_op_data is not None:
                                        print(f"  batch-aware torch_graph entry: {batch_specific_op_data}")
                                else:
                                    print(f"  torch_graph_stage_op_dict not available for stage_id {stage_id}")

                        operation = Operation(
                            name=cmd_name,
                            duration=duration,
                            batch_id=batch_id,
                            wrank_id=wrank_id,
                            stage_id=stage_id,
                            mg_state=mg_state,
                            op_kind=op_kind,
                            group_kind=group_kind,
                            description=description,
                            tensor_shape=tensor_shape,
                            tensor_dtype=tensor_dtype
                        )

                        # DEBUG: Verify Operation object attributes
                        if cmd_name in ['dp_allreduce', 'exp_dp_allreduce']:
                            print(f"  Created Operation.tensor_shape: {operation.tensor_shape}")
                            print(f"  Created Operation.tensor_dtype: {operation.tensor_dtype}")
                        # print(operation)
                        stages_dict[stage_or_wrank_id].add_operations_to_list(operation)

    return stages_dict

def process_ds_files(my_filepath, rank_instances_dict, running_mode, torch_graph_stage_op_dict, is_trace, mpu, num_files):
    stages_dict = {}
    
    for filename in os.listdir(my_filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(my_filepath, filename), 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        stage_rank_signal = line.split(":", 1)[0]
                        stage_step_part, cmds_str = line.split('_cmds:')
                        wrank_id_str, step_id_str = stage_step_part.split('_step_id:')
                        stage_or_wrank_id = int(wrank_id_str.split(':')[-1])
                        step_id = int(step_id_str.split(':')[-1])

                        print(f"stage_rank_signal:{stage_rank_signal}, is_trace:{is_trace}, num_files:{num_files}, mpu.world_size:{mpu.world_size}")
                        
                        if stage_rank_signal == "stage" and not is_trace and mpu.world_size != num_files:
                            stage_id = stage_or_wrank_id
                            wrank_id = None
                            rank = None
                        elif (stage_rank_signal == "rank" and is_trace) or (stage_rank_signal == "stage" and not is_trace and mpu.world_size == num_files):
                            wrank_id = stage_or_wrank_id
                            rank = rank_instances_dict[wrank_id]
                            stage_id = rank_instances_dict[wrank_id]._get_pp_local_rank()
                        else:
                            raise ValueError(f"Error: signal is invalid, please check the trace file...")
                        
                        if stage_or_wrank_id not in stages_dict:
                            stages_dict[stage_or_wrank_id] = Stage(
                                wrank_id=wrank_id,
                                stage_id=stage_id,
                                rank=rank,
                                steps_num=0,
                                framework="deepspeed"
                            )

                        cmds = parse_commands(cmds_str.strip())
                        for cmd_name, kwargs in cmds:
                            op_kind = "comp" if cmd_name in DS_COMP_OPERATION else "comm"
                            batch_id = kwargs.get('batch_id', None)
                            description = kwargs.get('description', None)
                            if description is not None:
                                description = re.sub(r'\W+', '', description)

                            # Note: dp情况下算作comp好像不影响逻辑？因为dp组里头的rank运行情况其实是完全一致的？
                            # 但是当前为tp的allreduce的时候要注意，tp group的rank的运行情况不一致（算子组成不同）
                            if cmd_name in ['ReduceGrads', 'ReduceTiedGrads']:
                                op_kind = "comp"
                
                            if running_mode == MODE_PROFILE:
                                duration = kwargs.get('duration', None)
                            elif running_mode == MODE_SIMULATE:
                                if is_trace:
                                    duration = kwargs.get('duration', None)
                                else:
                                    duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id, "deepspeed", batch_id=batch_id)
                            elif running_mode == MODE_MODEL: 
                                duration = 1
                            duration = round(float(duration), 2) if duration is not None else None

                            operation = Operation(
                                name=cmd_name,
                                duration=duration,
                                buffer_id=kwargs.get('buffer_id', None),
                                batch_id=batch_id,
                                step_id=step_id,
                                wrank_id=wrank_id,
                                stage_id=stage_id,
                                op_kind=op_kind,
                                description=description,
                            )
                            # print(operation)
                            stages_dict[stage_or_wrank_id].add_operations_to_list(operation)

                    stages_dict[stage_or_wrank_id].steps_num = max(stages_dict[stage_or_wrank_id].steps_num, step_id + 1)

    return stages_dict

# def parse_megatron_cmd(cmd):
#     # Parse the command part of the line
#     cmd_name, params = cmd.split('(', 1)
#     params = params.rstrip(')')
#     kwargs = {}

#     # Handle sub_operations separately
#     param_list = params.split(',')
#     param_count = len(param_list)
#     index = 0

#     while index < param_count:
#         param = param_list[index]
#         if 'sub_operations' in param:
#             # Join the rest of the list as the value for sub_operations
#             sub_operations_value = ','.join(param_list[index:])
#             key, value = sub_operations_value.split('=', 1)
#             kwargs[key.strip()] = eval(value.strip())  # Convert the string representation of the list to a Python list
#             break
#         else:
#             if '=' in param:
#                 key, value = param.split('=', 1)
#                 kwargs[key.strip()] = value.strip()
#             index += 1
                
#     return cmd_name.strip(), kwargs


def _is_top_level_trace_field_boundary(params: str, comma_index: int) -> bool:
    cursor = comma_index + 1
    while cursor < len(params) and params[cursor].isspace():
        cursor += 1
    if cursor >= len(params):
        return False
    if not (params[cursor].isalpha() or params[cursor] == "_"):
        return False

    cursor += 1
    while cursor < len(params) and (params[cursor].isalnum() or params[cursor] == "_"):
        cursor += 1
    while cursor < len(params) and params[cursor].isspace():
        cursor += 1
    return cursor < len(params) and params[cursor] == "="


def _split_top_level_trace_fields(params: str):
    fields = []
    index = 0
    length = len(params)

    while index < length:
        while index < length and params[index] in {",", " ", "\t", "\n"}:
            index += 1
        if index >= length:
            break

        key_start = index
        while index < length and params[index] != "=":
            index += 1
        if index >= length:
            raise ValueError(f"Malformed Megatron trace field: {params[key_start:]}")

        key = params[key_start:index].strip()
        if not key:
            raise ValueError(f"Empty Megatron trace field key in: {params}")

        index += 1
        value_start = index
        bracket_depth = 0
        brace_depth = 0
        paren_depth = 0
        quote_char = None

        while index < length:
            char = params[index]
            if quote_char is not None:
                if char == "\\":
                    index += 2
                    continue
                if char == quote_char:
                    quote_char = None
                index += 1
                continue

            if char in {"'", '"'}:
                quote_char = char
                index += 1
                continue

            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif (
                char == ","
                and bracket_depth == 0
                and brace_depth == 0
                and paren_depth == 0
                and _is_top_level_trace_field_boundary(params, index)
            ):
                break

            index += 1

        value = params[value_start:index].strip()
        fields.append((key, value))

        if index < length and params[index] == ",":
            index += 1

    return fields


def _parse_megatron_trace_value(raw_value: str):
    value = raw_value.strip()
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False

    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value):
        return float(value)

    if (
        (value.startswith("[") and value.endswith("]"))
        or (value.startswith("{") and value.endswith("}"))
        or (value.startswith("(") and value.endswith(")"))
        or (value.startswith("\"") and value.endswith("\""))
        or (value.startswith("'") and value.endswith("'"))
    ):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    return value


def parse_megatron_cmd(cmd):
    cmd_name, params = cmd.split("(", 1)
    params = params.rstrip(")")
    kwargs = {}

    for key, value in _split_top_level_trace_fields(params):
        kwargs[key] = _parse_megatron_trace_value(value)

    return cmd_name.strip(), kwargs



def process_trace_or_scheduling_files(my_filepath: str, mpu, rank_instances_dict: dict, running_mode, torch_graph_stage_op_dict, is_trace: bool, framework:str):

    assert my_filepath and mpu and rank_instances_dict, "Error: 初始化错误,请重新检查关键变量..."
    if running_mode == MODE_PROFILE:
        assert is_trace, "Error: MODE_PROFILE下只允许使用trace文件..."
    elif running_mode == MODE_SIMULATE:
        assert torch_graph_stage_op_dict and my_filepath, "Error: MODE_SIMULATE缺失必须文件..."

    stages_dict = {}
    # not_simulating_cmd_dict = {}

    num_files = get_num_txt_files(my_filepath)
    if num_files != mpu.world_size and num_files != mpu.pp_size:
        print(f"num_trace_files: {num_files}, mpu.world_size: {mpu.world_size}, mpu.pp_size: {mpu.pp_size}")
        raise ValueError(f"Error: mpu初始化设定与读取trace的GPU数量不一致, 请重新修改config...")
    
    if framework == "deepspeed":
        stages_dict = process_ds_files(my_filepath, rank_instances_dict, running_mode, torch_graph_stage_op_dict, is_trace, mpu, num_files)
    elif framework == "megatron-lm":
        stages_dict = process_mg_files(my_filepath, rank_instances_dict, running_mode, torch_graph_stage_op_dict, is_trace, mpu, num_files)
        # stages_dict = v1_process_mg_files(my_filepath, rank_instances_dict, running_mode, torch_graph_stage_op_dict, is_trace, mpu, num_files)
    else:
        raise ValueError(f"Error: framework is invalid, please check the trace file...")

    return stages_dict

def get_op_excution_time_from_trace(trace_stages_dict, operation):
    """ 当前的写法是将所有op都拆解开(每个Stage的operations_list) 因此直接遍历搜寻对应rank上的operations_list即可 """
    trace_stage = trace_stages_dict.get(operation.wrank_id)
    operation_duration = None

    if trace_stage is None:
        print(f"Warning: No trace stage found for wrank_id={operation.wrank_id}")
        return None

    # 添加调试信息
    if isinstance(operation, SubOperation) and any(moe_op in operation.name_with_id for moe_op in ['exp_allgather', 'exp_all_to_all', 'exp_dp_allreduce']):
        print(f"Debug: Searching for MoE operation: {operation.name_with_id} in wrank_id={operation.wrank_id}")
        available_moe_ops = [op.name_with_id for op in trace_stage.operations_list if isinstance(op, SubOperation) and any(moe in op.name_with_id for moe in ['exp_allgather', 'exp_all_to_all', 'exp_dp_allreduce'])]
        print(f"Debug: Available MoE operations in trace_stage: {available_moe_ops}")
        print(f"Debug: Total operations in trace_stage: {len(trace_stage.operations_list)}")

    # TODO：这里的查询方式可以优化？维护一个dict,key为OP的唯一ID
    for op in trace_stage.operations_list:
        if isinstance(operation, SubOperation):
            if isinstance(op, SubOperation):
                # 对于MoE SubOperation，使用放宽的匹配策略
                # 标准化name_with_id：去掉schedule侧添加的 _{batch_id}_{parent_op} 后缀
                base_name_with_id = operation.name_with_id
                is_moe_operation = any(moe_op in operation.name_with_id for moe_op in ['exp_allgather', 'exp_all_to_all', 'exp_dp_allreduce'])

                if is_moe_operation:
                    # 对于MoE操作，去掉可能的后缀 (例如: exp_allgather_0_0_forward_step -> exp_allgather_0)
                    if base_name_with_id.endswith('_forward_step') or base_name_with_id.endswith('_backward_step'):
                        base_name_with_id = base_name_with_id.rsplit('_', 2)[0]

                    # 使用放宽的匹配：支持多种匹配模式
                    # 1. 精确匹配 base_name_with_id
                    if op.name_with_id == base_name_with_id:
                        operation_duration = op.duration
                        print(f"Debug: Found exact MoE operation match: {op.name_with_id} -> {operation.name_with_id}")
                        break

                    # 2. 模糊匹配：基于操作类型匹配（忽略索引差异）
                    # 提取操作类型 (例如: exp_allgather_6 -> exp_allgather, exp_all_to_all_2 -> exp_all_to_all)
                    def extract_operation_type(name):
                        # 对于MoE操作，提取基础操作类型
                        if 'exp_allgather' in name:
                            return 'exp_allgather'
                        elif 'exp_all_to_all' in name:
                            return 'exp_all_to_all'
                        elif 'exp_dp_allreduce' in name:
                            return 'exp_dp_allreduce'
                        else:
                            return name.rsplit('_', 1)[0] if '_' in name else name

                    op_type = extract_operation_type(op.name_with_id)
                    schedule_type = extract_operation_type(base_name_with_id)

                    if op_type == schedule_type:
                        operation_duration = op.duration
                        print(f"Debug: Found type-based MoE operation match: {op.name_with_id} -> {operation.name_with_id} (type: {op_type})")
                        break
                else:
                    # 对于非MoE操作，保持严格匹配
                    if op.name_with_id == operation.name_with_id and op.batch_id == operation.batch_id:
                        operation_duration = op.duration
                        break
        else:
            if not isinstance(op, SubOperation) and op.name == operation.name and op.batch_id == operation.batch_id:
                operation_duration = op.duration
                break

    # # TODO:修正为network的接口
    # if "allreduce" in op.name:
    #     import random
    #     op_duration = float(op.duration) * random.uniform(0.9, 1.0)
    
    if operation_duration == None:
        print(f"operation = {operation}")
        if isinstance(operation, SubOperation):
            print(f"Warning: Could not find matching SubOperation in trace for wrank_id={operation.wrank_id}, name_with_id={operation.name_with_id}, batch_id={operation.batch_id}")

            # 对于MoE相关的通信操作，尝试使用网络估算器作为回退
            if any(moe_op in operation.name_with_id for moe_op in ['exp_allgather', 'exp_all_to_all', 'exp_dp_allreduce']):
                print(f"Info: Using network estimator for MoE operation: {operation.name_with_id}")
                # 使用默认的通信时间估算
                from src.core.comm_sim.nccl_comm import get_comm_op_exc_time

                # 尝试获取tensor shape和dtype信息
                tensor_shape = getattr(operation, 'tensor_shape', None)
                tensor_dtype = getattr(operation, 'tensor_dtype', None)

                # 如果没有tensor信息，尝试从operation的其他属性获取
                if tensor_shape is None:
                    tensor_shape = getattr(operation, 'input_shape', [1024, 1024])  # 默认shape
                if tensor_dtype is None:
                    tensor_dtype = getattr(operation, 'input_dtype', 'float16')  # 默认dtype

                # 估算通信时间
                try:
                    # 计算数据大小
                    if isinstance(tensor_shape, (list, tuple)) and len(tensor_shape) >= 2:
                        # 简单估算：shape[0] * shape[1] * dtype_size
                        dtype_size = 2 if 'float16' in str(tensor_dtype) else 4  # float16=2bytes, float32=4bytes
                        data_size = tensor_shape[0] * tensor_shape[1] * dtype_size
                    else:
                        data_size = 1024 * 1024 * 2  # 默认2MB数据

                    # 使用更真实的通信组进行估算
                    default_comm_group = [0, 1]  # 默认的2-rank组

                    # 尝试获取更真实的通信组信息
                    if hasattr(operation, 'wrank_id') and operation.wrank_id is not None:
                        wrank_id = operation.wrank_id
                        # 根据操作类型选择合适的通信组
                        if 'exp_allgather' in operation.name_with_id or 'exp_all_to_all' in operation.name_with_id:
                            # 对于expert parallel操作，使用expert group
                            default_comm_group = list(range(min(8, wrank_id + 2)))  # 简化的expert group
                        elif 'exp_dp_allreduce' in operation.name_with_id:
                            # 对于expert data parallel操作，使用data parallel group
                            default_comm_group = list(range(min(4, wrank_id + 2)))  # 简化的dp group

                    if 'allgather' in operation.name_with_id:
                        operation_duration = get_comm_op_exc_time(default_comm_group, data_size, "allgather")
                    elif 'all_to_all' in operation.name_with_id:
                        operation_duration = get_comm_op_exc_time(default_comm_group, data_size, "all_to_all")
                    elif 'allreduce' in operation.name_with_id:
                        operation_duration = get_comm_op_exc_time(default_comm_group, data_size, "allreduce")
                    else:
                        operation_duration = 0.001  # 1ms默认值

                    print(f"Info: Estimated duration for {operation.name_with_id}: {operation_duration}ms")
                except Exception as e:
                    print(f"Warning: Failed to estimate communication time: {e}")
                    operation_duration = 0.001  # 1ms默认值
            else:
                print(f"Error: Could not find matching SubOperation in trace for wrank_id={operation.wrank_id}, name_with_id={operation.name_with_id}, batch_id={operation.batch_id}")
                raise ValueError("Could not find operation in trace, see log for details.")
        else:
            print(f"Error: Could not find matching Operation in trace for wrank_id={operation.wrank_id}, name={operation.name}, batch_id={operation.batch_id}")
            raise ValueError("Could not find operation in trace, see log for details.")

    return operation_duration

def analysis_sub_ops_list_str(sub_operations_str):
    sub_comm_list = []
    for sub_op_str in sub_operations_str:
        # Initialize a dictionary to store key-value pairs for each sub-operation
        sub_op_dict = {}

        # Manually parse the sub-operation string to handle nested structures
        i = 0
        while i < len(sub_op_str):
            # Find the key
            key_start = i
            while i < len(sub_op_str) and sub_op_str[i] != '=':
                i += 1
            key = sub_op_str[key_start:i].strip()
            i += 1  # Skip the '='

            # Find the value
            if sub_op_str[i] == '[':
                # Value is a list, find the closing ']'
                value_start = i
                bracket_count = 1
                i += 1
                while i < len(sub_op_str) and bracket_count > 0:
                    if sub_op_str[i] == '[':
                        bracket_count += 1
                    elif sub_op_str[i] == ']':
                        bracket_count -= 1
                    i += 1
                value = sub_op_str[value_start:i].strip()
            else:
                # Value is a regular string, find the next comma or end of string
                value_start = i
                while i < len(sub_op_str) and sub_op_str[i] != ',':
                    i += 1
                value = sub_op_str[value_start:i].strip()

            # Store the key-value pair in the dictionary
            sub_op_dict[key] = value

            # Skip the comma and any whitespace
            while i < len(sub_op_str) and (sub_op_str[i] == ',' or sub_op_str[i].isspace()):
                i += 1

        sub_comm_list.append(sub_op_dict)

    return sub_comm_list

def extract_sub_operations(sub_operations_str_list, pt_start_time, pt_end_time, pt_duration, wrank_id, \
                           is_muti_gpus_trace, rank_instance:RankZoo=None)->list:
    '''
        非traces情况下,pt_duration记录的只是comp部分时间,sub_comm_op记录的是start的时间戳,由于来实例化不同的sub_comp_op
        在这种情况下,对comm_sub_op的duration额外进行, 不要修改current_XXX部分的迭代计算过程中的sub_op_duration
    
    '''
    # print(f"sub_operations_str: {sub_operations_str_list}")
    # print(f"pt_start_time: {pt_start_time}")
    # print(f"pt_end_time: {pt_end_time}")
    # print(f"pt_duration: {pt_duration}")
    # print(f"wrank_id: {wrank_id}")

    # Initialize the list to store sub-operation dictionaries
    if not sub_operations_str_list:
        return None
    sub_comm_list = analysis_sub_ops_list_str(sub_operations_str_list)
    # print(f"sub_comm_list:{sub_comm_list}")
    # print(f"type sub_operations_str_list:{type(sub_operations_str_list)}")

    all_sub_ops = []
    current_start_time = pt_start_time
    comp_index = 0
    last_comm_op_timestamp = None
    for index, each_comm_subop in enumerate(sub_comm_list):
        # print(f"each_comm_subop: {each_comm_subop}")
        sub_op_trace_src_func = each_comm_subop.get('trace_src_func', None)
        sub_op_func_name = each_comm_subop.get('func_name', None)
        sub_op_comm_func = each_comm_subop.get('comm_func', None)
        sub_op_group = each_comm_subop.get('group', None)
        sub_op_input_shape = each_comm_subop.get('input__shape', None)
        sub_op_dtype = each_comm_subop.get('input__dtype', None)
        # print(f"sub_op_dtype:{sub_op_dtype}")
        comm_duration_for_instance = 0
        assert sub_op_comm_func, f"Error: comm subop's {each_comm_subop} comm_func is None, please check the trace file..."

        # 处理记录的comm subop
        # 当是single gpu profile log时，comm duration应为0，只是记录调用API的时间戳(用于计算comp，隔离开了comm_duration_for_intance的使用)
        if is_muti_gpus_trace:
            sub_op_duration = float(each_comm_subop.get('duration', None))
            comm_duration_for_instance = round(sub_op_duration,2)
        else:
            sub_op_duration = 0

            if sub_op_group == "tp":
                comm_group = rank_instance.tp_groups
            elif sub_op_group == "dp":
                comm_group = rank_instance.dp_groups
            elif sub_op_group == "pp":
                comm_group = rank_instance.pp_groups
            elif sub_op_group == "exp":
                comm_group = rank_instance.exp_groups
            else:
                raise ValueError(f"Error: sub_op_group:{sub_op_group} is invalid")
            
            # print(f"sub_op_comm_func:{sub_op_comm_func}")
            comm_duration_for_instance = get_comm_op_exc_time(
                comm_group=comm_group,
                data_size=get_tensor_data_size(sub_op_input_shape,sub_op_dtype),
                comm_func=sub_op_comm_func
            )
        sub_op_timestamp = float(each_comm_subop.get('timestamp', None))

        current_finish_time = sub_op_timestamp - sub_op_duration
        current_duration = current_finish_time - current_start_time

        # Determine if there's a gap indicating a comp operation
        if current_duration > 0:
            comp_op = SubOperation(
                name='sub_comp',
                op_kind='comp',
                duration=round(current_duration,2),
                wrank_id=wrank_id,
                group_kind=None,
                description="Generated comp op",
                start_time=float(current_start_time),
                pt_start_time=float(pt_start_time),
                pt_duration=round(float(pt_duration),2),
                tensor_shape=None,
                tensor_dtype=None,
                trace_src_func=None,
                comm_func=None,
                name_with_id=f'sub_comp_{comp_index}'
            )
            all_sub_ops.append(comp_op)
            comp_index += 1

        comm_name = f'{sub_op_group}_{sub_op_comm_func}'
        # Create and add the comm operation
        comm_op = SubOperation(
            name=comm_name,
            op_kind='comm',
            duration=comm_duration_for_instance, #round(sub_op_duration,2)
            wrank_id=wrank_id,
            group_kind=sub_op_group,
            description=f"{sub_op_func_name}|shape={sub_op_input_shape}|dtype={sub_op_dtype}",
            start_time=float(current_finish_time),
            pt_start_time=float(pt_start_time),
            pt_duration=round(float(pt_duration),2),
            tensor_shape=sub_op_input_shape,
            tensor_dtype=sub_op_dtype,
            trace_src_func=sub_op_trace_src_func,
            comm_func=sub_op_comm_func,
            name_with_id=f'{comm_name}_{index}'
        )
        all_sub_ops.append(comm_op)

        current_start_time = sub_op_timestamp
        if index == len(sub_comm_list) - 1:
            last_comm_op_timestamp = sub_op_timestamp

    # Handle any remaining comp operation up to the end of the original operation
    # print(f"pt_end_time:{pt_end_time}, last_comm_op_timestamp:{last_comm_op_timestamp}")
    final_comp_duration = pt_end_time - last_comm_op_timestamp

    if final_comp_duration > 0:
        final_comp_op = SubOperation(
                name='sub_comp',
                op_kind='comp',
                duration=round(final_comp_duration,2),
                wrank_id=wrank_id,
                group_kind=None,
                description="Generated comp op",
                start_time=float(current_start_time),
                pt_start_time=float(pt_start_time),
                pt_duration=round(float(pt_duration),2),
                tensor_shape=None,
                tensor_dtype=None,
                trace_src_func=None,
                comm_func=None,
                name_with_id=f'sub_comp_{comp_index}'
        )
        all_sub_ops.append(final_comp_op)

    return all_sub_ops

def add_sub_ops_according_to_profile_dict(torch_graph_stage_op_dict, operation, rank_id, running_mode, trace_stages_dict):
    """ 
    cmds of pp-level schedule plan -> tp-level (依据single-gpu profile): 1. 存在部分OP需要拆分为subop, 2. 部分直接使用, 3. 部分不存在于single-gpu profile 
    
    dp_allreduce and ep_allreduce 的duration记录在在dict的duration
    subop的duration已经写入SubOperation的duration中
    """
    part_operations_list = []
    sub_ops_list_copy = []
    profile_op_data = get_batch_specific_profile_op_data(
        torch_graph_stage_op_dict,
        rank_id,
        operation.name,
        batch_id=getattr(operation, 'batch_id', None),
    )

    ''' process1: single-gpu profiler 完善OP/SUBOP duration '''
    if operation.name in GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST:
        if (
            isinstance(profile_op_data, dict)
            and profile_op_data.get('sub_ops_list')
        ):
            operation_duration = profile_op_data['duration']
            operation.hidden_duration = operation_duration
            operation.duration = 0.01
            sub_ops_list = profile_op_data['sub_ops_list']
            sub_ops_list_copy = copy.deepcopy(sub_ops_list)
            for sub_op in sub_ops_list_copy:
                sub_op.name_with_id += f"_{str(operation.batch_id)}_{operation.name}"
        else:
            if profile_op_data is not None:
                if isinstance(profile_op_data, dict):
                    duration_value = profile_op_data.get('duration', None)
                else:
                    duration_value = profile_op_data
                if duration_value is not None:
                    operation.duration = float(duration_value)
                    operation.hidden_duration = None
                    if operation.name in {"dp_allreduce", "ep_allreduce", "tp_allreduce"}:
                        print(f"allreduce的duration补充: operation.name: {operation.name}, duration{operation.duration}")
            elif operation.duration is None:
                operation.duration = 0.01
    else:
        print(f"{operation.name} not in GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST")

    ''' process2: muti-gpus trace 完善OP/SUBOP duration '''
    operation, sub_ops_list_copy, not_simulating_cmd_dict = check_and_supplement_duration_from_muti_gpus_trace(
        operation,
        sub_ops_list_copy,
        running_mode,
        trace_stages_dict,
        torch_graph_stage_op_dict,
        rank_id,
    )

    part_operations_list.append(operation)
    part_operations_list.extend(sub_ops_list_copy)

    return part_operations_list, not_simulating_cmd_dict


def check_and_supplement_duration_from_muti_gpus_trace(operation, sub_ops_list, running_mode, trace_stages_dict, torch_graph_stage_op_dict, rank_id):
    """ 
        对于SUBOP,如果存在于torch_graph_stage_op_dict(暂定comm的模拟数值存于该DICT),则直接使用;否则从trace中补充;
        对于OP和其余SUBOP, 从muti-gpus profile trace中补充operation和sub_ops_list的duration属性,并记录不支持的OP:
            1. 对于无sub_ops_list的独立OP,处理其duration属性
            2. 对于有sub_ops_list的OP,处理其duration属性和sub_ops_list中每个sub_op的duration属性
        
        return:
            operation: 修改后的OP
            sub_ops_list: 修改后的SUBOP
            not_simulating_cmd_dict: 记录SINGLE_GPU上没有PROFILE的OP或SUBOP or 没有被模拟的...
    """
    not_simulating_cmd_dict = {}

    if not sub_ops_list:
        if running_mode == MODE_SIMULATE:
            if operation.duration is None:
                communication_operations = {
                    'dp_allreduce', 'exp_dp_allreduce', 'ep_allreduce', 'tp_allreduce',
                    'send_forward', 'recv_forward', 'send_backward', 'recv_backward',
                    'exp_all_to_all', 'exp_allgather'
                }

                if operation.name in communication_operations:
                    print(f"Communication operation {operation.name}: Skipping trace duration, will use CC-predictor")
                    operation.set_duration(0.001)
                else:
                    if trace_stages_dict is None:
                        operation.set_duration(1)
                    elif getattr(operation, 'trace_metadata', None):
                        operation.set_duration(get_op_excution_time_from_trace(trace_stages_dict, operation))
                    else:
                        get_time = get_op_excution_time_from_trace(trace_stages_dict, operation) * random.uniform(0.93, 1.07)
                        operation.set_duration(get_time)

                assert operation.duration is not None
                if operation.name not in not_simulating_cmd_dict:
                    not_simulating_cmd_dict[operation.name] = operation.duration

        return operation, [], not_simulating_cmd_dict

    for each_sub_op in sub_ops_list:
        add_sub_op_prop(operation, each_sub_op)
        if running_mode == MODE_SIMULATE and each_sub_op.duration is None:
            duration_data = get_batch_specific_profile_op_data(
                torch_graph_stage_op_dict,
                rank_id,
                each_sub_op.name,
                batch_id=getattr(operation, 'batch_id', None),
            )
            if duration_data is not None:
                if isinstance(duration_data, dict) and 'duration' in duration_data:
                    each_sub_op.set_duration(duration_data['duration'])
                else:
                    each_sub_op.set_duration(duration_data)
                print(f"Debug: SubOperation duration set from torch_graph_stage_op_dict: {each_sub_op.name} = {each_sub_op.duration}")
            else:
                if trace_stages_dict is None:
                    each_sub_op.set_duration(1)
                else:
                    each_sub_op.set_duration(get_op_excution_time_from_trace(trace_stages_dict, each_sub_op))
                assert operation.duration is not None
                if operation.name not in not_simulating_cmd_dict:
                    not_simulating_cmd_dict[operation.name] = operation.duration

    return operation, sub_ops_list, not_simulating_cmd_dict



class SimulatorEngine():
    """ manager the whole workflow of the train simulation """
    def __init__(self, stages_num=None, stages_steps_dict=None, grad_acc=None, trace_filepath=None,
                framwork='megatron-lm', strategy="1F1B-none_interleaved", can_overlap=False, args=None, running_mode=None, torchgraph_filepath=None, stages_scheduling_filepath=None,
                cc_backend_name=None, cc_backend_options=None, moe_rank_selection="all"):

        self.stages_num = stages_num
        self.stages_steps_dict = stages_steps_dict
        self.grad_acc = grad_acc
        self.strategy = strategy
        self.args = args
        self.running_mode = running_mode
        self.can_overlap = can_overlap
        self.framwork = framwork
        self.trace_filepath = trace_filepath
        self.stages_scheduling_filepath = stages_scheduling_filepath
        self.torchgraph_filepath = torchgraph_filepath
        self.stages_timeline = []
        self.predictor = None
        self._op_excution_time_dict = {}
        self.compelete_wranks_list = None
        self.stages_dict = None
        self.mpu = None
        self.dependency_relationship = None
        self.comm_matching_relationship = None
        self.timeline_manager = None
        self.not_simulating_cmd_dict = {}

        self.global_waiting_pool: dict = {}
        self.global_finished_operations: dict = {}

        self.is_moe_model = False
        self.selected_ranks = set()
        self.optimization_enabled = True
        self.moe_rank_selection = moe_rank_selection

        self.cc_estimator = None
        self.cc_backend = None
        self.simulator_config = None
        self.cc_backend_name = cc_backend_name
        self.cc_backend_options = cc_backend_options or {}
        self._initialize_cc_backend()

        assert running_mode in RUNNING_MODE_OPTION, "Invalid running mode: {running_mode}"

    def _initialize_cc_backend(self):
        """Initialize pluggable communication backend for operation time prediction."""
        self.simulator_config = load_config_from_env()

        if self.cc_backend_name:
            self.simulator_config.communication.backend = self.cc_backend_name
        if self.cc_backend_options:
            merged_options = dict(getattr(self.simulator_config.communication, "backend_options", {}))
            selected_backend = self.simulator_config.communication.backend
            if selected_backend not in merged_options:
                merged_options[selected_backend] = {}
            if not isinstance(merged_options[selected_backend], dict):
                raise ValueError(
                    "simulator_config.communication.backend_options backend entry must be a dict"
                )
            selected_options = self.cc_backend_options
            if (
                isinstance(self.cc_backend_options, dict)
                and selected_backend in self.cc_backend_options
                and isinstance(self.cc_backend_options[selected_backend], dict)
            ):
                selected_options = self.cc_backend_options[selected_backend]
            if not isinstance(selected_options, dict):
                raise ValueError("cc_backend_options must be a dict")
            merged_options[selected_backend].update(selected_options)
            self.simulator_config.communication.backend_options = merged_options

        backend_name = self.simulator_config.communication.backend
        self.cc_backend = create_cc_backend(backend_name, self.simulator_config)

        # Keep legacy attribute for compatibility with existing tests/scripts.
        if backend_name == "cc-estimator" and hasattr(self.cc_backend, "estimator"):
            self.cc_estimator = self.cc_backend.estimator
        else:
            self.cc_estimator = None

        print(f"CC backend initialized: {backend_name}")

    def _initialize_cc_estimator(self):
        """Backward-compatible alias for legacy code paths."""
        self._initialize_cc_backend()

    @staticmethod
    def _is_moe_operation_name(operation_name: str) -> bool:
        moe_operations = {
            "exp_all_to_all",
            "exp_allgather",
            "exp_reducescatter",
            "exp_dp_allreduce",
            "ep_dp_allreduce",
        }
        return str(operation_name).strip().lower() in moe_operations

    def _contains_moe_signal_in_stage_dict(self, stage_dict) -> bool:
        if stage_dict is None or not hasattr(stage_dict, "operations_list"):
            return False

        for operation in stage_dict.operations_list:
            operation_name = str(getattr(operation, "name", "")).strip().lower()
            group_kind = str(getattr(operation, "group_kind", "")).strip().lower()

            if self._is_moe_operation_name(operation_name):
                return True
            if group_kind in {"exp", "exp_dp"}:
                return True

            sub_operations = getattr(operation, "sub_operations", None)
            if not isinstance(sub_operations, list):
                continue

            for sub_op_str in sub_operations:
                if not isinstance(sub_op_str, str):
                    continue
                lowered = sub_op_str.strip().lower()
                if "group=exp" in lowered or "group:exp" in lowered:
                    return True
                if "exp_all_to_all" in lowered or "exp_allgather" in lowered:
                    return True
                if "exp_dp_allreduce" in lowered or "ep_dp_allreduce" in lowered:
                    return True

        return False

    def _contains_moe_signal_in_profile_database(self, torch_graph_stage_op_dict) -> bool:
        if not isinstance(torch_graph_stage_op_dict, dict):
            return False

        for rank_or_stage_ops in torch_graph_stage_op_dict.values():
            if not isinstance(rank_or_stage_ops, dict):
                continue
            for operation_name in rank_or_stage_ops.keys():
                if self._is_moe_operation_name(operation_name):
                    return True
        return False

    def _detect_model_type(self, trace_stages_dict=None, stages_or_wranks_dict=None, torch_graph_stage_op_dict=None):
        """Detect whether current workload is MoE.

        Priority:
        1) Explicit topology signal in simulate mode (`mpu.exp_size > 1`)
        2) Operation signals from trace/schedule stage dictionaries
        3) Operation keys from database profile dictionary
        """

        # Strong topology evidence: in this engine exp_size denotes expert parallel degree.
        if (
            self.running_mode == MODE_SIMULATE
            and getattr(self, "mpu", None) is not None
            and int(getattr(self.mpu, "exp_size", 1) or 1) > 1
        ):
            return True

        for stage_source in (trace_stages_dict, stages_or_wranks_dict):
            if not isinstance(stage_source, dict):
                continue
            for stage_dict in stage_source.values():
                if self._contains_moe_signal_in_stage_dict(stage_dict):
                    return True

        if self._contains_moe_signal_in_profile_database(torch_graph_stage_op_dict):
            return True

        return False

    @staticmethod
    def parse_profile_filename_parallel_tokens(filename: str) -> dict:
        """Parse parallel tokens from database_profile file name.

        Semantics:
        - expX: expert parallel degree
        - epX: embedding parallel degree
        """
        token_info = {"exp": None, "ep": None}
        stem_name = os.path.splitext(str(filename).strip())[0]
        for raw_part in stem_name.split("_"):
            part = raw_part.strip().lower()
            if part.startswith("exp") and part[3:].isdigit():
                token_info["exp"] = int(part[3:])
            elif part.startswith("ep") and part[2:].isdigit():
                token_info["ep"] = int(part[2:])
        return token_info

    def validate_profile_filename_parallel_tokens(self, filename: str, parsed_tokens: dict) -> None:
        """Validate `exp`/`ep` tokens in profile file name against engine topology."""
        if self.mpu is None:
            raise ValueError("mpu_info must be initialized before validating database_profile file names.")

        expected_exp_size = int(getattr(self.mpu, "exp_size", 0) or 0)
        expected_ep_size = int(getattr(self.mpu, "ep_size", 0) or 0)
        file_exp_size = parsed_tokens.get("exp")
        file_ep_size = parsed_tokens.get("ep")

        if file_exp_size is not None and expected_exp_size > 0 and file_exp_size != expected_exp_size:
            raise ValueError(
                "Database profile filename semantic mismatch: "
                f"file `{filename}` has exp={file_exp_size} (expert parallel), "
                f"but engine topology expects exp_size={expected_exp_size}."
            )

        if file_ep_size is not None:
            if expected_ep_size <= 0:
                raise ValueError(
                    "Database profile filename semantic mismatch: "
                    f"file `{filename}` has ep={file_ep_size} (embedding parallel), "
                    "but engine topology does not provide a valid ep_size."
                )
            if file_ep_size != expected_ep_size:
                raise ValueError(
                    "Database profile filename semantic mismatch: "
                    f"file `{filename}` has ep={file_ep_size} (embedding parallel), "
                    f"but engine topology expects ep_size={expected_ep_size}."
                )

    def _validate_database_profile_filename_semantics(self) -> None:
        """Fail fast when database profile filename semantics conflict with engine topology."""
        if not self.torchgraph_filepath:
            return
        if not os.path.isdir(self.torchgraph_filepath):
            return

        txt_filenames = sorted(
            filename for filename in os.listdir(self.torchgraph_filepath) if filename.endswith(".txt")
        )
        for filename in txt_filenames:
            parsed_tokens = self.parse_profile_filename_parallel_tokens(filename)
            if parsed_tokens["exp"] is None and parsed_tokens["ep"] is None:
                continue
            self.validate_profile_filename_parallel_tokens(filename, parsed_tokens)

    def _select_optimization_ranks(self, rank_instances_dict):
        """Select the ranks that need explicit timeline replay."""
        selected_ranks = set()

        if not self.optimization_enabled:
            return set(rank_instances_dict.keys())

        if not hasattr(self, 'mpu') or self.mpu is None:
            return set(rank_instances_dict.keys())

        moe_rank_selection = getattr(self, 'moe_rank_selection', 'all')

        if self.is_moe_model:
            if moe_rank_selection == "all":
                selected_ranks = set(rank_instances_dict.keys())
                print(
                    "Warning: MoE workload uses all ranks for timeline construction and communication-barrier semantics."
                )
                print(
                    f"优化策略: MOE 模型(all)，选择了 {len(selected_ranks)} 个ranks: {sorted(selected_ranks)}"
                )
                return selected_ranks

            if moe_rank_selection != "pp-ep":
                raise ValueError(
                    f"Unsupported moe_rank_selection={moe_rank_selection}. Expected one of: all, pp-ep"
                )

            pp_size = self.mpu.pp_size
            tp_size = self.mpu.tp_size
            dp_size = self.mpu.dp_size
            exp_size = getattr(self.mpu, 'exp_size', 1)

            for pp_rank in range(pp_size):
                pp_base_rank = pp_rank * tp_size * dp_size
                for exp_rank in range(exp_size):
                    representative_rank = pp_base_rank + exp_rank * tp_size
                    if representative_rank in rank_instances_dict:
                        selected_ranks.add(representative_rank)

            print(
                f"优化策略: MOE 模型(pp-ep)，选择了 {len(selected_ranks)} 个ranks: {sorted(selected_ranks)}"
            )
            return selected_ranks

        pp_size = self.mpu.pp_size
        tp_size = self.mpu.tp_size
        dp_size = self.mpu.dp_size

        for pp_rank in range(pp_size):
            representative_rank = pp_rank * tp_size * dp_size
            if representative_rank in rank_instances_dict:
                selected_ranks.add(representative_rank)

        print(
            f"优化策略: Dense 模型，选择了 {len(selected_ranks)} 个ranks: {sorted(selected_ranks)}"
        )
        return selected_ranks

    def _init_3d_parallel_all_ranks(self, stages_or_wranks_dict, rank_instances_dict, trace_stages_dict, torch_graph_stage_op_dict):
        """
            根据mpu中的pp,tp,dp信息生成完整rank_list(包含所有ranks或优化后的部分ranks)
            1. 当profile文件与world_size一致,且为非SIMULATE MODE: 直接返回即可,OP or SubOP的初始化已经全部完成
            2. 否则,说明只是根据schedule plan(PP-level)进行的初始化,需要根据PP/TP/DP group进行完整初始化(SIMULATE MODE必然使用了stage标记的生成方式,即便文件数量与world_size一致)

        """
        assert len(rank_instances_dict) == self.mpu.world_size, "Invalid rank_instances_dict"
        not_simulating_cmd_dict = {}

        # 检测模型类型并选择优化的ranks
        self.is_moe_model = self._detect_model_type(
            trace_stages_dict=trace_stages_dict,
            stages_or_wranks_dict=stages_or_wranks_dict,
            torch_graph_stage_op_dict=torch_graph_stage_op_dict,
        )
        if (
            self.running_mode == MODE_SIMULATE
            and int(getattr(self.mpu, "exp_size", 1) or 1) > 1
            and not self.is_moe_model
        ):
            raise ValueError(
                "Fail-fast: simulate mode expects MoE workload when exp_size > 1, "
                "but model type detection returned Dense."
            )
        self.selected_ranks = self._select_optimization_ranks(rank_instances_dict)

        # WARNING:对于有trace的情况，stage_dict已经完整初始化，因此必须直接返回(否则操作的不是stage_id而是wrank_id）；
        if len(stages_or_wranks_dict) == self.mpu.world_size and (self.running_mode == MODE_PROFILE or self.running_mode == MODE_MODEL):
            # 此时stage list中已包含所有ranks (profile mode); 只要是simulating/logic mode,都需要重新初始化所有ranks
            if self.optimization_enabled:
                # 过滤出选中的ranks
                filtered_stages = [stages_or_wranks_dict[rank_id] for rank_id in self.selected_ranks if rank_id in stages_or_wranks_dict]
                return filtered_stages, not_simulating_cmd_dict
            else:
                return list(stages_or_wranks_dict.values()), not_simulating_cmd_dict

        # SIMULATE MODE 下只读取schedule plan生成的结果
        # stage_list中包含了所有的Stage对象,每个Stage instance只完成了stage_id的初始化,
        # 因此需要根据pp/tp/dp group完成所有rank的Stage实例的初始化,并返回该完整初始化后的list
        # 含MODEL和SIMULATING模式
        compelete_wranks_list = []

        # 根据优化策略决定处理哪些ranks
        ranks_to_process = self.selected_ranks if self.optimization_enabled else rank_instances_dict.keys()

        for rank_id in ranks_to_process:
            if rank_id not in rank_instances_dict:
                continue

            rank_instance = rank_instances_dict[rank_id]
            stage_id = rank_instance._get_pp_local_rank()
            print(f"check rank_id:{rank_id} -> stage_id:{stage_id}")

            new_stage_instance: Stage = copy.deepcopy(stages_or_wranks_dict[stage_id])
            new_stage_instance.set_stage_wrank_id(rank_id)
            new_stage_instance.set_stage_rank(rank_instance)
            override_operations_list = []
            trace_overlay_state = None
            representative_overlay_targets = {}
            reserved_representative_keys = set()
            if self.running_mode == MODE_SIMULATE and trace_stages_dict is not None:
                trace_overlay_state = self._build_trace_overlay_state(trace_stages_dict.get(rank_id))
                representative_overlay_targets = self._plan_representative_trace_overlay_targets(
                    new_stage_instance.operations_list,
                    trace_overlay_state,
                )
                reserved_representative_keys = set(representative_overlay_targets.values())

            for operation_index, operation in enumerate(new_stage_instance.operations_list):
                operation.set_wrank_id(rank_id)
                operation.set_stage_id(stage_id)

                matched_trace_operation = None
                if trace_overlay_state is not None:
                    representative_key = representative_overlay_targets.get(operation_index, None)
                    if representative_key is not None:
                        matched_trace_operation = self._consume_representative_trace_top_level_operation(
                            trace_overlay_state,
                            representative_key,
                        )
                    elif (
                        self._schedule_operation_requires_trace_overlay(operation)
                        and (operation.name, operation.mg_state) in reserved_representative_keys
                    ):
                        matched_trace_operation = None
                    else:
                        matched_trace_operation = self._consume_matching_trace_top_level_operation(
                            trace_overlay_state,
                            operation,
                        )

                    requires_trace_overlay = (
                        operation.name == 'dp_allreduce' and operation.mg_state == 'finalize'
                    )
                    if requires_trace_overlay and matched_trace_operation is None:
                        raise ValueError(
                            f'Missing matching trace operation for schedule op '
                            f'{operation.name} batch_id={operation.batch_id} mg_state={operation.mg_state} '
                            f'on rank {rank_id}'
                        )
                    if matched_trace_operation is not None:
                        self._apply_trace_overlay_metadata_to_schedule_operation(
                            operation,
                            matched_trace_operation,
                        )

                ddp_overlap_operations = []
                if trace_overlay_state is not None and operation.name == 'backward_step':
                    ddp_overlap_operations = self._collect_ddp_overlap_overlay_operations(
                        trace_overlay_state,
                        matched_trace_operation,
                        operation,
                    )

                slowdown_config = getattr(self.simulator_config, 'slowdown', None)
                is_trace_driven_slowdown_backward = (
                    self.running_mode == MODE_SIMULATE
                    and bool(getattr(slowdown_config, 'enabled', False))
                    and operation.name == 'backward_step'
                    and matched_trace_operation is not None
                    and operation.cmd_uid is not None
                    and bool(ddp_overlap_operations)
                )

                if is_trace_driven_slowdown_backward:
                    trace_duration = getattr(matched_trace_operation, 'duration', None)
                    if trace_duration is None:
                        raise ValueError(
                            'Trace-driven slowdown backward is missing trace duration: '
                            f'cmd_uid={operation.cmd_uid}'
                        )
                    legacy_operations_list, part_not_simulating_cmd_dict = (
                        add_sub_ops_according_to_profile_dict(
                            torch_graph_stage_op_dict,
                            operation,
                            rank_id,
                            self.running_mode,
                            trace_stages_dict,
                        )
                    )
                    if not legacy_operations_list or legacy_operations_list[0] is not operation:
                        raise ValueError(
                            'Legacy profile expansion did not retain the backward parent: '
                            f'cmd_uid={operation.cmd_uid}'
                        )
                    legacy_children = legacy_operations_list[1:]
                    unsupported_child_kinds = sorted(
                        {
                            child.op_kind
                            for child in legacy_children
                            if child.op_kind not in {'comp', 'comm'}
                        },
                        key=str,
                    )
                    if unsupported_child_kinds:
                        raise ValueError(
                            'Unsupported legacy sub-operation kind for trace-driven slowdown '
                            f'backward: cmd_uid={operation.cmd_uid}, '
                            f'op_kinds={unsupported_child_kinds}'
                        )
                    communication_children = [
                        child for child in legacy_children if child.op_kind == 'comm'
                    ]
                    operation.duration = float(trace_duration)
                    operation.hidden_duration = None
                    part_override_operations_list = [operation, *communication_children]
                else:
                    # check torch_graph的当前rank_id是否存在subop_list,有则拆分算子
                    part_override_operations_list, part_not_simulating_cmd_dict = add_sub_ops_according_to_profile_dict(torch_graph_stage_op_dict,                                                      operation, rank_id, self.running_mode, trace_stages_dict)

                override_operations_list.extend(part_override_operations_list)
                not_simulating_cmd_dict.update(part_not_simulating_cmd_dict)
                override_operations_list.extend(copy.deepcopy(ddp_overlap_operations))

            new_stage_instance.operations_list = override_operations_list
            compelete_wranks_list.append(new_stage_instance)

            if trace_overlay_state is not None:
                self._validate_trace_overlay_state_consumed(trace_overlay_state, rank_id)

        return compelete_wranks_list, not_simulating_cmd_dict

    @staticmethod
    def _get_trace_operation_start_ms(trace_operation: Operation):
        trace_metadata = getattr(trace_operation, 'trace_metadata', {}) or {}
        if trace_metadata.get('trace_event_type') == 'ddp_grad_comm':
            launch_timestamp_ms = trace_metadata.get('launch_timestamp_ms', None)
            return None if launch_timestamp_ms is None else float(launch_timestamp_ms)

        end_timestamp = getattr(trace_operation, 'end_timestamp', None)
        duration = getattr(trace_operation, 'duration', None)
        if end_timestamp is None or duration is None:
            return None
        return round(float(end_timestamp) - float(duration), 2)

    @staticmethod
    def _annotate_inferred_metadata_placeholder_durations(trace_stage):
        if trace_stage is None:
            return

        operations_list = list(trace_stage.operations_list)
        for index, trace_operation in enumerate(operations_list):
            if isinstance(trace_operation, SubOperation):
                continue
            if trace_operation.name != 'dp_allreduce':
                continue
            if getattr(trace_operation, 'op_semantics', None) != 'metadata_placeholder':
                continue

            trace_metadata = dict(getattr(trace_operation, 'trace_metadata', {}) or {})
            if trace_metadata.get('finalize_base_duration_ms', None) is not None:
                trace_operation.trace_metadata = trace_metadata
                continue
            if trace_metadata.get('inferred_base_duration_ms', None) is not None:
                trace_operation.trace_metadata = trace_metadata
                continue

            previous_finish_ms = None
            for prev_index in range(index - 1, -1, -1):
                previous_operation = operations_list[prev_index]
                if isinstance(previous_operation, SubOperation):
                    continue
                previous_trace_type = getattr(previous_operation, 'trace_metadata', {}) or {}
                if previous_trace_type.get('trace_event_type') == 'ddp_grad_comm':
                    continue
                previous_finish_ms = getattr(previous_operation, 'end_timestamp', None)
                if previous_finish_ms is not None:
                    previous_finish_ms = float(previous_finish_ms)
                    break

            next_start_ms = None
            for next_index in range(index + 1, len(operations_list)):
                next_operation = operations_list[next_index]
                if isinstance(next_operation, SubOperation):
                    continue
                next_trace_type = getattr(next_operation, 'trace_metadata', {}) or {}
                if next_trace_type.get('trace_event_type') == 'ddp_grad_comm':
                    continue
                next_start_ms = SimulatorEngine._get_trace_operation_start_ms(next_operation)
                if next_start_ms is not None:
                    break

            if previous_finish_ms is None or next_start_ms is None:
                continue

            inferred_base_duration_ms = round(float(next_start_ms) - float(previous_finish_ms), 2)
            if inferred_base_duration_ms < -0.05:
                raise ValueError(
                    'Invalid inferred metadata-placeholder duration from neighboring trace ops: '
                    f'prev_finish={previous_finish_ms}, next_start={next_start_ms}, '
                    f'cmd_uid={getattr(trace_operation, "cmd_uid", None)}'
                )
            if inferred_base_duration_ms <= 0:
                continue

            trace_metadata['inferred_base_duration_ms'] = inferred_base_duration_ms
            trace_metadata['base_duration_source'] = 'neighbor_gap'
            trace_operation.trace_metadata = trace_metadata

    @staticmethod
    def _build_trace_overlay_state(trace_stage):
        if trace_stage is None:
            return None

        SimulatorEngine._annotate_inferred_metadata_placeholder_durations(trace_stage)

        top_level_trace_queues = {}
        top_level_trace_queues_by_name_state = {}
        ddp_comm_by_trigger_cmd = {}
        pending_metadata_only_ddp_comm = []

        for trace_operation in trace_stage.operations_list:
            if isinstance(trace_operation, SubOperation):
                continue

            trace_metadata = dict(getattr(trace_operation, 'trace_metadata', {}) or {})
            trace_event_type = trace_metadata.get('trace_event_type', None)
            if trace_event_type == 'ddp_grad_comm':
                trigger_cmd_uid = trace_metadata.get('trigger_cmd_uid', None)
                if trigger_cmd_uid is None:
                    raise ValueError('ddp_grad_comm trace event is missing trigger_cmd_uid')

                copied_trace_operation = copy.deepcopy(trace_operation)
                copied_trace_operation.trace_metadata = trace_metadata
                ddp_comm_by_trigger_cmd.setdefault(trigger_cmd_uid, []).append(copied_trace_operation)

                wait_cmd_uid = trace_metadata.get('wait_cmd_uid', None)
                if bool(trace_metadata.get('metadata_only', False)) and wait_cmd_uid in {None, 'None'}:
                    pending_metadata_only_ddp_comm.append(copied_trace_operation)
                continue

            if (
                trace_operation.name == 'dp_allreduce'
                and trace_operation.mg_state == 'finalize'
                and pending_metadata_only_ddp_comm
            ):
                wait_cmd_uid = getattr(trace_operation, 'cmd_uid', None)
                if wait_cmd_uid not in {None, 'None'}:
                    for pending_operation in pending_metadata_only_ddp_comm:
                        pending_operation.trace_metadata['wait_cmd_uid'] = wait_cmd_uid
                    pending_metadata_only_ddp_comm = []

            queue_key = (trace_operation.name, trace_operation.batch_id, trace_operation.mg_state)
            top_level_trace_queues.setdefault(queue_key, deque()).append(trace_operation)
            name_state_key = (trace_operation.name, trace_operation.mg_state)
            top_level_trace_queues_by_name_state.setdefault(name_state_key, deque()).append(trace_operation)

        return {
            'top_level_trace_queues': top_level_trace_queues,
            'top_level_trace_queues_by_name_state': top_level_trace_queues_by_name_state,
            'ddp_comm_by_trigger_cmd': ddp_comm_by_trigger_cmd,
        }

    @staticmethod
    def _consume_matching_trace_top_level_operation(overlay_state, schedule_operation: Operation):
        if overlay_state is None:
            return None

        queue_key = (schedule_operation.name, schedule_operation.batch_id, schedule_operation.mg_state)
        candidate_queue = overlay_state['top_level_trace_queues'].get(queue_key, None)
        if candidate_queue:
            trace_operation = candidate_queue.popleft()
            name_state_key = (trace_operation.name, trace_operation.mg_state)
            candidate_name_state_queue = overlay_state.get('top_level_trace_queues_by_name_state', {}).get(
                name_state_key,
                None,
            )
            if candidate_name_state_queue is not None:
                try:
                    candidate_name_state_queue.remove(trace_operation)
                except ValueError as exc:
                    raise ValueError(
                        f'Exact trace overlay queue desynchronized for key {queue_key}'
                    ) from exc
            return trace_operation

        return None

    @staticmethod
    def _schedule_operation_requires_trace_overlay(operation: Operation) -> bool:
        return operation.name == 'backward_step' or (
            operation.name == 'dp_allreduce' and operation.mg_state == 'finalize'
        )

    @staticmethod
    def _plan_representative_trace_overlay_targets(schedule_operations, overlay_state):
        if overlay_state is None:
            return {}

        relevant_schedule_positions = {}
        for op_index, operation in enumerate(schedule_operations):
            if not SimulatorEngine._schedule_operation_requires_trace_overlay(operation):
                continue
            name_state_key = (operation.name, operation.mg_state)
            relevant_schedule_positions.setdefault(name_state_key, []).append(op_index)

        representative_targets = {}
        for name_state_key, candidate_queue in overlay_state.get(
            'top_level_trace_queues_by_name_state', {}
        ).items():
            schedule_positions = relevant_schedule_positions.get(name_state_key, [])
            if not schedule_positions:
                continue

            if len(candidate_queue) == len(schedule_positions):
                for schedule_position in schedule_positions:
                    representative_targets[schedule_position] = name_state_key
                continue

            if len(candidate_queue) == 1:
                representative_targets[schedule_positions[-1]] = name_state_key

        return representative_targets

    @staticmethod
    def _consume_representative_trace_top_level_operation(overlay_state, name_state_key):
        if overlay_state is None:
            return None

        candidate_queue = overlay_state.get('top_level_trace_queues_by_name_state', {}).get(
            name_state_key,
            None,
        )
        if not candidate_queue:
            return None

        trace_operation = candidate_queue.popleft()
        exact_key = (trace_operation.name, trace_operation.batch_id, trace_operation.mg_state)
        exact_queue = overlay_state['top_level_trace_queues'].get(exact_key, None)
        if exact_queue is None:
            raise ValueError(
                f'Representative trace overlay missing exact queue for key {exact_key}'
            )

        try:
            exact_queue.remove(trace_operation)
        except ValueError as exc:
            raise ValueError(
                f'Representative trace overlay queue desynchronized for key {exact_key}'
            ) from exc

        return trace_operation

    @staticmethod
    def _apply_trace_overlay_metadata_to_schedule_operation(schedule_operation: Operation, trace_operation: Operation):
        if trace_operation is None:
            return

        schedule_operation.cmd_uid = getattr(trace_operation, 'cmd_uid', None)
        schedule_operation.op_semantics = getattr(trace_operation, 'op_semantics', None)
        schedule_operation.trace_metadata = dict(getattr(trace_operation, 'trace_metadata', {}) or {})
        schedule_operation.end_timestamp = getattr(trace_operation, 'end_timestamp', None)

        if schedule_operation.tensor_shape is None and getattr(trace_operation, 'tensor_shape', None) is not None:
            schedule_operation.tensor_shape = copy.deepcopy(trace_operation.tensor_shape)
        if schedule_operation.tensor_dtype is None and getattr(trace_operation, 'tensor_dtype', None) is not None:
            schedule_operation.tensor_dtype = trace_operation.tensor_dtype

        if schedule_operation.name == 'dp_allreduce' and schedule_operation.op_semantics in {'wait_flush_only', 'metadata_placeholder'}:
            schedule_operation.op_kind = 'comp'
            explicit_base_duration_ms = schedule_operation.trace_metadata.get('finalize_base_duration_ms', None)
            inferred_base_duration_ms = schedule_operation.trace_metadata.get('inferred_base_duration_ms', None)
            if explicit_base_duration_ms is not None:
                explicit_base_duration_ms = float(explicit_base_duration_ms)
                if explicit_base_duration_ms < 0:
                    raise ValueError(
                        f'Invalid finalize_base_duration_ms={explicit_base_duration_ms} for cmd_uid={schedule_operation.cmd_uid}'
                    )
                schedule_operation.duration = explicit_base_duration_ms
            elif inferred_base_duration_ms is not None and getattr(schedule_operation, 'duration', None) in (None, 0, 0.0):
                inferred_base_duration_ms = float(inferred_base_duration_ms)
                if inferred_base_duration_ms < 0:
                    raise ValueError(
                        f'Invalid inferred_base_duration_ms={inferred_base_duration_ms} for cmd_uid={schedule_operation.cmd_uid}'
                    )
                schedule_operation.duration = inferred_base_duration_ms

    @staticmethod
    def _collect_ddp_overlap_overlay_operations(
        overlay_state,
        matched_trace_operation: Operation,
        schedule_operation=None,
    ):
        if overlay_state is None or matched_trace_operation is None:
            return []

        trigger_cmd_uid = getattr(matched_trace_operation, 'cmd_uid', None)
        if trigger_cmd_uid is None:
            return []

        overlay_operations = overlay_state['ddp_comm_by_trigger_cmd'].pop(trigger_cmd_uid, [])
        if schedule_operation is None:
            return overlay_operations

        remapped_operations = []
        for overlay_operation in overlay_operations:
            copied_operation = copy.deepcopy(overlay_operation)
            copied_operation.batch_id = schedule_operation.batch_id
            copied_operation.mg_state = schedule_operation.mg_state
            trace_metadata = dict(getattr(copied_operation, 'trace_metadata', {}) or {})
            trace_metadata['trigger_batch_id'] = schedule_operation.batch_id
            copied_operation.trace_metadata = trace_metadata
            remapped_operations.append(copied_operation)

        return remapped_operations

    @staticmethod
    def _validate_trace_overlay_state_consumed(overlay_state, rank_id: int):
        if overlay_state is None:
            return

        remaining_ddp_overlay = {
            key: value for key, value in overlay_state['ddp_comm_by_trigger_cmd'].items() if value
        }
        if remaining_ddp_overlay:
            raise ValueError(
                f'Unconsumed DDP overlap trace overlay remains for rank {rank_id}: '
                f'{sorted(remaining_ddp_overlay.keys())}'
            )

        remaining_top_level_overlay = []
        for candidate_queue in overlay_state['top_level_trace_queues'].values():
            for trace_operation in list(candidate_queue):
                if not SimulatorEngine._schedule_operation_requires_trace_overlay(trace_operation):
                    continue
                remaining_top_level_overlay.append(
                    (trace_operation.name, trace_operation.batch_id, trace_operation.mg_state)
                )

        if remaining_top_level_overlay:
            raise ValueError(
                f'Unconsumed top-level trace overlay remains for rank {rank_id}: '
                f'{sorted(remaining_top_level_overlay)}'
            )


    def _init_excution_time_predictor(self):
        """ init excution time predictor"""
        # import yaml
        # from model_zoo import ModelZoo
        # from benchmark_tools import BenchmarkTools
        
        # with open(self.args.config) as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)
        # model_zoo = ModelZoo(config)
        # self.predictor = BenchmarkTools(models,
        #                                 model_zoo,
        #                                 args.skip_coverage,
        #                                 args.skip_accuracy,
        #                                 config)
        pass

    def _get_excution_time_from_predictor(self, graph_json_path, database_json_path):
        """ get excution time from Merak, update self._op_excution_time_dict """
        pass
    
    # @staticmethod
    # def _add_subop_to_3d_parallel_all_ranks(compelete_wranks_list, torchgraph_filepath):

    #             # Calculate the start time of the operation
    #             start_time = timestamp - duration

    #             # Initialize sub_operations list
    #             # all_sub_ops = []
    #             all_sub_ops = extract_sub_operations(eval(sub_operations_str), start_time, duration, stage_rank.wrank_id, stage_rank.stage_id)

    #             # Append sub_operations to the operations list in the appropriate place
    #             for operation in stage_rank.operations_list:
    #                 if operation.name == cmd_name:
    #                     operation.duration = duration
    #                     index = stage_rank.operations_list.index(operation)
    #                     for sub_op in all_sub_ops:
    #                         sub_op.batch_id = operation.batch_id
    #                         sub_op.stage_id = operation.stage_id
    #                         sub_op.mg_state = operation.mg_state
    #                     stage_rank.operations_list[index:index+1] = [operation] + all_sub_ops




    def _init_tmp_stages_dataset_and_timeline_manager(self, rank_instances_dict, mpu_info: MPUInfo):
        self._validate_database_profile_filename_semantics()

        self.compelete_wranks_list, self.stages_dict, torch_graph_stage_op_dict, trace_stages_dict = self.generate_stages_and_cmds_info_from_datasets_and_schedules(self.trace_filepath, 
                                                                                    rank_instances_dict, self.running_mode, self.framwork, self.torchgraph_filepath,self.mpu, self.stages_scheduling_filepath)
        self.compelete_wranks_list, self.not_simulating_cmd_dict = self._init_3d_parallel_all_ranks(self.stages_dict, rank_instances_dict, trace_stages_dict, torch_graph_stage_op_dict)
        

        self.timeline_manager = TimelinesManager(dependency_relationship=self.dependency_relationship,
                                                 comm_matching_relationship=self.comm_matching_relationship,
                                                 compelete_wranks_list = self.compelete_wranks_list, strategy=self.strategy, can_overlap=self.can_overlap,
                                                 global_waiting_pool=self.global_waiting_pool, global_finished_operations=self.global_finished_operations,
                                                 mpu_info=mpu_info,running_mode=self.running_mode, torch_graph_stage_op_dict=torch_graph_stage_op_dict,
                                                 torchgraph_filepath=self.torchgraph_filepath, trace_filepath=self.trace_filepath, not_simulating_cmd_dict=self.not_simulating_cmd_dict,
                                                 rank_instances_dict=rank_instances_dict,trace_stages_dict=trace_stages_dict,
                                                 optimization_enabled=self.optimization_enabled, is_moe_model=self.is_moe_model, selected_ranks=self.selected_ranks,
                                                 cc_estimator=self.cc_estimator, simulator_config=self.simulator_config, cc_backend=self.cc_backend)
        self.can_overlap = self.timeline_manager.can_overlap
        print(f"INIT | tmp stages dataset and timeline manager have been set.")

    def  _set_mpu_info_and_init_key_relationship(self, mpu_info):
        self.mpu = mpu_info
        self.dependency_relationship = self._get_dependency_relationship(self.framwork, self.strategy)
        self.comm_matching_relationship = self._get_comm_matching_relationship(self.framwork, self.strategy)
        print(f"INIT | mpu info and key relationship have been set.")

    def _get_mpu_info(self):
        if self.mpu:
            return self.mpu
        else:
            raise ValueError(f"mpu_info is not set.")

    def validate_global_placement_requirements(self):
        """Fail fast when global placement mode lacks topology or comm-group semantics."""
        if self.cc_backend is None:
            return
        if str(getattr(self.cc_backend, "backend_name", "")).strip().lower() != "collective-sim":
            return
        if str(getattr(self.cc_backend, "placement_mode", "auto")).strip().lower() != "global":
            return

        missing_fields = []
        unresolved_comm_group_fields = set()

        mpu_info = self.mpu
        if mpu_info is None:
            missing_fields.append("mpu_info")
        else:
            for field in ("world_size", "tp_size", "dp_size", "pp_size", "exp_size"):
                if getattr(mpu_info, field, None) in (None, 0):
                    missing_fields.append(f"mpu_info.{field}")

        if self.timeline_manager is None:
            missing_fields.append("timeline_manager")
        else:
            group_attr_mapping = {
                "dp": "dp_groups",
                "tp": "tp_groups",
                "pp": "pp_groups",
                "ep": "ep_groups",
                "exp": "exp_groups",
                "exp_dp": "dp_modulo_exp_groups",
                "cp": "cp_groups",
            }

            if mpu_info is not None:
                used_group_kinds = set()
                for timeline in self.timeline_manager.stages_timeline_process_dict.values():
                    for operation in list(timeline.waiting_queue):
                        if getattr(operation, "op_kind", None) != "comm":
                            continue

                        try:
                            _, group_kind = self.timeline_manager._get_comm_operation_kind_and_parallel_dimension(
                                operation
                            )
                        except Exception:
                            unresolved_comm_group_fields.add("comm_group.group_kind")
                            continue

                        used_group_kinds.add(group_kind)
                        comm_group = self.timeline_manager._get_comm_group_for_operation(operation)
                        if not comm_group:
                            unresolved_comm_group_fields.add(f"comm_group.{group_kind}_ranks")

                for group_kind in sorted(used_group_kinds):
                    group_attr = group_attr_mapping.get(group_kind)
                    if group_attr is None:
                        unresolved_comm_group_fields.add(f"comm_group.{group_kind}_ranks")
                        continue
                    if getattr(mpu_info, group_attr, None) in (None, [], ()):
                        missing_fields.append(f"mpu_info.{group_attr}")

        if missing_fields or unresolved_comm_group_fields:
            details = sorted(set(missing_fields) | set(unresolved_comm_group_fields))
            detail_text = ", ".join(details)
            raise ValueError(
                "placement_mode=global requires complete mpu_info and comm_group metadata before simulation starts. "
                f"Missing or unresolved fields: {detail_text}. "
                "These fields are required to build physical placement-aware participant ranks for each communication "
                "domain (DP/TP/EP/PP/CP). Please provide complete topology degrees/groups and ensure trace/schedule "
                "communication ops include resolvable group_kind and comm_group mappings."
            )

    def _get_dependency_relationship(self, framwork=None, strategy=None):
        ''' 
        现阶段不存在跨stages的依赖项,目前维护的依赖项都是过程中关系, 而非模型内部的算子的关系X
        '''
        if framwork == "deepspeed":
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                return {"FirstStage": {'ForwardPass': ['LoadMicroBatch', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0]}, 
                        "MiddleStage": {'ForwardPass': ['RecvActivation', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0], 'SendGrad': ['BackwardPass', 0]},
                        "LastStage": {'ForwardPass': [['RecvActivation', 0], ['LoadMicroBatch', 0]], 'SendGrad': ['BackwardPass', 0]}}
            raise ValueError(f"Invalid mode, cannot get dependency relationship.")

        if framwork == "megatron-lm":
            if strategy == "no-pipelining" and self.mpu.pp_size == 1 and self.mpu.dp_size >= 1:
                return {"SingleStage": {}}

            # PP>1时的依赖关系
            # TODO: 目前dp_allreduce和comp的依赖关系还没写入，因此需要定制overlap策略和对应控制模块
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                # 1/2D pp并行: pp>1,dp==1,tp==1: 没有allreduce、没有broadcast、只涉及pp的P2P
                return {"FirstStage": {'forward_step': ['get_batch', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': [['recv_forward', 0], ['get_batch', 0]], 'send_backward': ['backward_step', 0]}}
            
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size > 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                # 3D or 2D pp,tp并行: pp>1,tp>1,dp=1: 没有dp的allreduce，涉及pp的p2p和tp的allreduce\broadcast
                # first stage会broadcast输入相关的数据，如 tokens、attention_mask 和 position_ids
                # last stage会broadcast与输出相关的数据，如 labels 和 loss_mask
                # 其他中间阶段不会执行任何 broadcast 操作
                # TODO:tp_load_batch_broadcast是不是可以改为load mb?(参考ds)
                return {"FirstStage": {'forward_step': ['get_batch', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': [['recv_forward', 0],['get_batch', 0]], 'send_backward': ['backward_step', 0]}}

            if strategy == "no-pipelining" and self.mpu.tp_size > 1 and self.mpu.dp_size == 1:
                return {"SingleStage": {}}

            raise ValueError(f"Invalid mode, cannot get dependency relationship.")

        raise ValueError(f"The framework '{framwork}' is not supported.")
        
    def _get_comm_matching_relationship(self, framwork=None, strategy=None):
        ''' 区别于dependency_relationship, 该dict用于确认comm op之间的匹配信息,而非依赖信息,匹配关系只与并行size相关 '''
        if framwork == "deepspeed":
            return_dict = {}

            if self.mpu.pp_size > 1:
                return_dict = {**return_dict, **{
                        'RecvGrad': ['SendGrad', 1],
                        'RecvActivation': ['SendActivation', -1],
                        'SendActivation': ['RecvActivation', 1],
                        'SendGrad': ['RecvGrad', -1]
                }}
            
            if self.mpu.dp_size > 1:
                return_dict = {**return_dict, **{
                    'ReduceGrads': ['ReduceGrads', 'dp_group'],
                    'ReduceTiedGrads': ['ReduceTiedGrads', 'dp_group']
                }}
            
            return return_dict

        elif framwork == "megatron-lm":
            # 根据return_dict[key][1]的类别判断，是int类则可直接用作offset，是str类则需要根据
            # tp_group的情况进行判断, rank==rank0时，matching为其余所有rank；否则，为matching为rank0
            # rank==rank0, 生成tp_size-1个broadcast ops(同join_time), 每个op看作一个独立的op，找对应的匹配op
            # 按照2部曲，对方已经注册，收尾操作；如果对方未注册，自己注册并block；由于如果是对方收尾，会进行bloc
            # 的解除（但此时rank0上并不是所有block都没有，这里还要再判断下pool里头是否还有别的broadcast的op，都没了才修改bool）
            return_dict = {}
            if self.mpu.pp_size > 1:
                # 只有pp的dict好像是有指导作用的？
                return_dict = {**return_dict, **{
                    'recv_backward': ['send_backward', 1],
                    'recv_forward': ['send_forward', -1],
                    'send_forward': ['recv_forward', 1],
                    'send_backward': ['recv_backward', -1]
                }}
            # TODO：是否会出错？
            if self.mpu.dp_size >= 1:
                return_dict = {**return_dict, **{
                    'dp_allreduce': ['dp_allreduce', 'dp_group'],
                    'dp_reducescatter': ['dp_reducescatter', 'dp_group'],
                }}

            if self.mpu.tp_size > 1:
                # tp_load_batch_broadcast 暂时没用，根据逻辑区分即可
                return_dict = {**return_dict, **{
                    'tp_load_batch_broadcast': ['tp_load_batch_broadcast','tp_group'],
                    'tp_broadcast': ['tp_broadcast', 'tp_group'],
                    'tp_all_to_all': ['tp_all_to_all', 'tp_group'],
                    'tp_allgather': ['tp_allgather', 'tp_group'],
                    'tp_reduce_scatter': ['tp_reduce_scatter', 'tp_group'],
                    'tp_allreduce': ['tp_allreduce', 'tp_group'],
                    'tp_reducescatter': ['tp_reducescatter', 'tp_group'],
                }}

            if self.mpu.ep_size > 1:
                return_dict = {**return_dict, **{
                    'ep_allreduce': ['ep_allreduce', 'ep_group']
                }}

            if self.mpu.exp_size > 1:
                return_dict = {**return_dict, **{
                    'exp_allgather': ['exp_allgather', 'exp_group'],
                    'exp_all_to_all': ['exp_all_to_all', 'exp_group'],
                    'exp_reducescatter': ['exp_reducescatter', 'exp_group'],
                }}

            if getattr(self.mpu, "cp_size", 1) > 1:
                return_dict = {**return_dict, **{
                    'cp_reducescatter': ['cp_reducescatter', 'cp_group'],
                }}

            if self.mpu.pep_size > 1:
                return_dict = {**return_dict, **{
                    'pep_allreduce': ['pep_allreduce', 'pep_group']
                }}

            # MoE模式下的expert data parallel allreduce
            if hasattr(self.mpu, 'dp_modulo_exp_groups') and self.mpu.dp_modulo_exp_groups is not None:
                return_dict = {**return_dict, **{
                    'exp_dp_allreduce': ['exp_dp_allreduce', 'dp_modulo_exp_group']
                }}

            return return_dict
        else:
            raise ValueError(f"The framework '{framwork}' is not supported.")

    @staticmethod
    def stages_task_init(stages_steps_dict) -> list:
        """初始化stages的task,实例化Operation objs、Stage objs
        
        Args:
            stages_steps_dict: {"wrank_id": {"step_id": [cmds]}, ...} # steps_num = len(stages_steps_dict["wrank_id"])

            return stages_list: [Stage obj, ...]
        """
        pass

    @staticmethod
    def v1_ds_handle_tmp_stages_dataset(filename: str, rank_instances_dict: dict):
        stages_dict = {}
        def parse_commands(cmds_str):
            # A simple parser to extract command names and arguments from the command strings
            cmds = []
            default_params = {'buffer_id': None, 'batch_id': None}  # Default parameters
            if cmds_str.startswith('[') and cmds_str.endswith(']'):
                cmds_str = cmds_str[1:-1].strip()
                if cmds_str:
                    # Split the commands by "), " assuming that no nested functions calls exist
                    cmd_parts = cmds_str.split('), ')
                    for part in cmd_parts:
                        if ')' not in part:
                            part += ')'
                        cmd_name, arg_str = part.split('(', 1)
                        arg_str = arg_str[:-1]  # Remove the closing ')'
                        kwargs = default_params.copy()  # Start with default parameters
                        if arg_str:
                            for arg in arg_str.split(', '):
                                key, value = arg.split('=')
                                # Convert value to the appropriate type
                                if value.isdigit():  # If the value is a digit, convert it to an integer
                                    value = int(value)
                                kwargs[key] = value
                        cmds.append((cmd_name, kwargs))
            return cmds

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split('_cmds:')
                    stage_step_part = parts[0] if parts else ''
                    cmds_str = parts[1].strip() if len(parts) > 1 else "[]"

                    try:
                        stage_prefix, step_id_prefix = stage_step_part.split('_step_id:')
                        wrank_id = int(stage_prefix.split(':')[-1])
                        step_id = int(step_id_prefix.split(':')[-1])
                    except ValueError as e:
                        print(f"Error parsing line: {line} | Error: {str(e)}")
                        continue

                    cmds = parse_commands(cmds_str)

                    if wrank_id not in stages_dict:
                        stages_dict[wrank_id] = Stage(wrank_id, 0, "deepspeed")

                    for cmd_name, kwargs in cmds:
                        kwargs.update({"wrank_id": wrank_id, "step_id": step_id, "duration": 1})
                        operation = Operation(name=cmd_name, **kwargs)
                        stages_dict[wrank_id].add_operations_to_list(operation)

                    stages_dict[wrank_id].steps_num = max(stages_dict[wrank_id].steps_num, step_id + 1)

        for i, stage in stages_dict.items():
            if i == 0:
                stage.set_stage_kind('FirstStage')
            elif i == max(stages_dict.keys()):
                stage.set_stage_kind('LastStage')
            else:
                stage.set_stage_kind('MiddleStage')

        return list(stages_dict.values()), stages_dict

    # v2_ds_handle_tmp_stages_dataset
    @staticmethod
    def generate_stages_and_cmds_info_from_datasets_and_schedules(trace_filepath: str, rank_instances_dict: dict, running_mode=None, 
                                                                  framework=None, torchgraph_filepath=None, mpu=None, stages_scheduling_filepath=None):
        """
        trace_filepath: realistic trace from trainning in 3D parallel in DS/MG framwork
        torchgraph_filepath: model operations runnnig database and graph files
        stages_scheduling_filepath: stages scheduling plans
        """

        # 获取torchgraph的算子执行时间并根据dp情况进行stage阶段id和rank id的映射处理、
        single_gpu_load_start_1 = time.time()
        if framework == "deepspeed":
            torch_graph_stage_op_dict = get_tmp_ds_simu_torchgraph_op_dict(torchgraph_filepath)
        elif framework == "megatron-lm":
            torch_graph_stage_op_dict = get_tmp_mg_simu_torchgraph_op_dict(torchgraph_filepath, mpu, rank_instances_dict)
        end_time = time.time()
        single_gpu_load_1 = end_time - single_gpu_load_start_1

        if running_mode == MODE_PROFILE:
            stages_dict = process_trace_or_scheduling_files(my_filepath=trace_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                 torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_trace=True, framework=framework)
            trace_stages_dict = stages_dict
        elif running_mode == MODE_MODEL:
            if trace_filepath is not None:
                stages_dict = process_trace_or_scheduling_files(my_filepath=trace_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                    torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_trace=True, framework=framework)
                trace_stages_dict = stages_dict
            else:
                stages_dict = process_trace_or_scheduling_files(my_filepath=stages_scheduling_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                            torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_trace=False, framework=framework)
                trace_stages_dict = None
        elif running_mode == MODE_SIMULATE:
            if trace_filepath is not None:
                trace_stages_dict = process_trace_or_scheduling_files(my_filepath=trace_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                        torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_trace=True, framework=framework)
            else:
                trace_stages_dict = None
            single_gpu_load_start_2 = time.time()
            stages_dict = process_trace_or_scheduling_files(my_filepath=stages_scheduling_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                    torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_trace=False, framework=framework)
            end_time = time.time()
            single_gpu_load_2 = end_time - single_gpu_load_start_2
        else:
            raise ValueError(f"Invalid running mode: {running_mode}")
        # stages_dict = process_trace_or_scheduling_files(stages_scheduling_filepath, mpu, rank_instances_dict, running_mode, torch_graph_stage_op_dict)

        # sim_load_time = single_gpu_load_1 + single_gpu_load_2
        return list(stages_dict.values()), stages_dict, torch_graph_stage_op_dict, trace_stages_dict

    @staticmethod
    def v1_ds_op_compare_simu_with_trace(folder_path_ds_trace: str, folder_path_torchgraph: str):
        """ 用于ds架构: 比较各个stage中的operation的实际执行时间和模拟执行时间之间的差距"""
        import os
        # 获取torchgraph的算子执行时间
        torch_graph_stage_op_dict = get_tmp_ds_simu_torchgraph_op_dict(folder_path_torchgraph)
        compare_operation_dict = {k: {k2: [] for k2 in v} for k, v in torch_graph_stage_op_dict.items()}

        def parse_commands(cmds_str):
            cmds = []
            default_params = {'buffer_id': None, 'batch_id': None, 'duration': None, 'fbd_time':None, 'pure_time':None, 'param_bytes':None}
            if cmds_str.startswith('[') and cmds_str.endswith(']'):
                cmds_str = cmds_str[1:-1].strip()
                if cmds_str:
                    # Split the commands by "), " assuming that no nested functions calls exist
                    cmd_parts = cmds_str.split('), ')
                    for part in cmd_parts:
                        if ')' not in part:
                            part += ')'
                        cmd_name, arg_str = part.split('(', 1)
                        arg_str = arg_str[:-1]
                        kwargs = default_params.copy() 
                        if arg_str:
                            for arg in arg_str.split(', '):
                                key, value = arg.split('=')

                                if value.isdigit():
                                    value = int(value)
                                kwargs[key] = value
                        cmds.append((cmd_name, kwargs))
            return cmds

        for filename in os.listdir(folder_path_ds_trace):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path_ds_trace, filename), 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            stage_step_part, cmds_str = line.split('_cmds:')
                            wrank_id_str, step_id_str = stage_step_part.split('_step_id:')
                            wrank_id = int(wrank_id_str.split(':')[-1])

                            cmds = parse_commands(cmds_str.strip())
                            for cmd_name, kwargs in cmds:
                                duration_from_simulator = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, 
                                                                                        cmd_name, wrank_id)
                                duration_from_trace = float(kwargs.get('duration', '-1'))

                                if duration_from_simulator is not None and duration_from_trace >= 0:
                                    error =  duration_from_simulator - duration_from_trace
                                    compare_operation_dict[wrank_id][cmd_name].append(error)
        # 打印误差信息并计算绝对误差之和
        for stages, cmd_type in compare_operation_dict.items():
            print()
            print(f"********************stage:{stages}********************")
            for cmd_name, error_list in cmd_type.items():
                abs_error_sum = sum(abs(error) for error in error_list)
                print(f"Operation:{cmd_name} | Absolute Error Sum: {abs_error_sum} | Errors List: {error_list}")



    # @staticmethod
    # def mg_handle_tmp_stages_dataset(folder_path: str, rank_instances_dict=None, running_mode=None, is_trace=None):
    #     """
    #     Process all txt files in the given folder path, each representing operations in a stage.
    #     Each line in a txt file represents an operation in the format:
    #     `wrank_id:mg_state:operation_name:batch_id`    #     Returns both a list of Stage objects and a dictionary mapping wrank_ids to Stage objects.
    #     """

    #     # stages = {}
    #     stages_dict = {}
    #     # Read each file in the folder
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(".txt"):
    #             with open(os.path.join(folder_path, filename), 'r') as file:
    #                 lines = file.readlines()
    #                 # Process each line in the file
    #                 for line in lines:
    #                     line = line.strip()
    #                     if line:
    #                         # stage:0:warmup:forward_step:0
    #                         # wrank:0:warmup:forward_step:0
    #                         # stage:0:forward_step(batch_id=0, mg_state=xxx, duration=xxx, description=xx)
    #                         stage_rank_signal, stage_or_wrank_id, cmd = line.split(':')
    #                         stage_or_wrank_id = int(stage_or_wrank_id)

    #                         if stage_rank_signal == "stage" and not is_trace:
    #                             stage_id = stage_or_wrank_id
    #                             wrank_id = None
    #                             rank = None
    #                         elif stage_rank_signal == "rank" and is_trace:
    #                             wrank_id = stage_or_wrank_id
    #                             rank = rank_instances_dict[wrank_id]
    #                             stage_id = rank_instances_dict[wrank_id]._get_pp_local_rank()
    #                         else:
    #                             raise ValueError(f"Error: signal is invalid, please check the trace file...")
                            
    #                         if stage_or_wrank_id not in stages_dict:
    #                             stages_dict[stage_or_wrank_id] = Stage(
    #                                 wrank_id=wrank_id,
    #                                 stage_id=stage_id,
    #                                 rank=rank,
    #                                 framework="megatron-lm"
    #                             )

    #                         cmd_name, kwargs = parse_megatron_cmd(cmd)
    #                         batch_id = int(kwargs.get('batch_id', 0))
    #                         mg_state = kwargs.get('mg_state', None)
    #                         description = kwargs.get('description', None)
    #                         # duration = float(kwargs.get('duration', 1))
    #                         op_kind = "comp" if cmd_name in MG_COMP_OPERATION else "comm" 
    #                         if description is not None:
    #                             description = re.sub(r'\W+', '', description)

    #                         if running_mode == MODE_PROFILE:
    #                             duration = kwargs.get('duration', None)
    #                         elif running_mode == MODE_SIMULATE:
    #                             if is_trace:
    #                                 duration = kwargs.get('duration', None)
    #                             else:
    #                                 duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id, batch_id=batch_id)
    #                         elif running_mode == MODE_MODEL: 
    #                             duration = 1
    #                         duration = round(float(duration), 2) if duration is not None else None


    #                         operation = Operation(
    #                             name=cmd_name,
    #                             duration=duration,
    #                             batch_id=batch_id,
    #                             wrank_id=wrank_id,
    #                             mg_state=mg_state,
    #                             op_kind=op_kind
    #                         )

    #                         stages_dict[stage_or_wrank_id].add_operations_to_list(operation)

    #     return stages_dict


    def start_running(self):
        if self.mpu is not None:
            self.timeline_manager.start_pop_operators()
        else:
            raise ValueError(f"mpu_info is not set.")

    def visualize_timelines(self, wrank_id_start_end:list, specific_ranks_list=None, show_x_lim:int=None, 
                            save_plot=True, output_dir="./log/visualization_outputs", show_gui=False):
        self.timeline_manager.visualize_timelines(self.running_mode, self.mpu, wrank_id_start_end, 
                                                specific_ranks_list, show_x_lim, save_plot, output_dir, show_gui)

    def check_error_inference(self):
        self.timeline_manager.check_error_inference()


    def ds_op_compare_simu_with_trace(self):
        self.timeline_manager.ds_op_compare_simu_with_trace()

    def get_global_operation_error(self):
        if self.running_mode == MODE_SIMULATE:
            self.timeline_manager.get_global_operation_error()

    def get_op_json_db_record(self, file_name):
        if self.running_mode == MODE_SIMULATE:
            self.timeline_manager.get_op_json_db_record(file_name)



if __name__ == '__main__':
    pass

    """ 测试torch graph得到的 model的fwd and bwd time"""
    # operation_dict = get_tmp_ds_simu_torchgraph_op_dict(r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0519_3pp\graph_and_database')
    # print(f"tmp_ds_simu_torchgraph_duration: {operation_dict}")

    # """ 已废弃使用(无法比较P2P):测试 ds trace中的operation的执行时间 与 simulator 模拟的执行时间的误差 """
    # folder_path_torchgraph = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0520_8pp_params\graph_and_database'
    # folder_path_ds_trace = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0520_8pp_params\stages"
    # SimulatorEngine.v1_ds_op_compare_simu_with_trace(folder_path_ds_trace=folder_path_ds_trace, folder_path_torchgraph=folder_path_torchgraph)

    # """ 初始化mpu设定, 新版本中必须需要的信息 """
    # from parallel_group_manager import ParallelGroupManager

    # manager = ParallelGroupManager(local_size=8, world_size=8, pp_size=8, tp_size=1)
    # mpu_info = manager._get_mpu_info()
    # print(mpu_info)

    # """ 单独测试ds_handle_tmp_stages_dataset函数 """
    # # stages_list, stages_dict = SimulatorEngine.ds_handle_tmp_stages_dataset(r'H:\HUBOther\ML_Sys_Merak\TorchGraph\log_pp_txt\2pp.txt')
    # # print(stages_list)

    # """ 单独测试mg_handle_tmp_stages_dataset函数 """
    # # stages_list, stages_dict = SimulatorEngine.mg_handle_tmp_stages_dataset(r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log")
    # # print(stages_list, stages_dict)


    # """ 测试SimulatorEngine类 | deepspeed"""
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\log_pp_txt\8pp.txt"
    # # simulator_engine = SimulatorEngine(tmp_filename=filename, framwork='deepspeed', strategy="1F1B-none_interleaved")
    # # simulator_engine._start_pipeline()
    # # simulator_engine.visualize_timelines()


    # """ 测试SimulatorEngine类 | megatron-lm"""
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\8pp_1_1"
    # simulator_engine = SimulatorEngine(tmp_filename=filename, framwork='megatron-lm', strategy="1F1B-none_interleaved")
    # simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)
    # simulator_engine._init_tmp_stages_dataset_and_timeline_manager()
    # simulator_engine._start_pipeline()
    # simulator_engine.visualize_timelines()
