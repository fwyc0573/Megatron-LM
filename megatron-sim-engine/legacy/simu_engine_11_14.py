import copy
import os
import re
# import queue
from collections import deque
from typing import Union
import ast
import time
import random
# from deepspeed.utils import logger
from StaticGraphs.rank_manager import RankZoo
from StaticGraphs.parallel_group_manager import MPUInfo
from comm_sim.nccl_comm import get_comm_op_exc_time

DS_COMP_OPERATION = ['ForwardPass', 'BackwardPass', 'OptimizerStep', 'LoadMicroBatch']
DS_COMM_OPERATION = ['SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','ReduceGrads', 'ReduceTiedGrads']
DS_FINAL_OPERATION = ['ReduceGrads', 'ReduceTiedGrads', 'OptimizerStep']


# MG_COMP_OPERATION = ['forward_step', 'backward_step', 'load_batch']
MG_COMP_OPERATION = ['forward_step', 'backward_step','optimizer_step', 'get_batch', 'loss_func']
MG_COMM_OPERATION = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward', 'tp_allreduce', 'tp_load_batch_broadcast', 'dp_allreduce', 'ep_allreduce']


P2P_COMM_COLLECTIVE = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward',
                        'SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation']
ALLREDUCE_COMM_COLLECTIVE = ['dp_allreduce', 'tp_allreduce', 'ep_allreduce', 'ReduceGrads', 'ReduceTiedGrads']

# DS可以被模拟以及未支持模拟的op
GLOBAL_DS_DIRECT_MAPPING_LIST = ['ForwardPass', 'BackwardPass', 'SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','OptimizerStep']
GLOBAL_DS_NOT_SUPPORTED_LIST = ['LoadMicroBatch','ReduceGrads', 'ReduceTiedGrads']


# 被支持的OP可以从SINGLE_GPU_PROFILE data中获取duration等信息；该部分只适用于TP=1时，通过profile的算子是否可以直接用来模拟真实场景的算子(像'loss_func'就不应该支持,因为涉及了通信操作)
# TODO: tp_load_batch_broadcast还要吗;
GLOBAL_NETWORK_ESTIMATOR_GET_LIST = ['dp_allreduce', 'ep_allreduce', 'tp_allreduce', 'send_backward', 'recv_backward', 'send_forward', 'recv_forward']
# GLOBAL_MG_DIRECT_MAPPING_LIST = ['forward_step', 'backward_step', 'optimizer_step', *GLOBAL_NETWORK_ESTIMATOR_GET_LIST]
GLOBAL_MG_DIRECT_MAPPING_LIST = ['forward_step', 'backward_step', *GLOBAL_NETWORK_ESTIMATOR_GET_LIST]
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
GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST = ['forward_step', 'backward_step', 'loss_func', 'dp_allreduce', 'ep_allreduce', 'tp_allreduce','optimizer_step'] # , 'optimizer_step'

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
                 waiting_acc=None, description=None, group_kind=None,end_timestamp=None,hidden_duration=None, tensor_shape=None, tensor_dtype=None):
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
        self.mg_state = mg_state # warmup/steady/cooldown/help
        self.group_kind = group_kind
        self.mg_is_last_iteration = False
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.ds_buffer_id = buffer_id
        self.ds_step_id = step_id
        self.description = description # 给dp类allreduce算子增加的描述用于相互区分
        self.end_timestamp = end_timestamp
        self.hidden_duration = hidden_duration
    
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
        self.waiting_time = round(waiting_time,2)
        self.finish_time = round(self.join_time + waiting_time + self.duration,2)
        # self.finish_time = self.join_time + self.duration

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
        self.duration = round(float(duration),2)

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
            "hidden_duration": self.hidden_duration
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
            return last_operation.waiting_acc if last_operation else 0
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
                  trace_filepath=None, torchgraph_filepath=None,not_simulating_cmd_dict=None,rank_instances_dict=None,trace_stages_dict=None):
        
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
        self.stages_timeline_process_dict = self._init_stages_timeline()

    def _init_stages_timeline(self):
        stages_timeline_process_dict = {}
        # Create IndividualTimeline objects for all stages
        for Stage in self.compelete_wranks_list:
            timeline = IndividualTimeline(stage=Stage, can_overlap=self.can_overlap)
            stages_timeline_process_dict[Stage.wrank_id] = timeline
        
        # Set the pre and post individual timelines based on the PP groups
        for pp_group in self.mpu_info.pp_groups:
            for index, wrank_id in enumerate(pp_group):
                timeline = stages_timeline_process_dict[wrank_id]
                if index == 0:
                    timeline.stage_kind = 'FirstStage'
                elif index == len(pp_group) - 1:
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
        if self.strategy == '1F1B-none_interleaved':
            # 当self.global_waiting_queue不为空,继续循环
            # 1. 从global_waiting_queue中pop第一个operation
            # 2. 检查operation的依赖operation是否已被完成（依赖关系可以从dependency_relationship中找到,被完成指的是不存在于global_waiting_queue中）,
            #    如果已经完成,将operation加入到对应stage的timeline中（依据操作的类型,是comp还是comm）。在加入过程中,如果该operation有依赖操作,查找依赖操作的finish_time,然后max(finish_time, 当前类型timeline最后一个operation的finish_time)
            #    如果没有完成,put回队列,continue到下一个operation
            while True:
                # 初始化一个标志,用于检查所有的waiting_queue是否都为空
                all_queues_empty = True

                # 对wrank_id进行排序
                # 每个IndividualTimeline对应一个rank：逐个地将waiting_queue中的op pop尝试加入到timelien中
                for wrank_id in sorted(self.stages_timeline_process_dict.keys()):
                    timeline: IndividualTimeline  = self.stages_timeline_process_dict[wrank_id]

                    if timeline.waiting_queue:
                        all_queues_empty = False

                        # 检查当前timeline是否blocked
                        operation: Operation = timeline.waiting_queue.popleft()
                        # if operation.name == "optimizer_step":
                        #     continue

                        is_timeline_blocked: bool = self._check_timelline_blocked_status(timeline=timeline, operation=operation)

                        if is_timeline_blocked:
                            # 返回队首等待
                            timeline.waiting_queue.appendleft(operation)
                        else:
                            self._add_operation_to_timeline(timeline=timeline, operation=operation)

                # 如果所有的waiting_queue都为空,退出循环
                if all_queues_empty:
                    break

        elif self.strategy == '1F1B-interleaved':
            pass

        elif self.strategy == 'F-then-B':
            pass

        elif self.strategy == 'no-pipelining':
            pass
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

        elif operation.op_kind == "comm":
            '''comm. op 中 p2p(1 to 1) allreduce(n to n) broadcast(1 to n) 的处理逻辑都不一致, 因此需要单独分类处理

                这里要区别megatron中通信算子合并情况,仅出现在【steady阶段】,send_forward_recv_backward 和 send_backward_recv_forward 是一起出现的,因此注册和查找都是同时进行的
                注意,send_forward_recv_backward 和 send_backward_recv_forward 互为matching operaitons,需要相互配合才能执行。谁先达到谁先注册,等待对方到达进行收尾操作
                可能的情况：
                    1. 当前ops未注册, 配对的comm. ops 已经注册,更新当前ops的join_time,当前对象执行收尾操作,更新当前ops的属性,从配对的comm_waiting_pool取出该配对的comm op（根据pre/post_timeline）,然后刷新各类型属性,最后加入timeline和global_finished_operations
                    2. 当前ops未注册, 配对的comm. ops 未注册,更新当前ops的join_time, 将当前ops加入到comm_waiting_pool中等对方查询, 等待对方对象执行收尾操作
            '''
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

            elif parallel_kind == "dp" or parallel_kind == "ep" or parallel_kind == "tp":
                # TODO：修正下这个写法，不太美观
                if comm_kind == "allreduce": # if "allreduce" in operation.name:
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

    def _can_start_matching_comm_group_process(self, operation_list: list, comm_kind: str, parallel_kind: str, timeline: IndividualTimeline) -> bool:
        """ 生成operation_list中的operation的formatname, 检查global_waiting_pool中是否包含这些operation, 如果有不存在的, 返回False"""
        # TODO: 对于allreduce一定能在global pool中找到正确的matching op/subop吗？(现在是根据_get_format_operation_name()确认的)
        matching_operation_format_name_list = []
        matching_operation_list = []

        # 生成comm group的formatname list
        for operation in operation_list:
            matching_operation_name, matching_operation_wrank_id_list, _ = self._get_comm_matching_operation_name_and_wrank_id(operation=operation, 
                                                                                comm_kind=comm_kind, parallel_kind=parallel_kind, timeline=timeline)
            for matching_operation_wrank_id in matching_operation_wrank_id_list:
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

        calcu_operation = current_operation
        if comm_kind == "allreduce":
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
            
        elif comm_kind == "allreduce":
            # 最晚注册的不一定join_time是最晚的,因此在matching_list中找到最晚的并计算返回
            return_value = calcu_operation.join_time - current_operation.join_time
            assert return_value >= 0

            return return_value
            
        elif comm_kind == "broadcast":
            raise 0
        
        else:
            raise ValueError(f"Invalid comm_kind name: {comm_kind}")
        

    def _calculate_comm_duration(self, comm_op_list):
        """ 进行NCCL comm, 更新comm op的duration """
        pass


    def _get_comm_matching_operation_name_and_wrank_id(self, operation: Operation, comm_kind: str, parallel_kind: str, timeline: IndividualTimeline):
        """
            返回matching的op_name, wrank_id or wrank_id list
        """
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
            if comm_kind == "allreduce":
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.tp_groups if rank_id != timeline.wrank_id], None
            elif comm_kind == "broadcast":
                 raise ValueError(f"broadcast is not supported...")
            else:
                raise ValueError(f"Invalid comm_kind name: {comm_kind}")
        
        elif parallel_kind == "dp": # and comm_kind == "allreduce":
            if comm_kind == "allreduce":
                # 获取 dp group wrank_id list
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.dp_groups if rank_id != timeline.wrank_id], None
        elif parallel_kind == "ep":
            if comm_kind == "allreduce":
                # 获取 ep group wrank_id list
                # print(f"self.comm_matching_relationship[operation.name]:{self.comm_matching_relationship[operation.name]}")
                # print(f"self.comm_matching_relationship[operation.name][0]:{self.comm_matching_relationship[operation.name][0]}")
                # print(f"timeline.stage_rank.ep_groups:{timeline.stage_rank.ep_groups}")
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.ep_groups if rank_id != timeline.wrank_id], None

        else:
            raise ValueError(f"Invalid comm_kind name: {comm_kind} and parallel_kind name: {parallel_kind}")


    # TODO: 判定方式修改一下，可以根据name直接拆分
    def _get_comm_operation_kind_and_parallel_dimension(self, operation: Operation):
        """ return: comm_kind, parallel_kind"""
        # print(f"operation.group_kind:{operation.group_kind}")
        # TODO: 修改为根据group_kind来确定parallel_kind
        # TODO： 还要区分出不同的通信原语的区别
        assert operation.group_kind and operation.name, "Invalid group kind"

        if operation.mg_state == "steady" and operation.name in P2P_COMM_COLLECTIVE:
            return "p2p_fused", operation.group_kind # 'pp'
        elif operation.mg_state != "steady" and operation.name in P2P_COMM_COLLECTIVE:
            return "p2p", operation.group_kind # 'pp'
        elif operation.name in ALLREDUCE_COMM_COLLECTIVE:
            # if "tp" in operation.name:
            #     return "allreduce", "tp"
            # elif "dp" in operation.name or operation.name in DS_COMM_OPERATION:
            #     return "allreduce", "dp"
            # else:
            #     raise ValueError(f"Invalid allreduce operation name: {operation.name}")
            return "allreduce", operation.group_kind
        elif "broadcast" in operation.name and "tp" in operation.name:
            return "broadcast", operation.group_kind # "tp"
        else:
            raise ValueError(f"Invalid comm operation name: {operation.name}")


    def _check_timelline_blocked_status(self, timeline:IndividualTimeline, operation:Operation)->bool:
        """ 根据overlap情况来判定当前timeline是否被阻塞
            返回True表示阻塞,返回False表示不阻塞
        """
        # 如果不允许overlap且operation是comp类型，或者operation是comm类型，那么检查timeline的is_comm_blocked状态
        # 其他情况返回False，表示不阻塞
        # TODO: 如何处理overlap
        return (not self.can_overlap and operation.op_kind == "comp" or operation.op_kind == "comm") and timeline.is_comm_blocked


    # def _get_format_operation_name(self, wrank_id:int, operation_name: str, stage_offset: int, batch_id: int) -> str:
    #     return str(wrank_id+stage_offset) + "_" + operation_name + "_" + str(batch_id)


    def _get_format_operation_name(self, wrank_id:int, operation: Operation, batch_id: int, description: str, matching_op_name:str=None) -> str:
        
        opration_name =  operation.name
        if matching_op_name:
            opration_name = matching_op_name
        
        # 暂时只考虑mg的subop中allreduce类型需要特定的format_name对应
        if opration_name in ['dp_allreduce', 'ep_allreduce', 'tp_allreduce'] and operation.op_kind == "comm":
            if isinstance(operation, SubOperation):
                # 对于某个需要sync的allreduce subop, 它们所在的OP的序号(即这是第几次该OP)和subop的序号是一致的
                print(f"operation:{operation}")
                # raise 0
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

    def visualize_timelines(self, running_mode, mpu, wrank_id_start_end=[1,100], specific_ranks_list=None, show_x_lim=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk
        from tkinter import ttk

        wrank_id_start, wrank_id_end = wrank_id_start_end[0], wrank_id_start_end[1]

        # Validate wrank_id range
        if specific_ranks_list is None:
            if wrank_id_end - wrank_id_start + 1 > 200:
                raise ValueError("The number of wrank_ids to visualize is too large. Please set a range of 200 or fewer.")

        # Filter the stages_timeline_process_dict based on the given range
        # filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id_start <= wrank_id <= wrank_id_end}
        
        # num_stages = len(filtered_stages)

        # filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id_start <= wrank_id <= wrank_id_end}
        
        if specific_ranks_list:
            filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id in specific_ranks_list}
        else:
            filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id_start <= wrank_id <= wrank_id_end}

        num_stages = len(filtered_stages)

        for wrank_id, timeline in filtered_stages.items():
            # Initialize accumulators for the current rank
            comp_timeline_time = 0
            comm_timeline_time = 0
            load_microbatch_time = 0

            # Calculate comp_timeline times
            for op in timeline.comp_timeline:
                # print(f"op:{op}")
                duration = op.finish_time - op.join_time
                comp_timeline_time += duration
                if op.name == 'get_batch':
                    load_microbatch_time += duration
            # raise 0

            # Calculate comm_timeline times
            for op in timeline.comm_timeline:
                duration = op.finish_time - op.join_time
                comm_timeline_time += duration

            comp_comm_sum_time = comp_timeline_time + comm_timeline_time
            # Output the results for this rank
            print(f"{running_mode}, rank{wrank_id} comp_time: {comp_timeline_time:.2f} ms / {(comp_timeline_time-load_microbatch_time):.2f}")
            print(f"{running_mode}, rank{wrank_id} comm_time: {comm_timeline_time:.2f} ms")
            # print(f"{running_mode}, rank{wrank_id} get_batch:{load_microbatch_time:.2f} ms")
            print(f"{running_mode}, rank{wrank_id} sum_time: {comp_comm_sum_time:.2f} ms")
        # raise 0


        # Create the main window
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

        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=num_stages, ncols=1, figsize=(10, num_stages * 2 + 1), squeeze=False)

        # Add a title to the figure
        # plt.suptitle(f"{running_mode} - PP{mpu.pp_size} - TP{mpu.tp_size} - DP{mpu.dp_size}")

        # Display the not_simulating_cmd_dict contents
        self.not_simulating_cmd_dict.pop('get_batch', None)
        # self.not_simulating_cmd_dict.pop('optimizer_step', None)
        not_supported_ops = ", ".join(self.not_simulating_cmd_dict.keys())
        if len(self.not_simulating_cmd_dict.keys()) != 0:
            not_supproted_string = f"Operations not yet supported: {not_supported_ops}"
        else:
            not_supproted_string = f"All operations are supported now."
        # TODO:change
        not_supproted_string = f"All operations are supported now."
        annotation_text = f"{running_mode} - PP{mpu.pp_size} - TP{mpu.tp_size} - DP{mpu.dp_size} \n {not_supproted_string}"
        plt.figtext(0.5, 0.98, annotation_text, ha='center', fontsize=10)

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
            'tp_load_batch_broadcast': 'BDC', 'tp_allreduce': 'tAR', 'dp_allreduce': 'dAR','ep_allreduce': 'eAR',
            'optimizer_step': 'OS', 'get_batch': "GB", "fwd_comp": "fc", "bwd_comp": "bc",
        }

        # Define special operations that require red coloring
        red_operations = {'tp_load_batch_broadcast', 'dp_allreduce', 'tp_allreduce', 'ReduceTiedGrads', 'ReduceGrads', 'ep_allreduce'}

        patches_dict = []

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
                # Store the patch and its properties for event handling
                patches_dict.append((rect, op.name, duration, op.join_time, op.finish_time, op.mg_state, op.stage_id, op.description, ax))
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

            # Add total time text
            if all_operations:
                total_time = round((all_operations[-1].finish_time), 2)
                ax.text(max_finish_time / 2, 1.05, f"total_time = {total_time}ms", horizontalalignment='center')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Render the figure onto the tkinter canvas
        canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add the toolbar for zoom and pan functionalities
        toolbar_frame = ttk.Frame(scrollable_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas_widget, toolbar_frame)
        toolbar.update()

        # Create an annotation for displaying operation info for each axis
        annots = []
        for ax in axs:
            annot = ax[0].annotate("", xy=(0,0), xytext=(20,20),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            annots.append(annot)

        def update_annot(rect, name, duration, join_time, finish_time, mg_state, stage_id, description, annot):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            annot.xy = (x, y)
            # text = f"Name: {name}\nDuration: {duration:.2f}\nJoin Time: {join_time:.2f}\nFinish Time: {finish_time:.2f}"
            text = f"Name: {name}\nDuration: {duration:.2f}\nStage ID: {stage_id}\nJoin Time: {join_time:.2f}\nFinish Time: {finish_time:.2f}\nMG State: {mg_state}\nDescription: {description}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            if not hover_active:
                return
            vis = any(annot.get_visible() for annot in annots)
            for rect, name, duration, join_time, finish_time, mg_state, stage_id, description, ax in patches_dict:
                cont, _ = rect.contains(event)
                if cont:
                    update_annot(rect, name, duration, join_time, finish_time, mg_state, stage_id, description, \
                                 annots[axs.tolist().index([ax])])
                    annots[axs.tolist().index([ax])].set_visible(True)
                    fig.canvas.draw_idle()
                    return
            if vis:
                for annot in annots:
                    annot.set_visible(False)
                fig.canvas.draw_idle()

        hover_active = False

        def toggle_hover():
            nonlocal hover_active
            hover_active = not hover_active
            button_text.set("Disable Hover" if hover_active else "Enable Hover")

        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X)
        button_text = tk.StringVar()
        button_text.set("Enable Hover")
        hover_button = ttk.Button(button_frame, textvariable=button_text, command=toggle_hover)
        hover_button.pack(side=tk.TOP)

        fig.canvas.mpl_connect("motion_notify_event", hover)

        root.mainloop()


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
        record_json_path = f"H:/HUBOther/ML_Sys_Merak/TorchGraph/megatron_operation_log/analysis_log/{file_name}.json"

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
        trace_dict_path = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\analysis_log\trace_dict.txt"
        comp_comm_dict_path = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\analysis_log\comp_comm_dict.txt"

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
                # record_cmd_times_dict = {}
                for line in lines:
                    line = line.strip()
                    if line:
                        stage_rank_signal, stage_or_wrank_id, cmd = line.split(':', 2)
                        wrank_id = int(stage_or_wrank_id)
                        cmd_name, kwargs = parse_megatron_cmd(cmd)
                        duration = kwargs.get('duration', None)
                        timestamp = float(kwargs.get('timestamp', 0))
                        start_time = timestamp - float(duration)
                        sub_ops_list = None

                        # if mpu.dp_size == 1 and cmd_name == "dp_allreduce":
                        #     # dp_size>1时才加入dp_allreduce op (grad部分的allreduce)
                        #     continue

                        # TODO: tp>1 才读取sub_operations信息(后续该根据什么来决定是否读取)
                        # break OP to SUBOP 是因为OP中额外包含了comm部分，因此需要break，否则作为整体即可
                        if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                            if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                                        (mpu.dp_size > 1 and cmd_name in ['loss_func']):
                                sub_operations_str = kwargs.get('sub_operations', '[]')
                                sub_ops_list = extract_sub_operations(sub_operations_str_list=sub_operations_str,pt_start_time=start_time,\
                                                                    pt_end_time=timestamp,pt_duration=duration,wrank_id=wrank_id,is_muti_gpus_trace=False,\
                                                                    rank_instance=rank_instances_dict[wrank_id])
                        # if mpu.tp_size > 1:
                        #     sub_operations_str = kwargs.get('sub_operations', '[]')
                        #     sub_ops_list = extract_sub_operations(sub_operations_str_list=sub_operations_str,pt_start_time=start_time,\
                        #                                         pt_end_time=timestamp,pt_duration=duration,wrank_id=wrank_id,is_muti_gpus_trace=False,\
                        #                                         rank_instance=rank_instances_dict[wrank_id])

                        if wrank_id not in torch_graph_op_excution_record_dict:
                            torch_graph_op_excution_record_dict[wrank_id] = {}

                        # if not sub_ops_list:
                        #     torch_graph_op_excution_record_dict[wrank_id][cmd_name] = duration
                        # else:
                        #     torch_graph_op_excution_record_dict[wrank_id][cmd_name] = {'duration': duration, 'sub_ops_list': sub_ops_list}

                        # sub_ops_list中的comm op已经包含了estimated的duration
                        # dp_allreduce还未赋值,其shape和dtype信息被记录了,在init_3d部分处理

                        # TODO: 额外单独处理dp_allreduce和ep_allreduce
                        # TODO： if overlap:
                            # 此时dp应该有subop？
                        if cmd_name == "dp_allreduce" or cmd_name == "ep_allreduce":
                            tensor_shape = kwargs.get('input__shape', None)
                            tenosr_dtype = kwargs.get('input__dtype', None)
                            group_kind = kwargs.get('group_kind', None)
                            if group_kind == "dp":
                                comm_group = rank_instances_dict[wrank_id].dp_groups
                            else:
                                comm_group = rank_instances_dict[wrank_id].ep_groups
                            duration = get_comm_op_exc_time(
                                comm_group=comm_group,
                                data_size=get_tensor_data_size(tensor_shape,tenosr_dtype),
                                comm_func="allreduce"
                            )
                            print(f"cmd_name:{cmd_name}, tensor_shape:{tensor_shape}, tenosr_dtype:{tenosr_dtype}, comm_group:{comm_group}, duration:{duration}")

                        torch_graph_op_excution_record_dict[wrank_id][cmd_name] = {'duration': duration, 'sub_ops_list': sub_ops_list}

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
    if not isinstance(tensor_shape, list):
        tensor_shape = ast.literal_eval(tensor_shape)
    assert isinstance(tensor_shape, list), f"Invalid type of tensor_shape:{type(tensor_shape)}, tensor_shape:{tensor_shape}"
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

def get_op_excution_time_from_torchgraph(operation_dict:dict, cmd_name:str, wrank_id:int, framwork)->float:
    if framwork == "deepspeed":
        reference_support_list = GLOBAL_DS_DIRECT_MAPPING_LIST
        reference_not_support_list = GLOBAL_DS_NOT_SUPPORTED_LIST
    elif framwork == "megatron-lm":
        reference_support_list = GLOBAL_MG_DIRECT_MAPPING_LIST
        reference_not_support_list = GLOBAL_MG_NOT_SUPPORTED_LIST
    else:
        raise ValueError(f"Invalid framwork: {framwork}")

    if cmd_name in reference_support_list:
        print(f"cmd_name:{cmd_name}, wrank_id:{wrank_id}")
        return operation_dict[wrank_id][cmd_name]
    elif cmd_name in reference_not_support_list:
        return None
    else:
        raise ValueError(f"Invalid cmd_name: {cmd_name}") 

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
                        batch_id = int(kwargs.get('batch_id', 0))
                        mg_state = kwargs.get('mg_state', None)
                        group_kind = kwargs.get('group_kind', None)
                        description = kwargs.get('description', None)
                        if description is not None:
                            description = re.sub(r'\W+', '', description)

                        op_kind = "comp" if cmd_name in MG_COMP_OPERATION else "comm" 
                        # if mpu.tp_size == 1:
                        # if cmd_name in ['ep_allreduce']:
                        #     op_kind = "comp"

                        end_timestamp = None
                        duration = None
                        if running_mode == MODE_PROFILE:
                            duration = kwargs.get('duration', None)
                            timestamp = float(kwargs.get('timestamp', 0))
                            end_timestamp = timestamp

                            if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                                if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                                            (mpu.dp_size > 1 and cmd_name in ['loss_func']):
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
                                timestamp = float(kwargs.get('timestamp', 0))
                                end_timestamp = timestamp

                                if cmd_name in GLOBAL_CMD_NEED_BREAK_LIST:
                                    if (mpu.tp_size > 1 and cmd_name in ['forward_step', 'backward_step']) or \
                                                (mpu.dp_size > 1 and cmd_name in ['loss_func']):
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
                                # simulate模式下预测p2p的comm duration(处理的是生成的schedule plan)
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
                                    print(f"cmd_name:{cmd_name}, tensor_shape:{tensor_shape}, tenosr_dtype:{tenosr_dtype}, comm_group:{comm_group}, duration:{duration}")
                                    # raise 0

                        elif running_mode == MODE_MODEL:
                            duration = 1
                        else:
                            raise ValueError(f"Error: running_mode is invalid, please check the trace file...")
                        
                        duration = round(float(duration), 2) if duration else None

                        # cmd_op_duration = 0.1 if (mpu.tp_size > 1 and running_mode == MODE_PROFILE and cmd_name in GLOBAL_CMD_NEED_BREAK_LIST) else duration

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
                            end_timestamp=end_timestamp,
                            hidden_duration=duration
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
                            (mpu.dp_size > 1 and cmd_name in ['loss_func']):
                                if running_mode == MODE_PROFILE or (running_mode == MODE_SIMULATE and is_trace):
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
                        if cmd_name in ['dp_allreduce']:
                            op_kind = "comp"

                        if running_mode == MODE_PROFILE:
                            duration = kwargs.get('duration', None)
                        elif running_mode == MODE_SIMULATE:
                            if is_trace:
                                duration = kwargs.get('duration', None)
                            else:
                                if torch_graph_stage_op_dict:
                                    duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id)
                                else:
                                    raise ValueError(f"Error: torch_graph_stage_op_dict is None, please check the trace file...")
                        elif running_mode == MODE_MODEL: 
                            duration = 1
                        duration = round(float(duration), 2) if duration else None

                        operation = Operation(
                            name=cmd_name,
                            duration=duration,
                            batch_id=batch_id,
                            wrank_id=wrank_id,
                            stage_id=stage_id,
                            mg_state=mg_state,
                            op_kind=op_kind,
                            group_kind=group_kind,
                            description=description
                        )
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
                                    duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id, "deepspeed")
                            elif running_mode == MODE_MODEL: 
                                duration = 1
                            duration = round(float(duration), 2) if duration else None

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


def parse_megatron_cmd(cmd):
    # 提取cmd_name和参数部分
    cmd_name, params = cmd.split('(', 1)
    params = params.rstrip(')')
    kwargs = {}

    # 使用正则表达式匹配 key=value 对，支持各种数据格式，包括嵌套列表和逗号间的空格
    # 修改后的正则表达式可以匹配包括点号的完整类型名，如torch.float16
    pattern = re.compile(r'(\w+)\s*=\s*(\[.*?\]|".*?"|\'.*?\'|None|\d+\.?\d*|[\w\.]+)')
    matches = pattern.findall(params)

    for match in matches:
        key = match[0]
        value = match[1].strip()

        # 对于 sub_operations 这样的嵌套复杂结构，需要手动处理
        if key == 'sub_operations':
            # 手动提取子操作的内容
            sub_ops_str = params.split('sub_operations=')[1]
            sub_ops_str = sub_ops_str.strip().lstrip('[').rstrip(']')
            sub_operations = []

            # 分割子操作内容
            sub_ops_list = re.findall(r"'(.*?)'", sub_ops_str)
            for sub_op in sub_ops_list:
                sub_operations.append(sub_op.strip())

            kwargs[key] = sub_operations
        else:
            # 处理普通的 key=value 对
            if value.startswith('[') and value.endswith(']'):
                kwargs[key] = eval(value)
            elif value == 'None':
                kwargs[key] = None
            elif value.isdigit():
                kwargs[key] = int(value)
            elif re.match(r'^\d+\.\d+$', value):  # 浮点数处理
                kwargs[key] = float(value)
            else:
                kwargs[key] = value

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

    # TODO：这里的查询方式可以优化？维护一个dict,key为OP的唯一ID
    for op in trace_stage.operations_list:
        if isinstance(operation, SubOperation):
            if isinstance(op, SubOperation):
                # print(f"wrank_id:{operation.wrank_id}, check op.name_with_id:{op.name_with_id}")
                #TODO: 维护一个可以用于一致性检查的ID
                if op.name_with_id == operation.name_with_id and op.batch_id == operation.batch_id:
                    # 对于subop，需要确认name_with_id和batch_id都相同(存在重复的同类型的subops)
                    operation_duration = op.duration
                    break
        else:
            if op.name == operation.name and op.batch_id == operation.batch_id:
                operation_duration = op.duration
                break

    # # TODO:修正为network的接口
    # if "allreduce" in op.name:
    #     import random
    #     op_duration = float(op.duration) * random.uniform(0.9, 1.0)
    
    if operation_duration == None:
        print(f"wrank_id:{operation.wrank_id}, type_op:{type(operation)}, operation.name_with_id:{operation.name_with_id}, operation.batch_id:{operation.name_with_id}")
        raise 0

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
    
    ''' process1: single-gpu profiler 完善OP/SUBOP duration '''
    # TODO：只有在GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST中OP/SUBOP才会被从profiler测量的数据中读取（torch_graph_stage_op_dict）
    if operation.name in GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST:
        if torch_graph_stage_op_dict[rank_id][operation.name]['sub_ops_list']:
        # if isinstance(torch_graph_stage_op_dict[rank_id][operation.name], dict) and 'sub_ops_list' in torch_graph_stage_op_dict[rank_id][operation.name]:
            # 需要拆分的OP的'sub_ops_list'的value存在
            operation_duration = torch_graph_stage_op_dict[rank_id][operation.name]['duration']
            operation.hidden_duration = operation_duration
            operation.duration = 0.01
            # print(f"here i am, can break, operation.name:{operation.name}, operation.batch_id:{operation.batch_id}")
            # raise 0
            sub_ops_list = torch_graph_stage_op_dict[rank_id][operation.name]['sub_ops_list']
            sub_ops_list_copy = copy.deepcopy(sub_ops_list)
            for sub_op in sub_ops_list_copy:
                sub_op.name_with_id += f"_{str(operation.batch_id)}_{operation.name}"
            # print(f"sub_ops_list:{sub_ops_list_copy}")
            # raise 0
        else:
            # 'sub_ops_list'无value,处理OP
            operation.duration = float(torch_graph_stage_op_dict[rank_id][operation.name]['duration'])
            operation.hidden_duration = None
            if operation.name == "dp_allreduce" or "ep_allreduce" or "tp_allreduce":
                print(f"allreduce的duration补充: operation.name: {operation.name}, duration{operation.duration}")
            
    else:
        print(f"{operation.name} not in GLOBAL_SINGLE_GPU_PROFILE_OP_NAME_LIST")


    ''' process2: muti-gpus trace 完善OP/SUBOP duration '''
    # 根据当前运行模式检查并补充operation和当前op的subop的duration,和其他属性
    # Note: 这个是根据trace补充，single profile里头存在一系列没有被支持模拟/没有进行模拟的op or subop
    # network op的duration应该在此之前被添加;如果没有被模拟则使用trace
    operation, sub_ops_list_copy, not_simulating_cmd_dict = check_and_supplement_duration_from_muti_gpus_trace(operation, \
                                                                            sub_ops_list_copy, running_mode, trace_stages_dict, torch_graph_stage_op_dict, rank_id)


    # 添加operation和suboperation到override_operations_list并返回
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

    # 如果sub_ops_list为空(代表该算子独立被使用), 需要查看operation的duration是否为空,为空则从trace补充duration
    if not sub_ops_list:
        # 单OP
        if running_mode == MODE_SIMULATE:
            # 查看duration属性是否被赋值（来自profile log）
            if not operation.duration:
                if trace_stages_dict is None:
                    operation.set_duration(1)

                else:
                    get_time = get_op_excution_time_from_trace(trace_stages_dict, operation) * random.uniform(0.93, 1.07)
                    operation.set_duration(get_time)
                assert operation.duration is not None
                if operation.name not in not_simulating_cmd_dict:
                    not_simulating_cmd_dict[operation.name] = operation.duration

        return operation, [], not_simulating_cmd_dict

    # 补充sub_ops_list中每个sub_op的属性(一般属性 + 根据model判定是否补充duration)
    # SUBOP的duration的处理:如果duration为空,则从trace中补充(SUBOP的duration没有记录在torch_graph_dict中,而是直接被赋值了？)
    for each_sub_op in sub_ops_list:
        # 一般属性例如mg_state,batch_id等
        add_sub_op_prop(operation, each_sub_op)
        # duration
        if running_mode == MODE_SIMULATE:
            if not each_sub_op.duration:
                # 如果当前的操作在torch_graph_stage_op_dict中存在，则直接使用;否则，通过trace获取
                # TODO: dict需要细化为{rank_id:{{op_name:{shape:{dtype:duration}}},...},...}
                if each_sub_op.name in torch_graph_stage_op_dict[rank_id]:
                    each_sub_op.set_duration(torch_graph_stage_op_dict[rank_id][each_sub_op.name])
                    #TODO：check这种情况会出现吗。我理解里，正常在torchdict中存下suboplist都已经完成duration的初始化？
                    raise 0
                else:
                    if trace_stages_dict is None:
                        each_sub_op.set_duration(1)
                    else:
                        # print(f"each_sub_op:{each_sub_op}")
                        each_sub_op.set_duration(get_op_excution_time_from_trace(trace_stages_dict, each_sub_op))
                    assert operation.duration is not None
                    if operation.name not in not_simulating_cmd_dict:
                        not_simulating_cmd_dict[operation.name] = operation.duration

    return operation, sub_ops_list, not_simulating_cmd_dict



class SimulatorEngine():
    """ manager the whole workflow of the train simulation """
    def __init__(self, stages_num=None, stages_steps_dict=None, grad_acc=None, trace_filepath=None, 
                framwork='megatron-lm', strategy="1F1B-none_interleaved",can_overlap=False, args=None, running_mode=None, torchgraph_filepath=None, stages_scheduling_filepath=None):
        
        self.stages_num = stages_num
        self.stages_steps_dict = stages_steps_dict
        self.grad_acc = grad_acc # 即megatron的num_microbatches
        self.strategy = strategy
        self.args = args
        self.running_mode = running_mode
        self.can_overlap = can_overlap
        self.framwork = framwork
        self.trace_filepath = trace_filepath
        # TODO:stages_scheduling_filepath
        self.stages_scheduling_filepath = stages_scheduling_filepath
        self.torchgraph_filepath = torchgraph_filepath
        self.stages_timeline = []
        self.predictor = None
        self._op_excution_time_dict = {} # predictor
        self.compelete_wranks_list = None
        self.stages_dict = None
        self.mpu = None
        self.dependency_relationship = None
        self.comm_matching_relationship = None
        self.timeline_manager = None
        self.not_simulating_cmd_dict = {} # not supported cmd in simulating

        # 需要lock进行维护的2个全局变量
        self.global_waiting_pool: dict = {} 
        self.global_finished_operations: dict = {} # global_finished_operations = {wrank_id_operation.name_operation.batch_id: Operation obj, ...}, e.g., {1_ForwardPass_0: Operation obj, 3_RecvActivation_0: Operation obj, ...
        assert running_mode in RUNNING_MODE_OPTION, "Invalid running mode: {running_mode}"
    
    def _init_3d_parallel_all_ranks(self, stages_or_wranks_dict, rank_instances_dict, trace_stages_dict, torch_graph_stage_op_dict):
        """ 
            根据mpu中的pp,tp,dp信息生成完整rank_list(包含所有ranks) 
            1. 当profile文件与world_size一致,且为非SIMULATE MODE: 直接返回即可,OP or SubOP的初始化已经全部完成
            2. 否则,说明只是根据schedule plan(PP-level)进行的初始化,需要根据PP/TP/DP group进行完整初始化(SIMULATE MODE必然使用了stage标记的生成方式,即便文件数量与world_size一致)

        """
        assert len(rank_instances_dict) == self.mpu.world_size, "Invalid rank_instances_dict"
        not_simulating_cmd_dict = {}
        
        # WARNING:对于有trace的情况，stage_dict已经完整初始化，因此必须直接返回(否则操作的不是stage_id而是wrank_id）；
        if len(stages_or_wranks_dict) == self.mpu.world_size and (self.running_mode == MODE_PROFILE or self.running_mode == MODE_MODEL):
            # 此时stage list中已包含所有ranks (profile mode); 只要是simulating/logic mode,都需要重新初始化所有ranks
            return list(stages_or_wranks_dict.values()), not_simulating_cmd_dict

        # SIMULATE MODE 下只读取schedule plan生成的结果
        # stage_list中包含了所有的Stage对象,每个Stage instance只完成了stage_id的初始化,
        # 因此需要根据pp/tp/dp group完成所有rank的Stage实例的初始化,并返回该完整初始化后的list
        # 含MODEL和SIMULATING模式
        compelete_wranks_list = []
        for rank_id, rank_instance in rank_instances_dict.items():
            stage_id = rank_instance._get_pp_local_rank()
            print(f"check rank_id:{rank_id} -> stage_id:{stage_id}")

            new_stage_instance: Stage = copy.deepcopy(stages_or_wranks_dict[stage_id])
            new_stage_instance.set_stage_wrank_id(rank_id)
            new_stage_instance.set_stage_rank(rank_instance)
            override_operations_list = []
            for operation in new_stage_instance.operations_list:
                operation.set_wrank_id(rank_id)
                operation.set_stage_id(stage_id)

                # check torch_graph的当前rank_id是否存在subop_list,有则拆分算子
                part_override_operations_list, part_not_simulating_cmd_dict = add_sub_ops_according_to_profile_dict(torch_graph_stage_op_dict, \
                                                     operation, rank_id, self.running_mode, trace_stages_dict)
                override_operations_list.extend(part_override_operations_list)
                not_simulating_cmd_dict.update(part_not_simulating_cmd_dict)

                # T0D0: 下面部分的代码放在add_sub_ops_according_to_profile_dict中，因为subop处理的逻辑和op是一致的，下面部分仅仅包含op
                # 补充或处理没有被SINGLE GPU PROFILE的算子duration(from MUTI-GPUS trace or just setting 1)
                # if self.running_mode == MODE_SIMULATE:
                #     if not operation.duration:
                #         if trace_stages_dict is None:
                #             operation.set_duration(1)
                #         else:
                #             operation.set_duration(get_op_excution_time_from_trace(trace_stages_dict, operation))
                #         assert operation.duration is not None
                #         if operation.name not in not_simulating_cmd_dict:
                #             not_simulating_cmd_dict[operation.name] = operation.duration
            # print(f"rank:{rank_id} 已替换,len_old={len(new_stage_instance.operations_list)}, len_new={len(override_operations_list)}")
            new_stage_instance.operations_list = override_operations_list
            compelete_wranks_list.append(new_stage_instance)

        return compelete_wranks_list, not_simulating_cmd_dict


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

        self.compelete_wranks_list, self.stages_dict, torch_graph_stage_op_dict, trace_stages_dict = self.generate_stages_and_cmds_info_from_datasets_and_schedules(self.trace_filepath, 
                                                                                    rank_instances_dict, self.running_mode, self.framwork, self.torchgraph_filepath,self.mpu, self.stages_scheduling_filepath)
        self.compelete_wranks_list, self.not_simulating_cmd_dict = self._init_3d_parallel_all_ranks(self.stages_dict, rank_instances_dict, trace_stages_dict, torch_graph_stage_op_dict)
        

        self.timeline_manager = TimelinesManager(dependency_relationship=self.dependency_relationship, 
                                                 comm_matching_relationship=self.comm_matching_relationship, 
                                                 compelete_wranks_list = self.compelete_wranks_list, strategy=self.strategy, can_overlap=self.can_overlap, 
                                                 global_waiting_pool=self.global_waiting_pool, global_finished_operations=self.global_finished_operations,
                                                 mpu_info=mpu_info,running_mode=self.running_mode, torch_graph_stage_op_dict=torch_graph_stage_op_dict,
                                                 torchgraph_filepath=self.torchgraph_filepath, trace_filepath=self.trace_filepath, not_simulating_cmd_dict=self.not_simulating_cmd_dict,
                                                 rank_instances_dict=rank_instances_dict,trace_stages_dict=trace_stages_dict)
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

    def _get_dependency_relationship(self, framwork=None, strategy=None):
        ''' 
        现阶段不存在跨stages的依赖项,目前维护的依赖项都是过程中关系, 而非模型内部的算子的关系X
        '''
        if framwork == "deepspeed":
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                return {"FirstStage": {'ForwardPass': ['LoadMicroBatch', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0]}, 
                        "MiddleStage": {'ForwardPass': ['RecvActivation', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0], 'SendGrad': ['BackwardPass', 0]},
                        "LastStage": {'ForwardPass': [['RecvActivation', 0], ['LoadMicroBatch', 0]], 'SendGrad': ['BackwardPass', 0]}}
            else:
                raise ValueError(f"Invalid mode, cannot get dependency relationship.")

        elif framwork == "megatron-lm":
            # PP>1时的依赖关系
            # TODO: 目前dp_allreduce和comp的依赖关系还没写入，因此需要定制overlap策略和对应控制模块
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                # 1/2D pp并行: pp>1,dp==1,tp==1: 没有allreduce、没有broadcast、只涉及pp的P2P
                return {"FirstStage": {'forward_step': ['get_batch', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': [['recv_forward', 0], ['get_batch', 0]], 'send_backward': ['backward_step', 0]}}
            
            elif strategy == "1F1B-none_interleaved" and self.mpu.tp_size > 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                # 3D or 2D pp,tp并行: pp>1,tp>1,dp=1: 没有dp的allreduce，涉及pp的p2p和tp的allreduce\broadcast
                # first stage会broadcast输入相关的数据，如 tokens、attention_mask 和 position_ids
                # last stage会broadcast与输出相关的数据，如 labels 和 loss_mask
                # 其他中间阶段不会执行任何 broadcast 操作
                # TODO:tp_load_batch_broadcast是不是可以改为load mb?(参考ds)
                return {"FirstStage": {'forward_step': ['get_batch', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': [['recv_forward', 0],['get_batch', 0]], 'send_backward': ['backward_step', 0]}}

            # elif strategy == "1F1B-none_interleaved" and self.mpu.tp_size > 1 and self.mpu.dp_size > 1:
            #     # 3D并行: 涉及dp的allreduce、pp的p2p和tp的allreduce、broadcast
            #     pass
            elif strategy == "no-pipelining" and self.mpu.tp_size > 1 and self.mpu.dp_size == 1:
                # 1D tp并行: pp==1,dp==1,tp>1: 只涉及tp的allreduce\broadcast
                # tokens、labels、loss_mask、attention_mask 和 position_ids都将被broadcast
                pass

            else:
                raise ValueError(f"Invalid mode, cannot get dependency relationship.")

        else:
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
                    'dp_allreduce': ['dp_allreduce', 'dp_group']
                }}

            if self.mpu.tp_size > 1:
                # tp_load_batch_broadcast 暂时没用，根据逻辑区分即可
                return_dict = {**return_dict, **{
                    'tp_load_batch_broadcast': ['tp_load_batch_broadcast','tp_group'],
                    'tp_allreduce': ['tp_allreduce', 'tp_group']
                }}

            if self.mpu.ep_size > 1:
                return_dict = {**return_dict, **{
                    'ep_allreduce': ['ep_allreduce', 'ep_group']
                }}

            if self.mpu.pep_size > 1:
                return_dict = {**return_dict, **{
                    'pep_allreduce': ['pep_allreduce', 'pep_group']
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
    #     `wrank_id:mg_state:operation_name:batch_id`
    #     Returns both a list of Stage objects and a dictionary mapping wrank_ids to Stage objects.
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
    #                                 duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id)
    #                         elif running_mode == MODE_MODEL: 
    #                             duration = 1
    #                         duration = round(float(duration), 2) if duration else None


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

    def visualize_timelines(self, wrank_id_start_end:list, specific_ranks_list:list, show_x_lim:int):
        self.timeline_manager.visualize_timelines(self.running_mode, self.mpu, wrank_id_start_end, specific_ranks_list, show_x_lim)

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