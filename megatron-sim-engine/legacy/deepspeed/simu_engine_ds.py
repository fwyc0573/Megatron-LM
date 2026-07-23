import copy
import os
import re
# import queue
from collections import deque
from typing import Union
# from deepspeed.utils import logger
from StaticGraphs.rank_manager import RankZoo
from StaticGraphs.parallel_group_manager import MPUInfo

DS_COMP_OPERATION = ['ForwardPass', 'BackwardPass', 'OptimizerStep', 'LoadMicroBatch']
DS_COMM_OPERATION = ['SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','ReduceGrads', 'ReduceTiedGrads']
DS_FINAL_OPERATION = ['ReduceGrads', 'ReduceTiedGrads', 'OptimizerStep']


# MG_COMP_OPERATION = ['forward_step', 'backward_step', 'load_batch']
MG_COMP_OPERATION = ['forward_step', 'backward_step']
MG_COMM_OPERATION = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward', 'tp_allreduce', 'tp_load_batch_broadcast', 'dp_allreduce']


P2P_COMM_COLLECTIVE = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward',
                        'SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation']
ALLREDUCE_COMM_COLLECTIVE = ['dp_allreduce', 'tp_allreduce', 'ReduceGrads', 'ReduceTiedGrads']

# DS可以被模拟以及未支持模拟的op
GLOBAL_DS_DIRECT_MAPPING_LIST = ['ForwardPass', 'BackwardPass', 'SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','OptimizerStep']
GLOBAL_DS_NOT_SUPPORTED_LIST = ['LoadMicroBatch','ReduceGrads', 'ReduceTiedGrads']

class Operation:
    def __init__(self, name="", duration=-1, buffer_id=-1, step_id=-1, batch_id=-1, 
                wrank_id=-1, stage_id=-1, mg_state=None, op_kind=None, waiting_acc=None, description=None):
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
        self.mg_is_last_iteration = False
        self.ds_buffer_id = buffer_id
        self.ds_step_id = step_id
        self.description = description # 给dp类allreduce算子增加的描述用于相互区分
    
    def __str__(self):
        return f"Operation(name={self.name}, duration={self.duration}, waiting_time={self.waiting_time}, waiting_acc={self.waiting_acc}, join_time={self.join_time}, finish_time={self.finish_time}, op_kind={self.op_kind}, pre_op={self.pre_op}, post_op={self.post_op}, batch_id={self.batch_id}, wrank_id={self.wrank_id}, mg_state={self.mg_state}, mg_is_last_iteration={self.mg_is_last_iteration}, ds_buffer_id={self.ds_buffer_id}, ds_step_id={self.ds_step_id})"

    def to_dict(self):
        return {
            "name": self.name,
            "duration": self.duration,
            "waiting_time": self.waiting_time,
            "waiting_acc": self.waiting_acc,
            "join_time": self.join_time,
            "finish_time": self.finish_time,
            "op_kind": self.op_kind,
            "pre_op": self.pre_op,
            "post_op": self.post_op,
            "batch_id": self.batch_id,
            "wrank_id": self.wrank_id,
            "stage_id": self.stage_id,
            "ds_buffer_id": self.ds_buffer_id,
            "ds_step_id": self.ds_step_id,
            "description": self.description
        }


    def _comp_set_join_finish_waiting_acc_time(self, join_time, waiting_acc):
        self.join_time = round(join_time,2)
        self.finish_time = round(join_time + self.duration,2)
        self.waiting_acc = waiting_acc
        self.waiting_time = 0

    def _comm_set_join_time(self, join_time):
        self.join_time = round(join_time,2)

    def _comm_set_waiting_finish_time(self, waiting_time):
        self.waiting_time = round(waiting_time,2)
        self.finish_time = round(self.join_time + waiting_time + self.duration,2)
        # self.finish_time = self.join_time + self.duration

    def _comm_set_waiting_acc_time(self, waiting_time, last_op_waiting_acc):
        self.waiting_acc = round(waiting_time + last_op_waiting_acc,2)
    
    def _set_pre_op(self, pre_op):
        self.pre_op = pre_op
    
    def _set_post_op(self, post_op):
        self.post_op = post_op
    
    def _set_wrank_id(self, wrank_id):
        self.wrank_id = int(wrank_id)

    def _set_duration(self, duration):
        self.duration = round(float(duration),2)

class Stage:
    """ 一个完整stage包含多个Operation obj"""
    def __init__(self, wrank_id, rank: RankZoo, steps_num, stage_id, framework=None):
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
        return f"Stage ID: {self.wrank_id}\nStage Kind: {self.stage_kind}\nOperations:\n{operations}"
    
    def _add_operations_to_list(self, operation: Operation):
        self.operations_list.append(operation)

    def _set_pre_stage(self, pre_stage):
        self.pre_stage = pre_stage
    
    def _set_post_stage(self, post_stage):
        self.post_stage = post_stage
    
    def _set_stage_kind(self, stage_kind: str):
        assert stage_kind in ['FirstStage', 'MiddleStage', 'LastStage'], "Invalid stage kind"
        self.stage_kind = stage_kind

    def _total_duration(self):
        """ Calculate the total duration of all operations in this stage """
        return sum(op.duration for op in self.operations_list)

    def _set_stage_wrank_id(self, wrank_id):
        self.wrank_id = int(wrank_id)

    def _set_stage_rank(self, rank: RankZoo):
        if self.wrank_id is not None:
            if rank.world_rank != self.wrank_id:
                raise ValueError("rank.world_rank and wrank_id must be consistent")
        self.rank = rank


class IndividualTimeline:
    """ 管理当前stage下的timeline """
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
        self.waiting_queue = deque()
        self.final_package_operation = [] # final_package_operation最后逐一处理
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
    def __init__(self, dependency_relationship, comm_matching_relationship, stages_list: list,
                  strategy: str='1F1B-none_interleaved', can_overlap=False, global_waiting_pool={}, 
                  global_finished_operations={},mpu_info=None,running_mode=None, torch_graph_stage_op_dict=None,
                  ds_trace_filepath=None, torchgraph_filepath=None,not_simulating_cmd_dict=None,rank_instances_dict=None):
        
        self.stages_list = sorted(stages_list, key=lambda s: s.wrank_id)  # 根据wrank_id进行排序
        self.strategy = strategy
        self.running_mode = running_mode
        self.mpu_info = mpu_info
        self.ds_trace_filepath = ds_trace_filepath
        self.torchgraph_filepath = torchgraph_filepath
        self.torch_graph_stage_op_dict = torch_graph_stage_op_dict
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
        for stage in self.stages_list:
            timeline = IndividualTimeline(stage=stage, can_overlap=self.can_overlap)
            stages_timeline_process_dict[stage.wrank_id] = timeline
        
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

    def _stages_pipeline_parallel(self):
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
                for wrank_id in sorted(self.stages_timeline_process_dict.keys()):
                    timeline: IndividualTimeline  = self.stages_timeline_process_dict[wrank_id]

                    if timeline.waiting_queue:
                        all_queues_empty = False

                        # 检查当前timeline是否blocked
                        operation: Operation = timeline.waiting_queue.popleft()
                        # if operation.name == "ReduceGrads":
                        #     raise ValueError(f"debug11111111111")
                        is_timeline_blocked: bool = self._check_timelline_blocked_status(timeline=timeline, operation=operation)

                        if is_timeline_blocked:
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
        if operation.name == "dp_allreduce" and operation.wrank_id == 2 and operation.batch_id == 0:
            print("debug")

        if operation.name in self.dependency_relationship[timeline.stage_kind]:
            dependency_operation = self._operation_dependency_finished(operation=operation, wrank_id=timeline.wrank_id, stage_kind=timeline.stage_kind)
            if not dependency_operation:
                can_go_on_sign = False

        if operation.op_kind == "comm":
            # 查看当前comm的所属类别和并行维度
            # is_p2p_fused_comm_operation: bool = self._is_p2p_fused_comm_operation(operation)
            comm_kind, parallel_kind = self._get_comm_operation_kind_and_parallel_dimension(operation)
            # next_dependency_operation: Operation = None # p2p_fused情况
            if operation.mg_state == "steady": 
                # P2P通信的特殊情况(该操作在实际过程中的一个算子被拆成2个，其余op都与实际相同为1个)
                # megatron的stready阶段的通信op是2个,检查另外一个op,如果未完成,返回等待队列
                print(f"operation.wrank_id: {operation.wrank_id}, operation.name: {operation.name}, operation.batch_id: {operation.batch_id}")
                # next_operation: Operation = None
                try:
                    next_operation: Operation = timeline.waiting_queue.popleft()
                    if next_operation.mg_state != "steady":
                        operation.mg_is_last_iteration = True
                        timeline.waiting_queue.appendleft(next_operation)
                        # TODO：修正p2p_fused为p2p，last_iter的send_backward目前没法在_get_comm_operation_kind_and_parallel_dimension中被正常标记
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
        if not can_go_on_sign:
            timeline.waiting_queue.appendleft(operation)
            print(f"wrank_id: {timeline.wrank_id}, stage_kind: {timeline.stage_kind}")
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
        current_format_operantion_name: str = self._get_format_operation_name(wrank_id=timeline.wrank_id, operation_name=operation.name, batch_id=operation.batch_id, description=operation.description)

        ''' 总体分为comp和comm两种情况处理 '''
        if operation.op_kind == "comp":
            if operation.name not in self.dependency_relationship[timeline.stage_kind]:
                # 无依赖项,更新该operation的属性,加入timeline和global_finished_operations
                # operation._comp_set_join_finish_waiting_acc_time(last_operation_time, last_operation.waiting_acc)
                # 加入timeline和global_finished_operations
                operation._comp_set_join_finish_waiting_acc_time(last_operation_time, 0)
                target_timeline.append(operation)
                self.global_finished_operations[current_format_operantion_name] = operation
                print(f"计算操作：{current_format_operantion_name}无依赖操作, 该operation已经完成属性更新,并加入到数据结构中")
            else:
                operation._comp_set_join_finish_waiting_acc_time(last_operation_time, 0)
                target_timeline.append(operation)
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
            operation._comm_set_join_time(last_operation_time)
            if comm_kind == "p2p_fused" and parallel_kind == "pp" and not operation.mg_is_last_iteration:
                '''
                pp类型fused_comm,2个comm. ops情况:
                    只有megatron的steady阶段,即operation: send_forward_recv_backward and send_backward_recv_forward,此时可取出下一个operation
                    steady的last_iteration只有send_backward(单op)
                '''
                # 设定另一个op的join时间
                next_operation._comm_set_join_time(last_operation_time)

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
                    _ = self._update_matching_comm_ops_properties(comm_matching_operation_list=comm_matching_operation_list, 
                                                                    comm_matching_operation_format_name_list=comm_matching_operation_format_name_list,
                                                                    current_operation=operation, comm_kind=comm_kind, parallel_kind=parallel_kind, 
                                                                    current_timeline=timeline)
                    # 更新current operation(2个)的属性
                    _ = self._update_current_comm_op_properties(comm_op=operation, current_timeline=timeline)
                    _ = self._update_current_comm_op_properties(comm_op=next_operation, current_timeline=timeline)

                    operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation_name=operation.name, batch_id=operation.batch_id, description=operation.description)
                    next_operation_format_name = self._get_format_operation_name(wrank_id=next_operation.wrank_id, operation_name=next_operation.name, batch_id=next_operation.batch_id, description=next_operation.description)
                    print(f"通信操作: p2p_fused {operation_format_name}和{next_operation_format_name}进行收尾操作.")
                else:
                    '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                    operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation_name=operation.name, batch_id=operation.batch_id, description=operation.description)
                    next_operation_format_name = self._get_format_operation_name(wrank_id=next_operation.wrank_id, operation_name=next_operation.name, batch_id=next_operation.batch_id, description=next_operation.description)
                    self.global_waiting_pool[operation_format_name] = operation
                    self.global_waiting_pool[next_operation_format_name] = next_operation
                    timeline._set_is_blocked_sign(True)
                    print(f"通信操作: p2p_fused {operation_format_name}和{next_operation_format_name}注册并等待.")

            else:
                assert parallel_kind == "dp" or parallel_kind == "pp", "tp is not supported."
                if (comm_kind == "p2p" and parallel_kind == "pp") or parallel_kind == "dp":
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

                        operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation_name=operation.name, batch_id=operation.batch_id, description=operation.description)
                        print(f"通信操作: {comm_kind}_{parallel_kind} |  {operation_format_name}进行收尾操作.")
                    else:
                        '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                        operation_format_name = self._get_format_operation_name(wrank_id=operation.wrank_id, operation_name=operation.name, batch_id=operation.batch_id, description=operation.description)
                        self.global_waiting_pool[operation_format_name] = operation
                        timeline._set_is_blocked_sign(True)

                        print(f"通信操作: {comm_kind}_{parallel_kind} | {operation_format_name}注册并等待.")

                elif comm_kind == "allreduce" and parallel_kind == "tp":
                    pass
                elif comm_kind == "broadcast" and parallel_kind == "tp":
                    pass

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
        matching_operation_format_name_list = []
        matching_operation_list = []

        # 生成comm group的formatname list
        for operation in operation_list:
            matching_operation_name, matching_operation_wrank_id_list, _ = self._get_comm_matching_operation_name_and_wrank_id(operation=operation, 
                                                                                comm_kind=comm_kind, parallel_kind=parallel_kind, timeline=timeline)
            for matching_operation_wrank_id in matching_operation_wrank_id_list:
                matching_operation_format_name = self._get_format_operation_name(wrank_id=matching_operation_wrank_id, 
                                                                                operation_name=matching_operation_name, batch_id=operation.batch_id, 
                                                                                description=operation.description)
                if matching_operation_format_name not in self.global_waiting_pool:
                    return False, None, None
                else:
                    matching_operation_list.append(self.global_waiting_pool[matching_operation_format_name])
                    matching_operation_format_name_list.append(matching_operation_format_name)
        return True, matching_operation_list, matching_operation_format_name_list

    def _update_current_comm_op_properties(self, comm_op: Operation, current_timeline: IndividualTimeline,gap_waiting_time:float):
        # 1. 更新waiting time/finish time, if当前op是刚加入的,不存在等待时间;else 更新更待时间
        if self.running_mode == "mapping":
            comm_op._comm_set_waiting_finish_time(0)
        else:
            if gap_waiting_time > 0:
                comm_op._comm_set_waiting_finish_time(gap_waiting_time)
            else:
                comm_op._comm_set_waiting_finish_time(0)
                # 说明current op是先注册的,因此补充等待时间
        # comm_op._comm_set_waiting_finish_time(gap_waiting_time)
            
        # 2. 加入到timeline中
        current_timeline._add_comm_op_to_timeline([comm_op])

        # 3. global_finished_operations
        operation_format_name = self._get_format_operation_name(wrank_id=comm_op.wrank_id, operation_name=comm_op.name, batch_id=comm_op.batch_id, description=comm_op.description)
        self.global_finished_operations[operation_format_name] = comm_op

    def _update_matching_comm_ops_properties(self, comm_matching_operation_list: list, comm_matching_operation_format_name_list: list, 
                                                                        current_operation: Operation, comm_kind: str, parallel_kind: str, 
                                                                        current_timeline: IndividualTimeline):
        for matching_operation in comm_matching_operation_list:
            # 1. 更新macthing comm ops的 waiting time/finish time
            # print(f"current_operation: {current_operation}")
            # print(f"matching_operation: {matching_operation}")
            # gap_waiting_time =  current_operation.join_time - matching_operation.join_time # 0
            abs_gap_waiting_time = abs(current_operation.join_time - matching_operation.join_time)
            if self.running_mode == "mapping":
                gap_waiting_time = 0
            else:
                # 先判定哪个操作注册的更早(依据join_time)
                if current_operation.join_time >= matching_operation.join_time:
                    # current晚于matching op 注册
                    gap_waiting_time = abs_gap_waiting_time
                else:
                    # matching晚于current op 注册, matching's watiing_time is 0.
                    gap_waiting_time = 0
            # gap_waiting_time = current_operation.join_time - matching_operation.join_time if not self.running_mode else 0
            matching_operation._comm_set_waiting_finish_time(gap_waiting_time)

            # 获取stage_offset从而得到matching_op_timeline；获取target_wrank_id从而获取operation_matching_format_name
            # _, matching_operation_wrank_id, stage_offset = self._get_comm_matching_operation_name_and_wrank_id(operation=current_operation, 
            #                                                                                       comm_kind=comm_kind, parallel_kind=parallel_kind, 
            #                                                                                       timeline=current_timeline)
            # matching_op_timeline: IndividualTimeline= current_timeline.post_individual_timeline if stage_offset > 0 else current_timeline.pre_individual_timeline
            # TODO: 检查一下
            matching_operation_wrank_id = matching_operation.wrank_id
            # Note: 这里应该返回的是当前通信group的上一个，current_timeline的post和pre皆是pp维度的
            matching_op_timeline: IndividualTimeline = self.stages_timeline_process_dict[matching_operation_wrank_id]
            # matching_op_timeline: IndividualTimeline = current_timeline.post_individual_timeline if matching_operation_wrank_id > current_timeline.wrank_id  \
            #                                                                                     else current_timeline.pre_individual_timeline

            # 2. 更新global_finished_operations
            operation_matching_format_name = self._get_format_operation_name(wrank_id=matching_operation_wrank_id, 
                                                                             operation_name=matching_operation.name, 
                                                                             batch_id=matching_operation.batch_id, description=matching_operation.description)
            self.global_finished_operations[operation_matching_format_name] = matching_operation

            # 3. 加入到timeline中
            matching_op_timeline._add_comm_op_to_timeline([matching_operation])

        # 4. 更新self.global_waiting_pool,遍历comm_matching_operation_format_name_list，删除global_waiting_pool中的key-value
        for operation_format_name in comm_matching_operation_format_name_list:
            self.global_waiting_pool.pop(operation_format_name, None)

        # 5. 更新blocked状态
        matching_op_timeline._set_is_blocked_sign(False)
        # gap_waiting_time==0则说明需要更新current op的watiting_time(在非mapping model下)
        if self.running_mode == "mapping" or gap_waiting_time != 0:
            return 0
        else:
            return abs_gap_waiting_time
        

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
            if comm_kind == "broadcast" or comm_kind == "allreduce":
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.tp_groups if rank_id != timeline.wrank_id], None
            else:
                raise ValueError(f"Invalid comm_kind name: {comm_kind}")
        
        elif parallel_kind == "dp" and comm_kind == "allreduce":
            # 获取 dp group wrank_id list
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.dp_groups if rank_id != timeline.wrank_id], None
        else:
            raise ValueError(f"Invalid comm_kind name: {comm_kind} and parallel_kind name: {parallel_kind}")


    # TODO: 判定方式修改一下，可以根据name直接拆分
    def _get_comm_operation_kind_and_parallel_dimension(self, operation: Operation):
        if operation.mg_state == "steady" and operation.name in P2P_COMM_COLLECTIVE:
            return "p2p_fused", "pp"
        elif operation.mg_state != "steady" and operation.name in P2P_COMM_COLLECTIVE:
            return "p2p", "pp"
        elif operation.name in ALLREDUCE_COMM_COLLECTIVE:
            if "tp" in operation.name:
                return "allreduce", "tp"
            elif "dp" in operation.name or operation.name in DS_COMM_OPERATION:
                return "allreduce", "dp"
            else:
                raise ValueError(f"Invalid allreduce operation name: {operation.name}")
        elif "broadcast" in operation.name and "tp" in operation.name:
            return "broadcast", "tp"
        else:
            raise ValueError(f"Invalid comm operation name: {operation.name}")


    def _check_timelline_blocked_status(self, timeline:IndividualTimeline, operation:Operation)->bool:
        """ 根据overlap情况来判定当前timeline是否被阻塞
            返回True表示阻塞,返回False表示不阻塞
        """
        # 如果不允许overlap且operation是comp类型，或者operation是comm类型，那么检查timeline的is_comm_blocked状态
        # 其他情况返回False，表示不阻塞
        return (not self.can_overlap and operation.op_kind == "comp" or operation.op_kind == "comm") and timeline.is_comm_blocked


    # def _get_format_operation_name(self, wrank_id:int, operation_name: str, stage_offset: int, batch_id: int) -> str:
    #     return str(wrank_id+stage_offset) + "_" + operation_name + "_" + str(batch_id)


    def _get_format_operation_name(self, wrank_id:int, operation_name: str, batch_id: int, description: str) -> str:
        if batch_id is not None:
            return str(wrank_id) + "_" + operation_name + "_" + str(batch_id)
        else:
            # Note：没有batch_id的op有reduce系列和optimizer_step
            return str(wrank_id) + "_" + operation_name + "_" + description
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
                all_finished = False
                break

        if all_finished:
            return latest_operation
        else:
            return False


    def _handle_final_package_operation(self):
        for wrank_id, timeline in self.stages_timeline_process_dict.items():
            for operation in timeline.final_package_operation:
                timeline._add_operation_to_timeline(operation, self.global_finished_operations, self.dependency_relationship, self.comm_matching_relationship, timeline.stage_kind)

    def v1_visualize_timelines(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Define the number of rows needed for our subplots based on the number of stages
        num_stages = len(self.stages_timeline_process_dict)

        # Set up the figure and axes
        fig, axs = plt.subplots(nrows=num_stages, ncols=1, figsize=(10, num_stages * 2), squeeze=False)

        # Determine the maximum finish time across all operations in all stages for consistent x-axis scale
        max_finish_time = max(
            [op.finish_time for timeline in self.stages_timeline_process_dict.values()
            for op in timeline.comp_timeline + timeline.comm_timeline],
            default=0
        )

        operation_labels = {
            'ForwardPass': 'FP', 'BackwardPass': 'BP', 'OptimizerStep': 'OS',
            'LoadMicroBatch': 'LB', 'SendGrad': 'SG', 'RecvGrad': 'RG',
            'SendActivation': 'SA', 'RecvActivation': 'RA', 'ReduceGrads': 'AR',
            'ReduceTiedGrads': 'AR', 'forward_step': 'FS', 'backward_step': 'BS',
            'recv_forward': 'RF', 'send_forward': 'SF', 'recv_backward': 'RB', 'send_backward': 'SB',
            'tp_load_batch_broadcast': 'BDC', 'tp_allreduce': 'AR', 'dp_allreduce': 'AR',
        }

        # Define special operations that require red coloring
        red_operations = {'tp_load_batch_broadcast', 'dp_allreduce', 'tp_allreduce', 'ReduceTiedGrads', 'ReduceGrads'}

        # Plot each stage's timelines
        for idx, (wrank_id, individual_timeline) in enumerate(self.stages_timeline_process_dict.items()):
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
                label = f"{operation_labels.get(op.name, 'NA')}{op.batch_id}"
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
            ax.set_xlim(0, max_finish_time)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_ylabel(f'Stage {wrank_id}')
            ax.set_xlabel('Time(ms)')
            ax.label_outer()

        plt.tight_layout()
        plt.show()

    def v2_visualize_timelines(self, running_mode, mpu):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk
        from tkinter import ttk

        # Define the number of rows needed for our subplots based on the number of stages
        num_stages = len(self.stages_timeline_process_dict)

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
        fig, axs = plt.subplots(nrows=num_stages, ncols=1, figsize=(10, num_stages * 2), squeeze=False)

        # Add a title to the figure
        plt.suptitle(f"{running_mode} - PP{mpu.pp_size} - TP{mpu.tp_size} - DP{mpu.dp_size}")

        # Determine the maximum finish time across all operations in all stages for consistent x-axis scale
        max_finish_time = max(
            [op.finish_time for timeline in self.stages_timeline_process_dict.values()
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
            'tp_load_batch_broadcast': 'BDC', 'tp_allreduce': 'AR', 'dp_allreduce': 'AR',
        }

        # Define special operations that require red coloring
        red_operations = {'tp_load_batch_broadcast', 'dp_allreduce', 'tp_allreduce', 'ReduceTiedGrads', 'ReduceGrads'}

        patches_dict = []

        # Plot each stage's timelines
        for idx, (wrank_id, individual_timeline) in enumerate(self.stages_timeline_process_dict.items()):
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
                patches_dict.append((rect, op.name, duration, op.join_time, op.finish_time, ax))
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
            ax.set_xlim(0, max_finish_time)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_ylabel(f'Rank {wrank_id}')

            # Add total time text
            if all_operations:
                total_time = round((all_operations[-1].finish_time), 2)
                ax.text(max_finish_time / 2, 1.05, f"total_time = {total_time}ms", horizontalalignment='center')

        plt.tight_layout()

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

        def update_annot(rect, name, duration, join_time, finish_time, annot):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            annot.xy = (x, y)
            text = f"Name: {name}\nDuration: {duration:.2f}\nJoin Time: {join_time:.2f}\nFinish Time: {finish_time:.2f}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            if not hover_active:
                return
            vis = any(annot.get_visible() for annot in annots)
            for rect, name, duration, join_time, finish_time, ax in patches_dict:
                cont, _ = rect.contains(event)
                if cont:
                    update_annot(rect, name, duration, join_time, finish_time, annots[axs.tolist().index([ax])])
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

    def v3_visualize_timelines(self, running_mode, mpu):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk
        from tkinter import ttk

        # Define the number of rows needed for our subplots based on the number of stages
        num_stages = len(self.stages_timeline_process_dict)

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
        not_supported_ops = ", ".join(self.not_simulating_cmd_dict.keys())
        if len(self.not_simulating_cmd_dict.keys()) != 0:
            not_supproted_string = f"Operations not yet supported: {not_supported_ops}"
        else:
            not_supproted_string = f"All operations are supported now."
        annotation_text = f"{running_mode} - PP{mpu.pp_size} - TP{mpu.tp_size} - DP{mpu.dp_size} \n {not_supproted_string}"
        plt.figtext(0.5, 0.98, annotation_text, ha='center', fontsize=10)

        # Determine the maximum finish time across all operations in all stages for consistent x-axis scale
        max_finish_time = max(
            [op.finish_time for timeline in self.stages_timeline_process_dict.values()
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
            'tp_load_batch_broadcast': 'BDC', 'tp_allreduce': 'AR', 'dp_allreduce': 'AR',
        }

        # Define special operations that require red coloring
        red_operations = {'tp_load_batch_broadcast', 'dp_allreduce', 'tp_allreduce', 'ReduceTiedGrads', 'ReduceGrads'}

        patches_dict = []

        # Plot each stage's timelines
        for idx, (wrank_id, individual_timeline) in enumerate(self.stages_timeline_process_dict.items()):
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
                patches_dict.append((rect, op.name, duration, op.join_time, op.finish_time, ax))
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
            ax.set_xlim(0, max_finish_time)
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

        def update_annot(rect, name, duration, join_time, finish_time, annot):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            annot.xy = (x, y)
            text = f"Name: {name}\nDuration: {duration:.2f}\nJoin Time: {join_time:.2f}\nFinish Time: {finish_time:.2f}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            if not hover_active:
                return
            vis = any(annot.get_visible() for annot in annots)
            for rect, name, duration, join_time, finish_time, ax in patches_dict:
                cont, _ = rect.contains(event)
                if cont:
                    update_annot(rect, name, duration, join_time, finish_time, annots[axs.tolist().index([ax])])
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

    # def visualize_timelines(self, running_mode, mpu, wrank_id_start_end=[1,100]):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk
        from tkinter import ttk

        wrank_id_start, wrank_id_end = wrank_id_start_end[0], wrank_id_start_end[1]

        # Validate wrank_id range
        if wrank_id_end - wrank_id_start + 1 > 200:
            raise ValueError("The number of wrank_ids to visualize is too large. Please set a range of 200 or fewer.")

        # Filter the stages_timeline_process_dict based on the given range
        filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id_start <= wrank_id <= wrank_id_end}
        
        num_stages = len(filtered_stages)

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
        not_supported_ops = ", ".join(self.not_simulating_cmd_dict.keys())
        if len(self.not_simulating_cmd_dict.keys()) != 0:
            not_supproted_string = f"Operations not yet supported: {not_supported_ops}"
        else:
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
            'tp_load_batch_broadcast': 'BDC', 'tp_allreduce': 'AR', 'dp_allreduce': 'AR',
        }

        # Define special operations that require red coloring
        red_operations = {'tp_load_batch_broadcast', 'dp_allreduce', 'tp_allreduce', 'ReduceTiedGrads', 'ReduceGrads'}

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
                patches_dict.append((rect, op.name, duration, op.join_time, op.finish_time, ax))
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
            ax.set_xlim(0, max_finish_time)
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

        def update_annot(rect, name, duration, join_time, finish_time, annot):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            annot.xy = (x, y)
            text = f"Name: {name}\nDuration: {duration:.2f}\nJoin Time: {join_time:.2f}\nFinish Time: {finish_time:.2f}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            if not hover_active:
                return
            vis = any(annot.get_visible() for annot in annots)
            for rect, name, duration, join_time, finish_time, ax in patches_dict:
                cont, _ = rect.contains(event)
                if cont:
                    update_annot(rect, name, duration, join_time, finish_time, annots[axs.tolist().index([ax])])
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

    def visualize_timelines(self, running_mode, mpu, wrank_id_start_end=[1, 100]):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk
        from tkinter import ttk

        wrank_id_start, wrank_id_end = wrank_id_start_end[0], wrank_id_start_end[1]

        # Validate wrank_id range
        if wrank_id_end - wrank_id_start + 1 > 200:
            raise ValueError("The number of wrank_ids to visualize is too large. Please set a range of 200 or fewer.")

        # Filter the stages_timeline_process_dict based on the given range
        filtered_stages = {wrank_id: timeline for wrank_id, timeline in self.stages_timeline_process_dict.items() if wrank_id_start <= wrank_id <= wrank_id_end}

        for wrank_id, timeline in filtered_stages.items():
            # Initialize accumulators for the current rank
            comp_timeline_time = 0
            comm_timeline_time = 0
            load_microbatch_time = 0

            # Calculate comp_timeline times
            for op in timeline.comp_timeline:
                duration = op.finish_time - op.join_time
                comp_timeline_time += duration
                if op.name == 'LoadMicroBatch':
                    load_microbatch_time += duration

            # Calculate comm_timeline times
            for op in timeline.comm_timeline:
                duration = op.finish_time - op.join_time
                comm_timeline_time += duration

            # Output the results for this rank
            print(f"rank{wrank_id} comp_timeline time: {comp_timeline_time:.2f} ms / {comp_timeline_time-load_microbatch_time}")
            print(f"rank{wrank_id} comm_timeline time: {comm_timeline_time:.2f} ms")
            print(f"rank{wrank_id} LoadMicroBatch time in comp_timeline: {load_microbatch_time:.2f} ms")

        # The rest of your visualization logic goes here...
        # This part remains unchanged for displaying the timeline as you originally had.



def process_json_files(directory):
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

def get_tmp_ds_simu_torchgraph_op_dict(filepath=None)->dict:
    operation_dict = process_json_files(filepath)
    
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

def get_parallel_torch_graph_stage_op_dict(torch_graph_dict, rank_instances_dict):
    """ 从ds获取的测试数据只包含Pipe 的stage id (非wrank_id), 需要扩展DP部分 """
    complete_torch_graph_dict = {}
    for id, rank_instance in rank_instances_dict.items():
        if rank_instance._get_dp_group_size() == 1:
            return torch_graph_dict

        complete_torch_graph_dict[id] = torch_graph_dict[rank_instance._get_pp_local_rank()]
    return complete_torch_graph_dict

def get_op_excution_time_from_torchgraph(operation_dict:dict, cmd_name:str, wrank_id:int)->float:
    if cmd_name in GLOBAL_DS_DIRECT_MAPPING_LIST:
        return operation_dict[wrank_id][cmd_name]
    elif cmd_name in GLOBAL_DS_NOT_SUPPORTED_LIST:
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

def process_trace_or_scheduling_files(my_filepath: str, mpu, rank_instances_dict: dict, running_mode, torch_graph_stage_op_dict, is_ds_trace: bool):
    assert my_filepath and mpu and rank_instances_dict and torch_graph_stage_op_dict, "Error: 初始化错误,请重新检查关键变量..."

    if running_mode == "mapping":
        assert is_ds_trace, "Error: mapping模式下只允许使用ds_trace文件..."

    stages_dict = {}
    # not_simulating_cmd_dict = {}

    num_ds_trace_files = get_num_txt_files(my_filepath)
    if num_ds_trace_files != mpu.world_size and num_ds_trace_files != mpu.pp_size:
        print(f"num_ds_trace_files: {num_ds_trace_files}, mpu.world_size: {mpu.world_size}, mpu.pp_size: {mpu.pp_size}")
        raise ValueError(f"Error: mpu初始化设定与读取trace的GPU数量不一致, 请重新修改config...")

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

                        if stage_rank_signal == "stage" and not is_ds_trace:
                            stage_id = stage_or_wrank_id
                            wrank_id = None
                            rank = None
                        elif stage_rank_signal == "rank" and is_ds_trace:
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

                            # 用于处理ds下dp=1时的allreduce算子.
                            # TODO: 因为dp为1时comm依赖项是不存在reduce系列的，因此单独处理为comp
                            if cmd_name in ['ReduceGrads', 'ReduceTiedGrads']: #and mpu.dp_size == 1:
                                op_kind = "comp"

                            # 根据模式来映射duration
                            if running_mode == "mapping":
                                duration = kwargs.get('duration', None)

                            elif running_mode == "simulating":
                                # 当是ds_trace时,正常获取实际执行时间；当是scheduling时,需要根据stage_id获取模拟值（当op未被模拟支持时,从真实中获取）
                                if is_ds_trace:
                                    duration = kwargs.get('duration', None)
                                else:
                                    # 从模拟器中获取值
                                    duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id)
                                    # if duration is None:
                                    #     # 从真实trace中获取, 推迟到3d_paralle函数中(根据每个rank依次取)
                                    #     not_simulating_cmd_dict[cmd_name] = 1

                            elif running_mode == "logic": 
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
                            print(operation)
                            stages_dict[stage_or_wrank_id]._add_operations_to_list(operation)

                    stages_dict[stage_or_wrank_id].steps_num = max(stages_dict[stage_or_wrank_id].steps_num, step_id + 1)

    return stages_dict

def get_op_excution_time_from_ds_trace(ds_trace_stages_dict, operation):
    ds_trace_stage = ds_trace_stages_dict.get(operation.wrank_id)
    for op in ds_trace_stage.operations_list:
        if op.name == operation.name and op.ds_step_id == operation.ds_step_id and op.description == operation.description and op.batch_id == operation.batch_id:
            return op.duration
    return None

class SimulatorEngine():
    """ manager the whole workflow of the train simulation """
    def __init__(self, stages_num=None, stages_steps_dict=None, grad_acc=None, ds_trace_filepath=None, 
                framwork='megatron-lm', strategy="1F1B-none_interleaved",can_overlap=False, args=None, running_mode=None,torchgraph_filepath=None, stages_scheduling_filepath=None):
        
        self.stages_num = stages_num
        self.stages_steps_dict = stages_steps_dict
        self.grad_acc = grad_acc # 即megatron的num_microbatches
        self.strategy = strategy
        self.args = args
        self.running_mode = running_mode
        self.can_overlap = can_overlap
        self.framwork = framwork
        self.ds_trace_filepath = ds_trace_filepath
        # TODO:stages_scheduling_filepath
        self.stages_scheduling_filepath = stages_scheduling_filepath
        self.torchgraph_filepath = torchgraph_filepath
        self.stages_timeline = []
        self.predictor = None
        self._op_excution_time_dict = {} # predictor
        self.stages_list = None
        self.stages_dict = None
        self.mpu = None
        self.dependency_relationship = None
        self.comm_matching_relationship = None
        self.timeline_manager = None
        self.not_simulating_cmd_dict = {} # not supported cmd in simulating

        # 需要lock进行维护的2个全局变量
        self.global_waiting_pool: dict = {} 
        self.global_finished_operations: dict = {} # global_finished_operations = {wrank_id_operation.name_operation.batch_id: Operation obj, ...}, e.g., {1_ForwardPass_0: Operation obj, 3_RecvActivation_0: Operation obj, ...
        assert running_mode in ['mapping', 'simulating', 'logic'], "Invalid running mode: {running_mode}"
    
    def _init_3d_parallel_all_ranks(self, stages_dict, rank_instances_dict, ds_trace_stages_dict):
        """ 根据mpu中的pp,tp,dp信息生成完整stages_list(包含所有ranks) """
        assert len(rank_instances_dict) == self.mpu.world_size, "Invalid rank_instances_dict"
        not_simulating_cmd_dict = {}

        if len(stages_dict) == self.mpu.world_size and (self.running_mode == "mapping" or self.running_mode == "logic"):
            # 此时stage list中已包含所有ranks (mapping mode); 只要是simulating/logic mode,都需要重新初始化所有ranks
            return list(stages_dict.values()), not_simulating_cmd_dict
        
        # stage_list中包含了所有的Stage对象,每个Stage instance只完成了stage_id的初始化,
        # 因此需要根据pp/tp/dp group完成所有rank的Stage实例的初始化,并返回该完整初始化后的list
        compelete_stages_list = []
        for rank_id, rank_instance in rank_instances_dict.items():
            # 遍历所有的rank,并根据当前wrank mapping to stage_id, 从而初始化所有的Stage实例
            stage_id = rank_instance._get_pp_local_rank()
            new_stage_instance: Stage = copy.deepcopy(stages_dict[stage_id])
            new_stage_instance._set_stage_wrank_id(rank_id)
            new_stage_instance._set_stage_rank(rank_instance)
            for operation in new_stage_instance.operations_list:
                operation._set_wrank_id(rank_id)
            compelete_stages_list.append(new_stage_instance)

        # 对于simulating模式,需要额外处理duration(为None且有trace,则取trace; 否则设定为1)
        if self.running_mode == "simulating":
            for stage in compelete_stages_list:
                for operation in stage.operations_list:
                    if operation.duration is None:
                        if ds_trace_stages_dict is None:
                            operation._set_duration(1)
                        else:
                            operation._set_duration(get_op_excution_time_from_ds_trace(ds_trace_stages_dict, operation))
                        assert operation.duration is not None
                        if operation.name not in not_simulating_cmd_dict:
                            not_simulating_cmd_dict[operation.name] = operation.duration

        return compelete_stages_list, not_simulating_cmd_dict


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

    def _init_tmp_stages_dataset_and_timeline_manager(self, rank_instances_dict, mpu_info: MPUInfo):
        # if self.running_mode == "simulating":
        #     self.predictor = self._init_excution_time_predictor()

        if self.framwork == 'deepspeed':
            # self.ds_trace_stages_list = self.v2_ds_handle_tmp_stages_dataset(self.ds_trace_filepath, rank_instances_dict, self.running_mode, 
            #                                                                  self.torchgraph_filepath,self.mpu, self.stages_scheduling_filepath)
            self.stages_list, self.stages_dict, torch_graph_stage_op_dict, ds_trace_stages_dict = self.v2_ds_handle_tmp_stages_dataset(self.ds_trace_filepath, 
                                                                                        rank_instances_dict, self.running_mode, self.torchgraph_filepath,self.mpu, self.stages_scheduling_filepath)
            self.stages_list, self.not_simulating_cmd_dict = self._init_3d_parallel_all_ranks(self.stages_dict, rank_instances_dict, ds_trace_stages_dict)

        elif self.framwork == 'megatron-lm':
            self.stages_list, self.stages_dict = self.mg_handle_tmp_stages_dataset(self.ds_trace_filepath, rank_instances_dict)
        
        self.timeline_manager = TimelinesManager(dependency_relationship=self.dependency_relationship, 
                                                 comm_matching_relationship=self.comm_matching_relationship, 
                                                 stages_list = self.stages_list, strategy=self.strategy, can_overlap=self.can_overlap, 
                                                 global_waiting_pool=self.global_waiting_pool, global_finished_operations=self.global_finished_operations,
                                                 mpu_info=mpu_info,running_mode=self.running_mode, torch_graph_stage_op_dict=torch_graph_stage_op_dict,
                                                 torchgraph_filepath=self.torchgraph_filepath, ds_trace_filepath=self.ds_trace_filepath, not_simulating_cmd_dict=self.not_simulating_cmd_dict,
                                                 rank_instances_dict=rank_instances_dict)
        print(f"INIT | tmp stages dataset and timeline manager have been set.")

    def _set_mpu_info_and_init_key_relationship(self, mpu_info):
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
        现阶段不存在跨stages的依赖项,目前维护的依赖项都是过程中关系, 而非模型内部的算子的关系
        
        '''
        if framwork == "deepspeed":
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                return {"FirstStage": {'ForwardPass': ['LoadMicroBatch', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0]}, 
                        "MiddleStage": {'ForwardPass': ['RecvActivation', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0], 'SendGrad': ['BackwardPass', 0]},
                        "LastStage": {'ForwardPass': [['RecvActivation', 0], ['LoadMicroBatch', 0]], 'SendGrad': ['BackwardPass', 0]}}
            else:
                raise ValueError(f"Invalid mode, cannot get dependency relationship.")

        elif framwork == "megatron-lm":
            # TODO： 考虑到同类型的op已是按序放入到待处理队列，因此同类型操作可以忽略依赖关系？（例如对于last stage，forward_step本身就在backward_step之前）
            # 另外，不同的pipeline模式，依赖关系不同。例如对于no-pipeline模式，就不存在p2p操作，不存在多种Stage
            # return {"FirstStage": {'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
            #         "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
            #         "LastStage": {'forward_step': ['recv_forward', 0], 'backward_step': ['forward_step', 0], 'send_backward': ['backward_step', 0]}}

            # PP>1时的依赖关系
            # TODO: 目前dp_allreduce和comp. 的依赖关系还没写入，因此需要定制overlap策略和对应控制模块
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                # 1D pp并行: pp>1,dp==1,tp==1: 没有allreduce、没有broadcast、只涉及pp的P2P
                return {"FirstStage": {'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': ['recv_forward', 0], 'send_backward': ['backward_step', 0]}}
            
            elif strategy == "1F1B-none_interleaved" and self.mpu.tp_size > 1 and self.mpu.dp_size >= 1 and self.mpu.pp_size > 1:
                # 3D or 2D pp,tp并行: pp>1,tp>1,dp=1: 没有dp的allreduce，涉及pp的p2p和tp的allreduce\broadcast
                # first stage会broadcast输入相关的数据，如 tokens、attention_mask 和 position_ids
                # last stage会broadcast与输出相关的数据，如 labels 和 loss_mask
                # 其他中间阶段不会执行任何 broadcast 操作
                return {"FirstStage": {'forward_step': ['tp_load_batch_broadcast', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': [['recv_forward', 0],['tp_load_batch_broadcast', 0]], 'send_backward': ['backward_step', 0]}}

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
                return_dict = {**return_dict, **{'RecvGrad': ['SendGrad', 1],
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
                
            if self.mpu.tp_size > 1:
                # tp_load_batch_broadcast 暂时没用，根据逻辑区分即可
                return_dict = {**return_dict, **{
                    'tp_load_batch_broadcast': ['tp_load_batch_broadcast', ['tp_src_rank', 'tp_group']],
                    'tp_allreduce': ['tp_allreduce', 'tp_group']
                }}

            if self.mpu.dp_size > 1:
                return_dict = {**return_dict, **{
                    'dp_allreduce': ['dp_allreduce', 'dp_group']
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
                        stages_dict[wrank_id]._add_operations_to_list(operation)

                    stages_dict[wrank_id].steps_num = max(stages_dict[wrank_id].steps_num, step_id + 1)

        for i, stage in stages_dict.items():
            if i == 0:
                stage._set_stage_kind('FirstStage')
            elif i == max(stages_dict.keys()):
                stage._set_stage_kind('LastStage')
            else:
                stage._set_stage_kind('MiddleStage')

        return list(stages_dict.values()), stages_dict

    
    @staticmethod
    def v2_ds_handle_tmp_stages_dataset(ds_trace_filepath: str, rank_instances_dict: dict, running_mode=None, torchgraph_filepath=None, mpu=None, stages_scheduling_filepath=None):
        """
        ds_trace_filepath: realistic trace from trainning in 3D parallel in DS framwork
        torchgraph_filepath: model operations runnnig database and graph files
        stages_scheduling_filepath: stages scheduling plans
        """

        # 获取torchgraph的算子执行时间并根据dp情况进行stage阶段id和rank id的映射处理
        torch_graph_stage_op_dict = get_tmp_ds_simu_torchgraph_op_dict(torchgraph_filepath)
        # torch_graph_stage_op_dict = get_parallel_torch_graph_stage_op_dict(tmp_torch_graph_dict, rank_instances_dict)

        # 请将以下部分打包为一个函数，最终返回值为stages_dict
        if running_mode == "mapping":
            stages_dict = process_trace_or_scheduling_files(my_filepath=ds_trace_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                 torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_ds_trace=True)
            ds_trace_stages_dict = stages_dict
        elif running_mode == "logic":
            if ds_trace_filepath is not None:
                stages_dict = process_trace_or_scheduling_files(my_filepath=ds_trace_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                    torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_ds_trace=True)
                ds_trace_stages_dict = stages_dict
            else:
                stages_dict = process_trace_or_scheduling_files(my_filepath=stages_scheduling_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                            torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_ds_trace=False)
                ds_trace_stages_dict = None
        elif running_mode == "simulating":
            if ds_trace_filepath is not None:
                ds_trace_stages_dict = process_trace_or_scheduling_files(my_filepath=ds_trace_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                        torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_ds_trace=True)
            else:
                ds_trace_stages_dict = None
            stages_dict = process_trace_or_scheduling_files(my_filepath=stages_scheduling_filepath, mpu=mpu, rank_instances_dict=rank_instances_dict, running_mode=running_mode, 
                                                                    torch_graph_stage_op_dict=torch_graph_stage_op_dict, is_ds_trace=False)
        # stages_dict = process_trace_or_scheduling_files(stages_scheduling_filepath, mpu, rank_instances_dict, running_mode, torch_graph_stage_op_dict)

        return list(stages_dict.values()), stages_dict, torch_graph_stage_op_dict, ds_trace_stages_dict




        num_ds_trace_files = get_num_txt_files(ds_trace_filepath)
        if num_ds_trace_files != mpu.world_size and num_ds_trace_files != mpu.pp_size:
            print(f"num_ds_trace_files: {num_ds_trace_files}, mpu.world_size: {mpu.world_size}, mpu.pp_size: {mpu.pp_size}")
            raise ValueError(f"Error: mpu初始化设定与读取trace的GPU数量不一致, 请重新修改config...")

        for filename in os.listdir(ds_trace_filepath):
            if filename.endswith(".txt"):
                with open(os.path.join(ds_trace_filepath, filename), 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            stage_rank_signal = line.split(":", 1)[0]
                            stage_step_part, cmds_str = line.split('_cmds:')
                            wrank_id_str, step_id_str = stage_step_part.split('_step_id:')
                            stage_or_wrank_id = int(wrank_id_str.split(':')[-1])
                            step_id = int(step_id_str.split(':')[-1])

                            if stage_rank_signal == "stage":
                                stage_id = stage_or_wrank_id
                                wrank_id = None
                                rank = None
                            elif stage_rank_signal == "rank":
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

                                # 用于处理ds下dp=1时的allreduce算子.
                                # TODO: 因为dp为1时comm依赖项是不存在reduce系列的，因此单独处理为comp
                                if cmd_name in ['ReduceGrads', 'ReduceTiedGrads']: #and mpu.dp_size == 1:
                                    op_kind = "comp"

                                # 根据模式来映射duration
                                if running_mode == "mapping":
                                    duration = kwargs.get('duration', None)
                                elif running_mode == "simulating":
                                    # DS不涉及TP,因此stage上的op耗时模拟也是唯一的;只需要根据stage_id得到每个op的耗时即可
                                    # TODO: 当MG中涉及TP并且希望TP可以进行不均衡分配,该如何解决该测试问题？
                                    duration = get_op_excution_time_from_torchgraph(torch_graph_stage_op_dict, cmd_name, stage_id)
                                    if duration is None:
                                        # 该op未被支持模拟, 暂用1替代
                                        duration = 1
                                        not_simulating_cmd_dict[cmd_name] = duration
                                    # if not duration and mpu.world_size == num_ds_trace_files:
                                    #     # 该op未被支持模拟, 直接使用trace
                                    #     duration = kwargs.get('duration', None)
                                    # else:
                                    #     # 该op未被支持模拟, 且没有trace可用
                                    #     duration = 1
                                elif running_mode == "logic": 
                                    duration = 1
                                # print(f"cmd_name:{cmd_name}")
                                operation = Operation(
                                    name=cmd_name,
                                    duration=round(float(duration), 2),
                                    buffer_id=kwargs.get('buffer_id', None),
                                    batch_id=batch_id,
                                    step_id=step_id,
                                    wrank_id=wrank_id,
                                    stage_id=stage_id,
                                    op_kind=op_kind,
                                    description=description,
                                )
                                print(operation)
                                stages_dict[stage_or_wrank_id]._add_operations_to_list(operation)

                        stages_dict[stage_or_wrank_id].steps_num = max(stages_dict[stage_or_wrank_id].steps_num, step_id + 1)

        return list(stages_dict.values()), stages_dict, torch_graph_stage_op_dict, not_simulating_cmd_dict


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

    @staticmethod
    def mg_handle_tmp_stages_dataset(folder_path: str, rank_instances_dict=None, running_mode=None):
        """
        Process all txt files in the given folder path, each representing operations in a stage.
        Each line in a txt file represents an operation in the format:
        `wrank_id:mg_state:operation_name:batch_id`
        Returns both a list of Stage objects and a dictionary mapping wrank_ids to Stage objects.
        """
        import os

        stages = {}
        # Read each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r') as file:
                    lines = file.readlines()
                    # Process each line in the file
                    for line in lines:
                        line = line.strip()
                        if line:
                            wrank_id, mg_state, operation_name, batch_id = line.split(':')
                            wrank_id, batch_id = int(wrank_id), int(batch_id)

                            # Create a new stage if it does not exist
                            if wrank_id not in stages:
                                stages[wrank_id] = Stage(wrank_id, rank_instances_dict[wrank_id], len(lines), "megatron-lm")
                            
                            # Create a new operation
                            op_kind = "comp" if operation_name in MG_COMP_OPERATION else "comm" 
                            # duration = kwargs.get('duration', '-1') if self.running_mode else float(1)
                            operation = Operation(
                                name=operation_name,
                                duration=1,  # Set duration uniformly as 1 for now
                                batch_id=batch_id,
                                wrank_id=wrank_id,
                                mg_state=mg_state,
                                op_kind=op_kind
                            )
                            
                            # Add operation to the stage
                            stages[wrank_id]._add_operations_to_list(operation)

                    # Update steps_num to the total number of lines in the current file
                    stages[wrank_id].steps_num = len(lines)

        return list(stages.values()), stages

    def _start_pipeline(self):
        if self.mpu is not None:
            self.timeline_manager._stages_pipeline_parallel()
        else:
            raise ValueError(f"mpu_info is not set.")

    def visualize_timelines(self, wrank_id_start_end:list):
        self.timeline_manager.visualize_timelines(self.running_mode, self.mpu, wrank_id_start_end)



    def ds_op_compare_simu_with_trace(self):
        self.timeline_manager.ds_op_compare_simu_with_trace()





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