import copy
# import queue
from collections import deque
from typing import Union
# from deepspeed.utils import logger

DS_COMP_OPERATION = ['ForwardPass', 'BackwardPass', 'OptimizerStep', 'LoadMicroBatch']
DS_COMM_OPERATION = ['SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','ReduceGrads', 'ReduceTiedGrads']
DS_FINAL_OPERATION = ['ReduceGrads', 'ReduceTiedGrads', 'OptimizerStep']


# MG_COMP_OPERATION = ['forward_step', 'backward_step', 'load_batch']
MG_COMP_OPERATION = ['forward_step', 'backward_step']
MG_COMM_OPERATION = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward']



class Operation:
    def __init__(self, name="", duration=-1, buffer_id=-1, step_id=-1, batch_id=-1, stage_id=-1, mg_state=None, waiting_acc=None):
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
        self.pre_op = None
        self.post_op = None
        self.batch_id = batch_id
        self.stage_id = stage_id
        self.mg_state = mg_state # warmup/steady/cooldown/help
        self.mg_is_last_iteration = False
        self.ds_buffer_id = buffer_id
        self.ds_step_id = step_id 

    def _comp_set_join_finish_waiting_acc_time(self, join_time, waiting_acc):
        self.join_time = join_time
        self.finish_time = self.join_time + self.duration
        self.waiting_acc = waiting_acc
        self.waiting_time = 0

    def _comm_set_join_time(self, join_time):
        self.join_time = join_time

    def _comm_set_waiting_finish_time(self, waiting_time):
        self.waiting_time = waiting_time
        self.finish_time = self.join_time + waiting_time + self.duration

    def _comm_set_waiting_acc_time(self, waiting_time, last_op_waiting_acc):
        self.waiting_acc = waiting_time + last_op_waiting_acc
    
    def _set_pre_op(self, pre_op):
        self.pre_op = pre_op
    
    def _set_post_op(self, post_op):
        self.post_op = post_op
    

class Stage:
    """ 一个完整stage包含多个Operation obj"""
    def __init__(self, stage_id, steps_num, framework="megatron-lm"):
        self.stage_id = stage_id
        self.steps_num = steps_num
        self.operations_list = []
        self.pre_stage = None
        self.post_stage = None
        self.stage_kind = None
        self.framework = framework

    def _add_operations_to_list(self, operation: Operation):
        self.operations_list.append(operation)

    def _set_pre_stage(self, pre_stage):
        self.pre_stage = pre_stage
    
    def _set_post_stage(self, post_stage):
        self.post_stage = post_stage
    
    def _set_stage_kind(self, stage_kind: str):
        assert stage_kind in ['FirstStage', 'MiddleStage', 'LastStage'], "Invalid stage kind"
        self.stage_kind = stage_kind



class IndividualTimeline:
    """ 管理当前stage下的timeline """
    def __init__(self, stage: Stage, can_overlap=False):
        ''' pre_stage、post_stage'''
        self.stage_id = stage.stage_id  
        self.pre_stage = stage.pre_stage
        self.post_stage = stage.post_stage
        self.stage_kind = stage.stage_kind
        self.pre_individual_timeline: IndividualTimeline = None
        self.post_individual_timeline: IndividualTimeline = None 
        self.comm_waiting_pool = {} # comm.加入需要特殊处理, key：匹配操作名称, val: Operation对象
        self.can_overlap = can_overlap
        self.is_blocked = False # 已注册但未完成的comm. ops会被阻塞,等待对方解除阻塞。对于comp. ops,不会出现这种情况
        self.comp_timeline = []
        self.comm_timeline = []
        self.waiting_queue = deque()
        self.final_package_operation = [] # final_package_operation最后逐一处理
        for operation in stage.operations_list:
            if stage.framework == "deepspeed":
                if operation.name in DS_FINAL_OPERATION:
                    self.final_package_operation.append(copy.deepcopy(operation))
                else:
                    self.waiting_queue.append(copy.deepcopy(operation))
            else:
                self.waiting_queue.append(copy.deepcopy(operation))

    def _set_pre_individual_timeline(self, pre_individual_timeline):
        self.pre_individual_timeline = pre_individual_timeline
    
    def _set_post_individual_timeline(self, post_individual_timeline):
        self.post_individual_timeline = post_individual_timeline


    def _add_operation_to_timeline(self, operation:Operation, global_finished_operations:dict, \
                                   dependency_relationship:dict, comm_matching_relationship:dict, stage_kind:str, strategy:str):
        target_timeline = self.comp_timeline if operation.name in MG_COMP_OPERATION or DS_COMP_OPERATION else self.comm_timeline

        # 获取当前timeline中最后一个operation的finish_time（依据是否overlap）
        last_operation_time: int = None
        last_operation: Operation = None
        if not self.can_overlap or strategy == "1F1B-none_interleaved":
            last_operation_time, last_operation = self._get_last_operation_time_and_op([self.comp_timeline, self.comm_timeline])
        elif self.can_overlap and strategy == "1F1B-interleaved":
            last_operation_time, last_operation = self._get_last_operation_time_and_op([target_timeline])
            raise ValueError(f"Overlap of Comm. and Comp. is not supported.")
        
        # 当前op的格式化命名
        current_format_operantion_name: str = self._get_format_operation_name(operation_name=operation.name, stage_offset=0, batch_id=operation.batch_id)

        ''' 统一检查dependency情况,对于dependency未完成的operation(该op存在依赖项且未完成),直接返回等待队列 '''
        can_go_on_sign = True
        next_operation = None
        if operation.name in dependency_relationship[stage_kind]:
            if self._operation_dependency_finished(global_finished_operations, dependency_relationship, operation) == False:
                can_go_on_sign = False
        if operation.stage_id == 0 and operation.name == "recv_backward" and operation.batch_id == 3:
            print("debug...")
        if operation.mg_state == "steady" and operation.name in MG_COMM_OPERATION:
            # megatron的stready阶段的通信op是2个,检查另外一个op,如果未完成,返回等待队列
            if operation.stage_id == 1 and operation.name == "send_backward" and operation.batch_id == 3:
                print("debug...")
            print(f"operation.stage_id: {operation.stage_id}, operation.name: {operation.name}, operation.batch_id: {operation.batch_id}")
            next_operation: Operation = None
            try:
                next_operation: Operation = self.waiting_queue.popleft()
                if next_operation.mg_state != "steady":
                    operation.mg_is_last_iteration = True
                    self.waiting_queue.appendleft(next_operation)
                else:
                    if next_operation.name in dependency_relationship[stage_kind]:
                        if self._operation_dependency_finished(global_finished_operations, dependency_relationship, next_operation) == False:
                            can_go_on_sign = False
                            self.waiting_queue.appendleft(next_operation)
            except:
                operation.mg_is_last_iteration = True
        if not can_go_on_sign:
            self.waiting_queue.appendleft(operation)
            print(f"{current_format_operantion_name}的依赖项未完成, 返回队首等待...")
            raise ValueError(f"现阶段是阻塞方式执行,现阶段不存在跨stage的依赖项,所有依赖项都在同一个stage上,按序被执行.所以当某个operation被pop到了,应该不会发生依赖项未完成情况.")

        # comp. op
        if (operation.name in MG_COMP_OPERATION) or (operation.name in DS_COMP_OPERATION):
            if operation.name not in dependency_relationship[stage_kind]:
                # 无依赖项,更新该operation的属性,加入timeline和global_finished_operations
                operation._comp_set_join_finish_waiting_acc_time(last_operation_time, last_operation.waiting_acc)
                # 加入timeline和global_finished_operations
                target_timeline.append(operation)
                global_finished_operations[current_format_operantion_name] = operation
                print(f"计算操作：{current_format_operantion_name}无依赖操作, 该operation已经完成属性更新,并加入到数据结构中")
            else:
                # 存在依赖项,且它必定已完成
                dependency_operation: Operation = self._operation_dependency_finished(global_finished_operations, dependency_relationship, operation)
                
                # 依赖项已经完成,查找依赖操作的finish_time,然后max(finish_time, last_operation_time),即选定依赖项和当前stage中最晚operation的结束时间
                if dependency_operation.finish_time > last_operation_time:
                    # 依赖项时间最晚(一般来说,当前场景下依赖项和当前operation在同一stage中,当前的写法使得依赖项可以跨stage)
                    operation._comp_set_join_finish_waiting_acc_time(dependency_operation.finish_time, dependency_operation.waiting_acc)
                    print(f"注意,{current_format_operantion_name}的依赖操作{dependency_operation.name}在当前timeline之前完成。即依赖项并非同一stage情况!")
                    print(f"dependency_operation.finish_time: {dependency_operation.finish_time}, last_operation_time: {last_operation_time}")
                    raise ValueError(f"当前使用阻塞方式执行, 所有operation的依赖项都在同一个stage上, 且排在之前, 理论上不会出现")
                else:
                    # 当前stage中最晚operation的时间最晚
                    operation._comp_set_join_finish_waiting_acc_time(last_operation_time, last_operation.waiting_acc)
                target_timeline.append(operation)
                # current_format_operantion_name: str = str(self.stage_id) + "_" + operation.name + "_" + str(operation.batch_id)
                global_finished_operations[current_format_operantion_name] = operation
                print(f"计算操作：{current_format_operantion_name}的依赖操作已完成, 该operation已经完成属性更新,并加入到数据结构中")

        else:
            '''comm. op
                这里要区别megatron中通信算子合并情况,仅出现在【steady阶段】,send_forward_recv_backward 和 send_backward_recv_forward 是一起出现的,因此注册和查找都是同时进行的
                注意,send_forward_recv_backward 和 send_backward_recv_forward 互为matching operaitons,需要相互配合才能执行。谁先达到谁先注册,等待对方到达进行收尾操作
                可能的情况：
                    1. 当前ops未注册, 配对的comm. ops 已经注册,更新当前ops的join_time,当前对象执行收尾操作,更新当前ops的属性,从配对的comm_waiting_pool取出该配对的comm op（根据pre/post_timeline）,然后刷新各类型属性,最后加入timeline和global_finished_operations
                    2. 当前ops未注册, 配对的comm. ops 未注册,更新当前ops的join_time, 将当前ops加入到comm_waiting_pool中等对方查询, 等待对方对象执行收尾操作
            '''
            if operation.mg_state == "steady" and not operation.mg_is_last_iteration:
                # steady的last_iteration只有send_backward
                # 2个comm. op情况：只有megatron的steady阶段,即operation：send_forward_recv_backward and send_backward_recv_forward,此时可取出下一个operation
                next_current_format_operantion_name: str = self._get_format_operation_name(operation_name=next_operation.name, stage_offset=0, batch_id=next_operation.batch_id)

                # TODO：当前所有comm的依赖项都在同一个stage上,所有last_operation_time中的值包含了依赖项完成时间。当依赖项跨stage时,需要变成max(dependency_op_finish_time, last_operation_time)
                operation._comm_set_join_time(last_operation_time)
                next_operation._comm_set_join_time(last_operation_time)

                ''' 校验对方情况, 是否注册 '''
                matching_operation_name, stage_offset = self._get_comm_matching_operation_name_and_offset(operation, comm_matching_relationship)
                next_matching_operation_name, next_stage_offset = self._get_comm_matching_operation_name_and_offset(next_operation, comm_matching_relationship)

                # 获取matching operaitons的名称(即对方2个op的名称,如果已注册,则已经在对方的pool中)
                operation_matching_format_name = self._get_format_operation_name(matching_operation_name, stage_offset, operation.batch_id)
                next_matching_format_name = self._get_format_operation_name(next_matching_operation_name, next_stage_offset, next_operation.batch_id)

                # matching operaitons 所在的 IndividualTimeline对象
                both_search_target_timeline: IndividualTimeline= self.post_individual_timeline if stage_offset > 0 else self.pre_individual_timeline

                # 根据上述信息检查对方注册情况
                if (operation_matching_format_name in both_search_target_timeline.comm_waiting_pool) and \
                                (next_matching_format_name in both_search_target_timeline.comm_waiting_pool):
                    '''类别1: 对方已经注册, 进行收尾工作'''
                    # 对方stage： |_____对方等待时间_____|———————— duration执行时间 ————————|
                    # 当前stage：                       |———————— duration执行时间 ————————|

                    # 更新对方operations的属性：
                    matching_target_op: Operation = both_search_target_timeline.comm_waiting_pool[operation_matching_format_name]
                    matching_target_next_op: Operation = both_search_target_timeline.comm_waiting_pool[next_matching_format_name]

                    # 更新对方2个operation的 waiting time and finish time (当前2个ops的join_time一致,因此只需计算一个gap即可)
                    gap_waiting_time = operation.join_time - matching_target_op.join_time
                    matching_target_op._comm_set_waiting_finish_time(gap_waiting_time)
                    matching_target_next_op._comm_set_waiting_finish_time(gap_waiting_time)

                    # 更新对方2个operation的waiting_acc
                    last_op_waiting_acc = both_search_target_timeline._get_last_op_waiting_acc(ignore_timeline_kind = True)
                    matching_target_op._comm_set_waiting_acc_time(gap_waiting_time, last_op_waiting_acc)
                    matching_target_next_op._comm_set_waiting_acc_time(gap_waiting_time, last_op_waiting_acc)

                    # 更新对方的IndividualTimeline：加入comm_timeline和global_finished_operations,并移除matching operaitons,解除对方的阻塞状态
                    both_search_target_timeline._add_comm_double_ops_to_timeline({operation_matching_format_name: matching_target_op, next_matching_format_name: matching_target_next_op}, \
                                                        global_finished_operations, need_fuse=False)
                    del both_search_target_timeline.comm_waiting_pool[operation_matching_format_name]
                    del both_search_target_timeline.comm_waiting_pool[next_matching_format_name]
                    both_search_target_timeline._set_is_blocked_sign(False)

                    # 同理,更新当前2个operation的属性
                    operation._comm_set_waiting_finish_time(0) # 当前op是刚加入的,不存在等待时间
                    next_operation._comm_set_waiting_finish_time(0)

                    last_op_waiting_acc = self._get_last_op_waiting_acc(ignore_timeline_kind = True)
                    operation._comm_set_waiting_acc_time(0, last_op_waiting_acc) # 当前op是刚加入的,不存在等待时间
                    next_operation._comm_set_waiting_acc_time(0, last_op_waiting_acc)

                    # 将当前2个operation加入到timeline中和global finished operation list中
                    self._add_comm_double_ops_to_timeline({current_format_operantion_name: operation, next_current_format_operantion_name: next_operation}, \
                                                        global_finished_operations, need_fuse=False)
                else:
                    '''类别2: 对方未注册,等待对方注册,将当前2个operation的匹配名称加入到自己的pool中,等待对方校验和收尾处理;设定当前stage的timeline处理情况为阻塞状态'''
                    self.comm_waiting_pool[current_format_operantion_name] = operation
                    self.comm_waiting_pool[next_current_format_operantion_name] = next_operation
                    self._set_is_blocked_sign(True)
            else:
                ''' 
                    1个comm. op情况:包含ds和非steady的其他阶段
                    依旧分为: 1.本地op未注册, 对方op已经注册 2. 双方op都未注册
                '''
                operation._comm_set_join_time(last_operation_time)

                # 校验对方情况, 是否注册
                matching_operation_name, stage_offset = self._get_comm_matching_operation_name_and_offset(operation, comm_matching_relationship)

                # 获取matching operaitons的名称(即对方2个op的名称,如果已注册,则已经在对方的pool中)
                operation_matching_format_name = self._get_format_operation_name(matching_operation_name, stage_offset, operation.batch_id)

                # matching operaitons 所在的 IndividualTimeline对象(matching op一定在当前timeline的上下stage中,可以根据stage_offset来判断)
                matching_search_target_timeline: IndividualTimeline= self.post_individual_timeline if stage_offset > 0 else self.pre_individual_timeline

                if operation_matching_format_name in matching_search_target_timeline.comm_waiting_pool:
                    ''' 类别1: 对方已经注册,进行收尾工作 ''' 

                    # 更新对方operations的属性：
                    matching_target_op: Operation = matching_search_target_timeline.comm_waiting_pool[operation_matching_format_name]

                    # 更新对方operation的 waiting time and finish time
                    gap_waiting_time = operation.join_time - matching_target_op.join_time
                    matching_target_op._comm_set_waiting_finish_time(gap_waiting_time)

                    # 更新对方operation的waiting_acc
                    last_op_waiting_acc = matching_search_target_timeline._get_last_op_waiting_acc(ignore_timeline_kind = True)
                    matching_target_op._comm_set_waiting_acc_time(gap_waiting_time, last_op_waiting_acc)

                    # 更新对方的IndividualTimeline：加入comm_timeline和global_finished_operations,并移除matching operaitons,解除对方的阻塞状态
                    matching_search_target_timeline._add_comm_double_ops_to_timeline({operation_matching_format_name: matching_target_op}, \
                                                        global_finished_operations, need_fuse=False)
                    del matching_search_target_timeline.comm_waiting_pool[operation_matching_format_name]
                    matching_search_target_timeline._set_is_blocked_sign(False)

                    # 同理,更新当前2个operation的属性
                    operation._comm_set_waiting_finish_time(0) # 当前op是刚加入的,不存在等待时间

                    last_op_waiting_acc = self._get_last_op_waiting_acc(ignore_timeline_kind = True)
                    operation._comm_set_waiting_acc_time(0, last_op_waiting_acc) # 当前op是刚加入的,不存在等待时间

                    # 将当前operation加入到timeline中和global finished operation list中
                    self._add_comm_double_ops_to_timeline({current_format_operantion_name: operation}, \
                                                        global_finished_operations, need_fuse=False)
                else:
                    ''' 类型2:对方未注册,等待对方注册,将当前2个operation的匹配名称加入到自己的pool中,
                        等待对方校验和收尾处理;设定当前stage的timeline处理情况为阻塞状态
                    ''' 
                    self.comm_waiting_pool[current_format_operantion_name] = operation
                    self._set_is_blocked_sign(True)


    def _set_is_blocked_sign(self, bool_value):
        self.is_blocked = bool_value

    def _add_comm_double_ops_to_timeline(self, ops_dict, global_finished_operations, need_fuse=False):
        if need_fuse:
            raise ValueError(f"Fusion of comm. ops is not supported.")
        else:
            for operation_format_name, operation in ops_dict.items():
                self.comm_timeline.append(operation)
                global_finished_operations[operation_format_name] = operation

    def _get_last_op_waiting_acc(self, ignore_timeline_kind: True):
        if ignore_timeline_kind:
            # 取出2个timeline中最后的操作的waiting_acc
            _, last_operation = self._get_last_operation_time_and_op([self.comm_timeline, self.comp_timeline])
            return last_operation.waiting_acc if last_operation else 0
        else:
            raise ValueError(f"Special kind of _get_last_op_waiting_acc is not supported.")

    def _get_format_operation_name(self, operation_name: str, stage_offset: int, batch_id: int) -> str:
        return  str(self.stage_id+stage_offset) + "_" + operation_name + "_" + str(batch_id)

    def _get_dependency_operation_name_and_offset(self, operation: Operation, dependency_relationship: dict):
        dep_operation_name, stage_offset = dependency_relationship[self.stage_kind][operation.name]
        return dep_operation_name, stage_offset

    def _get_comm_matching_operation_name_and_offset(self, operation: Operation, comm_matching_relationship:dict):
        matching_operation_name, stage_offset = comm_matching_relationship[operation.name]
        return matching_operation_name, stage_offset
    
    def _get_last_operation_time_and_op(self, check_timelines: list):
        """ 返回的是check_timelines中最迟的operation的finish time和对应的operation """
        if not check_timelines or all(not sublist for sublist in check_timelines):
            return 0, Operation(waiting_acc=0) # 返回一个空的Operation
        if len(check_timelines) == 1:
            return (check_timelines[0][-1].finish_time, check_timelines[0][-1]) if check_timelines[0] else (0, None)
        else:
            # 非overlap情况下,取出comp和comm中最后一个operation的finish_time
            last_operations = [timeline[-1] for timeline in check_timelines if timeline]
            last_operation = max(last_operations, key=lambda operation: operation.finish_time, default=None)
            return (last_operation.finish_time if last_operation else 0, last_operation)


    def _operation_dependency_finished(self, global_finished_operations: dict, dependency_relationship: dict, operation: Operation) -> Union[bool, Operation]:
        dep_operation, stage_offset = self._get_dependency_operation_name_and_offset(operation, dependency_relationship)
        # dependency_relationship[self.stage_kind][operation.name]
        target_stage_id = self.stage_id + stage_offset
        assert target_stage_id > -1, "Invalid stage offset"
        target_comb_operantion_name = str(target_stage_id) + "_" + dep_operation + "_" + str(operation.batch_id)

        if target_comb_operantion_name in global_finished_operations:
            return global_finished_operations[target_comb_operantion_name]
        else:
            # print(f"{operation.name}的依赖项{target_comb_operantion_name} not in global_finished_operations...")
            return False


class TimelinesManager:
    """ 管理所有stages的timeline """
    def __init__(self, dependency_relationship, comm_matching_relationship, stages_list: list, strategy: str = '1F1B-none_interleaved'):
        self.stages_list = sorted(stages_list, key=lambda s: s.stage_id)  # 根据stage_id进行排序
        self.strategy = strategy
        self.dependency_relationship: dict = dependency_relationship
        self.comm_matching_relationship: dict = comm_matching_relationship
        self.global_finished_operations: dict = {} # global_finished_operations = {stage_id_operation.name_operation.batch_id: Operation obj, ...}, e.g., {1_ForwardPass_0: Operation obj, 3_RecvActivation_0: Operation obj, ...
        self.stages_timeline_process_dict = self._init_stages_timeline()
        
    def _init_stages_timeline(self):
        """ 生成一个dict, key为stage_id, value为IndividualTimeline obj"""
        stages_timeline_process_dict = {}
        # 先创建所有IndividualTimeline对象并存入字典
        for stage in self.stages_list:
            stages_timeline_process_dict[stage.stage_id] = IndividualTimeline(stage)
        
        # 然后设置每个IndividualTimeline的前一个和后一个timeline
        for i, stage in enumerate(self.stages_list):
            current_timeline = stages_timeline_process_dict[stage.stage_id]
            if i > 0:  # 如果当前不是第一个stage,则设置前一个stage的timeline
                prev_stage = self.stages_list[i - 1]
                current_timeline.pre_individual_timeline = stages_timeline_process_dict[prev_stage.stage_id]
            if i < len(self.stages_list) - 1:  # 如果当前不是最后一个stage,则设置后一个stage的timeline
                next_stage = self.stages_list[i + 1]
                current_timeline.post_individual_timeline = stages_timeline_process_dict[next_stage.stage_id]
        
        return stages_timeline_process_dict

    def _stages_pipeline_parallel(self):
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

                # 对stage_id进行排序
                for stage_id in sorted(self.stages_timeline_process_dict.keys()):
                    timeline: IndividualTimeline  = self.stages_timeline_process_dict[stage_id]
                    # TODO: timeline.is_blocked应该要单独
                    if timeline.waiting_queue and not timeline.is_blocked:
                        # 如果waiting_queue非空,且当前为非阻塞状态,处理operation
                        all_queues_empty = False
                        # if :
                        operation: Operation = timeline.waiting_queue.popleft()
                        timeline._add_operation_to_timeline(operation, self.global_finished_operations, self.dependency_relationship, self.comm_matching_relationship, timeline.stage_kind, self.strategy)

                # 如果所有的waiting_queue都为空,退出循环
                if all_queues_empty:
                    break

        elif self.strategy == '1F1B-interleaved':
            pass

        elif self.strategy == 'F-then-B':
            pass

        # self._handle_final_package_operation()

    def _handle_final_package_operation(self):
        for stage_id, timeline in self.stages_timeline_process_dict.items():
            for operation in timeline.final_package_operation:
                timeline._add_operation_to_timeline(operation, self.global_finished_operations, self.dependency_relationship, self.comm_matching_relationship, timeline.stage_kind)


    # def _visualize_timelines(self):
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    #     import numpy as np

    #     # Define the number of rows needed for our subplots based on the number of stages
    #     num_stages = len(self.stages_timeline_process_dict)

    #     # Adjust the size of each subplot if there are more than 3 stages
    #     subplot_height = 2 if num_stages <= 3 else 1

    #     # Set up the figure and axes
    #     fig, axs = plt.subplots(nrows=num_stages, ncols=1, figsize=(10, num_stages * subplot_height), squeeze=False)

    #     # Determine the maximum finish time across all operations in all stages for consistent x-axis scale
    #     max_finish_time = max(
    #         [op.finish_time for timeline in self.stages_timeline_process_dict.values()
    #         for op in timeline.comp_timeline + timeline.comm_timeline],
    #         default=0
    #     )

    #     # Define a simple mapping from operation names to two-letter labels
    #     operation_labels = {
    #         'ForwardPass': 'FP',
    #         'BackwardPass': 'BP',
    #         'OptimizerStep': 'OS',
    #         'LoadMicroBatch': 'LB',
    #         'SendGrad': 'SG',
    #         'RecvGrad': 'RG',
    #         'SendActivation': 'SA',
    #         'RecvActivation': 'RA',
    #         'ReduceGrads': 'RG',
    #         'ReduceTiedGrads': 'RT',
    #         'forward_step': 'FS',
    #         'backward_step': 'BS',
    #         'recv_forward': 'RF',
    #         'send_forward': 'SF',
    #         'recv_backward': 'RB',
    #         'send_backward': 'SB'
    #     }

    #     # Plot each stage's timelines
    #     for idx, (stage_id, individual_timeline) in enumerate(self.stages_timeline_process_dict.items()):
    #         ax = axs[idx][0]

    #         # Create a patch (rectangle) for each operation in comp_timeline and comm_timeline
    #         for timeline in [individual_timeline.comp_timeline, individual_timeline.comm_timeline]:
    #             for operation in timeline:
    #                 label = f"{operation_labels.get(operation.name, 'NA')}{operation.batch_id}"  # Label with batch_id
    #                 color = 'skyblue' if operation in individual_timeline.comp_timeline else 'orange'
    #                 rect = patches.Rectangle(
    #                     (operation.join_time, 0.1 if color == 'skyblue' else -0.5),
    #                     operation.finish_time - operation.join_time,
    #                     0.4,
    #                     linewidth=1,
    #                     edgecolor='black',
    #                     facecolor=color,
    #                     label=label
    #                 )
    #                 ax.add_patch(rect)
    #                 ax.text(
    #                     (operation.join_time + operation.finish_time) / 2,
    #                     0.3 if color == 'skyblue' else -0.3,
    #                     label,
    #                     horizontalalignment='center',
    #                     verticalalignment='center',
    #                     fontsize=8  # Adjust font size for legibility
    #                 )

    #         # Formatting the subplot
    #         ax.set_xlim(0, max_finish_time)
    #         ax.set_ylim(-1, 1)
    #         ax.set_yticks([])
    #         ax.set_title(f'Stage {stage_id} Timelines')
    #         ax.set_xlabel('Time')
    #         ax.set_ylabel('Timelines')
    #         ax.label_outer()

    #     plt.tight_layout()
    #     plt.show()


    def _visualize_timelines(self):
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

        # Define a simple mapping from operation names to two-letter labels
        operation_labels = {
            'ForwardPass': 'FP', 'BackwardPass': 'BP', 'OptimizerStep': 'OS',
            'LoadMicroBatch': 'LB', 'SendGrad': 'SG', 'RecvGrad': 'RG',
            'SendActivation': 'SA', 'RecvActivation': 'RA', 'ReduceGrads': 'RG',
            'ReduceTiedGrads': 'RT', 'forward_step': 'FS', 'backward_step': 'BS',
            'recv_forward': 'RF', 'send_forward': 'SF', 'recv_backward': 'RB', 'send_backward': 'SB'
        }

        # Plot each stage's timelines
        for idx, (stage_id, individual_timeline) in enumerate(self.stages_timeline_process_dict.items()):
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
                    # If overlap, adjust y_offset, reduce the separation
                    y_offset = 0.2 if y_offset == 0 else 0
                else:
                    y_offset = 0  # Reset y_offset if no overlap
                y_positions.append(y_offset)
                last_finish_time = op.finish_time

            for op, y_pos in zip(all_operations, y_positions):
                label = f"{operation_labels.get(op.name, 'NA')}{op.batch_id}"  # Label with batch_id
                color = 'skyblue' if op in individual_timeline.comp_timeline else 'orange'
                base_y = 0.1 if color == 'skyblue' else -0.5
                rect = patches.Rectangle(
                    (op.join_time, base_y + y_pos),
                    op.finish_time - op.join_time,
                    0.2,  # Half height for overlapping items
                    linewidth=1,
                    edgecolor='black',
                    facecolor=color,
                    label=label
                )
                ax.add_patch(rect)
                ax.text(
                    (op.join_time + op.finish_time) / 2,
                    base_y + y_pos + 0.1,  # Center text in rectangle
                    label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8  # Adjust font size for legibility
                )

            # Formatting the subplot
            ax.set_xlim(0, max_finish_time)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_ylabel(f'Stage {stage_id}')  # Set y-axis label to Stage ID
            ax.set_xlabel('Time')
            ax.label_outer()

        plt.tight_layout()
        plt.show()





class SimulatorEngine():
    """ manager the whole workflow of the train simulation """
    def __init__(self, stages_num=None, stages_steps_dict=None, grad_acc=None, tmp_filename=None, framwork='megatron-lm'):
        self.stages_num = stages_num
        self.stages_steps_dict = stages_steps_dict
        self.grad_acc = grad_acc # 即megatron的num_microbatches
        self.stages_timeline = []
        self.stages_list = None
        self.stages_dict = None
        self.dependency_relationship = self._get_dependency_relationship(framwork)
        self.comm_matching_relationship = self._get_comm_matching_relationship(framwork)

        # self.stages_list = self.stages_task_init(stages_steps_dict)
        if framwork == 'deepspeed':
            self.stages_list, self.stages_dict = self.ds_handle_tmp_stages_dataset(tmp_filename)
        elif framwork == 'megatron-lm':
            self.stages_list, self.stages_dict = self.mg_handle_tmp_stages_dataset(tmp_filename)

        self.timeline_manager = TimelinesManager(dependency_relationship=self.dependency_relationship, comm_matching_relationship=self.comm_matching_relationship, stages_list = self.stages_list)

    def _get_dependency_relationship(self, framwork):
        ''' 现阶段不存在跨stages的依赖项 '''
        if framwork == "deepspeed":
            return {"FirstStage": {'ForwardPass': ['LoadMicroBatch', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0]}, 
                    "MiddleStage": {'ForwardPass': ['RecvActivation', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0], 'SendGrad': ['BackwardPass', 0]},
                    "LastStage": {'ForwardPass': ['RecvActivation', 0], 'BackwardPass': ['ForwardPass', 0], 'SendGrad': ['BackwardPass', 0]}}
        
        elif framwork == "megatron-lm":
            return {"FirstStage": {'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                    "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                    "LastStage": {'forward_step': ['recv_forward', 0], 'backward_step': ['forward_step', 0], 'send_backward': ['backward_step', 0]}}
        else:
            raise ValueError(f"The framework '{framwork}' is not supported.")
        
    def _get_comm_matching_relationship(self, framwork):
        ''' 区别于dependency_relationship, 该dict用于确认comm op之间的匹配信息,而非依赖信息'''
        if framwork == "deepspeed":
            return {'RecvGrad': ['SendGrad', 1],
                    'RecvActivation': ['SendActivation', -1],
                    'SendActivation': ['RecvActivation', 1],
                    'SendGrad': ['RecvGrad', -1]
            }
        elif framwork == "megatron-lm":
            return {'recv_backward': ['send_backward', 1],
                    'recv_forward': ['send_forward', -1],
                    'send_forward': ['recv_forward', 1],
                    'send_backward': ['recv_backward', -1]
            }
        else:
            raise ValueError(f"The framework '{framwork}' is not supported.")

    @staticmethod
    def stages_task_init(stages_steps_dict) -> list:
        """初始化stages的task,实例化Operation objs、Stage objs
        
        Args:
            stages_steps_dict: {"stage_id": {"step_id": [cmds]}, ...} # steps_num = len(stages_steps_dict["stage_id"])

            return stages_list: [Stage obj, ...]
        """
        pass

    @staticmethod
    def ds_handle_tmp_stages_dataset(filename: str):
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
                        stage_id = int(stage_prefix.split(':')[-1])
                        step_id = int(step_id_prefix.split(':')[-1])
                    except ValueError as e:
                        print(f"Error parsing line: {line} | Error: {str(e)}")
                        continue

                    cmds = parse_commands(cmds_str)

                    if stage_id not in stages_dict:
                        stages_dict[stage_id] = Stage(stage_id, 0, "deepspeed")

                    for cmd_name, kwargs in cmds:
                        kwargs.update({"stage_id": stage_id, "step_id": step_id, "duration": 1})
                        operation = Operation(name=cmd_name, **kwargs)
                        stages_dict[stage_id]._add_operations_to_list(operation)

                    stages_dict[stage_id].steps_num = max(stages_dict[stage_id].steps_num, step_id + 1)

        for i, stage in stages_dict.items():
            if i == 0:
                stage._set_stage_kind('FirstStage')
            elif i == max(stages_dict.keys()):
                stage._set_stage_kind('LastStage')
            else:
                stage._set_stage_kind('MiddleStage')

        return list(stages_dict.values()), stages_dict
    
    @staticmethod
    def mg_handle_tmp_stages_dataset(folder_path: str):
        """
        Process all txt files in the given folder path, each representing operations in a stage.
        Each line in a txt file represents an operation in the format:
        `stage_id:mg_state:operation_name:batch_id`
        Returns both a list of Stage objects and a dictionary mapping stage_ids to Stage objects.
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
                            stage_id, mg_state, operation_name, batch_id = line.split(':')
                            stage_id, batch_id = int(stage_id), int(batch_id)

                            # Create a new stage if it does not exist
                            if stage_id not in stages:
                                stages[stage_id] = Stage(stage_id, len(lines), "megatron-lm")
                            
                            # Create a new operation
                            operation = Operation(
                                name=operation_name,
                                duration=1,  # Set duration uniformly as 1 for now
                                batch_id=batch_id,
                                stage_id=stage_id,
                                mg_state=mg_state
                            )
                            
                            # Add operation to the stage
                            stages[stage_id]._add_operations_to_list(operation)

                    # Update steps_num to the total number of lines in the current file
                    stages[stage_id].steps_num = len(lines)
        
        # Set stage kinds
        stage_ids = sorted(stages.keys())
        for idx, stage_id in enumerate(stage_ids):
            if idx == 0:
                stages[stage_id]._set_stage_kind('FirstStage')
            elif idx == len(stage_ids) - 1:
                stages[stage_id]._set_stage_kind('LastStage')
            else:
                stages[stage_id]._set_stage_kind('MiddleStage')

        # Return both a list and a dictionary of stages
        return list(stages.values()), stages

    def _start_pipeline(self):
        self.timeline_manager._stages_pipeline_parallel()

    def _visualize_timelines(self):
        self.timeline_manager._visualize_timelines()



if __name__ == '__main__':
    """ 单独测试ds_handle_tmp_stages_dataset函数 """
    # stages_list, stages_dict = SimulatorEngine.ds_handle_tmp_stages_dataset(r'H:\HUBOther\ML_Sys_Merak\TorchGraph\log_pp_txt\2pp.txt')
    # print(stages_list)

    """ 单独测试mg_handle_tmp_stages_dataset函数 """
    # stages_list, stages_dict = SimulatorEngine.mg_handle_tmp_stages_dataset(r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log")
    # print(stages_list, stages_dict)


    """ 测试SimulatorEngine类 | deepspeed"""
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\log_pp_txt\8pp.txt"
    # simulator_engine = SimulatorEngine(tmp_filename=filename, framwork='deepspeed')
    # simulator_engine._start_pipeline()
    # simulator_engine._visualize_timelines()


    """ 测试SimulatorEngine类 | megatron-lm"""
    filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log"
    simulator_engine = SimulatorEngine(tmp_filename=filename, framwork='megatron-lm')
    simulator_engine._start_pipeline()
    simulator_engine._visualize_timelines()