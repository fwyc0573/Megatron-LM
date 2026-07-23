import copy
# import queue
from collections import deque
from typing import Union
# from deepspeed.utils import logger
from StaticGraphs.rank_manager import RankZoo

DS_COMP_OPERATION = ['ForwardPass', 'BackwardPass', 'OptimizerStep', 'LoadMicroBatch']
DS_COMM_OPERATION = ['SendGrad', 'RecvGrad', 'SendActivation', 'RecvActivation','ReduceGrads', 'ReduceTiedGrads']
DS_FINAL_OPERATION = ['ReduceGrads', 'ReduceTiedGrads', 'OptimizerStep']


# MG_COMP_OPERATION = ['forward_step', 'backward_step', 'load_batch']
MG_COMP_OPERATION = ['forward_step', 'backward_step']
MG_COMM_OPERATION = ['recv_forward', 'send_forward', 'recv_backward', 'send_backward']



class Operation:
    def __init__(self, name="", duration=-1, buffer_id=-1, step_id=-1, batch_id=-1, stage_id=-1, mg_state=None, op_kind=None, waiting_acc=None):
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
        self.stage_id = stage_id
        self.mg_state = mg_state # warmup/steady/cooldown/help
        self.mg_is_last_iteration = False
        self.ds_buffer_id = buffer_id
        self.ds_step_id = step_id 
    
    def __str__(self):
        return f"Operation(name={self.name}, duration={self.duration}, waiting_time={self.waiting_time}, waiting_acc={self.waiting_acc}, join_time={self.join_time}, finish_time={self.finish_time}, op_kind={self.op_kind}, pre_op={self.pre_op}, post_op={self.post_op}, batch_id={self.batch_id}, stage_id={self.stage_id}, mg_state={self.mg_state}, mg_is_last_iteration={self.mg_is_last_iteration}, ds_buffer_id={self.ds_buffer_id}, ds_step_id={self.ds_step_id})"

    def _comp_set_join_finish_waiting_acc_time(self, join_time, waiting_acc):
        self.join_time = join_time
        self.finish_time = join_time + self.duration
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
    def __init__(self, stage_id, rank: RankZoo, steps_num, framework="megatron-lm"):
        self.stage_id = stage_id # i.e., world_rank (唯一id)
        self.rank: RankZoo = rank
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
        self.stage_id = stage.stage_id # i.e., world_rank (唯一id)
        self.pre_stage = stage.pre_stage
        self.post_stage = stage.post_stage
        self.stage_kind = stage.stage_kind
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
                  strategy: str='1F1B-none_interleaved', can_overlap=False, global_waiting_pool={}, global_finished_operations={}):
        self.stages_list = sorted(stages_list, key=lambda s: s.stage_id)  # 根据stage_id进行排序
        self.strategy = strategy
        self.can_overlap = can_overlap
        self.dependency_relationship: dict = dependency_relationship
        self.comm_matching_relationship: dict = comm_matching_relationship
        self.global_waiting_pool: dict = global_waiting_pool # 同类型之间的操作不重叠，因此暂时维护一个全局的comm pool
        self.global_finished_operations: dict = global_finished_operations # global_finished_operations = {stage_id_operation.name_operation.batch_id: Operation obj, ...}, e.g., {1_ForwardPass_0: Operation obj, 3_RecvActivation_0: Operation obj, ...
        self.stages_timeline_process_dict = self._init_stages_timeline()
        
    def _init_stages_timeline(self):
        """ 生成一个dict, key为stage_id, value为IndividualTimeline obj"""
        stages_timeline_process_dict = {}
        # 先创建所有IndividualTimeline对象并存入字典
        for stage in self.stages_list:
            stages_timeline_process_dict[stage.stage_id] = IndividualTimeline(stage=stage, can_overlap=self.can_overlap)
        
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

                # 对stage_id进行排序
                for stage_id in sorted(self.stages_timeline_process_dict.keys()):
                    timeline: IndividualTimeline  = self.stages_timeline_process_dict[stage_id]

                    if timeline.waiting_queue:
                        all_queues_empty = False

                        # 检查当前timeline是否blocked
                        operation: Operation = timeline.waiting_queue.popleft()
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
        # print()
        # print(f"当前的waiting_pool: {self.global_waiting_pool}")
        # print(f"当前的finished ops: {self.global_finished_operations}")
        print(f"当前处理的 operation: {operation}, stage_id: {timeline.stage_id}, stage_kind: {timeline.stage_kind}")
        if operation.name == "send_backward" and operation.stage_id == 1 and operation.batch_id== 3:
            print("debug")
        # print()
        if operation.name in self.dependency_relationship[timeline.stage_kind]:
            dependency_operation = self._operation_dependency_finished(operation=operation, stage_id=timeline.stage_id, stage_kind=timeline.stage_kind)
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
                print(f"operation.stage_id: {operation.stage_id}, operation.name: {operation.name}, operation.batch_id: {operation.batch_id}")
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
                            next_dependency_operation = self._operation_dependency_finished(operation=next_operation, stage_id=timeline.stage_id, stage_kind=timeline.stage_kind)
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
            print(f"stage_id: {timeline.stage_id}, stage_kind: {timeline.stage_kind}")
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
        # print(f"dependency_operation:{dependency_operation}")
        # print(f"timeline_last_operation:{timeline_last_operation}")
        last_operation, last_operation_time = self._get_last_operation_and_finshed_time(timeline_last_operation, dependency_operation) 
        
        # last_operation: Operation = timeline_last_operation if (not dependency_operation or dependency_operation.finish_time <= timeline_last_operation.finish_time) else dependency_operation
        # last_operation_time: float = last_operation.finish_time if last_operation else 0
        print(f"last_operation-> {last_operation}")
        print(f"last_operation_time-> {last_operation_time}")

        # 当前op的格式化name
        current_format_operantion_name: str = self._get_format_operation_name(stage_id=timeline.stage_id, operation_name=operation.name, batch_id=operation.batch_id)

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
                # current_format_operantion_name: str = str(self.stage_id) + "_" + operation.name + "_" + str(operation.batch_id)
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

                    operation_format_name = self._get_format_operation_name(stage_id=operation.stage_id, operation_name=operation.name, batch_id=operation.batch_id)
                    next_operation_format_name = self._get_format_operation_name(stage_id=next_operation.stage_id, operation_name=next_operation.name, batch_id=next_operation.batch_id)
                    print(f"通信操作: p2p_fused {operation_format_name}和{next_operation_format_name}进行收尾操作.")
                else:
                    '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                    operation_format_name = self._get_format_operation_name(stage_id=operation.stage_id, operation_name=operation.name, batch_id=operation.batch_id)
                    next_operation_format_name = self._get_format_operation_name(stage_id=next_operation.stage_id, operation_name=next_operation.name, batch_id=next_operation.batch_id)
                    self.global_waiting_pool[operation_format_name] = operation
                    self.global_waiting_pool[next_operation_format_name] = next_operation
                    timeline._set_is_blocked_sign(True)
                    print(f"通信操作: p2p_fused {operation_format_name}和{next_operation_format_name}注册并等待.")

            else:
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
                        '''其余comm op都已注册, 因此开始nccl comm, 更新属性和状态'''
                        # 获取matching comm op对象的list
                        # comm_matching_operation_list: list = self.global_waiting_pool[key_in_waiting_pool]

                        # 进行nccl comm => 计算 comm op的 duration 数值
                        # TODO： 每个comm op的 duration 该如何确定？在哪里初始化？
                        _ = self._calculate_comm_duration(comm_matching_operation_list + [operation])

                        # 更新matching comm ops的属性,
                        _ = self._update_matching_comm_ops_properties(comm_matching_operation_list=comm_matching_operation_list, 
                                                                        comm_matching_operation_format_name_list=comm_matching_operation_format_name_list,
                                                                        current_operation=operation, comm_kind=comm_kind, parallel_kind=parallel_kind, 
                                                                        current_timeline=timeline)
                        # 更新current operation(1个)的属性
                        _ = self._update_current_comm_op_properties(comm_op=operation, current_timeline=timeline)

                        operation_format_name = self._get_format_operation_name(stage_id=operation.stage_id, operation_name=operation.name, batch_id=operation.batch_id)
                        print(f"通信操作: p2p {operation_format_name}进行收尾操作.")
                    else:
                        '''依旧存在其他comm op 未注册, 进行当前op的注册并等待'''
                        operation_format_name = self._get_format_operation_name(stage_id=operation.stage_id, operation_name=operation.name, batch_id=operation.batch_id)
                        self.global_waiting_pool[operation_format_name] = operation
                        timeline._set_is_blocked_sign(True)

                        print(f"通信操作: p2p {operation_format_name}注册并等待.")

                elif comm_kind == "allreduce" and parallel_kind == "tp":
                    pass
                elif comm_kind == "allreduce" and parallel_kind == "dp":
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

        # 生成p2p group的formatname list
        for operation in operation_list:
            matching_operation_name, matching_operation_stage_id_list, _ = self._get_comm_matching_operation_name_and_stageId(operation=operation, 
                                                                                comm_kind=comm_kind, parallel_kind=parallel_kind, timeline=timeline)
            for matching_operation_stage_id in matching_operation_stage_id_list:
                matching_operation_format_name = self._get_format_operation_name(stage_id=matching_operation_stage_id, 
                                                                                operation_name=matching_operation_name, batch_id=operation.batch_id)
                if matching_operation_format_name not in self.global_waiting_pool:
                    return False, None, None
                else:
                    matching_operation_list.append(self.global_waiting_pool[matching_operation_format_name])
                    matching_operation_format_name_list.append(matching_operation_format_name)
        return True, matching_operation_list, matching_operation_format_name_list


    def _update_current_comm_op_properties(self, comm_op: Operation, current_timeline: IndividualTimeline):
        # 1. 更新waiting time/finish time, 当前op是刚加入的,不存在等待时间
        comm_op._comm_set_waiting_finish_time(0)

        # 2. 加入到timeline中
        current_timeline._add_comm_op_to_timeline([comm_op])

        # 3. global_finished_operations
        operation_format_name = self._get_format_operation_name(stage_id=comm_op.stage_id, operation_name=comm_op.name, batch_id=comm_op.batch_id)
        self.global_finished_operations[operation_format_name] = comm_op


    def _update_matching_comm_ops_properties(self, comm_matching_operation_list: list, comm_matching_operation_format_name_list: list, 
                                                                        current_operation: Operation, comm_kind: str, parallel_kind: str, 
                                                                        current_timeline: IndividualTimeline):
        
        for matching_operation in comm_matching_operation_list:
            # 1. 更新macthing comm ops的 waiting time/finish time
            # print(f"current_operation: {current_operation}")
            # print(f"matching_operation: {matching_operation}")
            gap_waiting_time = current_operation.join_time - matching_operation.join_time
            matching_operation._comm_set_waiting_finish_time(gap_waiting_time)

            # 获取stage_offset从而得到matching_op_timeline；获取target_stage_id从而获取operation_matching_format_name
            # _, matching_operation_stage_id, stage_offset = self._get_comm_matching_operation_name_and_stageId(operation=current_operation, 
            #                                                                                       comm_kind=comm_kind, parallel_kind=parallel_kind, 
            #                                                                                       timeline=current_timeline)
            # matching_op_timeline: IndividualTimeline= current_timeline.post_individual_timeline if stage_offset > 0 else current_timeline.pre_individual_timeline
            # TODO: 检查一下
            matching_operation_stage_id = matching_operation.stage_id
            matching_op_timeline: IndividualTimeline = current_timeline.post_individual_timeline if matching_operation_stage_id > current_timeline.stage_id  \
                                                                                                else current_timeline.pre_individual_timeline

            # 2. 更新global_finished_operations
            operation_matching_format_name = self._get_format_operation_name(stage_id=matching_operation_stage_id, 
                                                                             operation_name=matching_operation.name, 
                                                                             batch_id=matching_operation.batch_id)
            self.global_finished_operations[operation_matching_format_name] = matching_operation

            # 3. 加入到timeline中
            matching_op_timeline._add_comm_op_to_timeline([matching_operation])

        # 4. 更新self.global_waiting_pool,遍历comm_matching_operation_format_name_list，删除global_waiting_pool中的key-value
        for operation_format_name in comm_matching_operation_format_name_list:
            self.global_waiting_pool.pop(operation_format_name, None)

        # 5. 更新blocked状态
        matching_op_timeline._set_is_blocked_sign(False)



    def _calculate_comm_duration(self, comm_op_list):
        """ 进行NCCL comm, 更新comm op的duration """
        pass


    def _get_comm_matching_operation_name_and_stageId(self, operation: Operation, comm_kind: str, parallel_kind: str, timeline: IndividualTimeline):
        """
            返回matching的op_name, stage_id or stage_id list
        """
        if (comm_kind == "p2p" or comm_kind == "p2p_fused") and parallel_kind == "pp":
            matching_operation_name, stage_offset = self.comm_matching_relationship[operation.name]
            if stage_offset == 1:
                target_stage_id = timeline.stage_rank._get_pp_next_world_rank()
            elif stage_offset == -1:
                target_stage_id = timeline.stage_rank._get_pp_previous_world_rank()
            else:
                raise ValueError(f"Invalid stage_offset value: {stage_offset}")
            
            # p2p类型返回的target_stage_id是单一值
            return matching_operation_name, [target_stage_id], stage_offset
        
        elif parallel_kind == "tp":
            # 获取 tp group stage_id list
            if comm_kind == "broadcast" or comm_kind == "allreduce":
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.tp_groups if rank_id != timeline.stage_id], None
            else:
                raise ValueError(f"Invalid comm_kind name: {comm_kind}")
        
        elif parallel_kind == "dp" and comm_kind == "allreduce":
            # 获取 dp group stage_id list
                return self.comm_matching_relationship[operation.name][0], [rank_id for rank_id in 
                                                                            timeline.stage_rank.dp_groups if rank_id != timeline.stage_id], None
        else:
            raise ValueError(f"Invalid comm_kind name: {comm_kind} and parallel_kind name: {parallel_kind}")


    # TODO: 判定方式修改一下，可以根据name直接拆分
    def _get_comm_operation_kind_and_parallel_dimension(self, operation: Operation):
        if operation.mg_state == "steady" and operation.name in ['recv_forward', 'send_forward', 'recv_backward', 'send_backward']:
            return "p2p_fused", "pp"
        elif operation.mg_state != "steady" and operation.name in ['recv_forward', 'send_forward', 'recv_backward', 'send_backward']:
            return "p2p", "pp"
        elif "allreduce" in operation.name:
            if "tp" in operation.name:
                return "allreduce", "tp"
            elif "dp" in operation.name:
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


    # def _get_format_operation_name(self, stage_id:int, operation_name: str, stage_offset: int, batch_id: int) -> str:
    #     return str(stage_id+stage_offset) + "_" + operation_name + "_" + str(batch_id)


    def _get_format_operation_name(self, stage_id:int, operation_name: str, batch_id: int) -> str:
        return str(stage_id) + "_" + operation_name + "_" + str(batch_id)


    def _get_dependency_operation_name_and_offset(self, operation: Operation, stage_kind:str):
        tmp_value = self.dependency_relationship[stage_kind][operation.name]
        if isinstance(tmp_value[0], list):
            # 多依赖情况，返回list
            return tmp_value
        else:
            # 单依赖情况
            return [tmp_value]

    def _operation_dependency_finished(self, operation: Operation, stage_id: int, stage_kind:str) -> Union[bool, Operation]:
        dependencies = self._get_dependency_operation_name_and_offset(operation=operation, stage_kind=stage_kind)
        latest_operation = None
        latest_finish_time = 0
        all_finished = True

        for dependency in dependencies:
            dep_operation_name, stage_offset = dependency
            target_stage_id = stage_id + stage_offset
            assert target_stage_id > -1, "Invalid stage offset"
            target_comb_operation_name = f"{target_stage_id}_{dep_operation_name}_{operation.batch_id}"

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
        for stage_id, timeline in self.stages_timeline_process_dict.items():
            for operation in timeline.final_package_operation:
                timeline._add_operation_to_timeline(operation, self.global_finished_operations, self.dependency_relationship, self.comm_matching_relationship, timeline.stage_kind)

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
                # Center text in rectangle for operation label
                ax.text(
                    (op.join_time + op.finish_time) / 2,
                    base_y + y_pos + 0.1,
                    label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8
                )
                # Add waiting_acc text below the operation
                # ax.text(
                #     (op.join_time + op.finish_time) / 2,
                #     base_y + y_pos - 0.15,  # Adjust to position below the operation rectangle
                #     f'{op.waiting_acc}',  # Display waiting_acc value
                #     horizontalalignment='center',
                #     verticalalignment='center',
                #     fontsize=8,
                #     color='black'  # Make waiting_acc value stand out
                # )

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
    def __init__(self, stages_num=None, stages_steps_dict=None, grad_acc=None, tmp_filename=None, framwork='megatron-lm', strategy="1F1B-none_interleaved",can_overlap=False):
        self.stages_num = stages_num
        self.stages_steps_dict = stages_steps_dict
        self.grad_acc = grad_acc # 即megatron的num_microbatches
        self.strategy = strategy
        self.can_overlap = can_overlap
        self.framwork = framwork
        self.tmp_filename = tmp_filename
        self.stages_timeline = []
        self.stages_list = None
        self.stages_dict = None
        self.mpu = None
        self.dependency_relationship = None
        self.comm_matching_relationship = None
        self.timeline_manager = None

        # 需要lock进行维护的2个全局变量
        self.global_waiting_pool: dict = {} 
        self.global_finished_operations: dict = {} # global_finished_operations = {stage_id_operation.name_operation.batch_id: Operation obj, ...}, e.g., {1_ForwardPass_0: Operation obj, 3_RecvActivation_0: Operation obj, ...


    def _init_tmp_stages_dataset_and_timeline_manager(self, rank_instances_dict):
        if self.framwork == 'deepspeed':
            self.stages_list, self.stages_dict = self.ds_handle_tmp_stages_dataset(self.tmp_filename)
        elif self.framwork == 'megatron-lm':
            self.stages_list, self.stages_dict = self.mg_handle_tmp_stages_dataset(self.tmp_filename, rank_instances_dict)
        
        # 每个 TimelinesManager 维护一个 pp group，data group信息交换由2个全局变量决定
        self.timeline_manager = TimelinesManager(dependency_relationship=self.dependency_relationship, 
                                                 comm_matching_relationship=self.comm_matching_relationship, 
                                                 stages_list = self.stages_list, strategy=self.strategy, can_overlap=self.can_overlap, 
                                                 global_waiting_pool=self.global_waiting_pool, global_finished_operations=self.global_finished_operations)
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
            return {"FirstStage": {'ForwardPass': ['LoadMicroBatch', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0]}, 
                    "MiddleStage": {'ForwardPass': ['RecvActivation', 0], 'SendActivation': ['ForwardPass', 0], 'BackwardPass': ['RecvGrad', 0], 'SendGrad': ['BackwardPass', 0]},
                    "LastStage": {'ForwardPass': ['RecvActivation', 0], 'BackwardPass': ['ForwardPass', 0], 'SendGrad': ['BackwardPass', 0]}}
        elif framwork == "megatron-lm":
            # TODO： 考虑到同类型的op已是按序放入到待处理队列，因此同类型操作可以忽略依赖关系？（例如对于last stage，forward_step本身就在backward_step之前）
            # 另外，不同的pipeline模式，依赖关系不同。例如对于no-pipeline模式，就不存在p2p操作，不存在多种Stage
            # return {"FirstStage": {'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
            #         "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
            #         "LastStage": {'forward_step': ['recv_forward', 0], 'backward_step': ['forward_step', 0], 'send_backward': ['backward_step', 0]}}

            # PP>1时的依赖关系
            if strategy == "1F1B-none_interleaved" and self.mpu.tp_size == 1 and self.mpu.dp_size == 1:
                # 1D pp并行: pp>1,dp==1,tp==1: 没有allreduce、没有broadcast、只涉及pp的P2P
                return {"FirstStage": {'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0]}, 
                        "MiddleStage": {'forward_step': ['recv_forward', 0], 'send_forward': ['forward_step', 0], 'backward_step': ['recv_backward', 0], 'send_backward': ['backward_step', 0]},
                        "LastStage": {'forward_step': ['recv_forward', 0], 'send_backward': ['backward_step', 0]}}
            
            elif strategy == "1F1B-none_interleaved" and self.mpu.tp_size > 1 and self.mpu.dp_size >= 1:
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
            raise ValueError(f"The framework '{framwork}' is not supported.")
        
    def _get_comm_matching_relationship(self, framwork=None, strategy=None):
        ''' 区别于dependency_relationship, 该dict用于确认comm op之间的匹配信息,而非依赖信息,匹配关系只与并行size相关'''
        if framwork == "deepspeed":
            if self.mpu.pp_size > 1:
                return {'RecvGrad': ['SendGrad', 1],
                        'RecvActivation': ['SendActivation', -1],
                        'SendActivation': ['RecvActivation', 1],
                        'SendGrad': ['RecvGrad', -1]
                }
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
    def mg_handle_tmp_stages_dataset(folder_path: str, rank_instances_dict=None):
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
                                stages[stage_id] = Stage(stage_id, rank_instances_dict[stage_id], len(lines), "megatron-lm")
                            
                            # Create a new operation
                            op_kind = "comp" if operation_name in DS_COMP_OPERATION or operation_name in MG_COMP_OPERATION else "comm" 
                            operation = Operation(
                                name=operation_name,
                                duration=1,  # Set duration uniformly as 1 for now
                                batch_id=batch_id,
                                stage_id=stage_id,
                                mg_state=mg_state,
                                op_kind=op_kind
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
        if self.mpu:
            self.timeline_manager._stages_pipeline_parallel()
        else:
            raise ValueError(f"mpu_info is not set.")

    def _visualize_timelines(self):
        self.timeline_manager._visualize_timelines()



if __name__ == '__main__':
    """ 初始化mpu设定, 新版本中必须需要的信息 """
    from parallel_group_manager import ParallelGroupManager

    manager = ParallelGroupManager(local_size=8, world_size=8, pp_size=8, tp_size=1)
    mpu_info = manager._get_mpu_info()
    print(mpu_info)

    """ 单独测试ds_handle_tmp_stages_dataset函数 """
    # stages_list, stages_dict = SimulatorEngine.ds_handle_tmp_stages_dataset(r'H:\HUBOther\ML_Sys_Merak\TorchGraph\log_pp_txt\2pp.txt')
    # print(stages_list)

    """ 单独测试mg_handle_tmp_stages_dataset函数 """
    # stages_list, stages_dict = SimulatorEngine.mg_handle_tmp_stages_dataset(r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log")
    # print(stages_list, stages_dict)


    """ 测试SimulatorEngine类 | deepspeed"""
    filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\log_pp_txt\8pp.txt"
    # simulator_engine = SimulatorEngine(tmp_filename=filename, framwork='deepspeed', strategy="1F1B-none_interleaved")
    # simulator_engine._start_pipeline()
    # simulator_engine._visualize_timelines()


    """ 测试SimulatorEngine类 | megatron-lm"""
    filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\8pp_1_1"
    simulator_engine = SimulatorEngine(tmp_filename=filename, framwork='megatron-lm', strategy="1F1B-none_interleaved")
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager()
    simulator_engine._start_pipeline()
    simulator_engine._visualize_timelines()