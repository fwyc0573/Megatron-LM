# import torch
# import torch.fx
# from module import PipelineModule
# from simu_module import PipelineModule
import os
# from parallel_group_manager import MPUInfo, ParallelGroupManager
# from rank_manager import RankManager


class schedulingPlan():
    def __init__(self, args):
        args.simu_micro_batch_ids = {"recv_forward":-1, "forward_step":-1, "send_forward":-1, "recv_backward":-1,
                                    "backward_step":-1, "send_backward":-1, "tp_load_batch_broadcast":-1, "dp_allreduce":-1,
                                    "tp_allreduce":-1, "optimizer_step": -1, "average_mbsloss": -1}
        args.simu_stage_operations_trace = []
        # group_manager = ParallelGroupManager(local_size=args.local_size, world_size=args.world_size, pp_size=args.pipeline_model_parallel_size, tp_size=args.tensor_model_parallel_size)
        # self.all_groups = group_manager.get_all_groups()
        # rank_manager = RankManager(args, self.all_groups)
        # self.rank_instances: dict = rank_manager.get_rank_zoos()
        self.args = args
        
    def print_sheduling_meta_info(self):
        print("megatron-lm sheduling meta info:")
        print(f"micro_batch_size:{self.args.micro_batch_size}")
        print(f"num_microbatches:{self.args.num_microbatches}")
        print(f"pp size:{self.args.pipeline_model_parallel_size}, tp size: {self.args.tensor_model_parallel_size}, dp size: {self.args.data_parallel_size}, world_size: {self.args.world_size}")

    def is_pipeline_last_stage(self, stage_id):
        return stage_id == self.args.pipeline_model_parallel_size - 1
    
    def is_pipeline_first_stage(self, stage_id):
        return stage_id == 0
    
    def get_write_scheduling_plan(self, write_to_file=False):
        self.write_to_file = write_to_file
        """ reference to implementation logic of training.py in megatron-lm """
        # 每个相同stage的scheduling策略是一致的，例如rank0和rankXX(大规模下)的plan一致
        for stage_id in range(self.args.pipeline_model_parallel_size):
            iteration = 0
            while iteration < self.args.train_iters:
                self.train_step(stage_id)
                iteration += 1


    def train_step(self, stage_id: int):
        """ reference to implementation logic of training.py in megatron-lm """
        # TODO: zero要加吗
        # Set grad to zero.

        # optimizer.zero_grad()
        # args.simu_micro_batch_ids["optimizer_zero"] += 1
        # record_name = str(world_rank) + ":init:optimizer_zero:" + str(args.simu_micro_batch_ids["optimizer_zero"])
        # args.simu_stage_operations_trace.append(record_name)
        
        # 1F1B 
        forward_backward_func = self.get_forward_backward_func()
        forward_backward_func(stage_id)
        
        # optimizer.step
        self.args.simu_micro_batch_ids["optimizer_step"] += 1
        record_name = "stage:" + str(stage_id) + ":finalize:optimizer_step:" + str(self.args.simu_micro_batch_ids["optimizer_step"])
        self.args.simu_stage_operations_trace.append(record_name)
        
        # average loss
        if self.is_pipeline_last_stage(stage_id):
            # Average loss across microbatches.
            self.args.simu_micro_batch_ids["average_mbsloss"] += 1
            record_name = str(stage_id) + ":finalize:average_mbsloss:" + str(self.args.simu_micro_batch_ids["average_mbsloss"])
            self.args.simu_stage_operations_trace.append(record_name)
            
    def get_forward_backward_func(self):
        """ reference to implementation logic of schedules.py in megatron-lm """
        # pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        if self.args.pipeline_model_parallel_size > 1:
            # TODO：补充交错式调度方法
            # if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            #     forward_backward_func = forward_backward_pipelining_with_interleaving
            # else:
            forward_backward_func = self.forward_backward_pipelining_without_interleaving
        else:
            # TODO: pp==0时的逻辑？
            forward_backward_func = None
        return forward_backward_func


    def forward_backward_pipelining_without_interleaving(self, stage_id):
        """ reference to implementation logic of schedules.py in megatron-lm """
        args = self.args
        forward_only = False
        num_microbatches = self.args.num_microbatches
        num_warmup_microbatches = (
            args.pipeline_model_parallel_size
            - stage_id
            - 1
        )
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        try:
            print(f"stage_id: {stage_id}, num_microbatches: {num_microbatches}, num_warmup_microbatches: {num_warmup_microbatches}, num_microbatches_remaining: {num_microbatches_remaining}")
        except:
            print("3D group info cannot be obtained...")

        ''' ------------------------------------------- warmup state -----------------------------------------'''
        # Run warmup forward passes.
        # last stage 不存在warmup阶段的，num_warmup_microbatches 是 0；num_warmup_microbatches = PP_size - PP_rank - 1
        args.simu_state = "warmup"
        for i in range(num_warmup_microbatches):

            ''' 1. recv_forward '''
            if not self.is_pipeline_first_stage(stage_id):
                args.simu_micro_batch_ids["recv_forward"] += 1
                record_name = str(stage_id) + ":warmup:recv_forward:" + str(args.simu_micro_batch_ids["recv_forward"])
                args.simu_stage_operations_trace.append(record_name)

            ''' 2. forward_step '''
            # TODO: first stage的loadbatch耗时？
            args.simu_micro_batch_ids["forward_step"] += 1
            record_name = str(stage_id) + ":warmup:forward_step:" + str(args.simu_micro_batch_ids["forward_step"])
            args.simu_stage_operations_trace.append(record_name)

            ''' 3. send_forward '''
            if not self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["send_forward"] += 1
                record_name = str(stage_id) + ":warmup:send_forward:" + str(args.simu_micro_batch_ids["send_forward"])
                args.simu_stage_operations_trace.append(record_name)

        ''' ------------------------------------------- warmup state -----------------------------------------'''
        # 1F1B开始前获取 (last stage - 1)上send的activation
        args.simu_state = "help"
        if num_microbatches_remaining > 0:
            # nvtx_add_label_micro_batch_id = None
            if not self.is_pipeline_first_stage(stage_id):
                args.simu_micro_batch_ids["recv_forward"] += 1
                record_name = str(stage_id) + ":help:recv_forward:" + str(args.simu_micro_batch_ids["recv_forward"])
                args.simu_stage_operations_trace.append(record_name)

        ''' ------------------------------------------- steady state -----------------------------------------'''
        # Run 1F1B in steady state.
        args.simu_state = "steady"
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)

            ''' 1. forward_step '''
            args.simu_micro_batch_ids["forward_step"] += 1
            record_name = str(stage_id) + ":steady:forward_step:" + str(args.simu_micro_batch_ids["forward_step"])
            args.simu_stage_operations_trace.append(record_name)

            ''' 2. send_forward_recv_backward '''
            # send_forward_recv_backward 和 forward_step 可以并行可以异步吗？
            # Note: 串行过程，先send后recv，comm.和comp.不重叠（通信是阻塞式的，but why？）
            if forward_only:
                pass
            else:
                # nvtx_add_label_micro_batch_id = None
                if not self.is_pipeline_last_stage(stage_id):
                    args.simu_micro_batch_ids["send_forward"] += 1
                    record_name = str(stage_id) + ":steady:send_forward:" + str(args.simu_micro_batch_ids["send_forward"])
                    args.simu_stage_operations_trace.append(record_name)
                    # print(f"sd_send_fwd_recv_bwd: {record_name}")
                    args.simu_micro_batch_ids["recv_backward"] += 1
                    record_name = str(stage_id) + ":steady:recv_backward:" + str(args.simu_micro_batch_ids["recv_backward"])
                    args.simu_stage_operations_trace.append(record_name)

                ''' 3. backward_step '''
                # bwd_n进行的input需求：fwd_n的input和output + output_tensor_grad (from next stage)
                args.simu_micro_batch_ids["backward_step"] += 1
                record_name = str(stage_id) + ":steady:backward_step:" + str(args.simu_micro_batch_ids["backward_step"])
                args.simu_stage_operations_trace.append(record_name)

                ''' 4. send_backward_recv_forward '''
                if last_iteration:

                    if not self.is_pipeline_first_stage(stage_id):
                        args.simu_micro_batch_ids["send_backward"] += 1
                        record_name = str(stage_id) + ":steady:send_backward:" + str(args.simu_micro_batch_ids["send_backward"])
                        args.simu_stage_operations_trace.append(record_name)

                else:
                    if not self.is_pipeline_first_stage(stage_id):
                        args.simu_micro_batch_ids["send_backward"] += 1
                        record_name = str(stage_id) + ":steady:send_backward:" + str(args.simu_micro_batch_ids["send_backward"])
                        args.simu_stage_operations_trace.append(record_name)
                        # print(f"sd_send_bwd_recv_fwd: {record_name}")
                        args.simu_micro_batch_ids["recv_forward"] += 1
                        record_name = str(stage_id) + ":steady:recv_forward:" + str(args.simu_micro_batch_ids["recv_forward"])
                        args.simu_stage_operations_trace.append(record_name)

        ''' ------------------------------------------- steady state -----------------------------------------'''
        
        ''' ------------------------------------------- cooldown state -----------------------------------------'''
        # Run cooldown backward passes.
        args.simu_state = "cooldown"
        if not forward_only:
            for i in range(num_warmup_microbatches):
                if i == num_warmup_microbatches - 1:
                    #TODO: 后续分桶部分可能相关
                    pass

                ''' 1. recv_backward '''
                if not self.is_pipeline_last_stage(stage_id):
                    args.simu_micro_batch_ids["recv_backward"] += 1
                    record_name = str(stage_id) + ":cooldown:recv_backward:" + str(args.simu_micro_batch_ids["recv_backward"])
                    args.simu_stage_operations_trace.append(record_name)
                
                ''' 2. backward_step '''
                args.simu_micro_batch_ids["backward_step"] += 1
                record_name = str(stage_id) + ":cooldown:backward_step:" + str(args.simu_micro_batch_ids["backward_step"])
                args.simu_stage_operations_trace.append(record_name)

                ''' 3. send_backward '''
                if not self.is_pipeline_first_stage(stage_id):
                    args.simu_micro_batch_ids["send_backward"] += 1
                    record_name = str(stage_id) + ":cooldown:send_backward:" + str(args.simu_micro_batch_ids["send_backward"])
                    args.simu_stage_operations_trace.append(record_name)

            
        # TODO: 加入非overlap-reudce的 allreduce. overlap情况这里是否要主动生成分桶信息？
        # if not args.overlap_grad_reduce:
        # grad allreduce
        args.simu_micro_batch_ids["dp_allreduce"] += 1
        record_name = str(stage_id) + ":finalize:dp_allreduce:" + str(args.simu_micro_batch_ids["dp_allreduce"])
        args.simu_stage_operations_trace.append(record_name)

        # embedding allreduce
        # TODO: 编码层应该在一个单独的group，需要改
        args.simu_micro_batch_ids["dp_allreduce"] += 1
        record_name = str(stage_id) + ":finalize:dp_allreduce:" + str(args.simu_micro_batch_ids["dp_allreduce"])
        args.simu_stage_operations_trace.append(record_name)

        ''' -------------------------------------------cooldown state-----------------------------------------'''
        def write_list_to_file(stage_id, list_to_write):
            os.makedirs("./mg_scheduling_plan_log", exist_ok=True)
            filename = f"./mg_scheduling_plan_log/0708mg_stage{stage_id}_scheduling_plan.txt"
            with open(filename, 'w') as f:
                for item in list_to_write:
                    f.write(f"{item}\n")
                    
        print(f"stage_id: {stage_id}, total operations num: {len(args.simu_stage_operations_trace)}")
        write_list_to_file(stage_id, args.simu_stage_operations_trace) if self.write_to_file else 0



    # def write_to_txt(self, folder_path):
    #     import os
    #     os.makedirs(folder_path, exist_ok=True)

    #     for stage_id, sched in self.sched.items():
    #         with open(f"{folder_path}/0604_stage{stage_id}_scheduling_plan.txt", "w") as file:
    #             for step_cmds, step_id, total_steps in sched:
    #                 file.write(f"stage:{stage_id}_step_id:{step_id}_cmds:{step_cmds}\n")
    #     print("write_to_txt func. finished, folder name: ds_scheduling_plan_log")