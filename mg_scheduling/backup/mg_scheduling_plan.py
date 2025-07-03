import os
import torch

class CMD:
    def __init__(self, stage_id, mg_state, cmd, batch_id, duration=None, description=None, group_kind=None, input__shape=None, input__dtype=None):
        self.stage_id = stage_id
        self.mg_state = mg_state
        self.cmd = cmd
        self.batch_id = batch_id
        self.duration = duration
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

    def __str__(self):
        return f"stage:{self.stage_id}:{self.cmd}(batch_id={self.batch_id}, mg_state={self.mg_state}, duration={self.duration}, description={self.description}, group_kind={self.group_kind}, input__shape={self.input__shape}, input__dtype={self.input__dtype})"


class SchedulingPlan:
    def __init__(self, args):
        args.simu_micro_batch_ids = {
            "recv_forward": -1, "forward_step": -1, "send_forward": -1, "recv_backward": -1,
            "backward_step": -1, "send_backward": -1, "tp_load_batch_broadcast": -1, "dp_allreduce": -1,
            "tp_allreduce": -1, "optimizer_step": -1, 'get_batch': -1, 'loss_func': -1, 'ep_allreduce': -1
        }
        self.initial_simu_micro_batch_ids = args.simu_micro_batch_ids.copy()
        self.stage_operations_trace = {}
        self.args = args
        self.args.pipeline_dtype = torch.float32
        if args.fp16:
            self.args.pipeline_dtype = torch.float16
            
    def print_scheduling_meta_info(self):
        print("megatron-lm scheduling meta info:")
        print(f"micro_batch_size:{self.args.micro_batch_size}")
        print(f"num_microbatches:{self.args.num_microbatches}")
        print(f"pp size:{self.args.pipeline_model_parallel_size}, tp size: {self.args.tensor_model_parallel_size}, dp size: {self.args.data_parallel_size}, world_size: {self.args.world_size}")
        print(f"seq_length:{self.args.seq_length}, hidden_size:{self.args.hidden_size}, pipeline_dtype: {self.args.pipeline_dtype}")


    def is_pipeline_last_stage(self, stage_id):
        return stage_id == self.args.pipeline_model_parallel_size - 1

    def is_pipeline_first_stage(self, stage_id):
        return stage_id == 0

    def get_write_scheduling_plan(self, write_to_file=False):
        self.write_to_file = write_to_file
        for stage_id in range(self.args.pipeline_model_parallel_size):
            iteration = 0
            self.stage_operations_trace[stage_id] = []
            while iteration < self.args.train_iters:
                # self.stage_operations_trace[stage_id] = []
                if iteration < self.args.trace_start:
                    self.stage_operations_trace[stage_id] = []
                self.args.simu_micro_batch_ids = self.initial_simu_micro_batch_ids.copy()  # Reset batch IDs
                self.train_step(stage_id)
                iteration += 1
            if self.write_to_file: # and self.args.trace_start >= iteration
                self.write_list_to_file(stage_id, self.stage_operations_trace[stage_id])

    def get_tensor_shapes(self):
        # TODO: add T5 and CP.
        tensor_shapes = []
        # seq_length = seq_length // parallel_state.get_context_parallel_world_size()
        tensor_shapes.append((self.args.seq_length, self.args.micro_batch_size, self.args.hidden_size))
        
        return tensor_shapes

    def train_step(self, stage_id: int):
        forward_backward_func = self.get_forward_backward_func()
        forward_backward_func(stage_id)
        
        self.args.simu_micro_batch_ids["optimizer_step"] += 1
        cmd = CMD(stage_id, "finalize", "optimizer_step", self.args.simu_micro_batch_ids["optimizer_step"])
        self.stage_operations_trace[stage_id].append(str(cmd))
        
        # if self.is_pipeline_last_stage(stage_id):
        #     self.args.simu_micro_batch_ids["calcu_loss"] += 1
        #     cmd = CMD(stage_id, "finalize", "calcu_loss", self.args.simu_micro_batch_ids["calcu_loss"])
        #     self.stage_operations_trace[stage_id].append(str(cmd))

    def get_forward_backward_func(self):
        if self.args.pipeline_model_parallel_size > 1:
            forward_backward_func = self.forward_backward_pipelining_without_interleaving
        else:
            forward_backward_func = None
        return forward_backward_func

    def forward_backward_pipelining_without_interleaving(self, stage_id):
        args = self.args
        forward_only = False
        num_microbatches = self.args.num_microbatches
        num_warmup_microbatches = args.pipeline_model_parallel_size - stage_id - 1
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        print(f"stage_id: {stage_id}, num_microbatches: {num_microbatches}, num_warmup_microbatches: {num_warmup_microbatches}, num_microbatches_remaining: {num_microbatches_remaining}")

        args.simu_state = "warmup"
        for i in range(num_warmup_microbatches):
            if not self.is_pipeline_first_stage(stage_id):
                args.simu_micro_batch_ids["recv_forward"] += 1
                cmd = CMD(stage_id, args.simu_state, "recv_forward", args.simu_micro_batch_ids["recv_forward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                self.stage_operations_trace[stage_id].append(str(cmd))

            if self.is_pipeline_first_stage(stage_id) or self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["get_batch"] += 1
                cmd = CMD(stage_id, args.simu_state, "get_batch", args.simu_micro_batch_ids["get_batch"])
                self.stage_operations_trace[stage_id].append(str(cmd))

            args.simu_micro_batch_ids["forward_step"] += 1
            cmd = CMD(stage_id, args.simu_state, "forward_step", args.simu_micro_batch_ids["forward_step"])
            self.stage_operations_trace[stage_id].append(str(cmd))

            if self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["loss_func"] += 1
                cmd = CMD(stage_id, args.simu_state, "loss_func", args.simu_micro_batch_ids["loss_func"])
                self.stage_operations_trace[stage_id].append(str(cmd))

            if not self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["send_forward"] += 1
                cmd = CMD(stage_id, args.simu_state, "send_forward", args.simu_micro_batch_ids["send_forward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                self.stage_operations_trace[stage_id].append(str(cmd))

        args.simu_state = "help"
        if num_microbatches_remaining > 0:
            if not self.is_pipeline_first_stage(stage_id):
                args.simu_micro_batch_ids["recv_forward"] += 1
                cmd = CMD(stage_id, args.simu_state, "recv_forward", args.simu_micro_batch_ids["recv_forward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                self.stage_operations_trace[stage_id].append(str(cmd))

        args.simu_state = "steady"
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)

            if self.is_pipeline_first_stage(stage_id) or self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["get_batch"] += 1
                cmd = CMD(stage_id, args.simu_state, "get_batch", args.simu_micro_batch_ids["get_batch"])
                self.stage_operations_trace[stage_id].append(str(cmd))

            args.simu_micro_batch_ids["forward_step"] += 1
            cmd = CMD(stage_id, args.simu_state, "forward_step", args.simu_micro_batch_ids["forward_step"])
            self.stage_operations_trace[stage_id].append(str(cmd))

            if self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["loss_func"] += 1
                cmd = CMD(stage_id, args.simu_state, "loss_func", args.simu_micro_batch_ids["loss_func"])
                self.stage_operations_trace[stage_id].append(str(cmd))
                
            if not self.is_pipeline_last_stage(stage_id):
                args.simu_micro_batch_ids["send_forward"] += 1
                cmd = CMD(stage_id, args.simu_state, "send_forward", args.simu_micro_batch_ids["send_forward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                self.stage_operations_trace[stage_id].append(str(cmd))

                args.simu_micro_batch_ids["recv_backward"] += 1
                cmd = CMD(stage_id, args.simu_state, "recv_backward", args.simu_micro_batch_ids["recv_backward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                self.stage_operations_trace[stage_id].append(str(cmd))

            args.simu_micro_batch_ids["backward_step"] += 1
            cmd = CMD(stage_id, args.simu_state, "backward_step", args.simu_micro_batch_ids["backward_step"])
            self.stage_operations_trace[stage_id].append(str(cmd))

            if last_iteration:
                if not self.is_pipeline_first_stage(stage_id):
                    args.simu_micro_batch_ids["send_backward"] += 1
                    cmd = CMD(stage_id, args.simu_state, "send_backward", args.simu_micro_batch_ids["send_backward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                    self.stage_operations_trace[stage_id].append(str(cmd))
            else:
                if not self.is_pipeline_first_stage(stage_id):
                    args.simu_micro_batch_ids["send_backward"] += 1
                    cmd = CMD(stage_id, args.simu_state, "send_backward", args.simu_micro_batch_ids["send_backward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                    self.stage_operations_trace[stage_id].append(str(cmd))

                    args.simu_micro_batch_ids["recv_forward"] += 1
                    cmd = CMD(stage_id, args.simu_state, "recv_forward", args.simu_micro_batch_ids["recv_forward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                    self.stage_operations_trace[stage_id].append(str(cmd))

        args.simu_state = "cooldown"
        if not forward_only:
            for i in range(num_warmup_microbatches):
                if not self.is_pipeline_last_stage(stage_id):
                    args.simu_micro_batch_ids["recv_backward"] += 1
                    cmd = CMD(stage_id, args.simu_state, "recv_backward", args.simu_micro_batch_ids["recv_backward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                    self.stage_operations_trace[stage_id].append(str(cmd))
                
                args.simu_micro_batch_ids["backward_step"] += 1
                cmd = CMD(stage_id, args.simu_state, "backward_step", args.simu_micro_batch_ids["backward_step"])
                self.stage_operations_trace[stage_id].append(str(cmd))

                if not self.is_pipeline_first_stage(stage_id):
                    args.simu_micro_batch_ids["send_backward"] += 1
                    cmd = CMD(stage_id, args.simu_state, "send_backward", args.simu_micro_batch_ids["send_backward"], group_kind='pp', input__shape=self.get_tensor_shapes()[0], input__dtype=self.args.pipeline_dtype)
                    self.stage_operations_trace[stage_id].append(str(cmd))

        args.simu_state = "finalize"
        args.simu_micro_batch_ids["dp_allreduce"] += 1
        cmd = CMD(stage_id, args.simu_state, "dp_allreduce", args.simu_micro_batch_ids["dp_allreduce"], description='model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas', group_kind='dp')
        self.stage_operations_trace[stage_id].append(str(cmd))
        
        if self.is_pipeline_first_stage(stage_id) or self.is_pipeline_last_stage(stage_id):
            args.simu_micro_batch_ids["ep_allreduce"] += 1
            cmd = CMD(stage_id, args.simu_state, "ep_allreduce", args.simu_micro_batch_ids["ep_allreduce"], description='_allreduce_embedding_grads, All-reduce embedding grads (for pipeline parallelism)', group_kind='ep')
            self.stage_operations_trace[stage_id].append(str(cmd))


    def write_list_to_file(self, stage_id, list_to_write):
        os.makedirs("./mg_scheduling_plan_log", exist_ok=True)
        filename = f"./mg_scheduling_plan_log/0712mg_stage{stage_id}_scheduling_plan.txt"
        with open(filename, 'w') as f:
            for item in list_to_write:
                f.write(f"{item}\n")
