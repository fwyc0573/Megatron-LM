class MPUInfo:
    def __init__(self, dp_size, tp_size, pp_size, dp_groups, pp_groups, tp_groups, mp_groups, ep_groups, pep_groups):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.mp_size = pp_size*tp_size
        self.world_size = self.mp_size*dp_size
        self.dp_groups = dp_groups
        self.pp_groups = pp_groups
        self.tp_groups = tp_groups
        self.mp_groups = mp_groups
        self.ep_groups = ep_groups
        self.pep_groups = pep_groups

    def __str__(self):
        return (
            f"MPUInfo:\n"
            f"\tdp_size={self.dp_size}\n"
            f"\ttp_size={self.tp_size}\n"
            f"\tpp_size={self.pp_size}\n"
            f"\tmp_size={self.mp_size}\n"
            f"\tworld_size={self.world_size}\n"
            f"\tdp_groups={self.dp_groups}\n"
            f"\tpp_groups={self.pp_groups}\n"
            f"\ttp_groups={self.tp_groups}\n"
            f"\tmp_groups={self.mp_groups}\n"
            f"\tep_groups={self.ep_groups}\n"
            f"\tpep_groups={self.pep_groups}"
        )
    
    def covert_stage_to_wrank_id(self, stage_id):
        # 请从遍历pp_groups（由多个sub list组成，每个list代表一个pp group），返回这个group[stage_id]的元素的值(stage_id即是index)
        pass
    
    def covert_wrank_to_stage_id(self, wrank_id):
        # 请从遍历pp_groups（由多个sub list组成，每个list代表一个pp group），找到wrank_id所在的group
        pass

class ParallelGroupManager:
    def __init__(self, local_size, world_size, pp_size, tp_size, dp_size=None):
        self.local_size = local_size
        self.world_size = world_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.mp_size = pp_size * tp_size
        self.dp_size = None

        # TODO: 如何决定？
        self.tp_src_rank = None
        self.dp_src_rank = None

        self.num_tp_groups: int = world_size // tp_size
        self.num_pp_groups: int = world_size // pp_size
        self.num_mp_groups: int = world_size // self.mp_size
        self.dp_groups = []
        self.pp_groups = []
        self.tp_groups = []
        self.mp_groups = []
        self.ep_groups = []
        self.pep_groups = []
        self._create_groups()

        
    def get_mpu_info(self):
        return MPUInfo(
            dp_size=self.dp_size,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            dp_groups=self.dp_groups,
            pp_groups=self.pp_groups,
            tp_groups=self.tp_groups,
            mp_groups=self.mp_groups,
            ep_groups=self.ep_groups,
            pep_groups=self.pep_groups
        )

    def _create_groups(self):
        # 假设 world_size 是 DP、PP 和 TP sizes 的乘积
        assert self.world_size % self.mp_size == 0, "ParallelGroupManager Error: world_size must be divisible by mp_size."
        self.dp_size = self.world_size // self.mp_size

        # 生成DP、PP、TP groups
        self._create_dp_groups()
        self._create_pp_groups()
        self._create_tp_groups()
        self._create_mp_groups()
        self._create_ep_groups()
        self._create_pep_groups()

    def _create_dp_groups(self):
        """
            # Build the data-parallel groups.
            for i in range(pipeline_model_parallel_size):
                start_rank = i * num_pipeline_model_parallel_groups
                end_rank = (i + 1) * num_pipeline_model_parallel_groups
                for j in range(context_parallel_size * tensor_model_parallel_size):
                    ranks = range(
                        start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                    )
                    group = torch.distributed.new_group(
                        ranks, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
                    )
                    group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                    if rank in ranks:
                        _DATA_PARALLEL_GROUP = group
                        _DATA_PARALLEL_GROUP_GLOO = group_gloo
                        _DATA_PARALLEL_GLOBAL_RANKS = ranks
                for j in range(tensor_model_parallel_size):
                    ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                    all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                    group_with_cp = torch.distributed.new_group(
                        ranks_with_cp, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
                    )
                    group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, backend="gloo")
                    if rank in ranks_with_cp:
                        _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                        _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                        _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp
        
        """
        for i in range(self.pp_size):
            start_rank = i * self.num_pp_groups
            end_rank = (i + 1) * self.num_pp_groups
            for j in range(self.tp_size):
                ranks = list(range(
                    start_rank + j, end_rank, self.tp_size
                ))
                self.dp_groups.append(ranks)


    def _create_pp_groups(self):
        """
            # Build the pipeline model-parallel groups and embedding groups
            # (first and last rank in each pipeline model-parallel group).
            for i in range(num_pipeline_model_parallel_groups):
                ranks = range(i, world_size, num_pipeline_model_parallel_groups)
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _PIPELINE_MODEL_PARALLEL_GROUP = group
                    _PIPELINE_GLOBAL_RANKS = ranks
        
        """
        for i in range(self.num_pp_groups):
            ranks = list(range(i, self.world_size, self.num_pp_groups))
            self.pp_groups.append(ranks)


    def _create_tp_groups(self):
        """
            # Build the tensor model-parallel groups.
            for i in range(num_tensor_model_parallel_groups):
                ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _TENSOR_MODEL_PARALLEL_GROUP = group
        """
        for i in range(self.num_tp_groups):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            self.tp_groups.append(ranks)

    def _create_mp_groups(self):
        """
            # Build the model-parallel groups.
            for i in range(data_parallel_size * context_parallel_size):
                ranks = [
                    data_parallel_group_ranks_with_cp[i]
                    for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
                ]
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _MODEL_PARALLEL_GROUP = group
        """
        # Note: Currently, we don't support context-paralel.
        all_data_parallel_group_ranks_with_cp = self.dp_groups[:]
        for i in range(self.dp_size):
            ranks = [
                data_parallel_group_ranks_with_cp[i]
                for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
            ]
            self.mp_groups.append(ranks)

    def _create_ep_groups(self):
        """
        for ranks in rank_generator.get_ranks('pp'):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks
        
            group = torch.distributed.new_group(
                embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
            )
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = group
            if rank in ranks:
                _EMBEDDING_GLOBAL_RANKS = embedding_ranks
                
            group = torch.distributed.new_group(
                position_embedding_ranks,
                timeout=timeout,
                pg_options=get_nccl_options('embd', nccl_comm_cfgs),
            )
            if rank in position_embedding_ranks:
                _POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks
        """
        for ranks in self.pp_groups:
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                pipeline_model_parallel_split_rank = None  # Replace with actual split rank if applicable

                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
            else:
                embedding_ranks = ranks

            self.ep_groups.append(embedding_ranks)

    def _create_pep_groups(self):
        for ranks in self.pp_groups:
            position_embedding_ranks = [ranks[0]]
            pipeline_model_parallel_split_rank = None  # Replace with actual split rank if applicable

            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]

            self.pep_groups.append(position_embedding_ranks)

        
    def get_dp_groups(self):
        return self.dp_groups

    def get_pp_groups(self):
        return self.pp_groups

    def get_tp_groups(self):
        return self.tp_groups
    
    def get_mp_groups(self):
        return self.mp_groups
    
    def get_ep_groups(self):
        return self.ep_groups
    
    def get_pep_groups(self):
        return self.pep_groups

    def get_all_groups(self):
        return {
            'pp_groups': self.pp_groups,
            'tp_groups': self.tp_groups,
            'dp_groups': self.dp_groups,
            'mp_groups': self.mp_groups,
            'ep_groups': self.ep_groups,
            'pep_groups': self.pep_groups,
        }
    

"""
    megatron原来的组织方式如下,rank对应的某个group中的序号,即local_rank
    PP=2 TP=2 DP=1


    current_rank: 0
    pp_rank: 0
    pp_global_ranks: [0, 2]
    pp_next_rank: 2
    pp_last_rank: 2
    tp_rank: 0
    tp_src_rank: 0
    tp_global_ranks: [0, 1]
    dp_rank: 0
    dp_src_rank: [0]

    current_rank: 2
    pp_rank: 1
    pp_global_ranks: [0, 2]
    pp_next_rank: 0
    pp_last_rank: 2
    tp_rank: 0
    tp_src_rank: 2
    tp_global_ranks: [2, 3]
    dp_rank: 0
    dp_src_rank: [2]
        
    current_rank: 1
    pp_rank: 0
    pp_global_ranks: [1, 3]
    pp_next_rank: 3
    pp_last_rank: 3
    tp_rank: 1
    tp_src_rank: 0
    tp_global_ranks: [0, 1]
    dp_rank: 0
    dp_src_rank: [1]

    current_rank: 3
    pp_rank: 1
    pp_global_ranks: [1, 3]
    pp_next_rank: 1
    pp_last_rank: 3
    tp_rank: 1
    tp_src_rank: 2
    tp_global_ranks: [2, 3]
    dp_rank: 0
    dp_src_rank: [3]

"""