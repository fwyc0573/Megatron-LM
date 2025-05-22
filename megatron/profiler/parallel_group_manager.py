class MPUInfo:
    def __init__(self, dp_size, tp_size, pp_size, dp_groups, pp_groups, tp_groups, mp_groups, ep_groups=None, pep_groups=None, exp_groups=None, cp_groups=None, cp_size=1):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.mp_size = pp_size * tp_size
        self.world_size = self.mp_size * dp_size * cp_size
        self.dp_groups = dp_groups
        self.pp_groups = pp_groups
        self.tp_groups = tp_groups
        self.mp_groups = mp_groups
        self.ep_groups = ep_groups
        self.pep_groups = pep_groups
        self.exp_groups = exp_groups
        self.cp_groups = cp_groups

    def __str__(self):
        return (
            f"MPUInfo:\n"
            f"\tdp_size={self.dp_size}\n"
            f"\ttp_size={self.tp_size}\n"
            f"\tpp_size={self.pp_size}\n"
            f"\tcp_size={self.cp_size}\n"
            f"\tmp_size={self.mp_size}\n"
            f"\tworld_size={self.world_size}\n"
            f"\tdp_groups={self.dp_groups}\n"
            f"\tpp_groups={self.pp_groups}\n"
            f"\ttp_groups={self.tp_groups}\n"
            f"\tmp_groups={self.mp_groups}\n"
            f"\tep_groups={self.ep_groups}\n"
            f"\tpep_groups={self.pep_groups}\n"
            f"\texp_groups={self.exp_groups}\n"
            f"\tcp_groups={self.cp_groups}"
        )
    
    def covert_stage_to_wrank_id(self, stage_id):
        """Convert pipeline stage ID to world rank ID.
        
        Args:
            stage_id: Stage ID in pipeline parallel group
            
        Returns:
            World rank ID corresponding to the stage ID
        """
        for pp_group in self.pp_groups:
            if stage_id < len(pp_group):
                return pp_group[stage_id]
        return None
    
    def covert_wrank_to_stage_id(self, wrank_id):
        """Convert world rank ID to pipeline stage ID.
        
        Args:
            wrank_id: World rank ID
            
        Returns:
            Stage ID in pipeline parallel group
        """
        for pp_group in self.pp_groups:
            if wrank_id in pp_group:
                return pp_group.index(wrank_id)
        return None


class ParallelGroupManager:
    def __init__(self, local_size, world_size, pp_size, tp_size, dp_size=None, cp_size=1, exp_size=1):
        """Initialize the parallel group manager.
        
        Args:
            local_size: Number of GPUs per node
            world_size: Total number of GPUs
            pp_size: Pipeline parallel size
            tp_size: Tensor parallel size
            dp_size: Data parallel size (calculated if None)
            cp_size: Context parallel size (default: 1)
            exp_size: Expert parallel size (default: 1)
        """
        self.local_size = local_size
        self.world_size = world_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.cp_size = cp_size if cp_size is not None else 1
        self.exp_size = exp_size if exp_size is not None else 1
        self.mp_size = pp_size * tp_size
        
        # Calculate data parallel size if not provided
        if dp_size is None:
            self.dp_size = self.world_size // (self.mp_size * self.cp_size)
        else:
            self.dp_size = dp_size
            
        # Validate configuration
        expected_world_size = self.dp_size * self.pp_size * self.tp_size * self.cp_size
        if self.world_size != expected_world_size:
            raise ValueError(
                f"Invalid parallel configuration: world_size ({self.world_size}) != "
                f"dp_size ({self.dp_size}) * pp_size ({self.pp_size}) * tp_size ({self.tp_size}) * "
                f"cp_size ({self.cp_size})) = {expected_world_size}"
            )
            
        # Initialize groups using sim_parallel_state
        self._initialize_groups()

    def _initialize_groups(self):
        """Initialize all parallel groups using sim_parallel_state."""
        from megatron.profiler.sim_parallel_state import sim_initialize_model_parallel
        
        groups_info = sim_initialize_model_parallel(
            world_size=self.world_size,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            context_parallel_size=self.cp_size,
            expert_model_parallel_size=self.exp_size
        )
        
        # Extract groups from groups_info
        self.dp_groups = groups_info['data_parallel_global_ranks']
        self.pp_groups = groups_info['pipeline_global_ranks']
        self.tp_groups = groups_info['tensor_model_parallel_global_ranks']
        self.mp_groups = groups_info['model_parallel_group']
        self.ep_groups = groups_info['embedding_global_ranks']
        self.pep_groups = groups_info['position_embedding_global_ranks']
        self.exp_groups = groups_info['expert_model_parallel_group']
        self.cp_groups = groups_info['context_parallel_global_ranks']
        
        # Additional groups that might be useful
        self.dp_cp_groups = groups_info['data_parallel_global_ranks_with_cp']
        self.tp_dp_groups = groups_info['tensor_and_data_parallel_group']
        self.tp_dp_cp_groups = groups_info['tensor_and_data_parallel_group_with_cp']
        self.tp_exp_groups = groups_info['tensor_and_expert_parallel_group']
        self.dp_modulo_exp_groups = groups_info['data_modulo_expert_parallel_group']
        
    def get_mpu_info(self):
        """Get MPU info object containing parallel group information.
        
        Returns:
            MPUInfo object with all parallel group information
        """
        return MPUInfo(
            dp_size=self.dp_size,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            dp_groups=self.dp_groups,
            pp_groups=self.pp_groups,
            tp_groups=self.tp_groups,
            mp_groups=self.mp_groups,
            ep_groups=self.ep_groups,
            pep_groups=self.pep_groups,
            exp_groups=self.exp_groups,
            cp_groups=self.cp_groups
        )
        
    def get_dp_groups(self):
        """Get data parallel groups."""
        return self.dp_groups

    def get_pp_groups(self):
        """Get pipeline parallel groups."""
        return self.pp_groups

    def get_tp_groups(self):
        """Get tensor parallel groups."""
        return self.tp_groups
    
    def get_mp_groups(self):
        """Get model parallel groups."""
        return self.mp_groups
    
    def get_ep_groups(self):
        """Get embedding parallel groups."""
        return self.ep_groups
    
    def get_pep_groups(self):
        """Get position embedding parallel groups."""
        return self.pep_groups
    
    def get_exp_groups(self):
        """Get expert model parallel groups."""
        return self.exp_groups
    
    def get_cp_groups(self):
        """Get context parallel groups."""
        return self.cp_groups

    def get_all_groups(self):
        """Get all parallel groups as a dictionary."""
        return {
            'pp_groups': self.pp_groups,
            'tp_groups': self.tp_groups,
            'dp_groups': self.dp_groups,
            'mp_groups': self.mp_groups,
            'ep_groups': self.ep_groups,
            'pep_groups': self.pep_groups,
            'exp_groups': self.exp_groups,
            'cp_groups': self.cp_groups,
            'dp_cp_groups': self.dp_cp_groups,
            'tp_dp_groups': self.tp_dp_groups,
            'tp_dp_cp_groups': self.tp_dp_cp_groups,
            'tp_exp_groups': self.tp_exp_groups,
            'dp_modulo_exp_groups': self.dp_modulo_exp_groups,
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