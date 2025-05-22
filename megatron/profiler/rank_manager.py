class RankManager:
    def __init__(self, args, groups_dict):
        """Initialize the rank manager.
        
        Args:
            args: Arguments containing configuration parameters
            groups_dict: Dictionary containing all parallel groups
        """
        self.args = args
        self.groups_dict = groups_dict
        self.rank_zoos = self._create_rank_zoos()

    def _create_rank_zoos(self):
        """Create rank zoo objects for all ranks.
        
        Returns:
            Dictionary mapping world ranks to RankZoo objects
        """
        rank_zoos = {}
        # Get cp_size from args, default to 1 if not present
        cp_size = getattr(self.args, 'fake_cp', 1)
        exp_size = getattr(self.args, 'fake_exp', 1)
        world_size = self.args.fake_pp * self.args.fake_tp * self.args.fake_dp * cp_size * exp_size
        
        for world_rank in range(world_size):
            local_rank = world_rank % self.args.fake_gpus_per_node
            server_id = world_rank // self.args.fake_gpus_per_node
            related_groups = self._find_related_groups(world_rank)
            rank_zoo = RankZoo(world_rank=world_rank, local_rank=local_rank,
                            server_id=server_id, model_config=None, **related_groups)
            rank_zoos[world_rank] = rank_zoo
        return rank_zoos

    def _find_related_groups(self, world_rank):
        """Find all groups that contain the given world rank.
        
        Args:
            world_rank: World rank to find groups for
            
        Returns:
            Dictionary mapping group types to groups containing the world rank
        """
        # Define basic groups that RankZoo expects
        related_groups = {
            'dp_groups': None, 
            'pp_groups': None, 
            'tp_groups': None, 
            'mp_groups': None,
            'ep_groups': None,
            'pep_groups': None,
            'exp_groups': None,
            'cp_groups': None,
            'dp_cp_groups': None,
            'tp_dp_groups': None,
            'tp_dp_cp_groups': None,
            'tp_exp_groups': None,
            'dp_modulo_exp_groups': None
        }
        
        # Handle groups from groups_dict that are accepted by RankZoo
        for group_type, groups in self.groups_dict.items():
            if groups is None:
                continue
                
            # Skip if group_type is not in related_groups
            if group_type not in related_groups:
                continue
                
            for group in groups:
                if world_rank in group:
                    related_groups[group_type] = group
                    
        return related_groups

    def get_rank_zoos(self):
        """Get all rank zoo objects.
        
        Returns:
            Dictionary mapping world ranks to RankZoo objects
        """
        return self.rank_zoos


class RankZoo:
    def __init__(self, world_rank, local_rank, server_id, model_config=None,
                 device_type='gpu', dp_groups=None, pp_groups=None, tp_groups=None, 
                 mp_groups=None, ep_groups=None, pep_groups=None, exp_groups=None, cp_groups=None,
                 dp_cp_groups=None, tp_dp_groups=None, tp_dp_cp_groups=None, tp_exp_groups=None,
                 dp_modulo_exp_groups=None):
        """Initialize a rank zoo object representing a single GPU rank.
        
        Args:
            world_rank: Global rank ID
            local_rank: Local rank ID within node
            server_id: Node ID
            model_config: Model configuration
            device_type: Device type (e.g., 'gpu')
            dp_groups: Data parallel group containing this rank
            pp_groups: Pipeline parallel group containing this rank
            tp_groups: Tensor parallel group containing this rank
            mp_groups: Model parallel group containing this rank
            ep_groups: Embedding parallel group containing this rank
            pep_groups: Position embedding parallel group containing this rank
            exp_groups: Expert model parallel group containing this rank
            cp_groups: Context parallel group containing this rank
            dp_cp_groups: Data parallel + context parallel group
            tp_dp_groups: Tensor parallel + data parallel group
            tp_dp_cp_groups: Tensor parallel + data parallel + context parallel group
            tp_exp_groups: Tensor parallel + expert model parallel group
            dp_modulo_exp_groups: Data parallel modulo expert model parallel group
        """
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.server_id = server_id
        self.model_config = model_config
        self.dp_groups = dp_groups
        self.pp_groups = pp_groups
        self.tp_groups = tp_groups
        self.mp_groups = mp_groups
        self.ep_groups = ep_groups
        self.pep_groups = pep_groups
        self.exp_groups = exp_groups
        self.cp_groups = cp_groups
        self.dp_cp_groups = dp_cp_groups
        self.tp_dp_groups = tp_dp_groups
        self.tp_dp_cp_groups = tp_dp_cp_groups
        self.tp_exp_groups = tp_exp_groups
        self.dp_modulo_exp_groups = dp_modulo_exp_groups
        self.device_type = device_type

    def __repr__(self):
        return (f"RankZoo(world_rank={self.world_rank}, local_rank={self.local_rank}, "
               f"server_id={self.server_id}, config={self.model_config}, "
               f"dp_groups={self.dp_groups}, pp_groups={self.pp_groups}, "
               f"tp_groups={self.tp_groups}, mp_groups={self.mp_groups}, "
               f"ep_groups={self.ep_groups}, pep_groups={self.pep_groups}, "
               f"exp_groups={self.exp_groups}, cp_groups={self.cp_groups}, "
               f"dp_cp_groups={self.dp_cp_groups}, tp_dp_groups={self.tp_dp_groups}, "
               f"tp_dp_cp_groups={self.tp_dp_cp_groups}, tp_exp_groups={self.tp_exp_groups}, "
               f"dp_modulo_exp_groups={self.dp_modulo_exp_groups})")

    def _get_dp_groups(self):
        """Get data parallel group."""
        return self.dp_groups

    def _get_dp_group_size(self):
        """Get size of data parallel group."""
        return len(self.dp_groups) if self.dp_groups else 0

    def _get_pp_group_size(self):
        """Get size of pipeline parallel group."""
        return len(self.pp_groups) if self.pp_groups else 0

    def _get_tp_group_size(self):
        """Get size of tensor parallel group."""
        return len(self.tp_groups) if self.tp_groups else 0

    def _get_mp_group_size(self):
        """Get size of model parallel group."""
        return len(self.mp_groups) if self.mp_groups else 0

    def _get_ep_group_size(self):
        """Get size of embedding parallel group."""
        return len(self.ep_groups) if self.ep_groups else 0
    
    def _get_pep_group_size(self):
        """Get size of position embedding parallel group."""
        return len(self.pep_groups) if self.pep_groups else 0
    
    def _get_exp_group_size(self):
        """Get size of expert model parallel group."""
        return len(self.exp_groups) if self.exp_groups else 0
    
    def _get_cp_group_size(self):
        """Get size of context parallel group."""
        return len(self.cp_groups) if self.cp_groups else 0
    
    def _get_dp_cp_group_size(self):
        """Get size of data parallel + context parallel group."""
        return len(self.dp_cp_groups) if self.dp_cp_groups else 0
    
    def _get_tp_dp_group_size(self):
        """Get size of tensor parallel + data parallel group."""
        return len(self.tp_dp_groups) if self.tp_dp_groups else 0
    
    def _get_tp_dp_cp_group_size(self):
        """Get size of tensor parallel + data parallel + context parallel group."""
        return len(self.tp_dp_cp_groups) if self.tp_dp_cp_groups else 0
    
    def _get_tp_exp_group_size(self):
        """Get size of tensor parallel + expert model parallel group."""
        return len(self.tp_exp_groups) if self.tp_exp_groups else 0
    
    def _get_dp_modulo_exp_group_size(self):
        """Get size of data parallel modulo expert model parallel group."""
        return len(self.dp_modulo_exp_groups) if self.dp_modulo_exp_groups else 0

    def _get_pp_next_world_rank(self):
        """Get next rank in pipeline parallel group."""
        if not self.pp_groups:
            return None
        index = self.pp_groups.index(self.world_rank)
        next_index = (index + 1) % len(self.pp_groups)
        return self.pp_groups[next_index]

    def _get_pp_previous_world_rank(self):
        """Get previous rank in pipeline parallel group."""
        if not self.pp_groups:
            return None
        index = self.pp_groups.index(self.world_rank)
        prev_index = (index - 1) % len(self.pp_groups)
        return self.pp_groups[prev_index]

    def _get_pp_local_rank(self):
        """Get local rank (stage ID) in pipeline parallel group."""
        if not self.pp_groups:
            return None
        return self.pp_groups.index(self.world_rank)

    def _get_tp_local_rank(self):
        """Get local rank in tensor parallel group."""
        if not self.tp_groups:
            return None
        return self.tp_groups.index(self.world_rank)
    
    def _get_dp_local_rank(self):
        """Get local rank in data parallel group."""
        if not self.dp_groups:
            return None
        return self.dp_groups.index(self.world_rank)

    def _get_mp_local_rank(self):
        """Get local rank in model parallel group."""
        if not self.mp_groups:
            return None
        return self.mp_groups.index(self.world_rank)
    
    def _get_exp_local_rank(self):
        """Get local rank in expert model parallel group."""
        if not self.exp_groups:
            return None
        return self.exp_groups.index(self.world_rank)
    
    def _get_cp_local_rank(self):
        """Get local rank in context parallel group."""
        if not self.cp_groups:
            return None
        return self.cp_groups.index(self.world_rank)
    
    def _get_dp_cp_local_rank(self):
        """Get local rank in data parallel + context parallel group."""
        if not self.dp_cp_groups:
            return None
        return self.dp_cp_groups.index(self.world_rank)
    
    def _get_tp_dp_local_rank(self):
        """Get local rank in tensor parallel + data parallel group."""
        if not self.tp_dp_groups:
            return None
        return self.tp_dp_groups.index(self.world_rank)
    
    def _get_tp_dp_cp_local_rank(self):
        """Get local rank in tensor parallel + data parallel + context parallel group."""
        if not self.tp_dp_cp_groups:
            return None
        return self.tp_dp_cp_groups.index(self.world_rank)
    
    def _get_tp_exp_local_rank(self):
        """Get local rank in tensor parallel + expert model parallel group."""
        if not self.tp_exp_groups:
            return None
        return self.tp_exp_groups.index(self.world_rank)
    
    def _get_dp_modulo_exp_local_rank(self):
        """Get local rank in data parallel modulo expert model parallel group."""
        if not self.dp_modulo_exp_groups:
            return None
        return self.dp_modulo_exp_groups.index(self.world_rank)

    def _convert_pp_world_to_local_rank(self, pp_world_rank):
        """Convert world rank to local rank in pipeline parallel group."""
        if not self.pp_groups or pp_world_rank not in self.pp_groups:
            return None
        return self.pp_groups.index(pp_world_rank)

    def _convert_pp_local_to_world_rank(self, pp_local_rank):
        """Convert local rank in pipeline parallel group to world rank."""
        if not self.pp_groups or pp_local_rank >= len(self.pp_groups):
            return None
        return self.pp_groups[pp_local_rank]

    def _convert_tp_world_to_local_rank(self, tp_world_rank):
        """Convert world rank to local rank in tensor parallel group."""
        if not self.tp_groups or tp_world_rank not in self.tp_groups:
            return None
        return self.tp_groups.index(tp_world_rank)

    def _convert_tp_local_to_world_rank(self, tp_local_rank):
        """Convert local rank in tensor parallel group to world rank."""
        if not self.tp_groups or tp_local_rank >= len(self.tp_groups):
            return None
        return self.tp_groups[tp_local_rank]
    
    def _convert_exp_world_to_local_rank(self, exp_world_rank):
        """Convert world rank to local rank in expert model parallel group."""
        if not self.exp_groups or exp_world_rank not in self.exp_groups:
            return None
        return self.exp_groups.index(exp_world_rank)

    def _convert_exp_local_to_world_rank(self, exp_local_rank):
        """Convert local rank in expert model parallel group to world rank."""
        if not self.exp_groups or exp_local_rank >= len(self.exp_groups):
            return None
        return self.exp_groups[exp_local_rank]

    def is_pre_process(self):
        """Check if this rank is in the first stage of pipeline."""
        pp_id = self._get_pp_local_rank()
        if pp_id == 0:
            return True
        return False

    def is_post_process(self):
        """Check if this rank is in the last stage of pipeline."""
        pp_id = self._get_pp_local_rank()
        pp_world_size = self._get_pp_group_size()
        if pp_id == pp_world_size - 1:
            return True
        return False

    def is_rank_in_embedding_group(self):
        """Check if this rank is in the embedding group."""
        if self.ep_groups is None:
            return False
        if self.world_rank in self.ep_groups:
            return True
        return False
    
    def is_rank_in_expert_group(self):
        """Check if this rank is in the expert model parallel group."""
        if self.exp_groups is None:
            return False
        if self.world_rank in self.exp_groups:
            return True
        return False

    def _get_tp_src_rank(self):
        """Get source rank in tensor parallel group."""
        if not self.tp_groups:
            return None
        return self.tp_groups[0]  # First rank in TP group is typically the source

    def _get_dp_src_rank(self):
        """Get source rank in data parallel group."""
        if not self.dp_groups:
            return None
        return self.dp_groups[0]  # First rank in DP group is typically the source



"""
local_size = 8
world_size = 16
args.nnodes = 2
pp_size = 2
tp_size = 4

manager.get_dp_groups() -> : [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]
manager.get_pp_groups() -> : [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]
manager.get_tp_groups() -> : [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
manager.get_mp_groups() -> : [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]
RankZoo(world_rank=0, local_rank=0, server_id=0, config=None, dp_groups=[[0, 4]], pp_groups=[[0, 8]], tp_groups=[[0, 1, 2, 3]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=1, local_rank=1, server_id=0, config=None, dp_groups=[[1, 5]], pp_groups=[[1, 9]], tp_groups=[[0, 1, 2, 3]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=2, local_rank=2, server_id=0, config=None, dp_groups=[[2, 6]], pp_groups=[[2, 10]], tp_groups=[[0, 1, 2, 3]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=3, local_rank=3, server_id=0, config=None, dp_groups=[[3, 7]], pp_groups=[[3, 11]], tp_groups=[[0, 1, 2, 3]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=4, local_rank=4, server_id=0, config=None, dp_groups=[[0, 4]], pp_groups=[[4, 12]], tp_groups=[[4, 5, 6, 7]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=5, local_rank=5, server_id=0, config=None, dp_groups=[[1, 5]], pp_groups=[[5, 13]], tp_groups=[[4, 5, 6, 7]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=6, local_rank=6, server_id=0, config=None, dp_groups=[[2, 6]], pp_groups=[[6, 14]], tp_groups=[[4, 5, 6, 7]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=7, local_rank=7, server_id=0, config=None, dp_groups=[[3, 7]], pp_groups=[[7, 15]], tp_groups=[[4, 5, 6, 7]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=8, local_rank=0, server_id=1, config=None, dp_groups=[[8, 12]], pp_groups=[[0, 8]], tp_groups=[[8, 9, 10,   11]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=9, local_rank=1, server_id=1, config=None, dp_groups=[[9, 13]], pp_groups=[[1, 9]], tp_groups=[[8, 9, 10, 11]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=10, local_rank=2, server_id=1, config=None, dp_groups=[[10, 14]], pp_groups=[[2, 10]], tp_groups=[[8, 9, 10, 11]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=11, local_rank=3, server_id=1, config=None, dp_groups=[[11, 15]], pp_groups=[[3, 11]], tp_groups=[[8, 9, 10, 11]], mp_groups=[[0, 1, 2, 3, 8, 9, 10, 11]])
RankZoo(world_rank=12, local_rank=4, server_id=1, config=None, dp_groups=[[8, 12]], pp_groups=[[4, 12]], tp_groups=[[12, 13, 14, 15]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=13, local_rank=5, server_id=1, config=None, dp_groups=[[9, 13]], pp_groups=[[5, 13]], tp_groups=[[12, 13, 14, 15]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=14, local_rank=6, server_id=1, config=None, dp_groups=[[10, 14]], pp_groups=[[6, 14]], tp_groups=[[12, 13, 14, 15]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
RankZoo(world_rank=15, local_rank=7, server_id=1, config=None, dp_groups=[[11, 15]], pp_groups=[[7, 15]], tp_groups=[[12, 13, 14, 15]], mp_groups=[[4, 5, 6, 7, 12, 13, 14, 15]])
"""