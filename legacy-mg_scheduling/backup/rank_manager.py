class RankManager:
    def __init__(self, args, groups_dict):
        self.args = args
        self.groups_dict = groups_dict
        self.rank_zoos = self._create_rank_zoos()

    def _create_rank_zoos(self):
        rank_zoos = {}
        world_size = self.args.nproc_per_node * self.args.nnodes
        for world_rank in range(world_size):
            local_rank = world_rank % self.args.nproc_per_node
            server_id = world_rank // self.args.nproc_per_node
            related_groups = self._find_related_groups(world_rank)
            rank_zoo = RankZoo(world_rank=world_rank, local_rank=local_rank,
                            server_id=server_id, model_config=None, **related_groups)
            rank_zoos[world_rank] = rank_zoo
        return rank_zoos

    def _find_related_groups(self, world_rank):
        related_groups = {'dp_groups': None, 'pp_groups': None, 'tp_groups': None, 'mp_groups': None}
        for group_type, groups in self.groups_dict.items():
            for group in groups:
                if world_rank in group:
                    related_groups[group_type] = group
        return related_groups

    def get_rank_zoos(self):
        return self.rank_zoos


class RankZoo:
    def __init__(self, world_rank, local_rank, server_id, model_config=None,
                 device_type='gpu', dp_groups=None, pp_groups=None, tp_groups=None, mp_groups=None):
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.server_id = server_id
        self.model_config = model_config
        self.dp_groups = dp_groups
        self.pp_groups = pp_groups
        self.tp_groups = tp_groups
        self.mp_groups = mp_groups
        self.device_type = device_type

    def __repr__(self):
        return f"RankZoo(world_rank={self.world_rank}, local_rank={self.local_rank}, " \
               f"server_id={self.server_id}, config={self.model_config}, " \
               f"dp_groups={self.dp_groups}, pp_groups={self.pp_groups}, " \
               f"tp_groups={self.tp_groups}, mp_groups={self.mp_groups})"

    def _get_dp_group_size(self):
        return len(self.dp_groups)

    def _get_pp_group_size(self):
        return len(self.pp_groups)

    def _get_tp_group_size(self):
        return len(self.tp_groups)

    def _get_mp_group_size(self):
        return len(self.mp_groups)

    def _get_pp_next_world_rank(self):
        index = self.pp_groups.index(self.world_rank)
        next_index = (index + 1) % len(self.pp_groups)
        return self.pp_groups[next_index]

    def _get_pp_previous_world_rank(self):
        index = self.pp_groups.index(self.world_rank)
        prev_index = (index - 1) % len(self.pp_groups)
        return self.pp_groups[prev_index]

    def _get_pp_local_rank(self):
        """ Get the stage id in the pp group"""
        return self.pp_groups.index(self.world_rank)

    def _is_pp_last_stage(self):
        return self._get_pp_local_rank() == self._get_pp_group_size() - 1
    
    def _is_pp_first_stage(self):
        return self._get_pp_local_rank() == 0
    
    def _convert_pp_world_to_local_rank(self, pp_world_rank):
        if pp_world_rank in self.pp_groups:
            return self.pp_groups.index(pp_world_rank)
        return None

    def _convert_pp_local_to_world_rank(self, pp_local_rank):
        if pp_local_rank < len(self.pp_groups):
            return self.pp_groups[pp_local_rank]
        return None

    def _get_tp_local_rank(self):
        return self.tp_groups.index(self.world_rank)

    def _convert_tp_world_to_local_rank(self, tp_world_rank):
        if tp_world_rank in self.tp_groups:
            return self.tp_groups.index(tp_world_rank)
        return None

    def _convert_tp_local_to_world_rank(self, tp_local_rank):
        if tp_local_rank < len(self.tp_groups):
            return self.tp_groups[tp_local_rank]
        return None

    def _get_tp_src_rank(self):
        pass

    def _get_dp_src_rank(self):
        pass



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