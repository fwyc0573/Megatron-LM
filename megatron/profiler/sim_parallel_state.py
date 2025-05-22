from typing import List, Optional



def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example, 
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then 
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the 
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the 
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).
        
        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks



class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        if 'ep' in order:
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)
        return ranks


def sim_initialize_model_parallel(
    world_size: int = None,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
) -> dict:
    """Initialize model parallel groups for simulation.

    Arguments:
        world_size: Total number of GPUs. If None, calculated as 
                   tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        virtual_pipeline_model_parallel_size: Degree of virtual pipeline model parallelism.
        pipeline_model_parallel_split_rank: Rank where encoder and decoder are split.
        use_sharp: Use SHARP for data parallel communications.
        context_parallel_size: Degree of context parallelism.
        expert_model_parallel_size: Degree of expert model parallelism.
        nccl_communicator_config_path: Path to NCCL communicator config.
        distributed_timeout_minutes: Timeout for distributed operations.
        order: Order of parallelism dimensions (e.g., "tp-cp-ep-dp-pp").

    Returns:
        Dictionary containing all parallel groups information.
    """
    # Initialize dictionary to store all group information
    groups_info = {
        'data_parallel_group': None,
        'data_parallel_global_ranks': None,
        'data_parallel_group_with_cp': None,
        'data_parallel_global_ranks_with_cp': None,
        'context_parallel_group': None,
        'context_parallel_global_ranks': None,
        'model_parallel_group': None,
        'tensor_model_parallel_group': None,
        'tensor_model_parallel_global_ranks': None,
        'pipeline_model_parallel_group': None,
        'pipeline_global_ranks': None,
        'embedding_group': None,
        'embedding_global_ranks': None,
        'position_embedding_group': None,
        'position_embedding_global_ranks': None,
        'tensor_and_data_parallel_group': None,
        'tensor_and_data_parallel_group_with_cp': None,
        'expert_model_parallel_group': None,
        'tensor_and_expert_parallel_group': None,
        'data_modulo_expert_parallel_group': None,
    }

    # Calculate world_size if not provided
    if world_size is None:
        world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size

    if world_size <= 0:
        raise RuntimeError(
            f"Invalid world_size: {world_size}. Check tensor_model_parallel_size={tensor_model_parallel_size}, "
            f"pipeline_model_parallel_size={pipeline_model_parallel_size}, "
            f"context_parallel_size={context_parallel_size}"
        )

    # Calculate data parallel size
    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
            f"({expert_model_parallel_size})"
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"Combination of expert model parallelism and context parallelism is not supported"
        )

    # Initialize rank generator
    rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
    )

    # Get data parallel group information
    dp_groups = rank_generator.get_ranks('dp')
    groups_info['data_parallel_global_ranks'] = dp_groups
    
    # Get data parallel + context parallel group information
    dp_cp_groups = rank_generator.get_ranks('dp-cp')
    groups_info['data_parallel_global_ranks_with_cp'] = dp_cp_groups

    # Get context parallel group information
    cp_groups = rank_generator.get_ranks('cp')
    groups_info['context_parallel_global_ranks'] = cp_groups

    # Get model parallel group information
    mp_groups = rank_generator.get_ranks('tp-pp')
    groups_info['model_parallel_group'] = mp_groups

    # Get tensor parallel group information
    tp_groups = rank_generator.get_ranks('tp')
    groups_info['tensor_model_parallel_global_ranks'] = tp_groups

    # Get pipeline parallel group information
    pp_groups = rank_generator.get_ranks('pp')
    groups_info['pipeline_global_ranks'] = pp_groups
    
    # Set up embedding group information
    embedding_groups = []
    position_embedding_groups = []
    for ranks in pp_groups:
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
            
        embedding_groups.append(embedding_ranks)
        position_embedding_groups.append(position_embedding_ranks)
    
    groups_info['embedding_global_ranks'] = embedding_groups
    groups_info['position_embedding_global_ranks'] = position_embedding_groups

    # Get tensor + data parallel group information
    tp_dp_cp_groups = rank_generator.get_ranks('tp-dp-cp')
    groups_info['tensor_and_data_parallel_group_with_cp'] = tp_dp_cp_groups
    
    tp_dp_groups = rank_generator.get_ranks('tp-dp')
    groups_info['tensor_and_data_parallel_group'] = tp_dp_groups

    # Get tensor + expert parallel group information
    tp_ep_groups = rank_generator.get_ranks('tp-ep', independent_ep=True)
    groups_info['tensor_and_expert_parallel_group'] = tp_ep_groups

    # Get expert parallel group information
    ep_groups = rank_generator.get_ranks('ep', independent_ep=True)
    groups_info['expert_model_parallel_group'] = ep_groups

    # Get data modulo expert parallel group information
    dp_modulo_ep_groups = rank_generator.get_ranks('dp', independent_ep=True)
    groups_info['data_modulo_expert_parallel_group'] = dp_modulo_ep_groups

    return groups_info