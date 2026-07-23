import unittest
from src.core.static_graphs.parallel_group_manager import ParallelGroupManager
from src.core.static_graphs.rank_manager import RankManager
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description='Static Graphs Parser')
    # parser.add_argument('--model-type', type=str, required=True,
    #                     choices=['gpt', 'llama', 'bert', 'CV', 'others'],
    #                     help='Type of the model to partition.')
    # parser.add_argument('--tensor-model-parallel-size', type=int, default=1,
    #                     help='Size of the tensor model parallel.')
    # parser.add_argument('--pipeline-model-parallel-size', type=int, default=1,
    #                     help='Size of the pipeline model parallel.')
    # parser.add_argument('--micro-batch-size', type=int, required=True,
    #                     help='Size of the micro batch.')
    # parser.add_argument('--global-batch-size', type=int, required=True,
    #                     help='Size of the global batch.')

    # # Arguments required for certain model types
    # parser.add_argument('--num-layers', type=int, default=None,
    #                     help='Number of layers in the model.')
    # parser.add_argument('--hidden-size', type=int, default=None,
    #                     help='Size of the hidden layers.')
    # parser.add_argument('--num-attention-heads', type=int, default=None,
    #                     help='Number of attention heads.')
    # parser.add_argument('--seq-length', type=int, default=None,
    #                     help='Sequence length.')
    # parser.add_argument('--max-position-embeddings', type=int, default=None,
    #                     help='Max position embeddings.')
    return parser.parse_args()


class TestRankInt(unittest.TestCase):
    def test_groups(self):
        args = _parse_args()
        local_size = 4
        world_size = 4
        pp_size = 2
        tp_size = 2
        args.nproc_per_node = local_size
        args.nnodes = 1

        manager = ParallelGroupManager(local_size, world_size, pp_size, tp_size)
        print(f"manager.get_dp_groups() -> : {manager.get_dp_groups()}")
        print(f"manager.get_pp_groups() -> : {manager.get_pp_groups()}")
        print(f"manager.get_tp_groups() -> : {manager.get_tp_groups()}")
        print(f"manager.get_mp_groups() -> : {manager.get_mp_groups()}")

        all_groups = manager.get_all_groups()
        rank_manager = RankManager(args, all_groups)
        rank_instances: list = rank_manager.get_rank_zoos()
        for rank_instance in rank_instances:
            print(rank_instance)


if __name__ == '__main__':
    unittest.main()