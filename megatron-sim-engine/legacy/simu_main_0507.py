from StaticGraphs.static_graphs_module import StaticGraphsModule
import argparse

from StaticGraphs.parallel_group_manager import ParallelGroupManager
from simu_engine import SimulatorEngine
from StaticGraphs.rank_manager import RankManager

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

    # Arguments required for certain model types
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



if __name__ == "__main__":
    # args = _parse_args()
    # static_graphs_module = StaticGraphsModule(args, json_save=True)
    # layer_assignments = static_graphs_module.allocate_model_layers()
    # print(layer_assignments)
    
    # TODO: 暂时用于衔接engine部分
    """ 根据 config setting 初始化 mpu 控制流的group信息 """
    local_size = 8
    world_size = local_size
    manager = ParallelGroupManager(local_size=local_size, world_size=world_size, pp_size=8, tp_size=1)
    mpu_info = manager._get_mpu_info()
    print(mpu_info)

    """ 结合 mpu info 完整初始化 rank """
    args = _parse_args()
    args.nproc_per_node = local_size
    args.nnodes = world_size // local_size
    all_groups = manager.get_all_groups()

    rank_manager = RankManager(args, all_groups)
    rank_instances: dict = rank_manager.get_rank_zoos()
    for rank_instance in rank_instances.values():
        print(rank_instance)
        # print(f"next: {rank_instance._get_pp_next_world_rank()}")
        # print(f"previous: {rank_instance._get_pp_previous_world_rank()}")
        # local_rank = rank_instance._get_pp_local_rank()
        # print(f"local rank: {local_rank}")
        # print(f"local rank cover to world rank: {rank_instance._convert_pp_local_to_world_rank(local_rank)}")


    # """ 测试SimulatorEngine类 | megatron-lm"""
    filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\8pp_1_1"
    simulator_engine: SimulatorEngine = SimulatorEngine(tmp_filename=filename, framwork='megatron-lm', strategy="1F1B-none_interleaved")
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances)
    simulator_engine._start_pipeline()
    simulator_engine._visualize_timelines()
