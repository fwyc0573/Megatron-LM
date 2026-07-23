from StaticGraphs.static_graphs_module import StaticGraphsModule
import argparse

from StaticGraphs.parallel_group_manager import MPUInfo, ParallelGroupManager
from simu_engine_ds import SimulatorEngine
from StaticGraphs.rank_manager import RankManager

def _parse_args():
    parser = argparse.ArgumentParser(description='Static Graphs Parser')

    # args for predictor
    parser.add_argument('--skip-coverage',
                        dest='skip_coverage', action='store_true',
                        help='skip testing the databse coverage')
    parser.add_argument('--skip-accuracy',
                        dest='skip_accuracy', action='store_true',
                        help='skip testting the simulator accuracy')
    parser.add_argument('-c', '--config_path',
                        dest='config', default=r'H:\HUBOther\ML_Sys_Merak\ai_simulator\simulator_benchmark\config\config_torch.yaml',
                        help='config setting.')
    
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
    """ vgg19 144M, PP=4, DP=2 """
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\graph_and_database'
    # current_pp_size = 4
    # current_tp_size = 1
    # current_world_size = 8

    """ vgg19 144M, PP=1, DP=8 暂时无法执行,需要manually统计"""
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp1\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp1\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp1\graph_and_database'
    # current_pp_size = 1
    # current_tp_size = 1
    # current_world_size = 8

    """ vgg19 144M, PP=2, DP=4 """
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp2\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp2\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp2\graph_and_database'
    # current_pp_size = 2
    # current_tp_size = 1
    # current_world_size = 8


    """ vgg19 144M, PP=4, DP=2 """
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp4\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp4\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp4\graph_and_database'
    # current_pp_size = 4
    # current_tp_size = 1
    # current_world_size = 8

    """ compare vgg19 144M, PP=4, DP=2 """
    ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp4_h800\ranks'
    stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp4_h800\stages'
    torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp4_h800\graph_and_database'
    current_pp_size = 4
    current_tp_size = 1
    current_world_size = 8

    """ vgg19 144M, PP=2, DP=4 """
    ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp2_h800\ranks'
    stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp2_h800\stages'
    torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\comp_vgg19_8gpu_pp2_h800\graph_and_database'
    current_pp_size = 2
    current_tp_size = 1
    current_world_size = 8

    """ llama 25B, PP=8 """
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0604_pp8_26B_batch4_4_layers\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0604_pp8_26B_batch4_4_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0604_pp8_26B_batch4_4_layers\graph_and_database'
    # current_pp_size = 8
    # current_tp_size = 1
    # current_world_size = 8

    """ llama 25B, PP=16, DP=32, world_size=512 """
    # ds_trace_filepath = None
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0605_pp16dp32_26B_batch4_4_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0605_pp16dp32_26B_batch4_4_layers\graph_and_database'


    """ 根据 config setting 初始化 mpu 控制流的group信息 """
    local_size = 8
    world_size = current_world_size
    running_mode = "simulating" # simulating logic mapping
    manager = ParallelGroupManager(local_size=local_size, world_size=world_size, pp_size=current_pp_size, tp_size=current_tp_size)
    mpu_info: MPUInfo = manager.get_mpu_info()
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
        print(rank_instance._get_pp_local_rank())

    """ 测试v2 deepspeed 数据读取 """
    filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log"
    # filename = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0511'
    # stages_list, stages_dict, _ = SimulatorEngine.v2_ds_handle_tmp_stages_dataset(filename, rank_instances, True)
    # for stage in stages_list:
    #     total_duration = stage._total_duration()  # Calculate the total duration for this stage
    #     print(f"stage:{stage.stage_id}, operation_exec_sum = {total_duration:.2f}")


    """ 测试SimulatorEngine类 | megatron-lm"""
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\8pp_1_1"
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\3pp_2dp_1"

    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_nooverlap"
    # simulator_engine: SimulatorEngine = SimulatorEngine(tmp_filename=filename, framwork='megatron-lm', strategy="1F1B-none_interleaved")
    # simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)
    # simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    # simulator_engine._start_pipeline()
    # simulator_engine._visualize_timelines()


    """ 测试SimulatorEngine类 | deepspeed"""
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log"
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0520_8pp_params\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0520_8pp_params\graph_and_database'

    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0520_8pp_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0520_8pp_layers\graph_and_database'

    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0521_pp2dp3_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0521_pp2dp3_layers\graph_and_database'

    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp8_layers\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp8_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp8_layers\graph_and_database'
    
    # ds_trace_filepath = None
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0602_pp32_dp4_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\graph_and_database'

    simulator_engine: SimulatorEngine = SimulatorEngine(ds_trace_filepath=ds_trace_filepath, framwork='deepspeed', 
                                                        strategy="1F1B-none_interleaved", args=args, running_mode=running_mode, torchgraph_filepath=torchgraph_filepath,
                                                        stages_scheduling_filepath=stages_scheduling_filepath)
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    simulator_engine._start_pipeline()


    # simulator_engine.ds_op_compare_simu_with_trace()
    simulator_engine.visualize_timelines(wrank_id_start_end=[0,100])