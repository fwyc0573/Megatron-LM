from StaticGraphs.static_graphs_module import StaticGraphsModule
import argparse
import time
from StaticGraphs.parallel_group_manager import MPUInfo, ParallelGroupManager
from simu_engine import SimulatorEngine
from StaticGraphs.rank_manager import RankManager
from simu_engine import MODE_MODEL, MODE_PROFILE, MODE_SIMULATE, RUNNING_MODE_OPTION


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
    """ ds: vgg19 144M, PP=4, DP=2 """
    # framwork='deepspeed'
    # ds_trace_filepath = None #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0601_pp4dp2_layers\graph_and_database'
    # curr_world_size = 8
    # curr_pp_size = 4
    # curr_tp_size = 1

    """ ds: llama 25B, PP=8 """
    # TODO: DS的profiler格式需要调整
    # framwork='deepspeed'
    # ds_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0604_pp8_26B_batch4_4_layers\ranks'
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0604_pp8_26B_batch4_4_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0604_pp8_26B_batch4_4_layers\graph_and_database'
    # curr_world_size = 8
    # curr_pp_size = 8
    # curr_tp_size = 1


    """ ds: llama 25B, PP=16, DP=32, world_size=512 """
    # ds_trace_filepath = None
    # stages_scheduling_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0605_pp16dp32_26B_batch4_4_layers\stages'
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log\0605_pp16dp32_26B_batch4_4_layers\graph_and_database'

    """ mg: llama tiny, PP=3 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\new_pp3\profile'
    # # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_1_1"# r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\new_pp3\schedule"# r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp'
    # # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_"
    # torchgraph_filepath = None

    """ mg: llama 7B, PP=2 DP=2 TP=2"""
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2_2_7llama\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2_2_7llama\schedule"
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2_2_7llama\database_profile'

    """ mg: llama 13B, PP=2 DP=2 TP=2 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_13llama\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_13llama\schedule"
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_13llama\database_profile'
    # curr_world_size = 8
    # curr_pp_size = 2
    # curr_tp_size = 2

    """ mg: llama 13B, PP=2 DP=2 TP=2 time_perf"""
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_13llama_timeperf\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_13llama_timeperf\schedule"
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2dp_13llama_timeperf\database_profile'

    """ mg: llama tiny, PP=2 DP=2 TP=2 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2_2_tiny_llama\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2_2_tiny_llama\schedule"
    # torchgraph_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\2pp_2_2_tiny_llama\database_profile'


    """ mg: llama 7B, PP=4 DP=2 TP=1 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_2dp_7llama\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_2dp_7llama\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_2dp_7llama\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 4
    # curr_tp_size = 1

    """ mg: llama tiny, PP=4 DP=2 TP=1 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_2dp_tiny_llama\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_2dp_tiny_llama\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\4pp_2dp_tiny_llama\database_profile"

    """ 16gpu """
    # framwork='megatron-lm'
    # mg_trace_filepath = None #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\16_4pp_2tp_2dp_13BLLAMA\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\16_4pp_2tp_2dp_13BLLAMA\database_profile"
    # curr_world_size = 16
    # curr_pp_size = 4
    # curr_tp_size = 2


    """ 32gpu """
    # framwork='megatron-lm'
    # mg_trace_filepath = None #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\32_4pp_4tp_2dp_13BLLAMA\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\32_4pp_4tp_2dp_13BLLAMA\database_profile"
    # curr_world_size = 32
    # curr_pp_size = 4
    # curr_tp_size = 4

    """ 64gpu """
    # framwork='megatron-lm'
    # mg_trace_filepath = None #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64_4pp_8tp_2dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64_4pp_8tp_2dp_70BLLAMA\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64_4pp_8tp_2dp_70BLLAMA\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 4
    # curr_tp_size = 8


    """ 128gpu """
    # framwork='megatron-lm'
    # mg_trace_filepath = None #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64_4pp_8tp_2dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\128_4pp_8tp_4dp_70BLLAMA\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\128_4pp_8tp_4dp_70BLLAMA\database_profile"
    # curr_world_size = 128
    # curr_pp_size = 4
    # curr_tp_size = 8


    """ 256gpu """
    # framwork='megatron-lm'
    # mg_trace_filepath = None #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\database_profile"
    # curr_world_size = 256
    # curr_pp_size = 4
    # curr_tp_size = 8

    """ a800 comp_gpt13b_8gpu_pp4_tp2 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 4
    # curr_tp_size = 2

    """ a800 comp_gpt13b_8gpu_pp8_tp1 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp8_tp1\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp8_tp1\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp8_tp1\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 8
    # curr_tp_size = 1


    """ profiling mode test """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_tmp_profile_mode\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_tmp_profile_mode\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_tmp_profile_mode\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 4
    # curr_tp_size = 2


    """ h800 fp32 comp_gpt13b_8gpu_pp4_tp2 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_h800_fp32\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_h800_fp32\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_h800_fp32\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 4
    # curr_tp_size = 2


    # """  64gpus_4pp_8tp_13b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_13b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_13b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_13b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 4
    # curr_tp_size = 8

    # """  64gpus_8pp_4tp_13b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_13b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_13b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_13b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 8
    # curr_tp_size = 4

    # """  64gpus_8pp_4tp_30b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_30b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_30b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_30b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 8
    # curr_tp_size = 4

    # """  64gpus_4pp_8tp_30b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_30b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_30b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_30b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 4
    # curr_tp_size = 8


    # """  64gpus_4pp_8tp_40b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_40b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_40b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_4pp_8tp_40b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 4
    # curr_tp_size = 8


    # """  64gpus_8pp_4tp_40b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_40b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_40b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\64gpus_8pp_4tp_40b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 8
    # curr_tp_size = 4


    # """  realistic_trace_175b_12_8_32 """
    framwork='megatron-lm'
    mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\realistic_trace_175b_12_8_32\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\realistic_trace_175b_12_8_32\schedule"
    torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\realistic_trace_175b_12_8_32\database_profile"
    curr_world_size = 96
    curr_pp_size = 12
    curr_tp_size = 8

    # """  realistic_trace_175b_12_8_16 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\realistic_trace_175b_12_8_16\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\realistic_trace_175b_12_8_16\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\realistic_trace_175b_12_8_16\database_profile"
    # curr_world_size = 96
    # curr_pp_size = 12
    # curr_tp_size = 8


    # """  (cuda)_ realistic_trace_175b_12_8_16 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\cuda_realistic_trace_175b_12_8_16\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\cuda_realistic_trace_175b_12_8_16\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\cuda_realistic_trace_175b_12_8_16\database_profile"
    # curr_world_size = 96
    # curr_pp_size = 12
    # curr_tp_size = 8


    # """  96gpus_12pp_4tp_70b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\96gpus_12pp_4tp_70b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\96gpus_12pp_4tp_70b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\96gpus_12pp_4tp_70b\database_profile"
    # curr_world_size = 96
    # curr_pp_size = 12
    # curr_tp_size = 4


    # """  comp_gpt13b_8gpu_pp4_tp2_h800_fp32 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_h800_fp32\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_h800_fp32\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp4_tp2_h800_fp32\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 4
    # curr_tp_size = 2

    # """ comp_gpt13b_8gpu_pp8_tp1_h800_fp32 """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp8_tp1_h800_fp32\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp8_tp1_h800_fp32\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\comp_gpt13b_8gpu_pp8_tp1_h800_fp32\database_profile"
    # curr_world_size = 8
    # curr_pp_size = 8
    # curr_tp_size = 1



    # """  h800_64gpus_8pp_4tp_13b """
    framwork='megatron-lm'
    mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_13b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_13b\schedule"
    torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_13b\database_profile"
    curr_world_size = 64
    curr_pp_size = 8
    curr_tp_size = 4

    # """  h800_64gpus_8pp_4tp_30b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_30b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_30b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_30b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 8
    # curr_tp_size = 4

    """  h800_64gpus_8pp_4tp_40b """
    # framwork='megatron-lm'
    # mg_trace_filepath = r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_40b\global_ranks_profile' #r'H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\256_4pp_8tp_8dp_70BLLAMA\global_ranks_profile'
    # stages_scheduling_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_40b\schedule"
    # torchgraph_filepath = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\megatron_operation_log\h800_64gpus_8pp_4tp_40b\database_profile"
    # curr_world_size = 64
    # curr_pp_size = 8
    # curr_tp_size = 4



    # # tmp test for comb gpus
    # curr_world_size = 64
    # curr_pp_size = 8
    # curr_tp_size = 4


    """ 根据 config setting 初始化 mpu 控制流的group信息 """
    local_size = 8
    world_size = curr_world_size # 48*8*8
    running_mode = MODE_SIMULATE # MODE_PROFILE MODE_SIMULATE MODE_MODEL
    manager = ParallelGroupManager(local_size=local_size, world_size=world_size, pp_size=curr_pp_size, tp_size=curr_tp_size)
    mpu_info: MPUInfo = manager.get_mpu_info()
    print(mpu_info)
    # raise 0

    """ 结合 mpu info 完整初始化 rank """
    args = _parse_args()
    args.nproc_per_node = local_size
    args.nnodes = world_size // local_size
    all_groups = manager.get_all_groups()

    rank_manager = RankManager(args, all_groups)
    rank_instances: dict = rank_manager.get_rank_zoos()
    for rank_instance in rank_instances.values():
        print(rank_instance)
        # print(rank_instance._get_pp_local_rank())
        print(rank_instance.is_rank_in_embedding_group())
    # raise 0

    """ 测试v2 deepspeed 数据读取 """
    # filename = r"H:\HUBOther\ML_Sys_Merak\TorchGraph\deepspeed_operation_log"
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



    simulator_engine: SimulatorEngine = SimulatorEngine(trace_filepath=mg_trace_filepath, framwork=framwork, 
                                                        strategy="1F1B-none_interleaved", args=args, running_mode=running_mode, torchgraph_filepath=torchgraph_filepath,
                                                        stages_scheduling_filepath=stages_scheduling_filepath)
    simulator_engine._set_mpu_info_and_init_key_relationship(mpu_info)

    time_load_start = time.time()
    simulator_engine._init_tmp_stages_dataset_and_timeline_manager(rank_instances, mpu_info)
    load_time = time.time() - time_load_start
    print(f"sim load time:{load_time}")

    # simulator_engine.check_error_inference()

    time_execution_start = time.time()
    simulator_engine.start_running()
    execution_time = time.time() - time_execution_start
    print(f"sim execution time:{execution_time}")

    # simulator_engine.get_global_operation_error()
    simulator_engine.get_op_json_db_record(file_name="moye_sim_"+"h800_64gpus_8pp_4tp_13b") # h800_96gpus_12pp_8tp_175b
    # simulator_engine.ds_op_compare_simu_with_trace() 
    # specific_ranks_list = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88] [0, 1, 2, 3 ,4 ,5 ,6 ,7] [0, 8, 16, 24, 32, 40, 48, 56]
    # simulator_engine.visualize_timelines(wrank_id_start_end=[0,100], specific_ranks_list=[0, 8, 16, 24, 32, 40, 48, 56], show_x_lim=None)