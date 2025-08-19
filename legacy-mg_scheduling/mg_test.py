"""
This module is used for generating scheduling plan for megatron-lm in stage-level.
We focus on the stage level ops like send/recv/forward/backward/optimizer_step/loss_func, etc, 
but not the micro-batch level ops (like tp_allreduce, ep_alltoall, etc).

ps: I think in moe mode, we donot need to add any other stage-level ops?


The scheduling plan is generated in the following steps:
1. Calculate the number of micro-batches in one global-batch.
2. Generate the scheduling plan for each stage.
3. Write the scheduling plan to a file.

"""

import argparse
from mg_scheduling_plan import SchedulingPlan


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-tp','--tensor-model-parallel-size', type=int, default=1, help='Tensor model parallelism size')
    parser.add_argument('-pp','--pipeline-model-parallel-size', type=int, default=1, help='Pipeline model parallelism size')
    parser.add_argument('-exp','--expert-model-parallel-size', type=int, default=1, help='Expert model parallelism size')
    parser.add_argument('-exp_num','--num-experts', type=int, default=1, help='Number of experts')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true', help='no embeding sharing, no ep allreduce')
    parser.add_argument("--local-size", type=int, default=None, help="Local rank of the process within the node")
    parser.add_argument("--world-size", type=int, default=None, help="Total number of processes in the execution")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Batch size per model instance")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Total effective batch size across all instances")
    parser.add_argument("--seq-length", type=int, default=None, help="")
    parser.add_argument("--hidden-size", type=int, default=None, help="")
    parser.add_argument("--model-size", type=str, default="tiny", help="Model size (e.g., tiny, 13, 30, 40, 70, 175)")
    parser.add_argument("--fp16", action='store_true', help="Enable fp16 precision (for pipeline_dtype)")
    parser.add_argument("--train-iters", type=int, default=None, help="Number of training iterations")
    parser.add_argument("--trace-start", type=int, default=None, help="When to trace")
    return parser.parse_args()


def validate_and_calculate_parameters(args):
    # 验证world-size/tp/pp为整数
    assert args.world_size % (args.tensor_model_parallel_size*args.pipeline_model_parallel_size) == 0, \
                                                            "world-size 必须能被tp和pp整除"
    # 计算数据并行度dp
    dp = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size)
    # 验证global-batch-size/micro-batch-size/dp为整数
    assert args.global_batch_size % (args.micro_batch_size * dp) == 0, "global-batch-size 必须能被 micro-batch-size 和 dp 的乘积整除"
    # 计算num_micro_batches
    num_microbatches = args.global_batch_size // (args.micro_batch_size * dp)
    args.data_parallel_size = dp
    assert args.data_parallel_size % args.expert_model_parallel_size == 0, "DP必须能被EP整除"

    if args.expert_model_parallel_size > 1:
        args.exp_dp_size = dp // args.expert_model_parallel_size
    else:
        args.exp_dp_size = 1
    args.num_microbatches = num_microbatches
    # print(args)
    return args


def main():
    args = parse_arguments()
    args = validate_and_calculate_parameters(args)

    sp = SchedulingPlan(args)
    sp.print_scheduling_meta_info()
    sp.get_write_scheduling_plan(write_to_file=True)


if __name__ == "__main__":
    main()