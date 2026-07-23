# import json
# import torch
# import torchvision
# from torch_graph import TorchGraph
# import torch.optim as optim
# import argparse


# parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--model', type=str, default='resnet50',
#                     help='model to benchmark')
# args = parser.parse_args()

# args.model="gpt2"
# args.type="NLP"

# from torchvision import models
# module = getattr(models, args.model)().cuda()
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# g = TorchGraph(module, example, optimizer, 'GPT2')
# for node in g.get_output_json():
#     print(node)
# g.dump_graph(args.model + "test.json")

import json
import torch
import torchvision
from TorchGraph.torch_graph import TorchGraph
import torch.optim as optim
import argparse

'''
没被切割(TP、PP)的模型的完整过程可以通过class torch_graph构建

Q:
    1. 异步情况是否包含?
    2. NCCL通信是否已经记录在grad中?
    3. overlap情况如果正常发生,单从新加入属性1无法确认cpu和kernel的过程,但是如果可以确认kernel发生在这个时间段,具体的时间点或许不重要?考虑到可能并发执行的只有
    不同model中,如果记录的operation除了id完全一致,是否其执行时间也一致呢？
    4. 对于一个新设计的model,怎么样的模拟DAG才能还原最可能的overlap发生情况?因为预测overlap的影响本身是建立在overlap发生的基础上的


需要添加的属性:
    1. start/end point (time)
    2. which layer (layer1 -> [..operations...] -> layer2), so operations belong to layer1
    3. which rank
'''

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--type', type=str, default='CV',
                    help='model types')
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument('--path', type=str, default='mytest-DDP.json',
                    help='path')
parser.add_argument('--path_var', type=str, default='DDP.json',
                    help='path')
args = parser.parse_args()

from torchvision import models
import transformer

args.model="resnet50"
args.type="CV"

model = args.model
# timer = Timer(100, args.model)
if args.type == 'CV':
    # module = getattr(models, args.model)().cuda()
    module = getattr(models, args.model)().cpu()
    # example = torch.rand(args.batchsize, 3, 224, 224).cuda()
    example = torch.rand(args.batchsize, 3, 224, 224).cpu()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
elif args.type == 'NLP':
    # module = getattr(transformer, args.model)().cuda()
    module = getattr(transformer, args.model)().cpu()
    # example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cuda()
    example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cpu()
    optimizer = optim.SGD(module.parameters(), lr=0.01)

import time
# 使用 TorchDatabase 类来收集关于模型前向传播、后向传播和优化器的性能数据。
time_1 = time.time()
g = TorchGraph(module, example, optimizer, 'resnet50')
time_cume = time.time() - time_1
print(f"time = {time.time() - time_1}")
# db = (g._get_overall_database())
g.dump_graph(args.path)

# json.dump(db,
#             open(args.path, 'w'),
#             indent=4)
# var = (g._get_overall_variance())
# json.dump(var,
#             open(args.path_var, 'w'),
#             indent=4)

