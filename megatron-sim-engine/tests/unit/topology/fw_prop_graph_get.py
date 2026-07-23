import torch
import torch.nn as nn
import time

class ForwardHookRecorder:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.recorded_operations = []

        # 注册前向钩子到所有模块
        self._register_hooks()

    def _register_hooks(self):
        # 遍历所有模块并附加钩子
        for name, module in self.model.named_modules():
            # 跳过总模型本身，只对子模块操作
            if module != self.model:
                hook = module.register_forward_hook(self._forward_hook(name))
                self.hooks.append(hook)

    def _forward_hook(self, module_name):
        # 创建并返回钩子函数
        def hook(module, inputs, outputs):
            # 移除所有钩子
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()


            print(inputs[0])
            torch.cuda.synchronize()
            ss = time.perf_counter()
            module(inputs[0])
            torch.cuda.synchronize()
            ee = time.perf_counter()
            

            # 记录操作信息
            self.recorded_operations.append({
                'module_name': module_name,
                'module_type': type(module).__name__,
                'inputs': [input.shape for input in inputs] if isinstance(inputs, tuple) else [inputs.shape],
                # 'outputs': outputs.shape
                'outputs': [output.shape for output in outputs if torch.is_tensor(output)] if isinstance(outputs, tuple) else outputs.shape if torch.is_tensor(outputs) else 'Not a tensor',
                'execution_time': ee - ss
            })
            self._register_hooks()
        return hook

    def remove_hooks(self):
        # 移除所有钩子，防止内存泄漏
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def print_recorded_operations(self):
        # 打印所有记录的操作
        for op in self.recorded_operations:
            print(op)

# class Conv1D(nn.Module):
#     def __init__(self, nf, nx):
#         super().__init__()
#         self.nf = nf
#         self.weight = nn.Parameter(torch.empty(nx, nf))
#         self.bias = nn.Parameter(torch.zeros(nf))
#         nn.init.normal_(self.weight, std=0.02)

#     def forward(self, x):
#         size_out = x.size()[:-1] + (self.nf,)
#         x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
#         x = x.view(size_out)
#         return x

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.my_conv_1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
#         self.my_relu_2 = nn.ReLU()
#         self.my_conv1d_3 = Conv1D(10*32*32, 10)  # Change the input features of Conv1D layer to 10*32*32
#         self.my_linear_4 = nn.Linear(10, 5)

#     def forward(self, x):
#         x = self.my_conv_1(x)
#         x = self.my_relu_2(x)
#         x = torch.add(x, 1)
#         x = x.view(x.size(0), -1)
#         x = self.my_conv1d_3(x)  # Apply Conv1D layer
#         return self.my_linear_4(x)
    
# 创建模型和记录器
# model = SimpleModel()

from torchvision import models
import transformer

model = getattr(models, "resnet50")()
model = getattr(transformer, "gpt2")()

recorder = ForwardHookRecorder(model)

# 输入示例
# input_tensor = torch.randn(1, 3, 32, 32)
# example = torch.rand(1, 3, 224, 224)
example = (torch.LongTensor(1,512).random_() % 1000)

# 执行前向传播
output = model(example)

# 打印记录的操作
recorder.print_recorded_operations()

# 清理钩子
recorder.remove_hooks()



"""
{'module_name': 'h.10.mlp.dropout', 'module_type': 'Dropout', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.00020710565149784088}
{'module_name': 'h.10.mlp', 'module_type': 'GPT2MLP', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.00468544103205204}
{'module_name': 'h.10', 'module_type': 'GPT2Block', 'inputs': [torch.Size([1, 512, 768])], 'outputs': [torch.Size([1, 512, 768])], 'execution_time': 0.010782206431031227}
{'module_name': 'h.11.ln_1', 'module_type': 'LayerNorm', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 9.9916011095047e-05}
{'module_name': 'h.11.attn.c_attn', 'module_type': 'Conv1D', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 2304]), 'execution_time': 0.0010919757187366486}
{'module_name': 'h.11.attn.attn_dropout', 'module_type': 'Dropout', 'inputs': [torch.Size([1, 12, 512, 512])], 'outputs': torch.Size([1, 12, 512, 512]), 'execution_time': 0.0013134554028511047}
{'module_name': 'h.11.attn.c_proj', 'module_type': 'Conv1D', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.0005749203264713287}
{'module_name': 'h.11.attn.resid_dropout', 'module_type': 'Dropout', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.00024471431970596313}
{'module_name': 'h.11.attn', 'module_type': 'GPT2Attention', 'inputs': [torch.Size([1, 512, 768])], 'outputs': [torch.Size([1, 512, 768])], 'execution_time': 0.00838763639330864}
{'module_name': 'h.11.ln_2', 'module_type': 'LayerNorm', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 9.97055321931839e-05}
{'module_name': 'h.11.mlp.c_fc', 'module_type': 'Conv1D', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 3072]), 'execution_time': 0.0014229826629161835}
{'module_name': 'h.11.mlp.act', 'module_type': 'NewGELUActivation', 'inputs': [torch.Size([1, 512, 3072])], 'outputs': torch.Size([1, 512, 3072]), 'execution_time': 0.0016909856349229813}
{'module_name': 'h.11.mlp.c_proj', 'module_type': 'Conv1D', 'inputs': [torch.Size([1, 512, 3072])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.0015755612403154373}
{'module_name': 'h.11.mlp.dropout', 'module_type': 'Dropout', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.00020884349942207336}
{'module_name': 'h.11.mlp', 'module_type': 'GPT2MLP', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.004416767507791519}
{'module_name': 'h.11', 'module_type': 'GPT2Block', 'inputs': [torch.Size([1, 512, 768])], 'outputs': [torch.Size([1, 512, 768])], 'execution_time': 0.010745113715529442}
{'module_name': 'ln_f', 'module_type': 'LayerNorm', 'inputs': [torch.Size([1, 512, 768])], 'outputs': torch.Size([1, 512, 768]), 'execution_time': 0.00509590283036232}
"""