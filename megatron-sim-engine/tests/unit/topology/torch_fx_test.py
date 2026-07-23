import math
import torch
from torch.fx import symbolic_trace, GraphModule

def example_1():
    def my_func(x):
        return torch.relu(x).neg()
    traced: GraphModule = symbolic_trace(my_func)
    print(traced.code)

    class SampleModule(torch.nn.Module):
        def forward(self, x):
            return self.act(x+math.pi)

    sm = SampleModule()
    sm.act = traced

    input_tensor = torch.randn(10)
    output = sm(input_tensor)
    print(output)

    traced: GraphModule = symbolic_trace(sm)
    # print(traced.code)

    output = traced(input_tensor)
    print(output)
    loss = output.sum()
    loss.backward()
    """
    def forward(self, x):
        relu = torch.relu(x);  x = None
        neg = relu.neg();  relu = None
        return neg
    -----------------------------------------
    def forward(self, x):
        add = x + 3.141592653589793;  x = None
        relu = torch.relu(add);  add = None
        neg = relu.neg();  relu = None
        return neg
    """

def exmaple_change_save():
    def my_func(x):
        return torch.relu(x).neg()
    traced: GraphModule = symbolic_trace(my_func)
    print(traced.code)

    class SampleModule(torch.nn.Module):
        def forward(self, x):
            return self.act(x+math.pi)

    sm = SampleModule()
    sm.act = traced
    traced: GraphModule = symbolic_trace(sm)
    traced.to_folder(folder="/data/ytyang/yichengfeng/StaticGraphs/test/save_file", module_name="after_change_model")
    print("finish save...")



def exmaple_2():
    import torch.nn as nn
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(10*32*32, 5)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)
            return self.linear(x)
        
    sm = SimpleModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    traced: GraphModule = symbolic_trace(sm)
    output = traced(input_tensor)
    print(output)

    loss = output.sum()
    loss.backward()


def exmaple_3():
    import torch.nn as nn
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.my_conv_1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
            self.my_relu_2 = nn.ReLU()
            self.my_linear_3 = nn.Linear(10*32*32, 5)

        def forward(self, x):
            x = self.my_conv_1(x)
            x = self.my_relu_2(x)
            x = torch.add(x, 1)
            x = x.view(x.size(0), -1)
            return self.my_linear_3(x)
        
    sm = SimpleModel()
    traced: GraphModule = symbolic_trace(sm)
    traced.print_readable()
    """
        class SimpleModel(torch.nn.Module):
        def forward(self, x):
            # No stacktrace found for following nodes
            conv = self.conv(x);  x = None
            relu = self.relu(conv);  conv = None
            size = relu.size(0)
            view = relu.view(size, -1);  relu = size = None
            linear = self.linear(view);  view = None
            return linear
    """
    # print(traced.code)
    """
        def forward(self, x):
            conv = self.conv(x);  x = None
            relu = self.relu(conv);  conv = None
            size = relu.size(0)
            view = relu.view(size, -1);  relu = size = None
            linear = self.linear(view);  view = None
            return linear
    """
    for n in traced.graph.nodes:
        print(f"{n.name} = {n.op} target={n.target} args={n.args}")

def exmaple_use_warp_module():
    import torch.nn as nn

    # @torch.fx.wrap
    class Conv1D(nn.Module):
        """
        1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

        Basically works like a linear layer but the weights are transposed.

        Args:
            nf (`int`): The number of output features.
            nx (`int`): The number of input features.
        """

        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = nn.Parameter(torch.empty(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))
            nn.init.normal_(self.weight, std=0.02)

        # @torch.fx.wrap
        def forward(self, x):
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(size_out)
            return x
    # torch.fx.wrap(Conv1D.forward)
    # torch.fx.wrap(Conv1D)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.my_conv_1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
            self.my_relu_2 = nn.ReLU()
            # torch.fx.wrap(Conv1D)
            self.my_conv1d_3 = Conv1D(10, 10*32*32)  # Add Conv1D layer
            self.my_linear_4 = nn.Linear(10*32*32, 5)

        def forward(self, x):
            x = self.my_conv_1(x)
            x = self.my_relu_2(x)
            x = torch.add(x, 1)
            x = x.view(x.size(0), -1)
            x = self.my_conv1d_3(x)  # Apply Conv1D layer
            return self.my_linear_4(x)
        
    sm = SimpleModel()
    traced: GraphModule = symbolic_trace(sm)
    traced.print_readable()
    for n in traced.graph.nodes:
        print(f"{n.name} = {n.op} target={n.target} args={n.args}")



def example_save_class_test():
    class after_change_model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # self.load_state_dict(torch.load(r'/data/ytyang/yichengfeng/StaticGraphs/test/save_file/state_dict.pt'))

        def forward(self, x):
            add = x + 3.141592653589793;  x = None
            relu = torch.relu(add);  add = None
            neg = relu.neg();  relu = None
            return neg
        
    sm = after_change_model()
    # input_tensor = torch.randn(1, 3, 32, 32)
    traced: GraphModule = symbolic_trace(sm)
    traced.print_readable()
    # output = sm(input_tensor)
    # print(output)


def example_split_test():
    import torch
    # from torch.fx import symbolic_trace
    # from torch.fx.graph_module import GraphModule
    from torch.fx.node import Node
    from torch.fx.passes.split_module import split_module
    import torch.nn as nn
    # class MyModule(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.param = torch.nn.Parameter(torch.rand(3, 4))
    #         self.linear = torch.nn.Linear(4, 5)

    #     def forward(self, x, y):
    #         z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
    #         w = self.linear(y).clamp(min=0.0, max=1.0)
    #         return z + w
    # @torch.fx.wrap
    # def func_wrap_test():
    #     return nn.Linear(10*32*32, 5)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.my_conv_1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
            self.my_relu_2 = nn.ReLU()
            # torch.fx.wrap(Conv1D)
            # self.my_conv1d_3 = Conv1D(10, 10*32*32)  # Add Conv1D layer
            self.my_linear_4 = nn.Linear(10*32*32, 5) #  func_wrap_test()
            # torch.fx.wrap(self.my_linear_4.forward)

        def forward(self, x):
            x = self.my_conv_1(x)
            x = self.my_relu_2(x)
            # x = torch.add(x, 1)
            # x = x.view(x.size(0), -1)
            # x = self.my_conv1d_3(x)  # Apply Conv1D layer
            x = torch.flatten(x, start_dim=1)
            return self.my_linear_4(x)
        
    # symbolically trace model
    my_module = SimpleModel()
    my_module_traced = symbolic_trace(my_module)
    print(my_module_traced.graph)
    # scripted_model = torch.jit.script(my_module)  # 使用 Scripting 转换模型
    # print(scripted_model.graph)  # 打印 TorchScript 计算图
    print("------------------------------------------------------")
    # # random mod partitioning
    # partition_counter = [0]
    # NPARTITIONS = 4

    # def mod_partition(node: Node):
    #     # global partition_counter
    #     partition = partition_counter[0] % NPARTITIONS
    #     partition_counter[0] = (partition_counter[0] + 1) % NPARTITIONS
    #     return partition

    # # split module in module with submodules
    # module_with_submodules = split_module(
    #     my_module_traced, my_module, mod_partition
    # )
    # print(module_with_submodules)
    # print("------------------------------------------------------")
    # """
    #     GraphModule(
    #     (submod_0): GraphModule(
    #         (linear): Linear(in_features=4, out_features=5, bias=True)
    #     )
    #     (submod_1): GraphModule(
    #         (linear): Linear(in_features=4, out_features=5, bias=True)
    #     )
    #     (submod_2): GraphModule()
    #     )

    #     def forward(self, x, y):
    #         param = self.param
    #         submod_0 = self.submod_0(x, param, y);  x = param = y = None
    #         getitem = submod_0[0]
    #         getitem_1 = submod_0[1];  submod_0 = None
    #         submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None
    #         getitem_2 = submod_1[0]
    #         getitem_3 = submod_1[1];  submod_1 = None
    #         submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    #         return submod_2
            
    #     # To see more debug info, please use `graph_module.print_readable()`
    # """
    # submod_0 = module_with_submodules.submod_0
    # print(submod_0)
    # print("------------------------------------------------------")
    # # x = torch.rand(3, 4)
    # # y = torch.rand(3, 4)
    # # param = torch.nn.Parameter(torch.rand(3, 4))
    # # output = submod_0(x,param,y)
    # # print(output)

    # x = torch.randn(1, 3, 32, 32)
    # output = submod_0(x)
    # print(output)

def example_op_aten_test():
    import torch
    import time

    # 对于aten::add
    start_time = time.time()
    result = torch.ops.aten.add(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    elapsed_time = time.time() - start_time
    print(f"aten::add took {elapsed_time} seconds.")

    # 对于aten::neg
    start_time = time.time()
    result = torch.ops.aten.neg(torch.tensor([1.0, 2.0]))
    elapsed_time = time.time() - start_time
    print(f"aten::neg took {elapsed_time} seconds.")


def exmaple_aotgrad_test():
    import time
    # import torch
    import torch.utils._pytree as pytree
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.weak import WeakIdKeyDictionary
    from graphviz import Digraph

    __all__ = ["capture"]

    class CaptureGraph(TorchDispatchMode):
        def __init__(self, fname="graph.dot"):
            self.fname = fname
            self._graph = Digraph(format="svg")
            self._tensors = WeakIdKeyDictionary()
            self._n_tensors = 0
            self._n_ops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # print(f"Function: {func}")
            # print(f"Types: {types}")
            # print(f"Arguments: {args}")
            # shape = args[0].shape
            # print(f"Shape: {shape}")
            # tensor_type = args[0].dtype
            # print(f"Tensor Type: {tensor_type}")
            # print(f"Keyword Arguments: {kwargs}")

            out = func(*args, **kwargs)
            # print(f"Out: {out}")

            op = f"{func}_{self._n_ops}"
            print(f"op = {op}")
            self._n_ops += 1
            self._graph.node(op, str(func), fillcolor="green")
            self._add_to_graph((args, kwargs), op, is_in=True)
            self._add_to_graph(out, op, is_in=False)
            return out

        def _add_to_graph(self, args, op, is_in=True):
            flat_args, _ = pytree.tree_flatten(args)
            # print(f"flat_args: {flat_args}")
            for t in flat_args:
                if not torch.is_tensor(t):
                    continue
                if t not in self._tensors:
                    tensor = f"tensor_{self._n_tensors}"
                    self._graph.node(tensor, fillcolor="skyblue")
                    self._tensors[t] = tensor
                    self._n_tensors += 1
                else:
                    tensor = self._tensors[t]
                if is_in:
                    self._graph.edge(tensor, op)
                else:
                    self._graph.edge(op, tensor)

        # def __exit__(self, exc_type, exc_value, traceback):
        #     super().__exit__(exc_type, exc_value, traceback)
        #     self._graph.render(self.fname)

    def capture(model, *inputs):
        primals = [p for p in model.parameters() if p.requires_grad]
        primals.extend([p for p in inputs if torch.is_tensor(p) and p.requires_grad])
        with CaptureGraph(f"dispatch.{time.time()}.dot"):
            output = model(*inputs)
            # loss = model(*inputs).sum()
            loss = output.sum()
            grads = torch.autograd.grad(loss, primals)
            print(loss,grads)

    import torch.nn as nn
    # model = nn.Sequential(
    #     nn.Conv2d(16, 32, 3),
    #     nn.BatchNorm2d(32),
    #     nn.SiLU(),
    # ).cuda()
    # x = torch.randn((2, 16, 8, 8), requires_grad=True, device="cuda")

    from torchvision import models
    import transformer
    # model = getattr(models, "resnet50")()
    model = getattr(transformer, "gpt2")().cuda()


    # 输入示例
    # input_tensor = torch.randn(1, 3, 32, 32)
    # example = torch.rand(1, 3, 224, 224)
    x = (torch.LongTensor(1,512).random_() % 1000).cuda()

    output = model(x)
    loss = output.loss
    print(f"loss = {loss}")
    # capture(model, x)



if __name__ == '__main__':
    # example_1()
    # exmaple_2()
    # exmaple_3()
    # exmaple_use_warp_module()
    example_split_test()
    # example_op_aten_test()
    # exmaple_aotgrad_test()
    # exmaple_change_save()
    # example_save_class_test()