# import torch
# import torch.nn as nn
# from torch.autograd import Function
# import torch.onnx

# #custom op for onnx representation
# class MyStrangeOp2(Function):
# 	#for onnx graph
#    @staticmethod
#    def symbolic(g, input1, input2, bias, int_attr1, int_attr2, str_attr3):
#       return g.op("MyStrangeOp2", input1, input2, bias, int_attr1_i=int_attr1, int_attr2_i=int_attr2, str_attr3_s=str_attr3)

#    @staticmethod
#    def forward(ctx, input1, input2, bias, int_attr1, int_attr2, str_attr3):
#       return input1 + input2 - bias

# myStrangeOp2_forward = MyStrangeOp2.apply

# #layer
# class MyStrangeOp2Layer(nn.Module):
#     def __init__(self, bias, int_attr1, int_attr2, str_attr3):
#         super(MyStrangeOp2Layer, self).__init__()
#         self.bias = bias
#         self.int_attr1 = int_attr1
#         self.int_attr2 = int_attr2
#         self.str_attr3 = str_attr3

#     def forward(self, in1, in2):
#       assert in1.dim() != 100
#       if in1.requires_grad:
#          print(in1)
#       if in2.numel():
#            return myStrangeOp2_forward(in1, in2, self.bias, self.int_attr1, self.int_attr2, self.str_attr3)
#       else:
#            return myStrangeOp2_forward(in1, in2, self.bias, self.int_attr1, self.int_attr2, self.str_attr3)

# #net
# class MyStrangeNet2(nn.Module):
#    def __init__(self):
#       super(MyStrangeNet2, self).__init__()
#       self.myLayer1 = MyStrangeOp2Layer(bias=nn.Parameter(torch.ones(1, 3, 4, 4)), int_attr1=10, int_attr2=[3, 5], str_attr3="fuck" )
#       self.myLayer2 = MyStrangeOp2Layer(bias=nn.Parameter(torch.ones(1, 3, 4, 4)), int_attr1=40, int_attr2=[12, 22], str_attr3="shit")
#       self.conv1    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
#       self.conv2    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)

#    def forward(self, in1, in2, in3, in4):
#       assert in1.dim() != 100
#       if in1.requires_grad:
#          print(in1)
#       if in4.numel():
#          x1 = self.myLayer1(in1, in2)
#          x2 = self.myLayer2(in3, in4)
#          x1 = self.conv1(x1)
#          x2 = self.conv2(x2)
#       else:
#          x1 = self.myLayer1(in1, in2)
#          x2 = self.myLayer2(in3, in4)
#          x1 = self.conv1(x1)
#          x2 = self.conv2(x2)
#       return x1 + x2


# #fake input
# model = MyStrangeNet2()
# t1 = torch.ones(1, 3, 4, 4, dtype=torch.float32)
# t2= torch.ones(1, 3, 4, 4, dtype=torch.float32)
# t3 = torch.ones(1, 3, 4, 4, dtype=torch.float32)
# t4 = torch.ones(1, 3, 4, 4, dtype=torch.float32)
# #save onnx
# torch.onnx.export(model, (t1, t2, t3, t4), 'specific_net.onnx', opset_version=13, input_names=["input1", "input2", "input3", "input4",], output_names=["outputTensor"],
#                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


import onnx
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
model_path = 'specific_net.onnx'
onnx_model = onnx.load(model_path)
# onnx.checker.check_model(onnx_model)

# 启用profiling
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

# 创建ONNX Runtime session
ort_session = ort.InferenceSession(model_path, sess_options)

# 准备输入数据
input1 = np.random.randn(1, 3, 224, 224).astype(np.float32)  # 例子输入
input2 = np.random.randn(1, 3, 224, 224).astype(np.float32)
input3 = np.random.randn(1, 3, 224, 224).astype(np.float32)
input4 = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 运行模型
outputs = ort_session.run(None, {
    "input1": input1,
    "input2": input2,
    "input3": input3,
    "input4": input4
})

profile_file = ort_session.end_profiling()
print(f"Profiler output saved to: {profile_file}")