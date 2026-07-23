import torch
import onnx
from torch import nn

# # Define a simple PyTorch model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x

# # Instantiate the model and prepare a dummy input
# # model = SimpleModel()
# # dummy_input = torch.randn(1, 1, 28, 28)


# from torchvision import models
# # import transformer

# model = getattr(models, "resnet50")()
# # dummy_input = (torch.LongTensor(1,512).random_() % 1000)
# dummy_input = torch.rand(1, 3, 224, 224)

# # Export the model to an ONNX file
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, opset_version=11)

# # Load the ONNX model
# onnx_model = onnx.load("model.onnx")
# onnx.checker.check_model(onnx_model)



import onnx
import onnxruntime as ort
import numpy as np

# 启用profiling
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

dummy_input = torch.rand(1, 3, 224, 224)
# 创建ONNX Runtime session
ort_session = ort.InferenceSession("model.onnx", sess_options)
sess_options.enable_profiling = True

input_name = ort_session.get_inputs()[0].name
dummy_input_np = dummy_input.numpy()
outputs = ort_session.run(None, {input_name: dummy_input_np})


profile_file = ort_session.end_profiling()
print(f"Profiler output saved to: {profile_file}")