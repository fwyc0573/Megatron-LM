
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class after_change_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.load_state_dict(torch.load(r'/data/ytyang/yichengfeng/StaticGraphs/test/save_file/state_dict.pt'))

    
    
    def forward(self, x):
        add = x + 3.141592653589793;  x = None
        relu = torch.relu(add);  add = None
        neg = relu.neg();  relu = None
        return neg
        
