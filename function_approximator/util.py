from typing import Tuple
import torch
import torch.nn as nn

'''
Utility pytorch modules
'''

'''
Passes a single input to multipe modules,
return outputs in a tuple
'''
class ModuleTuple(nn.Module):
    def __init__(self, modules: list[nn.Module]) -> None:
        super(ModuleTuple, self).__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        return [module(x) for module in self.module_list]

'''
Module which ignores input and returns value of trainable parameters
'''
class ParameterModule(nn.Module):
    def __init__(self, shape: list[int]) -> None:
        super(ParameterModule, self).__init__()
        self.param = nn.Parameter(torch.zeros(shape), requires_grad = True)
        
    def forward(self, x):
        return self.param
    
'''
Converts an unbounded input to a bounded output
'''
class BoundedModule(nn.Module):
    def __init__(self, lower: torch.FloatTensor, upper: torch.FloatTensor) -> None:
        super(BoundedModule, self).__init__()
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        
    def forward(self, x):
        return self.lower + torch.sigmoid(x) * (self.upper - self.lower)    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)