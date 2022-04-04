from enum import Enum
import torch.nn as nn

'''
Enum class for referencing activation types, and creating activation layers on command.
'''
class Activation(Enum):
    RELU = 1
    LRELU = 2
    SIGMOID = 3
    TANH = 4
    
    def create(self) -> nn.Module:
        if self is Activation.RELU:
            return nn.ReLU()
        
        if self is Activation.LRELU:
            return nn.LeakyReLU()
        
        if self is Activation.SIGMOID:
            return nn.Sigmoid()
        
        if self is Activation.TANH:
            return nn.Tanh()