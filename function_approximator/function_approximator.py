from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn

'''
Interface for dataclasses specifying the architecture of function approximators.
'''
class FunctionApproximator(ABC):
    '''
    Creates an instance of the module specified by a particular initialized
    FunctionApproximator subclass.
    Returns:
        (nn.Module):    Callable and trainable network instance
    '''
    @abstractmethod
    def create(self) -> nn.Module:
        raise NotImplementedError
    
    '''
    Return input shape, ignoring batch dimension.
    Outer tuple allows multiple input modalities.
    '''
    @abstractmethod
    def input_shape(self) -> Tuple[Tuple[int]]:
        raise NotImplementedError
    
    '''
    Return output shape, ignoring batch dimension.
    Outer tuple allows multiple output modalities.
    '''
    @abstractmethod
    def output_shape(self) -> Tuple[Tuple[int]]:
        raise NotImplementedError