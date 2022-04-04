from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .activation import Activation
from .util import ModuleTuple, ParameterModule, BoundedModule
from .function_approximator import FunctionApproximator

'''
Simple trainable parameter, ignoring input
'''
@dataclass
class Parameter(FunctionApproximator):
    shape: Tuple[int, ...]
    
    '''
    Creates an instance of the module specified by a particular initialized
    FunctionApproximator subclass.
    Returns:
        (nn.Module):    Callable and trainable network instance
    '''
    def create(self) -> nn.Module:
        return ParameterModule(self.shape)
    
    def input_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return None
    
    def output_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return (self.shape,)

'''
Simple linear transform
'''
@dataclass
class Linear(FunctionApproximator):
    input_len: int
    output_len: int
    
    '''
    Creates an instance of the module specified by a particular initialized
    FunctionApproximator subclass.
    Returns:
        (nn.Module):    Callable and trainable network instance
    '''
    def create(self) -> nn.Module:
        return nn.Linear(self.input_len, self.output_len)
    
    def input_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return ((self.input_len,),)
    
    def output_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return ((self.output_len,),)
    
    
    
'''
Basic multi-layer network.
'''
@dataclass
class MLP(FunctionApproximator):
    layer_sizes: List[int]
    activation: Activation = Activation.RELU
    final_layer_activation: bool = False
    bounded_output: Optional[Tuple[float, float]] = None
    
    '''
    Creates an instance of the module specified by a particular initialized
    FunctionApproximator subclass.
    Returns:
        (nn.Module):    Callable and trainable network instance
    '''
    def create(self) -> nn.Module:
        layers = []
        for i in range(1, len(self.layer_sizes)):
            layers.append(nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))
            if (self.final_layer_activation or i != len(self.layer_sizes) - 1) and self.bounded_output is None:
                layers.append(self.activation.create())
        
        if self.bounded_output is not None:
            layers.append(BoundedModule(torch.ones(self.layer_sizes[-1]) * self.bounded_output[0], torch.ones(self.layer_sizes[-1]) * self.bounded_output[1]))
                
        return nn.Sequential(*layers)
    
    def input_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return ((self.layer_sizes[0],),)
    
    def output_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return ((self.layer_sizes[-1],),)
    
    

'''
Multi-headed network
'''
@dataclass
class MultiheadModule(FunctionApproximator):
    shared_module: FunctionApproximator
    head_modules: Tuple[FunctionApproximator, ...]
    
    def __post_init__(self):
        for m in self.head_modules:
            if len(m.output_shape()) > 1:
                raise ValueError(f"Cannot create MultiheadModule where individual head modules have multiple outputs themselves.")

    '''
    Creates an instance of the module specified by a particular initialized
    FunctionApproximator subclass.
    Returns:
        (nn.Module):    Callable and trainable network instance
    '''
    def create(self) -> nn.Module:
        return nn.Sequential(
            self.shared_module.create(),
            ModuleTuple(tuple(m.create() for m in self.head_modules))
        )
        
    def input_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return self.shared_module.input_shape()
    
    def output_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(m.output_shape()[0] for m in self.head_modules)
    
'''
Sequential Networks
'''
@dataclass
class SequentialModule(FunctionApproximator):
    modules: List[FunctionApproximator]
    
    def create(self) -> nn.Module:
        return nn.Sequential(
            module.create() for module in self.modules
        )
        
    def input_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return self.modules[0].input_shape()
    
    def output_shape(self) -> Tuple[Tuple[int, ...], ...]:
        return self.modules[-1].output_shape()