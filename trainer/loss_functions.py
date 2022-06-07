from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn.functional as F


'''
Dataclass subclasses for specifying different loss functions in config
'''

class Loss(ABC):
    @abstractmethod
    def apply(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    
    
@dataclass
class MSE_Loss(Loss):
    def apply(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target)
    
@dataclass
class Huber_Loss(Loss):
    def apply(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(input, target)    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)