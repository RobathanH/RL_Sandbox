from abc import ABC, abstractmethod
from typing import Any
import numpy as np

'''
Interface for converting actions into a format different from the default of the
chosen environment
'''

class ActionTransform(ABC):
    '''
    Convert an action from env format to internal format.
    Args:
        action (Any):   env-format action
    Returns:
        (Any):          internal-format action
    '''
    @abstractmethod
    def from_env(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    '''
    Convert an action from internal format to env format.
    Args:
        action (Any):   internal-format action
    Returns:
        (Any):          env-format action
    '''
    @abstractmethod
    def to_env(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)