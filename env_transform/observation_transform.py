from abc import ABC, abstractmethod
import numpy as np

'''
Interface for converting observations into a format different from the default of the
chosen environment
'''

class ObservationTransform(ABC):
    '''
    Convert an observation from env format to internal format.
    Args:
        observation (np.ndarray):   env-format observation
    Returns:
        (np.ndarray):               internal-format observation
    '''
    @abstractmethod
    def from_env(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    '''
    Convert an action from internal format to env format.
    Args:
        observation (np.ndarray):   internal-format observation
    Returns:
        (np.ndarray):               env-format observation
    '''
    @abstractmethod
    def to_env(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)