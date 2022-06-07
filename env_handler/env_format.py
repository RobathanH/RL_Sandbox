from dataclasses import dataclass
from typing import Optional
import numpy as np

'''
Action Space Specifier. Config may specify a DiscreteActionSpace or a ContinuousActionSpace
'''
@dataclass
class DiscreteActionSpace:
    count: int
    
@dataclass
class ContinuousActionSpace:
    shape: list[int]
    lower_bound: Optional[np.ndarray]
    upper_bound: Optional[np.ndarray]
    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)