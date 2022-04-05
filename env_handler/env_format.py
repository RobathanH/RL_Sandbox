from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

'''
Action Space Specifier. Config may specify a DiscreteActionSpace or a ContinuousActionSpace
'''
@dataclass
class DiscreteActionSpace:
    count: int
    
@dataclass
class ContinuousActionSpace:
    shape: Tuple[int, ...]
    lower_bound: Optional[np.ndarray]
    upper_bound: Optional[np.ndarray]