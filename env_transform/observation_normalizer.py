from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .observation_transform import ObservationTransform

'''
Normalizes observation element values into a specified range of floats.
Assumes env-format range and desired internal-format range is the same for all
elements of the observation array.
'''
@dataclass
class ObservationNormalizer(ObservationTransform):
    env_lower_bound: float
    env_upper_bound: float
    internal_lower_bound: float
    internal_upper_bound: float
    
    def from_env(self, observation: np.ndarray) -> np.ndarray:
        return self.internal_lower_bound + (self.internal_upper_bound - self.internal_lower_bound) * (observation - self.env_lower_bound) / (self.env_upper_bound - self.env_lower_bound)
    
    def to_env(self, observation: np.ndarray) -> np.ndarray:
        return self.env_lower_bound + (self.env_upper_bound - self.env_lower_bound) * (observation - self.internal_lower_bound) / (self.internal_upper_bound - self.internal_lower_bound)