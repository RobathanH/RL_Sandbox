from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .action_transform import ActionTransform

'''
Discretize continuous actions into discrete categories
'''

@dataclass
class ActionDiscretizer(ActionTransform):
    action_shape: Tuple[int]            # Shape of env-format actions, and thus also the shape of the other variables
    disc_counts: np.ndarray             # Number of bins to discretize each action element into
    cont_lower_bound: np.ndarray        # Minimum value for each action element
    cont_upper_bound: np.ndarray        # Maximum value for each action element
        
    
    def from_env(self, action: np.ndarray) -> np.ndarray:
        cont_bin_size = (self.cont_upper_bound - self.cont_lower_bound) / self.disc_counts
        rescaled_action = np.clip((action - self.cont_lower_bound) / cont_bin_size, np.zeros(self.action_shape), self.disc_counts - 1)
        discretized_action = np.floor(rescaled_action)
        return discretized_action
    
    def to_env(self, action: np.ndarray) -> np.ndarray:
        cont_bin_size = (self.cont_upper_bound - self.cont_lower_bound) / self.disc_counts
        rescaled_action = self.cont_lower_bound + action * cont_bin_size
        bin_centered_action = rescaled_action + 0.5 * cont_bin_size
        return bin_centered_action