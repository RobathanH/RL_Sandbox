from dataclasses import dataclass
import numpy as np

from .observation_transform import ObservationTransform

'''
Unpacks an array of bytes into a larger array containing binary bit elements.
Also scales 0,1 binary values to -1,+1
'''
@dataclass
class ByteUnpacker(ObservationTransform):
    
    def from_env(self, observation: np.ndarray) -> np.ndarray:
        return np.unpackbits(observation).astype(np.float32) * 2 - 1
    
    # TODO
    def to_env(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)