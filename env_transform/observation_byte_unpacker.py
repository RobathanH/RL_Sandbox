from dataclasses import dataclass
import numpy as np

from .observation_transform import ObservationTransform

'''
Unpacks an array of bytes into a larger array containing binary bit elements
'''
@dataclass
class ByteUnpacker(ObservationTransform):
    
    def from_env(self, observation: np.ndarray) -> np.ndarray:
        return np.unpackbits(observation).astype(np.float32)
    
    # TODO
    def to_env(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError