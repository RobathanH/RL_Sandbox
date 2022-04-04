from abc import ABC, abstractmethod
import numpy as np

'''
Interface class for instances of a policy, which will make actions when given states in the environment.
TODO: Clearly differentiate between discrete and continuous policies
'''
class Policy(ABC):
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError