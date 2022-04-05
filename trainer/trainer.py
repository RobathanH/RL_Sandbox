from abc import ABC, abstractmethod
from typing import Optional, Type

from exp_buffer.exp_format import Trajectory
from exp_buffer.exp_buffer import ExpBuffer
from policy.policy import Policy

'''
Interface class for taking training data and iteratively improving a policy using an arbitrary RL algorithm.
Also handles experience buffer
'''
class Trainer(ABC):

    # Environment Interaction

    '''
    Returns the current trained policy, for choosing actions in the env
    Returns:
        (Policy)
    '''
    @abstractmethod
    def current_policy(self) -> Policy:
        raise NotImplementedError
    
    '''
    Returns the current training step (Number of training loops completed).
    Returns:
        (int)
    '''
    @abstractmethod
    def current_train_step(self) -> int:
        raise NotImplementedError



    # Improvement

    '''
    Performs a training session using a given set of experience tuples from
    the buffer.
    Args:
        exp:    List of experience tuples from the experience buffer.
    Returns:
        (Dict):     Dictionary of metric names and values over this train step
    '''
    @abstractmethod
    def train(self, exp_buffer: ExpBuffer) -> dict:
        raise NotImplementedError
    
    '''
    Returns true if the trainer is on-policy, and the exp buffer should be
    cleared after each train step
    '''
    @abstractmethod
    def on_policy(self) -> bool:
        raise NotImplementedError



    # Loading and Saving

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, folder: Optional[str] = None) -> None:
        raise NotImplementedError
    
    
'''
Interface class for storing trainer-specific config
'''
class Trainer_Config(ABC):
    '''
    Return Trainer class associated with this config
    '''
    @abstractmethod
    def get_class(self) -> Type[Trainer]:
        raise NotImplementedError