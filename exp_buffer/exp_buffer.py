from abc import ABC, abstractmethod
from typing import Optional, Type

from .exp_format import *

'''
Store data from trajectories, allowing training with and without experience replay.
Stored and managed by policy trainer (to allow both online and offline policies)
'''

EXP_BUFFER_SAVENAME = "exp_buffer.buf"
class ExpBuffer(ABC):

    # File Operations

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, folder: Optional[str] = None) -> None:
        raise NotImplementedError



    # Buffer Editing

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
    
    def add_trajectories(self, trajs: list[Trajectory]) -> None:
        for traj in trajs:
            self.add_trajectory(traj)

    @abstractmethod
    def add_trajectory(self, traj: Trajectory) -> None:
        raise NotImplementedError

    

    # Buffer Retrieval

    '''
    Returns a list of experience tuples in the exp buffer.
    Args:
        count   (Optional[int]):    Number of experiences to return.
        shuffle (bool):             Whether to shuffle the returned experiences.
    '''
    @abstractmethod
    def get(self, count: Optional[int] = None, shuffle: bool = True) -> list:
        raise NotImplementedError
    
    '''
    Returns the current number of stored experiences
    '''
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    


'''
ExpBuffer Config
'''
class ExpBuffer_Config(ABC):
    '''
    Return the ExpBuffer type associated with this config
    '''
    @abstractmethod
    def get_class(self) -> Type[ExpBuffer]:
        raise NotImplementedError