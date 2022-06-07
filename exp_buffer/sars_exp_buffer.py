import os
import numpy as np
import torch
from typing import Optional

from config.config import Config

from .exp_buffer import *
from .exp_format import *

'''
Basic Experience Replay Buffer for Q-Learning, saving tuples of state, action, reward, and next-state
'''

class SarsExpBuffer(ExpBuffer):
    '''
    Args:
        config (Config)
    '''
    def __init__(self, config: Config) -> None:
        self.config = config

        # State variables
        self.data = {}
        self.next_index = 0



    # File Operations

    def save(self) -> None:
        # Create dirs if needed
        os.makedirs(Config.instance_save_folder(self.config.name, self.config.instance), exist_ok = True)
        
        savepath = os.path.join(Config.instance_save_folder(self.config.name, self.config.instance), EXP_BUFFER_SAVENAME)
        torch.save(self.data, savepath)

    def load(self, folder: Optional[str] = None) -> None:
        if folder is None:
            folder = Config.instance_save_folder(self.config.name, self.config.instance)

        savepath = os.path.join(folder, EXP_BUFFER_SAVENAME)
        
        if not os.path.exists(savepath):
            return
        
        self.data = torch.load(savepath)



    # Buffer Editing

    def clear(self) -> None:
        self.data = {}
        self.next_index = 0
    
    def add_trajectories(self, trajs: List[Trajectory]) -> None:
        for traj in trajs:
            self.add_trajectory(traj)

    def add_trajectory(self, traj: Trajectory) -> None:
        cur_state = traj.start.state
        for step in traj.steps:
            if not step.done:
                self.data[self.next_index] = ExpSars(cur_state, step.action, step.reward, step.next_state)
            else:
                self.data[self.next_index] = ExpSars(cur_state, step.action, step.reward)

            self.next_index = (self.next_index + 1) % self.config.exp_buffer_capacity
            cur_state = step.next_state

    

    # Buffer Retrieval

    '''
    Returns a list of experience tuples in the exp buffer.
    Args:
        count   (Optional[int]):    Number of experiences to return.
        shuffle (bool):             Whether to shuffle the returned experiences.
    '''
    def get(self, count: Optional[int] = None, shuffle: bool = True) -> List[Any]:
        if count is None:
            count = len(self.data)
        else:
            count = min(count, len(self.data))

        if shuffle:
            order = np.random.permutation(len(self.data))[:count]
        else:
            order = np.arange(len(self.data) - count, len(self.data))

        return [self.data[i] for i in order]
    
    '''
    Returns the current number of stored experiences
    '''
    def size(self) -> int:
        return len(self.data)    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)