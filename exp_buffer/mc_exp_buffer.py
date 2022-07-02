import os
import numpy as np
import torch
from typing import Optional

from config.config import Config

from .exp_buffer import *
from .exp_format import *

'''
Experience Buffer for MC evaluation, storing discounted return from visited states
Every-Visit: Stores tuples for every s-a tuple in each trajectory
'''

class MCExpBuffer(ExpBuffer):
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
        savepath = os.path.join(Config.checkpoint_folder(), EXP_BUFFER_SAVENAME)
        torch.save(self.data, savepath)

    def load(self) -> None:
        savepath = os.path.join(Config.checkpoint_folder(), EXP_BUFFER_SAVENAME)
        
        if not os.path.exists(savepath):
            return
        
        self.data = torch.load(savepath)



    # Buffer Editing

    def clear(self) -> None:
        self.data = {}
        self.next_index = 0
    
    def add_trajectories(self, trajs: list[Trajectory]) -> None:
        for traj in trajs:
            self.add_trajectory(traj)

    def add_trajectory(self, traj: Trajectory) -> None:
        returns = traj.steps[-1].reward
        action = traj.steps[-1].action
        state = None # Must be taken from previous step next_state
        for step in traj.steps[-2::-1]:
            state = step.next_state

            self.data[self.next_index] = ExpMC(state, action, returns)
            self.next_index = (self.next_index + 1) % self.config.exp_buffer_capacity

            returns = step.reward + returns * self.config.env_discount_rate
            action = step.action

        state = traj.start.state
        self.data[self.next_index] = ExpMC(state, action, returns)
        self.next_index = (self.next_index + 1) % self.config.exp_buffer_capacity

    

    # Buffer Retrieval

    '''
    Returns a list of experience tuples in the exp buffer.
    Args:
        count   (Optional[int]):    Number of experiences to return.
        shuffle (bool):             Whether to shuffle the returned experiences.
    '''
    def get(self, count: Optional[int] = None, shuffle: bool = True) -> list[Any]:
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