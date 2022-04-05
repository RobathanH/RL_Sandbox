import os
import numpy as np
import torch
from typing import Optional

from config.config import Config

from .exp_buffer import *
from .exp_format import *

'''
Basic Experience Replay Buffer for Q-Learning, saving tuples of state, action, reward, and next-state
TDExpBuffer stores tuples with aggregated discounted return for an arbitrary number of steps forward,
thus it can represent both SARS tuples (td_steps = 1) and MC tuples (td_steps = None, store full return
to end of episode)
'''

@dataclass
class TDExpBuffer_Config(ExpBuffer_Config):
    capacity: int
    td_steps: Optional[int]
    
    def __post_init__(self) -> None:
        if self.td_steps is not None and self.td_steps <= 0:
            raise ValueError(f"TDExpBuffer_Config.td_steps must be None or positive")
        
    def get_class(self) -> Type[ExpBuffer]:
        return TDExpBuffer
    

class TDExpBuffer(ExpBuffer):
    '''
    Args:
        config (Config)
    '''
    def __init__(self, config: Config) -> None:
        if type(config.exp_buffer) is not TDExpBuffer_Config:
            raise ValueError(f"TDExpBuffer requires TDExpBuffer_Config")
        
        self.config: Config = config
        self.exp_buffer_config: TDExpBuffer_Config = config.exp_buffer

        # State variables
        self.data = {}
        self.next_index = 0



    # File Operations

    def save(self) -> None:
        # Create dirs if needed
        os.makedirs(self.config.instance_savefolder(), exist_ok = True)
        
        savepath = os.path.join(self.config.instance_savefolder(), EXP_BUFFER_SAVENAME)
        torch.save(self.data, savepath)

    def load(self, folder: Optional[str] = None) -> None:
        if folder is None:
            folder = self.config.instance_savefolder()

        savepath = os.path.join(folder, EXP_BUFFER_SAVENAME)
        
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
        N = self.exp_buffer_config.td_steps
        if N is None or N > traj.length():
            N = traj.length()
        
        cur_state = traj.start.state
        cur_action = traj.steps[0].action
        cur_reward = sum(traj.steps[i].reward * self.config.env_handler.discount_rate**i for i in range(N))
        cur_next_state = traj.steps[N - 1].next_state if not traj.steps[N - 1].done else None
        
        self.data[self.next_index] = ExpTD(cur_state, cur_action, cur_reward, cur_next_state)
        self.next_index = (self.next_index + 1) % self.exp_buffer_config.capacity
        
        done_reached = traj.steps[N - 1].done

        for i in range(1, traj.length()):
            cur_state = traj.steps[i - 1].next_state
            cur_action = traj.steps[i].action
            
            cur_reward -= traj.steps[i - 1].reward
            cur_reward /= self.config.env_handler.discount_rate
            if i + N - 1 < traj.length():
                cur_reward += traj.steps[i + N - 1].reward * self.config.env_handler.discount_rate**(N - 1)
                
            if i + N - 1 < traj.length():
                if not traj.steps[i + N - 1].done:
                    cur_next_state = traj.steps[i + N - 1].next_state
                else:
                    cur_next_state = None
                    done_reached = True
            else:
                if done_reached:
                    cur_next_state = None
                else:
                    # If trajectory ends without episode officially ending (non-terminal state), 
                    # n-step estimate is not valid, so we stop collecting tuples which go beyond those bounds.
                    # Optionally, we could include these tuples for k-step estimates (bootstrap from non-terminal
                    # end state)
                    return
                
            self.data[self.next_index] = ExpTD(cur_state, cur_action, cur_reward, cur_next_state)
            self.next_index = (self.next_index + 1) % self.exp_buffer_config.capacity

    

    # Buffer Retrieval

    '''
    Returns a list of experience tuples in the exp buffer.
    Args:
        count   (Optional[int]):    Number of experiences to return.
        shuffle (bool):             Whether to shuffle the returned experiences.
    '''
    def get(self, count: Optional[int] = None, shuffle: bool = True) -> list:
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