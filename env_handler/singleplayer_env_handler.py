import json
import gym
import os
from typing import Optional, Union, Tuple
from tqdm import trange
import numpy as np

from config.config import Config
from policy.policy import Policy
from env_transform.action_transform import ActionTransform
from env_transform.observation_transform import ObservationTransform
from exp_buffer.exp_format import *

from .env_handler import *
from .env_format import *

'''
Wrapper for storing and running singleplpayer environments.
'''



'''
EnvHandler Config Information
'''

@dataclass
class SinglePlayerEnvHandler_Config(EnvHandler_Config):
    name: str
    
    # Environment input/output shape (after any transforms)
    action_space: Union[DiscreteActionSpace, ContinuousActionSpace]
    observation_space: Tuple[int, ...]
    
    # Environment RL Constants
    discount_rate: float = 0.99
    
    # Environment input/output transforms
    action_transform: Optional[ActionTransform] = None
    observation_transform: Optional[ObservationTransform] = None
    
    # Environment Running Constants
    max_episode_steps: Optional[int] = None
    time_limit_counts_as_terminal_state: bool = False
    
    '''
    Return the associated EnvHandler class type
    '''
    def get_class(self):
        return SinglePlayerEnvHandler
    



# File Constants
RECORDING_DIR = "recordings"

'''
Base EnvHandler Class
'''

class SinglePlayerEnvHandler:
    def __init__(self, config: Config):
        if type(config.env_handler) is not SinglePlayerEnvHandler_Config:
            raise ValueError(f"config.env_handler must be of type SinglePlayerEnvHandler_Config")
        
        self.config: Config = config
        self.env_handler_config: SinglePlayerEnvHandler_Config = config.env_handler
        
        # Create env
        self.env = gym.make(self.env_handler_config.name)

    def run_episode(self, policy: Policy, wrapped_env: Optional[gym.Wrapper] = None) -> Trajectory:
        
        # Optional env wrapper, useful for recording
        if wrapped_env is not None:
            env = wrapped_env
        else:
            env = self.env
        
        traj = Trajectory()
        
        env_obs = env.reset()
        if self.env_handler_config.observation_transform is None:
            obs = env_obs
        else:
            obs = self.env_handler_config.observation_transform.from_env(env_obs)
            
        traj.add_start(obs)
        steps_taken = 0
        episode_done = False
        while not episode_done and (self.env_handler_config.max_episode_steps is None or steps_taken < self.env_handler_config.max_episode_steps):
            action = policy.get_action(obs)
            if self.env_handler_config.action_transform is None:
                env_action = action
            else:
                env_action = self.env_handler_config.action_transform.to_env(action)
            
            env_next_obs, reward, episode_done, info = env.step(env_action)
            if self.env_handler_config.observation_transform is None:
                next_obs = env_next_obs
            else:
                next_obs = self.env_handler_config.observation_transform.from_env(env_next_obs)
                
            if not self.env_handler_config.time_limit_counts_as_terminal_state:
                # Do not mark trajectory steps as 'done' if episode is ended by a timelimit, otherwise env is not MDP
                trajectory_done = episode_done and not ("TimeLimit.truncated" in info and info["TimeLimit.truncated"])
            
            traj.add_step(action, reward, next_obs, trajectory_done)
            steps_taken += 1
            obs = next_obs

        return traj

    def run_episodes(self, policy: Policy, count: int) -> list[Trajectory]:
        trajectories = []
        for i in trange(count, leave = False):
            trajectories.append(self.run_episode(policy))
        return trajectories
    
    def record_episodes(self, policy: Policy, count: int, train_step: int) -> None:
        record_path = os.path.join(self.config.instance_savefolder(), RECORDING_DIR)
        #wrapped_env = gym.wrappers.RecordVideo(self.env, record_path, episode_trigger = lambda x: True, name_prefix = f"step-{train_step}")
        wrapped_env = gym.wrappers.Monitor(self.env, record_path, video_callable = lambda x: True, uid = f"step-{train_step}", resume = True)
        
        for i in trange(count, leave = False):
            self.run_episode(policy, wrapped_env)
            