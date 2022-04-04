import json
import gym
import os
from typing import List, Optional
from tqdm import trange

from policy.policy import Policy
from config.config import Config

from exp_buffer.exp_format import *

'''
Wrapper for storing and running particular environments.
Every chosen environment from AI gym will be wrapped as a subclass of this function
'''
RECORDING_DIR = "recordings"
class EnvHandler:
    def __init__(self, config: Config):
        self.config = config
        
        # Env
        self.env = gym.make(config.env_name)

    def run_episode(self, policy: Policy, wrapped_env: Optional[gym.Wrapper] = None) -> Trajectory:
        
        # Optional env wrapper, useful for recording
        if wrapped_env is not None:
            env = wrapped_env
        else:
            env = self.env
        
        traj = Trajectory()
        
        env_obs = env.reset()
        if self.config.env_observation_transform is None:
            obs = env_obs
        else:
            obs = self.config.env_observation_transform.from_env(env_obs)
            
        traj.add_start(obs)
        steps_taken = 0
        episode_done = False
        while not episode_done and (self.config.env_max_episode_steps is None or steps_taken < self.config.env_max_episode_steps):
            action = policy.get_action(obs)
            if self.config.env_action_transform is None:
                env_action = action
            else:
                env_action = self.config.env_action_transform.to_env(action)
            
            env_next_obs, reward, episode_done, info = env.step(env_action)
            if self.config.env_observation_transform is None:
                next_obs = env_next_obs
            else:
                next_obs = self.config.env_observation_transform.from_env(env_next_obs)
                
            # Do not mark trajectory steps as 'done' if episode is ended by a timelimit, otherwise env is not MDP
            trajectory_done = episode_done and not ("TimeLimit.truncated" in info and info["TimeLimit.truncated"])
            
            traj.add_step(action, reward, next_obs, trajectory_done)
            steps_taken += 1
            obs = next_obs

        return traj

    def run_episodes(self, policy: Policy, count: int) -> List[Trajectory]:
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
            