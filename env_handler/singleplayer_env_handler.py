import gym
import os, shutil
from typing import Optional, Union
from tqdm import trange
from tempfile import TemporaryDirectory

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
    observation_space: list[int]
    
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
            else:
                trajectory_done = episode_done or ("TimeLimit.truncated" in info and info["TimeLimit.truncated"])
            
            traj.add_step(action, reward, next_obs, trajectory_done)
            steps_taken += 1
            obs = next_obs

        return traj

    def run_episodes(self, policy: Policy, count: int) -> list[Trajectory]:
        trajectories = []
        for i in trange(count, leave = False):
            trajectories.append(self.run_episode(policy))
        return trajectories
    
    def run_parallel_episodes(self, batch_policy: Policy, count: int) -> list[Trajectory]:
        env = gym.vector.make(self.env_handler_config.name, num_envs=count)
        
        batch_trajectories = [Trajectory() for _ in range(count)]
        
        batch_env_obs = env.reset()
            
        # Preprocess raw observations from env
        if self.env_handler_config.observation_transform is None:
            batch_obs = batch_env_obs
        else:
            batch_obs = np.concatenate([
                self.env_handler_config.observation_transform.from_env(env_obs)[None]
                for env_obs in batch_env_obs
            ], axis=0)
                
        # Add to trajectories
        for i in range(count):
            batch_trajectories[i].add_start(batch_obs[i])
            
        steps_taken = 0
        batch_trajectories_complete = [False for _ in range(count)]
        while not all(batch_trajectories_complete) and (self.env_handler_config.max_episode_steps is None or steps_taken < self.env_handler_config.max_episode_steps):
            batch_actions = batch_policy.get_action(batch_obs)
            
            # Postprocess action before sending to env
            if self.env_handler_config.action_transform is None:
                batch_env_actions = batch_actions
            else:
                batch_env_actions = np.concatenate([
                    self.env_handler_config.action_transform.to_env(action)[None]
                    for action in batch_actions
                ], axis=0)
                
            batch_env_next_obs, batch_rewards, batch_episode_done, batch_info = env.step(batch_env_actions)
            
            # Preprocess raw observations from env
            if self.env_handler_config.observation_transform is None:
                batch_next_obs = batch_env_next_obs
            else:
                batch_next_obs = np.concatenate([
                    self.env_handler_config.observation_transform.from_env(env_next_obs)[None]
                    for env_next_obs in batch_env_next_obs
                ], axis=0)
                
            # Add step to trajectories
            for i in range(count):
                # If this index has already run a full episode and reset, do not save further partial episodes
                if batch_trajectories_complete[i]:
                    continue
                
                episode_done = batch_episode_done[i]
                timelimit_truncated = (
                    "TimeLimit.truncated" in batch_info and
                    "_TimeLimit.truncated" in batch_info and
                    batch_info["_TimeLimit.truncated"][i] and
                    batch_info["TimeLimit.truncated"][i]
                )
                if not self.env_handler_config.time_limit_counts_as_terminal_state:
                    # Do not mark trajectory steps as 'done' if episode is ended by a timelimit, otherwise env is not MDP
                    trajectory_done = episode_done and not timelimit_truncated
                else:
                    trajectory_done = episode_done or timelimit_truncated
                    
                batch_trajectories[i].add_step(batch_actions[i], batch_rewards[i], batch_next_obs[i], trajectory_done)
                
                # Mark batch elements which have completed a full trajectory
                if trajectory_done:
                    batch_trajectories_complete[i] = True
                    
            # Increment
            steps_taken += 1
            batch_obs = batch_next_obs
            
        return batch_trajectories
                    
    
    '''
    Records example episodes under the given policy, saving them as a concatenated gif (converted from default mp4 format).
    The resulting gif is saved in the checkpoint folder, overwriting any previous recordings.
    Args:
        policy (Policy):    The policy to query throughout the episode
        count (int):        The number of episodes to record and save
        train_step (int):   The training iteration this policy represents, which is included in the save video names
    Returns:
        str:                Path to the newly recorded gif
    '''
    def record_episodes(self, policy: Policy, count: int, train_step: int) -> str:
        # Import here, so any import errors can be avoided by disabling recording
        # os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg" # Bugfix for linux
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        with TemporaryDirectory() as record_dir:
            step_id = f"step-{train_step}"
            #wrapped_env = gym.wrappers.RecordVideo(self.env, record_path, episode_trigger = lambda x: True, name_prefix = f"step-{train_step}")
            wrapped_env = gym.wrappers.Monitor(self.env, record_dir, video_callable = lambda x: True, uid = step_id, resume = True)
            
            for i in trange(count, leave = False):
                self.run_episode(policy, wrapped_env)
                
            wrapped_env.close()
                
            # Collect mp4 videos for each episode
            recorded_clips = []
            for file in os.scandir(record_dir):
                if file.name.startswith("openaigym") and file.name.endswith(".mp4"):
                    name_parts = file.name.split('.')
                    name_step_id = name_parts[3]
                    
                    if name_step_id == step_id:
                        recorded_clips.append(VideoFileClip(file.path))
            
            # Combine individual episode recordings into one (concatenation)
            final_clip = concatenate_videoclips(recorded_clips)
            final_clip = final_clip.subclip(t_end=final_clip.duration - 1 / final_clip.fps) # bug workaround, since concatenation adds an extra frame
            
            # Save concatenated clips as gif in checkpoint folder
            recording_filepath = os.path.join(Config.checkpoint_folder(), "recording.gif")
            final_clip.write_gif(recording_filepath, loop=True)
            
            final_clip.close()
            for clip in recorded_clips:
                clip.close()
                                
        return recording_filepath
    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)