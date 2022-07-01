import gym
import os, shutil
from typing import Optional, Union
from tqdm import trange

# Bugfix for linux
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
    
    '''
    Records example episodes under the given policy, saving them as a concatenated gif (converted from default mp4 format).
    The resulting gif is saved in the saves folder, and the most recent step recording is saved in the examples folder.
    Args:
        policy (Policy):    The policy to query throughout the episode
        count (int):        The number of episodes to record and save
        train_step (int):   The training iteration this policy represents, which is included in the save video names
    Returns:
        str:                Path to the newly recorded gif
    '''
    def record_episodes(self, policy: Policy, count: int, train_step: int) -> str:
        record_path = os.path.join(Config.instance_save_folder(self.config.name, self.config.instance), RECORDING_DIR)
        step_id = f"step-{train_step}"
        #wrapped_env = gym.wrappers.RecordVideo(self.env, record_path, episode_trigger = lambda x: True, name_prefix = f"step-{train_step}")
        wrapped_env = gym.wrappers.Monitor(self.env, record_path, video_callable = lambda x: True, uid = step_id, resume = True)
        
        for i in trange(count, leave = False):
            self.run_episode(policy, wrapped_env)
            
        wrapped_env.close()
            
        # Remove unnecessary folders, and collect mp4 videos into a single gif
        recorded_clips = []
        recorded_clip_filepaths = [] # Save original filepaths to be deleted after use
        for file in os.scandir(record_path):
            # All newly created files have this prefix
            if file.name.startswith("openaigym"):
                # Remove stats, manifest and metadata jsons
                if file.name.endswith(".json"):
                    os.remove(file.path)
                    
                # Rename mp4 files and convert to gifs
                if file.name.endswith(".mp4"):
                    name_parts = file.name.split('.')
                    name_step_id = name_parts[3]
                    record_it = int(name_parts[4][5:]) # Remove "video" prefix and leading 0s from number
                    
                    if name_step_id == step_id:
                        recorded_clips.append(VideoFileClip(file.path))
                        recorded_clip_filepaths.append(file.path)
        
        # Combine individual episode recordings into one (concatenation)
        final_clip = concatenate_videoclips(recorded_clips)
        final_clip = final_clip.subclip(t_end=final_clip.duration - 1 / final_clip.fps) # bug workaround, since concatenation adds an extra frame
        
        # Save concatenated clips as gif in saves folder (ignored by git)
        filename = f"{step_id}.gif"
        saves_filepath = os.path.join(record_path, filename)
        final_clip.write_gif(saves_filepath)
        
        # Delete original mp4 episode recordings (now that they've been rewritten)
        final_clip.close()
        for clip in recorded_clips:
            clip.close()
        for clip_filepath in recorded_clip_filepaths:
            os.remove(clip_filepath)
        
        # Copy latest recording to example folder (synced with git, only contains most recent recording)
        instance_example_folder = Config.instance_example_folder(self.config.name, self.config.instance)
        os.makedirs(instance_example_folder, exist_ok = True)
        for file in os.scandir(instance_example_folder):
            os.remove(file.path)
            
        examples_filepath = os.path.join(instance_example_folder, filename)
        shutil.copyfile(saves_filepath, examples_filepath)
                
        return saves_filepath
    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)