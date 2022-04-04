from dataclasses import dataclass, is_dataclass
from enum import Enum
import os
import json
from typing import Tuple, Type, Optional, Union
import numpy as np

from env_transform.action_transform import ActionTransform
from env_transform.observation_transform import ObservationTransform
from exp_buffer.exp_buffer import ExpBuffer
from trainer.trainer import Trainer, Trainer_Config

'''
Config class that specifies a particular instance of an RL method on a particular environment.
Specific instances will override variables and be stored in the config registry.
'''

'''
Action Space Specifier. Config may specify a DiscreteActionSpace or a ContinuousActionSpace
'''
@dataclass
class DiscreteActionSpace:
    count: Tuple[int]
    
@dataclass
class ContinuousActionSpace:
    shape: Tuple[int]
    lower_bound: Optional[np.ndarray]
    upper_bound: Optional[np.ndarray]
    
    
'''
JSON Encoder for Config instances
'''
class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        # Encode Class Types by their names alone
        if isinstance(obj, type):
            return obj.__name__
        
        # Encode enums by their value name
        if isinstance(obj, Enum):
            return obj.name
        
        # Encode numpy arrays as nested lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Encode dataclasses by their dictionary formats
        if is_dataclass(obj):
            return [type(obj).__name__, obj.__dict__]
        
        return json.JSONEncoder.default(self, obj)
        
    
    

CONFIG_SAVENAME = "config.json"
INSTANCE_FOLDER_PREFIX = "instance_"
@dataclass
class Config:
    name: str                                   # Unique config identifier

    # Environment Specifier
    env_name: str                               # Exact AI Gym env string
    env_action_space: Union[DiscreteActionSpace, ContinuousActionSpace]     # Specifies whether action space is discrete or continuous, plus its shape and bounds
    env_observation_shape: Tuple[int]           # Shape of observation inputs
    
    # Environment Transforms
    env_action_transform: Optional[ActionTransform]             # Optional action format transform operation
    env_observation_transform: Optional[ObservationTransform]   # Optional observation format transform operation

    # Environment Running Constants
    env_max_episode_steps: Optional[int]        # Max episode length before termination. If None, no enforced limit
    env_episodes_per_step: int                  # Number of episodes to collect per step (before each training step)
    
    # Env RL Constants
    env_discount_rate: float                    # Discount rate for future rewards

    # RL method
    trainer_class: Type[Trainer]                # (Trainer subclass)
    trainer_config: Trainer_Config              # (Trainer-specific config dataclass)

    # Exp Buffer
    exp_buffer_class: Type[ExpBuffer]           # (ExpBuffer subclass)
    exp_buffer_capacity: int                    # Max number of tuples to store in exp buffer
    exp_buffer_td_level: Optional[int] = None   # Number of episodes to look ahead for exact rewards before bootstrapping. None indicates MC experience
    
    
    
    # Save Info (model data, etc saved in 'global_save_folder/name/instance/...')
    # Generally not specified in registered config
    instance_index: int = 0                     # Allows multiple saves and versions of the same config
    global_save_folder: str = "saves"           # Folder to save data for any and all config instances



    # General Save Folder for Config. Contains consistency file and instance saves separately

    '''
    Folder path for this specific configuration
    Returns:
        (str)
    '''
    def config_savefolder(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", self.global_save_folder, self.name))
    
    
    
    # Config Consistency File. Store json of config to ensure valid loads/saves
    
    '''
    Returns the full path for the config json file.
    Config json file is saved in general config-specific folder, separate from specific instances
    '''
    def consistency_filepath(self) -> str:
        return os.path.join(self.config_savefolder(), CONFIG_SAVENAME)
    
    '''
    Checks if this config has been saved before
    Returns:
        (bool)
    '''
    def config_save_exists(self) -> bool:
        return os.path.exists(self.consistency_filepath())
    
    '''
    Returns the JSON encoded string for this Config object
    '''
    def consistency_string(self) -> str:
        return json.dumps(self, indent = 4, cls = ConfigEncoder)

    '''
    Save the config as a json file inside its name-specific savefolder.
    Used to ensure config consistency between saves of the same name.
    '''
    def save_consistency_file(self) -> None:
        # Make config dir if needed
        os.makedirs(self.config_savefolder(), exist_ok = True)
        
        with open(self.consistency_filepath(), 'w') as fp:
            fp.write(self.consistency_string())

    '''
    Checks config consistency against an existing saved config file for
    this config name.
    '''
    def check_consistency(self) -> bool:
        if self.config_save_exists():
            with open(self.consistency_filepath(), 'r') as fp:
                saved_config_str = fp.read()
            current_config_str = self.consistency_string()
            
            if saved_config_str == current_config_str:
                return True

        return False
    
    
    
    # Instance Saves. Store saves of particular instances of this config in separate named folders.
    
    '''
    Folder path for this specific instance of this configuration
    Returns:
        (str)
    '''
    def instance_savefolder(self) -> str:
        return os.path.join(self.config_savefolder(), INSTANCE_FOLDER_PREFIX + str(self.instance_index))
    
    '''
    Checks if saves exist for this instance of this config
    Returns:
        (bool)
    '''
    def instance_save_exists(self) -> bool:
        return os.path.exists(self.instance_savefolder())

    '''
    Returns index of maximum saved instance for this config.
    Returns:
        (int | None):   Instance index or None if no instances saved
    '''
    def max_saved_instance(self) -> Optional[int]:
        max_instance = None
        for name in os.listdir(self.config_savefolder()):
            if name.startswith(INSTANCE_FOLDER_PREFIX):
                instance = int(name[len(INSTANCE_FOLDER_PREFIX):])
                if max_instance is None or instance > max_instance:
                    max_instance = instance
                    
        return max_instance