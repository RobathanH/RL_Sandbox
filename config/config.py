from dataclasses import dataclass, is_dataclass
from enum import Enum
import os
import json
from typing import Tuple, Type, Optional, Union
import numpy as np
from env_handler.env_handler import EnvHandler_Config

from env_transform.action_transform import ActionTransform
from env_transform.observation_transform import ObservationTransform
from exp_buffer.exp_buffer import ExpBuffer, ExpBuffer_Config
from trainer.trainer import Trainer, Trainer_Config

'''
Config dataclass that specifies a particular instance of an RL method on a particular environment.
Contains multiple config dataclasses for specific components
'''
    
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

    # Environment Handler
    env_handler: EnvHandler_Config              # EnvHandler Specifier
    
    # Experience Buffer
    exp_buffer: ExpBuffer_Config                # ExpBuffer Specifier

    # RL method
    trainer: Trainer_Config                     # Trainer Specifier
    
    
    
    # Save Info (model data, etc saved in 'global_save_folder/name/instance/...')
    # Generally not specified in registered config
    instance_index: int = 0                     # Allows multiple saves and versions of the same config
    global_save_folder: str = "saves"           # Folder to save data for any and all config instances
    global_example_folder: str = "examples"     # Folder to save examples from most recent training for any and all config instances



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
    
    '''
    Return example folder for this config instance
    Returns:
        (str)
    '''
    def instance_examplefolder(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", self.global_example_folder, self.name, INSTANCE_FOLDER_PREFIX + str(self.instance_index)))