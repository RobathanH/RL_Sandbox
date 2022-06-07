from dataclasses import dataclass, is_dataclass
from enum import Enum
import os
import json
from typing import Any, Optional
import numpy as np

from env_handler.env_handler import EnvHandler_Config
from exp_buffer.exp_buffer import ExpBuffer_Config
from trainer.trainer import Trainer_Config

'''
Config dataclass that specifies a particular instance of an RL method on a particular environment.
Contains multiple config dataclasses for specific components
'''
        
    
    

# Save Info (model data, etc saved in 'global_save_folder/name/instance/...')
GLOBAL_SAVE_FOLDER = "saves" # Folder to save data for any and all config instances
GLOBAL_EXAMPLE_FOLDER = "examples" # Folder to save examples from most recent training for any and all config instances
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
    
    
    
    instance: int = 0                           # Allows multiple saves and versions of the same base config



    '''
    Turn this base config into a new instance, with optional value overrides
    Args:
        instance (int):                 Instance index for new instance
        overrides (dict[str, Any]):     Values to override. Fields must be period-separated string of nested fields
                                        Ex: {"exp_buffer.capacity": 10000}
    '''
    def to_new_instance(self, instance: int, overrides: dict[str, Any] = {}) -> None:
        self.instance = instance
            
        for key, val in overrides.items():
            obj = self
            fields = key.split('.')
            for field in fields[:-1]:
                obj = getattr(obj, field)
            setattr(obj, fields[-1], val)
    



    # Static Config JSON Save/Load Functions
    
    @staticmethod
    def global_save_folder() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", GLOBAL_SAVE_FOLDER))
    
    @staticmethod
    def config_save_folder(config_name: str) -> str:
        return os.path.join(Config.global_save_folder(), config_name)
    
    @staticmethod
    def instance_save_folder(config_name: str, config_instance: int) -> str:
        return os.path.join(Config.config_save_folder(config_name), f"{INSTANCE_FOLDER_PREFIX}{config_instance}")
    
    @staticmethod
    def instance_save_path(config_name: int, config_instance: int) -> str:
        return os.path.join(Config.instance_save_folder(config_name, config_instance), CONFIG_SAVENAME)
    
    @staticmethod
    def instance_save_exists(config_name: str, config_instance: int) -> str:
        return os.path.exists(Config.instance_save_path(config_name, config_instance))
    
    @staticmethod
    def max_saved_instance(config_name: str) -> Optional[int]:
        max_instance = None
        if os.path.exists(Config.config_save_folder(config_name)):
            for name in os.listdir(Config.config_save_folder(config_name)):
                if name.startswith(INSTANCE_FOLDER_PREFIX):
                    instance = int(name[len(INSTANCE_FOLDER_PREFIX):])
                    if max_instance is None or instance > max_instance:
                        max_instance = instance
        return max_instance
    
    
    @staticmethod
    def global_example_folder() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", GLOBAL_EXAMPLE_FOLDER))
    
    @staticmethod
    def config_example_folder(config_name: str) -> str:
        return os.path.join(Config.global_example_folder(), config_name)
    
    @staticmethod
    def instance_example_folder(config_name: str, config_instance: int) -> str:
        return os.path.join(Config.config_example_folder(config_name), f"{INSTANCE_FOLDER_PREFIX}{config_instance}")
    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)