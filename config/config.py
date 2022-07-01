from dataclasses import dataclass, is_dataclass
from enum import Enum
import os
import json
from typing import Any, Optional
import numpy as np

from env_handler.env_handler import EnvHandler_Config
from exp_buffer.exp_buffer import ExpBuffer_Config
from trainer.trainer import Trainer_Config

from .config_io import ConfigDecoder, ConfigEncoder

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
    Return a new Config instance, copied from the current, with the desired instance index and all overrides applied.
    Args:
        instance (int):                 Instance index for new instance
        overrides (dict[str, Any]):     Values to override. Fields must be period-separated string of nested fields.
                                        Fields may be in Config format or dict format (which contains "__value__" fields, etc).
                                        Ex: {"exp_buffer.capacity": 10000} OR {"__value__.exp_buffer.__value__.capacity": 10000}
    '''
    def create_new_instance(self, instance: int, overrides: dict[str, Any] = {}) -> 'Config':
        # Determine overrides format
        dict_format = any("__" in key for key in overrides.keys())
        
        # If in dict format, perform overrides on dict copy
        # If in config format, perform overrides on Config copy
        if dict_format:
            new = self.to_dict()
            def get(obj, field):
                return obj[field]
            def set(obj, field, value):
                obj[field] = value
        else:
            new = Config.from_str(self.to_str())
            def get(obj, field):
                return getattr(obj, field)
            def set(obj, field, value):
                setattr(obj, field, value)
            
        for key, val in overrides.items():
            obj = new
            fields = key.split('.')
            for field in fields[:-1]:
                obj = get(obj, field)
            set(obj, fields[-1], val)
        
        if dict_format:
            new = Config.from_dict(new)
            
        # Update instance number
        new.instance = instance
        
        return new
    


    '''
    File I/O
    '''
    
    def save(self) -> None:
        os.makedirs(Config.instance_save_folder(self.name, self.instance), exist_ok=True)
        
        with open(Config.instance_save_path(self.name, self.instance), "w") as fp:
            json.dump(self, fp, cls = ConfigEncoder, indent=2)
        
    def to_str(self, **kargs) -> str:
        return json.dumps(self, cls=ConfigEncoder, **kargs)    
    
    def to_dict(self) -> dict:
        return json.loads(self.to_str())
    
    @staticmethod
    def load(config_name: str, config_instance: int) -> 'Config':
        with open(Config.instance_save_path(config_name, config_instance), "r") as fp:
            return json.load(fp, cls = ConfigDecoder)

    @staticmethod
    def from_str(config_str: str) -> 'Config':
        return json.loads(config_str, cls=ConfigDecoder)
    
    @staticmethod
    def from_dict(config_dict: dict) -> 'Config':
        return Config.from_str(json.dumps(config_dict))
    
    

    '''
    Static Config JSON File Organization
    '''
    
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