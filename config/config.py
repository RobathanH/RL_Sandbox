from dataclasses import dataclass, is_dataclass
from enum import Enum
import os
import json
from typing import Any, Optional
import numpy as np
import wandb

from env_handler.env_handler import EnvHandler_Config
from exp_buffer.exp_buffer import ExpBuffer_Config
from trainer.trainer import Trainer_Config

from .config_io import ConfigDecoder, ConfigEncoder

'''
Config dataclass that specifies a particular instance of an RL method on a particular environment.
Contains multiple config dataclasses for specific components
'''

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
    Wandb run name for this config instance, combining both config name and instance index.
    '''
    def wandb_run_name(self) -> str:
        return f"{self.name}.{self.instance}"


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
    Directory for storing checkpoint files which will be synced with wandb for each run
    '''
    @staticmethod
    def checkpoint_folder() -> str:
        return wandb.run.dir



    '''
    Format Transforms
    '''
        
    def to_str(self, **kargs) -> str:
        return json.dumps(self, cls=ConfigEncoder, **kargs)    
    
    def to_dict(self) -> dict:
        return json.loads(self.to_str())

    @staticmethod
    def from_str(config_str: str) -> 'Config':
        return json.loads(config_str, cls=ConfigDecoder)
    
    @staticmethod
    def from_dict(config_dict: dict) -> 'Config':
        return Config.from_str(json.dumps(config_dict))
    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)