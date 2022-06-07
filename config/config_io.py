import os, json
from dataclasses import is_dataclass
import numpy as np
from enum import Enum

from .config import Config
from .module_importer import GET_CLASS

'''
Methods for saving and loading Config objects
'''

def save_config(config: Config) -> None:
    os.makedirs(Config.instance_save_folder(config.name, config.instance), exist_ok=True)
    
    with open(Config.instance_save_path(config.name, config.instance), "w") as fp:
        json.dump(config, fp, cls = ConfigEncoder, indent=2)
        
def load_config(config_name: str, config_instance: int) -> Config:
    with open(Config.instance_save_path(config_name, config_instance), "r") as fp:
        return json.load(fp, cls = ConfigDecoder)

'''
JSON Encoder/Decoder for Config instances
'''
class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        # Encode Class Types by their names alone
        if isinstance(obj, type):
            return dict(__type__ = "class", __class__ = obj.__name__)
        
        # Encode enums by their value name
        if isinstance(obj, Enum):
            return dict(__type__ = "enum", __class__ = type(obj).__name__, __value__ = obj.name)
        
        # Encode numpy arrays as nested lists
        if isinstance(obj, np.ndarray):
            return dict(__type__ = "np_array", __value__ = obj.tolist())
        
        # Encode dataclasses by their dictionary formats
        if is_dataclass(obj):
            return dict(__type__ = "dataclass", __class__ = type(obj).__name__, __value__ = obj.__dict__)
        
        return json.JSONEncoder.default(self, obj)
    
class ConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(ConfigDecoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)
        
    def object_hook(self, dct):
        # If not a special case, use default object_hook
        if "__type__" not in dct:
            return dct
        
        # Class instance
        if dct["__type__"] == "class":
            return GET_CLASS(dct["__class__"])
            #return globals()[dct["__class__"]]
        
        # Enum
        if dct["__type__"] == "enum":
            class_ref = GET_CLASS(dct["__class__"])
            #class_ref = globals()[dct["__class__"]]
            return getattr(class_ref, dct["__value__"])
        
        # Np Array
        if dct["__type__"] == "np_array":
            return np.array(dct["__value__"])
        
        # Dataclass
        if dct["__type__"] == "dataclass":
            class_ref = GET_CLASS(dct["__class__"])
            #class_ref = globals()[dct["__class__"]]
            return class_ref(**dct["__value__"])
        
        raise ValueError(f"Unrecognized JSON-encoded config __type__: {dct}")