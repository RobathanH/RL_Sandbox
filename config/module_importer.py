import importlib
import json
import os
import sys
import inspect

'''
Store package location for all named classes, allowing them to be dynamically imported from their name alone
'''

# Store registry as editable json file
MODULE_REGISTRY_FILENAME = "module_registry.json"
with open(os.path.join(os.path.dirname(__file__), MODULE_REGISTRY_FILENAME), "r") as fp:
    MODULE_REGISTRY = json.load(fp)
    
'''
Appends the given class and package as an entry in the module registry,
if it doesn't already exist.
Allows defining and registering new classes in the same place.
'''
def REGISTER_CLASS(cls: type, override=False):
    if cls.__name__ in MODULE_REGISTRY:
        if MODULE_REGISTRY[cls.__name__] == cls.__module__:
            return
        
        if not override:
            raise ValueError(f"MODULE REGISTRY entry '{cls.__name__}' already references package '{MODULE_REGISTRY[cls.__name__]}', not {cls.__module__}.")
        
    print(f"Registering '{cls.__name__}' to module '{cls.__module__}'")
    MODULE_REGISTRY[cls.__name__] = cls.__module__
    with open(os.path.join(os.path.dirname(__file__), MODULE_REGISTRY_FILENAME), "w") as fp:
        json.dump(MODULE_REGISTRY, fp, indent=2)
            
    
'''
Register all classes defined in the current file.
Should be called using the __name__ builtin variable: 
"REGISTER_MODULE(__name__)"
'''
def REGISTER_MODULE(module_name: str):
    for cls_name, cls in inspect.getmembers(sys.modules[module_name], inspect.isclass):
        REGISTER_CLASS(cls)
    

'''
Dynamically import a class reference if it has an entry stored in the registry
'''
def GET_CLASS(cls_name: str):
    if cls_name not in MODULE_REGISTRY:
        raise ValueError(f"Class '{cls_name}' has no entry in MODULE REGISTRY.")
    
    module = importlib.import_module(MODULE_REGISTRY[cls_name])
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise ValueError(f"MODULE REGISTRY registered package '{MODULE_REGISTRY[cls_name]}' has no member named '{cls_name}'")
    
    return cls