import os
import wandb

from config.config import Config
from config.config_registry import get_config


'''
Entry Point Functions
Create or Load a Config Instance, and initialize a wandb run, potentially resumed
from an existing wandb run
'''

WANDB_ENTITY_NAME = "rharries"
WANDB_PROJECT_NAME = "RL_Sandbox"


'''
Initialize a wandb run by loading a given config name and instance saved on the cloud
'''
def load_run(config_name: str, config_instance: int) -> Config:
    wandb.login()
    
    # Find the run id which matches this config instance
    runs = wandb.Api().runs(
        f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}",
        filters={
            "config.name": config_name,
            "config.instance": config_instance
        }
    )
    if len(runs) == 0:
        raise ValueError(f"Could not find wandb run for config {config_name}, instance {config_instance}")
    if len(runs) > 1:
        raise ValueError(f"Found multiple wandb runs for config {config_name}, instance {config_instance}")
    run = runs[0]
    
    # Load config instance from run
    config = Config.from_dict(run.config)
    
    # Initialize wandb run, resumed from previous run
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=config.name,
        name=config.wandb_run_name(),
        config=config.to_dict(),
        id=run.id,
        resume="must"
    )
    
    # Restore all files from wandb cloud run, which requires referencing them by name
    for file in run.files():
        wandb.restore(file.name, replace=True, root=Config.checkpoint_folder())
        
    return config



'''
Initialize a wandb run as a new instance of the chosen config, with optional config value overrides
'''
def create_run(config_name: str, config_overrides: dict = {}) -> Config:
    wandb.login()
    
    # Load base config by this name
    base_config = get_config(config_name)
    
    # Determine the first unused instance index for this config name
    next_instance_index = 0
    runs = wandb.Api().runs(
        f"{WANDB_ENTITY_NAME}/{WANDB_PROJECT_NAME}",
        filters={
            "config.name": config_name
        }
    )
    if len(runs) == 0:
        instance = 0
    else:
        instance = max(run.config["instance"] for run in runs) + 1
                
    # Create new config instance
    config = base_config.create_new_instance(instance, config_overrides)
    
    # Initialize wandb run
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=config.name,
        name=config.wandb_run_name(),
        config=config.to_dict()
    )
    
    return config