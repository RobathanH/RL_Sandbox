import argparse
import os, shutil
from tqdm import trange
import numpy as np
import time
import json
import wandb

from torch.utils import tensorboard

from config.config_registry import get_config
from config.config import Config
from env_handler.singleplayer_env_handler import EnvHandler
from exp_buffer.exp_buffer import ExpBuffer
from trainer.trainer import Trainer


'''
Manages and handles all other components based on a particular config and run type
'''

# Constants
WANDB_PROJECT_NAME = "RL_Sandbox"
PERIODIC_RECORD_TIME = 5 # Minutes between triggering new policy recording
PERIODIC_RECORD_EPISODE_COUNT = 3 # Number of episodes to record at each periodic record trigger
PERIODIC_SAVE_TIME = 10 # Minutes between saving training state


'''
Core training function.
Requires wandb run to be initialized.
'''
def train(config: Config, train_iterations: int, wandb_run: wandb.run) -> None:
    if wandb_run is None:
        raise ValueError(f"wandb run must be initialized")
    
    '''
    Component Setup
    '''
    
    # Major Components
    env_handler: EnvHandler = config.env_handler.get_class()(config)
    exp_buffer: ExpBuffer = config.exp_buffer.get_class()(config)
    trainer: Trainer = config.trainer.get_class()(config)
    
    # Load component states if they exist
    if not trainer.on_policy():
        exp_buffer.load()
    trainer.load()
    
    '''
    Experiment/Training Loop
    '''
    try:
        
        last_record_time = time.time()
        last_save_time = time.time()
        
        for i in trange(train_iterations):
            # Dict for log info at this step
            log_dict = {}
            
            # Experience Step
            trajectories = env_handler.run_episodes(trainer.current_policy(), config.trainer.episodes_per_step)
            
            # Log Experience Metrics
            log_dict["episode_reward"] = np.mean([t.total_reward() for t in trajectories])
            log_dict["episode_length"] = np.mean([t.length() for t in trajectories])
            
            # Add Experience to Buffer
            exp_buffer.add_trajectories(trajectories)
            
            # Train Step
            train_metrics = trainer.train(exp_buffer)
            
            # Log Train Metrics
            log_dict.update(train_metrics)
                
            # Clear exp buffer after training if on-policy
            if trainer.on_policy():
                exp_buffer.clear()
            
            # Save after each training step
            if time.time() - last_save_time >= PERIODIC_SAVE_TIME * 60:
                if not trainer.on_policy():
                    exp_buffer.save()
                trainer.save()
                last_save_time = time.time()
            
            # Record policy periodically
            if time.time() - last_record_time >= PERIODIC_RECORD_TIME * 60:
                recording_path = env_handler.record_episodes(trainer.current_policy(), PERIODIC_RECORD_EPISODE_COUNT, trainer.current_train_step())
                log_dict["recording"] = wandb.Video(recording_path, format="gif")
                last_record_time = time.time()
                
            # Upload logs for this train step
            wandb.log(log_dict)
                
    except KeyboardInterrupt:
        print("Training loop aborted. Saving final state before exit...")
    finally:
        # Save final state
        if not trainer.on_policy():
            exp_buffer.save()
        trainer.save()    
        
        # Record final policy
        recording_path = env_handler.record_episodes(trainer.current_policy(), PERIODIC_RECORD_EPISODE_COUNT, trainer.current_train_step())
        log_dict["recording"] = wandb.Video(recording_path, format="gif")
        wandb.log(log_dict)









if __name__ == '__main__':
    '''
    Command-line Arguments
    '''
    argparser = argparse.ArgumentParser(description = "Run a particular RL method and environment. By default creates a new instance of the given config name (with optional overrides), but can also load existing saves for further training.")
    
    # Specify Config Base (required)
    argparser.add_argument("name", type=str, help="Config name to look up and use")
    
    # Specify Action (required)
    argparser.add_argument("action", choices=['new', 'load'], help="Action to perform with chosen base config.\n\
        'new' creates a new instance for this config (with optional overrides), then trains.\n\
        'load' loads saves from existing instance for this config, then trains.")
    
    # Specify instance index
    argparser.add_argument("-i", "--instance", type=int, default=None, help="Instance to target. Required for 'load' action, allows overriding existing instances for 'new' action.")
    
    # Training Options
    argparser.add_argument("-t", "--train_iter", type=int, default=1, help="Number of experimentation-training loops to perform.")
    
    # Config Overrides
    argparser.add_argument(
        "-o", "--overrides", type=json.loads, default={},
        help=   "Allows overriding parts of base config, valid only for 'new' action.\
                Overrides are dictionary strings with period-separated double-quoted strings referencing the particular nested config field to be changed.\
                Ex: python main.py <base_config_name> new --overrides \"{\\\"trainer.weight_decay\\\": 1e-2}\"")
    
    args = argparser.parse_args()
    
    
    
    '''
    Config Setup
    '''
    
    # Create new instance
    if args.action == "new":
        name = args.name
        if args.instance is not None:
            instance = args.instance
        else:
            max_saved_instance = Config.max_saved_instance(name)
            if max_saved_instance is not None:
                instance = max_saved_instance + 1
            else:
                instance = 0
        
        # Load base config
        config = get_config(name)
        
        # Convert to new instance (change instance index and apply optional overrides)
        config = config.create_new_instance(instance, args.overrides)
        
        # Clear any existing files for this instance
        if os.path.exists(Config.instance_save_folder(config.name, config.instance)):
            shutil.rmtree(Config.instance_save_folder(config.name, config.instance))
        if os.path.exists(Config.instance_example_folder(config.name, config.instance)):
            shutil.rmtree(Config.instance_example_folder(config.name, config.instance))
        
        # Save config file (for future loading)
        config.save()
        
    # Load existing instance
    elif args.action == "load":
        name = args.name
        instance = args.instance
        
        # Error if instance not specified
        if instance is None:
            raise ValueError(f"'load' action requires specified instance.")
        
        # Error if instance does not exist
        if not Config.instance_save_exists(name, instance):
            raise ValueError(f"No config instance exists with name = {name}, instance = {instance}")
        
        config = Config.load(name, instance)
        
    # Unrecognized action
    else:
        raise ValueError(f"Unrecognized action {args.action}")
    
    
    '''
    Wandb Setup
    '''
    wandb.login()
    wandb_run = wandb.init(
        project=WANDB_PROJECT_NAME,
        name=f"{config.name}.i{config.instance}",
        config=config.to_dict()
    )
    
    print(json.dumps(config.to_dict(), indent=2))
    
    
    '''
    Training function
    '''
    train(config, args.train_iter, wandb_run)