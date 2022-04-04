import argparse
import os, shutil
from tqdm import trange
import numpy as np
import time

from torch.utils import tensorboard

from config.config_registry import CONFIG_REGISTRY
from config.config import Config
from env_handler.env_handler import EnvHandler
from exp_buffer.exp_buffer import ExpBuffer
from trainer.trainer import Trainer


'''
Manages and handles all other components based on a particular config and run type
'''

# Constants
LOG_DIR_NAME = "logs"
PERIODIC_RECORD_TIME = 5 # Minutes between triggering new policy recording
PERIODIC_SAVE_TIME = 10 # Minutes between saving training state


if __name__ == '__main__':
    '''
    Command-line Arguments
    '''
    argparser = argparse.ArgumentParser(description = "Run a particular RL method and environment")
    argparser.add_argument("name", type=str, help="Config name to look up and use")
    
    # Instance targeting - new, load are mutually exclusive
    argparser.add_argument("-n", "--new", action="store_true", help="Train new instance from scratch. Otherwise loads from specified or highest-index saved instance.")
    argparser.add_argument("-l", "--load", type=int, default=None, help="Load from particular instance index. By default load from highest-index saved instance.")
    
    # Config folder editing
    argparser.add_argument("--replace", action="store_true", help="Replace any existing configs by this name, but attempt to load any corresponding save data.")
    argparser.add_argument("--restart", action="store_true", help="Remove all existing save data for the chosen config and recreate from scratch.")
 
    # Training Options
    argparser.add_argument("-t", "--train_iter", type=int, default=1, help="Number of experimentation-training loops to perform.")
    
    args = argparser.parse_args()
    
    
    
    '''
    Config Setup
    '''

    # Config
    if args.name not in CONFIG_REGISTRY:
        raise ValueError(f"{args.name} not found in config registry.")
    config: Config = CONFIG_REGISTRY[args.name]
    
    # Optional config replacement
    if args.replace and config.config_save_exists():
        config.save_consistency_file()
    
    # Optional config reset
    if args.restart and config.config_save_exists():
        shutil.rmtree(config.config_savefolder())
    
    # If no save for this config exists, create new config folder and consistency file
    if not config.config_save_exists():
        config.save_consistency_file()
    # If config saves exist, confirm config consistency
    else:
        if not config.check_consistency():
            raise ValueError(f"Registered config does not match saved config for name: {args.name}.")
    
    # Determine Instance Index
    if args.new:
        max_saved_instance = config.max_saved_instance()
        if max_saved_instance is not None:
            config.instance_index = max_saved_instance + 1
    elif args.load is not None:
        config.instance_index = args.load
        if not config.instance_save_exists():
            raise ValueError(f"Specified instance {args.load} for config {args.name} does not exist and cannot be loaded.")
    else:
        max_saved_instance = config.max_saved_instance()
        if max_saved_instance is not None:
            config.instance_index = max_saved_instance
    
    
    
    '''
    Component Setup
    '''
    
    # Major Components
    env_handler: EnvHandler = EnvHandler(config)
    exp_buffer: ExpBuffer = config.exp_buffer_class(config)
    trainer: Trainer = config.trainer_class(config)
    
    # Load component states if instance previously saved
    if config.instance_save_exists():
        if not trainer.on_policy():
            exp_buffer.load()
        trainer.load()
    
    # Log Writer
    log_dir_path = os.path.join(config.instance_savefolder(), LOG_DIR_NAME)
    log_writer = tensorboard.SummaryWriter(log_dir = log_dir_path)
    
    
    
    '''
    Experiment/Training Loop
    '''
    try:
        
        last_record_time = time.time()
        last_save_time = time.time()
        
        for i in trange(args.train_iter):
            # Experience Step
            trajectories = env_handler.run_episodes(trainer.current_policy(), config.env_episodes_per_step)
            
            # Log Experience Metrics
            log_writer.add_scalar("total_reward", np.mean([t.total_reward() for t in trajectories]), trainer.current_train_step())
            log_writer.add_scalar("average_episode_length", np.mean([t.length() for t in trajectories]), trainer.current_train_step())
            
            # Add Experience to Buffer
            exp_buffer.add_trajectories(trajectories)
            
            # Train Step
            train_metrics = trainer.train(exp_buffer)
            
            # Log Train Metrics
            for key, val in train_metrics.items():
                log_writer.add_scalar(key, val, trainer.current_train_step())
                
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
                env_handler.record_episodes(trainer.current_policy(), 3, trainer.current_train_step())
                last_record_time = time.time()
                
    except KeyboardInterrupt:
        print("Training loop aborted. Saving final state before exit...")
    finally:
        # Save final state
        if not trainer.on_policy():
            exp_buffer.save()
        trainer.save()    
        
        # Record final policy
        env_handler.record_episodes(trainer.current_policy(), 3, trainer.current_train_step())