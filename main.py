import argparse
import os, shutil
from tqdm import trange
import numpy as np
import time
import json
import wandb

from config.config import Config
from env_handler.singleplayer_env_handler import EnvHandler
from exp_buffer.exp_buffer import ExpBuffer
from trainer.trainer import Trainer
from run_manager import load_run, create_run

'''
Manages and handles all other components based on a particular config and run type
'''

# Constants
PERIODIC_RECORD_TIME = 5 # Minutes between triggering new policy recording
PERIODIC_RECORD_EPISODE_COUNT = 3 # Number of episodes to record at each periodic record trigger
PERIODIC_SAVE_TIME = 10 # Minutes between saving training state


'''
Core training function.
Requires wandb run to be initialized.
'''
def train(config: Config, train_iterations: int, record: bool) -> None:
    if wandb.run is None:
        raise ValueError(f"wandb run must be initialized")
    
    '''
    Component Setup
    '''
    
    # Major Components
    env_handler: EnvHandler = config.env_handler.get_class()(config)
    exp_buffer: ExpBuffer = config.exp_buffer.get_class()(config)
    trainer: Trainer = config.trainer.get_class()(config)
    
    # Load component states from wandb-synced checkpoints if they exist
    if not trainer.on_policy():
        exp_buffer.load()
    trainer.load()
    
    # Setup wandb pytorch module watching
    wandb.watch(trainer.get_trainable_modules(), log="all", log_freq=1)
    
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
            trajectories = env_handler.run_parallel_episodes(trainer.current_batch_policy(), config.trainer.episodes_per_step)
            
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
            if record and time.time() - last_record_time >= PERIODIC_RECORD_TIME * 60:
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
        if record:
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
    
    # Load Existing Instance
    argparser.add_argument("-l", "--load", type=int, default=None, help="Load an existing instance from the cloud for further training.")
    
    # Training Options
    argparser.add_argument("-t", "--train_iter", type=int, default=1, help="Number of experimentation-training loops to perform.")

    # Disable Periodic Policy Recording
    argparser.add_argument("--disable_recording", action="store_true", help="Disable periodic policy episode recording")
    
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
    if args.load is None:
        # Initialize wandb run and config
        config = create_run(args.name, args.overrides)
        
    # Load existing instance
    else: 
        # Initialize wandb run and config
        config = load_run(args.name, args.load)
    
    # Print config json
    print(config.to_str(indent=2))
    
    
    '''
    Training function
    '''
    train(config, args.train_iter, not args.disable_recording)