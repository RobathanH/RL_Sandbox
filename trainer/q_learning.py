from dataclasses import dataclass
import os
from typing import List, Optional, Type, Any
import numpy as np
import json
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils import tensorboard

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from config.config import Config, DiscreteActionSpace
from exp_buffer.exp_buffer import ExpBuffer
from exp_buffer.sars_exp_buffer import SarsExpBuffer
from function_approximator.function_approximator import FunctionApproximator
from policy.function_policy import Policy, EpsilonGreedyFunctionPolicy
from util.schedule import Schedule

from .trainer import Trainer, Trainer_Config


'''
Basic Q-Learning Policy Trainer
(No improvements)
'''



'''
Trainer-specific Config
'''
@dataclass
class QLearning_Config(Trainer_Config):
    # Epsilon Schedule
    epsilon_schedule: Schedule
        
    # Learning Constants
    epochs_per_step: int
    batch_size: Optional[int]
    minibatch_size: int
    optimizer_class: Type[torch.optim.Optimizer]
    optimizer_args: dict
    
    # Q-Learning Constants
    soft_target_update_fraction: float
    
    # Function Approximator
    q_network_architecture: FunctionApproximator


Q_LEARNING_NETWORK_SAVENAME = "trainer_network.pth"
Q_LEARNING_TARGET_NETWORK_SAVENAME = "trainer_target_network.pth"
Q_LEARNING_STATE_SAVENAME = "trainer_state.json"
class QLearning(Trainer):
    def __init__(self, config: Config):
        # Confirm correct trainer config
        if not isinstance(config.trainer_config, QLearning_Config):
            raise ValueError(f"QLearning requires trainer_config of type QLearning_Config, not {type(config.trainer_config)}")
        
        # Config
        self.config: Config = config
        self.trainer_config: QLearning_Config = config.trainer_config

        # Check env validity
        if not isinstance(config.env_action_space, DiscreteActionSpace):
            raise ValueError(f"QLearning Trainer class requires DiscreteActionSpace, not {type(config.env_action_space)}")
        if len(config.env_action_space.count) != 1:
            raise ValueError(f"QLearning can currently only handle single-dimensional discrete action spaces.")
        if not issubclass(config.exp_buffer_class, SarsExpBuffer):
            raise ValueError(f"QLearning requires SARS exp buffer, not {config.exp_buffer_class}")

        # Confirm network shapes match env input/output shapes
        if not (len(config.trainer_config.q_network_architecture.input_shape()) == 1 and config.trainer_config.q_network_architecture.input_shape()[0] == config.env_observation_shape):
            raise ValueError(f"Q Network Architecture input must match observation shape {config.env_observation_shape}")
        if not (len(config.trainer_config.q_network_architecture.output_shape()) == 1 and config.trainer_config.q_network_architecture.output_shape()[0] == config.env_action_space.count):
            raise ValueError(f"Q Network Architecture output must match space of possible actions")

        # Create q network (and target network)
        self.q_net = config.trainer_config.q_network_architecture.create()
        self.q_net.to(DEVICE)
        self.target_q_net = config.trainer_config.q_network_architecture.create()
        self.target_q_net.to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Create optimizer
        self.optimizer = config.trainer_config.optimizer_class(self.q_net.parameters(), **config.trainer_config.optimizer_args)

        # State Variables
        self.train_step = 0
        
    '''
    Returns true if the trainer is on-policy, and the exp buffer should be
    cleared after each train step
    '''
    def on_policy(self) -> bool:
        return False



    # Environment Interaction

    '''
    Returns the current trained policy, for choosing actions in the env
    Returns:
        (Policy)
    '''
    def current_policy(self) -> Policy:

        # Define policy function which takes an observation and returns an integer action index
        def policyFunction(observation) -> int:
            observation = torch.from_numpy(observation).type(torch.float32).to(DEVICE)
            q = self.q_net(observation)
            return int(torch.argmax(q))

        return EpsilonGreedyFunctionPolicy(
            policyFunction,
            self.config.trainer_config.epsilon_schedule.value(self.train_step),
            lambda: np.random.choice(self.config.env_action_space.count[0])
        )

    '''
    Returns the current training step (Number of training loops completed).
    Returns:
        (int)
    '''
    def current_train_step(self) -> int:
        return self.train_step
    

    # Improvement

    '''
    Performs a training session using a given set of experience tuples from
    the buffer. Assumes experience tuples are of type ExpSars.
    Args:
        exp:    List of experience tuples from the experience buffer.
    Returns:
        (Dict):     Dictionary of metric names and values over this train step
    '''
    def train(self, exp_buffer: ExpBuffer) -> dict:
        # Batch ignores final incomplete minibatch
        batch_size = exp_buffer.size() if self.config.trainer_config.batch_size is None else self.config.trainer_config.batch_size
        aligned_batch_size = (batch_size // self.config.trainer_config.minibatch_size) * self.config.trainer_config.minibatch_size
        
        # Keep track of average loss over entire epoch 
        total_loss = 0
        
        for epoch in trange(self.config.trainer_config.epochs_per_step, leave = False):
            epoch_loss = 0 # Accumulate total square error over single epoch
            
            # Collect batch of exp
            exp = exp_buffer.get(count = aligned_batch_size)
            
            minibatch_iterator = trange(0, len(exp), self.config.trainer_config.minibatch_size, leave = False)
            for i in minibatch_iterator:
                exp_minibatch = exp[i : i + self.config.trainer_config.minibatch_size]

                s = torch.from_numpy(np.array([e.state for e in exp_minibatch])).type(torch.float32).to(DEVICE)
                a_ind = torch.from_numpy(np.array([e.action for e in exp_minibatch])).type(torch.int64)
                a = nn.functional.one_hot(a_ind, self.config.env_action_space.count[0]).type(torch.bool).to(DEVICE)
                r = torch.from_numpy(np.array([e.reward for e in exp_minibatch])).type(torch.float32).to(DEVICE)
                done = torch.tensor([e.done() for e in exp_minibatch]).type(torch.bool).to(DEVICE)
                next_s = torch.from_numpy(np.array([e.next_state if not e.done() else np.zeros(self.config.env_observation_shape).astype(np.float32) for e in exp_minibatch])).type(torch.float32).to(DEVICE)

                self.optimizer.zero_grad()
                minibatch_mean_loss = self.calc_mean_loss(s, a, r, done, next_s)
                minibatch_mean_loss.backward()
                self.optimizer.step()
                
                # Soft update target network
                tau = self.config.trainer_config.soft_target_update_fraction
                for current_param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                    target_param.data.copy_(tau * current_param.data + (1 - tau) * target_param.data)
                
                # Update displayed batch MSE
                epoch_loss += minibatch_mean_loss.item() * self.config.trainer_config.minibatch_size
                minibatch_iterator.set_postfix({"Batch MSE": epoch_loss / (i + self.config.trainer_config.minibatch_size)})
                
            # Accumulate loss over epochs
            total_loss += epoch_loss
        
        # Increment training step index
        self.train_step += 1
            
        # Return metrics
        train_mse = total_loss / (aligned_batch_size * self.config.trainer_config.epochs_per_step)
        return {"train_mse": train_mse}

    

    # Loading and Saving

    def save(self) -> None:
        # Create dirs if needed
        os.makedirs(self.config.instance_savefolder(), exist_ok = True)
        
        network_savepath = os.path.join(self.config.instance_savefolder(), Q_LEARNING_NETWORK_SAVENAME)
        target_network_savepath = os.path.join(self.config.instance_savefolder(), Q_LEARNING_TARGET_NETWORK_SAVENAME)
        state_savepath = os.path.join(self.config.instance_savefolder(), Q_LEARNING_STATE_SAVENAME)
        
        # Save network weights
        torch.save(self.q_net.state_dict(), network_savepath)
        torch.save(self.target_q_net.state_dict(), target_network_savepath)
        
        # Save current state info
        state = {
            "train_step": self.train_step
        }
        with open(state_savepath, 'w') as fp:
            json.dump(state, fp, indent=2)

    def load(self, folder: Optional[str] = None) -> None:
        if folder is None:
            folder = self.config.instance_savefolder()
        network_savepath = os.path.join(folder, Q_LEARNING_NETWORK_SAVENAME)
        target_network_savepath = os.path.join(folder, Q_LEARNING_TARGET_NETWORK_SAVENAME)
        state_savepath = os.path.join(folder, Q_LEARNING_STATE_SAVENAME)
        
        if not os.path.exists(network_savepath) or not os.path.exists(target_network_savepath) or not os.path.exists(state_savepath):
            return
        
        # Load network weights
        self.q_net.load_state_dict(torch.load(network_savepath))
        self.target_q_net.load_state_dict(torch.load(target_network_savepath))
        
        # Load current state info
        with open(state_savepath, 'r') as fp:
            state = json.load(fp)
        self.train_step = state['train_step']
        
        
        
    # HELPER FUNCTIONS - Helpful to override in subclasses
    
    '''
    Calculates minibatch mean loss, which is the mean-squared TD error. Accumulates pytorch grads, and 
    expects pytorch grads to have been zeroed before this function.
    Args:
        s (torch.FloatTensor):      Minibatch of observation tensors. Shape = (B,) + config.env_observation_shape
        a (torch.BoolTensor):       Minibatch of one-hot action tensors. Shape = (B,) + config.env_action_shape
        r (torch.FloatTensor):      Minibatch of reward floats. Shape = (B,)
        done (torch.BoolTensor):    Minibatch of boolean flags for if a tuple ended the episode. Shape = (B,)
        next_s (torch.FloatTensor): Minibatch of observation tensors. Shape = (B,) + config.env_observation_shape
    Returns:
        (torch.FloatTensor):        Mean loss of minibatch. Shape = (1,)
    '''
    def calc_mean_loss(self, s: torch.FloatTensor, a: torch.BoolTensor, r: torch.FloatTensor, done: torch.BoolTensor, next_s: torch.FloatTensor) -> torch.FloatTensor:
        q_predicted = torch.sum(self.q_net(s) * a, dim = -1)
        q_target = r + self.config.env_discount_rate * torch.bitwise_not(done) * torch.max(self.target_q_net(next_s), dim = -1).values
        minibatch_mean_loss = torch.mean((q_predicted - q_target)**2)
        return minibatch_mean_loss
    