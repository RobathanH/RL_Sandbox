from dataclasses import dataclass, field
import itertools
from multiprocessing.sharedctypes import Value
import os, json
from enum import Enum
from re import T
from typing import Optional, Type, Union, Tuple
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from config.config import Config
from function_approximator.function_approximator import FunctionApproximator
from policy.policy import Policy
from policy.function_policy import FunctionPolicy
from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
from env_handler.env_format import DiscreteActionSpace, ContinuousActionSpace
from exp_buffer.exp_buffer import ExpBuffer
from exp_buffer.td_exp_buffer import TDExpBuffer_Config
from exp_buffer.exp_format import *
from util.schedule import Schedule, Constant

from .trainer import Trainer, Trainer_Config
from .loss_functions import Huber_Loss, Loss, MSE_Loss

'''
Default: Advantage Actor-Critic (A2C)
Train both a policy network and a value network, allowing applying the
REINFORCE policy-gradient algorithm to estimated advantage for state-action
tuples.

TODO: Support for multiple versions of actor-critic, both online and offline,
using different kinds of baseline estimator, or monte-carlo return estimates (online)
instead of estimated future value

TODO: ActorCritic subtypes: (subclasses or config?)
- Discrete (Categorical) vs Continuous (Gaussian)
- Bootstrapped Advantage vs Monte-Carlo with V-estimator baseline
'''
@dataclass
class ActorCritic_Config(Trainer_Config):
    # Function Approximator
    policy_network_architecture: FunctionApproximator
    v_network_architecture: FunctionApproximator
    
    # Learning Constants
    epochs_per_step: int = 1
    episodes_per_step: int = 1
    batch_size: Optional[int] = None
    minibatch_size: int = 32
    optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD
    learning_rate: Schedule = Constant(1e-3)
    weight_decay: float = 0
    gradient_norm_clip_threshold: Optional[float] = None
    
    # Value loss type
    value_loss_function: Loss = MSE_Loss()

    # TD-Learning Constants
    soft_target_update_fraction: float = 0.01
    
    # Entropy Loss Constant
    entropy_loss_constant: float = 0
    
    def get_class(self) -> Type[Trainer]:
        return ActorCritic
    
    

# ActorCritic Subtype Enums
class ActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2

ACTOR_CRITIC_POLICY_NETWORK_SAVENAME = "policy_network.pth"
ACTOR_CRITIC_V_NETWORK_SAVENAME = "v_network.pth"
ACTOR_CRITIC_V_TARGET_NETWORK_SAVENAME = "v_target_network.pth"
ACTOR_CRITIC_STATE_SAVENAME = "trainer_state.json"
class ActorCritic(Trainer):
    def __init__(self, config: Config):
        # Config Validity Check
        if not isinstance(config.trainer, ActorCritic_Config):
            raise ValueError(f"ActorCritic trainer requires ActorCritic_Config, not {type(config.trainer)}")
        
        # Config
        self.config: Config = config
        self.trainer_config: ActorCritic_Config = config.trainer
        
        # Check config validity
        if not isinstance(config.env_handler, SinglePlayerEnvHandler_Config):
            raise ValueError(f"EnvHandler must be SinglePlayerEnvHandler")
        if not isinstance(config.exp_buffer, TDExpBuffer_Config):
            raise ValueError(f"ExpBuffer must be TDExpBuffer")
        
        # Value Loss validity
        if not (isinstance(self.trainer_config.value_loss_function, MSE_Loss) or isinstance(self.trainer_config.value_loss_function, Huber_Loss)):
            raise ValueError(f"Unsupported value loss function: {type(self.trainer_config.value_loss_function)}")
        
        # Check actionspace validity
        if isinstance(config.env_handler.action_space, DiscreteActionSpace):
            self.action_type = ActionType.DISCRETE
        elif isinstance(config.env_handler.action_space, ContinuousActionSpace):
            self.action_type = ActionType.CONTINUOUS
        else:
            raise ValueError(f"Unsupported ActionSpace type: {type(config.env_handler.action_space)}")

        
        # Confirm network architecture input/output shape matches
        p_net_arch = self.trainer_config.policy_network_architecture
        v_net_arch = self.trainer_config.v_network_architecture
        if self.action_type is ActionType.DISCRETE:
            p_net_expected_input = [config.env_handler.observation_space]
            p_net_expected_output = [[config.env_handler.action_space.count]]
            v_net_expected_input = [config.env_handler.observation_space]
            v_net_expected_output = [[1]]
        elif self.action_type is ActionType.CONTINUOUS:
            p_net_expected_input = [config.env_handler.observation_space]
            p_net_expected_output = [config.env_handler.action_space.shape, config.env_handler.action_space.shape] # mean and log-variance
            v_net_expected_input = [config.env_handler.observation_space]
            v_net_expected_output = [[1]]
        else:
            raise ValueError(f"Missing network architecture check for action_type {self.action_type.name}")
            
        if p_net_arch.input_shape() != p_net_expected_input:
            raise ValueError(f"Expected policy network input shape {p_net_expected_input}, actual input shape is {p_net_arch.input_shape()}")
        if p_net_arch.output_shape() != p_net_expected_output:
            raise ValueError(f"Expected policy network output shape {p_net_expected_output}, actual output shape is {p_net_arch.output_shape()}")
        if v_net_arch.input_shape() != v_net_expected_input:
            raise ValueError(f"Expected value network input shape {v_net_expected_input}, actual input shape is {v_net_arch.input_shape()}")
        if v_net_arch.output_shape() != v_net_expected_output:
            raise ValueError(f"Expected value network output shape {v_net_expected_output}, actual output shape is {v_net_arch.output_shape()}")
        
        # Create networks
        self.policy_network = self.trainer_config.policy_network_architecture.create()
        self.policy_network.to(DEVICE)
        self.v_network = self.trainer_config.v_network_architecture.create()
        self.v_network.to(DEVICE)

        # Create target network if using bootstrapping for training - if exp_buffer.td_step is not None (Monte-Carlo samples)
        self.v_target_network = self.trainer_config.v_network_architecture.create()
        self.v_target_network.to(DEVICE)
        self.v_target_network.load_state_dict(self.v_network.state_dict())
        
        # Create optimizer
        self.optimizer = self.trainer_config.optimizer(itertools.chain(self.v_network.parameters(), self.policy_network.parameters()), lr = self.trainer_config.learning_rate.value(), weight_decay = self.trainer_config.weight_decay)
        
        # State Variables
        self.train_step = 0
    
    
    # Environment Interaction
    
    '''
    Returns the current trained policy, for choosing actions in the env
    Returns:
        (Policy)
    '''
    def current_policy(self) -> Policy:
        
        def policyFunction(observation: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                observation = torch.from_numpy(observation).type(torch.float32).to(DEVICE)
                if self.action_type is ActionType.DISCRETE:
                    action_logits = self.policy_network(observation)
                    dist = torch.distributions.Categorical(logits = action_logits)
                    action = dist.sample().cpu().numpy()
                elif self.action_type is ActionType.CONTINUOUS:
                    mean, log_var = self.policy_network(observation)
                    dist = torch.distributions.MultivariateNormal(mean, torch.diag(torch.exp(log_var)))
                    action = dist.sample().cpu().numpy()
                    action = np.clip(action, self.config.env_handler.action_space.lower_bound, self.config.env_handler.action_space.upper_bound)
                else:
                    raise ValueError(f"Missing policy function for action type: {self.action_type.name}")
                return action
        
        return FunctionPolicy(policyFunction)
    
    '''
    Returns the current trained policy, in batch format.
    Takes batches of observations, returns batches of actions
    Returns:
        (Policy)
    '''
    def current_batch_policy(self) -> Policy:
        def policyFunction(batch_observation: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                batch_observation = torch.from_numpy(batch_observation).type(torch.float32).to(DEVICE)
                if self.action_type is ActionType.DISCRETE:
                    batch_action_logits = self.policy_network(batch_observation)
                    
                    batch_dist = torch.distributions.Categorical(logits = batch_action_logits)
                    batch_action = batch_dist.sample().cpu().numpy()
                    
                elif self.action_type is ActionType.CONTINUOUS:
                    batch_mean, batch_log_var = self.policy_network(batch_observation)
                    
                    batch_action = []
                    for i in range(len(batch_observation)):
                        dist = torch.distributions.MultivariateNormal(batch_mean[i], torch.diag(torch.exp(batch_log_var[i])))
                        action = dist.sample().cpu().numpy()
                        action = np.clip(action, self.config.env_handler.action_space.lower_bound, self.config.env_handler.action_space.upper_bound)
                        batch_action.append(action[None])
                    batch_action = np.concatenate(batch_action, axis=0)
                else:
                    raise ValueError(f"Missing policy function for action type: {self.action_type.name}")
                return batch_action
        
        return FunctionPolicy(policyFunction)
    
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
        batch_size = exp_buffer.size() if self.trainer_config.batch_size is None else self.trainer_config.batch_size
        aligned_batch_size = (batch_size // self.trainer_config.minibatch_size) * self.trainer_config.minibatch_size
        
        # Keep track of average loss over entire epoch
        total_policy_loss = 0
        total_v_loss = 0
        
        # Set learning rate for this train step
        for g in self.optimizer.param_groups:
            g['lr'] = self.trainer_config.learning_rate.value(self.train_step)
        
        for epoch in trange(self.trainer_config.epochs_per_step, leave = False):
            # Accumulate total square error over single epoch
            epoch_policy_loss = 0
            epoch_v_loss = 0
            
            # Collect batch of exp
            exp = exp_buffer.get(count = aligned_batch_size)
            
            '''
            s = torch.from_numpy(np.array([e.state for e in exp])).type(torch.float32).to(DEVICE)
            _, log_var = self.policy_network(s)
            print(f" log_var: {torch.min(log_var).item()}, {torch.max(log_var).item()}")
            '''
            
            minibatch_pbar = tqdm(total=len(exp), leave=False)
            i = 0
            while i < len(exp):
                exp_minibatch = exp[i : i + self.trainer_config.minibatch_size]
                
                # Compute loss gradients
                self.optimizer.zero_grad()
                minibatch_mean_policy_loss, minibatch_mean_v_loss = self.calc_mean_losses(exp_minibatch)
                minibatch_mean_loss = minibatch_mean_v_loss + minibatch_mean_policy_loss
                minibatch_mean_loss.backward()
                
                # Optional gradient clipping
                if self.trainer_config.gradient_norm_clip_threshold is not None:
                    nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.trainer_config.gradient_norm_clip_threshold)
                    nn.utils.clip_grad_norm_(self.v_network.parameters(), self.trainer_config.gradient_norm_clip_threshold)
                
                self.optimizer.step()
                
                # Soft update target network
                tau = self.trainer_config.soft_target_update_fraction
                for current_param, target_param in zip(self.v_network.parameters(), self.v_target_network.parameters()):
                    target_param.data.copy_(tau * current_param.data + (1 - tau) * target_param.data)
                
                # Update displayed batch MSE
                epoch_policy_loss += minibatch_mean_policy_loss.item() * self.trainer_config.minibatch_size
                epoch_v_loss += minibatch_mean_v_loss.item() * self.trainer_config.minibatch_size
                minibatch_pbar.set_postfix({
                    "Batch Policy Loss": epoch_policy_loss / (i + self.trainer_config.minibatch_size),
                    "Batch V Loss": epoch_v_loss / (i + self.trainer_config.minibatch_size)
                })
                
                # Increment
                i += len(exp_minibatch)
                minibatch_pbar.update(len(exp_minibatch))
                
            # Accumulate loss over epochs
            total_policy_loss += epoch_policy_loss
            total_v_loss += epoch_v_loss
        
        # Increment training step index
        self.train_step += 1
            
        # Return metrics
        train_policy_loss = total_policy_loss / (aligned_batch_size * self.trainer_config.epochs_per_step)
        train_v_loss = total_v_loss / (aligned_batch_size * self.trainer_config.epochs_per_step)
        return {
            "policy_loss": train_policy_loss,
            "value_loss": train_v_loss
        }
        
    '''
    Returns true if the trainer is on-policy, and the exp buffer should be
    cleared after each train step
    '''
    def on_policy(self) -> bool:
        return True
    
    
    
    # Loading and Saving

    def save(self) -> None:
        policy_network_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_POLICY_NETWORK_SAVENAME)
        v_network_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_V_NETWORK_SAVENAME)
        v_target_network_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_V_TARGET_NETWORK_SAVENAME)
        state_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_STATE_SAVENAME)
        
        # Save network weights
        torch.save(self.policy_network.state_dict(), policy_network_savepath)
        torch.save(self.v_network.state_dict(), v_network_savepath)
        torch.save(self.v_target_network.state_dict(), v_target_network_savepath)
        
        # Save current state info
        state = {
            "train_step": self.train_step
        }
        with open(state_savepath, 'w') as fp:
            json.dump(state, fp, indent=2)

    def load(self) -> None:
        policy_network_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_POLICY_NETWORK_SAVENAME)
        v_network_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_V_NETWORK_SAVENAME)
        v_target_network_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_V_TARGET_NETWORK_SAVENAME)
        state_savepath = os.path.join(Config.checkpoint_folder(), ACTOR_CRITIC_STATE_SAVENAME)
        
        if not os.path.exists(policy_network_savepath) or \
            not os.path.exists(v_network_savepath) or \
            not os.path.exists(v_target_network_savepath) or \
            not os.path.exists(state_savepath):
            return
        
        # Load network weights
        self.policy_network.load_state_dict(torch.load(policy_network_savepath))
        self.v_network.load_state_dict(torch.load(v_network_savepath))
        self.v_target_network.load_state_dict(torch.load(v_target_network_savepath))
        
        # Load current state info
        with open(state_savepath, 'r') as fp:
            state = json.load(fp)
        self.train_step = state['train_step']
        
        
        
    # Logging Setup
    
    '''
    Returns a list of the trainable pytorch modules used (for logging)
    '''
    def get_trainable_modules(self) -> list[nn.Module]:
        return [self.policy_network, self.v_network]



    # Helper Functions

    '''
    Calculates minibatch mean loss for both value function estimator and policy function
    Args:
        exp_minibatch ([ExpSars] or [ExpMC]):   Minibatch list of exp tuples
    Returns:
        (torch.FloatTensor):                    Loss for policy function (Using REINFORCE objective with baseline and entropy objective)
        (torch.FloatTensor):                    Loss for value function (TD error or MC error depending on exp type)
    '''
    def calc_mean_losses(self, exp_minibatch: Union[list[ExpSars], list[ExpMC]]) -> list[torch.FloatTensor, torch.FloatTensor]:
        if self.action_type is ActionType.DISCRETE:
            # Exp Tensors
            s = torch.from_numpy(np.array([e.state for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            a_ind = torch.from_numpy(np.array([e.action for e in exp_minibatch])).type(torch.int64)
            a = F.one_hot(a_ind, self.config.env_handler.action_space.count).type(torch.bool).to(DEVICE)
            r = torch.from_numpy(np.array([e.reward for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            done = torch.tensor([e.done() for e in exp_minibatch]).type(torch.bool).to(DEVICE)
            next_s = torch.from_numpy(np.array([e.next_state if not e.done() else np.zeros(self.config.env_handler.observation_space).astype(np.float32) for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            
            # Policy Loss
            with torch.no_grad():
                TDn_val_estimate = r + torch.bitwise_not(done) * self.v_network(next_s).squeeze() * self.config.env_handler.discount_rate**self.config.exp_buffer.td_steps
                TD0_val_estimate = self.v_network(s).squeeze()
                advantage = TDn_val_estimate - TD0_val_estimate

            logits = self.policy_network(s)
            log_prob = self.log_prob_categorical(logits, a)
            entropy = torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1)
            mean_policy_loss = torch.mean(-log_prob * advantage - self.trainer_config.entropy_loss_constant * entropy)
            
            # V Loss
            v_predicted = self.v_network(s).squeeze()
            v_target = r + torch.bitwise_not(done) * self.v_target_network(next_s).squeeze() * self.config.env_handler.discount_rate**self.config.exp_buffer.td_steps
            mean_v_loss = self.trainer_config.value_loss_function.apply(v_predicted, v_target)
            
            return mean_policy_loss, mean_v_loss
                
        if self.action_type is ActionType.CONTINUOUS:
            # Exp Tensors
            s = torch.from_numpy(np.array([e.state for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            a = torch.from_numpy(np.array([e.action for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            r = torch.from_numpy(np.array([e.reward for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            done = torch.tensor([e.done() for e in exp_minibatch]).type(torch.bool).to(DEVICE)
            next_s = torch.from_numpy(np.array([e.next_state if not e.done() else np.zeros(self.config.env_handler.observation_space).astype(np.float32) for e in exp_minibatch])).type(torch.float32).to(DEVICE)
            
            # Policy Loss
            with torch.no_grad():
                TDn_val_estimate = r + torch.bitwise_not(done) * self.v_network(next_s).squeeze() * self.config.env_handler.discount_rate**self.config.exp_buffer.td_steps
                TD0_val_estimate = self.v_network(s).squeeze()
                advantage = TDn_val_estimate - TD0_val_estimate
            mean, log_var = self.policy_network(s)
            log_prob = self.log_prob_gaussian(mean, log_var, a)
            entropy = torch.sum(torch.flatten(0.5 * (1 + np.log(2 * np.pi) + log_var), start_dim = 1), dim = -1)
            mean_policy_loss = torch.mean(-log_prob * advantage - self.trainer_config.entropy_loss_constant * entropy)
            
            # V Loss
            v_predicted = self.v_network(s).squeeze()
            v_target = r + torch.bitwise_not(done) * self.v_target_network(next_s).squeeze() * self.config.env_handler.discount_rate**self.config.exp_buffer.td_steps
            mean_v_loss = self.trainer_config.value_loss_function.apply(v_predicted, v_target)
            
            return mean_policy_loss, mean_v_loss

        raise ValueError(f"Unsupported ActionType: {self.action_type.name}")
    
    
    
    # Helper functions for batch calculating log probs of distributions
    
    '''
    Return log probabilities for each row in a minibatch of Categorical action distributions and chosen actions
    Args:
        logits (torch.FloatTensor):     Minibatch of Categorical logits (unnormalized log probs). Shape = (B, config.env_handler.action_space.count)
        actions (torch.BoolTensor):     Minibatch of one-hot action tensors. Shape = (B, config.env_handler.action_space.count)
    Returns:
        (torch.FloatTensor):            Log probability of chosen action for each row. Shape = (B,)
    '''
    def log_prob_categorical(self, logits: torch.FloatTensor, actions: torch.BoolTensor) -> torch.FloatTensor:
        return torch.sum(torch.log_softmax(logits, dim = -1) * actions, dim = -1)
    
    '''
    Return log probabilities for each row in a minibatch of Gaussian action distributions and chosen actions.
    Improve stability outside action bounds using Clipped Action Policy Gradient from https://arxiv.org/pdf/1802.07564.pdf
    Args:
        means (torch.FloatTensor):      Minibatch of Gaussian mean vectors. Shape = (B,) + config.env_handler.action_space.shape
        log_vars (torch.FloatTensor):   Minibatch of Gaussian log_var vectors. Shape = (B,) + config.env_handler.action_space.shape
        actions (torch.FloatTensor):    Minibatch of chosen action tensors. Shape = (B,) + config.env_handler.action_space.shape
    Returns:
        (torch.FloatTensor):            Log probability of chosen action for each row. Shape = (B,)
    '''
    def log_prob_gaussian(self, means: torch.FloatTensor, log_vars: torch.FloatTensor, actions: torch.FloatTensor) -> torch.FloatTensor:        
        # Determine which dimensions are out of bounds in each row
        lower_bound = torch.from_numpy(self.config.env_handler.action_space.lower_bound).reshape([1] + self.config.env_handler.action_space.shape).to(DEVICE)
        upper_bound = torch.from_numpy(self.config.env_handler.action_space.upper_bound).reshape([1] + self.config.env_handler.action_space.shape).to(DEVICE)
        oob_low = (actions <= lower_bound)
        oob_high = (actions >= upper_bound)
        in_bounds = torch.bitwise_not(torch.bitwise_or(oob_low, oob_high))
        
        # Calculate log cumulative probability for each given gaussian distribution and each dimension on both lower and upper segments
        stddev = torch.sqrt(torch.exp(log_vars))
        lower_cum_log_prob_per_dim = torch.log(0.5 * (1 + torch.erf((lower_bound - means) / (stddev * np.sqrt(2)))) + 1e-6)
        upper_cum_log_prob_per_dim = torch.log(1 - 0.5 * (1 + torch.erf((upper_bound - means) / (stddev * np.sqrt(2)))) + 1e-6)
        
        # Calculate log prob density for each chosen action under each gaussian distribution
        action_log_prob_per_dim = -0.5 * (actions - means)**2 / torch.exp(log_vars) - 0.5 * log_vars - 0.5 * np.log(2 * np.pi) # Log prob for each action dimension individually
        
        # For each row and action dimension, log prob is cumulative if action out of bounds, and normal density within bounds
        # Log prob is then summed over all action dimensions
        log_prob_per_dim = oob_low * lower_cum_log_prob_per_dim + \
                           oob_high * upper_cum_log_prob_per_dim + \
                           in_bounds * action_log_prob_per_dim
        log_prob = torch.sum(torch.flatten(log_prob_per_dim, start_dim = 1), dim = -1)
        return log_prob    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)