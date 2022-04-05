from typing import Type
import torch

from trainer.trainer import Trainer

from .q_learning import QLearning, QLearning_Config


'''
Q-Learning version, avoiding maximization bias by calculating TD target using both 
target and online network.
Uses the same config as QLearning (QLearning_Config)
'''

class DoubleQLearning_Config(QLearning_Config):
    def get_class(self) -> Type[Trainer]:
        return DoubleQLearning

class DoubleQLearning(QLearning):
    '''
    Calculates minibatch mean loss, which is the mean-squared TD error. Accumulates pytorch grads, and 
    expects pytorch grads to have been zeroed before this function.
    Instead of maximizing next-action-value using target network alone, double DQN chooses the best
    next action according to current network, and gets estimated value of that next action using target network.
    Args:
        s (torch.FloatTensor):      Minibatch of observation tensors. Shape = (B,) + config.env_handler.observation_space
        a (torch.BoolTensor):       Minibatch of one-hot action tensors. Shape = (B, config.env_handler.action_space.count)
        r (torch.FloatTensor):      Minibatch of reward floats. Shape = (B,)
        done (torch.BoolTensor):    Minibatch of boolean flags for if a tuple ended the episode. Shape = (B,)
        next_s (torch.FloatTensor): Minibatch of observation tensors. Shape = (B,) + config.env_handler.observation_space
    Returns:
        (torch.FloatTensor):        Mean loss of minibatch. Shape = (1,)
    '''
    def calc_mean_loss(self, s: torch.FloatTensor, a: torch.BoolTensor, r: torch.FloatTensor, done: torch.BoolTensor, next_s: torch.FloatTensor) -> torch.FloatTensor:
        q_predicted = torch.sum(self.q_net(s) * a, dim = -1)
        
        with torch.no_grad():
            best_next_action_ind = torch.argmax(self.q_net(next_s), dim = -1)
            best_next_action_onehot = torch.nn.functional.one_hot(best_next_action_ind, self.config.env_handler.action_space.count)
            q_target = r + self.config.env_handler.discount_rate * torch.bitwise_not(done) * torch.sum(self.target_q_net(next_s) * best_next_action_onehot, dim = -1)
        
        minibatch_mean_loss = torch.mean((q_predicted - q_target)**2)
        return minibatch_mean_loss