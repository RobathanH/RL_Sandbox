from re import M
import torch
import numpy as np

from .config import *

'''
Registry to store a dictionary of all configurations
'''

def get_config(name: str) -> Config:
    '''
    Lunar Lander
    '''
    if name == "lander.ddqn.1":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import DiscreteActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.double_q_learning import DoubleQLearning_Config
        from function_approximator.basic_networks import MLP
        from util.schedule import LogarithmicSchedule
        
        return Config(
            name = "lander.ddqn.1",
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLander-v2",
                action_space = DiscreteActionSpace(
                    count = 4
                ),
                observation_space = (8,)
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 50000,
                td_steps = 1
            ),
            trainer = DoubleQLearning_Config(
                q_network_architecture = MLP(
                    layer_sizes = [8, 32, 64, 32, 4]
                ),
                epsilon_schedule = LogarithmicSchedule(
                    start = 0.5,
                    end = 0.01,
                    duration = 1000
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                optimizer = torch.optim.Adam,
                learning_rate = 2e-4,
                weight_decay = 1e-4
            )
        )
        
    if name == "lander.a2c.1":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import DiscreteActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.actor_critic import ActorCritic_Config
        from function_approximator.basic_networks import MLP
        
        return Config(
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLander-v2",
                action_space = DiscreteActionSpace(
                    count = 4
                ),
                observation_space = (8,)
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 1
            ),
            trainer = ActorCritic_Config(
                policy_network_architecture = MLP(
                    layer_sizes = [8, 32, 64, 32, 4]
                ),
                v_network_architecture = MLP(
                    layer_sizes = [8, 32, 64, 32, 1]
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                optimizer = torch.optim.RMSprop,
                learning_rate = 1e-4,
                weight_decay = 1e-5,
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-5
            )
        )

    '''
    Continuous Lunar Lander
    '''
    
    if name == "lander_cont.a2c.1":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import ContinuousActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.actor_critic import ActorCritic_Config
        from function_approximator.basic_networks import MLP, MultiheadModule
        from function_approximator.activation import Activation
        
        return Config(
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLander-v2",
                action_space = ContinuousActionSpace(
                    shape = (2,),
                    lower_bound = np.array([-1, -1]),
                    upper_bound = np.array([1, 1])
                ),
                observation_space = (8,)
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 1
            ),
            trainer = ActorCritic_Config(
                policy_network_architecture = MultiheadModule(
                    shared_module = MLP(
                        layer_sizes = [8, 32, 64, 32],
                        activation = Activation.TANH,
                        final_layer_activation = True
                    ),
                    head_modules = (
                        MLP(
                            layer_sizes = [32, 2],
                            activation = Activation.TANH
                        ),
                        MLP(
                            layer_sizes = [32, 2],
                            activation = Activation.TANH,
                            bounded_output = (-5, 5)
                        )
                    )
                ),
                v_network_architecture = MLP(
                    layer_sizes = [8, 32, 64, 32, 1],
                    activation = Activation.TANH
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                optimizer = torch.optim.RMSprop,
                learning_rate = 1e-3,
                weight_decay = 1e-5,
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-5
            )
        )
        
    raise ValueError(f"Config {name} not found")