from torch.optim import *

from .config import *
from .module_importer import REGISTER_MODULE

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
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLander-v2",
                action_space = DiscreteActionSpace(
                    count = 4
                ),
                observation_space = [8]
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
                optimizer = Adam,
                learning_rate = LogarithmicSchedule(1e-3, 1e-5, 2000),
                weight_decay = 1e-4
            )
        )
        
    if name == "lander.a2c.1":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import DiscreteActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.actor_critic import ActorCritic_Config
        from function_approximator.basic_networks import MLP
        from util.schedule import LogarithmicSchedule
        from trainer.loss_functions import Huber_Loss
        from trainer.loss_functions import MSE_Loss
        
        return Config(
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLander-v2",
                action_space = DiscreteActionSpace(
                    count = 4
                ),
                observation_space = [8],
                time_limit_counts_as_terminal_state = True
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 10
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
                optimizer = RMSprop,
                learning_rate = LogarithmicSchedule(1e-3, 1e-5, 2000),
                weight_decay = 1e-5,
                value_loss_function = MSE_Loss(),
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-3,
                gradient_norm_clip_threshold = 1
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
        from util.schedule import LogarithmicSchedule
        
        return Config(
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLanderContinuous-v2",
                action_space = ContinuousActionSpace(
                    shape = [2],
                    lower_bound = np.array([-1, -1]),
                    upper_bound = np.array([1, 1])
                ),
                observation_space = [8],
                time_limit_counts_as_terminal_state = True
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 10
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
                            bounded_output = [-5, 5]
                        ),
                        MLP(
                            layer_sizes = [32, 2],
                            bounded_output = [-5, 5]
                        )
                    )
                ),
                v_network_architecture = MLP(
                    layer_sizes = [8, 32, 64, 32, 1],
                    activation = Activation.TANH
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                optimizer = RMSprop,
                learning_rate = LogarithmicSchedule(1e-4, 1e-5, 3000),
                weight_decay = 1e-5,
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-5
            )
        )

    if name == "lander_cont.a2c.2":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import ContinuousActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.actor_critic import ActorCritic_Config
        from function_approximator.basic_networks import MLP, MultiheadModule
        from function_approximator.activation import Activation
        from util.schedule import Constant
        
        return Config(
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "LunarLanderContinuous-v2",
                action_space = ContinuousActionSpace(
                    shape = [2],
                    lower_bound = np.array([-1, -1]),
                    upper_bound = np.array([1, 1])
                ),
                observation_space = [8],
                time_limit_counts_as_terminal_state = False
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 8
            ),
            trainer = ActorCritic_Config(
                policy_network_architecture = MultiheadModule(
                    shared_module = MLP(
                        layer_sizes = [8, 64, 256, 64],
                        activation = Activation.TANH,
                        final_layer_activation = True
                    ),
                    head_modules = (
                        MLP(
                            layer_sizes = [64, 2],
                            bounded_output = [-5, 5]
                        ),
                        MLP(
                            layer_sizes = [64, 2],
                            bounded_output = [-5, 5]
                        )
                    )
                ),
                v_network_architecture = MLP(
                    layer_sizes = [8, 64, 256, 64, 1],
                    activation = Activation.TANH
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                optimizer = RMSprop,
                learning_rate = Constant(1e-5),
                weight_decay = 1e-5,
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-5
            )
        )
        
    '''
    Box2D Walker
    '''
    if name == "walker_cont.a2c.1":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import ContinuousActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.actor_critic import ActorCritic_Config
        from function_approximator.basic_networks import MLP, MultiheadModule
        from function_approximator.activation import Activation
        from util.schedule import Constant
        
        return Config(
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "BipedalWalker-v3",
                action_space = ContinuousActionSpace(
                    shape = [4],
                    lower_bound = np.array([-1, -1, -1, -1]),
                    upper_bound = np.array([1, 1, 1, 1])
                ),
                observation_space = [24],
                time_limit_counts_as_terminal_state = False
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 8
            ),
            trainer = ActorCritic_Config(
                policy_network_architecture = MultiheadModule(
                    shared_module = MLP(
                        layer_sizes = [24, 48, 64, 128, 32],
                        activation = Activation.TANH,
                        final_layer_activation = True
                    ),
                    head_modules = (
                        MLP(
                            layer_sizes = [32, 4],
                            bounded_output = [-1, 1]
                        ),
                        MLP(
                            layer_sizes = [32, 4],
                            bounded_output = [-1, 1]
                        )
                    )
                ),
                v_network_architecture = MLP(
                    layer_sizes = [24, 48, 64, 128, 32, 1],
                    activation = Activation.TANH
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                optimizer = RMSprop,
                learning_rate = Constant(1e-4),
                weight_decay = 1e-5,
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-5
            )
        )

    if name == "walker_cont.a2c.2":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import ContinuousActionSpace
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.actor_critic import ActorCritic_Config
        from function_approximator.basic_networks import MLP, MultiheadModule
        from function_approximator.activation import Activation
        from util.schedule import Constant
        
        return Config(
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "BipedalWalker-v3",
                action_space = ContinuousActionSpace(
                    shape = [4],
                    lower_bound = np.array([-1, -1, -1, -1]),
                    upper_bound = np.array([1, 1, 1, 1])
                ),
                observation_space = [24],
                time_limit_counts_as_terminal_state = False
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 10000,
                td_steps = 8
            ),
            trainer = ActorCritic_Config(
                policy_network_architecture = MultiheadModule(
                    shared_module = MLP(
                        layer_sizes = [24, 128, 512, 128, 32],
                        activation = Activation.TANH,
                        final_layer_activation = True
                    ),
                    head_modules = (
                        MLP(
                            layer_sizes = [32, 4],
                            bounded_output = [-1, 1]
                        ),
                        MLP(
                            layer_sizes = [32, 4],
                            bounded_output = [-1, 1]
                        )
                    )
                ),
                v_network_architecture = MLP(
                    layer_sizes = [24, 128, 512, 128, 32, 1],
                    activation = Activation.TANH
                ),
                epochs_per_step = 1,
                episodes_per_step = 1,
                optimizer = RMSprop,
                learning_rate = Constant(1e-4),
                weight_decay = 1e-5,
                soft_target_update_fraction = 1e-2,
                entropy_loss_constant = 1e-5
            )
        )
        
    '''
    Atari Space Invaders
    '''
    if name == "space_invaders_ram.ddqn.1":
        from env_handler.singleplayer_env_handler import SinglePlayerEnvHandler_Config
        from env_handler.env_format import DiscreteActionSpace
        from env_transform.observation_byte_unpacker import ByteUnpacker
        from exp_buffer.td_exp_buffer import TDExpBuffer_Config
        from trainer.double_q_learning import DoubleQLearning_Config
        from function_approximator.basic_networks import MLP
        from util.schedule import LogarithmicSchedule
        
        return Config(
            name = name,
            env_handler = SinglePlayerEnvHandler_Config(
                name = "SpaceInvaders-ram-v0",
                action_space = DiscreteActionSpace(
                    count = 6
                ),
                observation_space = [1024],
                discount_rate = 0.99,
                observation_transform = ByteUnpacker(),
                time_limit_counts_as_terminal_state = True
            ),
            exp_buffer = TDExpBuffer_Config(
                capacity = 50000,
                td_steps = 1
            ),
            trainer = DoubleQLearning_Config(
                q_network_architecture = MLP(
                    layer_sizes = [1024, 1024, 512, 256, 64, 32, 6]
                ),
                epsilon_schedule = LogarithmicSchedule(
                    start = 0.5,
                    end = 0.01,
                    duration = 2000
                ),
                epochs_per_step = 2,
                episodes_per_step = 5,
                minibatch_size = 64,
                optimizer = Adam,
                learning_rate = LogarithmicSchedule(1e-4, 1e-6, 2000),
                weight_decay = 1e-4
            )
        )
        
    raise ValueError(f"Config {name} not found")
    
    
    
# Register for imports
# Importantly this will register third-party classes used in configs if they are imported in global scope,
# like torch optimizers
REGISTER_MODULE(__name__)