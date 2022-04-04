import torch
import numpy as np

from env_transform.action_discretizer import ActionDiscretizer
from env_transform.observation_byte_unpacker import ByteUnpacker
from function_approximator.activation import Activation
from util.schedule import LogarithmicSchedule
from trainer.double_q_learning import DoubleQLearning
from trainer.q_learning import QLearning, QLearning_Config
from trainer.actor_critic import ActorCritic, ActorCritic_Config
from function_approximator.basic_networks import MLP, Linear, MultiheadModule
from exp_buffer.sars_exp_buffer import SarsExpBuffer
from exp_buffer.mc_exp_buffer import MCExpBuffer
from exp_buffer.td_exp_buffer import TDExpBuffer

from .config import *

'''
Registry to store a dictionary of all configurations

TODO: Make into function which can import relevant parts dynamically, rather than importing everything at once.
'''
CONFIG_REGISTRY = {}


# Simplest possible linear Q-Learner for Cartpole environment

c = Config(
    name = "cartpole.dqn.linear.1",
    env_name = "CartPole-v1",
    env_action_space = DiscreteActionSpace(
        count = (2,)
    ),
    env_observation_shape = (4,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 100,
    env_discount_rate = 1,
    trainer_class = QLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.Adam,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = Linear(
            input_len = 4,
            output_len = 2
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

'''
Pendulum Swing-Up and Hold
'''

c = Config(
    name = "pendulum.act2.dqn.linear.1",
    env_name = "Pendulum-v0",
    env_action_space = DiscreteActionSpace(
        count = (2,)
    ),
    env_observation_shape = (3,),
    env_action_transform = ActionDiscretizer(
        action_shape = (1,),
        disc_counts = np.array([2]),
        cont_lower_bound = np.array([-2]),
        cont_upper_bound = np.array([2])
    ),
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 20,
    env_discount_rate = 0.9,
    trainer_class = QLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.Adam,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = Linear(
            input_len = 3,
            output_len = 2
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "pendulum.act5.dqn.linear.1",
    env_name = "Pendulum-v0",
    env_action_space = DiscreteActionSpace(
        count = (5,)
    ),
    env_observation_shape = (3,),
    env_action_transform = ActionDiscretizer(
        action_shape = (1,),
        disc_counts = np.array([5]),
        cont_lower_bound = np.array([-2]),
        cont_upper_bound = np.array([2])
    ),
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 20,
    env_discount_rate = 0.9,
    trainer_class = QLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.05,
            duration = 300
        ),
        epochs_per_step = 4,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = MLP(
            layer_sizes = [3, 8, 5]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "pendulum.a2c.sars.1",
    env_name = "Pendulum-v0",
    env_action_space = ContinuousActionSpace(
        shape = (1,),
        lower_bound = np.array([-2]),
        upper_bound = np.array([2])
    ),
    env_observation_shape = (3,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 5,
    env_discount_rate = 0.99,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        policy_network_architecture = MultiheadModule(
            shared_module = MLP(
                layer_sizes = [3, 8, 4],
                final_layer_activation = True
            ),
            head_modules = (
                Linear(4, 1),
                Linear(4, 1)
            )
        ),
        v_network_architecture = MLP(
            layer_sizes = (3, 8, 4, 1)
        ),
        epochs_per_step = 1,
        batch_size = None,
        minibatch_size = 32,
        optimizer_class = torch.optim.Adam,
        optimizer_args = {
            'lr': 1e-2
        },
        soft_target_update_fraction = 0.01,
        gradient_norm_clip_threshold = 0.5
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "pendulum.a2c.td.1",
    env_name = "Pendulum-v0",
    env_action_space = ContinuousActionSpace(
        shape = (1,),
        lower_bound = np.array([-2]),
        upper_bound = np.array([2])
    ),
    env_observation_shape = (3,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 4,
    env_discount_rate = 0.95,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        policy_network_architecture = MultiheadModule(
            shared_module = MLP(
                layer_sizes = [3, 8, 8],
                final_layer_activation = True
            ),
            head_modules = (
                Linear(8, 1),
                Linear(8, 1)
            )
        ),
        v_network_architecture = MLP(
            layer_sizes = (3, 8, 8, 1)
        ),
        epochs_per_step = 4,
        batch_size = None,
        minibatch_size = 32,
        optimizer_class = torch.optim.SGD,
        optimizer_args = {
            'lr': 5e-4
        },
        gradient_norm_clip_threshold = 0.5,
        entropy_loss_constant = 0
    ),
    exp_buffer_class = TDExpBuffer,
    exp_buffer_capacity = 100000,
    exp_buffer_td_level = 10
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "pendulum.act5.a2c.1",
    env_name = "Pendulum-v0",
    env_action_space = DiscreteActionSpace(
        count = (5,)
    ),
    env_observation_shape = (3,),
    env_action_transform = ActionDiscretizer(
        action_shape = (1,),
        disc_counts = np.array([5]),
        cont_lower_bound = np.array([-2]),
        cont_upper_bound = np.array([2]),
    ),
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 5,
    env_discount_rate = 0.99,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        epochs_per_step = 1,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.Adam,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        policy_network_architecture = MLP(
            layer_sizes = [3, 8, 5]
        ),
        v_network_architecture = MLP(
            layer_sizes = [3, 8, 1]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

'''
Double-Pendulum Swing-up
'''
c = Config(
    name = "acrobot.ddqn.1",
    env_name = "Acrobot-v1",
    env_action_space = DiscreteActionSpace(
        count = (3,)
    ),
    env_observation_shape = (6,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 20,
    env_discount_rate = 0.95,
    trainer_class = DoubleQLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-4
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = MLP(
            layer_sizes = [6, 32, 16, 3]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c


'''
Atari Assault
'''
c = Config(
    name = "assault_ram.dqn.1",
    env_name = "Assault-ram-v0",
    env_action_space = DiscreteActionSpace(
        count = (7,)
    ),
    env_observation_shape = (1024,),
    env_action_transform = None,
    env_observation_transform = ByteUnpacker(),
    env_max_episode_steps = None,
    env_episodes_per_step = 10,
    env_discount_rate = 0.95,
    trainer_class = QLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 20000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = MLP(
            layer_sizes = [1024, 512, 256, 256, 64, 7]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 1000000
)
CONFIG_REGISTRY[c.name] = c

'''
Atari Skiing
'''
c = Config(
    name = "skiing_ram.dqn.1",
    env_name = "Skiing-ram-v0",
    env_action_space = DiscreteActionSpace(
        count = (3,)
    ),
    env_observation_shape = (1024,),
    env_action_transform = None,
    env_observation_transform = ByteUnpacker(),
    env_max_episode_steps = None,
    env_episodes_per_step = 10,
    env_discount_rate = 0.95,
    trainer_class = QLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 20000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = MLP(
            layer_sizes = [1024, 512, 256, 128, 64, 3]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 1000000
)
CONFIG_REGISTRY[c.name] = c

'''
Lunar Lander
'''
c = Config(
    name = "lander.ddqn.1",
    env_name = "LunarLander-v2",
    env_action_space = DiscreteActionSpace(
        count = (4,)
    ),
    env_observation_shape = (8,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 10,
    env_discount_rate = 0.95,
    trainer_class = DoubleQLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-3
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = MLP(
            layer_sizes = [8, 32, 64, 32, 4]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "lander.ddqn.2",
    env_name = "LunarLander-v2",
    env_action_space = DiscreteActionSpace(
        count = (4,)
    ),
    env_observation_shape = (8,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 10,
    env_discount_rate = 0.95,
    trainer_class = DoubleQLearning,
    trainer_config = QLearning_Config(
        epsilon_schedule = LogarithmicSchedule(
            start = 0.5,
            end = 0.01,
            duration = 500
        ),
        epochs_per_step = 4,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-4
        },
        soft_target_update_fraction = 0.01,
        q_network_architecture = MLP(
            layer_sizes = [8, 32, 4]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "lander.a2c.1",
    env_name = "LunarLander-v2",
    env_action_space = DiscreteActionSpace(
        count = (4,)
    ),
    env_observation_shape = (8,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 3,
    env_discount_rate = 0.95,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        epochs_per_step = 2,
        batch_size = 10000,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-4
        },
        soft_target_update_fraction = 0.01,
        policy_network_architecture = MLP(
            layer_sizes = [8, 32, 4]
        ),
        v_network_architecture = MLP(
            layer_sizes = [8, 32, 1]
        )
    ),
    exp_buffer_class = SarsExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

'''
Continuous Lunar Lander
'''
c = Config(
    name = "lander_cont.a2c.mc.1",
    env_name = "LunarLanderContinuous-v2",
    env_action_space = ContinuousActionSpace(
        shape = (2,),
        lower_bound = np.array([-1, -1]),
        upper_bound = np.array([1, 1])
    ),
    env_observation_shape = (8,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 3,
    env_discount_rate = 0.99,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        policy_network_architecture = MultiheadModule(
            shared_module = MLP(
                layer_sizes = [8, 32],
                activation = Activation.TANH,
                final_layer_activation = True
            ),
            head_modules = (
                MLP(
                    layer_sizes = [32, 2]
                ),
                MLP(
                    layer_sizes = [32, 2]
                )
            )
        ),
        v_network_architecture = MLP(
            layer_sizes = [8, 32, 1],
            activation = Activation.TANH
        ),
        epochs_per_step = 2,
        batch_size = None,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-3
        },
        gradient_norm_clip_threshold = 0.5,
        entropy_loss_constant = 0.01
    ),
    exp_buffer_class = MCExpBuffer,
    exp_buffer_capacity = 100000
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "lander_cont.a2c.td.1",
    env_name = "LunarLanderContinuous-v2",
    env_action_space = ContinuousActionSpace(
        shape = (2,),
        lower_bound = np.array([-1, -1]),
        upper_bound = np.array([1, 1])
    ),
    env_observation_shape = (8,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 3,
    env_discount_rate = 0.99,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
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
        batch_size = None,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        gradient_norm_clip_threshold = 0.5,
        entropy_loss_constant = 1e-4
    ),
    exp_buffer_class = TDExpBuffer,
    exp_buffer_capacity = 100000,
    exp_buffer_td_level = 10
)
CONFIG_REGISTRY[c.name] = c

c = Config(
    name = "lander_cont.a2c.td.2",
    env_name = "LunarLanderContinuous-v2",
    env_action_space = ContinuousActionSpace(
        shape = (2,),
        lower_bound = np.array([-1, -1]),
        upper_bound = np.array([1, 1])
    ),
    env_observation_shape = (8,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 5,
    env_discount_rate = 0.97,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        policy_network_architecture = MultiheadModule(
            shared_module = MLP(
                layer_sizes = [8, 64],
                activation = Activation.RELU,
                final_layer_activation = True
            ),
            head_modules = (
                MLP(
                    layer_sizes = [64, 8, 2],
                    activation = Activation.RELU,
                    bounded_output = (-10, 10)
                ),
                MLP(
                    layer_sizes = [64, 8, 2],
                    activation = Activation.RELU,
                    bounded_output = (-10, 10)
                )
            )
        ),
        v_network_architecture = MLP(
            layer_sizes = [8, 64, 8, 1],
            activation = Activation.RELU
        ),
        epochs_per_step = 2,
        batch_size = None,
        minibatch_size = 64,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-4,
            'weight_decay': 0
        },
        gradient_norm_clip_threshold = 0.5,
        entropy_loss_constant = 1e-4
    ),
    exp_buffer_class = TDExpBuffer,
    exp_buffer_capacity = 100000,
    exp_buffer_td_level = 4
)
CONFIG_REGISTRY[c.name] = c

'''
Continuous Mountain Car
'''
c = Config(
    name = "mountaincar_cont.a2c.td.1",
    env_name = "MountainCarContinuous-v0",
    env_action_space = ContinuousActionSpace(
        shape = (1,),
        lower_bound = np.array([-1]),
        upper_bound = np.array([1])
    ),
    env_observation_shape = (2,),
    env_action_transform = None,
    env_observation_transform = None,
    env_max_episode_steps = None,
    env_episodes_per_step = 1,
    env_discount_rate = 0.99,
    trainer_class = ActorCritic,
    trainer_config = ActorCritic_Config(
        policy_network_architecture = MultiheadModule(
            shared_module = MLP(
                layer_sizes = [2, 8, 8],
                final_layer_activation = True
            ),
            head_modules = (
                Linear(8, 1),
                Linear(8, 1)
            )
        ),
        v_network_architecture = MLP(
            layer_sizes = (2, 8, 8, 1)
        ),
        epochs_per_step = 2,
        batch_size = None,
        minibatch_size = 32,
        optimizer_class = torch.optim.RMSprop,
        optimizer_args = {
            'lr': 1e-3
        },
        gradient_norm_clip_threshold = 0.5,
        entropy_loss_constant = 0.01
    ),
    exp_buffer_class = TDExpBuffer,
    exp_buffer_capacity = 100000,
    exp_buffer_td_level = 20
)
CONFIG_REGISTRY[c.name] = c