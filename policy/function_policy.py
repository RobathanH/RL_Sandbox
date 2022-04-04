from typing import Callable
import numpy as np

from .policy import Policy

'''
Simplest policy, which is simply a wrapper for an arbitrary function, which takes observations and outputs actions.
'''
class FunctionPolicy(Policy):
    '''
    Args:
        policy_function (observation -> action): Arbitrary policy function to instantiate
    '''
    def __init__(self, policy_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.policy_function = policy_function

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.policy_function(observation)

'''
Wrapper for an arbitrary policy function with an added epsilon chance of taking a random action
'''
class EpsilonGreedyFunctionPolicy(FunctionPolicy):
    '''
    Args:
        policy_function (observation -> action):    Arbitrary policy function to instantiate
        epsilon (float):                            Fractional probability of taking a random action at any turn
        random_action_sampler (None -> action):     Function which returns a random action
    '''
    def __init__(self, policy_function: Callable, epsilon: float, random_action_sampler: Callable) -> None:
        super().__init__(policy_function)
        self.epsilon = epsilon
        self.random_action_sampler = random_action_sampler

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if np.random.uniform() < self.epsilon:
            return self.random_action_sampler()
        else:
            return super().get_action(observation)