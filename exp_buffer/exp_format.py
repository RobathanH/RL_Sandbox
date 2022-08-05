from dataclasses import dataclass
from typing import Any, Optional



'''
Stores the experience acquired during an episode in the environment.
Using AI gym environment format.
'''
class Trajectory:
    @dataclass
    class TrajectoryStart:
        state: Any

    @dataclass
    class TrajectoryStep:
        action: Any
        reward: float
        next_state: Any
        done: bool

    def __init__(self) -> None:
        self.start = None
        self.steps = []
        
        
        
    # Building trajectory

    def add_start(self, state) -> None:
        self.start = Trajectory.TrajectoryStart(state)

    def add_step(self, action, reward: float, next_state, done: bool) -> None:
        self.steps.append(Trajectory.TrajectoryStep(action, reward, next_state, done))
        
    
    
    # Metrics
    
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)
    
    def length(self) -> int:
        return len(self.steps)    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)