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



'''
s,a,r,s' exp type
If next_state is None, tuple is assumed to have reached end of episode (done = True)
'''
@dataclass
class ExpSars:
    state: Any
    action: Any
    reward: float
    next_state: Optional[Any] = None

    def done(self) -> bool:
        return self.next_state is None

'''
s,a,r,s',a' tuple type
If next_state and/or next_action is None, tuple is assumed to have reached end of episode (done = True)
'''
@dataclass
class ExpSarsa:
    state: Any
    action: Any
    reward: float
    next_state: Optional[Any] = None
    next_action: Optional[Any] = None

    def done(self) -> bool:
        return self.next_state is None or self.next_action is None

'''
Full return tuple type, for Monte-Carlo estimators
'''
@dataclass
class ExpMC:
    state: Any
    action: Any
    returns: float
    
'''
Tuple type for Variable TD levels. Can represent both ExpSars and ExpMC
Stores state-action tuple and discounted rewards over the next n actions,
as well as the final next state for bootstrapping if not done.
'''
@dataclass
class ExpTD:
    state: Any
    action: Any
    reward: float
    next_state: Optional[Any] = None
    
    def done(self) -> bool:
        return self.next_state is None    
    
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)