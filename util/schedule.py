from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

'''
Schedules for adjusting float variables over time.
Mainly for epsilon decay or learning rate decay
'''

class Schedule(ABC):
    '''
    Retrieve the current value of the variable under the schedule.
    Args:
        step (int):     Current step index.
    Returns:
        (float)
    '''
    @abstractmethod
    def value(self, step: int = 0) -> float:
        raise NotImplementedError

'''
Constant schedule
'''
@dataclass
class Constant(Schedule):
    val: float
    
    '''
    Retrieve the current value of the variable under the schedule.
    Args:
        step (int):     Current step index.
    Returns:
        (float)
    '''
    def value(self, step: int = 0) -> float:
        return self.val

'''
Linear schedule from starting value to ending value in a set number of steps
'''
@dataclass
class LinearSchedule(Schedule):
    start: float            # Initial value
    end: float              # Final value (for step >= duration)
    duration: float         # Duration of schedule (in steps)

    '''
    Retrieve the current value of the variable under the schedule.
    Args:
        step (int):     Current step index.
    Returns:
        (float)
    '''
    def value(self, step: int = 0) -> float:
        return self.start + (self.end - self.start) * min(step / self.duration, 1)



'''
Logarithmic schedule
'''
@dataclass
class LogarithmicSchedule(Schedule):
    start: float            # Initial value
    end: float              # Final value (for step >= duration)
    duration: float         # Duration of schedule (in steps)

    '''
    Retrieve the current value of the variable under the schedule.
    Args:
        step (int):     Current step index.
    Returns:
        (float)
    '''
    def value(self, step: int = 0) -> float:
        lin_schedule = LinearSchedule(np.log(self.start), np.log(self.end), np.log(self.duration))
        return np.exp(lin_schedule.value(step))
    
'''
Effectively the same as LogarithmicSchedule, but
with different parameters which are less dependent on
each other (better for hyperparameter sweeps)
'''
@dataclass
class LogarithmicDecaySchedule(Schedule):
    start: float            # Initial value
    end: float              # Minimum value, after which decay stops
    decay: float            # Decay multiplier per step
    
    '''
    Retrieve the current value of the variable under the schedule.
    Args:
        step (int):     Current step index.
    Returns:
        (float)
    '''
    def value(self, step: int = 0) -> float:
        log_schedule = LogarithmicSchedule(
            start = self.start,
            end = self.end,
            duration = (np.log(self.start) - np.log(self.end)) // np.log(self.decay)
        )
        return log_schedule.value(step)
    
# Register for importing
from config.module_importer import REGISTER_MODULE
REGISTER_MODULE(__name__)