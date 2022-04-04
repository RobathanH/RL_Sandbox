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
    def value(self, step: int) -> float:
        raise NotImplementedError


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
    def value(self, step: int) -> float:
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
    def value(self, step: int) -> float:
        lin_schedule = LinearSchedule(np.log(self.start), np.log(self.end), np.log(self.duration))
        return np.exp(lin_schedule.value(step))