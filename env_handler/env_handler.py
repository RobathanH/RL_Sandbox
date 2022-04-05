from abc import ABC, abstractmethod
from typing import Type


'''
Simple interface for EnvHandler config and class
'''
class EnvHandler(ABC):
    pass

class EnvHandler_Config(ABC):
    '''
    Return the EnvHandler type associated with this config.
    '''
    @abstractmethod
    def get_class(self) -> Type[EnvHandler]:
        raise NotImplementedError
    
