from abc import ABC, abstractmethod

class Trainable(ABC):
    
    @abstractmethod
    def train(self, **kwargs):
        pass