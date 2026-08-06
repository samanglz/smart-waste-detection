from abc import ABC, abstractmethod

class Predictable(ABC):
    
    @abstractmethod
    def predict(self, **kwargs):
        pass