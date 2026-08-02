class Predictable(ABC):
    
    @abstractmethod
    def predict(self, **kwargs):
        pass