class Trainable(ABC):
    
    @abstractmethod
    def train(self, **kwargs):
        pass