from abc import ABC, abstractmethod



class BaseTrainer(ABC):
    
    def __init__(self, config):
    
        self.config = config
        self.model = None
        self.logger = None
        self.metrics = None

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass