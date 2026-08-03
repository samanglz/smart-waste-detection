from abc import ABC, abstractmethod
from src.logging.logger import get_logger


class BaseTrainer(ABC):
    
    def __init__(self, config):
    
        self.config = config
        self.model = None
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = None
        self.optimizer = None

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
