from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def train(self, **kwargs):
        pass


    @abstractmethod
    def predict(self, **kwargs):
        pass


    @abstractmethod
    def val(self, **kwargs):
        pass


    @abstractmethod
    def export(self, **kwargs):
        pass