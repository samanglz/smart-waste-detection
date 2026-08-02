from pathlib import Path
from ultralytics import YOLO

from src.models.capabilities.predictable import Predictable
from src.models.capabilities.trainable import Trainable


class YOLOModel(Trainable, Predictable):
    """
    Wrapper around Ultralytics YOLO model.
    """

    def __init__(self, weight_path: Path):
        self._weight_path = weight_path
        self._model = YOLO(str(weight_path))

    @property
    def model(self) -> YOLO:
        return self._model

    @property
    def weight_path(self) -> Path:
        return self._weight_path
    
    
    
    def train(self, **kwargs):
        return self._model.train(**kwargs)

    def predict(self, **kwargs):
        return self._model.predict(**kwargs)

    def val(self, **kwargs):
        return self._model.val(**kwargs)

    def export(self, **kwargs):
        return self._model.export(**kwargs)