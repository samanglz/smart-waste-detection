from pathlib import Path

from ultralytics import YOLO


class YOLOModel:
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