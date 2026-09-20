from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from src.models.capabilities.predictable import Predictable
from src.models.capabilities.trainable import Trainable


class YOLOModel(Trainable, Predictable):
    """
    Wrapper around Ultralytics YOLO model.

    Supports:
        1. Standard Ultralytics checkpoints (.pt)
        2. Custom training checkpoints containing model_state_dict
    """

    def __init__(self, weight_path: Path, base_model: Path = Path("yolo11m.pt")):
        self._weight_path = Path(weight_path)

        print(f"🔵 weight_path: {self._weight_path}")

        # ------------------------------------------------------
        # Try reading checkpoint metadata
        # ------------------------------------------------------

        ckpt = torch.load(
            self._weight_path,
            map_location="cpu",
            weights_only=False,
        )

        # ------------------------------------------------------
        # Custom checkpoint (E5)
        # ------------------------------------------------------
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:

            print("🟢 Custom E5 checkpoint detected")

            # Build 5-class architecture
            model = DetectionModel(
                cfg="yolo11m.yaml",
                nc=5,
            )

            model.load_state_dict(
                ckpt["model_state_dict"],
                strict=True,
            )

            self._model = YOLO("yolo11m.yaml")
            self._model.model = model
        # ------------------------------------------------------
        # Standard Ultralytics checkpoint
        # ------------------------------------------------------

        else:

            print("🟢 Standard Ultralytics checkpoint")

            self._model = YOLO(str(self._weight_path))

    @property
    def model(self):
        return self._model

    @property
    def weight_path(self):
        return self._weight_path

    def train(self, **kwargs):
        return self._model.train(**kwargs)

    def predict(self, **kwargs):
        return self._model.predict(**kwargs)

    def val(self, **kwargs):
        return self._model.val(**kwargs)

    def export(self, **kwargs):
        return self._model.export(**kwargs)