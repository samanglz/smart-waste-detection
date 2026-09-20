from pathlib import Path

from src.trainers.yolo_trainer import YOLOTrainer
from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger

logger = get_logger(__name__)


class HardMiningTrainer(YOLOTrainer):

    EPOCHS = 30
    BATCH = 2
    IMGSZ = 640

    LR0 = 0.0005
    OPTIMIZER = "AdamW"

    PATIENCE = 30
    COS_LR = True

    PROJECT = "runs/E6_hard_mining"
    RUN_NAME = "hard_mining"

    DISABLE_AUG = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
    }

    def __init__(self, config, dataset, pretrained_path: Path):
        super().__init__(config, dataset)
        self.pretrained_path = Path(pretrained_path)

    def build_model(self):
        if not self.pretrained_path.exists():
            raise FileNotFoundError(self.pretrained_path)

        self.model = YOLOModel(self.pretrained_path)

    def prepare_dataset(self):
        if not Path(self.dataset.yaml_path).exists():
            raise FileNotFoundError(self.dataset.yaml_path)

    def train(self):

        self.prepare_dataset()
        self.build_model()

        params = {
            "data": str(self.dataset.yaml_path),
            "epochs": self.EPOCHS,
            "imgsz": self.IMGSZ,
            "batch": self.BATCH,
            "lr0": self.LR0,
            "optimizer": self.OPTIMIZER,
            "cos_lr": self.COS_LR,
            "patience": self.PATIENCE,
            "project": self.PROJECT,
            "name": self.RUN_NAME,
            "save_period": 10,
            **self.DISABLE_AUG,
        }

        self.model.train(**params)

    def validate(self):
        return self.model.val(
            data=str(self.dataset.yaml_path)
        )