"""
E1 - Fine-tuning trainer.

E1 changes only the learning-rate strategy relative to E0.

No:
    - class weighting
    - class-specific augmentation
    - hard-example sampling
    - embedding mining
"""

from pathlib import Path

from src.trainers.yolo_trainer import YOLOTrainer
from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger


logger = get_logger(__name__)


class FineTuneTrainer(YOLOTrainer):
    """
    Trainer for E1 fine-tuning from an existing YOLO checkpoint.

    E1 evaluates whether continued training with a lower learning
    rate improves the detector.
    """

    def __init__(
        self,
        config,
        dataset,
        pretrained_path: Path,
    ):
        super().__init__(config, dataset)

        self.pretrained_path = Path(pretrained_path)

        logger.info("FineTuneTrainer initialized")
        logger.info(
            "  Pre-trained model: %s",
            self.pretrained_path,
        )

    # ============================================================
    # MODEL
    # ============================================================

    def build_model(self):
        """
        Load the exact E0 checkpoint.

        No fallback to another model is allowed.
        """

        logger.info(
            "Loading pretrained checkpoint: %s",
            self.pretrained_path,
        )

        if not self.pretrained_path.exists():
            raise FileNotFoundError(
                "E1 requires the E0 checkpoint, but it was not found:\n"
                f"{self.pretrained_path}"
            )

        self.model = YOLOModel(
            self.pretrained_path
        )

        logger.info(
            "E0 checkpoint loaded successfully."
        )

    # ============================================================
    # TRAIN
    # ============================================================

    def train(self):
        """
        Run E1 fine-tuning.

        The only experimental variable is the lower learning rate.
        """

        logger.info("=" * 60)
        logger.info("E1 - LOWER LEARNING RATE FINE-TUNING")
        logger.info("=" * 60)

        aug_params = {}

        if hasattr(self.config, "AUGMENTATION"):
            aug_params = self.config.AUGMENTATION.to_dict()

        self.model.train(
            data=str(self.dataset.yaml_path),
            epochs=self.config.EPOCHS,
            imgsz=self.config.IMAGE_SIZE,
            batch=self.config.BATCH_SIZE,

            # E1 experimental variable
            lr0=self.config.LR0,
            lrf=self.config.LRF,
            cos_lr=True,

            # Early stopping
            patience=getattr(
                self.config,
                "PATIENCE",
                20,
            ),

            # Output
            project=getattr(
                self.config,
                "PROJECT_NAME",
                "runs",
            ),
            name=getattr(
                self.config,
                "RUN_NAME",
                "E1",
            ),

            # Same augmentation as E0
            **aug_params,
        )

        logger.info("=" * 60)
        logger.info("E1 TRAINING COMPLETED")
        logger.info("=" * 60)