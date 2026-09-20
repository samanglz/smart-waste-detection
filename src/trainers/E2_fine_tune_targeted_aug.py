"""
E2 - Targeted Augmentation Trainer.

E2 uses the already-built offline targeted-augmentation dataset.

Experimental variables:
    - offline targeted augmentation for glass and plastic
    - lr0 = 0.0005
    - epochs = 30

Inherited from E1:
    - same model architecture
    - same optimizer strategy
    - same image size
    - same batch size

Disabled:
    - class weighting
    - hard-example sampling
    - embedding mining
    - loss modification
    - additional online augmentation
"""

from pathlib import Path

from src.trainers.yolo_trainer import YOLOTrainer
from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger


logger = get_logger(__name__)


class TargetedAugmentationTrainer(YOLOTrainer):
    """
    E2 trainer.

    IMPORTANT:
    The E2 dataset is already generated offline by
    TargetedAugmentationDatasetBuilder.

    Therefore this trainer must NOT rebuild or augment
    the dataset again.
    """

    # ============================================================
    # E2 EXPERIMENT CONSTANTS
    # ============================================================

    E2_EPOCHS = 30
    E2_LR0 = 0.0005

    # Inherited training parameters
    E2_BATCH_SIZE = 2
    E2_IMAGE_SIZE = 640

    # E1/E2 optimizer
    E2_OPTIMIZER = "AdamW"

    # Scheduler
    E2_COS_LR = True

    # Early stopping
    E2_PATIENCE = 30

    # Output
    E2_PROJECT = "runs/E2_targeted_augmentation"
    E2_RUN_NAME = "targeted_aug"

    # Online augmentation MUST remain disabled.
    E2_ONLINE_AUGMENTATION = {
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

    def __init__(
        self,
        config,
        dataset,
        pretrained_path: Path,
    ):
        super().__init__(config, dataset)

        self.pretrained_path = Path(pretrained_path)

        logger.info(
            "TargetedAugmentationTrainer initialized"
        )

        logger.info(
            "  E1 checkpoint: %s",
            self.pretrained_path,
        )

        logger.info(
            "  E2 epochs: %d",
            self.E2_EPOCHS,
        )

        logger.info(
            "  E2 lr0: %s",
            self.E2_LR0,
        )

        logger.info(
            "  E2 optimizer: %s",
            self.E2_OPTIMIZER,
        )

    # ============================================================
    # MODEL
    # ============================================================

    def build_model(self):
        """
        Load E1 best.pt.

        E2 has no fallback model.
        """

        if not self.pretrained_path.exists():
            raise FileNotFoundError(
                "E2 requires the E1 best checkpoint:\n"
                f"{self.pretrained_path}"
            )

        logger.info(
            "Loading E1 checkpoint: %s",
            self.pretrained_path,
        )

        self.model = YOLOModel(
            self.pretrained_path
        )

        logger.info(
            "E1 checkpoint loaded successfully."
        )

    # ============================================================
    # DATASET
    # ============================================================

    def prepare_dataset(self):
        """
        E2 dataset has already been built offline.

        DO NOT run TargetedAugmentation again.
        """

        yaml_path = Path(self.dataset.yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(
                "E2 dataset YAML was not found:\n"
                f"{yaml_path}"
            )

        logger.info(
            "Using pre-built E2 dataset: %s",
            yaml_path,
        )

        logger.info(
            "Offline targeted augmentation will NOT "
            "be executed again."
        )

    # ============================================================
    # TRAIN
    # ============================================================

    def train(self):
        """
        Execute the E2 ablation.

        Only E2 experimental change:
            offline targeted augmentation.
        """

        logger.info("=" * 60)
        logger.info(
            "E2 - TARGETED AUGMENTATION TRAINING"
        )
        logger.info("=" * 60)

        # --------------------------------------------------------
        # Dataset
        # --------------------------------------------------------

        self.prepare_dataset()

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        self.build_model()

        # --------------------------------------------------------
        # Training parameters
        # --------------------------------------------------------

        train_params = {
            "data": str(self.dataset.yaml_path),

            "epochs": self.E2_EPOCHS,
            "imgsz": self.E2_IMAGE_SIZE,
            "batch": self.E2_BATCH_SIZE,

            "lr0": self.E2_LR0,
            "optimizer": self.E2_OPTIMIZER,
            "cos_lr": self.E2_COS_LR,

            "patience": self.E2_PATIENCE,

            "project": self.E2_PROJECT,
            "name": self.E2_RUN_NAME,

            "save_period": 10,

            # ----------------------------------------------------
            # IMPORTANT:
            # The dataset already contains offline augmentation.
            # Disable Ultralytics online augmentation.
            # ----------------------------------------------------
            **self.E2_ONLINE_AUGMENTATION,
        }

        logger.info(
            "E2 training parameters:"
        )

        for key, value in train_params.items():
            logger.info(
                "  %s = %s",
                key,
                value,
            )

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        self.model.train(
            **train_params
        )

        logger.info("=" * 60)
        logger.info(
            "E2 TRAINING COMPLETED"
        )
        logger.info("=" * 60)

    # ============================================================
    # VALIDATE
    # ============================================================

    def validate(self):
        """
        Validate using the original validation dataset
        defined in E2 data.yaml.
        """

        logger.info(
            "Running E2 validation..."
        )

        return self.model.val(
            data=str(self.dataset.yaml_path)
        )

    # ============================================================
    # CHECKPOINT
    # ============================================================

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""

        self.model.model.save(
            str(path)
        )

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""

        self.model = YOLOModel(
            path
        )