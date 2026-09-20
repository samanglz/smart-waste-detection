from pathlib import Path
from typing import Dict
import json

import torch
from torch.utils.data import WeightedRandomSampler

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader, seed_worker

from src.trainers.base_trainer import BaseTrainer
from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger


logger = get_logger(__name__)


# =================================================================
# Hard Sampling Sampler
# =================================================================

class EpochWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler with epoch-aware random seed.

    This allows the sampled order to change between epochs
    while keeping reproducibility.

    This sampler is intended for single-process training.
    """

    def __init__(
        self,
        weights,
        num_samples: int,
        replacement: bool = True,
        seed: int = 6148914691236517205,
    ):
        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
        )

        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch.

        Kept for compatibility with training loops that
        expect samplers to expose set_epoch().
        """
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()

        generator.manual_seed(
            self.seed + self.epoch
        )

        return iter(
            torch.multinomial(
                self.weights,
                self.num_samples,
                self.replacement,
                generator=generator,
            ).tolist()
        )


# =================================================================
# Ultralytics Trainer
# =================================================================

class HardSamplingDetectionTrainer(DetectionTrainer):
    """
    Ultralytics DetectionTrainer with hard-example sampling.

    IMPORTANT:

    TRAIN:
        WeightedRandomSampler is applied.

    VAL:
        Standard Ultralytics DataLoader is used.

    TEST:
        Never used for training.

    The sampler uses image-level weights generated from
    hard-example mining.
    """

    def __init__(
        self,
        *args,
        sampling_weights_path: Path,
        sampling_replacement: bool = True,
        sampling_seed: int = 6148914691236517205,
        **kwargs,
    ):
        self.sampling_weights_path = (
            Path(sampling_weights_path).resolve()
        )

        self.sampling_replacement = sampling_replacement
        self.sampling_seed = sampling_seed

        self.sampling_weights = (
            self._load_sampling_weights()
        )

        super().__init__(
            *args,
            **kwargs,
        )
    
    # =============================================================
    # Sampling weights
    # =============================================================

    def _load_sampling_weights(
        self,
    ) -> Dict[str, float]:
        """
        Load image-path -> sampling-weight mapping.
        """

        if not self.sampling_weights_path.exists():
            raise FileNotFoundError(
                "Sampling weights file not found: "
                f"{self.sampling_weights_path}"
            )

        with open(
            self.sampling_weights_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        weights = data.get("weights")

        if not weights:
            raise ValueError(
                "No 'weights' section found in "
                f"{self.sampling_weights_path}"
            )

        normalized_weights = {
            self._normalize_path(path): float(weight)
            for path, weight in weights.items()
        }

        logger.info(
            "Loaded %d sampling weights.",
            len(normalized_weights),
        )

        statistics = data.get(
            "statistics",
            {},
        )

        if statistics:
            logger.info(
                "Sampling weight statistics:"
            )

            logger.info(
                "  Total train images : %s",
                statistics.get(
                    "total_train_images",
                    "?",
                ),
            )

            logger.info(
                "  Hard images        : %s",
                statistics.get(
                    "hard_images",
                    "?",
                ),
            )

            logger.info(
                "  Normal images      : %s",
                statistics.get(
                    "normal_images",
                    "?",
                ),
            )

            logger.info(
                "  Hard weight        : %s",
                statistics.get(
                    "hard_weight",
                    "?",
                ),
            )

        return normalized_weights

    # =============================================================
    # DataLoader
    # =============================================================

    def get_dataloader(
        self,
        dataset_path,
        batch_size=16,
        rank=0,
        mode="train",
    ):
        """
        Build Ultralytics dataloader.

        TRAIN:
            Uses hard-example WeightedRandomSampler.

        VAL:
            Uses the standard Ultralytics dataloader.
        """

        # ---------------------------------------------------------
        # Safety: current implementation is single-process
        # ---------------------------------------------------------

        if mode == "train" and rank != -1:

            raise RuntimeError(
                "HardSamplingDetectionTrainer currently "
                "supports single-process training only. "
                "Distributed hard sampling has not been "
                "implemented yet."
            )

        # ---------------------------------------------------------
        # Build normal Ultralytics dataset
        # ---------------------------------------------------------

        dataloader = super().get_dataloader(
            dataset_path,
            batch_size=batch_size,
            rank=rank,
            mode=mode,
        )

        # ---------------------------------------------------------
        # Validation remains untouched
        # ---------------------------------------------------------

        if mode != "train":

            logger.info(
                "Mode=%s → standard Ultralytics DataLoader.",
                mode,
            )

            return dataloader

        logger.info(
            "Applying HARD EXAMPLE SAMPLING to TRAIN..."
        )

        # ---------------------------------------------------------
        # Build image-level weights
        # ---------------------------------------------------------

        weights = self._build_dataset_weights(
            dataloader.dataset
        )

        weights_tensor = torch.tensor(
            weights,
            dtype=torch.double,
        )

        sampler = EpochWeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(weights),
            replacement=self.sampling_replacement,
            seed=self.sampling_seed,
        )
        # ---------------------------------------------------------
        # Preserve Ultralytics InfiniteDataLoader
        # ---------------------------------------------------------

        workers = (
            self.args.workers
            if mode == "train"
            else self.args.workers * 2
        )

        dataset_len = len(dataloader.dataset)

        batch_size = min(
            batch_size,
            dataset_len,
        )

        num_workers = min(
            torch.get_num_threads(),
            workers,
            0
            if batch_size == 0
            else max(
                1,
                dataset_len // batch_size,
            ),
        )

        # ---------------------------------------------------------
        # Rebuild InfiniteDataLoader
        # ---------------------------------------------------------

        hard_dataloader = InfiniteDataLoader(
            dataset=dataloader.dataset,

            batch_size=batch_size,

            shuffle=False,

            sampler=sampler,

            num_workers=num_workers,

            prefetch_factor=(
                4
                if num_workers > 0
                else None
            ),

            pin_memory=(
                torch.cuda.is_available()
            ),

            collate_fn=getattr(
                dataloader.dataset,
                "collate_fn",
                None,
            ),

            worker_init_fn=seed_worker,

            drop_last=(
                self.args.compile
            ),

        )

        logger.info(
            "Hard-example WeightedRandomSampler enabled."
        )

        logger.info(
            "TRAIN loader type: %s",
            type(hard_dataloader).__name__,
        )

        logger.info(
            "Sampler type: %s",
            type(sampler).__name__,
        )

        return hard_dataloader

    # =============================================================
    # Build dataset weights
    # =============================================================

    def _build_dataset_weights(
        self,
        dataset,
    ):
        """
        Match every training image in the Ultralytics
        dataset with its corresponding sampling weight.

        If an image is missing from sampling_weights.json,
        weight=1.0 is used.

        This means missing images remain normal samples.
        """

        image_paths = dataset.im_files

        weights = []

        missing = []

        hard_count = 0
        normal_count = 0

        for image_path in image_paths:

            normalized_path = (
                self._normalize_path(
                    image_path
                )
            )

            weight = (
                self.sampling_weights.get(
                    normalized_path
                )
            )

            if weight is None:

                missing.append(
                    normalized_path
                )

                weight = 1.0

            if weight > 1.0:

                hard_count += 1

            else:

                normal_count += 1

            weights.append(
                float(weight)
            )

        # ---------------------------------------------------------
        # Safety
        # ---------------------------------------------------------

        if len(weights) != len(image_paths):

            raise RuntimeError(
                "Number of sampling weights does not "
                "match number of training images."
            )

        if not weights:

            raise RuntimeError(
                "No training images found."
            )

        if missing:

            logger.warning(
                "%d training images were not found "
                "in sampling_weights.json. "
                "Default weight=1.0 applied.",
                len(missing),
            )

        logger.info(
            "=" * 60
        )

        logger.info(
            "HARD SAMPLING STATISTICS"
        )

        logger.info(
            "Total train images : %d",
            len(weights),
        )

        logger.info(
            "Hard images        : %d",
            hard_count,
        )

        logger.info(
            "Normal images      : %d",
            normal_count,
        )

        logger.info(
            "Weight range       : %.2f → %.2f",
            min(weights),
            max(weights),
        )

        logger.info(
            "=" * 60
        )

        return weights

    # =============================================================
    # Path normalization
    # =============================================================

    @staticmethod
    def _normalize_path(
        path,
    ) -> str:
        """
        Normalize Windows paths.

        Ensures paths from:

            sampling_weights.json

        and:

            Ultralytics dataset.im_files

        match correctly.
        """

        return str(
            Path(path)
            .resolve()
        ).lower()


# =================================================================
# Project-level Trainer
# =================================================================

class YOLOHardSamplerTrainer(BaseTrainer):
    """
    Project-level trainer for E3 Hard Example Sampling.

    E3:

        hard_example_mining.json
                    ↓
        sampling_weights.json
                    ↓
        WeightedRandomSampler
                    ↓
        TRAIN

    TEST images are never added to TRAIN.
    """

    def __init__(
        self,
        config,
        dataset,
    ):
        super().__init__(
            config
        )

        self.config = config
        self.dataset = dataset
        self.model = None

        self.ultralytics_trainer = None

    # =============================================================
    # Model
    # =============================================================

    def build_model(self):

        self.model = YOLOModel(
            self.config.MODEL_PATH
        )

    # =============================================================
    # Train
    # =============================================================

    def train(self):
        
        sampling_weights_path = Path(
            self.config.SAMPLING_WEIGHTS_PATH
        ).resolve()

        if not sampling_weights_path.exists():
            raise FileNotFoundError(
                "Hard sampling weights not found:\n"
                f"{sampling_weights_path}"
            )

        logger.info("=" * 70)
        logger.info("STARTING E3 HARD EXAMPLE SAMPLING")
        logger.info("=" * 70)

        logger.info("Model: %s", self.config.MODEL_PATH)
        logger.info("Dataset YAML: %s", self.dataset.yaml_path)
        logger.info(
            "Sampling weights: %s",
            sampling_weights_path,
        )

        logger.info("Epochs: %s", self.config.EPOCHS)
        logger.info("Batch size: %s", self.config.BATCH_SIZE)
        logger.info("Image size: %s", self.config.IMAGE_SIZE)
        logger.info("Learning rate: %s", self.config.LR0)
        logger.info("Optimizer: %s", self.config.OPTIMIZER)
        logger.info("Cos LR: %s", self.config.COS_LR)
        logger.info("Patience: %s", self.config.PATIENCE)

        # =========================================================
        # Create Ultralytics trainer
        # =========================================================

        self.ultralytics_trainer = HardSamplingDetectionTrainer(
            overrides={
                # -------------------------------------------------
                # Model
                # -------------------------------------------------

                "model": str(
                    self.config.MODEL_PATH
                ),

                # -------------------------------------------------
                # Dataset
                # -------------------------------------------------

                "data": str(
                    self.dataset.yaml_path
                ),

                # -------------------------------------------------
                # Training
                # -------------------------------------------------

                "epochs": self.config.EPOCHS,
                "imgsz": self.config.IMAGE_SIZE,
                "batch": self.config.BATCH_SIZE,

                "lr0": self.config.LR0,
                "optimizer": self.config.OPTIMIZER,
                "cos_lr": self.config.COS_LR,
                "patience": self.config.PATIENCE,

                # -------------------------------------------------
                # Output
                # -------------------------------------------------

                "project": self.config.PROJECT_NAME,
                "name": self.config.RUN_NAME,
                "save_period": self.config.SAVE_PERIOD,

                # -------------------------------------------------
                # Hardware
                # -------------------------------------------------

                "device": self.config.DEVICE,
                "workers": self.config.WORKERS,

                # -------------------------------------------------
                # E3:
                # Online augmentation MUST remain disabled.
                # -------------------------------------------------

                **self.config.ONLINE_AUGMENTATION,
            },

            sampling_weights_path=(
                sampling_weights_path
            ),
            
            sampling_replacement=( self.config.SAMPLING_REPLACEMENT ),  
            sampling_seed=( self.config.SAMPLING_SEED ), 
        )

        # =========================================================
        # Train
        # =========================================================

        self.ultralytics_trainer.train()

        # =========================================================
        # Load BEST checkpoint
        # =========================================================

        best_checkpoint = Path(
            self.ultralytics_trainer.best
        ).resolve()

        if not best_checkpoint.exists():

            logger.warning(
                "best.pt was not found at expected path: %s",
                best_checkpoint,
            )

            last_checkpoint = Path(
                self.ultralytics_trainer.last
            ).resolve()
            
            if not last_checkpoint.exists():
                raise FileNotFoundError(
                    "Neither best.pt nor last.pt "
                    "was found after training."
                )

            best_checkpoint = last_checkpoint

        logger.info(
            "Loading trained checkpoint: %s",
            best_checkpoint,
        )

        self.model = YOLOModel(
            best_checkpoint
        )

        logger.info("=" * 70)
        logger.info("E3 HARD EXAMPLE SAMPLING COMPLETED")
        logger.info(
            "Best checkpoint: %s",
            best_checkpoint,
        )
        logger.info("=" * 70)
        
        
        
        
        
        
    def validate(self):
    
        if self.model is None:
            self.build_model()

        return self.model.val()

    def save_checkpoint(self, path: Path):

        if self.model is None:
            raise RuntimeError(
                "Model has not been built or trained."
            )

        self.model.model.save(str(path))

        logger.info(
            "Checkpoint saved to: %s",
            path,
        )

    def load_checkpoint(self, path: Path):

        self.model = YOLOModel(path)

        logger.info(
            "Checkpoint loaded: %s",
            path,
        )