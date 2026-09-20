#!/usr/bin/env python

"""
E1 — Fine-Tuning from YOLO11m Baseline

Ablation:
    E0 = Baseline
    E1 = Same model + lower learning rate

Only the learning-rate strategy is changed.
No class weighting.
No targeted augmentation.
No hard-example sampling.
No embedding mining.
"""

from pathlib import Path
import sys

from configs import config

from src.data.yolo_dataset import YOLODataset
from src.trainers.fine_tune_trainer import FineTuneTrainer
from src.pipelines.training_pipeline import TrainingPipeline
from src.logging.logger import get_logger


logger = get_logger(__name__)


def main():

    logger.info("=" * 70)
    logger.info("E1 — FINE-TUNING FROM YOLO11m BASELINE")
    logger.info("=" * 70)

    # ============================================================
    # 1. E1 Configuration
    # ============================================================

    config.EPOCHS = 30
    config.BATCH_SIZE = 4
    config.WORKERS = 4

    # Lower LR for fine-tuning
    config.LR0 = 0.0005
    config.LRF = 0.001

    config.PROJECT_NAME = "runs"
    config.RUN_NAME = "E1_fine_tune"

    logger.info("Configuration:")
    logger.info("  Epochs     : %d", config.EPOCHS)
    logger.info("  Batch size : %d", config.BATCH_SIZE)
    logger.info("  Accumulate  : %d", config.ACCUMULATE)
    logger.info("  LR0         : %.6f", config.LR0)
    logger.info("  LRF         : %.6f", config.LRF)

    # ============================================================
    # 2. Dataset
    # ============================================================

    logger.info("-" * 70)
    logger.info("Loading dataset...")

    dataset = YOLODataset(
        config.FINAL_DATASET_DIR
    )

    logger.info(
        "Classes: %s",
        dataset.get_class_names()
    )

    logger.info(
        "Training samples: %d",
        len(dataset.get_train_data())
    )

    # ============================================================
    # 3. Pretrained YOLO11m checkpoint
    # ============================================================

    pretrained_path = Path(
        "runs/detect/runs/yolo11m/detect/weights/best.pt"
    )

    if not pretrained_path.exists():

        raise FileNotFoundError(
            f"YOLO11m baseline checkpoint not found:\n"
            f"{pretrained_path}\n\n"
            "E1 must start from the E0 baseline checkpoint."
        )

    logger.info(
        "Starting checkpoint: %s",
        pretrained_path
    )

    # ============================================================
    # 4. FineTuneTrainer
    # ============================================================

    trainer = FineTuneTrainer(
        config=config,
        dataset=dataset,
        pretrained_path=pretrained_path,
    )

    # ============================================================
    # 5. Training Pipeline
    # ============================================================

    logger.info("-" * 70)
    logger.info("Starting E1...")
    logger.info("-" * 70)

    pipeline = TrainingPipeline(
        config=config,
        trainer=trainer,
    )

    pipeline.run()

    # ============================================================
    # 6. Result
    # ============================================================

    output_path = (
        Path(config.PROJECT_NAME)
        / config.RUN_NAME
        / "weights"
        / "best.pt"
    )

    logger.info("=" * 70)
    logger.info("E1 COMPLETED")
    logger.info("=" * 70)

    logger.info(
        "Best model: %s",
        output_path
    )

    logger.info("")
    logger.info("Compare E1 against E0:")
    logger.info("")
    logger.info("E0:")
    logger.info("  Glass   AP = 0.7442")
    logger.info("  Plastic AP = 0.7190")
    logger.info("")
    logger.info("E1:")
    logger.info("  Evaluate before deciding E2.")
    logger.info("=" * 70)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        logger.info(
            "Training interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        logger.exception(
            "E1 failed: %s",
            exc
        )

        sys.exit(1)