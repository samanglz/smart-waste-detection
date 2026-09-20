#!/usr/bin/env python

from pathlib import Path
import sys
from types import SimpleNamespace

from configs import training_config as config
from src.trainers.E6_hard_mining_trainer import HardMiningTrainer
from src.pipelines.training_pipeline import TrainingPipeline
from src.logging.logger import get_logger

logger = get_logger(__name__)

E2_BEST_PT = Path(
    "runs/detect/runs/E2_targeted_augmentation/targeted_aug/weights/best.pt"
)

E6_DATASET_YAML = Path(
    "data/data.yaml"
)


def main():

    logger.info("=" * 70)
    logger.info("E6 — HARD MINING")
    logger.info("=" * 70)

    config.EPOCHS = 30
    config.BATCH_SIZE = 2
    config.IMAGE_SIZE = 640
    config.LR0 = 0.0005
    config.OPTIMIZER = "AdamW"
    config.PATIENCE = 30

    config.PROJECT_NAME = "runs/E6_hard_mining"
    config.RUN_NAME = "hard_mining"

    if not E2_BEST_PT.exists():
        raise FileNotFoundError(E2_BEST_PT)

    if not E6_DATASET_YAML.exists():
        raise FileNotFoundError(E6_DATASET_YAML)

    dataset = SimpleNamespace(
        yaml_path=E6_DATASET_YAML
    )

    trainer = HardMiningTrainer(
        config=config,
        dataset=dataset,
        pretrained_path=E2_BEST_PT,
    )

    pipeline = TrainingPipeline(
        config=config,
        trainer=trainer,
    )

    pipeline.run()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        sys.exit(1)

    except Exception:
        logger.exception("E6 failed")
        sys.exit(1)