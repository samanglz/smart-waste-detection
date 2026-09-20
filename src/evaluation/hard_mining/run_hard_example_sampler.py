from pathlib import Path
import json

from src.logging.logger import get_logger
from src.evaluation.hard_mining.hard_example_sampler import (
    HardExampleSampler,
)

logger = get_logger(__name__)


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = Path(
    "outputs/hard_example_mining/E2/hard_samples_manifest.json"
)

TRAIN_IMAGES_DIR = Path(
    "data/processed/train/images"
)

OUTPUT_DIR = Path(
    "outputs/hard_example_mining/E2"
)

HARD_WEIGHT = 3.0


# ============================================================
# Main
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("HARD EXAMPLE SAMPLER")
    logger.info("=" * 70)

    sampler = HardExampleSampler(
        manifest_path=MANIFEST_PATH,
        hard_weight=HARD_WEIGHT,
    )

    # Build image-level weights
    weights = sampler.build_sampling_weights(
        train_images_dir=TRAIN_IMAGES_DIR,
    )

    stats = sampler.get_statistics(
        train_images_dir=TRAIN_IMAGES_DIR,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / "sampling_weights.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "statistics": stats,
                "weights": weights,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("-" * 70)
    logger.info("Sampling statistics")
    logger.info("-" * 70)

    logger.info(
        "Total TRAIN images : %d",
        stats["total_train_images"],
    )

    logger.info(
        "Hard TRAIN images  : %d",
        stats["hard_images"],
    )

    logger.info(
        "Normal TRAIN images: %d",
        stats["normal_images"],
    )

    logger.info(
        "Hard weight        : %.1f",
        stats["hard_weight"],
    )

    logger.info("-" * 70)

    logger.info(
        "Sampling weights saved to:"
    )

    logger.info(
        "  %s",
        output_path,
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()