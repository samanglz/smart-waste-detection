from pathlib import Path
import json

from src.logging.logger import get_logger
from src.evaluation.hard_mining.hard_case_similarity import (
    HardCaseSimilarityAnalyzer,
)
from src.evaluation.hard_mining.hard_example_miner import (
    HardExampleMiner,
)

logger = get_logger(__name__)


# ============================================================
# Configuration
# ============================================================

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

# E2 outputs
PRIORITY_SUMMARY_PATH = Path(
    "outputs/hard_case_analysis/E2/priority_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/hard_example_mining/E2"
)

# TRAIN only
TRAIN_IMAGES_DIR = Path(
    "data/processed/train/images"
)

TRAIN_LABELS_DIR = Path(
    "data/processed/train/labels"
)

# Mining configuration
TOP_K_PER_CASE = 5
MAX_SAMPLES_PER_CONFUSION = 65

DEVICE = "cuda"


# ============================================================
# Helpers
# ============================================================

def load_priority_summary(
    path: Path,
) -> dict:
    """
    Load prioritized confusion summary.

    Priority information comes directly from the
    previous ErrorPrioritizer stage.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Priority summary not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_result(
    result: dict,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("HARD EXAMPLE MINING")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Load prioritization result
    # --------------------------------------------------------

    logger.info(
        "Loading priority summary: %s",
        PRIORITY_SUMMARY_PATH,
    )

    priority_summary = load_priority_summary(
        PRIORITY_SUMMARY_PATH
    )

    prioritized_confusions = (
        priority_summary.get(
            "priority_confusions",
            [],
        )
    )

    if not prioritized_confusions:
        raise ValueError(
            "No prioritized confusions found."
        )

    logger.info(
        "Prioritized confusions: %d",
        len(prioritized_confusions),
    )

    # --------------------------------------------------------
    # 2. Log selected priorities
    # --------------------------------------------------------

    for item in prioritized_confusions:

        logger.info(
            "Priority #%s: %s → %s | cases=%s",
            item.get("rank", "?"),
            item["gt_class"],
            item["pred_class"],
            item["count"],
        )

    # --------------------------------------------------------
    # 3. Build TRAIN embedding index
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("Building TRAIN embedding index")
    logger.info("=" * 70)

    analyzer = HardCaseSimilarityAnalyzer(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        class_names=CLASS_NAMES,
        device=DEVICE,
    )

    # Only classes that appear as GT in prioritized
    # confusions are required.
    target_classes = sorted(
        {
            item["gt_class"]
            for item in prioritized_confusions
        }
    )

    logger.info(
        "TRAIN target classes: %s",
        target_classes,
    )

    analyzer.build_train_index(
        target_classes=target_classes
    )

    # --------------------------------------------------------
    # 4. Create Hard Example Miner
    # --------------------------------------------------------

    miner = HardExampleMiner(
        similarity_analyzer=analyzer,
    )

    # --------------------------------------------------------
    # 5. Mine hard TRAIN samples
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("Mining hard TRAIN samples")
    logger.info("=" * 70)

    mining_result = miner.mine_prioritized_confusions(
        prioritized_confusions=prioritized_confusions,
        top_k_per_case=TOP_K_PER_CASE,
        max_samples_per_confusion=MAX_SAMPLES_PER_CONFUSION,
    )

    # --------------------------------------------------------
    # 6. Add experiment metadata
    # --------------------------------------------------------

    mining_result["experiment"] = "E2"

    mining_result["configuration"] = {
        "top_k_per_case": TOP_K_PER_CASE,
        "max_samples_per_confusion": (
            MAX_SAMPLES_PER_CONFUSION
        ),
        "train_images_dir": str(
            TRAIN_IMAGES_DIR
        ),
        "train_labels_dir": str(
            TRAIN_LABELS_DIR
        ),
        "device": str(
            analyzer.device
        ),
    }

    # --------------------------------------------------------
    # 7. Save result
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "hard_example_mining.json"
    )

    save_result(
        result=mining_result,
        output_path=output_path,
    )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("HARD EXAMPLE MINING COMPLETED")
    logger.info("=" * 70)

    logger.info(
        "Priority source: ErrorPrioritizer.rank_confusions"
    )

    logger.info(
        "Test samples added to train: NO"
    )

    for result in mining_result["confusions"]:

        logger.info(
            "Priority #%d: %s → %s | "
            "selected TRAIN samples: %d",
            result["rank"],
            result["gt_class"],
            result["pred_class"],
            result["selected_train_samples"],
        )

    logger.info(
        "Output: %s",
        output_path,
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()  