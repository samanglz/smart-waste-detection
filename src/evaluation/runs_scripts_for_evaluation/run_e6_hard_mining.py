from pathlib import Path
import json

from src.logging.logger import get_logger
from src.evaluation.prioritization.error_prioritizer import (
    ErrorPrioritizer,
)
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

# E6 raw TEST error analysis
ERROR_ANALYSIS_PATH = Path(
    "outputs/yolo11m/E6_hard_mining_targeted_aug/error_analysis.json"
)

# E6 hard mining output
OUTPUT_DIR = Path(
    "outputs/hard_example_mining/E6"
)

# TRAIN only
TRAIN_IMAGES_DIR = Path(
    "data/processed/train/images"
)

TRAIN_LABELS_DIR = Path(
    "data/processed/train/labels"
)

# Mining configuration
TOP_K_CONFUSIONS = 5
TOP_K_PER_CASE = 5
MAX_SAMPLES_PER_CONFUSION = 65

DEVICE = "cuda"


# ============================================================
# Helpers
# ============================================================

def load_error_analysis(
    path: Path,
) -> dict:
    """
    Load raw E6 error analysis.

    TEST is used only as the query source.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"E6 error analysis not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    if "error_analysis" not in report:
        raise ValueError(
            "Missing 'error_analysis' section."
        )

    error_analysis = report[
        "error_analysis"
    ]

    if "confusion" not in error_analysis:
        raise ValueError(
            "Missing 'confusion' section."
        )

    return error_analysis


def build_prioritized_confusions(
    error_analysis: dict,
) -> list:
    """
    Dynamically prioritize E6 TEST confusions.

    Priority is based on confusion count.
    """

    confusion = error_analysis[
        "confusion"
    ]

    prioritizer = ErrorPrioritizer()

    prioritized = (
        prioritizer.rank_confusions(
            confusion=confusion,
            criterion="count",
            top_k=TOP_K_CONFUSIONS,
        )
    )

    # --------------------------------------------------------
    # Attach actual TEST cases.
    #
    # HardExampleMiner expects:
    #
    # {
    #     "gt_class": ...,
    #     "pred_class": ...,
    #     "cases": [...]
    # }
    # --------------------------------------------------------

    for item in prioritized:

        gt_class = item[
            "gt_class"
        ]

        pred_class = item[
            "pred_class"
        ]

        pair_data = (
            confusion
            .get(gt_class, {})
            .get(pred_class, {})
        )

        item["cases"] = pair_data.get(
            "cases",
            [],
        )

    return prioritized


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
    logger.info("E6 HARD EXAMPLE MINING")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Load E6 error analysis
    # --------------------------------------------------------

    logger.info(
        "Loading E6 error analysis: %s",
        ERROR_ANALYSIS_PATH,
    )

    error_analysis = load_error_analysis(
        ERROR_ANALYSIS_PATH
    )

    confusion = error_analysis[
        "confusion"
    ]

    total_cases = sum(
        len(pair.get("cases", []))
        for gt_data in confusion.values()
        if isinstance(gt_data, dict)
        for pair in gt_data.values()
        if isinstance(pair, dict)
    )

    logger.info(
        "Total E6 confusion cases: %d",
        total_cases,
    )

    # --------------------------------------------------------
    # 2. Dynamic prioritization
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("Prioritizing E6 confusions")
    logger.info("=" * 70)

    prioritized_confusions = (
        build_prioritized_confusions(
            error_analysis
        )
    )

    if not prioritized_confusions:
        raise ValueError(
            "No prioritized E6 confusions found."
        )

    for rank, item in enumerate(
        prioritized_confusions,
        start=1,
    ):

        logger.info(
            "Priority #%d: %s → %s | "
            "count=%s | cases=%d",
            rank,
            item["gt_class"],
            item["pred_class"],
            item.get("count", "?"),
            len(item["cases"]),
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

    # Only GT classes of prioritized confusions.
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
    # 4. Create miner
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

    mining_result = (
        miner.mine_prioritized_confusions(
            prioritized_confusions=(
                prioritized_confusions
            ),
            top_k_per_case=TOP_K_PER_CASE,
            max_samples_per_confusion=(
                MAX_SAMPLES_PER_CONFUSION
            ),
        )
    )

    # --------------------------------------------------------
    # 6. Add E6 metadata
    # --------------------------------------------------------

    mining_result["experiment"] = "E6"

    mining_result["strategy"] = (
        "failure_pattern_guided_"
        "similarity_mining"
    )

    mining_result["configuration"] = {
        "top_k_confusions": TOP_K_CONFUSIONS,
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

    mining_result["source"] = {
        "error_analysis": str(
            ERROR_ANALYSIS_PATH
        ),
        "split": "test",
        "priority_source": (
            "ErrorPrioritizer.rank_confusions"
        ),
        "test_as_query": True,
        "train_as_reference": True,
        "test_samples_added_to_train": False,
    }

    # --------------------------------------------------------
    # 7. Save
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
    logger.info("E6 HARD EXAMPLE MINING COMPLETED")
    logger.info("=" * 70)

    logger.info(
        "Priority source: "
        "ErrorPrioritizer.rank_confusions"
    )

    logger.info(
        "Test samples added to train: NO"
    )

    for result in (
        mining_result["confusions"]
    ):

        logger.info(
            "Priority #%d: %s → %s | "
            "selected TRAIN samples: %d",
            result["rank"],
            result["gt_class"],
            result["pred_class"],
            result[
                "selected_train_samples"
            ],
        )

    logger.info(
        "Output: %s",
        output_path,
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()