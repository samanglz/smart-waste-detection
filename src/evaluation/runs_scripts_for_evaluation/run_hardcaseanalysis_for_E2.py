#!/usr/bin/env python

"""
E2 — Dynamic Hard Case Similarity Analysis

Pipeline
--------
E2 error_analysis.json
        |
        v
    ErrorPrioritizer
        |
        v
 Top-K prioritized confusions
        |
        v
HardCaseSimilarityAnalyzer
        |
        v
Embedding-based similarity analysis
        |
        v
Hard-case reports

IMPORTANT
---------
- Priority confusions are NOT hard-coded.
- Existing E2 error_analysis.json is reused.
- No model training is performed.
- No new evaluation/inference is performed.
- The analyzer only investigates the dynamically
  prioritized confusion cases.
"""

import json
import sys
from pathlib import Path

from src.evaluation.hard_case_detect.hard_case_similarity import (
    HardCaseSimilarityAnalyzer,
)

from src.logging.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

E2_ERROR_ANALYSIS = Path(
    "outputs/yolo11m/E2_targeted_augmentation/error_analysis.json"
)

TRAIN_IMAGES_DIR = Path(
    "data/processed/train/images"
)

TRAIN_LABELS_DIR = Path(
    "data/processed/train/labels"
)

OUTPUT_DIR = Path(
    "outputs/hard_case_analysis/E2"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

TOP_K = 5

NEIGHBORS_TOP_K = 5

DEVICE = "cuda"


# ============================================================
# ERROR PRIORITIZER
# ============================================================

def _load_error_prioritizer():
    """
    Load the existing ErrorPrioritizer used by
    ModelEvaluator_copy.

    The fallback import makes the run script tolerant to
    the exact module location used in the framework.
    """

    try:
        from src.evaluation.prioritization.error_prioritizer import ( 
            ErrorPrioritizer,
                                                                    
            )

        return ErrorPrioritizer

    except ImportError:

        try:
            from src.evaluation import ErrorPrioritizer

            return ErrorPrioritizer

        except ImportError as exc:

            raise ImportError(
                "Could not import ErrorPrioritizer.\n"
                "Use the same ErrorPrioritizer class that is "
                "used by ModelEvaluator_copy.prioritize_errors()."
            ) from exc


# ============================================================
# LOAD ERROR ANALYSIS
# ============================================================

def load_error_analysis() -> dict:
    """
    Load the existing E2 error-analysis report.
    """

    if not E2_ERROR_ANALYSIS.exists():

        raise FileNotFoundError(
            "E2 error_analysis.json was not found:\n"
            f"{E2_ERROR_ANALYSIS}"
        )

    logger.info(
        "Loading E2 error analysis: %s",
        E2_ERROR_ANALYSIS,
    )

    with E2_ERROR_ANALYSIS.open(
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    if "error_analysis" not in report:

        raise ValueError(
            "Invalid error analysis structure: "
            "'error_analysis' key was not found."
        )

    return report["error_analysis"]


# ============================================================
# DYNAMIC PRIORITIZATION
# ============================================================

def prioritize_confusions(
    error_analysis: dict,
) -> list:
    """
    Dynamically rank confusion cases using the same
    ErrorPrioritizer used by ModelEvaluator_copy.

    Criterion:
        count

    No confusion pair is hard-coded.
    """

    ErrorPrioritizer = _load_error_prioritizer()

    prioritizer = ErrorPrioritizer()

    confusion = error_analysis.get(
        "confusion",
        {},
    )

    prioritized = prioritizer.rank_confusions(
        confusion=confusion,
        criterion="count",
        top_k=TOP_K,
    )

    return prioritized


# ============================================================
# BUILD TRAIN EMBEDDING INDEX
# ============================================================

def build_analyzer() -> HardCaseSimilarityAnalyzer:
    """
    Initialize analyzer and build the training embedding index.
    """

    if not TRAIN_IMAGES_DIR.exists():

        raise FileNotFoundError(
            "Training images directory not found:\n"
            f"{TRAIN_IMAGES_DIR}"
        )

    if not TRAIN_LABELS_DIR.exists():

        raise FileNotFoundError(
            "Training labels directory not found:\n"
            f"{TRAIN_LABELS_DIR}"
        )

    logger.info(
        "Initializing HardCaseSimilarityAnalyzer..."
    )

    analyzer = HardCaseSimilarityAnalyzer(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        class_names=CLASS_NAMES,
        device=DEVICE,
    )

    logger.info(
        "Building training embedding index..."
    )

    analyzer.build_train_index(
        CLASS_NAMES
    )

    logger.info(
        "Training embedding index built successfully."
    )

    return analyzer


# ============================================================
# ANALYZE ONE PRIORITY CONFUSION
# ============================================================

def analyze_confusion(
    analyzer: HardCaseSimilarityAnalyzer,
    confusion_data: dict,
    item: dict,
    rank: int,
) -> dict | None:

    gt_class = item["gt_class"]
    pred_class = item["pred_class"]

    logger.info("=" * 70)
    logger.info(
        "Priority #%d: %s → %s",
        rank,
        gt_class,
        pred_class,
    )
    logger.info("=" * 70)

    cases = (
        confusion_data
        .get(gt_class, {})
        .get(pred_class, {})
        .get("cases", [])
    )

    if not cases:

        logger.warning(
            "No cases found for %s → %s",
            gt_class,
            pred_class,
        )

        return None

    logger.info(
        "Cases found: %d",
        len(cases),
    )

    results = analyzer.analyze_cases(
        cases,
        top_k=NEIGHBORS_TOP_K,
    )

    output_dir = (
        OUTPUT_DIR
        / f"{gt_class}_to_{pred_class}"
    )

    report_path = analyzer.generate_report(
        results,
        output_dir,
    )

    nearest_counts = {}

    margins = []

    for result in results:

        nearest = result.get(
            "nearest_class"
        )

        if nearest:

            nearest_counts[nearest] = (
                nearest_counts.get(
                    nearest,
                    0,
                )
                + 1
            )

        margin = result.get(
            "margin"
        )

        if margin is not None:

            margins.append(
                margin
            )

    return {
        "rank": rank,
        "gt_class": gt_class,
        "pred_class": pred_class,
        "total_cases": len(cases),
        "analyzed_cases": len(results),
        "nearest_class_counts": nearest_counts,
        "average_margin": (
            sum(margins) / len(margins)
            if margins
            else None
        ),
        "report": str(report_path),
    }


# ============================================================
# SAVE GLOBAL SUMMARY
# ============================================================

def save_summary(
    prioritized_confusions: list,
    analyzed_results: list,
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "priority_summary.json"
    )

    summary = {
        "source": {
            "error_analysis": str(
                E2_ERROR_ANALYSIS
            ),
            "prioritization": (
                "ErrorPrioritizer.rank_confusions"
            ),
            "criterion": "count",
            "top_k": TOP_K,
        },
        "priority_confusions": (
            prioritized_confusions
        ),
        "analyzed": analyzed_results,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info(
        "E2 — DYNAMIC HARD CASE SIMILARITY ANALYSIS"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Load existing error analysis
    # --------------------------------------------------------

    error_analysis = load_error_analysis()

    confusion_data = error_analysis.get(
        "confusion",
        {},
    )

    # --------------------------------------------------------
    # 2. Dynamic prioritization
    # --------------------------------------------------------

    logger.info("-" * 70)
    logger.info(
        "Running dynamic confusion prioritization..."
    )
    logger.info("-" * 70)

    prioritized_confusions = prioritize_confusions(
        error_analysis
    )

    if not prioritized_confusions:

        logger.warning(
            "No prioritized confusions were found."
        )

        return

    logger.info(
        "Top-%d prioritized confusions:",
        len(prioritized_confusions),
    )

    for rank, item in enumerate(
        prioritized_confusions,
        start=1,
    ):

        logger.info(
            "  %d. %s → %s | count=%s",
            rank,
            item["gt_class"],
            item["pred_class"],
            item.get("count"),
        )

    # --------------------------------------------------------
    # 3. Build embedding analyzer
    # --------------------------------------------------------

    analyzer = build_analyzer()

    # --------------------------------------------------------
    # 4. Analyze priority cases
    # --------------------------------------------------------

    analyzed_results = []

    for rank, item in enumerate(
        prioritized_confusions,
        start=1,
    ):

        try:

            result = analyze_confusion(
                analyzer=analyzer,
                confusion_data=confusion_data,
                item=item,
                rank=rank,
            )

            if result is not None:

                analyzed_results.append(
                    result
                )

        except Exception as exc:

            logger.exception(
                "Failed to analyze %s → %s: %s",
                item["gt_class"],
                item["pred_class"],
                exc,
            )

    # --------------------------------------------------------
    # 5. Save global summary
    # --------------------------------------------------------

    summary_path = save_summary(
        prioritized_confusions,
        analyzed_results,
    )

    # --------------------------------------------------------
    # 6. Final report
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info(
        "HARD CASE ANALYSIS COMPLETED"
    )
    logger.info("=" * 70)

    logger.info(
        "Priority source:"
    )

    logger.info(
        "  ErrorPrioritizer.rank_confusions()"
    )

    logger.info(
        "Hard-coded priority list: NONE"
    )

    logger.info(
        "Analyzed priority cases: %d",
        len(analyzed_results),
    )

    logger.info(
        "Global summary: %s",
        summary_path,
    )

    logger.info(
        "Output directory: %s",
        OUTPUT_DIR,
    )

    logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        logger.info(
            "Hard case analysis interrupted."
        )

        sys.exit(1)

    except Exception as exc:

        logger.exception(
            "Hard case analysis failed: %s",
            exc,
        )

        sys.exit(1)