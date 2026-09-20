"""
Run hard-case similarity analysis on validation failure patterns.

Validation error cases are used only as queries.
Training objects are used as the reference embedding index.

No dataset files are modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.failure_patterns.hard_case_similarity_analyzer import (
    HardCaseSimilarityAnalyzer,
)
from src.logging.logger import get_logger


logger = get_logger(__name__)


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return data


def main() -> None:

    logger.info("=" * 70)
    logger.info("HARD CASE SIMILARITY ANALYSIS")
    logger.info("=" * 70)

    # ============================================================
    # PATHS
    # ============================================================

    project_dir = Path(
        r"C:\Users\ASUS\Desktop\smart-waste-detection"
    )

    evaluation_dir = (
        project_dir
        / "outputs"
        / "yolo11m"
        / "E2_targeted_augmentation"
    )

    failure_patterns_path = (
        evaluation_dir
        / "val_failure_patterns.json"
    )

    output_dir = (
        evaluation_dir
        / "hard_case_similarity"
    )

    # ============================================================
    # DATASET PATHS
    # ============================================================

    train_images_dir = (
        project_dir
        / "data"
        / "processed"
        / "train"
        / "images"
    )

    train_labels_dir = (
        project_dir
        / "data"
        / "processed"
        / "train"
        / "labels"
    )

    # ============================================================
    # LOAD FAILURE PATTERNS
    # ============================================================

    failure_patterns = load_json(
        failure_patterns_path
    )

    confusion_patterns = failure_patterns.get(
        "confusion_patterns",
        [],
    )

    # ============================================================
    # FIND GLASS → PLASTIC
    # ============================================================

    target_pattern = None

    for item in confusion_patterns:

        if (
            item.get("source_class") == "glass"
            and item.get("target_class") == "plastic"
        ):
            target_pattern = item
            break
    
    
    if target_pattern is None:
        raise ValueError(
            "Could not find glass → plastic confusion pattern."
        )

    cases: List[Dict[str, Any]] = target_pattern.get(
        "cases",
        [],
    )

    logger.info(
        "Found glass → plastic pattern."
    )

    logger.info(
        "Validation cases: %d",
        len(cases),
    )

    if not cases:
        raise ValueError(
            "No validation cases found for glass → plastic."
        )

    # ============================================================
    # ANALYZER
    # ============================================================

    analyzer = HardCaseSimilarityAnalyzer(
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        class_names=CLASS_NAMES,
        device="cuda",
    )

    # ============================================================
    # BUILD TRAIN INDEX
    # ============================================================

    analyzer.build_train_index(
        target_classes=[
            "glass",
            "plastic",
        ]
    )

    # ============================================================
    # ANALYZE VALIDATION CASES
    # ============================================================

    results = analyzer.analyze_cases(
        cases=cases,
        top_k=5,
    )

    logger.info(
        "Successfully analyzed %d/%d cases.",
        len(results),
        len(cases),
    )

    # ============================================================
    # GENERATE REPORT
    # ============================================================

    report_path = analyzer.generate_report(
        results=results,
        output_dir=output_dir,
    )

    logger.info(
        "Report saved to: %s",
        report_path,
    )

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

