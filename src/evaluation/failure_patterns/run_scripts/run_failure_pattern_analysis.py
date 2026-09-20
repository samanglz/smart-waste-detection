"""
Run failure-pattern analysis on the VALIDATION error report.

Pipeline:

    VAL error report
            ↓
    FailurePatternAnalyzer
            ↓
    val_failure_patterns.json

IMPORTANT
---------
- Validation data is used only for failure-pattern discovery.
- Test data must never be used here.
- This script does not modify the dataset.
- This script does not perform hard-example mining.
- This script does not copy validation samples into TRAIN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.evaluation.failure_patterns.failure_pattern_analyzer import (
    FailurePatternAnalyzer,
)
from src.logging.logger import get_logger


logger = get_logger(__name__)


DEFAULT_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


# ============================================================
# IO
# ============================================================


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Error report not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Error report must contain a JSON object."
        )

    return data


def save_json(
    data: Dict[str, Any],
    path: Path,
) -> None:
    """Save analysis result as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Validation
# ============================================================


def validate_val_report(
    error_report: Dict[str, Any],
) -> None:
    """
    Validate that the supplied report belongs to VAL.

    We intentionally fail instead of silently accepting
    an unknown split because this component must never
    mine failure patterns from TEST.
    """

    dataset = error_report.get(
        "dataset",
        {},
    )

    split = str(
        dataset.get(
            "split",
            "",
        )
    ).lower()

    if split and split not in {
        "val",
        "validation",
    }:
        raise ValueError(
            "Failure-pattern analysis requires a VAL "
            f"error report, but received split='{split}'."
        )

    # Some older evaluator reports may not contain
    # dataset.split. In that case we allow the report,
    # but make the assumption explicit in the output.
    if not split:
        logger.warning(
            "Dataset split is not explicitly stored "
            "in the error report."
        )


# ============================================================
# Analysis
# ============================================================


def run_analysis(
    error_report_path: Path,
    output_path: Path,
    class_names: list[str],
) -> Path:
    """
    Run failure-pattern analysis on a VAL error report.
    """

    logger.info("=" * 70)
    logger.info("FAILURE PATTERN ANALYSIS")
    logger.info("=" * 70)

    logger.info(
        "Input error report: %s",
        error_report_path,
    )

    error_report = load_json(
        error_report_path
    )

    validate_val_report(
        error_report
    )

    logger.info(
        "Confirmed analysis source: VALIDATION"
    )

    analyzer = FailurePatternAnalyzer(
        class_names=class_names,
    )

    result = analyzer.analyze(
        error_report
    )

    # Add explicit provenance information.
    result["source"] = {
        "dataset_split": "val",
        "input_error_report": str(
            error_report_path.resolve()
        ),
        "test_data_used": False,
        "training_data_modified": False,
        "validation_samples_added_to_train": False,
    }

    save_json(
        result,
        output_path,
    )

    _log_summary(result)

    logger.info(
        "Failure-pattern report saved to: %s",
        output_path,
    )

    return output_path


# ============================================================
# Summary
# ============================================================


def _log_summary(
    result: Dict[str, Any],
) -> None:
    """Log a concise analysis summary."""

    confusion_patterns = result.get(
        "confusion_patterns",
        [],
    )

    fn_patterns = result.get(
        "false_negative_patterns",
        [],
    )

    fp_patterns = result.get(
        "false_positive_patterns",
        [],
    )

    prioritized = result.get(
        "prioritized_patterns",
        [],
    )

    logger.info("-" * 70)

    logger.info(
        "Confusion patterns: %d",
        len(confusion_patterns),
    )

    logger.info(
        "False-negative patterns: %d",
        len(fn_patterns),
    )

    logger.info(
        "False-positive patterns: %d",
        len(fp_patterns),
    )

    logger.info(
        "Prioritized patterns: %d",
        len(prioritized),
    )

    logger.info("-" * 70)

    if prioritized:

        logger.info(
            "Top failure patterns:"
        )

        for pattern in prioritized[:10]:

            logger.info(
                "  #%d | %s | %s → %s | count=%d",
                pattern.get(
                    "priority",
                    0,
                ),
                pattern.get(
                    "pattern_id",
                    "unknown",
                ),
                pattern.get(
                    "source_class",
                    "unknown",
                ),
                pattern.get(
                    "target_class",
                    "-",
                ),
                pattern.get(
                    "count",
                    0,
                ),
            )

    logger.info("=" * 70)




# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run failure-pattern analysis on the VAL error report."""

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    evaluation_dir = Path(
        r"C:\Users\ASUS\Desktop\smart-waste-detection"
        r"\outputs\yolo11m\E2_targeted_augmentation"
    )

    error_report_path = (
        evaluation_dir / "val_error_analysis.json"
    )

    output_path = (
        evaluation_dir / "val_failure_patterns.json"
    )

    # ---------------------------------------------------------
    # Class names
    # ---------------------------------------------------------

    class_names = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    ]

    # ---------------------------------------------------------
    # Run analysis
    # ---------------------------------------------------------

    run_analysis(
        error_report_path=error_report_path,
        output_path=output_path,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()



