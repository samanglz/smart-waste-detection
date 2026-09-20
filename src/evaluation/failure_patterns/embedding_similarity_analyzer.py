from collections import defaultdict
from typing import Dict, Any, List

from src.evaluation.failure_patterns.failure_pattern import (
    FailurePattern,
)


class FailurePatternAnalyzer:
    """
    Discover actionable failure patterns from validation errors.

    IMPORTANT
    ---------
    This analyzer is intended for validation-set analysis only.

    Validation data is used to discover failure patterns.
    Validation images must NEVER be copied into the training dataset.

    The analyzer does not perform:
        - embedding extraction
        - similarity search
        - training sample selection
        - augmentation

    It only converts the validation error report into structured
    FailurePattern objects.
    """

    def __init__(
        self,
        class_names: List[str],
    ):
        self.class_names = list(class_names)

    # ============================================================
    # PUBLIC API
    # ============================================================

    def analyze(
        self,
        error_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Discover failure patterns from a validation error report.

        Args:
            error_report:
                Output produced by ModelEvaluator.analyze_errors().

        Returns:
            Structured failure-pattern analysis.
        """

        if error_report.get("split") != "val":
            raise ValueError(
                "FailurePatternAnalyzer must operate on "
                "validation data only."
            )

        error_analysis = error_report.get(
            "error_analysis",
            {},
        )

        confusion = error_analysis.get(
            "confusion",
            {},
        )

        false_positives = error_analysis.get(
            "false_positives",
            {},
        )

        false_negatives = error_analysis.get(
            "false_negatives",
            {},
        )

        confusion_patterns = self._analyze_confusions(
            confusion
        )

        false_negative_patterns = (
            self._analyze_false_negatives(
                false_negatives
            )
        )

        false_positive_patterns = (
            self._analyze_false_positives(
                false_positives
            )
        )

        class_priority = self._build_class_priority(
            confusion_patterns,
            false_negative_patterns,
            false_positive_patterns,
        )

        return {
            "split": "val",
            "confusion_patterns": confusion_patterns,
            "false_negative_patterns": false_negative_patterns,
            "false_positive_patterns": false_positive_patterns,
            "class_priority": class_priority,
        }

    # ============================================================
    # CONFUSIONS
    # ============================================================

    def _analyze_confusions(
        self,
        confusion: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert validation confusion statistics into patterns.
        """

        patterns = []

        for gt_class, predictions in confusion.items():

            for pred_class, data in predictions.items():

                count = int(
                    data.get("count", 0)
                )

                if count <= 0:
                    continue

                pattern = FailurePattern(
                    pattern_id=(
                        f"confusion_"
                        f"{gt_class}_to_{pred_class}"
                    ),
                    error_type="confusion",
                    source_class=gt_class,
                    target_class=pred_class,
                    min_count=count,
                    priority=0,
                )

                patterns.append(
                    {
                        "pattern": pattern,
                        "count": count,
                        "mean_iou": float(
                            data.get(
                                "mean_iou",
                                0.0,
                            )
                        ),
                        "mean_confidence": float(
                            data.get(
                                "mean_confidence",
                                0.0,
                            )
                        ),
                        "images": list(
                            data.get(
                                "images",
                                [],
                            )
                        ),
                        "cases": list(
                            data.get(
                                "cases",
                                [],
                            )
                        ),
                    }
                )

        patterns.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        return patterns

    # ============================================================
    # FALSE NEGATIVES
    # ============================================================

    def _analyze_false_negatives(
        self,
        false_negatives: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert validation false-negative statistics into patterns.
        """

        patterns = []

        for class_name, data in false_negatives.items():

            count = int(
                data.get("count", 0)
            )

            if count <= 0:
                continue

            pattern = FailurePattern(
                pattern_id=f"false_negative_{class_name}",
                error_type="false_negative",
                source_class=class_name,
                target_class=None,
                min_count=count,
                priority=0,
            )

            patterns.append(
                {
                    "pattern": pattern,
                    "count": count,
                    "images": list(
                        data.get(
                            "images",
                            [],
                        )
                    ),
                }
            )

        patterns.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        return patterns

    # ============================================================
    # FALSE POSITIVES
    # ============================================================

    def _analyze_false_positives(
        self,
        false_positives: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert validation false-positive statistics into patterns.
        """

        patterns = []

        for class_name, data in false_positives.items():

            count = int(
                data.get("count", 0)
            )

            if count <= 0:
                continue

            pattern = FailurePattern(
                pattern_id=f"false_positive_{class_name}",
                error_type="false_positive",
                source_class=class_name,
                target_class=None,
                min_count=count,
                priority=0,
            )

            patterns.append(
                {
                    "pattern": pattern,
                    "count": count,
                    "mean_confidence": float(
                        data.get(
                            "mean_confidence",
                            0.0,
                        )
                    ),
                    "images": list(
                        data.get(
                            "images",
                            [],
                        )
                    ),
                }
            )

        patterns.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        return patterns

    # ============================================================
    # CLASS PRIORITY
    # ============================================================

    def _build_class_priority(
        self,
        confusion_patterns: List[Dict[str, Any]],
        fn_patterns: List[Dict[str, Any]],
        fp_patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build a class-level failure priority score.

        This score is NOT a model metric.

        It is only used to identify classes that deserve
        targeted investigation or augmentation.
        """

        scores = defaultdict(
            lambda: {
                "confusion": 0,
                "false_negative": 0,
                "false_positive": 0,
            }
        )

        # --------------------------------------------------------
        # Confusion contribution
        # --------------------------------------------------------

        for item in confusion_patterns:

            pattern = item["pattern"]

            scores[
                pattern.source_class
            ]["confusion"] += item["count"]

        # --------------------------------------------------------
        # False-negative contribution
        # --------------------------------------------------------

        for item in fn_patterns:

            pattern = item["pattern"]

            scores[
                pattern.source_class
            ]["false_negative"] += item["count"]

        # --------------------------------------------------------
        # False-positive contribution
        # --------------------------------------------------------

        for item in fp_patterns:

            pattern = item["pattern"]

            scores[
                pattern.source_class
            ]["false_positive"] += item["count"]

        # --------------------------------------------------------
        # Weighted score
        # --------------------------------------------------------

        result = []

        for class_name, values in scores.items():

            score = (
                2.0 * values["confusion"]
                + 2.0 * values["false_negative"]
                + 1.0 * values["false_positive"]
            )

            result.append(
                {
                    "class": class_name,
                    "score": score,
                    "confusion_count": values[
                        "confusion"
                    ],
                    "false_negative_count": values[
                        "false_negative"
                    ],
                    "false_positive_count": values[
                        "false_positive"
                    ],
                }
            )

        result.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return result

