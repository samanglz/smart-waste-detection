"""
Failure Pattern Extraction.

Converts prioritized validation errors produced by ModelEvaluator
into structured, actionable failure patterns.

Important design rule:
    Validation data is used only for failure-pattern discovery.

Validation samples must NEVER be passed to training or augmentation.
"""

from typing import Any, Dict, List, Optional


class FailurePatternExtractor:
    """
    Extract actionable failure patterns from prioritized errors.

    This class does NOT:
        - run model evaluation
        - perform prediction
        - analyze confusion
        - analyze false positives
        - analyze false negatives
        - perform prioritization

    Those responsibilities already belong to:
        ModelEvaluator
        ErrorAnalyzer
        ErrorPrioritizer

    This class only transforms their output into a unified
    failure-pattern representation.
    """

    def __init__(
        self,
        min_count: int = 1,
        top_k: Optional[int] = None,
    ):
        if min_count < 1:
            raise ValueError(
                "min_count must be >= 1."
            )

        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be None or > 0."
            )

        self.min_count = min_count
        self.top_k = top_k

    # ============================================================
    # PUBLIC API
    # ============================================================

    def extract(
        self,
        prioritized_errors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract failure patterns from ModelEvaluator.prioritize_errors().

        Args:
            prioritized_errors:
                Output of:

                    evaluator.prioritize_errors(
                        split="val"
                    )

        Returns:
            Structured failure-pattern report.
        """

        if not isinstance(
            prioritized_errors,
            dict,
        ):
            raise TypeError(
                "prioritized_errors must be a dictionary."
            )

        patterns: List[Dict[str, Any]] = []

        patterns.extend(
            self._extract_confusion_patterns(
                prioritized_errors.get(
                    "confusions",
                    [],
                )
            )
        )

        patterns.extend(
            self._extract_false_negative_patterns(
                prioritized_errors.get(
                    "false_negatives",
                    [],
                )
            )
        )

        patterns.extend(
            self._extract_false_positive_patterns(
                prioritized_errors.get(
                    "false_positives",
                    [],
                )
            )
        )

        patterns = self._sort_patterns(patterns)

        if self.top_k is not None:
            patterns = patterns[:self.top_k]

        return {
            "source_split": "val",
            "num_patterns": len(patterns),
            "patterns": patterns,
        }

    # ============================================================
    # CONFUSIONS
    # ============================================================

    def _extract_confusion_patterns(
        self,
        confusions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert prioritized confusion errors into patterns.

        Expected input item:

            {
                "gt_class": "plastic",
                "pred_class": "glass",
                "count": 8,
                ...
            }

        The exact field extraction is kept defensive because
        ErrorPrioritizer output may evolve.
        """

        patterns = []

        for item in confusions:

            count = self._get_count(item)

            if count < self.min_count:
                continue

            gt_class = (
                item.get("gt_class")
                or item.get("ground_truth_class")
                or item.get("name")
            )

            pred_class = (
                item.get("pred_class")
                or item.get("prediction_class")
            )

            if gt_class is None or pred_class is None:
                continue

            pattern = {
                "type": "confusion",
                "gt_class": gt_class,
                "pred_class": pred_class,
                "count": count,
                "priority_score": item.get(
                    "priority_score"
                ),
                "augmentation_target": gt_class,
                "source": "validation",
            }

            patterns.append(pattern)

        return patterns

    # ============================================================
    # FALSE NEGATIVES
    # ============================================================

    def _extract_false_negative_patterns(
        self,
        false_negatives: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert prioritized false-negative errors into patterns.
        """

        patterns = []

        for item in false_negatives:

            count = self._get_count(item)

            if count < self.min_count:
                continue

            class_name = (
                item.get("class_name")
                or item.get("gt_class")
                or item.get("name")
            )

            if class_name is None:
                continue

            pattern = {
                "type": "false_negative",
                "gt_class": class_name,
                "pred_class": None,
                "count": count,
                "priority_score": item.get(
                    "priority_score"
                ),
                "augmentation_target": class_name,
                "source": "validation",
            }

            patterns.append(pattern)

        return patterns

    # ============================================================
    # FALSE POSITIVES
    # ============================================================

    def _extract_false_positive_patterns(
        self,
        false_positives: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert prioritized false-positive errors into patterns.
        """

        patterns = []

        for item in false_positives:

            count = self._get_count(item)

            if count < self.min_count:
                continue

            class_name = (
                item.get("class_name")
                or item.get("pred_class")
                or item.get("name")
            )

            if class_name is None:
                continue

            pattern = {
                "type": "false_positive",
                "gt_class": None,
                "pred_class": class_name,
                "count": count,
                "priority_score": item.get(
                    "priority_score"
                ),
                "augmentation_target": class_name,
                "source": "validation",
            }

            patterns.append(pattern)

        return patterns

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _get_count(
        item: Dict[str, Any],
    ) -> int:
        """
        Safely extract an error count.
        """

        try:
            return int(
                item.get("count", 0)
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _sort_patterns(
        patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Sort patterns by priority score when available,
        otherwise by error count.
        """

        return sorted(
            patterns,
            key=lambda item: (
                item["priority_score"]
                if item["priority_score"] is not None
                else item["count"]
            ),
            reverse=True,
        )