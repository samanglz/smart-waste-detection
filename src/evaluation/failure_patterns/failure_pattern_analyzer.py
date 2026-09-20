from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.failure_patterns.failure_pattern import (
    FailurePattern,
)


class FailurePatternAnalyzer:
    """
    Discover actionable failure patterns from a validation
    error report.

    IMPORTANT
    ---------
    This analyzer is used only for failure-pattern discovery.

    Validation data:
        - may be used to discover patterns
        - may be used to prioritize patterns

    Validation data:
        - must NEVER be copied into TRAIN
        - must NEVER become a training sample

    Test data:
        - must NEVER be used for failure-pattern discovery
    """

    def __init__(
        self,
        class_names: List[str],
        min_confusion_count: int = 2,
        min_fn_count: int = 2,
        min_fp_count: int = 2,
    ):
        self.class_names = list(
            class_names
        )

        self.min_confusion_count = (
            min_confusion_count
        )

        self.min_fn_count = (
            min_fn_count
        )

        self.min_fp_count = (
            min_fp_count
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def analyze(
        self,
        error_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Discover failure patterns from an error report.

        The expected report structure is the output of
        ModelEvaluator.analyze_errors().
        """

        self._validate_report(
            error_report
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

        confusion_patterns = (
            self._analyze_confusions(
                confusion
            )
        )

        fn_patterns = (
            self._analyze_false_negatives(
                false_negatives
            )
        )

        fp_patterns = (
            self._analyze_false_positives(
                false_positives
            )
        )

        prioritized_patterns = (
            self._build_prioritized_patterns(
                confusion_patterns,
                fn_patterns,
                fp_patterns,
            )
        )

        class_priority = (
            self._build_class_priority(
                confusion_patterns,
                fn_patterns,
                fp_patterns,
            )
        )

        return {
            "confusion_patterns": (
                confusion_patterns
            ),
            "false_negative_patterns": (
                fn_patterns
            ),
            "false_positive_patterns": (
                fp_patterns
            ),
            "prioritized_patterns": (
                prioritized_patterns
            ),
            "class_priority": (
                class_priority
            ),
        }

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_report(
        self,
        error_report: Dict[str, Any],
    ) -> None:
        """
        Validate the basic structure of the error report.

        This method intentionally does not require a particular
        report schema beyond the error_analysis section because
        older evaluator reports may differ slightly.

        The VAL-only restriction is enforced by the runner.
        """

        if not isinstance(
            error_report,
            dict,
        ):
            raise ValueError(
                "error_report must be a dictionary."
            )

        error_analysis = error_report.get(
            "error_analysis"
        )

        if error_analysis is None:
            raise ValueError(
                "Invalid error report: "
                "'error_analysis' section is missing."
            )

        if not isinstance(
            error_analysis,
            dict,
        ):
            raise ValueError(
                "'error_analysis' must be a dictionary."
            )

    # ============================================================
    # CONFUSIONS
    # ============================================================

    def _analyze_confusions(
        self,
        confusion: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Discover class-to-class confusion patterns.

        Example:

            GT glass → predicted plastic

        becomes:

            FailurePattern(
                error_type="confusion",
                source_class="glass",
                target_class="plastic",
            )
        """

        patterns: List[Dict[str, Any]] = []

        for gt_class, predictions in confusion.items():

            if not isinstance(
                predictions,
                dict,
            ):
                continue

            for pred_class, data in predictions.items():

                if not isinstance(
                    data,
                    dict,
                ):
                    continue

                count = int(
                    data.get(
                        "count",
                        0,
                    )
                )

                if count < self.min_confusion_count:
                    continue

                mean_iou = float(
                    data.get(
                        "mean_iou",
                        0.0,
                    )
                )

                mean_confidence = float(
                    data.get(
                        "mean_confidence",
                        0.0,
                    )
                )

                images = list(
                    data.get(
                        "images",
                        [],
                    )
                )

                cases = list(
                    data.get(
                        "cases",
                        [],
                    )
                )

                pattern = FailurePattern(
                    pattern_id=(
                        f"confusion_"
                        f"{gt_class}_to_"
                        f"{pred_class}"
                    ),
                    error_type="confusion",
                    source_class=gt_class,
                    target_class=pred_class,
                    min_count=(
                        self.min_confusion_count
                    ),
                )

                patterns.append(
                    self._pattern_to_dict(
                        pattern,
                        count=count,
                        mean_iou=mean_iou,
                        mean_confidence=(
                            mean_confidence
                        ),
                        images=images,
                        cases=cases,
                    )
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
        Discover false-negative patterns.

        Example:

            glass → missed detection
        """

        patterns: List[Dict[str, Any]] = []

        for class_name, data in false_negatives.items():

            if not isinstance(
                data,
                dict,
            ):
                continue

            count = int(
                data.get(
                    "count",
                    0,
                )
            )

            if count < self.min_fn_count:
                continue

            images = list(
                data.get(
                    "images",
                    [],
                )
            )

            pattern = FailurePattern(
                pattern_id=(
                    f"false_negative_"
                    f"{class_name}"
                ),
                error_type="false_negative",
                source_class=class_name,
                target_class=None,
                min_count=self.min_fn_count,
            )

            patterns.append(
                self._pattern_to_dict(
                    pattern,
                    count=count,
                    images=images,
                    cases=[],
                )
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
        Discover false-positive patterns.

        Example:

            predicted plastic incorrectly
        """

        patterns: List[Dict[str, Any]] = []

        for class_name, data in false_positives.items():

            if not isinstance(
                data,
                dict,
            ):
                continue

            count = int(
                data.get(
                    "count",
                    0,
                )
            )

            if count < self.min_fp_count:
                continue

            mean_confidence = float(
                data.get(
                    "mean_confidence",
                    0.0,
                )
            )

            images = list(
                data.get(
                    "images",
                    [],
                )
            )

            pattern = FailurePattern(
                pattern_id=(
                    f"false_positive_"
                    f"{class_name}"
                ),
                error_type="false_positive",
                source_class=class_name,
                target_class=None,
                min_count=self.min_fp_count,
            )

            patterns.append(
                self._pattern_to_dict(
                    pattern,
                    count=count,
                    mean_confidence=(
                        mean_confidence
                    ),
                    images=images,
                    cases=[],
                )
            )

        patterns.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        return patterns

    # ============================================================
    # PRIORITIZATION
    # ============================================================

    def _build_prioritized_patterns(
        self,
        confusion_patterns: List[Dict[str, Any]],
        fn_patterns: List[Dict[str, Any]],
        fp_patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build a unified ranking of failure patterns.

        Priority philosophy:

            confusion      → highest
            false negative → high
            false positive → lower

        Count remains the main signal.

        This priority is NOT a model metric.
        It is only used to decide which failure patterns
        deserve further investigation / targeted augmentation.
        """

        candidates: List[Dict[str, Any]] = []

        # --------------------------------------------------------
        # Confusion
        # --------------------------------------------------------

        for pattern in confusion_patterns:

            count = pattern["count"]

            priority_score = (
                2.0 * count
            )

            item = dict(pattern)

            item["priority_score"] = (
                priority_score
            )

            candidates.append(item)

        # --------------------------------------------------------
        # False negatives
        # --------------------------------------------------------

        for pattern in fn_patterns:

            count = pattern["count"]

            priority_score = (
                2.0 * count
            )

            item = dict(pattern)

            item["priority_score"] = (
                priority_score
            )

            candidates.append(item)

        # --------------------------------------------------------
        # False positives
        # --------------------------------------------------------

        for pattern in fp_patterns:

            count = pattern["count"]

            priority_score = (
                1.0 * count
            )

            item = dict(pattern)

            item["priority_score"] = (
                priority_score
            )

            candidates.append(item)

        # --------------------------------------------------------
        # Sort
        # --------------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item["priority_score"],
                item["count"],
            ),
            reverse=True,
        )

        # --------------------------------------------------------
        # Assign deterministic priority rank
        # --------------------------------------------------------

        for priority, pattern in enumerate(
            candidates,
            start=1,
        ):
            pattern["priority"] = priority

        return candidates

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
        Build class-level failure priority.

        This is not a model metric.

        It is an auxiliary signal for deciding which classes
        deserve targeted investigation.
        """

        scores: Dict[
            str,
            Dict[str, int],
        ] = {}

        def ensure_class(
            class_name: str,
        ) -> None:

            if class_name not in scores:
                scores[class_name] = {
                    "confusion": 0,
                    "false_negative": 0,
                    "false_positive": 0,
                }

        # --------------------------------------------------------
        # Confusions
        # --------------------------------------------------------

        for pattern in confusion_patterns:

            class_name = pattern[
                "source_class"
            ]

            ensure_class(
                class_name
            )

            scores[class_name][
                "confusion"
            ] += pattern["count"]

        # --------------------------------------------------------
        # False negatives
        # --------------------------------------------------------

        for pattern in fn_patterns:

            class_name = pattern[
                "source_class"
            ]

            ensure_class(
                class_name
            )

            scores[class_name][
                "false_negative"
            ] += pattern["count"]

        # --------------------------------------------------------
        # False positives
        # --------------------------------------------------------

        for pattern in fp_patterns:

            class_name = pattern[
                "source_class"
            ]

            ensure_class(
                class_name
            )

            scores[class_name][
                "false_positive"
            ] += pattern["count"]

        # --------------------------------------------------------
        # Weighted score
        # --------------------------------------------------------

        result: List[
            Dict[str, Any]
        ] = []

        for class_name, values in scores.items():

            score = (
                2.0
                * values["confusion"]
                + 2.0
                * values["false_negative"]
                + 1.0
                * values["false_positive"]
            )

            result.append(
                {
                    "class": class_name,
                    "score": score,
                    "confusion_count": (
                        values["confusion"]
                    ),
                    "false_negative_count": (
                        values["false_negative"]
                    ),
                    "false_positive_count": (
                        values["false_positive"]
                    ),
                }
            )

        result.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return result

    # ============================================================
    # SERIALIZATION
    # ============================================================

    @staticmethod
    def _pattern_to_dict(
        pattern: FailurePattern,
        count: int,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Convert FailurePattern into a JSON-compatible dictionary.
        """

        result = {
            "pattern_id": pattern.pattern_id,
            "type": pattern.error_type,
            "error_type": pattern.error_type,
            "source_class": pattern.source_class,
            "target_class": pattern.target_class,
            "min_count": pattern.min_count,
            "count": count,
        }

        result.update(
            extra
        )

        return result

