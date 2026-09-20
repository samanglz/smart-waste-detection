from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


class ErrorAnalysisComparator:
    """
    Production-ready comparator for ModelEvaluator error_analysis.json files.

    Designed for:
        E2 vs E6

    Main responsibilities
    ---------------------
    1. Compare per-class metrics.
    2. Compare confusion aggregates.
    3. Compare false-positive aggregates.
    4. Compare false-negative aggregates.
    5. Perform object-level matching for confusion cases.
    6. Detect persistent confusions.
    7. Detect prediction changes for the same GT object.
    8. Detect newly appearing / disappearing confusion cases.
    9. Compare IoU and confidence distributions.
    10. Highlight class-specific regressions, especially Metal.
    11. Produce machine-readable JSON and human-readable TXT reports.

    Important methodological note
    ------------------------------
    This comparator does NOT assume that:
        "confusion disappeared" == "object became correct".

    If an E2 confusion case cannot be found as a confusion in E6,
    the result is reported conservatively as:

        "confusion_resolved_or_changed"

    It may have become:
        - correct,
        - false negative,
        - another error type,
        - or an unmatched case.

    Exact correct -> wrong transitions require a complete prediction
    inventory, not only error_analysis.json.
    """

    DEFAULT_GT_BBOX_IOU_THRESHOLD = 0.95
    DEFAULT_HIGH_IOU_THRESHOLD = 0.80
    DEFAULT_LOW_IOU_THRESHOLD = 0.50

    ERROR_TYPES = (
        "confusion",
        "false_positive",
        "false_negative",
    )

    def __init__(
        self,
        baseline_path: Path | str,
        experiment_path: Path | str,
        baseline_name: str = "E2",
        experiment_name: str = "E6",
        gt_bbox_iou_threshold: float = DEFAULT_GT_BBOX_IOU_THRESHOLD,
        high_iou_threshold: float = DEFAULT_HIGH_IOU_THRESHOLD,
        low_iou_threshold: float = DEFAULT_LOW_IOU_THRESHOLD,
        focus_classes: Optional[List[str]] = None,
    ):
        self.baseline_path = Path(baseline_path)
        self.experiment_path = Path(experiment_path)

        self.baseline_name = baseline_name
        self.experiment_name = experiment_name

        self.gt_bbox_iou_threshold = gt_bbox_iou_threshold
        self.high_iou_threshold = high_iou_threshold
        self.low_iou_threshold = low_iou_threshold

        self.focus_classes = focus_classes or [
            "metal",
            "glass",
            "plastic",
            "paper",
            "cardboard",
        ]

        self.baseline = self._load_json(self.baseline_path)
        self.experiment = self._load_json(self.experiment_path)

        self._validate_document(
            self.baseline,
            self.baseline_name,
            self.baseline_path,
        )
        self._validate_document(
            self.experiment,
            self.experiment_name,
            self.experiment_path,
        )

    # =========================================================================
    # JSON / VALIDATION
    # =========================================================================

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Root JSON object must be a dictionary: {path}"
            )

        return data

    @staticmethod
    def _validate_document(
        data: Dict[str, Any],
        name: str,
        path: Path,
    ) -> None:
        required_top_level = [
            "split",
            "per_class_metrics",
            "class_performance_summary",
            "error_analysis",
        ]

        missing = [
            key for key in required_top_level
            if key not in data
        ]

        if missing:
            raise ValueError(
                f"{name} is missing required keys {missing}: {path}"
            )

        if not isinstance(data["error_analysis"], dict):
            raise ValueError(
                f"{name}.error_analysis must be a dictionary: {path}"
            )

        if data["split"] != "test":
            LOGGER.warning(
                "%s split is '%s', not 'test'. "
                "For E2/E6 final comparison this should normally be 'test'.",
                name,
                data["split"],
            )

    # =========================================================================
    # BASIC HELPERS
    # =========================================================================

    @staticmethod
    def _image_name(path: Optional[str]) -> Optional[str]:
        if not path:
            return None

        normalized = str(path).replace("\\", "/")
        return Path(normalized).name

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            value = float(value)

            if not math.isfinite(value):
                return None

            return value
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bbox_valid(bbox: Any) -> bool:
        if not isinstance(bbox, (list, tuple)):
            return False

        if len(bbox) != 4:
            return False

        values = []

        for value in bbox:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False

            if not math.isfinite(value):
                return False

            values.append(value)

        x1, y1, x2, y2 = values

        return (
            x2 >= x1
            and y2 >= y1
        )

    @staticmethod
    def _normalize_bbox(
        bbox: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        if not ErrorAnalysisComparator._bbox_valid(bbox):
            return None

        x1, y1, x2, y2 = map(float, bbox)

        return x1, y1, x2, y2

    @staticmethod
    def _bbox_iou(
        box_a: Any,
        box_b: Any,
    ) -> float:
        a = ErrorAnalysisComparator._normalize_bbox(box_a)
        b = ErrorAnalysisComparator._normalize_bbox(box_b)

        if a is None or b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)

        intersection = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    # =========================================================================
    # METRICS
    # =========================================================================

    @staticmethod
    def _extract_metrics(
        data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        raw = data.get("per_class_metrics", {})

        if not isinstance(raw, dict):
            return {}

        result = {}

        for class_name, metrics in raw.items():
            if not isinstance(metrics, dict):
                continue

            result[class_name] = {
                "precision": ErrorAnalysisComparator._safe_float(
                    metrics.get("precision")
                ),
                "recall": ErrorAnalysisComparator._safe_float(
                    metrics.get("recall")
                ),
                "ap": ErrorAnalysisComparator._safe_float(
                    metrics.get("ap")
                ),
            }

        return result

    def _compare_metrics(self) -> Dict[str, Any]:
        baseline_metrics = self._extract_metrics(self.baseline)
        experiment_metrics = self._extract_metrics(self.experiment)

        classes = sorted(
            set(baseline_metrics.keys())
            | set(experiment_metrics.keys())
        )

        comparison = []

        for class_name in classes:
            base = baseline_metrics.get(class_name, {})
            exp = experiment_metrics.get(class_name, {})

            row = {
                "class": class_name,
            }

            for metric in ("precision", "recall", "ap"):
                baseline_value = base.get(metric)
                experiment_value = exp.get(metric)

                delta = None

                if (
                    baseline_value is not None
                    and experiment_value is not None
                ):
                    delta = experiment_value - baseline_value

                row[metric] = {
                    self.baseline_name: baseline_value,
                    self.experiment_name: experiment_value,
                    "delta": delta,
                }

            comparison.append(row)

        return {
            "classes": comparison,
            "largest_recall_regressions": self._top_metric_regressions(
                comparison,
                "recall",
            ),
            "largest_ap_regressions": self._top_metric_regressions(
                comparison,
                "ap",
            ),
            "largest_precision_regressions": self._top_metric_regressions(
                comparison,
                "precision",
            ),
        }

    @staticmethod
    def _top_metric_regressions(
        comparison: List[Dict[str, Any]],
        metric: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        valid = [
            item
            for item in comparison
            if item[metric]["delta"] is not None
        ]

        return sorted(
            valid,
            key=lambda item: item[metric]["delta"],
        )[:limit]

    # =========================================================================
    # CONFUSION AGGREGATES
    # =========================================================================

    @staticmethod
    def _extract_confusion(
        data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        confusion = (
            data
            .get("error_analysis", {})
            .get("confusion", {})
        )

        if not isinstance(confusion, dict):
            return {}

        result = {}

        for gt_class, predictions in confusion.items():
            if not isinstance(predictions, dict):
                continue

            result[gt_class] = {}

            for pred_class, info in predictions.items():
                if not isinstance(info, dict):
                    continue

                result[gt_class][pred_class] = {
                    "count": ErrorAnalysisComparator._safe_int(
                        info.get("count", 0)
                    ),
                    "mean_iou": ErrorAnalysisComparator._safe_float(
                        info.get("mean_iou")
                    ),
                    "mean_confidence": ErrorAnalysisComparator._safe_float(
                        info.get("mean_confidence")
                    ),
                    "images": list(info.get("images", []))
                    if isinstance(info.get("images", []), list)
                    else [],
                    "cases": list(info.get("cases", []))
                    if isinstance(info.get("cases", []), list)
                    else [],
                }

        return result

    @staticmethod
    def _flatten_confusion(
        confusion: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        flat = {}

        for gt_class, predictions in confusion.items():
            for pred_class, info in predictions.items():
                key = f"{gt_class} -> {pred_class}"

                flat[key] = {
                    "gt_class": gt_class,
                    "pred_class": pred_class,
                    **info,
                }

        return flat

    def _classify_confusion(
        self,
        mean_iou: Optional[float],
    ) -> str:
        if mean_iou is None:
            return "unknown"

        if mean_iou >= self.high_iou_threshold:
            return "classification_dominant"

        if mean_iou < self.low_iou_threshold:
            return "localization_dominant"

        return "mixed"

    def _compare_confusions(self) -> Dict[str, Any]:
        baseline = self._flatten_confusion(
            self._extract_confusion(self.baseline)
        )

        experiment = self._flatten_confusion(
            self._extract_confusion(self.experiment)
        )

        all_pairs = sorted(
            set(baseline.keys()) | set(experiment.keys())
        )

        comparison = []

        improved = []
        regressed = []
        unchanged = []
        new_errors = []
        removed_errors = []

        for pair in all_pairs:
            base = baseline.get(pair)
            exp = experiment.get(pair)

            base_count = base["count"] if base else 0
            exp_count = exp["count"] if exp else 0

            delta = exp_count - base_count

            if base_count == 0 and exp_count > 0:
                status = "new_error"
                new_errors.append(pair)

            elif base_count > 0 and exp_count == 0:
                status = "removed_error"
                removed_errors.append(pair)

            elif exp_count < base_count:
                status = "improved"
                improved.append(pair)

            elif exp_count > base_count:
                status = "regressed"
                regressed.append(pair)

            else:
                status = "unchanged"
                unchanged.append(pair)

            relative_change = None

            if base_count > 0:
                relative_change = delta / base_count

            row = {
                "confusion": pair,
                "gt_class": (
                    exp["gt_class"]
                    if exp
                    else base["gt_class"]
                ),
                "pred_class": (
                    exp["pred_class"]
                    if exp
                    else base["pred_class"]
                ),
                self.baseline_name: {
                    "count": base_count,
                    "mean_iou": (
                        base.get("mean_iou")
                        if base else None
                    ),
                    "mean_confidence": (
                        base.get("mean_confidence")
                        if base else None
                    ),
                },
                self.experiment_name: {
                    "count": exp_count,
                    "mean_iou": (
                        exp.get("mean_iou")
                        if exp else None
                    ),
                    "mean_confidence": (
                        exp.get("mean_confidence")
                        if exp else None
                    ),
                },
                "delta_count": delta,
                "relative_count_change": relative_change,
                "status": status,
                "experiment_error_type": self._classify_confusion(
                    exp.get("mean_iou")
                    if exp else None
                ),
            }

            comparison.append(row)

        total_baseline = sum(
            item["count"]
            for item in baseline.values()
        )

        total_experiment = sum(
            item["count"]
            for item in experiment.values()
        )

        return {
            "total_confusions": {
                self.baseline_name: total_baseline,
                self.experiment_name: total_experiment,
                "delta": total_experiment - total_baseline,
            },
            "summary": {
                "improved": len(improved),
                "regressed": len(regressed),
                "unchanged": len(unchanged),
                "new_errors": len(new_errors),
                "removed_errors": len(removed_errors),
            },
            "comparison": comparison,
            "top_regressions": sorted(
                [
                    row
                    for row in comparison
                    if row["delta_count"] > 0
                ],
                key=lambda x: x["delta_count"],
                reverse=True,
            )[:10],
            "top_improvements": sorted(
                [
                    row
                    for row in comparison
                    if row["delta_count"] < 0
                ],
                key=lambda x: x["delta_count"],
            )[:10],
            "new_errors": new_errors,
            "removed_errors": removed_errors,
            "improved_confusions": improved,
            "regressed_confusions": regressed,
            "unchanged_confusions": unchanged,
        }

    # =========================================================================
    # FP / FN AGGREGATES
    # =========================================================================

    @staticmethod
    def _extract_error_category(
        data: Dict[str, Any],
        category: str,
    ) -> Dict[str, Dict[str, Any]]:
        raw = (
            data
            .get("error_analysis", {})
            .get(category, {})
        )

        if not isinstance(raw, dict):
            return {}

        result = {}

        for class_name, info in raw.items():
            if not isinstance(info, dict):
                continue

            result[class_name] = {
                "count": ErrorAnalysisComparator._safe_int(
                    info.get("count", 0)
                ),
                "mean_confidence": ErrorAnalysisComparator._safe_float(
                    info.get("mean_confidence")
                ),
                "images": list(info.get("images", []))
                if isinstance(info.get("images", []), list)
                else [],
                "cases": list(info.get("cases", []))
                if isinstance(info.get("cases", []), list)
                else [],
            }

        return result
    
    def _compare_error_category(
        self,
        category: str,
    ) -> Dict[str, Any]:
        baseline = self._extract_error_category(
            self.baseline,
            category,
        )

        experiment = self._extract_error_category(
            self.experiment,
            category,
        )

        classes = sorted(
            set(baseline.keys())
            | set(experiment.keys())
        )

        comparison = []

        for class_name in classes:
            base = baseline.get(
                class_name,
                {"count": 0, "mean_confidence": None},
            )

            exp = experiment.get(
                class_name,
                {"count": 0, "mean_confidence": None},
            )

            base_count = base["count"]
            exp_count = exp["count"]

            delta = exp_count - base_count

            comparison.append({
                "class": class_name,
                self.baseline_name: {
                    "count": base_count,
                    "mean_confidence": base.get(
                        "mean_confidence"
                    ),
                },
                self.experiment_name: {
                    "count": exp_count,
                    "mean_confidence": exp.get(
                        "mean_confidence"
                    ),
                },
                "delta_count": delta,
                "status": (
                    "regressed"
                    if delta > 0
                    else "improved"
                    if delta < 0
                    else "unchanged"
                ),
            })

        return {
            "category": category,
            "comparison": comparison,
            "total": {
                self.baseline_name: sum(
                    item["count"]
                    for item in baseline.values()
                ),
                self.experiment_name: sum(
                    item["count"]
                    for item in experiment.values()
                ),
            },
            "top_regressions": sorted(
                [
                    item
                    for item in comparison
                    if item["delta_count"] > 0
                ],
                key=lambda x: x["delta_count"],
                reverse=True,
            )[:10],
            "top_improvements": sorted(
                [
                    item
                    for item in comparison
                    if item["delta_count"] < 0
                ],
                key=lambda x: x["delta_count"],
            )[:10],
        }

    # =========================================================================
    # CASE EXTRACTION
    # =========================================================================

    @staticmethod
    def _normalize_confusion_case(
        case: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(case, dict):
            return None

        image_path = case.get("image_path")
        gt_class = case.get("gt_class")
        pred_class = case.get("pred_class")

        if not image_path or not gt_class or not pred_class:
            return None

        return {
            "image_path": str(image_path),
            "image_name": ErrorAnalysisComparator._image_name(
                image_path
            ),
            "gt_class": gt_class,
            "pred_class": pred_class,
            "gt_bbox": case.get("gt_bbox"),
            "pred_bbox": case.get("pred_bbox"),
            "iou": ErrorAnalysisComparator._safe_float(
                case.get("iou")
            ),
            "confidence": ErrorAnalysisComparator._safe_float(
                case.get("confidence")
            ),
        }

    def _extract_confusion_cases(
        self,
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        confusion = self._extract_confusion(data)

        cases = []

        for gt_class, predictions in confusion.items():
            for pred_class, info in predictions.items():
                for raw_case in info.get("cases", []):
                    case = self._normalize_confusion_case(raw_case)

                    if case is None:
                        continue

                    # Protect against malformed aggregate/case mismatch.
                    case["aggregate_gt_class"] = gt_class
                    case["aggregate_pred_class"] = pred_class

                    cases.append(case)

        return cases

    @staticmethod
    def _normalize_fp_case(
        case: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(case, dict):
            return None

        image_path = case.get("image_path")
        pred_class = case.get("pred_class")

        if not image_path or not pred_class:
            return None

        return {
            "image_path": str(image_path),
            "image_name": ErrorAnalysisComparator._image_name(
                image_path
            ),
            "pred_class": pred_class,
            "pred_bbox": case.get("pred_bbox"),
            "confidence": ErrorAnalysisComparator._safe_float(
                case.get("confidence")
            ),
        }

    @staticmethod
    def _normalize_fn_case(
        case: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(case, dict):
            return None

        image_path = case.get("image_path")
        gt_class = case.get("gt_class")

        if not image_path or not gt_class:
            return None

        return {
            "image_path": str(image_path),
            "image_name": ErrorAnalysisComparator._image_name(
                image_path
            ),
            "gt_class": gt_class,
            "gt_bbox": case.get("gt_bbox"),
            "confidence": ErrorAnalysisComparator._safe_float(
                case.get("confidence")
            ),
        }

    def _extract_cases(
        self,
        data: Dict[str, Any],
        category: str,
    ) -> List[Dict[str, Any]]:
        raw = (
            data
            .get("error_analysis", {})
            .get(category, {})
        )

        if not isinstance(raw, dict):
            return []

        cases = []

        for class_name, info in raw.items():
            if not isinstance(info, dict):
                continue

            raw_cases = info.get("cases", [])

            if not isinstance(raw_cases, list):
                continue

            for raw_case in raw_cases:
                if category == "false_positives":
                    case = self._normalize_fp_case(raw_case)

                elif category == "false_negatives":
                    case = self._normalize_fn_case(raw_case)

                else:
                    case = None

                if case is not None:
                    case["aggregate_class"] = class_name
                    cases.append(case)

        return cases

    # =========================================================================
    # OBJECT-LEVEL MATCHING
    # =========================================================================

    @staticmethod
    def _greedy_match(
        baseline_cases: List[Dict[str, Any]],
        experiment_cases: List[Dict[str, Any]],
        similarity_function,
        threshold: float,
    ) -> Tuple[
        List[Tuple[Dict[str, Any], Dict[str, Any], float]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Deterministic greedy matching.

        Returns:
            matched
            unmatched_baseline
            unmatched_experiment
        """

        candidates = []

        for base_idx, base in enumerate(baseline_cases):
            for exp_idx, exp in enumerate(experiment_cases):
                score = similarity_function(base, exp)

                if score >= threshold:
                    candidates.append(
                        (score, base_idx, exp_idx)
                    )

        # Highest-quality matches first.
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            )
        )

        used_baseline = set()
        used_experiment = set()

        matched = []

        for score, base_idx, exp_idx in candidates:
            if base_idx in used_baseline:
                continue

            if exp_idx in used_experiment:
                continue

            used_baseline.add(base_idx)
            used_experiment.add(exp_idx)

            matched.append(
                (
                    baseline_cases[base_idx],
                    experiment_cases[exp_idx],
                    score,
                )
            )

        unmatched_baseline = [
            case
            for idx, case in enumerate(baseline_cases)
            if idx not in used_baseline
        ]

        unmatched_experiment = [
            case
            for idx, case in enumerate(experiment_cases)
            if idx not in used_experiment
        ]

        return (
            matched,
            unmatched_baseline,
            unmatched_experiment,
        )

    def _confusion_similarity(
        self,
        baseline_case: Dict[str, Any],
        experiment_case: Dict[str, Any],
    ) -> float:
        if (
            baseline_case.get("image_name")
            != experiment_case.get("image_name")
        ):
            return 0.0

        if (
            baseline_case.get("gt_class")
            != experiment_case.get("gt_class")
        ):
            return 0.0

        return self._bbox_iou(
            baseline_case.get("gt_bbox"),
            experiment_case.get("gt_bbox"),
        )

    def _compare_confusion_cases(self) -> Dict[str, Any]:
        baseline_cases = self._extract_confusion_cases(
            self.baseline
        )

        experiment_cases = self._extract_confusion_cases(
            self.experiment
        )

        matched, unmatched_baseline, unmatched_experiment = (
            self._greedy_match(
                baseline_cases,
                experiment_cases,
                self._confusion_similarity,
                self.gt_bbox_iou_threshold,
            )
        )

        persistent = []
        prediction_changed = []

        for base, exp, match_iou in matched:
            record = {
                "image_name": base["image_name"],
                "gt_class": base["gt_class"],
                "gt_bbox_match_iou": match_iou,
                self.baseline_name: {
                    "pred_class": base["pred_class"],
                    "iou": base["iou"],
                    "confidence": base["confidence"],
                    "pred_bbox": base["pred_bbox"],
                },
                self.experiment_name: {
                    "pred_class": exp["pred_class"],
                    "iou": exp["iou"],
                    "confidence": exp["confidence"],
                    "pred_bbox": exp["pred_bbox"],
                },
            }

            record["confidence_delta"] = (
                (
                    exp["confidence"]
                    - base["confidence"]
                )
                if (
                    exp["confidence"] is not None
                    and base["confidence"] is not None
                )
                else None
            )

            record["iou_delta"] = (
                (
                    exp["iou"]
                    - base["iou"]
                )
                if (
                    exp["iou"] is not None
                    and base["iou"] is not None
                )
                else None
            )

            if base["pred_class"] == exp["pred_class"]:
                record["transition"] = "persistent_error"
                persistent.append(record)
            else:
                record["transition"] = "prediction_changed"
                prediction_changed.append(record)

        resolved_or_changed = []

        for case in unmatched_baseline:
            resolved_or_changed.append({
                "image_name": case["image_name"],
                "gt_class": case["gt_class"],
                "previous_pred_class": case["pred_class"],
                "previous_iou": case["iou"],
                "previous_confidence": case["confidence"],
                "transition": "confusion_resolved_or_changed",
                "note": (
                    "No matching confusion case was found in the "
                    f"{self.experiment_name} confusion inventory. "
                    "This does not by itself prove the object became correct."
                ),
            })

        new_confusion_cases = []

        for case in unmatched_experiment:
            new_confusion_cases.append({
                "image_name": case["image_name"],
                "gt_class": case["gt_class"],
                "pred_class": case["pred_class"],
                "iou": case["iou"],
                "confidence": case["confidence"],
                "transition": "new_confusion_case",
            })

        return {
            "matching": {
                "method": "same_image + same_gt_class + GT_bbox_IoU",
                "gt_bbox_iou_threshold": self.gt_bbox_iou_threshold,
                "baseline_cases": len(baseline_cases),
                "experiment_cases": len(experiment_cases),
                "matched_cases": len(matched),
                "unmatched_baseline_cases": len(
                    unmatched_baseline
                ),
                "unmatched_experiment_cases": len(
                    unmatched_experiment
                ),
            },
            "summary": {
                "persistent_errors": len(persistent),
                "prediction_changed": len(prediction_changed),
                "confusion_resolved_or_changed": len(
                    resolved_or_changed
                ),
                "new_confusion_cases": len(
                    new_confusion_cases
                ),
            },
            "persistent_errors": persistent,
            "prediction_changed": prediction_changed,
            "confusion_resolved_or_changed": resolved_or_changed,
            "new_confusion_cases": new_confusion_cases,
        }

    # =========================================================================
    # FP / FN CASE-LEVEL COMPARISON
    # =========================================================================

    def _fp_similarity(
        self,
        baseline_case: Dict[str, Any],
        experiment_case: Dict[str, Any],
    ) -> float:
        if (
            baseline_case.get("image_name")
            != experiment_case.get("image_name")
        ):
            return 0.0

        if (
            baseline_case.get("pred_class")
            != experiment_case.get("pred_class")
        ):
            return 0.0

        return self._bbox_iou(
            baseline_case.get("pred_bbox"),
            experiment_case.get("pred_bbox"),
        )

    def _fn_similarity(
        self,
        baseline_case: Dict[str, Any],
        experiment_case: Dict[str, Any],
    ) -> float:
        if (
            baseline_case.get("image_name")
            != experiment_case.get("image_name")
        ):
            return 0.0

        if (
            baseline_case.get("gt_class")
            != experiment_case.get("gt_class")
        ):
            return 0.0

        return self._bbox_iou(
            baseline_case.get("gt_bbox"),
            experiment_case.get("gt_bbox"),
        )

    def _compare_case_category(
        self,
        category: str,
        similarity_function,
    ) -> Dict[str, Any]:
        baseline_cases = self._extract_cases(
            self.baseline,
            category,
        )

        experiment_cases = self._extract_cases(
            self.experiment,
            category,
        )

        matched, unmatched_baseline, unmatched_experiment = (
            self._greedy_match(
                baseline_cases,
                experiment_cases,
                similarity_function,
                self.gt_bbox_iou_threshold,
            )
        )

        persistent = []

        for base, exp, match_iou in matched:
            confidence_delta = None

            if (
                base.get("confidence") is not None
                and exp.get("confidence") is not None
            ):
                confidence_delta = (
                    exp["confidence"]
                    - base["confidence"]
                )

            persistent.append({
                "image_name": base["image_name"],
                "match_bbox_iou": match_iou,
                "class": (
                    base.get("pred_class")
                    or base.get("gt_class")
                ),
                self.baseline_name: {
                    "confidence": base.get("confidence"),
                },
                self.experiment_name: {
                    "confidence": exp.get("confidence"),
                },
                "confidence_delta": confidence_delta,
            })

        return {
            "category": category,
            "matching": {
                "threshold": self.gt_bbox_iou_threshold,
                "matched": len(matched),
                "unmatched_baseline": len(
                    unmatched_baseline
                ),
                "unmatched_experiment": len(
                    unmatched_experiment
                ),
            },
            "persistent_cases": persistent,
            "baseline_only_cases": unmatched_baseline,
            "experiment_only_cases": unmatched_experiment,
        }

    # =========================================================================
    # IMAGE-LEVEL ANALYSIS
    # =========================================================================

    @staticmethod
    def _collect_error_images(
        data: Dict[str, Any],
    ) -> Dict[str, set]:
        result = {
            category: set()
            for category in ErrorAnalysisComparator.ERROR_TYPES
        }

        error_analysis = data.get("error_analysis", {})

        if not isinstance(error_analysis, dict):
            return result

        mapping = {
            "confusion": "confusion",
            "false_positive": "false_positives",
            "false_negative": "false_negatives",
        }

        for normalized_name, json_name in mapping.items():
            raw = error_analysis.get(json_name, {})

            if not isinstance(raw, dict):
                continue

            for _, info in raw.items():
                if not isinstance(info, dict):
                    continue

                images = info.get("images", [])

                if not isinstance(images, list):
                    continue

                for image in images:
                    image_name = ErrorAnalysisComparator._image_name(
                        image
                    )

                    if image_name:
                        result[normalized_name].add(
                            image_name
                        )

        return result

    def _compare_error_images(self) -> Dict[str, Any]:
        baseline = self._collect_error_images(
            self.baseline
        )

        experiment = self._collect_error_images(
            self.experiment
        )

        all_baseline = set().union(*baseline.values())
        all_experiment = set().union(*experiment.values())

        both = all_baseline & all_experiment
        baseline_only = all_baseline - all_experiment
        experiment_only = all_experiment - all_baseline

        return {
            "baseline_error_images": len(all_baseline),
            "experiment_error_images": len(all_experiment),
            "both_error_images": len(both),
            "baseline_only_error_images": len(
                baseline_only
            ),
            "experiment_only_error_images": len(
                experiment_only
            ),
            "baseline_only_images": sorted(baseline_only),
            "experiment_only_images": sorted(experiment_only),
        }

    # =========================================================================
    # IOU / CONFIDENCE ANALYSIS
    # =========================================================================

    def _case_statistics(
        self,
        cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ious = [
            case["iou"]
            for case in cases
            if case.get("iou") is not None
        ]

        confidences = [
            case["confidence"]
            for case in cases
            if case.get("confidence") is not None
        ]

        def mean(values):
            if not values:
                return None

            return sum(values) / len(values)

        high_iou_count = sum(
            value >= self.high_iou_threshold
            for value in ious
        )

        low_iou_count = sum(
            value < self.low_iou_threshold
            for value in ious
        )

        return {
            "case_count": len(cases),
            "iou": {
                "count": len(ious),
                "mean": mean(ious),
                "high_iou_count": high_iou_count,
                "high_iou_ratio": (
                    high_iou_count / len(ious)
                    if ious
                    else None
                ),
                "low_iou_count": low_iou_count,
                "low_iou_ratio": (
                    low_iou_count / len(ious)
                    if ious
                    else None
                ),
            },
            "confidence": {
                "count": len(confidences),
                "mean": mean(confidences),
            },
        }

    def _compare_case_statistics(self) -> Dict[str, Any]:
        baseline_cases = self._extract_confusion_cases(
            self.baseline
        )

        experiment_cases = self._extract_confusion_cases(
            self.experiment
        )

        return {
            self.baseline_name: self._case_statistics(
                baseline_cases
            ),
            self.experiment_name: self._case_statistics(
                experiment_cases
            ),
        }

    # =========================================================================
    # FOCUS CLASS ANALYSIS
    # =========================================================================

    def _focus_class_analysis(self) -> Dict[str, Any]:
        baseline_confusion = self._flatten_confusion(
            self._extract_confusion(self.baseline)
        )

        experiment_confusion = self._flatten_confusion(
            self._extract_confusion(self.experiment)
        )

        result = {}

        for class_name in self.focus_classes:
            base_outgoing = [
                item
                for item in baseline_confusion.values()
                if item["gt_class"] == class_name
            ]

            exp_outgoing = [
                item
                for item in experiment_confusion.values()
                if item["gt_class"] == class_name
            ]

            base_incoming = [
                item
                for item in baseline_confusion.values()
                if item["pred_class"] == class_name
            ]

            exp_incoming = [
                item
                for item in experiment_confusion.values()
                if item["pred_class"] == class_name
            ]

            result[class_name] = {
                "outgoing_confusions": {
                    self.baseline_name: sum(
                        item["count"]
                        for item in base_outgoing
                    ),
                    self.experiment_name: sum(
                        item["count"]
                        for item in exp_outgoing
                    ),
                    "delta": (
                        sum(
                            item["count"]
                            for item in exp_outgoing
                        )
                        -
                        sum(
                            item["count"]
                            for item in base_outgoing
                        )
                    ),
                },
                "incoming_confusions": {
                    self.baseline_name: sum(
                        item["count"]
                        for item in base_incoming
                    ),
                    self.experiment_name: sum(
                        item["count"]
                        for item in exp_incoming
                    ),
                    "delta": (
                        sum(
                            item["count"]
                            for item in exp_incoming
                        )
                        -
                        sum(
                            item["count"]
                            for item in base_incoming
                        )
                    ),
                },
            }

        return result

    # =========================================================================
    # ROOT-CAUSE HEURISTICS
    # =========================================================================

    def _root_cause_summary(
        self,
        metrics: Dict[str, Any],
        confusions: Dict[str, Any],
        case_stats: Dict[str, Any],
        fp: Dict[str, Any],
        fn: Dict[str, Any],
    ) -> Dict[str, Any]:
        hypotheses = []

        metric_rows = metrics["classes"]

        metal_metric = next(
            (
                row
                for row in metric_rows
                if row["class"].lower() == "metal"
            ),
            None,
        )

        metal_recall_delta = None

        if metal_metric:
            metal_recall_delta = metal_metric[
                "recall"
            ]["delta"]

            if (
                metal_recall_delta is not None
                and metal_recall_delta < 0
            ):
                hypotheses.append(
                    {
                        "type": "metal_recall_regression",
                        "severity": abs(metal_recall_delta),
                        "evidence": (
                            f"Metal recall changed by "
                            f"{metal_recall_delta:.4f}"
                        ),
                    }
                )

        total_confusion_delta = confusions[
            "total_confusions"
        ]["delta"]

        base_stats = case_stats[self.baseline_name]
        exp_stats = case_stats[self.experiment_name]

        base_high_ratio = base_stats["iou"][
            "high_iou_ratio"
        ]

        exp_high_ratio = exp_stats["iou"][
            "high_iou_ratio"
        ]

        if (
            total_confusion_delta > 0
            and exp_high_ratio is not None
            and (
                base_high_ratio is None
                or exp_high_ratio >= base_high_ratio
            )
        ):
            hypotheses.append(
                {
                    "type": "classification_dominant_regression",
                    "evidence": (
                        "Confusion count increased while a high "
                        "fraction of confusion cases still have "
                        f"IoU >= {self.high_iou_threshold:.2f}."
                    ),
                }
            )

        base_low_ratio = base_stats["iou"][
            "low_iou_ratio"
        ]

        exp_low_ratio = exp_stats["iou"][
            "low_iou_ratio"
        ]

        if (
            exp_low_ratio is not None
            and (
                base_low_ratio is None
                or exp_low_ratio > base_low_ratio
            )
        ):
            hypotheses.append(
                {
                    "type": "localization_regression_signal",
                    "evidence": (
                        "The proportion of low-IoU confusion "
                        "cases increased."
                    ),
                }
            )

        top_confusion_regressions = confusions[
            "top_regressions"
        ]

        if top_confusion_regressions:
            dominant = top_confusion_regressions[0]

            hypotheses.append(
                {
                    "type": "dominant_confusion",
                    "confusion": dominant["confusion"],
                    "delta_count": dominant["delta_count"],
                    "experiment_mean_iou": (
                        dominant[self.experiment_name][
                            "mean_iou"
                        ]
                    ),
                    "evidence": (
                        "This confusion pair has the largest "
                        "increase in case count."
                    ),
                }
            )

        metal_fp_delta = self._class_delta(
            fp,
            "metal",
        )

        metal_fn_delta = self._class_delta(
            fn,
            "metal",
        )

        if metal_fp_delta is not None and metal_fp_delta > 0:
            hypotheses.append(
                {
                    "type": "metal_false_positive_increase",
                    "delta_count": metal_fp_delta,
                    "evidence": (
                        "False positives predicted as Metal increased."
                    ),
                }
            )

        if metal_fn_delta is not None and metal_fn_delta > 0:
            hypotheses.append(
                {
                    "type": "metal_false_negative_increase",
                    "delta_count": metal_fn_delta,
                    "evidence": (
                        "Metal false negatives increased."
                    ),
                }
            )

        return {
            "interpretation": (
                "These are evidence-based diagnostic signals, "
                "not causal proof. Root cause must be confirmed "
                "against the E6 hard-mined training samples."
            ),
            "hypotheses": hypotheses,
        }

    @staticmethod
    def _class_delta(
        category_result: Dict[str, Any],
        class_name: str,
    ) -> Optional[int]:
        for item in category_result.get(
            "comparison",
            [],
        ):
            if item["class"] == class_name:
                return item["delta_count"]

        return None

    # =========================================================================
    # MAIN COMPARISON
    # =========================================================================

    def compare(self) -> Dict[str, Any]:
        LOGGER.info(
            "Comparing %s against %s",
            self.baseline_name,
            self.experiment_name,
        )

        metrics = self._compare_metrics()
        confusions = self._compare_confusions()

        false_positives = self._compare_error_category(
            "false_positives"
        )

        false_negatives = self._compare_error_category(
            "false_negatives"
        )

        confusion_cases = self._compare_confusion_cases()

        fp_cases = self._compare_case_category(
            "false_positives",
            self._fp_similarity,
        )

        fn_cases = self._compare_case_category(
            "false_negatives",
            self._fn_similarity,
        )

        image_analysis = self._compare_error_images()

        case_statistics = self._compare_case_statistics()

        focus_classes = self._focus_class_analysis()

        root_cause = self._root_cause_summary(
            metrics=metrics,
            confusions=confusions,
            case_stats=case_statistics,
            fp=false_positives,
            fn=false_negatives,
        )

        return {
            "comparison": {
                "baseline": self.baseline_name,
                "experiment": self.experiment_name,
                "baseline_path": str(
                    self.baseline_path
                ),
                "experiment_path": str(
                    self.experiment_path
                ),
                "split_baseline": self.baseline.get(
                    "split"
                ),
                "split_experiment": self.experiment.get(
                    "split"
                ),
            },

            "configuration": {
                "gt_bbox_iou_threshold": (
                    self.gt_bbox_iou_threshold
                ),
                "high_iou_threshold": (
                    self.high_iou_threshold
                ),
                "low_iou_threshold": (
                    self.low_iou_threshold
                ),
                "focus_classes": self.focus_classes,
            },

            "metrics": metrics,

            "confusions": confusions,

            "false_positives": false_positives,

            "false_negatives": false_negatives,

            "case_level": {
                "confusions": confusion_cases,
                "false_positives": fp_cases,
                "false_negatives": fn_cases,
            },

            "case_statistics": case_statistics,

            "image_level": image_analysis,

            "focus_classes": focus_classes,

            "root_cause": root_cause,

            "methodology_notes": [
                (
                    "Confusion object matching uses "
                    "same image + same GT class + GT bbox IoU."
                ),
                (
                    "A removed confusion is not automatically "
                    "considered a correct prediction."
                ),
                (
                    "Object-level correct->wrong transitions cannot "
                    "be proven from error-only records when the "
                    "correct prediction inventory is unavailable."
                ),
                (
                    "High-IoU confusion is treated as a "
                    "classification-dominant diagnostic signal."
                ),
                (
                    "Low-IoU confusion is treated as a "
                    "localization-dominant diagnostic signal."
                ),
                (
                    "Root-cause hypotheses must be confirmed by "
                    "inspection of E6 hard-mined training samples."
                ),
            ],
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, set):
            return sorted(obj)

        raise TypeError(
            f"Object of type {type(obj).__name__} "
            "is not JSON serializable"
        )

    @staticmethod
    def _atomic_write_text(
        path: Path,
        content: str,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=str(path.parent),
            text=True,
        )

        try:
            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(content)

            os.replace(
                temp_path,
                path,
            )

        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            raise

    def save_json(
        self,
        result: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)

        content = json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=self._json_default,
        )

        self._atomic_write_text(
            output_path,
            content,
        )

        return output_path

    # =========================================================================
    # TEXT REPORT
    # =========================================================================

    def _format_float(
        self,
        value: Optional[float],
        digits: int = 4,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.{digits}f}"

    def _format_delta(
        self,
        value: Optional[float],
        digits: int = 4,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:+.{digits}f}"

    def generate_text_report(
        self,
        result: Dict[str, Any],
    ) -> str:
        lines = []

        append = lines.append

        append("=" * 90)
        append(
            f"{self.baseline_name} vs "
            f"{self.experiment_name} "
            "ERROR ANALYSIS COMPARISON"
        )
        append("=" * 90)
        append("")

        append("1. EXPERIMENTS")
        append("-" * 90)
        append(
            f"Baseline   : {self.baseline_name}"
        )
        append(
            f"Experiment : {self.experiment_name}"
        )
        append(
            f"Baseline split   : "
            f"{result['comparison']['split_baseline']}"
        )
        append(
            f"Experiment split : "
            f"{result['comparison']['split_experiment']}"
        )
        append("")

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        append("2. PER-CLASS METRICS")
        append("-" * 90)

        for row in result["metrics"]["classes"]:
            class_name = row["class"]

            append(f"[{class_name}]")

            for metric in (
                "precision",
                "recall",
                "ap",
            ):
                data = row[metric]

                append(
                    f"  {metric:<10} "
                    f"{self.baseline_name}="
                    f"{self._format_float(data[self.baseline_name])}  "
                    f"{self.experiment_name}="
                    f"{self._format_float(data[self.experiment_name])}  "
                    f"Δ="
                    f"{self._format_delta(data['delta'])}"
                )

        append("")

        # ---------------------------------------------------------------------
        # Confusions
        # ---------------------------------------------------------------------

        append("3. CONFUSION ANALYSIS")
        append("-" * 90)

        total = result["confusions"]["total_confusions"]

        append(
            f"Total confusions: "
            f"{self.baseline_name}={total[self.baseline_name]}  "
            f"{self.experiment_name}={total[self.experiment_name]}  "
            f"Δ={total['delta']:+d}"
        )

        summary = result["confusions"]["summary"]

        append(
            f"Improved={summary['improved']}  "
            f"Regressed={summary['regressed']}  "
            f"Unchanged={summary['unchanged']}  "
            f"New={summary['new_errors']}  "
            f"Removed={summary['removed_errors']}"
        )

        append("")
        append("Top confusion regressions:")

        for item in result["confusions"]["top_regressions"]:
            append(
                f"  {item['confusion']:<30} "
                f"Δcount={item['delta_count']:+d}  "
                f"IoU="
                f"{self._format_float(item[self.experiment_name]['mean_iou'])}  "
                f"conf="
                f"{self._format_float(item[self.experiment_name]['mean_confidence'])}"
            )

        append("")

        append("Top confusion improvements:")

        for item in result["confusions"]["top_improvements"]:
            append(
                f"  {item['confusion']:<30} "
                f"Δcount={item['delta_count']:+d}"
            )

        append("")

        # ---------------------------------------------------------------------
        # Case-level confusion
        # ---------------------------------------------------------------------

        case_conf = result["case_level"]["confusions"]

        append("4. OBJECT-LEVEL CONFUSION ANALYSIS")
        append("-" * 90)

        matching = case_conf["matching"]

        append(
            f"Baseline cases   : "
            f"{matching['baseline_cases']}"
        )
        append(
            f"Experiment cases : "
            f"{matching['experiment_cases']}"
        )
        append(
            f"Matched objects  : "
            f"{matching['matched_cases']}"
        )
        append(
            f"GT bbox IoU threshold: "
            f"{matching['gt_bbox_iou_threshold']}"
        )

        case_summary = case_conf["summary"]

        append(
            f"Persistent errors       : "
            f"{case_summary['persistent_errors']}"
        )
        append(
            f"Prediction changed     : "
            f"{case_summary['prediction_changed']}"
        )
        append(
            f"Resolved/changed       : "
            f"{case_summary['confusion_resolved_or_changed']}"
        )
        append(
            f"New confusion cases    : "
            f"{case_summary['new_confusion_cases']}"
        )

        if case_conf["prediction_changed"]:
            append("")
            append("Prediction changes:")

            for item in case_conf["prediction_changed"][:20]:
                append(
                    f"  {item['image_name']} | "
                    f"GT={item['gt_class']} | "
                    f"{self.baseline_name}:"
                    f"{item[self.baseline_name]['pred_class']} "
                    f"-> "
                    f"{self.experiment_name}:"
                    f"{item[self.experiment_name]['pred_class']} | "
                    f"GT-match-IoU="
                    f"{self._format_float(item['gt_bbox_match_iou'])}"
                )

        append("")

        # ---------------------------------------------------------------------
        # FP / FN
        # ---------------------------------------------------------------------

        append("5. FALSE POSITIVE ANALYSIS")
        append("-" * 90)

        fp_total = result["false_positives"]["total"]

        append(
            f"Total FP: "
            f"{self.baseline_name}="
            f"{fp_total[self.baseline_name]}  "
            f"{self.experiment_name}="
            f"{fp_total[self.experiment_name]}"
        )

        for item in result["false_positives"]["top_regressions"]:
            append(
                f"  {item['class']:<15} "
                f"Δcount={item['delta_count']:+d}"
            )

        append("")

        append("6. FALSE NEGATIVE ANALYSIS")
        append("-" * 90)

        fn_total = result["false_negatives"]["total"]

        append(
            f"Total FN: "
            f"{self.baseline_name}="
            f"{fn_total[self.baseline_name]}  "
            f"{self.experiment_name}="
            f"{fn_total[self.experiment_name]}"
        )

        for item in result["false_negatives"]["top_regressions"]:
            append(
                f"  {item['class']:<15} "
                f"Δcount={item['delta_count']:+d}"
            )

        append("")

        # ---------------------------------------------------------------------
        # Case statistics
        # ---------------------------------------------------------------------

        append("7. CONFUSION CASE STATISTICS")
        append("-" * 90)

        for name in (
            self.baseline_name,
            self.experiment_name,
        ):
            stats = result["case_statistics"][name]

            append(f"{name}:")

            append(
                f"  Mean IoU        : "
                f"{self._format_float(stats['iou']['mean'])}"
            )

            append(
                f"  High-IoU ratio  : "
                f"{self._format_float(stats['iou']['high_iou_ratio'])}"
            )

            append(
                f"  Low-IoU ratio   : "
                f"{self._format_float(stats['iou']['low_iou_ratio'])}"
            )

            append(
                f"  Mean confidence : "
                f"{self._format_float(stats['confidence']['mean'])}"
            )

        append("")

        # ---------------------------------------------------------------------
        # Focus classes
        # ---------------------------------------------------------------------

        append("8. FOCUS CLASS ANALYSIS")
        append("-" * 90)

        for class_name, data in result[
            "focus_classes"
        ].items():
            outgoing = data["outgoing_confusions"]
            incoming = data["incoming_confusions"]

            append(f"{class_name}:")

            append(
                f"  Outgoing confusion Δ: "
                f"{outgoing['delta']:+d}"
            )

            append(
                f"  Incoming confusion Δ: "
                f"{incoming['delta']:+d}"
            )

        append("")

        # ---------------------------------------------------------------------
        # Root cause
        # ---------------------------------------------------------------------

        append("9. ROOT-CAUSE DIAGNOSTIC SIGNALS")
        append("-" * 90)

        root = result["root_cause"]

        append(root["interpretation"])
        append("")

        for hypothesis in root["hypotheses"]:
            hypothesis_type = hypothesis["type"]

            append(
                f"- {hypothesis_type}"
            )

            for key in (
                "confusion",
                "delta_count",
                "severity",
                "evidence",
            ):
                if key in hypothesis:
                    append(
                        f"    {key}: "
                        f"{hypothesis[key]}"
                    )

        append("")

        # ---------------------------------------------------------------------
        # Methodology
        # ---------------------------------------------------------------------

        append("10. METHODOLOGY NOTES")
        append("-" * 90)

        for note in result["methodology_notes"]:
            append(f"- {note}")

        append("")
        append("=" * 90)

        return "\n".join(lines)

    def save_text_report(
        self,
        result: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        output_path = Path(output_path)

        report = self.generate_text_report(
            result
        )

        self._atomic_write_text(
            output_path,
            report,
        )

        return output_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    # -------------------------------------------------------------------------
    # Project paths
    # -------------------------------------------------------------------------

    E2_ERROR_PATH = Path(
        "outputs/hard_case_analysis/E2/error_analysis.json"
    )

    E6_ERROR_PATH = Path(
        "outputs/hard_case_analysis/E6/error_analysis.json"
    )

    OUTPUT_DIR = Path(
        "outputs/hard_case_analysis/E2_vs_E6"
    )

    OUTPUT_JSON = (
        OUTPUT_DIR
        / "e2_vs_e6_error_comparison.json"
    )

    OUTPUT_TXT = (
        OUTPUT_DIR
        / "e2_vs_e6_error_comparison.txt"
    )

    # -------------------------------------------------------------------------
    # Comparator
    # -------------------------------------------------------------------------

    comparator = ErrorAnalysisComparator(
        baseline_path=E2_ERROR_PATH,
        experiment_path=E6_ERROR_PATH,
        baseline_name="E2",
        experiment_name="E6",

        # Same GT object if GT boxes overlap at >= 95%.
        gt_bbox_iou_threshold=0.95,

        # Diagnostic thresholds.
        high_iou_threshold=0.80,
        low_iou_threshold=0.50,

        # Important project classes.
        focus_classes=[
            "metal",
            "glass",
            "plastic",
            "paper",
            "cardboard",
        ],
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    result = comparator.compare()

    json_path = comparator.save_json(
        result,
        OUTPUT_JSON,
    )

    txt_path = comparator.save_text_report(
        result,
        OUTPUT_TXT,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("E2 vs E6 ERROR ANALYSIS COMPARISON")
    print("=" * 90)

    metrics = result["metrics"]

    for row in metrics["classes"]:
        recall_delta = row["recall"]["delta"]
        ap_delta = row["ap"]["delta"]

        print(
            f"{row['class']:<12} "
            f"Recall Δ={recall_delta:+.4f} | "
            f"AP Δ={ap_delta:+.4f}"
        )

    print("-" * 90)

    confusion_total = result[
        "confusions"
    ]["total_confusions"]

    print(
        f"Total Confusions: "
        f"E2={confusion_total['E2']} | "
        f"E6={confusion_total['E6']} | "
        f"Δ={confusion_total['delta']:+d}"
    )

    case_summary = result[
        "case_level"
    ]["confusions"]["summary"]

    print(
        f"Persistent: {case_summary['persistent_errors']} | "
        f"Changed: {case_summary['prediction_changed']} | "
        f"New: {case_summary['new_confusion_cases']} | "
        f"Resolved/Changed: "
        f"{case_summary['confusion_resolved_or_changed']}"
    )

    print("-" * 90)

    print("Top confusion regressions:")

    for item in result[
        "confusions"
    ]["top_regressions"][:5]:
        print(
            f"  {item['confusion']:<30} "
            f"Δ={item['delta_count']:+d}"
        )

    print("-" * 90)

    print(
        f"JSON report: {json_path}"
    )

    print(
        f"TXT report : {txt_path}"
    )

    print("=" * 90)