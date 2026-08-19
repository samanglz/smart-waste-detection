"""
Visualization utilities for model error analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.logging.logger import get_logger


logger = get_logger(__name__)


class ErrorVisualizer:
    """
    Visualize error cases produced by the evaluation pipeline.

    Responsibilities:
        - Load image files.
        - Draw ground-truth bounding boxes.
        - Draw prediction bounding boxes.
        - Save annotated error cases.
        - Save visualization summaries.

    This class does NOT:
        - Perform prediction.
        - Match predictions with ground truths.
        - Calculate IoU.
        - Classify errors.
        - Generate evaluation metrics.
    """

    def __init__(self, class_names: List[str]):
        """
        Initialize the error visualizer.

        Args:
            class_names:
                Class names indexed by class ID.
        """

        self.class_names = class_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def visualize_confusion(
        self,
        cases: Dict[str, Dict[str, Any]],
        gt_class: str,
        pred_class: str,
        output_dir: Path,
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Visualize confusion cases between two classes.

        This method retrieves confusion cases for a specific
        ground-truth/prediction class pair, delegates case-level
        visualization to ``_save_cases()``, and stores a summary
        describing the generated visualizations.

        Args:
            cases:
                Confusion data produced by ``ErrorAnalyzer``.
                The expected structure is:

                    {
                        "gt_class": {
                            "pred_class": {
                                "count": int,
                                "mean_iou": float,
                                "mean_confidence": float,
                                "cases": [...]
                            }
                        }
                    }

            gt_class:
                Ground-truth class name.

            pred_class:
                Predicted class name.

            output_dir:
                Root directory where confusion visualizations
                will be stored.

            top_k:
                Optional maximum number of confusion cases to
                visualize. If ``None``, all available cases
                are processed.

        Returns:
            Path:
                Directory containing the generated confusion
                visualizations and summary.
        """

        output_dir = (
            Path(output_dir)
            / "confusion"
            / f"{gt_class}_to_{pred_class}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        confusion_data = (
            cases
            .get(gt_class, {})
            .get(pred_class)
        )

        if not confusion_data:
            logger.warning(
                "No confusion cases found: %s -> %s",
                gt_class,
                pred_class,
            )

            return output_dir

        error_cases = confusion_data.get(
            "cases",
            [],
        )

        saved_images = self._save_cases(
            cases=error_cases,
            output_dir=output_dir,
            top_k=top_k,
        )

        requested_cases = (
            min(len(error_cases), top_k)
            if top_k is not None
            else len(error_cases)
        )

        self._save_summary(
            output_dir=output_dir,
            summary={
                "error_type": "confusion",
                "gt_class": gt_class,
                "pred_class": pred_class,
                "count": confusion_data.get(
                    "count",
                    0,
                ),
                "mean_iou": confusion_data.get(
                    "mean_iou",
                    0.0,
                ),
                "mean_confidence": confusion_data.get(
                    "mean_confidence",
                    0.0,
                ),
                "requested_cases": requested_cases,
                "saved_images": saved_images,
            },
        )

        return output_dir



    def visualize_false_positives(
        self,
        cases: Dict[str, Dict[str, Any]],
        class_name: str,
        output_dir: Path,
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Visualize false-positive cases for a class.

        Args:
            cases:
                False-positive data produced by ErrorAnalyzer.

            class_name:
                Predicted class name.

            output_dir:
                Output directory.

            top_k:
                Maximum number of cases to save.

        Returns:
            Path to output directory.
        """

        output_dir = (
            Path(output_dir)
            / "false_positives"
            / class_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Get class data
        # --------------------------------------------------

        class_data = cases.get(class_name)

        if not class_data:
            logger.warning(
                "No false-positive cases found for %s",
                class_name,
            )

            return output_dir

        # --------------------------------------------------
        # Get error cases
        # --------------------------------------------------

        error_cases = class_data.get(
            "cases",
            [],
        )

        # --------------------------------------------------
        # Save visualized cases
        # --------------------------------------------------

        saved_images = self._save_cases(
            cases=error_cases,
            output_dir=output_dir,
            top_k=top_k,
        )
        # --------------------------------------------------
        # Calculate requested cases
        # --------------------------------------------------

        requested_cases = (
            min(len(error_cases), top_k)
            if top_k is not None
            else len(error_cases)
        )

        # --------------------------------------------------
        # Save summary
        # --------------------------------------------------

        self._save_summary(
            output_dir=output_dir,
            summary={
                "error_type": "false_positive",
                "class_name": class_name,
                "count": class_data.get(
                    "count",
                    0,
                ),
                "mean_confidence": class_data.get(
                    "mean_confidence",
                    0.0,
                ),
                "requested_cases": requested_cases,
                "saved_images": saved_images,
            },
        )

        return output_dir



    def visualize_false_negatives(
        self,
        cases: Dict[str, Dict[str, Any]],
        class_name: str,
        output_dir: Path,
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Visualize false-negative cases for a class.

        Args:
            cases:
                False-negative data produced by ErrorAnalyzer.

            class_name:
                Ground-truth class name.

            output_dir:
                Output directory.

            top_k:
                Maximum number of cases to save.

        Returns:
            Path to output directory.
        """

        output_dir = (
            Path(output_dir)
            / "false_negatives"
            / class_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Get class data
        # --------------------------------------------------

        class_data = cases.get(class_name)

        if not class_data:
            logger.warning(
                "No false-negative cases found for %s",
                class_name,
            )

            return output_dir

        # --------------------------------------------------
        # Get error cases
        # --------------------------------------------------

        error_cases = class_data.get(
            "cases",
            [],
        )

        # --------------------------------------------------
        # Save visualized cases
        # --------------------------------------------------
        
        saved_images = self._save_cases(
            cases=error_cases,
            output_dir=output_dir,
            top_k=top_k,
        )
        
        # --------------------------------------------------
        # Calculate requested cases
        # --------------------------------------------------

        requested_cases = (
            min(len(error_cases), top_k)
            if top_k is not None
            else len(error_cases)
        )

        # --------------------------------------------------
        # Save summary
        # --------------------------------------------------

        self._save_summary(
            output_dir=output_dir,
            summary={
                "error_type": "false_negative",
                "class_name": class_name,
                "count": class_data.get(
                    "count",
                    0,
                ),
                "requested_cases": requested_cases,
                "saved_images": saved_images,
            },
        )

        return output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _save_summary(
        self,
        output_dir: Path,
        summary: Dict[str, Any],
    ) -> None:
        """
        Save metadata about the generated visualization.
        """

        summary_path = (
            output_dir / "summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Visualization summary saved to %s",
            summary_path,
        )
        
        
    def _save_cases(
        self,
        cases: List[Dict[str, Any]],
        output_dir: Path,
        top_k: Optional[int] = None,
    ) -> int:
        """
        Save case-level error visualizations to disk.

        This method is responsible only for the persistence of
        already-analyzed error cases. It does not perform error
        detection, prediction/ground-truth matching, IoU calculation,
        or error classification.

        Each error case is expected to contain an ``image_path`` and
        may optionally contain additional information required by
        ``_draw_case()`` for visualization, including:

            - ``image_path``:
                Path to the source image.

            - ``gt_bbox``:
                Ground-truth bounding box in
                ``[x1, y1, x2, y2]`` format.

            - ``pred_bbox``:
                Predicted bounding box in
                ``[x1, y1, x2, y2]`` format.

            - ``gt_class``:
                Ground-truth class name.

            - ``pred_class``:
                Predicted class name.

            - ``confidence``:
                Prediction confidence score.

            - ``iou``:
                IoU between the prediction and the corresponding
                ground-truth bounding box.

        Args:
            cases:
                List of structured error cases produced by
                ``ErrorAnalyzer``. Each case must contain at least
                ``image_path``.

            output_dir:
                Directory where the generated error visualizations
                will be stored. The directory is expected to be
                created by the calling public visualization method.

            top_k:
                Optional maximum number of error cases to visualize.
                If ``None``, all provided cases are processed.

        Returns:
            int:
                Number of error cases that were successfully loaded,
                annotated, and written to disk.

        Notes:
            - Missing image files are skipped.
            - Images that cannot be decoded by OpenCV are skipped.
            - Visualization-specific drawing logic is delegated to
            ``_draw_case()``.
            - The method does not modify the original ``cases`` list.
        """

        # --------------------------------------------------
        # Select cases
        # --------------------------------------------------

        selected_cases = (
            cases[:top_k]
            if top_k is not None
            else cases
        )

        saved_count = 0

        # --------------------------------------------------
        # Process cases
        # --------------------------------------------------

        for index, case in enumerate(
            selected_cases,
            start=1,
        ):

            image_path = case.get("image_path")

            if not image_path:
                logger.warning(
                    "Error case does not contain image_path"
                )
                continue

            image_path = Path(image_path)

            # --------------------------------------------------
            # Validate image path
            # --------------------------------------------------

            if not image_path.exists():
                logger.warning(
                    "Image not found: %s",
                    image_path,
                )
                continue

            # --------------------------------------------------
            # Load image
            # --------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                logger.warning(
                    "Could not read image: %s",
                    image_path,
                )
                continue

            # --------------------------------------------------
            # Draw error information
            # --------------------------------------------------

            image = self._draw_case(
                image=image,
                case=case,
            )

            # --------------------------------------------------
            # Save visualization
            # --------------------------------------------------

            save_path = (
                output_dir
                / f"{index:03d}_{image_path.name}"
            )

            if cv2.imwrite(
                str(save_path),
                image,
            ):
                saved_count += 1

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        logger.info(
            "Saved %d/%d error cases to %s",
            saved_count,
            len(selected_cases),
            output_dir,
        )

        return saved_count
    
    
    
    
    def _draw_case(
        self,
        image: np.ndarray,
        case: Dict[str, Any],
    ) -> np.ndarray:
        """
        Draw ground-truth and prediction information
        for a single error case.
        """

        image = image.copy()

        # --------------------------------------------------
        # Ground truth
        # --------------------------------------------------

        gt_bbox = case.get("gt_bbox")

        if gt_bbox is not None:

            x1, y1, x2, y2 = map(
                int,
                gt_bbox,
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            gt_class = case.get(
                "gt_class",
                "",
            )

            cv2.putText(
                image,
                f"GT: {gt_class}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        pred_bbox = case.get("pred_bbox")

        if pred_bbox is not None:

            x1, y1, x2, y2 = map(
                int,
                pred_bbox,
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            pred_class = case.get(
                "pred_class",
                "",
            )

            confidence = case.get(
                "confidence",
                0.0,
            )

            cv2.putText(
                image,
                f"Pred: {pred_class} ({confidence:.2f})",
                (x1, min(y2 + 22, image.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # IoU
        # --------------------------------------------------

        if "iou" in case:

            iou = case["iou"]

            cv2.putText(
                image,
                f"IoU: {iou:.3f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return image
        
    
    
    
    
        
        
        