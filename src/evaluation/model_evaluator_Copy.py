"""
Model evaluation implementation.

Provides a clean interface for evaluating trained YOLO models
on dataset splits with metrics extraction and report generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional,Tuple, List
from collections import defaultdict
from PIL import Image



from src.models.yolo.yolo_model import YOLOModel
from src.data.yolo_dataset import YOLODataset
from src.logging.logger import get_logger
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.evaluation.prioritization.error_prioritizer import ( 
    ErrorPrioritizer,
                                                             
    )



logger = get_logger(__name__)


class ModelEvaluator_copy:
    """
    Evaluates trained YOLO models on dataset splits.

    Supports:
        - Evaluation on train/val/test splits
        - Per-class metric extraction
        - JSON report generation
    """

    def __init__(self, model: YOLOModel, dataset: YOLODataset):
        """
        Initialize evaluator with model and dataset.

        Args:
            model: Trained YOLOModel instance.
            dataset: YOLODataset instance providing data access.
        """
        self.model = model
        self.dataset = dataset
        self._cached_results = {}
        logger.info("ModelEvaluator initialized")
        self.error_analyzer = ErrorAnalyzer(
            self.dataset.get_class_names()
)


    def _get_val_results(self, split: str = "test"):
        """
        Get validation results with caching to avoid multiple calls.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Validation results object.
        """
        cache_key = f"val_{split}"

        if cache_key not in self._cached_results:
            logger.info("Running validation on %s split (cached)", split)
            results = self.model.val(
                data=str(self.dataset.yaml_path),
                split=split,
            )
            self._cached_results[cache_key] = results
        else:
            logger.info("Using cached validation results for %s split", split)

        return self._cached_results[cache_key]



    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        Evaluate model on the specified dataset split.

        Args:
            split: One of 'train', 'val', 'test' (default: 'test')

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating on %s split...", split)

        results = self._get_val_results(split)
        metrics = self._extract_metrics(results)
        logger.info("Evaluation complete on %s split", split)

        return metrics

    def evaluate_per_class(self, split: str = "test") -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics for detailed analysis.

        This method provides a public interface for accessing per-class metrics
        without generating a full report.

        Args:
            split: 'train', 'val', or 'test' (default: 'test')

        Returns:
            Dictionary mapping class names to metrics (precision, recall, ap).
        """
        logger.info("Computing per-class metrics on %s split...", split)
        results = self._get_val_results(split)
        return self._extract_per_class_metrics(results)



    def prioritize_errors(
        self,
        split: str = "test",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Prioritize detected errors on a dataset split.

        The method reuses the existing error-analysis pipeline
        and delegates prioritization to ErrorPrioritizer.
        """

        error_report = self.analyze_errors(
            split=split,
        )

        prioritizer = ErrorPrioritizer()

        return {
            "classes": prioritizer.rank_classes(
                metrics=error_report["per_class_metrics"],
                criterion="ap",
                top_k=top_k,
            ),
            "confusions": prioritizer.rank_confusions(
                confusion=error_report["error_analysis"]["confusion"],
                criterion="count",
                top_k=top_k,
            ),
            "false_positives": prioritizer.rank_false_positives(
                false_positives=error_report["error_analysis"]["false_positives"],
                criterion="count",
                top_k=top_k,
            ),
            "false_negatives": prioritizer.rank_false_negatives(
                false_negatives=error_report["error_analysis"]["false_negatives"],
                criterion="count",
                top_k=top_k,
            ),
        }



    def generate_report(self, output_path: Path, split: str = "test") -> Path:
        """
        Generate a comprehensive JSON evaluation report.

        Args:
            output_path: Path to save the report (JSON file).
            split: One of 'train', 'val', 'test' (default: 'test')

        Returns:
            Path to the saved report.
        """
        logger.info("Generating evaluation report: %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = self._get_val_results(split)

        metrics = self._extract_metrics(results)
        per_class =  self._extract_per_class_metrics(results)

        # Get sample count for the split
        split_samples = {
            "train": len(self.dataset.get_train_data()),
            "val": len(self.dataset.get_val_data()),
            "test": len(self.dataset.get_test_data()),
        }

        report = {
            "dataset": {
                "split": split,
                "num_classes": self.dataset.get_num_classes(),
                "class_names": self.dataset.get_class_names(),
                "total_samples": split_samples.get(split, 0),
            },
            "metrics": metrics,
            "per_class_metrics": per_class,
            "model": {
                "type": "YOLO",
                "weight_path": str(self.model.weight_path),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Report saved to: %s", output_path)
        return output_path

    def _extract_metrics(self, results) -> Dict[str, float]:
        """
        Extract standard YOLO metrics from validation results.

        Args:
            results: YOLO validation results object.

        Returns:
            Dictionary of extracted metrics.
        """
        metrics = {}

        if hasattr(results, "box"):
            if hasattr(results.box, "map50"):
                metrics["mAP_0.5"] = float(results.box.map50)
            if hasattr(results.box, "map75"):
                metrics["mAP_0.75"] = float(results.box.map75)
            if hasattr(results.box, "map"):
                metrics["mAP_0.5_0.95"] = float(results.box.map)

        if hasattr(results, "speed"):
            metrics["speed"] = results.speed

        return metrics
    
    
    
    def _extract_per_class_metrics(self, results) -> Dict[str, Dict[str, float]]:
        """Extract per-class metrics from validation results."""
        class_names = self.dataset.get_class_names()
        per_class = {}

        # استفاده از ap_class_index و maps (مربوط به هر کلاس)
        if hasattr(results, 'ap_class_index') and hasattr(results.box, 'maps'):
            ap_class_index = results.ap_class_index
            maps = results.box.maps  # mAP برای هر کلاس (لیست)
            
            # استخراج precision و recall از results.box
            p_list = getattr(results.box, 'p', None)  # precision per class
            r_list = getattr(results.box, 'r', None)  # recall per class
            ap_list = getattr(results.box, 'ap', None)  # AP per class

            for idx, class_idx in enumerate(ap_class_index):
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                
                # مقادیر پیش‌فرض
                prec_val = 0.0
                rec_val = 0.0
                ap_val = 0.0
                
                # اگر p_list موجود باشد
                if p_list is not None and idx < len(p_list):
                    prec_val = float(p_list[idx])
                # اگر r_list موجود باشد
                if r_list is not None and idx < len(r_list):
                    rec_val = float(r_list[idx])
                # اگر ap_list موجود باشد
                if ap_list is not None and idx < len(ap_list):
                    ap_val = float(ap_list[idx])
                # اگر maps موجود باشد و ap_list نبود
                elif maps is not None and class_idx < len(maps):
                    ap_val = float(maps[class_idx])
                
                per_class[class_name] = {
                    'precision': prec_val,
                    'recall': rec_val,
                    'ap': ap_val,
                }
        else:
            self.logger.warning("ap_class_index or maps not found in results")

        return per_class
    
    
        # ============================================================



    # Error Analysis Methods
    # ============================================================
    def analyze_errors(
        self,
        split: str = "test",
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Analyze model errors on a dataset split.

        This method orchestrates the error-analysis pipeline:
            1. Load standard evaluation metrics.
            2. Load predictions and ground truths.
            3. Match predictions with ground truths.
            4. Analyze confusion, false positives, and false negatives.
            5. Build class performance summary.
            6. Generate recommendations.
            7. Build and optionally save the final report.

        Args:
            split: Dataset split to analyze.
            output_path: Optional path for saving the error-analysis report.

        Returns:
            Complete error-analysis report.
        """

        logger.info("=" * 60)
        logger.info("Error Analysis on %s split", split)
        logger.info("=" * 60)

        # ---------------------------------------------------------
        # 1. Standard evaluation metrics
        # ---------------------------------------------------------

        results = self._get_val_results(split)
        per_class = self._extract_per_class_metrics(results)

        # ---------------------------------------------------------
        # 2. Collect error-analysis data
        # ---------------------------------------------------------

        analysis_data = self._collect_error_analysis_data(split)

        # ---------------------------------------------------------
        # 3. Build class performance summary
        # ---------------------------------------------------------

        class_performance_summary = (
            self._build_class_performance_summary(per_class)
        )

        # ---------------------------------------------------------
        # 4. Generate recommendations
        # ---------------------------------------------------------

        recommendations = self._generate_recommendations(
            per_class
        )

        # ---------------------------------------------------------
        # 5. Build final report
        # ---------------------------------------------------------

        error_report = {
            "split": split,
            "per_class_metrics": per_class,
            "class_performance_summary": class_performance_summary,
            "error_analysis": {
                "confusion": analysis_data["confusion"],
                "false_positives": analysis_data["false_positives"],
                "false_negatives": analysis_data["false_negatives"],
            },
            "recommendations": recommendations,
        }

        # ---------------------------------------------------------
        # 6. Save report
        # ---------------------------------------------------------

        if output_path:
            self._save_error_report(
                error_report,
                output_path,
            )

        return error_report
        
    
    


    def _calculate_iou(self, box1, box2) -> float:
        """
        Calculate IoU between two bounding boxes.

        Boxes are expected in:
        [x1, y1, x2, y2] format.
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0.0, x2 - x1)
        intersection_height = max(0.0, y2 - y1)

        intersection = (
            intersection_width *
            intersection_height
        )

        area1 = (
            max(0.0, box1[2] - box1[0]) *
            max(0.0, box1[3] - box1[1])
        )

        area2 = (
            max(0.0, box2[2] - box2[0]) *
            max(0.0, box2[3] - box2[1])
        )

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union







    def _analyze_confusion(
        self,
        matches,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze class confusion from matched predictions.
        """

        class_names = self.dataset.get_class_names()

        confusion = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "count": 0,
                    "mean_iou": 0.0,
                    "mean_confidence": 0.0,
                    "images": [],
                    "cases": [],
                }
            )
        )

        for match in matches:

            pred_class = match["pred_class"]
            gt_class = match["gt_class"]

            # Correct classification → not a confusion
            if pred_class == gt_class:
                continue

            gt_name = (
                class_names[gt_class]
                if gt_class < len(class_names)
                else f"class_{gt_class}"
            )

            pred_name = (
                class_names[pred_class]
                if pred_class < len(class_names)
                else f"class_{pred_class}"
            )

            confusion_data = confusion[gt_name][pred_name]

            confusion_data["count"] += 1

            confusion_data["mean_iou"] += match["iou"]

            confidence = match["prediction"].get(
                "confidence",
                0.0,
            )

            confusion_data["mean_confidence"] += confidence

            image_path = match.get("image_path")

            if image_path is not None:

                confusion_data["images"].append(
                    image_path
                )

                confusion_data["cases"].append({
                    "image_path": image_path,
                    "gt_class": gt_name,
                    "pred_class": pred_name,
                    "gt_bbox": match["ground_truth"]["bbox"],
                    "pred_bbox": match["prediction"]["bbox"],
                    "iou": match["iou"],
                    "confidence": confidence,
                })

        # Convert accumulated sums to means
        for gt_name, predictions in confusion.items():

            for pred_name, data in predictions.items():

                count = data["count"]

                if count > 0:
                    data["mean_iou"] /= count
                    data["mean_confidence"] /= count

        return {
            gt_name: dict(predictions)
            for gt_name, predictions in confusion.items()
        }
                
        
        



    def _analyze_false_positives(
        self,
        unmatched_predictions,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze false-positive predictions.

        Returns:
            Dictionary containing count, mean confidence,
            and affected images per predicted class.
        """

        class_names = self.dataset.get_class_names()

        false_positives = defaultdict(
            lambda: {
                "count": 0,
                "mean_confidence": 0.0,
                "images": [],
            }
        )

        for item in unmatched_predictions:

            prediction = item["prediction"]

            class_id = prediction["class_id"]

            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            )

            data = false_positives[class_name]

            data["count"] += 1

            confidence = prediction.get("confidence", 0.0)

            data["mean_confidence"] += confidence

            image_path = item.get("image_path")

            if image_path is not None:
                data["images"].append(image_path)

        # Convert accumulated confidence to mean confidence
        for class_name, data in false_positives.items():

            if data["count"] > 0:
                data["mean_confidence"] /= data["count"]

        return dict(false_positives)






    def _analyze_false_negatives(
        self,
        unmatched_ground_truths,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze false-negative ground-truth objects.

        Returns:
            Dictionary containing count and affected images
            per ground-truth class.
        """

        class_names = self.dataset.get_class_names()

        false_negatives = defaultdict(
            lambda: {
                "count": 0,
                "images": [],
            }
        )

        for item in unmatched_ground_truths:

            ground_truth = item["ground_truth"]

            class_id = ground_truth["class_id"]

            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            )

            data = false_negatives[class_name]

            data["count"] += 1

            image_path = item.get("image_path")

            if image_path is not None:
                data["images"].append(image_path)

        return dict(false_negatives)









    def _match_predictions_to_ground_truth(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        image_path: str,
        iou_threshold: float = 0.5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Match predictions with ground-truth objects using IoU.

        Matching is class-agnostic intentionally.

        This allows us to detect classification confusions such as:

            GT: glass
            Pred: plastic
            IoU: 0.91

        which is a classification error rather than a localization error.

        Matching strategy:
            1. Sort predictions by confidence descending.
            2. For each prediction, find the unmatched GT
            with the highest IoU.
            3. If IoU >= threshold -> match.
            4. Otherwise -> unmatched prediction (FP).
            5. Remaining GTs -> unmatched ground truths (FN).

        Args:
            predictions:
                Model predictions for one image.

            ground_truths:
                Ground-truth objects for one image.

            image_path:
                Path of the image.

            iou_threshold:
                Minimum IoU required for matching.

        Returns:
            Dictionary containing:
                matches
                unmatched_predictions
                unmatched_ground_truths
        """

        matches: List[Dict[str, Any]] = []
        unmatched_predictions: List[Dict[str, Any]] = []
        unmatched_ground_truths: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # 1. Normalize image path
        # ---------------------------------------------------------

        image_path = str(Path(image_path))

        # ---------------------------------------------------------
        # 2. Sort predictions by confidence
        # ---------------------------------------------------------

        sorted_predictions = sorted(
            enumerate(predictions),
            key=lambda item: item[1].get("confidence", 0.0),
            reverse=True,
        )

        matched_gt_indices = set()

        # ---------------------------------------------------------
        # 3. Match predictions to GT
        # ---------------------------------------------------------

        for pred_idx, prediction in sorted_predictions:

            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, ground_truth in enumerate(ground_truths):

                # GT already matched
                if gt_idx in matched_gt_indices:
                    continue

                iou = self._calculate_iou(
                    prediction["bbox"],
                    ground_truth["bbox"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # -----------------------------------------------------
            # Successful spatial match
            # -----------------------------------------------------

            if (
                best_gt_idx is not None
                and best_iou >= iou_threshold
            ):

                matched_gt_indices.add(best_gt_idx)

                ground_truth = ground_truths[best_gt_idx]

                matches.append(
                    {
                        "image_path": image_path,

                        "prediction_index": pred_idx,
                        "ground_truth_index": best_gt_idx,

                        "pred_class": prediction["class_id"],
                        "gt_class": ground_truth["class_id"],

                        "iou": float(best_iou),

                        "prediction": {
                            "class_id": prediction["class_id"],
                            "confidence": float(
                                prediction.get("confidence", 0.0)
                            ),
                            "bbox": prediction["bbox"],
                        },

                        "ground_truth": {
                            "class_id": ground_truth["class_id"],
                            "bbox": ground_truth["bbox"],
                        },
                    }
                )

            # -----------------------------------------------------
            # No suitable GT -> false positive
            # -----------------------------------------------------

            else:

                unmatched_predictions.append(
                    {
                        "image_path": image_path,

                        "prediction_index": pred_idx,

                        "prediction": {
                            "class_id": prediction["class_id"],
                            "confidence": float(
                                prediction.get("confidence", 0.0)
                            ),
                            "bbox": prediction["bbox"],
                        },
                    }
                )

        # ---------------------------------------------------------
        # 4. Remaining GTs -> false negatives
        # ---------------------------------------------------------

        for gt_idx, ground_truth in enumerate(ground_truths):

            if gt_idx in matched_gt_indices:
                continue

            unmatched_ground_truths.append(
                {
                    "image_path": image_path,

                    "ground_truth_index": gt_idx,

                    "ground_truth": {
                        "class_id": ground_truth["class_id"],
                        "bbox": ground_truth["bbox"],
                    },
                }
            )

        return {
            "matches": matches,
            "unmatched_predictions": unmatched_predictions,
            "unmatched_ground_truths": unmatched_ground_truths,
        }





    def _aggregate_detection_errors(
        self,
        predictions_by_image: Dict[str, List[Dict[str, Any]]],
        ground_truths_by_image: Dict[str, List[Dict[str, Any]]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Aggregate detection errors across all images.

        The aggregation combines image-level matching results into
        dataset-level error categories.

        Returns:
            {
                "matches": [...],
                "unmatched_predictions": [...],
                "unmatched_ground_truths": [...]
            }
        """

        all_matches: List[Dict[str, Any]] = []
        all_unmatched_predictions: List[Dict[str, Any]] = []
        all_unmatched_ground_truths: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Normalize image keys
        # ---------------------------------------------------------

        normalized_predictions = {
            str(Path(image_path)): predictions
            for image_path, predictions
            in predictions_by_image.items()
        }

        normalized_ground_truths = {
            str(Path(image_path)): ground_truths
            for image_path, ground_truths
            in ground_truths_by_image.items()
        }

        # ---------------------------------------------------------
        # All images
        # ---------------------------------------------------------

        image_paths = (
            set(normalized_predictions.keys())
            | set(normalized_ground_truths.keys())
        )

        logger.info(
            "Aggregating detection errors for %d images",
            len(image_paths),
        )

        # ---------------------------------------------------------
        # Process each image
        # ---------------------------------------------------------

        for image_path in image_paths:

            predictions = normalized_predictions.get(
                image_path,
                [],
            )

            ground_truths = normalized_ground_truths.get(
                image_path,
                [],
            )

            result = self._match_predictions_to_ground_truth(
                predictions=predictions,
                ground_truths=ground_truths,
                image_path=image_path,
                iou_threshold=iou_threshold,
            )

            all_matches.extend(
                result["matches"]
            )

            all_unmatched_predictions.extend(
                result["unmatched_predictions"]
            )

            all_unmatched_ground_truths.extend(
                result["unmatched_ground_truths"]
            )

        logger.info(
            "Detection matching completed: "
            "%d matches, %d unmatched predictions, "
            "%d unmatched ground truths",
            len(all_matches),
            len(all_unmatched_predictions),
            len(all_unmatched_ground_truths),
        )

        return {
            "matches": all_matches,
            "unmatched_predictions":
                all_unmatched_predictions,
            "unmatched_ground_truths":
                all_unmatched_ground_truths,
        }





    def _get_predictions_by_image(
        self,
        split: str,
        conf_threshold: float = 0.25,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate model predictions for every image in a dataset split.

        Predictions are normalized into a framework-independent format:

            {
                "image_path": [
                    {
                        "class_id": int,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2]
                    }
                ]
            }
        """

        logger.info(
            "Generating predictions for %s split "
            "(confidence >= %.2f)",
            split,
            conf_threshold,
        )

        split_getters = {
            "train": self.dataset.get_train_data,
            "val": self.dataset.get_val_data,
            "test": self.dataset.get_test_data,
        }

        if split not in split_getters:

            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of: "
                f"{list(split_getters.keys())}"
            )

        samples = split_getters[split]()

        predictions_by_image: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        for sample in samples:

            image_path = Path(
                sample["image_path"]
            )

            if not image_path.exists():

                logger.warning(
                    "Image not found: %s",
                    image_path,
                )

                continue

            try:

                results = self.model.predict(
                    source=str(image_path),
                    conf=conf_threshold,
                    verbose=False,
                )

            except Exception as exc:

                logger.error(
                    "Prediction failed for %s: %s",
                    image_path,
                    exc,
                )

                continue

            predictions: List[
                Dict[str, Any]
            ] = []

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    predictions.append(
                        {
                            "class_id": int(
                                box.cls[0]
                            ),

                            "confidence": float(
                                box.conf[0]
                            ),

                            "bbox": [
                                float(value)
                                for value in box.xyxy[0].tolist()
                            ],
                        }
                    )

            # Normalize path to string
            normalized_path = str(
                image_path
            )

            predictions_by_image[
                normalized_path
            ] = predictions

        logger.info(
            "Generated predictions for %d images",
            len(predictions_by_image),
        )

        return predictions_by_image
        
      
    

    def _get_ground_truths_by_image(
        self,
        split: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load ground-truth annotations for every image.

        YOLO annotations are stored as:

            [cx, cy, width, height]

        normalized to [0, 1].

        They are converted into pixel coordinates:

            [x1, y1, x2, y2]
        """

        logger.info(
            "Loading ground truths for %s split",
            split,
        )

        split_getters = {
            "train": self.dataset.get_train_data,
            "val": self.dataset.get_val_data,
            "test": self.dataset.get_test_data,
        }

        if split not in split_getters:

            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of: "
                f"{list(split_getters.keys())}"
            )

        samples = split_getters[split]()

        ground_truths_by_image: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        for sample in samples:

            image_path = Path(
                sample["image_path"]
            )

            if not image_path.exists():

                logger.warning(
                    "Image not found: %s",
                    image_path,
                )

                continue

            boxes = sample["boxes"]

            try:

                with Image.open(image_path) as image:

                    image_width, image_height = (
                        image.size
                    )

            except Exception as exc:

                logger.error(
                    "Failed to read image %s: %s",
                    image_path,
                    exc,
                )

                continue

            ground_truths: List[
                Dict[str, Any]
            ] = []

            for box in boxes:

                class_id = int(
                    box["class_id"]
                )

                cx, cy, width, height = (
                    box["bbox"]
                )

                # -------------------------------------------------
                # Convert normalized YOLO coordinates
                # to pixel coordinates
                # -------------------------------------------------

                cx_pixel = cx * image_width
                cy_pixel = cy * image_height

                width_pixel = (
                    width * image_width
                )

                height_pixel = (
                    height * image_height
                )

                x1 = (
                    cx_pixel
                    - width_pixel / 2
                )

                y1 = (
                    cy_pixel
                    - height_pixel / 2
                )

                x2 = (
                    cx_pixel
                    + width_pixel / 2
                )

                y2 = (
                    cy_pixel
                    + height_pixel / 2
                )

                ground_truths.append(
                    {
                        "class_id": class_id,

                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],
                    }
                )

            # -----------------------------------------------------
            # IMPORTANT:
            # Always use str(Path(...))
            # -----------------------------------------------------

            normalized_path = str(
                image_path
            )

            ground_truths_by_image[
                normalized_path
            ] = ground_truths

        logger.info(
            "Loaded ground truths for %d images",
            len(ground_truths_by_image),
        )

        return ground_truths_by_image


    def _get_split_image_paths(
        self,
        split: str,
    ):
        """
        Return image paths belonging to a dataset split.
        """

        split_getters = {
            "train": self.dataset.get_train_data,
            "val": self.dataset.get_val_data,
            "test": self.dataset.get_test_data,
        }

        try:
            getter = split_getters[split]
        except KeyError:
            raise ValueError(
                f"Unsupported split: {split}. "
                f"Expected one of: {list(split_getters)}"
            )

        data = getter()

        return [
            Path(item["image_path"])
            for item in data
        ]
  
  
  


    def _generate_recommendations(self, per_class: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Generate recommendations for improving model performance.

        Args:
            per_class: Per-class metrics dictionary.

        Returns:
            Dictionary mapping class names to recommendations.
        """
        recommendations = {}

        for class_name, metrics in per_class.items():
            ap = metrics['ap']
            recall = metrics['recall']
            precision = metrics['precision']

            if ap < 0.85:
                if recall < 0.8:
                    recommendations[class_name] = "Add more training samples for this class"
                elif precision < 0.8:
                    recommendations[class_name] = "Reduce false positives by adjusting confidence threshold"
                else:
                    recommendations[class_name] = "Consider data augmentation specific to this class"
            else:
                recommendations[class_name] = "Performance is good, no specific recommendation"

        return recommendations
    
    
    
    
    
    
    
    def _collect_error_analysis_data(
        self,
        split: str,
    ) -> Dict[str, Any]:
        """
        Collect all data required for error analysis.
        """

        # ---------------------------------------------------------
        # Standard evaluation metrics
        # ---------------------------------------------------------

        results = self._get_val_results(split)

        per_class = self._extract_per_class_metrics(results)

        # ---------------------------------------------------------
        # Predictions and ground truths
        # ---------------------------------------------------------

        predictions_by_image = (
            self._get_predictions_by_image(split)
        )

        ground_truths_by_image = (
            self._get_ground_truths_by_image(split)
        )

        # ---------------------------------------------------------
        # Match predictions with ground truths
        # ---------------------------------------------------------

        detection_errors = self._aggregate_detection_errors(
            predictions_by_image=predictions_by_image,
            ground_truths_by_image=ground_truths_by_image,
            iou_threshold=0.5,
        )

        # ---------------------------------------------------------
        # Error categories
        # ---------------------------------------------------------

        confusion = self.error_analyzer.analyze_confusion(
            detection_errors["matches"]
        )

        false_positives = self.error_analyzer.analyze_false_positives(
            detection_errors["unmatched_predictions"]
        )

        false_negatives = self.error_analyzer.analyze_false_negatives(
            detection_errors["unmatched_ground_truths"]
        )


        return {
            "per_class_metrics": per_class,
            "confusion": confusion,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        
    
    
    def _get_worst_class(
        self,
        per_class: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """
        Return the class with the lowest AP.
        """

        if not per_class:
            return None

        return min(
            per_class.items(),
            key=lambda item: item[1]["ap"],
        )[0]
        
        
        
        
        
        
    def _build_class_performance_summary(
        self,
        per_class: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build qualitative performance summary for each class.
        """

        summary = {}

        for class_name, metrics in per_class.items():

            ap = metrics["ap"]

            if ap > 0.9:
                performance_level = "good"
            elif ap > 0.8:  
                performance_level = "medium"
            else:
                performance_level = "poor"

            summary[class_name] = {
                "ap": metrics["ap"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "performance_level": performance_level,
            }

        return summary
    
    
        
    def _save_error_report(
        self,
        error_report: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Save error-analysis report as JSON.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                error_report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Error analysis saved to: %s",
            output_path,
        )