"""
Model evaluation implementation.

Provides a clean interface for evaluating trained YOLO models
on dataset splits with metrics extraction and report generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.models.yolo.yolo_model import YOLOModel
from src.data.yolo_dataset import YOLODataset
from src.logging.logger import get_logger


logger = get_logger(__name__)


class ModelEvaluator:
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
        logger.info("ModelEvaluator initialized")

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        Evaluate model on the specified dataset split.

        Args:
            split: One of 'train', 'val', 'test' (default: 'test')

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating on %s split...", split)

        results = self.model.val(
            data=str(self.dataset.yaml_path),
            split=split,
        )

        metrics = self._extract_metrics(results)
        logger.info("Evaluation complete on %s split", split)

        return metrics

    def evaluate_per_class(self, split: str = "test") -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics for detailed analysis.

        Args:
            split: One of 'train', 'val', 'test' (default: 'test')

        Returns:
            Dictionary mapping class names to precision/recall/ap.
        """
        logger.info("Computing per-class metrics on %s split...", split)

        results = self.model.val(
            data=self.dataset.get_dataset_config(),
            split=split,
        )

        class_names = self.dataset.get_class_names()
        per_class = {}

        if hasattr(results, 'ap_class_index'):
            for idx, class_idx in enumerate(results.ap_class_index):
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                per_class[class_name] = {
                    'precision': float(results.class_precision[idx]) if hasattr(results, 'class_precision') else 0.0,
                    'recall': float(results.class_recall[idx]) if hasattr(results, 'class_recall') else 0.0,
                    'ap': float(results.class_ap[idx]) if hasattr(results, 'class_ap') else 0.0,
                }

        return per_class

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

        metrics = self.evaluate(split)
        per_class = self.evaluate_per_class(split)

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