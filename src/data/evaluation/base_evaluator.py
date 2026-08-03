"""
Base evaluator abstraction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from src.logging.logger import get_logger


logger = get_logger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all model evaluators.

    Each model-specific evaluator (YOLO, MMDetection, GroundingDINO)
    must implement the abstract methods.
    """

    def __init__(self, model, dataset):
        """
        Initialize evaluator with model and dataset.

        Args:
            model: Model instance (YOLO, MMDet, etc.)
            dataset: Dataset instance (must implement get_*_data methods)
        """
        self.model = model
        self.dataset = dataset
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("%s initialized", self.__class__.__name__)

    @abstractmethod
    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        Evaluate model on specified dataset split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    @abstractmethod
    def evaluate_per_class(self, split: str = "test") -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Dictionary mapping class names to metrics.
        """
        pass

    @abstractmethod
    def generate_report(self, output_path: Path, split: str = "test") -> Path:
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Path to save the report.
            split: 'train', 'val', or 'test'

        Returns:
            Path to the saved report.
        """
        pass

    @abstractmethod
    def _extract_metrics(self, results) -> Dict[str, float]:
        """
        Extract metrics from framework-specific results.

        Args:
            results: Framework-specific results object.

        Returns:
            Dictionary of extracted metrics.
        """
        pass

    def _get_split_samples(self, split: str) -> int:
        """Get number of samples in a split."""
        split_map = {
            "train": self.dataset.get_train_data,
            "val": self.dataset.get_val_data,
            "test": self.dataset.get_test_data,
        }
        return len(split_map.get(split, lambda: [])())