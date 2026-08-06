from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class BaseDataset(ABC):


    """
    Unified interface for all dataset implementations.

    This abstraction ensures that trainers and pipelines work with a
    consistent API regardless of the underlying dataset format (YOLO, COCO,
    Pascal VOC, etc.). Each concrete implementation encapsulates its own
    filesystem logic and format-specific parsing.

    Attributes:
        dataset_root: Root directory path of the dataset.
    """



    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)


    @abstractmethod
    def get_train_data(self) -> List[Dict[str, Any]]:
        """
        Load and return all training samples.

        Returns:
            List of dictionaries, each containing:
                - 'image_path': str, absolute path to the image
                - 'boxes': List[Dict], each with 'class_id' and 'bbox'
                (format may vary by implementation)
        """
        pass

    @abstractmethod
    def get_val_data(self) -> List[Dict[str, Any]]:
        """Load and return all validation samples."""
        pass

    @abstractmethod
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Load and return all test samples."""
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Return the list of class names.

        Returns:
            List of class names in order of their indices (0, 1, 2, ...).
        """
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return the total number of classes."""
        pass


    # ===== Auxiliary Methods =====


    @abstractmethod
    def load(self):
        """Load dataset resources."""
        pass

    @abstractmethod
    def validate(self):
        """Validate dataset integrity.
        
        
                Checks for:
            - Required directory structure
            - Matching image-label pairs
            - Valid annotation formats

        Raises:
            FileNotFoundError: If required directories are missing.
            ValueError: If dataset structure is invalid.
        
        """
        pass

    @abstractmethod
    def get_metadata(self):
        """Return dataset information.
        
        
                Returns:
            Dictionary containing:
                - num_classes: int
                - class_names: List[str]
                - train_samples: int
                - val_samples: int
                - test_samples: int
                - and any additional implementation-specific info.
        
        
        
        """
        pass