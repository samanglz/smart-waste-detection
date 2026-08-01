from pathlib import Path
import yaml
from .base_dataset import BaseDataset
from src.logging.logger import get_logger



logger = get_logger(__name__)


class YOLODataset(BaseDataset):

    """
    Dataset implementation for YOLO formatted datasets.
    """


    def __init__(self, dataset_root: Path):

        super().__init__(dataset_root)
        self.yaml_path = self.dataset_root / "data.yaml"
        self.classes = []
        self.num_classes = 0


    def validate(self):
        """
        Validate YOLO dataset structure.
        """
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset yaml not found: {self.yaml_path}"
            )

        logger.info(
            "Yaml file exists: %s",
            self.yaml_path
        )
        

        required_dirs = [
            self.dataset_root / "images",
            self.dataset_root / "labels",
        ]

        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(
                    f"Missing directory: {directory}"
                )

            logger.info(
                "Found directory: %s",
                directory
            )
        logger.info(
            "YOLO dataset validation passed"
        )
        
        

    def load(self) -> None:
        """
        Load YOLO dataset metadata.
        """

        with open(self.yaml_path, "r") as file:
            data = yaml.safe_load(file)

        self.num_classes = data["nc"]
        self.classes = data["names"]
        
        if len(data["names"]) != data["nc"]:
            raise ValueError(
                "Class count mismatch"
            )
        
        logger.info(
            "Loaded YOLO dataset with %d classes",
            self.num_classes
        )

        logger.info(
            "Classes: %s",
            self.classes
        )