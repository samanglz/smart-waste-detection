"""
YOLO dataset implementation.
"""

from pathlib import Path
import yaml
from .base_dataset import BaseDataset
from src.logging.logger import get_logger
from typing import List, Dict, Any
import json
from torch.utils.data import WeightedRandomSampler



logger = get_logger(__name__)


class YOLODataset(BaseDataset):

    """
    YOLO dataset handler.

    Only this class knows about YOLO directory structure.
    If paths change, only this class is modified.
    """


    def __init__(self, dataset_root: Path):

        super().__init__(dataset_root)
        self._yaml_path = self.dataset_root / "data.yaml"
        self.classes = []
        self.num_classes = 0



        # ===== Encapsulated: only this class knows these paths =====
        self.train_img_dir = self.dataset_root / "train" / "images"
        self.train_label_dir = self.dataset_root / "train" / "labels"
        self.val_img_dir = self.dataset_root / "val" / "images"
        self.val_label_dir = self.dataset_root / "val" / "labels"
        self.test_img_dir = self.dataset_root / "test" / "images"
        self.test_label_dir = self.dataset_root / "test" / "labels"
        # ============================================================

        # برای راحتی کار در متدهای دیگر، یک دیکشنری هم بسازیم
        self._paths = {
            "train": {
                "images": self.train_img_dir,
                "labels": self.train_label_dir,
            },
            "val": {
                "images": self.val_img_dir,
                "labels": self.val_label_dir,
            },
            "test": {
                "images": self.test_img_dir,
                "labels": self.test_label_dir,
            },
        }


        self.load()
        self.validate()

    @property
    def yaml_path(self) -> Path:
        """Return the path to data.yaml file (read-only)."""
        return self._yaml_path




    def validate(self) -> bool:
        """
        Validate YOLO dataset structure.

        Checks:
            - data.yaml exists
            - train/val/test directories exist
            - Each directory contains images/ and labels/ subdirectories
            - Warns about images without labels

        Returns:
            True if validation passes.

        Raises:
            FileNotFoundError: If required files or directories are missing.
        """
        #1. checking existence of data.ymal file.
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset yaml not found: {self.yaml_path}"
            )

        logger.info("YAML file exists: %s", self.yaml_path)

        # 2.check for existence train/val/test
        split_dirs = [
            ("train", self.train_img_dir, self.train_label_dir),
            ("val", self.val_img_dir, self.val_label_dir),
            ("test", self.test_img_dir, self.test_label_dir),
        ]

        for split_name, img_dir, label_dir in split_dirs:
            if not img_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {img_dir} for {split_name} split"
                )
            if not label_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {label_dir} for {split_name} split"
                )
            logger.info("Found %s split directories", split_name)

        # 3. warning for images without labels
        for split_name, img_dir, label_dir in split_dirs:
            img_files = {
                f.stem for f in img_dir.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            }
            label_files = {
                f.stem for f in label_dir.iterdir()
                if f.suffix == '.txt'
            }

            missing_labels = img_files - label_files
            if missing_labels:
                logger.warning(
                    "%d images without labels in %s split",
                    len(missing_labels),
                    split_name
                )

        logger.info("YOLO dataset validation passed")
        return True     

    def load(self) -> None:
        """
        Load YOLO dataset metadata.
        """
        with open(self.yaml_path, "r", encoding='utf-8') as file:
            data = yaml.safe_load(file)

        self.num_classes = data["nc"]
        self.classes = data["names"]
        
        # اگر names به صورت دیکشنری بود، به لیست تبدیل کن
        if isinstance(self.classes, dict):
            self.classes = [self.classes[i] for i in sorted(self.classes.keys())]
        
        if len(self.classes) != self.num_classes:
            raise ValueError("Class count mismatch")
        
        logger.info("Loaded YOLO dataset with %d classes", self.num_classes)
        logger.info("Classes: %s", self.classes)


    def _load_split(self, split: str) -> List[Dict[str, Any]]:  # ← اینجا دیگر داخل load نیست
        """
        Load all samples from a split (train/val/test).

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            List of sample dictionaries, each containing:
                - image_path: str, absolute path to the image
                - boxes: List[Dict], each with 'class_id' and 'bbox'
        """
        if split not in self._paths:
            raise ValueError(f"Invalid split: {split}")

        img_dir = self._paths[split]["images"]
        label_dir = self._paths[split]["labels"]
        samples = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        if not img_dir.exists():
            logger.warning("Image directory not found: %s", img_dir)
            return samples

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in valid_exts:
                continue

            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            boxes = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, cx, cy, w, h = map(float, parts)
                            boxes.append({
                                'class_id': int(class_id),
                                'bbox': [cx, cy, w, h]
                            })
            except Exception as e:
                logger.error("Error reading label file %s: %s", label_path, e)
                continue

            samples.append({
                'image_path': str(img_path),
                'boxes': boxes
            })

        logger.debug("Loaded %d samples from %s split", len(samples), split)
        return samples
        
    
    def get_train_data(self) -> List[Dict[str, Any]]:
        """Load and return all training samples."""
        return self._load_split("train")

    def get_val_data(self) -> List[Dict[str, Any]]:
        """Load and return all validation samples."""
        return self._load_split("val")

    def get_test_data(self) -> List[Dict[str, Any]]:
        """Load and return all test samples."""
        return self._load_split("test")

    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return self.classes

    def get_num_classes(self) -> int:
        """Return the total number of classes."""
        return self.num_classes


    def get_metadata(self) -> Dict[str, Any]:
        """Dataset statistics."""
        return {
            'num_classes': self.num_classes,
            'class_names': self.classes,
            'train_samples': len(self.get_train_data()),
            'val_samples': len(self.get_val_data()),
            'test_samples': len(self.get_test_data()),
        }
        
    
    

    def build_weighted_sampler(
        self,
        image_paths: list[Path],
        weight_file: Path,
    ):
        """
        Build WeightedRandomSampler from sampling_weights.json.

        Supports absolute paths and filenames.
        """

        with open(
            weight_file,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        raw_weight_table = data["weights"]

        # Normalize path-based keys
        weight_table = {
            str(Path(path).resolve()).lower(): float(weight)
            for path, weight in raw_weight_table.items()
        }

        weights = []

        for image_path in image_paths:

            image_path = Path(image_path)

            normalized_path = str(
                image_path.resolve()
            ).lower()

            weight = weight_table.get(
                normalized_path,
                1.0,
            )

            weights.append(
                weight
            )

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )