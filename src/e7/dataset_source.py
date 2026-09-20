"""
E7 dataset source.

E7 uses:
    - E2 train split as the training background/source dataset
    - Original processed val/test splits for evaluation

Validation and test must remain unchanged across experiments.
"""

from pathlib import Path
from typing import Any, Dict, List

from src.data.yolo_dataset import YOLODataset


class E7DatasetSource:
    """
    Provides the dataset splits required by E7.

    Training:
        data/e2_targeted_test/train

    Validation/Test:
        data/processed/val
        data/processed/test

    The original YOLODataset implementation remains unchanged.
    """

    def __init__(
        self,
        train_dataset_root: Path,
        evaluation_dataset_root: Path,
    ):
        self.train_dataset_root = Path(train_dataset_root)
        self.evaluation_dataset_root = Path(evaluation_dataset_root)

        # E2 train uses the same YOLO structure, but contains
        # only the train split, so we load its paths directly.
        self.train_img_dir = (
            self.train_dataset_root / "images"
        )
        self.train_label_dir = (
            self.train_dataset_root / "labels"
        )



        self._validate_train_root()

        # Original complete dataset.
        # This gives us:
        #   - class metadata
        #   - validation split
        #   - test split
        self._evaluation_dataset = YOLODataset(
            self.evaluation_dataset_root
        )


    def _validate_train_root(self) -> None:
        """Validate that the E2 training split exists."""

        if not self.train_dataset_root.exists():
            raise FileNotFoundError(
                f"E7 training source not found: "
                f"{self.train_dataset_root}"
            )

        if not self.train_img_dir.exists():
            raise FileNotFoundError(
                f"E7 training images directory not found: "
                f"{self.train_img_dir}"
            )

        if not self.train_label_dir.exists():
            raise FileNotFoundError(
                f"E7 training labels directory not found: "
                f"{self.train_label_dir}"
            )

    def get_train_data(self) -> List[Dict[str, Any]]:
        """
        Return E2 training samples.

        The E2 train split is intentionally used as the
        background/source pool for E7.
        """

        return self._load_train_split()

    def _load_train_split(self) -> List[Dict[str, Any]]:
        """Load samples from the E2 train split."""

        samples: List[Dict[str, Any]] = []

        valid_exts = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
        }

        for img_path in self.train_img_dir.iterdir():

            if img_path.suffix.lower() not in valid_exts:
                continue

            label_path = (
                self.train_label_dir
                / f"{img_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            boxes = []

            try:
                with open(
                    label_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    for line in f:

                        parts = line.strip().split()

                        if len(parts) != 5:
                            continue

                        class_id, cx, cy, w, h = map(
                            float,
                            parts,
                        )

                        boxes.append(
                            {
                                "class_id": int(class_id),
                                "bbox": [
                                    cx,
                                    cy,
                                    w,
                                    h,
                                ],
                            }
                        )

            except Exception:
                continue

            samples.append(
                {
                    "image_path": str(img_path),
                    "boxes": boxes,
                }
            )

        return samples

    def get_val_data(self) -> List[Dict[str, Any]]:
        """Return the unchanged original validation split."""

        return self._evaluation_dataset.get_val_data()

    def get_test_data(self) -> List[Dict[str, Any]]:
        """Return the unchanged original test split."""

        return self._evaluation_dataset.get_test_data()

    def get_class_names(self) -> List[str]:
        """Return class names from the original dataset."""

        return self._evaluation_dataset.get_class_names()

    def get_num_classes(self) -> int:
        """Return number of classes."""

        return self._evaluation_dataset.get_num_classes()

    def get_metadata(self) -> Dict[str, Any]:
        """Return E7 dataset source metadata."""

        return {
            "num_classes": self.get_num_classes(),
            "class_names": self.get_class_names(),
            "train_samples": len(self.get_train_data()),
            "val_samples": len(self.get_val_data()),
            "test_samples": len(self.get_test_data()),
            "train_source": str(
                self.train_dataset_root
            ),
            "evaluation_source": str(
                self.evaluation_dataset_root
            ),
        }