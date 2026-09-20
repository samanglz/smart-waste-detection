"""
E2 - Targeted Augmentation Dataset Builder.

This builder reproduces the exact E2 dataset construction that was
used in the original Colab experiment.

E2 rules:

- Original training images are copied unchanged.
- Images containing glass (class 1) or plastic (class 4) receive
  one additional augmented copy.
- Augmented samples use TargetedAugmentation.
- Validation and test datasets are NOT modified.
- Original labels are copied unchanged.
- Augmented labels are generated from the transformed bounding boxes.
- The resulting dataset contains the original training samples plus
  one augmented sample for every training image containing a target
  class.

Target classes:
    glass   -> 1
    plastic -> 4
"""

from pathlib import Path
from typing import Optional

import shutil
import yaml
import os

from src.logging.logger import get_logger


logger = get_logger(__name__)


class TargetedAugmentationDatasetBuilder:
    """
    Build the exact E2 targeted-augmentation dataset.

    Example
    -------
    builder = TargetedAugmentationDatasetBuilder(
        source_root=Path("data"),
        output_root=Path("data/e2_targeted"),
        augmenter=TargetedAugmentation(),
    )

    dataset = builder.build()
    """

    TARGET_CLASS_IDS = {1, 4}

    CLASS_NAMES = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    ]

    NUM_CLASSES = 5

    def __init__(
        self,
        source_root: Path,
        output_root: Path,
        augmenter,
    ):
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.augmenter = augmenter

        self.source_train_images = (
            self.source_root / "train" / "images"
        )

        self.source_train_labels = (
            self.source_root / "train" / "labels"
        )

        self.source_val_images = (
            self.source_root / "val" / "images"
        )

        self.source_test_images = (
            self.source_root / "test" / "images"
        )

        self.output_train_images = (
            self.output_root / "train" / "images"
        )

        self.output_train_labels = (
            self.output_root / "train" / "labels"
        )

        self.data_yaml_path = (
            self.output_root / "data.yaml"
        )

        logger.info(
            "TargetedAugmentationDatasetBuilder initialized"
        )

        logger.info(
            "  Source root: %s",
            self.source_root,
        )

        logger.info(
            "  Output root: %s",
            self.output_root,
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_inputs(self) -> None:
        """
        Validate all required source directories/files before
        modifying the output dataset.
        """

        required_paths = [
            self.source_train_images,
            self.source_train_labels,
            self.source_val_images,
            self.source_test_images,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required E2 source path was not found:\n{path}"
                )

        if self.augmenter is None:
            raise ValueError(
                "E2 requires a TargetedAugmentation instance."
            )

        if not hasattr(
            self.augmenter,
            "contains_target_class",
        ):
            raise TypeError(
                "The augmenter must provide "
                "contains_target_class()."
            )

        if not hasattr(
            self.augmenter,
            "apply",
        ):
            raise TypeError(
                "The augmenter must provide apply()."
            )

        logger.info(
            "E2 dataset input validation passed."
        )

    # ============================================================
    # OUTPUT
    # ============================================================

    def _reset_output_directory(self) -> None:
        """
        Reproduce the original Colab behavior:

            if E2_ROOT.exists():
                shutil.rmtree(E2_ROOT)

        This guarantees that the E2 dataset is rebuilt from scratch.
        """

        if self.output_root.exists():
            logger.warning(
                "Removing existing E2 dataset: %s",
                self.output_root,
            )

            shutil.rmtree(self.output_root)

        self.output_train_images.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_train_labels.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # TRAINING DATA
    # ============================================================

    def _copy_original_training_sample(
        self,
        image_path: Path,
        label_path: Path,
    ) -> None:
        """
        Copy one original training image and its label unchanged.
        """

        shutil.copy2(
            image_path,
            self.output_train_images / image_path.name,
        )

        shutil.copy2(
            label_path,
            self.output_train_labels / label_path.name,
        )

    def _create_augmented_sample(
        self,
        image_path: Path,
        label_path: Path,
    ) -> bool:
        """
        Create exactly one E2 augmented sample.

        Naming convention is identical to the original Colab code:

            image.jpg
            image_e2.jpg

            image.txt
            image_e2.txt
        """

        output_image = (
            self.output_train_images
            / f"{image_path.stem}_e2{image_path.suffix}"
        )

        output_label = (
            self.output_train_labels
            / f"{image_path.stem}_e2.txt"
        )

        return self.augmenter.apply(
            image_path,
            label_path,
            output_image,
            output_label,
        )

    def _build_training_dataset(self) -> dict:
        """
        Build E2 training data.

        This reproduces the exact Colab loop:

            for image_path in sorted(
                (TRAIN_SRC / "images").glob("*.jpg")
            ):

                copy original

                if contains_target_class:
                    create augmented sample
        """

        original_count = 0
        target_count = 0
        augmented_count = 0

        image_paths = sorted(
            self.source_train_images.glob("*.jpg")
        )

        logger.info(
            "Found %d original training images.",
            len(image_paths),
        )

        for image_path in image_paths:

            label_path = (
                self.source_train_labels
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    "Missing label for training image:\n"
                    f"Image: {image_path}\n"
                    f"Label: {label_path}"
                )

            # ----------------------------------------------------
            # Original sample
            # ----------------------------------------------------

            self._copy_original_training_sample(
                image_path=image_path,
                label_path=label_path,
            )

            original_count += 1

            # ----------------------------------------------------
            # Target-class detection
            # ----------------------------------------------------

            if self.augmenter.contains_target_class(
                label_path
            ):
                target_count += 1

                # ------------------------------------------------
                # Targeted augmentation
                # ------------------------------------------------

                success = self._create_augmented_sample(
                    image_path=image_path,
                    label_path=label_path,
                )

                if success:
                    augmented_count += 1

        total_images = (
            original_count + augmented_count
        )

        logger.info("=" * 60)
        logger.info("E2 DATASET BUILD SUMMARY")
        logger.info("=" * 60)

        logger.info(
            "Original training images: %d",
            original_count,
        )

        logger.info(
            "Images containing target classes: %d",
            target_count,
        )

        logger.info(
            "Augmented images created: %d",
            augmented_count,
        )

        logger.info(
            "Final training images: %d",
            total_images,
        )

        return {
            "original": original_count,
            "target": target_count,
            "augmented": augmented_count,
            "total": total_images,
        }

    # ============================================================
    # DATA YAML
    # ============================================================

    def _create_data_yaml(self) -> Path:
        """
        Create the E2 data.yaml.

        Train points to the generated E2 training dataset.
        Validation and test point to the original processed datasets.
        """

        val_path = os.path.relpath(
            self.source_val_images,
            self.output_root,
        )

        test_path = os.path.relpath(
            self.source_test_images,
            self.output_root,
        )
        
        config = {
            "path": str(self.output_root).replace("\\", "/"),
            "train": "train/images",
            "val": val_path.replace("\\", "/"),
            "test": test_path.replace("\\", "/"),
            "nc": self.NUM_CLASSES,
            "names": self.CLASS_NAMES,
        }
        
        
        with self.data_yaml_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                config,
                file,
                sort_keys=False,
            )

        logger.info(
            "E2 data.yaml created: %s",
            self.data_yaml_path,
        )

        return self.data_yaml_path

    # ============================================================
    # DATASET VALIDATION
    # ============================================================

    def _validate_generated_dataset(self) -> dict:
        """
        Validate the generated E2 dataset.

        Checks:
            - image count
            - label count
            - missing labels
            - YOLO bounding-box validity
        """

        images = list(
            self.output_train_images.glob("*")
        )

        labels = list(
            self.output_train_labels.glob("*.txt")
        )

        invalid_boxes = 0
        missing_labels = 0

        for image_path in images:

            label_path = (
                self.output_train_labels
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                missing_labels += 1
                continue

            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():

                parts = line.split()

                if len(parts) != 5:
                    invalid_boxes += 1
                    continue

                try:
                    _, x, y, w, h = map(
                        float,
                        parts,
                    )
                except ValueError:
                    invalid_boxes += 1
                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    invalid_boxes += 1

        result = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": missing_labels,
            "invalid_boxes": invalid_boxes,
        }

        logger.info("=" * 60)
        logger.info("E2 DATASET VALIDATION")
        logger.info("=" * 60)

        logger.info(
            "Images: %d",
            result["images"],
        )

        logger.info(
            "Labels: %d",
            result["labels"],
        )

        logger.info(
            "Missing labels: %d",
            result["missing_labels"],
        )

        logger.info(
            "Invalid boxes: %d",
            result["invalid_boxes"],
        )

        if result["missing_labels"] != 0:
            raise ValueError(
                "E2 dataset validation failed: "
                "missing labels detected."
            )

        if result["invalid_boxes"] != 0:
            raise ValueError(
                "E2 dataset validation failed: "
                "invalid YOLO bounding boxes detected."
            )

        if result["images"] != result["labels"]:
            raise ValueError(
                "E2 dataset validation failed: "
                "image/label count mismatch."
            )

        logger.info(
            "E2 dataset validation passed."
        )

        return result

    # ============================================================
    # BUILD
    # ============================================================

    def build(self):
        """
        Build and validate the complete E2 dataset.

        Returns
        -------
        Path
            Path to E2 data.yaml.
        """

        logger.info("=" * 60)
        logger.info(
            "E2 - BUILDING TARGETED AUGMENTATION DATASET"
        )
        logger.info("=" * 60)

        self._validate_inputs()

        self._reset_output_directory()

        build_stats = (
            self._build_training_dataset()
        )

        self._create_data_yaml()

        validation_stats = (
            self._validate_generated_dataset()
        )

        logger.info("=" * 60)
        logger.info("E2 DATASET BUILD COMPLETED")
        logger.info("=" * 60)

        return {
            "root": self.output_root,
            "yaml_path": self.data_yaml_path,
            "build": build_stats,
            "validation": validation_stats,
        }