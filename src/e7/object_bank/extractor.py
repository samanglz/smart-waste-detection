"""
Object extraction pipeline for E7 Object Bank.

This module:
    - reads training samples from YOLODataset
    - converts YOLO bounding boxes to pixel coordinates
    - requests segmentation masks from MaskGenerator
    - crops objects using the segmentation mask
    - creates RGBA object crops
    - saves object, mask, and metadata
    - creates an Object Bank manifest

Only the training split is used.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import json

import cv2
import numpy as np

from src.data.yolo_dataset import YOLODataset
from src.e7.object_bank.mask_generator import MaskGenerator
from src.e7.object_bank.metadata import ObjectMetadata
from src.logging.logger import get_logger


logger = get_logger(__name__)


class ObjectExtractor:
    """
    Build the E7 Object Bank from the training split.
    """

    def __init__(
        self,
        dataset: YOLODataset,
        output_dir: Path,
        mask_generator: MaskGenerator,
    ):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.mask_generator = mask_generator

        self.class_names = dataset.get_class_names()

        self.object_counter = 0
        self.success_count = 0
        self.failure_count = 0

        self.manifest_records = []

    def extract(
        self,
        max_samples: int | None = None,
    ) -> None:
        """
        Extract objects from the training split.

        Args:
            max_samples:
                Maximum number of training images to process.
                None means process the complete training split.
        """

        train_samples = self.dataset.get_train_data()

        if max_samples is not None:
            train_samples = train_samples[:max_samples]

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Starting E7 Object Bank extraction."
        )

        logger.info(
            "Training samples to process: %d",
            len(train_samples),
        )

        for sample_index, sample in enumerate(
            train_samples,
            start=1,
        ):
            try:
                self._process_sample(sample)

            except Exception as exc:
                self.failure_count += 1

                logger.error(
                    "Failed to process sample %d: %s",
                    sample_index,
                    exc,
                )

            if (
                sample_index == 1
                or sample_index % 10 == 0
                or sample_index == len(train_samples)
            ):
                logger.info(
                    "Progress: %d/%d samples | "
                    "objects=%d | failures=%d",
                    sample_index,
                    len(train_samples),
                    self.success_count,
                    self.failure_count,
                )

        self._save_manifest()

        logger.info(
            "Object Bank extraction completed."
        )

        logger.info(
            "Samples processed: %d",
            len(train_samples),
        )

        logger.info(
            "Objects extracted: %d",
            self.success_count,
        )

        logger.info(
            "Failures: %d",
            self.failure_count,
        )

    def _process_sample(
        self,
        sample: Dict[str, Any],
    ) -> None:

        image_path = Path(
            sample["image_path"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image_height, image_width = (
            image.shape[:2]
        )

        label_path = (
            self.dataset.train_label_dir
            / f"{image_path.stem}.txt"
        )

        for box_data in sample["boxes"]:

            self._process_object(
                image=image,
                image_path=image_path,
                label_path=label_path,
                image_width=image_width,
                image_height=image_height,
                box_data=box_data,
            )

    def _process_object(
        self,
        image: np.ndarray,
        image_path: Path,
        label_path: Path,
        image_width: int,
        image_height: int,
        box_data: Dict[str, Any],
    ) -> None:

        class_id = int(
            box_data["class_id"]
        )

        if not (
            0 <= class_id < len(self.class_names)
        ):
            logger.warning(
                "Invalid class ID %d in %s",
                class_id,
                label_path,
            )
            self.failure_count += 1
            return

        class_name = self.class_names[
            class_id
        ]

        bbox_normalized = box_data["bbox"]

        bbox_pixel = self._yolo_to_pixel_bbox(
            bbox=bbox_normalized,
            image_width=image_width,
            image_height=image_height,
        )

        try:
            mask, score = (
                self.mask_generator.generate(
                    image=image,
                    bbox=bbox_pixel,
                )
            )

        except Exception as exc:
            self.failure_count += 1

            logger.error(
                "Segmentation failed for %s: %s",
                image_path,
                exc,
            )
            return

        if mask is None:
            self.failure_count += 1

            logger.warning(
                "Empty mask returned for %s",
                image_path,
            )
            return

        if mask.ndim != 2:
            self.failure_count += 1

            logger.warning(
                "Invalid mask dimensions for %s",
                image_path,
            )
            return

        if mask.shape != image.shape[:2]:
            self.failure_count += 1

            logger.warning(
                "Mask/image shape mismatch for %s: "
                "mask=%s image=%s",
                image_path,
                mask.shape,
                image.shape[:2],
            )
            return

        # ---------------------------------------------------------
        # Full-image mask quality information
        # ---------------------------------------------------------

        touches_image_border = (
            bool(mask[0].any())
            or bool(mask[-1].any())
            or bool(mask[:, 0].any())
            or bool(mask[:, -1].any())
        )

        # ---------------------------------------------------------
        # Crop using original bounding box
        # ---------------------------------------------------------

        x1, y1, x2, y2 = bbox_pixel

        cropped_image = image[
            y1:y2,
            x1:x2,
        ]

        cropped_mask = mask[
            y1:y2,
            x1:x2,
        ]

        if (
            cropped_image.size == 0
            or cropped_mask.size == 0
        ):
            self.failure_count += 1

            logger.warning(
                "Empty crop for %s",
                image_path,
            )
            return

        if (
            cropped_image.shape[:2]
            != cropped_mask.shape[:2]
        ):
            self.failure_count += 1

            logger.warning(
                "Image/mask crop size mismatch: %s",
                image_path,
            )
            return

        # ---------------------------------------------------------
        # Create RGBA object
        # ---------------------------------------------------------

        rgba_object = self._create_rgba(
            image=cropped_image,
            mask=cropped_mask,
        )

        # ---------------------------------------------------------
        # Create object ID
        # ---------------------------------------------------------

        self.object_counter += 1

        object_id = (
            f"{class_name}_"
            f"{self.object_counter:06d}"
        )

        object_dir = (
            self.output_dir
            / "objects"
            / class_name
            / object_id
        )

        object_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # Save object
        # ---------------------------------------------------------

        object_path = (
            object_dir / "object.png"
        )

        self._save_rgba(
            rgba=rgba_object,
            save_path=object_path,
        )

        # ---------------------------------------------------------
        # Save mask
        # ---------------------------------------------------------

        mask_path = (
            object_dir / "mask.png"
        )

        self._save_mask(
            mask=cropped_mask,
            save_path=mask_path,
        )

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------

        metadata = ObjectMetadata.create(
            object_id=object_id,
            source_image=str(image_path),
            source_label=str(label_path),
            split="train",
            class_id=class_id,
            class_name=class_name,
            image_width=image_width,
            image_height=image_height,
            bbox_normalized=bbox_normalized,
            bbox_pixel=bbox_pixel,
            mask=cropped_mask,
            segmentation_score=score,
            touches_image_border=touches_image_border,
        )

        metadata_path = (
            object_dir / "metadata.json"
        )

        metadata.save(
            metadata_path
        )

        # ---------------------------------------------------------
        # Manifest record
        # ---------------------------------------------------------

        self.manifest_records.append(
            {
                "object_id": object_id,
                "class_id": class_id,
                "class_name": class_name,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "object_path": str(object_path),
                "mask_path": str(mask_path),
                "metadata_path": str(metadata_path),
                "segmentation_score": float(score),
                "touches_image_border": bool(
                    touches_image_border
                ),
            }
        )

        self.success_count += 1

    def _save_manifest(self) -> None:
        """
        Save Object Bank manifest.
        """

        manifest = {
            "experiment": "E7",
            "stage": "object_bank",
            "split": "train",
            "num_objects": self.success_count,
            "num_failures": self.failure_count,
            "classes": self.class_names,
            "objects": self.manifest_records,
        }

        manifest_path = (
            self.output_dir
            / "manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Manifest saved: %s",
            manifest_path,
        )

    @staticmethod
    def _yolo_to_pixel_bbox(
        bbox,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:

        cx, cy, width, height = map(
            float,
            bbox,
        )

        x1 = int(
            (cx - width / 2)
            * image_width
        )

        y1 = int(
            (cy - height / 2)
            * image_height
        )

        x2 = int(
            (cx + width / 2)
            * image_width
        )

        y2 = int(
            (cy + height / 2)
            * image_height
        )

        x1 = max(
            0,
            min(x1, image_width - 1),
        )

        y1 = max(
            0,
            min(y1, image_height - 1),
        )

        x2 = max(
            x1 + 1,
            min(x2, image_width),
        )

        y2 = max(
            y1 + 1,
            min(y2, image_height),
        )

        return x1, y1, x2, y2

    @staticmethod
    def _create_rgba(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:

        alpha = (
            mask.astype(np.uint8)
            * 255
        )

        rgba = np.dstack(
            [
                image[:, :, 0],
                image[:, :, 1],
                image[:, :, 2],
                alpha,
            ]
        )

        return rgba

    @staticmethod
    def _save_rgba(
        rgba: np.ndarray,
        save_path: Path,
    ) -> None:

        bgra = cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2BGRA,
        )

        success = cv2.imwrite(
            str(save_path),
            bgra,
        )

        if not success:
            raise IOError(
                f"Failed to save RGBA object: "
                f"{save_path}"
            )

    @staticmethod
    def _save_mask(
        mask: np.ndarray,
        save_path: Path,
    ) -> None:

        binary_mask = (
            mask.astype(np.uint8)
            * 255
        )

        success = cv2.imwrite(
            str(save_path),
            binary_mask,
        )

        if not success:
            raise IOError(
                f"Failed to save mask: "
                f"{save_path}"
            )