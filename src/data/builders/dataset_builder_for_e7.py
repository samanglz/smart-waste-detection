"""
Generic dataset builder for synthetic object composition.

The builder creates a training dataset by:
    1. Reading the original training dataset.
    2. Reading objects from an Object Bank.
    3. Selecting a generation count through a strategy.
    4. Compositing objects onto training backgrounds.
    5. Calculating the visible bounding box from the final object mask.
    6. Writing synthetic images and YOLO labels.

Validation and test splits are never modified.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.yolo_dataset import YOLODataset
from src.e7.context_engine import ContextEngine
from src.e7.generation_strategy import GenerationCountStrategy
from src.e7.object_sampling import BalancedObjectSampler


class DatasetBuilder:
    """
    Builds a synthetic training dataset from an Object Bank.

    The builder is intentionally independent from a specific experiment.
    Experiment-specific behavior is injected through strategies and the
    ContextEngine.
    """

    def __init__(
        self,
        dataset: YOLODataset,
        object_bank_dir: Path,
        output_dir: Path,
        generation_strategy: GenerationCountStrategy,
        context_engine: ContextEngine,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.object_bank_dir = Path(object_bank_dir)
        self.output_dir = Path(output_dir)

        self.generation_strategy = generation_strategy
        self.context_engine = context_engine

        self.object_sampler = BalancedObjectSampler(
    seed=seed,
)

        self.object_bank_manifest = (
            self.object_bank_dir / "manifest.json"
        )

        self.train_images_dir = (
            self.output_dir / "train" / "images"
        )

        self.train_labels_dir = (
            self.output_dir / "train" / "labels"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        max_images: int | None = None,
        copy_original: bool = True,
    ) -> dict[str, Any]:
        """
        Build the synthetic training dataset.

        Args:
            max_images:
                Optional limit for smoke testing.

            copy_original:
                If True, original training samples are copied to the
                output dataset before synthetic samples are generated.

        Returns:
            Build summary.
        """

        self._validate_inputs()
        self._prepare_output_dirs()

        manifest = self._load_manifest()
        objects = manifest["objects"]

        train_data = self.dataset.get_train_data()

        if max_images is not None:
            if max_images <= 0:
                raise ValueError(
                    "max_images must be greater than zero."
                )

            train_data = train_data[:max_images]

        summary = {
            "source_images": len(train_data),
            "original_images": 0,
            "synthetic_images": 0,
            "objects_processed": 0,
            "failures": 0,
        }

        object_pool = self._build_object_pool(objects)

        if not object_pool:
            raise RuntimeError(
                "Object Bank contains no valid objects."
            )

        for index, sample in enumerate(train_data, start=1):
            try:
                self._process_sample(
                    sample=sample,
                    image_index=index,
                    object_pool=object_pool,
                    summary=summary,
                    copy_original=copy_original,
                )

            except Exception as exc:
                summary["failures"] += 1

                print(
                    f"[WARNING] Failed sample "
                    f"{index}: {exc}"
                )

        self._save_build_summary(summary)

        return summary

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        if not self.object_bank_dir.exists():
            raise FileNotFoundError(
                f"Object Bank not found: "
                f"{self.object_bank_dir}"
            )

        if not self.object_bank_manifest.exists():
            raise FileNotFoundError(
                f"Object Bank manifest not found: "
                f"{self.object_bank_manifest}"
            )

        if not self.dataset.train_img_dir.exists():
            raise FileNotFoundError(
                f"Training image directory not found: "
                f"{self.dataset.train_img_dir}"
            )

        if not self.dataset.train_label_dir.exists():
            raise FileNotFoundError(
                f"Training label directory not found: "
                f"{self.dataset.train_label_dir}"
            )

    def _prepare_output_dirs(self) -> None:
        if self.train_images_dir.exists():
            shutil.rmtree(self.train_images_dir)

        if self.train_labels_dir.exists():
            shutil.rmtree(self.train_labels_dir)

        self.train_images_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.train_labels_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Object Bank
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        with self.object_bank_manifest.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise ValueError(
                "Object Bank manifest must be a dictionary."
            )

        if manifest.get("experiment") != "E7":
            raise ValueError(
                "Object Bank manifest does not belong to E7."
            )

        if manifest.get("stage") != "object_bank":
            raise ValueError(
                "Invalid Object Bank manifest stage."
            )

        if manifest.get("split") != "train":
            raise ValueError(
                "E7 Object Bank must be extracted from train split."
            )

        if "objects" not in manifest:
            raise ValueError(
                "Object Bank manifest does not contain objects."
            )

        return manifest

    def _build_object_pool(
        self,
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pool = []

        for entry in objects:
            object_path = self.object_bank_dir / entry["object_path"]
            metadata_path = (
                self.object_bank_dir
                / entry["metadata_path"]
            )

            if not object_path.exists():
                continue

            if not metadata_path.exists():
                continue

            pool.append(
                {
                    "object_id": entry["object_id"],
                    "class_id": int(entry["class_id"]),
                    "class_name": entry["class_name"],
                    "object_path": object_path,
                    "metadata_path": metadata_path,
                }
            )

        return pool

    def _load_object_metadata(
        self,
        object_entry: dict[str, Any],
    ) -> dict[str, Any]:
        with object_entry["metadata_path"].open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # ------------------------------------------------------------------
    # Sample processing
    # ------------------------------------------------------------------

    def _process_sample(
        self,
        sample: dict[str, Any],
        image_index: int,
        object_pool: list[dict[str, Any]],
        summary: dict[str, Any],
        copy_original: bool,
    ) -> None:
        image_path = Path(sample["image_path"])

        background = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if background is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # --------------------------------------------------------------
        # Original sample
        # --------------------------------------------------------------

        if copy_original:
            original_image_name = image_path.name

            original_output_image = (
                self.train_images_dir
                / original_image_name
            )

            original_output_label = (
                self.train_labels_dir
                / f"{image_path.stem}.txt"
            )

            shutil.copy2(
                image_path,
                original_output_image,
            )

            self._write_original_labels(
                sample=sample,
                output_path=original_output_label,
            )

            summary["original_images"] += 1

        # --------------------------------------------------------------
        # Select an object
        # --------------------------------------------------------------

        object_entry = self._select_object(
            object_pool=object_pool,
        )

        metadata = self._load_object_metadata(
            object_entry
        )

        object_area = int(
            metadata["mask"]["area_pixels"]
        )

        image_height, image_width = background.shape[:2]

        decision = self.generation_strategy.decide(
            object_area_pixels=object_area,
            image_width=int(
                metadata["image"]["width"]
            ),
            image_height=int(
                metadata["image"]["height"]
            ),
        )

        # --------------------------------------------------------------
        # Generate synthetic samples
        # --------------------------------------------------------------

        for generation_index in range(
            decision.count
        ):
            object_rgba = self._load_object(
                object_entry["object_path"]
            )

            result = self.context_engine.generate(
                background=background.copy(),
                object_rgba=object_rgba,
                enable_reflection=(
                    object_entry["class_name"]
                    in {"glass", "plastic"}
                ),
                enable_occlusion=True,
            )
            
            
            bbox = self._mask_to_yolo_bbox(
                mask=result.object_mask,
                placement=result.placement,
                image_width=image_width,
                image_height=image_height,
            )

            if bbox is None:
                raise ValueError(
                    "Generated object has no visible pixels."
                )

            output_stem = (
                f"{image_path.stem}"
                f"_e7_{generation_index + 1}"
                f"_{object_entry['object_id']}"
            )

            output_image = (
                self.train_images_dir
                / f"{output_stem}{image_path.suffix}"
            )

            output_label = (
                self.train_labels_dir
                / f"{output_stem}.txt"
            )

            cv2.imwrite(
                str(output_image),
                result.image,
            )

            self._write_yolo_label(
                output_path=output_label,
                class_id=object_entry["class_id"],
                bbox=bbox,
            )

            summary["synthetic_images"] += 1
            summary["objects_processed"] += 1

    # ------------------------------------------------------------------
    # Object selection
    # ------------------------------------------------------------------

    def _select_object(
        self,
        object_pool: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.object_sampler.sample(
            object_pool
        )


    def _load_object(
        self,
        object_path: Path,
    ) -> np.ndarray:
        object_rgba = cv2.imread(
            str(object_path),
            cv2.IMREAD_UNCHANGED,
        )

        if object_rgba is None:
            raise ValueError(
                f"Could not read Object Bank object: "
                f"{object_path}"
            )

        if object_rgba.ndim != 3:
            raise ValueError(
                f"Object must have 4 channels: "
                f"{object_path}"
            )

        if object_rgba.shape[2] != 4:
            raise ValueError(
                f"Object must be RGBA: "
                f"{object_path}"
            )

        # OpenCV loads PNG as BGRA.
        # Context Engine expects RGBA.
        return cv2.cvtColor(
            object_rgba,
            cv2.COLOR_BGRA2RGBA,
        )

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_yolo_bbox(
        mask: np.ndarray,
        placement: Any,
        image_width: int,
        image_height: int,
    ) -> list[float] | None:
        """
        Convert the visible object mask into a YOLO bbox.

        The mask coordinates are local to the placement.
        They are first converted into full-image coordinates.
        """

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        local_x1 = int(xs.min())
        local_y1 = int(ys.min())
        local_x2 = int(xs.max()) + 1
        local_y2 = int(ys.max()) + 1

        x1 = placement.x + local_x1
        y1 = placement.y + local_y1
        x2 = placement.x + local_x2
        y2 = placement.y + local_y2

        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width <= 0 or bbox_height <= 0:
            return None

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return [
            cx / image_width,
            cy / image_height,
            bbox_width / image_width,
            bbox_height / image_height,
        ]

    # ------------------------------------------------------------------
    # Label writing
    # ------------------------------------------------------------------

    @staticmethod
    def _write_yolo_label(
        output_path: Path,
        class_id: int,
        bbox: list[float],
    ) -> None:
        values = [
            str(class_id),
            f"{bbox[0]:.6f}",
            f"{bbox[1]:.6f}",
            f"{bbox[2]:.6f}",
            f"{bbox[3]:.6f}",
        ]

        output_path.write_text(
            " ".join(values) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_original_labels(
        sample: dict[str, Any],
        output_path: Path,
    ) -> None:
        lines = []

        for box in sample["boxes"]:
            class_id = int(box["class_id"])
            cx, cy, width, height = box["bbox"]

            lines.append(
                f"{class_id} "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

        output_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _save_build_summary(
        self,
        summary: dict[str, Any],
    ) -> None:
        summary_path = (
            self.output_dir / "build_summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )