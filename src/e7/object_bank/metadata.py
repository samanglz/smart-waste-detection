"""
Metadata representation for E7 Object Bank.

This module is responsible only for:
    - representing object metadata
    - validating metadata
    - serializing metadata to JSON

It does not perform:
    - segmentation
    - image processing
    - augmentation
    - dataset building
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SourceInfo:
    image: str
    label: str
    split: str


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int


@dataclass(frozen=True)
class NormalizedBBox:
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class PixelBBox:
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


@dataclass(frozen=True)
class CropInfo:
    width: int
    height: int


@dataclass(frozen=True)
class MaskInfo:
    area_pixels: int
    bbox_area_pixels: int
    occupancy_ratio: float
    touches_image_border: bool
    segmentation_score: float


@dataclass(frozen=True)
class ObjectInfo:
    width: int
    height: int
    area_pixels: int
    aspect_ratio: float


@dataclass(frozen=True)
class QualityInfo:
    valid: bool


@dataclass(frozen=True)
class ObjectMetadata:
    """
    Complete metadata for one extracted object.
    """

    object_id: str

    source: SourceInfo
    class_info: ClassInfo
    image: ImageInfo

    bbox_normalized: NormalizedBBox
    bbox_pixel: PixelBBox

    crop: CropInfo

    mask: MaskInfo
    object: ObjectInfo

    quality: QualityInfo

    @classmethod
    def create(
        cls,
        object_id: str,
        source_image: str,
        source_label: str,
        split: str,
        class_id: int,
        class_name: str,
        image_width: int,
        image_height: int,
        bbox_normalized: List[float],
        bbox_pixel: Tuple[int, int, int, int],
        mask: np.ndarray,
        segmentation_score: float,
        touches_image_border: bool,
    ) -> "ObjectMetadata":
        """
        Create metadata for one cropped object.

        Args:
            object_id:
                Unique identifier of the object.

            source_image:
                Path to the original source image.

            source_label:
                Path to the original YOLO label file.

            split:
                Dataset split. For E7 Object Bank this must be 'train'.

            class_id:
                YOLO class index.

            class_name:
                Human-readable class name.

            image_width:
                Width of the original image.

            image_height:
                Height of the original image.

            bbox_normalized:
                Original YOLO bbox:
                [cx, cy, width, height].

            bbox_pixel:
                Pixel bbox in original image:
                (x1, y1, x2, y2).

            mask:
                Cropped binary object mask.

            segmentation_score:
                SAM segmentation confidence score.

            touches_image_border:
                Whether the original full-image mask touches
                the original image boundary.
        """

        if len(bbox_normalized) != 4:
            raise ValueError(
                "bbox_normalized must contain exactly 4 values."
            )

        x1, y1, x2, y2 = bbox_pixel

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        bbox_area = bbox_width * bbox_height

        if mask.ndim != 2:
            raise ValueError(
                "Object mask must be a 2D array."
            )

        if mask.dtype != np.bool_:
            mask = mask.astype(bool)

        mask_area = int(mask.sum())

        occupancy_ratio = (
            mask_area / bbox_area
            if bbox_area > 0
            else 0.0
        )

        aspect_ratio = (
            bbox_width / bbox_height
            if bbox_height > 0
            else 0.0
        )

        metadata = cls(
            object_id=object_id,

            source=SourceInfo(
                image=str(source_image),
                label=str(source_label),
                split=split,
            ),

            class_info=ClassInfo(
                id=int(class_id),
                name=str(class_name),
            ),

            image=ImageInfo(
                width=int(image_width),
                height=int(image_height),
            ),

            bbox_normalized=NormalizedBBox(
                cx=float(bbox_normalized[0]),
                cy=float(bbox_normalized[1]),
                width=float(bbox_normalized[2]),
                height=float(bbox_normalized[3]),
            ),

            bbox_pixel=PixelBBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                width=int(bbox_width),
                height=int(bbox_height),
            ),

            crop=CropInfo(
                width=int(bbox_width),
                height=int(bbox_height),
            ),

            mask=MaskInfo(
                area_pixels=mask_area,
                bbox_area_pixels=int(bbox_area),
                occupancy_ratio=round(
                    occupancy_ratio,
                    4,
                ),
                touches_image_border=bool(
                    touches_image_border
                ),
                segmentation_score=round(
                    float(segmentation_score),
                    4,
                ),
            ),

            object=ObjectInfo(
                width=int(bbox_width),
                height=int(bbox_height),
                area_pixels=mask_area,
                aspect_ratio=round(
                    aspect_ratio,
                    4,
                ),
            ),

            quality=QualityInfo(
                valid=True,
            ),
        )

        metadata.validate()

        return metadata

    def validate(self) -> None:
        """
        Validate metadata integrity.
        """

        if not self.object_id:
            raise ValueError(
                "Object ID cannot be empty."
            )

        if self.source.split != "train":
            raise ValueError(
                "E7 Object Bank must be built from train split only."
            )

        if self.class_info.id < 0:
            raise ValueError(
                "Class ID cannot be negative."
            )

        if not self.class_info.name:
            raise ValueError(
                "Class name cannot be empty."
            )

        if (
            self.image.width <= 0
            or self.image.height <= 0
        ):
            raise ValueError(
                "Invalid original image dimensions."
            )

        if (
            self.bbox_pixel.x1 < 0
            or self.bbox_pixel.y1 < 0
            or self.bbox_pixel.x2 > self.image.width
            or self.bbox_pixel.y2 > self.image.height
        ):
            raise ValueError(
                "Pixel bbox lies outside the original image."
            )

        if (
            self.bbox_pixel.width <= 0
            or self.bbox_pixel.height <= 0
        ):
            raise ValueError(
                "Invalid pixel bbox dimensions."
            )

        if (
            self.crop.width != self.bbox_pixel.width
            or self.crop.height != self.bbox_pixel.height
        ):
            raise ValueError(
                "Crop dimensions must match bbox dimensions."
            )

        if not (
            0.0 <= self.mask.segmentation_score <= 1.0
        ):
            raise ValueError(
                "Segmentation score must be in [0, 1]."
            )

        if self.mask.area_pixels <= 0:
            raise ValueError(
                "Object mask is empty."
            )

        if self.mask.bbox_area_pixels <= 0:
            raise ValueError(
                "Bounding box area must be positive."
            )

        if not (
            0.0 <= self.mask.occupancy_ratio <= 1.0
        ):
            raise ValueError(
                "Occupancy ratio must be in [0, 1]."
            )

    def to_dict(self) -> dict:
        """
        Convert metadata into JSON-serializable dictionary.
        """

        data = asdict(self)

        data["class"] = data.pop("class_info")

        data["bbox"] = {
            "normalized": data.pop(
                "bbox_normalized"
            ),
            "pixel": data.pop(
                "bbox_pixel"
            ),
        }

        return data

    def save(self, save_path: Path) -> None:
        """
        Save metadata as JSON.
        """

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=4,
                ensure_ascii=False,
            )