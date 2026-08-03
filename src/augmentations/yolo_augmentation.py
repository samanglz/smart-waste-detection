"""
YOLO augmentation configuration.

Provides a clean interface for configuring YOLO augmentations
during training and inference.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class YOLOAugmentationConfig:
    """
    Configuration class for YOLO augmentations.

    All parameters are passed directly to Ultralytics YOLO's train() method.
    Default values are based on Ultralytics YOLO v11.
    """

    # ===== Geometric Augmentations =====
    hsv_h: float = 0.015      # HSV Hue augmentation (0-1)
    hsv_s: float = 0.7        # HSV Saturation augmentation (0-1)
    hsv_v: float = 0.4        # HSV Value augmentation (0-1)

    degrees: float = 0.0      # Rotation angle in degrees
    translate: float = 0.1    # Translation as fraction of image size
    scale: float = 0.5        # Scale as fraction of image size
    shear: float = 0.0        # Shear angle in degrees

    flipud: float = 0.0       # Vertical flip probability (0-1)
    fliplr: float = 0.5       # Horizontal flip probability (0-1)

    # ===== Mosaic and MixUp =====
    mosaic: float = 1.0       # Mosaic probability (0-1)
    mixup: float = 0.0        # MixUp probability (0-1)
    copy_paste: float = 0.0   # Copy-Paste probability (0-1)

    # ===== Other Augmentations =====
    erasing: float = 0.4      # Random erasing probability (0-1)
    crop_fraction: float = 1.0 # Crop fraction for mosaic (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for passing to YOLO train().

        Returns:
            Dictionary of augmentation parameters.
        """
        return {
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "erasing": self.erasing,
            "crop_fraction": self.crop_fraction,
        }

    def disable_all(self) -> "YOLOAugmentationConfig":
        """Disable all augmentations (useful for validation/testing)."""
        self.hsv_h = 0.0
        self.hsv_s = 0.0
        self.hsv_v = 0.0
        self.degrees = 0.0
        self.translate = 0.0
        self.scale = 0.0
        self.shear = 0.0
        self.flipud = 0.0
        self.fliplr = 0.0
        self.mosaic = 0.0
        self.mixup = 0.0
        self.copy_paste = 0.0
        self.erasing = 0.0
        self.crop_fraction = 1.0
        return self

    def enable_light_augmentations(self) -> "YOLOAugmentationConfig":
        """Enable only light augmentations (for fine-tuning)."""
        self.hsv_h = 0.01
        self.hsv_s = 0.5
        self.hsv_v = 0.3
        self.degrees = 0.0
        self.translate = 0.05
        self.scale = 0.3
        self.shear = 0.0
        self.flipud = 0.0
        self.fliplr = 0.3
        self.mosaic = 0.5
        self.mixup = 0.0
        self.copy_paste = 0.0
        return self

    def enable_heavy_augmentations(self) -> "YOLOAugmentationConfig":
        """Enable heavy augmentations (for small datasets with overfitting)."""
        self.hsv_h = 0.02
        self.hsv_s = 0.8
        self.hsv_v = 0.5
        self.degrees = 5.0
        self.translate = 0.2
        self.scale = 0.7
        self.shear = 5.0
        self.flipud = 0.0
        self.fliplr = 0.5
        self.mosaic = 1.0
        self.mixup = 0.2
        self.copy_paste = 0.3
        return self