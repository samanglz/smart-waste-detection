"""
RGBA object compositor for E7 Context Engine.
"""



import cv2
import numpy as np

from src.e7.context_engine.scale import Placement


class Compositor:
    """
    Composites an RGBA object onto a BGR background.
    """

    def composite(
        self,
        background: np.ndarray,
        object_rgba: np.ndarray,
        placement: Placement,
    ) -> np.ndarray:

        if background.ndim != 3 or background.shape[2] != 3:
            raise ValueError(
                "Background must be a BGR image with shape (H, W, 3)."
            )

        if object_rgba.ndim != 3 or object_rgba.shape[2] != 4:
            raise ValueError(
                "Object must be an RGBA image with shape (H, W, 4)."
            )

        object_resized = cv2.resize(
            object_rgba,
            (placement.width, placement.height),
            interpolation=cv2.INTER_AREA,
        )

        x1 = placement.x
        y1 = placement.y
        x2 = x1 + placement.width
        y2 = y1 + placement.height

        background_height, background_width = background.shape[:2]

        if x1 < 0 or y1 < 0:
            raise ValueError("Placement cannot have negative coordinates.")

        if x2 > background_width or y2 > background_height:
            raise ValueError(
                "Object placement exceeds background boundaries."
            )

        roi = background[y1:y2, x1:x2]

        object_rgb = object_resized[:, :, :3]
        alpha = object_resized[:, :, 3].astype(np.float32) / 255.0

        alpha = alpha[:, :, np.newaxis]

        composited = (
            object_rgb.astype(np.float32) * alpha
            + roi.astype(np.float32) * (1.0 - alpha)
        )

        background[y1:y2, x1:x2] = np.clip(
            composited,
            0,
            255,
        ).astype(np.uint8)

        return background