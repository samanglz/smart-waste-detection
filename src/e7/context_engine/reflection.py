
import cv2
import numpy as np

class ReflectionEngine:

    def apply(
        self,
        rgba: np.ndarray,
        strength: float = 0.3,
    ) -> np.ndarray:

        h, w = rgba.shape[:2]

        gradient = np.tile(
            np.linspace(1.0, 0.0, w),
            (h, 1),
        )

        highlight = gradient[:, :, None] * 255

        rgb = rgba[:, :, :3].astype(np.float32)

        rgb += highlight * strength

        rgb = np.clip(rgb, 0, 255)

        out = rgba.copy()
        out[:, :, :3] = rgb.astype(np.uint8)

        return out