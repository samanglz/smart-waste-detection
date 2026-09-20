
import cv2
import numpy as np

class OcclusionEngine:

    def apply(
        self,
        rgba: np.ndarray,
        ratio: float,
    ) -> np.ndarray:

        out = rgba.copy()

        h, w = out.shape[:2]

        occ_h = int(h * ratio)

        out[h-occ_h:h, :, 3] = 0

        return out