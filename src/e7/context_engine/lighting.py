
import cv2
import numpy as np

class LightingEngine:

    def apply(
        self,
        rgba: np.ndarray,
        brightness_factor: float,
    ) -> np.ndarray:

        rgb = rgba[:, :, :3].astype(np.float32)

        rgb *= brightness_factor
        rgb = np.clip(rgb, 0, 255)

        result = rgba.copy()
        result[:, :, :3] = rgb.astype(np.uint8)

        return result

    def estimate_scene_brightness(
        self,
        background: np.ndarray,
    ) -> float:

        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        return float(gray.mean() / 127.5)