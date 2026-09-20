
import numpy as np

class ColorTemperatureEngine:

    def apply(
        self,
        rgba: np.ndarray,
        temperature: int,
    ) -> np.ndarray:

        rgb = rgba[:, :, :3].astype(np.float32)

        if temperature > 0:
            rgb[:, :, 0] += temperature
            rgb[:, :, 2] -= temperature

        else:
            value = abs(temperature)
            rgb[:, :, 2] += value
            rgb[:, :, 0] -= value

        rgb = np.clip(rgb, 0, 255)

        out = rgba.copy()
        out[:, :, :3] = rgb.astype(np.uint8)

        return out