"""
SAM 2.1 mask generator for E7 Object Bank.
"""

from pathlib import Path

import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class MaskGenerator:

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint_path}"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"SAM2 config not found: {self.config_path}"
            )

        print("Loading SAM 2.1 model...")

        model = build_sam2(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )

        self.predictor = SAM2ImagePredictor(model)

        print("SAM 2.1 MaskGenerator: OK")

    def generate(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, float]:

        x1, y1, x2, y2 = bbox

        input_box = np.array(
            [[x1, y1, x2, y2]],
            dtype=np.float32,
        )

        self.predictor.set_image(image)

        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        best_index = int(np.argmax(scores))

        mask = masks[best_index].astype(np.uint8)

        score = float(scores[best_index])

        return mask, score