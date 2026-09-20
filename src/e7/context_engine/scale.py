
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    width: int
    height: int

class RandomScaleStrategy:

    def __init__(
        self,
        min_relative_height: float = 0.10,
        max_relative_height: float = 0.35,
    ):
        self.min_h = min_relative_height
        self.max_h = max_relative_height

    def calculate(
        self,
        obj_w,
        obj_h,
        img_w,
        img_h,
        rng: np.random.Generator,
    ) -> Placement:

        target_h = int(img_h * rng.uniform(self.min_h, self.max_h))
        scale = target_h / obj_h

        target_w = int(obj_w * scale)

        x = int(rng.integers(0, img_w - target_w + 1))
        y = int(rng.integers(0, img_h - target_h + 1))

        return Placement(x, y, target_w, target_h)