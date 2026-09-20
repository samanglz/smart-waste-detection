from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class ImagePreprocessor:
    """
    YOLO-compatible image preprocessing.

    Output:
        tensor : (1,3,H,W) float32
        ratio  : resize ratio
        pad    : (pad_x, pad_y)
        original_image : BGR image
    """

    def __init__(self, input_size: int = 640):
        self.input_size = input_size

    def preprocess(
        self,
        image_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        letterboxed, ratio, pad = self._letterbox(image)

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, image, ratio, pad

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:

        h, w = image.shape[:2]
        size = self.input_size

        ratio = min(size / h, size / w)

        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full((size, size, 3), 114, dtype=np.uint8)

        pad_x = (size - new_w) / 2
        pad_y = (size - new_h) / 2

        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))

        canvas[
            top:top + new_h,
            left:left + new_w,
        ] = resized

        return canvas, ratio, (pad_x, pad_y)
    
    
    
    def preprocess_frame(self, frame: np.ndarray):
        """
        Preprocess an OpenCV BGR frame.
        """
        letterboxed, ratio, pad = self._letterbox(frame)

        rgb = cv2.cvtColor(
            letterboxed,
            cv2.COLOR_BGR2RGB,
        )

        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, ratio, pad