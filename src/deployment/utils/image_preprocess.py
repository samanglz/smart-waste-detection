from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a square deployment size.

    Args:
        image:
            BGR OpenCV image with shape (H, W, 3).

        size:
            Target square size.

        color:
            Padding color in BGR format.

    Returns:
        BGR image with shape (size, size, 3).
    """

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Input image must be a numpy.ndarray."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected image with 3 dimensions (H, W, C), "
            f"got shape: {image.shape}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel BGR image, "
            f"got shape: {image.shape}"
        )

    if size <= 0:
        raise ValueError(
            f"Image size must be positive, got: {size}"
        )

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid image dimensions: {image.shape}"
        )

    # Preserve aspect ratio.
    scale = min(
        size / height,
        size / width,
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Calculate required padding.
    pad_width = size - new_width
    pad_height = size - new_height

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top

    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return output


def preprocess_image(
    image: np.ndarray,
    size: int = 640,
) -> np.ndarray:
    """
    Convert a BGR OpenCV image into YOLO ONNX input format.

    Processing pipeline:
        BGR image
        -> Letterbox
        -> RGB
        -> HWC -> CHW
        -> float32
        -> Normalize to [0, 1]
        -> Add batch dimension

    Args:
        image:
            BGR OpenCV image with shape (H, W, 3).

        size:
            Target YOLO input size.

    Returns:
        Float32 tensor with shape:
            (1, 3, size, size)
    """

    image = letterbox(
        image,
        size=size,
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1),
    )

    # uint8 -> float32
    image = image.astype(
        np.float32,
    )

    # [0, 255] -> [0, 1]
    image /= 255.0

    # Add batch dimension.
    image = np.expand_dims(
        image,
        axis=0,
    )

    # Ensure C-contiguous memory layout.
    return np.ascontiguousarray(image)


def preprocess_image_path(
    image_path: str,
    size: int = 640,
) -> np.ndarray:
    """
    Load an image from disk and convert it into
    YOLO ONNX input format.

    Args:
        image_path:
            Path to the image.

        size:
            Target YOLO input size.

    Returns:
        Float32 tensor with shape:
            (1, 3, size, size)
    """

    image = cv2.imread(
        image_path,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return preprocess_image(
        image,
        size=size,
    )