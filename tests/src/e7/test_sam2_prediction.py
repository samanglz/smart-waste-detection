from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.yolo_dataset import YOLODataset
from src.e7.object_bank.mask_generator import MaskGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "data" / "processed"

SAM2_CHECKPOINT = (
    PROJECT_ROOT
    / "sam2"
    / "checkpoints"
    / "sam2.1_hiera_small.pt"
)


SAM2_CONFIG = (
    Path.home()
    / "AppData"
    / "Local"
    / "Programs"
    / "Python"
    / "Python311"
    / "Lib"
    / "site-packages"
    / "sam2"
    / "configs"
    / "sam2.1"
    / "sam2.1_hiera_s.yaml"
)

def yolo_to_pixel_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert YOLO normalized bbox:
        [cx, cy, w, h]

    to pixel bbox:
        [x1, y1, x2, y2]

    x2 and y2 are exclusive.
    """

    cx, cy, w, h = bbox

    x1 = int((cx - w / 2) * image_width)
    y1 = int((cy - h / 2) * image_height)

    x2 = int((cx + w / 2) * image_width)
    y2 = int((cy + h / 2) * image_height)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))

    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2


def main():
    print("=" * 70)
    print("E7 - SAM 2.1 REAL PREDICTION TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------

    print("\nLoading YOLO dataset...")

    dataset = YOLODataset(DATASET_ROOT)

    train_data = dataset.get_train_data()

    if not train_data:
        raise RuntimeError("Training dataset is empty.")

    print(f"Train samples: {len(train_data)}")

    # ------------------------------------------------------------------
    # 3. First sample
    # ------------------------------------------------------------------

    sample = train_data[0]

    image_path = Path(sample["image_path"])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not sample["boxes"]:
        raise RuntimeError(
            f"No bounding boxes found in: {image_path}"
        )

    box_info = sample["boxes"][0]

    class_id = box_info["class_id"]
    normalized_bbox = box_info["bbox"]

    class_name = dataset.get_class_names()[class_id]

    print("\nSample:")
    print(f"  Image: {image_path.name}")
    print(f"  Class ID: {class_id}")
    print(f"  Class: {class_name}")
    print(f"  YOLO bbox: {normalized_bbox}")

    # ------------------------------------------------------------------
    # 4. Load image
    # ------------------------------------------------------------------

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_height, image_width = image.shape[:2]

    print(f"  Image size: {image_width} x {image_height}")

    # ------------------------------------------------------------------
    # 5. Convert bbox
    # ------------------------------------------------------------------

    bbox_pixel = yolo_to_pixel_bbox(
        bbox=normalized_bbox,
        image_width=image_width,
        image_height=image_height,
    )

    print(f"  Pixel bbox: {bbox_pixel}")

    # ------------------------------------------------------------------
    # 6. Create SAM2 MaskGenerator
    # ------------------------------------------------------------------

    print("\nLoading MaskGenerator...")

    mask_generator = MaskGenerator(
        checkpoint_path=SAM2_CHECKPOINT,
        config_path=SAM2_CONFIG,
        device=device,
    )

    print("MaskGenerator: OK")

    # ------------------------------------------------------------------
    # 7. Generate mask
    # ------------------------------------------------------------------

    print("\nRunning SAM 2.1 prediction...")

    mask, score = mask_generator.generate(
        image=image,
        bbox=bbox_pixel,
    )

    # ------------------------------------------------------------------
    # 8. Validate result
    # ------------------------------------------------------------------

    print("\nPrediction result:")

    print(f"  Mask shape: {mask.shape}")
    print(f"  Image shape: {image.shape[:2]}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Mask score: {score:.4f}")

    mask_pixels = int(np.count_nonzero(mask))

    print(f"  Mask pixels: {mask_pixels}")

    if mask.shape != image.shape[:2]:
        raise RuntimeError(
            "Mask shape does not match image shape."
        )

    if mask_pixels == 0:
        raise RuntimeError(
            "SAM generated an empty mask."
        )

    print("\n" + "=" * 70)
    print("SAM 2.1 REAL PREDICTION TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()