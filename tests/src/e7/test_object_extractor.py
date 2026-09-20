from pathlib import Path

import torch

from src.data.yolo_dataset import YOLODataset
from src.e7.object_bank.extractor import ObjectExtractor
from src.e7.object_bank.mask_generator import MaskGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "E7_object_bank_test"
)

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


def main():

    print("=" * 70)
    print("E7 OBJECT BANK - SMOKE TEST")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = YOLODataset(
        DATASET_ROOT
    )

    # ---------------------------------------------------------
    # SAM2
    # ---------------------------------------------------------

    mask_generator = MaskGenerator(
        checkpoint_path=SAM2_CHECKPOINT,
        config_path=SAM2_CONFIG,
        device=device,
    )

    # ---------------------------------------------------------
    # Extractor
    # ---------------------------------------------------------

    extractor = ObjectExtractor(
        dataset=dataset,
        output_dir=OUTPUT_DIR,
        mask_generator=mask_generator,
    )

    # ---------------------------------------------------------
    # Process only first 5 images
    # ---------------------------------------------------------

    extractor.extract(
        max_samples=5
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETED")
    print("=" * 70)

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()