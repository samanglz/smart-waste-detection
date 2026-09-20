from pathlib import Path

import torch

from src.data.yolo_dataset import YOLODataset
from src.e7.object_bank.extractor import ObjectExtractor
from src.e7.object_bank.mask_generator import MaskGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "data" / "E7_object_bank"

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
    print("E7 OBJECT BANK - FULL EXTRACTION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"SAM2 checkpoint: {SAM2_CHECKPOINT}")
    print(f"SAM2 config: {SAM2_CONFIG}")

    # ------------------------------------------------------------------
    # Validate required paths
    # ------------------------------------------------------------------

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {DATASET_ROOT}"
        )

    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}"
        )

    if not SAM2_CONFIG.exists():
        raise FileNotFoundError(
            f"SAM2 config not found: {SAM2_CONFIG}"
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    print("\nLoading YOLO dataset...")

    dataset = YOLODataset(DATASET_ROOT)

    train_data = dataset.get_train_data()

    print(f"Training samples: {len(train_data)}")

    # ------------------------------------------------------------------
    # SAM 2.1
    # ------------------------------------------------------------------

    print("\nLoading SAM 2.1 model...")

    mask_generator = MaskGenerator(
        checkpoint_path=SAM2_CHECKPOINT,
        config_path=SAM2_CONFIG,
        device=device,
    )

    # ------------------------------------------------------------------
    # Extractor
    # ------------------------------------------------------------------

    print("\nInitializing ObjectExtractor...")

    extractor = ObjectExtractor(
        dataset=dataset,
        output_dir=OUTPUT_DIR,
        mask_generator=mask_generator,
    )

    # ------------------------------------------------------------------
    # Full extraction
    # ------------------------------------------------------------------

    print("\nStarting FULL Object Bank extraction...")
    print("Validation/Test data will NOT be modified.")
    print("-" * 70)

    extractor.extract()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("E7 OBJECT BANK - FULL EXTRACTION COMPLETED")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()