from pathlib import Path


class E2HardCaseAnalysisConfig:

    # ============================================================
    # DATASET
    # ============================================================

    DATASET_ROOT = Path(
        r"C:\Users\ASUS\Desktop\smart-waste-detection\data\processed"
    )
    TRAIN_IMAGES_DIR = (
        DATASET_ROOT / "train" / "images"
    )

    TRAIN_LABELS_DIR = (
        DATASET_ROOT / "train" / "labels"
    )

    # ============================================================
    # MODEL
    # ============================================================

    MODEL_PATH = Path(
        r"runs\detect\runs\E2_targeted_augmentation\targeted_aug\weights\best.pt"
    )

    # ============================================================
    # ANALYSIS
    # ============================================================

    TARGET_CLASSES = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    ]

    TOP_K = 5

    DEVICE = "cuda"

    # ============================================================
    # OUTPUT
    # ============================================================

    OUTPUT_DIR = Path(
        "outputs/hard_case_analysis/E2"
    )

    ERROR_ANALYSIS_PATH = (
        OUTPUT_DIR / "error_analysis.json"
    )