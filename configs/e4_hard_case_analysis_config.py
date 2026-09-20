from pathlib import Path


class E4HardCaseAnalysisConfig:

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
        r"runs\detect\runs\E3_hard_example_sampling\hard_sampling-6\weights\best.pt"
    )

    # ============================================================
    # ANALYSIS
    # ============================================================

    TARGET_CLASSES = [
        "glass",
        "plastic",
        "metal",
        "paper",
        "cardboard",
    ]

    TOP_K = 5

    DEVICE = "cuda"

    # ============================================================
    # OUTPUT
    # ============================================================

    OUTPUT_DIR = Path(
        "outputs/hard_case_analysis/E4"
    )

    ERROR_ANALYSIS_PATH = (
        OUTPUT_DIR / "error_analysis.json"
    )