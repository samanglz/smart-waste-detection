from pathlib import Path


class E6HardCaseAnalysisConfig:

    # ============================================================
    # EVALUATION DATASET
    # ============================================================

    DATASET_ROOT = Path(
        r"C:\Users\ASUS\Desktop\smart-waste-detection\data\processed"
    )

    # ============================================================
    # E6 TRAINING DATASET FOR HARD-CASE SIMILARITY
    # ============================================================

    E6_TRAIN_ROOT = Path(
        r"C:\Users\ASUS\Desktop\smart-waste-detection\data\E6_hard_mining"
    )

    TRAIN_IMAGES_DIR = E6_TRAIN_ROOT / "images"
    TRAIN_LABELS_DIR = E6_TRAIN_ROOT / "labels"
    # ============================================================
    # MODEL
    # ============================================================

    MODEL_PATH = Path(
        r"runs\detect\runs\E6_hard_mining_targeted_aug\weights\best.pt"
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
        "outputs/hard_case_analysis/E6"
    )

    ERROR_ANALYSIS_PATH = (
        OUTPUT_DIR / "error_analysis.json"
    )