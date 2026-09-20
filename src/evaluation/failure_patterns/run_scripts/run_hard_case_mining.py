from pathlib import Path
import json
from typing import List, Dict, Any

from src.evaluation.failure_patterns.hard_case_similarity_analyzer import (
    HardCaseSimilarityAnalyzer,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "images"
)

TRAIN_LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "labels"
)

VAL_ERROR_ANALYSIS = (
    PROJECT_ROOT
    / "outputs"
    / "yolo11m"
    / "E2_targeted_augmentation"
    / "val_error_analysis.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "yolo11m"
    / "E2_targeted_augmentation"
    / "hard_case_similarity_mining.json"
)


# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

TARGET_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

TOP_K = 5


# ============================================================
# LOAD HARD CASES
# ============================================================

def load_hard_cases(path: Path) -> List[Dict[str, Any]]:
    """
    Load hard cases from the nested confusion structure
    produced by val_error_analysis.json.

    Expected structure:

        error_analysis
        └── confusion
            ├── gt_class
            │   ├── pred_class
            │   │   └── cases
            │   └── ...
            └── ...

    Validation cases are used only as query objects.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    error_analysis = data.get(
        "error_analysis",
        {},
    )

    confusion = error_analysis.get(
        "confusion",
        {},
    )

    hard_cases = []

    for source_class, predictions in confusion.items():

        if not isinstance(predictions, dict):
            continue

        for target_class, pattern in predictions.items():

            if not isinstance(pattern, dict):
                continue

            cases = pattern.get(
                "cases",
                [],
            )

            if not isinstance(cases, list):
                continue

            for case in cases:

                if not isinstance(case, dict):
                    continue

                # Make sure class information is explicit.
                case = dict(case)

                case.setdefault(
                    "gt_class",
                    source_class,
                )

                case.setdefault(
                    "pred_class",
                    target_class,
                )

                hard_cases.append(case)

    return hard_cases

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HARD CASE SIMILARITY MINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    print("\n[1] Checking paths...")

    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Train images not found:\n{TRAIN_IMAGES_DIR}"
        )

    if not TRAIN_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Train labels not found:\n{TRAIN_LABELS_DIR}"
        )

    if not VAL_ERROR_ANALYSIS.exists():
        raise FileNotFoundError(
            f"VAL error analysis not found:\n"
            f"{VAL_ERROR_ANALYSIS}"
        )

    print(
        f"Train images : {TRAIN_IMAGES_DIR}"
    )

    print(
        f"Train labels : {TRAIN_LABELS_DIR}"
    )

    print(
        f"VAL analysis : {VAL_ERROR_ANALYSIS}"
    )

    # --------------------------------------------------------
    # Load hard cases
    # --------------------------------------------------------

    print("\n[2] Loading hard cases...")

    hard_cases = load_hard_cases(
        VAL_ERROR_ANALYSIS
    )

    print(
        f"Hard cases found: {len(hard_cases)}"
    )

    if not hard_cases:
        raise RuntimeError(
            "No hard cases found in "
            "val_error_analysis.json"
        )

    # --------------------------------------------------------
    # Show distribution
    # --------------------------------------------------------

    pattern_counts = {}

    for case in hard_cases:

        pattern_id = case.get(
            "pattern_id",
            "unknown",
        )

        pattern_counts[
            pattern_id
        ] = (
            pattern_counts.get(
                pattern_id,
                0,
            )
            + 1
        )

    print("\nHard-case distribution:")

    for pattern_id, count in sorted(
        pattern_counts.items()
    ):
        print(
            f"  {pattern_id}: {count}"
        )

    # --------------------------------------------------------
    # Create analyzer
    # --------------------------------------------------------

    print(
        "\n[3] Creating similarity analyzer..."
    )

    analyzer = HardCaseSimilarityAnalyzer(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        class_names=CLASS_NAMES,
        device="cuda",
    )

    # --------------------------------------------------------
    # Build TRAIN embedding index
    # --------------------------------------------------------

    print(
        "\n[4] Building TRAIN embedding index..."
    )

    analyzer.build_train_index(
        target_classes=TARGET_CLASSES
    )

    for class_name in TARGET_CLASSES:

        count = len(
            analyzer.train_embeddings.get(
                class_name,
                [],
            )
        )

        print(
            f"  {class_name}: {count}"
        )

    # --------------------------------------------------------
    # Mine similar TRAIN objects
    # --------------------------------------------------------

    print(
        "\n[5] Mining similar TRAIN objects..."
    )

    mining_result = (
        analyzer.mine_similar_training_objects(
            cases=hard_cases,
            top_k=TOP_K,
        )
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    print(
        "\n[6] Saving mining report..."
    )

    analyzer.save_mining_report(
        mining_result,
        OUTPUT_PATH,
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MINING SUMMARY")
    print("=" * 70)

    print(
        f"Hard cases analyzed: "
        f"{mining_result['total_hard_cases']}"
    )

    print(
        f"Top-K per case: "
        f"{mining_result['top_k_per_case']}"
    )

    print(
        f"Unique TRAIN objects selected: "
        f"{mining_result['total_unique_training_objects']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()