# src/data/build_e4_hard_mining_dataset.py

import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "yolo11m"
    / "E2_targeted_augmentation"
    / "E6_hard_mining_manifest.json"
)

SOURCE_TRAIN_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "e2_targeted_test"
    / "train"
    / "images"
)

SOURCE_TRAIN_LABELS = (
    PROJECT_ROOT
    / "data"
    / "e2_targeted_test"
    / "train"
    / "labels"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "E6_hard_mining"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def find_image(image_path: Path) -> Path:
    """
    Resolve the source image path from the manifest.

    The manifest contains absolute Windows paths.
    We first try that exact path, then fall back to the
    filename inside the current project's train/images directory.
    """

    image_path = Path(image_path)

    if image_path.exists():
        return image_path

    fallback = SOURCE_TRAIN_IMAGES / image_path.name

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Training image not found:\n"
        f"Manifest path: {image_path}\n"
        f"Fallback path: {fallback}"
    )


def find_label(image_path: Path) -> Path:
    """
    Find YOLO label corresponding to an image.
    """

    label_path = (
        SOURCE_TRAIN_LABELS
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label not found for image:\n"
            f"{image_path}\n"
            f"Expected:\n"
            f"{label_path}"
        )

    return label_path


def copy_training_sample(
    image_path: Path,
    label_path: Path,
    output_stem: str,
):
    """
    Copy an image + its original YOLO label.
    """

    image_ext = image_path.suffix.lower()

    output_image = (
        OUTPUT_IMAGES
        / f"{output_stem}{image_ext}"
    )

    output_label = (
        OUTPUT_LABELS
        / f"{output_stem}.txt"
    )

    shutil.copy2(
        image_path,
        output_image,
    )

    shutil.copy2(
        label_path,
        output_label,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("E6 HARD-MINING DATASET BUILDER")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    print("\n[1] Validating inputs...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    if not SOURCE_TRAIN_IMAGES.exists():
        raise FileNotFoundError(
            f"Source train images directory not found:\n"
            f"{SOURCE_TRAIN_IMAGES}"
        )

    if not SOURCE_TRAIN_LABELS.exists():
        raise FileNotFoundError(
            f"Source train labels directory not found:\n"
            f"{SOURCE_TRAIN_LABELS}"
        )

    print(f"Manifest       : {MANIFEST_PATH}")
    print(f"Source images  : {SOURCE_TRAIN_IMAGES}")
    print(f"Source labels  : {SOURCE_TRAIN_LABELS}")
    print(f"Output         : {OUTPUT_ROOT}")

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print("\n[2] Loading E6 hard-mining manifest...")

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    objects = manifest["objects"]

    print(
        f"Hard cases                 : "
        f"{manifest['hard_cases']}"
    )

    print(
        f"Top-K per case             : "
        f"{manifest['top_k_per_case']}"
    )

    print(
        f"Unique hard objects        : "
        f"{manifest['total_unique_training_objects']}"
    )

    # --------------------------------------------------------
    # Calculate expected copies
    # --------------------------------------------------------

    total_selections = sum(
        int(obj["selection_count"])
        for obj in objects
    )

    print(
        f"Total hard selections      : "
        f"{total_selections}"
    )

    print(
        f"Original E2 train images   : "
        f"{len(list(SOURCE_TRAIN_IMAGES.iterdir()))}"
    )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    print("\n[3] Preparing output dataset...")

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"\nOutput dataset already exists:\n"
            f"{OUTPUT_ROOT}\n\n"
            f"Delete it manually if you intentionally "
            f"want to rebuild E4."
        )

    OUTPUT_IMAGES.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_LABELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy complete E2 training set
    # --------------------------------------------------------

    print("\n[4] Copying original E2 training dataset...")

    source_images = sorted(
        p
        for p in SOURCE_TRAIN_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    copied_original = 0

    for image_path in source_images:

        label_path = find_label(
            image_path
        )

        copy_training_sample(
            image_path=image_path,
            label_path=label_path,
            output_stem=image_path.stem,
        )

        copied_original += 1

    print(
        f"Original samples copied: "
        f"{copied_original}"
    )

    # --------------------------------------------------------
    # Hard-mining oversampling
    # --------------------------------------------------------

    print(
        "\n[5] Adding hard-mined training samples..."
    )

    class_counts = Counter()
    selection_frequency = Counter()
    image_copy_counts = defaultdict(int)

    hard_copies = 0

    for obj in objects:

        image_path = find_image(
            Path(obj["image_path"])
        )

        label_path = find_label(
            image_path
        )

        selection_count = int(
            obj["selection_count"]
        )

        class_name = obj["class"]

        selection_frequency[
            selection_count
        ] += 1

        for repeat_idx in range(
            selection_count
        ):

            hard_copies += 1

            image_copy_counts[
                image_path.name
            ] += 1

            output_stem = (
                f"{image_path.stem}"
                f"__e4hm"
                f"_rank{int(obj['hard_mining_rank']):04d}"
                f"_rep{repeat_idx + 1:02d}"
            )

            copy_training_sample(
                image_path=image_path,
                label_path=label_path,
                output_stem=output_stem,
            )

            class_counts[
                class_name
            ] += 1

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    final_images = sorted(
        p
        for p in OUTPUT_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    final_labels = sorted(
        p
        for p in OUTPUT_LABELS.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    )

    print("\n" + "=" * 80)
    print("E4 DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Original E2 images       : "
        f"{copied_original}"
    )

    print(
        f"Hard-mining copies       : "
        f"{hard_copies}"
    )

    print(
        f"Final training images    : "
        f"{len(final_images)}"
    )

    print(
        f"Final training labels    : "
        f"{len(final_labels)}"
    )

    print(
        f"Expected final images    : "
        f"{copied_original + hard_copies}"
    )

    print("\nHard-mined class distribution:")

    for cls, count in sorted(
        class_counts.items()
    ):
        print(
            f"  {cls:10} {count}"
        )

    print("\nSelection frequency:")

    for count, num_objects in sorted(
        selection_frequency.items(),
        reverse=True,
    ):
        print(
            f"  selected {count}x : "
            f"{num_objects} objects"
        )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    print("\n[6] Running integrity checks...")

    if len(final_images) != len(final_labels):
        raise RuntimeError(
            "Image/label count mismatch!"
        )

    if len(final_images) != (
        copied_original + hard_copies
    ):
        raise RuntimeError(
            "Final dataset size does not match "
            "expected size!"
        )

    print("✓ Image/label counts match.")
    print("✓ Dataset size matches expected count.")

    # --------------------------------------------------------
    # Save build metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": "E6",
        "method": "embedding_based_hard_mining",
        "source_manifest": str(
            MANIFEST_PATH
        ),
        "source_dataset": "E2_targeted_augmentation",
        "original_training_images": copied_original,
        "unique_hard_mining_objects": len(objects),
        "total_hard_selections": total_selections,
        "hard_mining_copies": hard_copies,
        "final_training_images": len(final_images),
        "final_training_labels": len(final_labels),
        "selection_frequency": {
            str(k): v
            for k, v in sorted(
                selection_frequency.items(),
                reverse=True,
            )
        },
        "hard_mined_class_distribution": dict(
            class_counts
        ),
        "oversampling_strategy": (
            "Each selected training object contributes "
            "selection_count copies of its original "
            "training image and original YOLO label."
        ),
    }

    metadata_path = (
        OUTPUT_ROOT
        / "dataset_build_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nMetadata saved to:\n"
        f"{metadata_path}"
    )

    print("\n" + "=" * 80)
    print("E4 HARD-MINING DATASET READY")
    print("=" * 80)

    print(
        f"\nDataset:\n"
        f"{OUTPUT_ROOT}"
    )

    print(
        "\nNext step:"
        "\n  validate dataset integrity"
        "\n  then train E6 from this dataset."
    )


if __name__ == "__main__":
    main()