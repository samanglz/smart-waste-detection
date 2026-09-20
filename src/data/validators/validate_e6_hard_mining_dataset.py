# src/data/validators/validate_e6_hard_mining_dataset.py

import json
from collections import Counter
from pathlib import Path


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

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "E6_hard_mining"
    / "dataset_build_metadata.json"
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

E6_ROOT = (
    PROJECT_ROOT
    / "data"
    / "E6_hard_mining"
)

E6_IMAGES = E6_ROOT / "images"
E6_LABELS = E6_ROOT / "labels"


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

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_image_files(directory: Path):
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_label_files(directory: Path):
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    )


def normalize_path(path: str) -> str:
    """
    Normalize Windows paths for comparison.
    """

    return str(
        Path(path)
    ).replace(
        "\\",
        "/",
    ).lower()


def find_source_image(filename: str) -> Path:
    """
    Find source E2 image by filename.
    """

    candidate = (
        SOURCE_TRAIN_IMAGES
        / filename
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Source E2 image not found:\n{candidate}"
    )


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8"
    )


def fail(message: str):
    raise RuntimeError(
        f"\n❌ VALIDATION FAILED:\n{message}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("E6 HARD-MINING DATASET VALIDATION")
    print("=" * 80)

    # ========================================================
    # 1. Validate required files/directories
    # ========================================================

    print("\n[1] Validating paths...")

    required_paths = [
        MANIFEST_PATH,
        METADATA_PATH,
        SOURCE_TRAIN_IMAGES,
        SOURCE_TRAIN_LABELS,
        E6_IMAGES,
        E6_LABELS,
    ]

    for path in required_paths:

        if not path.exists():
            fail(
                f"Required path does not exist:\n{path}"
            )

    print("✓ All required paths exist.")

    # ========================================================
    # 2. Load manifest and metadata
    # ========================================================

    print("\n[2] Loading manifest and metadata...")

    manifest = load_json(
        MANIFEST_PATH
    )

    metadata = load_json(
        METADATA_PATH
    )

    print(
        f"Manifest experiment : "
        f"{manifest.get('experiment')}"
    )

    print(
        f"Metadata experiment : "
        f"{metadata.get('experiment')}"
    )

    if manifest.get("experiment") != "E6":
        fail(
            "Manifest experiment is not E6."
        )

    if metadata.get("experiment") != "E6":
        fail(
            "Dataset metadata experiment is not E6."
        )

    print("✓ Experiment identifiers are E6.")

    # ========================================================
    # 3. Validate manifest ↔ metadata
    # ========================================================

    print(
        "\n[3] Validating manifest ↔ metadata consistency..."
    )

    manifest_objects = manifest[
        "objects"
    ]

    manifest_unique = int(
        manifest[
            "total_unique_training_objects"
        ]
    )

    metadata_unique = int(
        metadata[
            "unique_hard_mining_objects"
        ]
    )

    if manifest_unique != metadata_unique:
        fail(
            "Unique hard object count mismatch:\n"
            f"Manifest : {manifest_unique}\n"
            f"Metadata : {metadata_unique}"
        )

    manifest_selection_total = sum(
        int(obj["selection_count"])
        for obj in manifest_objects
    )

    metadata_selection_total = int(
        metadata[
            "total_hard_selections"
        ]
    )

    if (
        manifest_selection_total
        != metadata_selection_total
    ):
        fail(
            "Total hard selection mismatch:\n"
            f"Manifest : {manifest_selection_total}\n"
            f"Metadata : {metadata_selection_total}"
        )

    metadata_copies = int(
        metadata[
            "hard_mining_copies"
        ]
    )

    if (
        manifest_selection_total
        != metadata_copies
    ):
        fail(
            "Hard-mining copy count mismatch:\n"
            f"Manifest selections : "
            f"{manifest_selection_total}\n"
            f"Metadata copies     : "
            f"{metadata_copies}"
        )

    print("✓ Manifest and metadata agree.")

    # ========================================================
    # 4. Count physical files
    # ========================================================

    print("\n[4] Counting dataset files...")

    source_images = get_image_files(
        SOURCE_TRAIN_IMAGES
    )

    source_labels = get_label_files(
        SOURCE_TRAIN_LABELS
    )

    e6_images = get_image_files(
        E6_IMAGES
    )

    e6_labels = get_label_files(
        E6_LABELS
    )

    print(
        f"E2 source images : "
        f"{len(source_images)}"
    )

    print(
        f"E2 source labels : "
        f"{len(source_labels)}"
    )

    print(
        f"E6 images        : "
        f"{len(e6_images)}"
    )

    print(
        f"E6 labels        : "
        f"{len(e6_labels)}"
    )

    if len(source_images) != 2367:
        fail(
            "Expected 2367 E2 training images, "
            f"found {len(source_images)}."
        )

    if len(source_labels) != 2367:
        fail(
            "Expected 2367 E2 training labels, "
            f"found {len(source_labels)}."
        )

    if len(e6_images) != 2662:
        fail(
            "Expected 2662 E6 training images, "
            f"found {len(e6_images)}."
        )

    if len(e6_labels) != 2662:
        fail(
            "Expected 2662 E6 training labels, "
            f"found {len(e6_labels)}."
        )

    print("✓ File counts are correct.")

    # ========================================================
    # 5. Image ↔ label pairing
    # ========================================================

    print(
        "\n[5] Checking image ↔ label pairing..."
    )

    e6_image_stems = {
        p.stem
        for p in e6_images
    }

    e6_label_stems = {
        p.stem
        for p in e6_labels
    }

    missing_labels = (
        e6_image_stems
        - e6_label_stems
    )

    orphan_labels = (
        e6_label_stems
        - e6_image_stems
    )

    if missing_labels:
        fail(
            "Images without labels:\n"
            + "\n".join(
                sorted(missing_labels)[:20]
            )
        )

    if orphan_labels:
        fail(
            "Labels without images:\n"
            + "\n".join(
                sorted(orphan_labels)[:20]
            )
        )

    print("✓ Every E6 image has exactly one label.")

    # ========================================================
    # 6. Validate original E2 samples are preserved
    # ========================================================

    print(
        "\n[6] Checking preservation of original E2 samples..."
    )

    e6_original_images = {
        p.name
        for p in e6_images
        if "__e4hm_" not in p.stem
    }

    source_image_names = {
        p.name
        for p in source_images
    }

    missing_originals = (
        source_image_names
        - e6_original_images
    )

    unexpected_originals = (
        e6_original_images
        - source_image_names
    )

    if missing_originals:
        fail(
            "Original E2 images missing from E6:\n"
            + "\n".join(
                sorted(missing_originals)[:20]
            )
        )

    if unexpected_originals:
        fail(
            "Unexpected non-hard-mining images in E6:\n"
            + "\n".join(
                sorted(unexpected_originals)[:20]
            )
        )

    if len(e6_original_images) != 2367:
        fail(
            "E6 does not contain exactly "
            "2367 original E2 images."
        )

    print(
        "✓ All 2367 original E2 images are preserved."
    )

    # ========================================================
    # 7. Validate original labels are identical
    # ========================================================

    print(
        "\n[7] Comparing original E2 labels..."
    )

    checked_labels = 0

    for source_label in source_labels:

        e6_label = (
            E6_LABELS
            / source_label.name
        )

        if not e6_label.exists():
            fail(
                f"Missing original E6 label:\n"
                f"{source_label.name}"
            )

        source_text = read_text(
            source_label
        )

        e6_text = read_text(
            e6_label
        )

        if source_text != e6_text:
            fail(
                "Original label was modified:\n"
                f"{source_label.name}"
            )

        checked_labels += 1

    print(
        f"✓ {checked_labels} original labels "
        "are byte-identical."
    )

    # ========================================================
    # 8. Validate hard-mining copies
    # ========================================================

    print(
        "\n[8] Validating hard-mining copies..."
    )

    expected_copy_counts = Counter()

    expected_class_counts = Counter()

    for obj in manifest_objects:

        image_path = Path(
            obj["image_path"]
        )

        filename = image_path.name

        selection_count = int(
            obj["selection_count"]
        )

        class_name = obj["class"]

        expected_copy_counts[
            filename
        ] += selection_count

        expected_class_counts[
            class_name
        ] += selection_count

    actual_copy_counts = Counter()

    hard_copy_files = []

    for image in e6_images:

        if "__e4hm_" not in image.stem:
            continue

        hard_copy_files.append(
            image
        )

        parts = image.stem.split(
            "__e4hm_"
        )

        if len(parts) != 2:
            fail(
                f"Invalid hard-mining filename:\n"
                f"{image.name}"
            )

        source_stem = parts[0]

        source_candidates = [
            p
            for p in source_images
            if p.stem == source_stem
        ]

        if len(source_candidates) != 1:
            fail(
                "Could not uniquely resolve "
                f"hard-mining source for:\n"
                f"{image.name}"
            )

        source_image = (
            source_candidates[0]
        )

        actual_copy_counts[
            source_image.name
        ] += 1

    if len(hard_copy_files) != metadata_copies:
        fail(
            "Physical hard-mining copy count mismatch:\n"
            f"Expected : {metadata_copies}\n"
            f"Actual   : {len(hard_copy_files)}"
        )

    if actual_copy_counts != expected_copy_counts:
        print("\nExpected copies:")
        for name, count in (
            expected_copy_counts.items()
        ):
            print(
                f"  {name}: {count}"
            )

        print("\nActual copies:")
        for name, count in (
            actual_copy_counts.items()
        ):
            print(
                f"  {name}: {count}"
            )

        fail(
            "Hard-mining copy distribution does not "
            "match manifest."
        )

    print(
        f"✓ All {metadata_copies} hard-mining copies "
        "match manifest selection counts."
    )

    # ========================================================
    # 9. Validate hard-mining labels
    # ========================================================

    print(
        "\n[9] Checking hard-mining labels..."
    )

    for hard_image in hard_copy_files:

        source_stem = hard_image.stem.split(
            "__e4hm_"
        )[0]

        source_image = next(
            (
                p
                for p in source_images
                if p.stem == source_stem
            ),
            None,
        )

        if source_image is None:
            fail(
                f"Source image not found for "
                f"{hard_image.name}"
            )

        source_label = (
            SOURCE_TRAIN_LABELS
            / f"{source_image.stem}.txt"
        )

        hard_label = (
            E6_LABELS
            / f"{hard_image.stem}.txt"
        )

        if not hard_label.exists():
            fail(
                f"Hard-mining label missing:\n"
                f"{hard_label.name}"
            )

        if (
            read_text(source_label)
            != read_text(hard_label)
        ):
            fail(
                "Hard-mining label differs from "
                f"source label:\n"
                f"{hard_image.name}"
            )

    print(
        "✓ All hard-mining labels match their "
        "original source labels."
    )

    # ========================================================
    # 10. Validate selection frequency
    # ========================================================

    print(
        "\n[10] Validating selection frequency..."
    )

    actual_selection_frequency = Counter(
        int(obj["selection_count"])
        for obj in manifest_objects
    )

    expected_frequency = Counter({
        1: 209,
        2: 37,
        3: 4,
    })

    if (
        actual_selection_frequency
        != expected_frequency
    ):
        fail(
            "Manifest selection frequency differs "
            "from expected E6 values."
        )

    print(
        "✓ Selection frequency:"
    )

    for count in sorted(
        actual_selection_frequency,
        reverse=True,
    ):
        print(
            f"  selected {count}x : "
            f"{actual_selection_frequency[count]} objects"
        )

    # ========================================================
    # 11. Validate hard-mined class distribution
    # ========================================================

    print(
        "\n[11] Validating hard-mined class distribution..."
    )

    metadata_class_counts = Counter(
        metadata[
            "hard_mined_class_distribution"
        ]
    )

    if (
        metadata_class_counts
        != expected_class_counts
    ):
        fail(
            "Hard-mined class distribution mismatch."
        )

    print(
        "✓ Hard-mined class distribution:"
    )

    for cls, count in sorted(
        expected_class_counts.items()
    ):
        print(
            f"  {cls:10} {count}"
        )

    # ========================================================
    # 12. Final dataset size equation
    # ========================================================

    print(
        "\n[12] Validating final dataset size..."
    )

    original_count = len(
        source_images
    )

    hard_count = len(
        hard_copy_files
    )

    expected_final = (
        original_count
        + hard_count
    )

    actual_final = len(
        e6_images
    )

    print(
        f"Original E2 : {original_count}"
    )

    print(
        f"Hard copies : {hard_count}"
    )

    print(
        f"Expected    : {expected_final}"
    )

    print(
        f"Actual      : {actual_final}"
    )

    if actual_final != expected_final:
        fail(
            "Final dataset size equation failed."
        )

    print(
        "✓ 2367 + 295 = 2662"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 80)
    print("E6 HARD-MINING DATASET VALIDATION PASSED")
    print("=" * 80)

    print("\nVerified:")
    print("  ✓ E6 manifest")
    print("  ✓ E6 metadata")
    print("  ✓ 2367 original E2 images")
    print("  ✓ 2367 original E2 labels")
    print("  ✓ Original labels unchanged")
    print("  ✓ 295 hard-mining copies")
    print("  ✓ Hard-mining selection frequencies")
    print("  ✓ Hard-mining class distribution")
    print("  ✓ Image/label pairing")
    print("  ✓ Hard-mining labels")
    print("  ✓ Final dataset size = 2662")

    print("\nDataset is ready for E6 training.")
    print("=" * 80)


if __name__ == "__main__":
    main()