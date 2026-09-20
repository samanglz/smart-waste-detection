"""
E7 Dataset Builder Pre-flight Test.

This test validates the E7 pipeline contracts before running
the full dataset generation on the complete training split.

Important:
    This test does NOT generate a new dataset.
    It only validates the existing smoke-test output and
    Object Bank structure.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OBJECT_BANK_DIR = (
    PROJECT_ROOT
    / "data"
    / "E7_object_bank"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "tests"
    / "trash"
    / "E7_dataset_test"
)

MANIFEST_PATH = (
    OBJECT_BANK_DIR / "manifest.json"
)

SUMMARY_PATH = (
    DATASET_DIR / "build_summary.json"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

NUM_CLASSES = len(CLASS_NAMES)

BBOX_TOLERANCE = 1e-5


def load_json(path: Path) -> dict:
    assert path.exists(), (
        f"Missing JSON file: {path}"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert isinstance(data, dict), (
        f"Expected JSON object: {path}"
    )

    return data


def validate_object_bank() -> dict:
    print("[1] Validating Object Bank...")

    assert OBJECT_BANK_DIR.exists(), (
        f"Object Bank not found: {OBJECT_BANK_DIR}"
    )

    manifest = load_json(MANIFEST_PATH)

    assert manifest["experiment"] == "E7"
    assert manifest["stage"] == "object_bank"
    assert manifest["split"] == "train"

    objects = manifest.get("objects")

    assert isinstance(objects, list)
    assert len(objects) > 0

    class_counts = Counter()

    for entry in objects:

        assert "object_id" in entry
        assert "class_id" in entry
        assert "class_name" in entry
        assert "object_path" in entry
        assert "metadata_path" in entry

        class_id = int(entry["class_id"])
        class_name = entry["class_name"]

        assert 0 <= class_id < NUM_CLASSES

        assert class_name in CLASS_NAMES

        object_path = (
            OBJECT_BANK_DIR
            / entry["object_path"]
        )

        metadata_path = (
            OBJECT_BANK_DIR
            / entry["metadata_path"]
        )

        assert object_path.exists(), (
            f"Missing object: {object_path}"
        )

        assert metadata_path.exists(), (
            f"Missing metadata: {metadata_path}"
        )

        class_counts[class_name] += 1

    missing_classes = (
        set(CLASS_NAMES)
        - set(class_counts.keys())
    )

    assert not missing_classes, (
        f"Missing Object Bank classes: "
        f"{missing_classes}"
    )

    print(
        f"✓ Objects: {len(objects)}"
    )

    for class_name in CLASS_NAMES:
        print(
            f"  - {class_name}: "
            f"{class_counts[class_name]}"
        )

    return manifest


def validate_dataset_structure() -> tuple[Path, Path]:
    print("[2] Validating dataset structure...")

    assert DATASET_DIR.exists(), (
        f"Dataset directory not found: {DATASET_DIR}"
    )

    train_dir = DATASET_DIR / "train"

    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"

    assert images_dir.exists()
    assert labels_dir.exists()

    assert not (
        DATASET_DIR / "val"
    ).exists(), (
        "Validation split must not be generated."
    )

    assert not (
        DATASET_DIR / "test"
    ).exists(), (
        "Test split must not be generated."
    )

    print("✓ Train structure valid")
    print("✓ Val untouched")
    print("✓ Test untouched")

    return images_dir, labels_dir


def validate_filename_collisions(
    images_dir: Path,
    labels_dir: Path,
) -> None:
    print("[3] Checking filename collisions...")

    image_files = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
    ]

    label_files = list(
        labels_dir.glob("*.txt")
    )

    image_stems = [
        path.stem
        for path in image_files
    ]

    label_stems = [
        path.stem
        for path in label_files
    ]

    assert len(image_stems) == len(
        set(image_stems)
    ), "Duplicate image filenames detected."

    assert len(label_stems) == len(
        set(label_stems)
    ), "Duplicate label filenames detected."

    assert set(image_stems) == set(
        label_stems
    ), (
        "Image/label filename sets do not match."
    )

    print(
        f"✓ Unique image stems: "
        f"{len(image_stems)}"
    )


def validate_labels(
    labels_dir: Path,
) -> Counter:
    print("[4] Validating label distribution...")

    label_files = sorted(
        labels_dir.glob("*.txt")
    )

    assert label_files

    class_counts = Counter()

    total_annotations = 0

    for label_path in label_files:

        content = label_path.read_text(
            encoding="utf-8"
        ).strip()

        assert content, (
            f"Empty label file: {label_path}"
        )

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            values = line.split()

            assert len(values) == 5, (
                f"Invalid annotation at "
                f"{label_path}:{line_number}"
            )

            class_id = int(values[0])

            cx = float(values[1])
            cy = float(values[2])
            width = float(values[3])
            height = float(values[4])

            assert 0 <= class_id < NUM_CLASSES

            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0

            assert 0.0 < width <= 1.0
            assert 0.0 < height <= 1.0

            left = cx - width / 2.0
            top = cy - height / 2.0
            right = cx + width / 2.0
            bottom = cy + height / 2.0

            assert left >= -BBOX_TOLERANCE
            assert top >= -BBOX_TOLERANCE
            assert right <= 1.0 + BBOX_TOLERANCE
            assert bottom <= 1.0 + BBOX_TOLERANCE

            class_counts[
                CLASS_NAMES[class_id]
            ] += 1

            total_annotations += 1

    assert total_annotations > 0

    print(
        f"✓ Total annotations: "
        f"{total_annotations}"
    )

    for class_name in CLASS_NAMES:
        print(
            f"  - {class_name}: "
            f"{class_counts[class_name]}"
        )

    return class_counts


def validate_summary(
    summary: dict,
    image_count: int,
    label_count: int,
    total_annotations: int,
) -> None:
    print("[5] Validating build summary...")

    assert summary["source_images"] == 5

    assert summary["original_images"] == 5

    assert summary["synthetic_images"] > 0

    assert summary["objects_processed"] > 0

    assert summary["failures"] == 0

    expected_files = (
        summary["original_images"]
        + summary["synthetic_images"]
    )

    assert image_count == expected_files

    assert label_count == expected_files

    assert total_annotations > 0

    print(
        f"✓ Original images: "
        f"{summary['original_images']}"
    )

    print(
        f"✓ Synthetic images: "
        f"{summary['synthetic_images']}"
    )

    print(
        f"✓ Objects processed: "
        f"{summary['objects_processed']}"
    )

    print(
        f"✓ Failures: "
        f"{summary['failures']}"
    )


def validate_generation_contract(
    summary: dict,
) -> None:
    print("[6] Validating generation contract...")

    synthetic_images = int(
        summary["synthetic_images"]
    )

    objects_processed = int(
        summary["objects_processed"]
    )

    assert synthetic_images == objects_processed, (
        "Each generated synthetic image must "
        "correspond to one processed object."
    )

    print(
        "✓ Synthetic images == objects processed"
    )

    print(
        "✓ Generation strategy contract valid"
    )


def validate_readiness(
    manifest: dict,
    summary: dict,
) -> None:
    print("[7] Final E7 readiness check...")

    assert manifest["split"] == "train"

    assert summary["failures"] == 0

    assert len(
        manifest["objects"]
    ) > 0

    print("✓ Object Bank ready")
    print("✓ Dataset Builder output valid")
    print("✓ No validation/test contamination")
    print("✓ E7 Full Build is ready")


def main():
    print("=" * 70)
    print("E7 DATASET BUILDER PRE-FLIGHT")
    print("=" * 70)

    manifest = validate_object_bank()

    images_dir, labels_dir = (
        validate_dataset_structure()
    )

    validate_filename_collisions(
        images_dir,
        labels_dir,
    )

    class_counts = validate_labels(
        labels_dir
    )

    summary = load_json(
        SUMMARY_PATH
    )

    image_count = len(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file()
        ]
    )

    label_count = len(
        list(labels_dir.glob("*.txt"))
    )

    total_annotations = sum(
        class_counts.values()
    )

    validate_summary(
        summary=summary,
        image_count=image_count,
        label_count=label_count,
        total_annotations=total_annotations,
    )

    validate_generation_contract(
        summary
    )

    validate_readiness(
        manifest=manifest,
        summary=summary,
    )

    print()
    print("=" * 70)
    print("E7 DATASET BUILDER PRE-FLIGHT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()