"""
Validation test for E7 Dataset Builder output.
"""

from pathlib import Path
import json

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "E7_dataset"
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


def load_summary() -> dict:
    summary_path = DATASET_DIR / "build_summary.json"

    assert summary_path.exists(), (
        f"Missing build summary: {summary_path}"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    assert isinstance(summary, dict)

    return summary


def validate_directories() -> tuple[Path, Path]:
    train_dir = DATASET_DIR / "train"

    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"

    assert train_dir.exists(), (
        f"Missing train directory: {train_dir}"
    )

    assert images_dir.exists(), (
        f"Missing images directory: {images_dir}"
    )

    assert labels_dir.exists(), (
        f"Missing labels directory: {labels_dir}"
    )

    # E7 Builder must not create validation/test data.
    assert not (DATASET_DIR / "val").exists(), (
        "Dataset Builder unexpectedly created a val directory."
    )

    assert not (DATASET_DIR / "test").exists(), (
        "Dataset Builder unexpectedly created a test directory."
    )

    return images_dir, labels_dir


def validate_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> tuple[list[Path], list[Path]]:

    image_files = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file()
        ]
    )

    label_files = sorted(
        labels_dir.glob("*.txt")
    )

    assert image_files, "No images found."
    assert label_files, "No labels found."

    image_stems = {
        path.stem
        for path in image_files
    }

    label_stems = {
        path.stem
        for path in label_files
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    assert not missing_labels, (
        f"Images without labels: {missing_labels}"
    )

    assert not missing_images, (
        f"Labels without images: {missing_images}"
    )

    return image_files, label_files


def validate_images(
    image_files: list[Path],
) -> None:

    for image_path in image_files:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        assert image is not None, (
            f"Could not read image: {image_path}"
        )

        assert image.ndim == 3, (
            f"Invalid image dimensions: {image_path}"
        )

        height, width = image.shape[:2]

        assert height > 0, (
            f"Invalid image height: {image_path}"
        )

        assert width > 0, (
            f"Invalid image width: {image_path}"
        )


def validate_labels(
    label_files: list[Path],
) -> tuple[int, int]:

    total_annotations = 0
    empty_labels = 0

    for label_path in label_files:

        content = label_path.read_text(
            encoding="utf-8"
        ).strip()

        # -----------------------------------------------------
        # Empty label = valid negative/background-only sample
        # -----------------------------------------------------

        if not content:
            empty_labels += 1
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            values = line.split()

            assert len(values) == 5, (
                f"Invalid annotation at "
                f"{label_path}:{line_number}. "
                f"Expected 5 values, got {len(values)}."
            )

            try:
                class_id = int(values[0])

                cx = float(values[1])
                cy = float(values[2])
                width = float(values[3])
                height = float(values[4])

            except ValueError as exc:
                raise AssertionError(
                    f"Non-numeric annotation at "
                    f"{label_path}:{line_number}: "
                    f"{line}"
                ) from exc

            # -------------------------------------------------
            # Class ID
            # -------------------------------------------------

            assert 0 <= class_id < NUM_CLASSES, (
                f"Invalid class ID {class_id} at "
                f"{label_path}:{line_number}. "
                f"Expected range: 0-{NUM_CLASSES - 1}."
            )

            # -------------------------------------------------
            # YOLO normalized values
            # -------------------------------------------------

            assert 0.0 <= cx <= 1.0, (
                f"Invalid cx={cx} at "
                f"{label_path}:{line_number}"
            )

            assert 0.0 <= cy <= 1.0, (
                f"Invalid cy={cy} at "
                f"{label_path}:{line_number}"
            )

            assert 0.0 < width <= 1.0, (
                f"Invalid width={width} at "
                f"{label_path}:{line_number}"
            )

            assert 0.0 < height <= 1.0, (
                f"Invalid height={height} at "
                f"{label_path}:{line_number}"
            )

            # -------------------------------------------------
            # Bounding box boundaries
            # -------------------------------------------------

            left = cx - width / 2.0
            top = cy - height / 2.0
            right = cx + width / 2.0
            bottom = cy + height / 2.0

            if left < -BBOX_TOLERANCE:
                raise AssertionError(
                    f"BBox exceeds LEFT boundary at "
                    f"{label_path}:{line_number}. "
                    f"cx={cx}, width={width}, "
                    f"left={left}"
                )

            if top < -BBOX_TOLERANCE:
                raise AssertionError(
                    f"BBox exceeds TOP boundary at "
                    f"{label_path}:{line_number}. "
                    f"cy={cy}, height={height}, "
                    f"top={top}"
                )

            if right > 1.0 + BBOX_TOLERANCE:
                raise AssertionError(
                    f"BBox exceeds RIGHT boundary at "
                    f"{label_path}:{line_number}. "
                    f"cx={cx}, width={width}, "
                    f"right={right}"
                )

            if bottom > 1.0 + BBOX_TOLERANCE:
                raise AssertionError(
                    f"BBox exceeds BOTTOM boundary at "
                    f"{label_path}:{line_number}. "
                    f"cy={cy}, height={height}, "
                    f"bottom={bottom}"
                )

            total_annotations += 1

    return total_annotations, empty_labels


def validate_summary(
    summary: dict,
    image_files: list[Path],
    label_files: list[Path],
    total_annotations: int,
) -> None:

    # ---------------------------------------------------------
    # Summary values
    # ---------------------------------------------------------

    source_images = summary["source_images"]
    original_images = summary["original_images"]
    synthetic_images = summary["synthetic_images"]
    failures = summary["failures"]

    assert source_images > 0, (
        "Source image count must be greater than zero."
    )

    assert original_images >= 0, (
        "Original image count cannot be negative."
    )

    assert synthetic_images > 0, (
        "No synthetic images were generated."
    )

    assert failures >= 0, (
        "Failure count cannot be negative."
    )

    # ---------------------------------------------------------
    # Expected total output images
    # ---------------------------------------------------------

    expected_files = (
        original_images
        + synthetic_images
    )

    assert len(image_files) == expected_files, (
        f"Summary expects {expected_files} images, "
        f"but found {len(image_files)}."
    )

    assert len(label_files) == expected_files, (
        f"Summary expects {expected_files} labels, "
        f"but found {len(label_files)}."
    )

    # At least the synthetic dataset must contain annotations.
    assert total_annotations > 0, (
        "No annotations found in the E7 dataset."
    )


def main():

    print("=" * 70)
    print("E7 DATASET BUILDER VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load summary
    # ---------------------------------------------------------

    print("[1] Loading build summary...")

    summary = load_summary()

    print("✓ Build summary loaded")

    # ---------------------------------------------------------
    # 2. Validate directories
    # ---------------------------------------------------------

    print("[2] Validating dataset directories...")

    images_dir, labels_dir = validate_directories()

    print("✓ Dataset directories valid")

    # ---------------------------------------------------------
    # 3. Validate image-label pairs
    # ---------------------------------------------------------

    print("[3] Validating image-label pairs...")

    image_files, label_files = (
        validate_image_label_pairs(
            images_dir,
            labels_dir,
        )
    )

    print(
        f"✓ Images: {len(image_files)}"
    )

    print(
        f"✓ Labels: {len(label_files)}"
    )

    # ---------------------------------------------------------
    # 4. Validate images
    # ---------------------------------------------------------

    print("[4] Validating image readability...")

    validate_images(image_files)

    print("✓ All images readable")

    # ---------------------------------------------------------
    # 5. Validate labels
    # ---------------------------------------------------------

    print("[5] Validating YOLO annotations...")

    total_annotations, empty_labels = validate_labels(
        label_files
    )

    print(
        f"✓ Annotations: {total_annotations}"
    )

    print(
        f"✓ Empty labels: {empty_labels}"
    )

    # ---------------------------------------------------------
    # 6. Validate summary
    # ---------------------------------------------------------

    print("[6] Validating summary consistency...")

    validate_summary(
        summary=summary,
        image_files=image_files,
        label_files=label_files,
        total_annotations=total_annotations,
    )

    print("✓ Summary is consistent")

    # ---------------------------------------------------------
    # 7. Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("E7 DATASET BUILDER VALIDATION PASSED")
    print("=" * 70)
    print()

    print(
        f"Dataset: {DATASET_DIR}"
    )


if __name__ == "__main__":
    main()