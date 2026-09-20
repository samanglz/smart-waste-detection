from pathlib import Path
import tempfile

import cv2
import numpy as np

from src.augmentations.targeted_augmentation import (
    TargetedAugmentation,
)


def create_test_sample(tmp_dir: Path):
    """
    Create a minimal YOLO image + label for testing.

    Class 1 = glass
    """

    image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Add a visible object-like region
    cv2.rectangle(
        image,
        (200, 200),
        (400, 400),
        (255, 255, 255),
        -1,
    )

    image_path = tmp_dir / "sample.jpg"
    label_path = tmp_dir / "sample.txt"

    cv2.imwrite(str(image_path), image)

    # YOLO format:
    # class x_center y_center width height
    label_path.write_text(
        "1 0.468750 0.468750 0.312500 0.312500\n"
    )

    return image_path, label_path


def test_initialization():

    augmenter = TargetedAugmentation()

    assert augmenter.target_class_ids == {1, 4}

    print("✅ Initialization test passed")


def test_contains_target_class():

    augmenter = TargetedAugmentation()

    with tempfile.TemporaryDirectory() as tmp:

        tmp_dir = Path(tmp)

        image_path, label_path = create_test_sample(tmp_dir)

        assert augmenter.contains_target_class(label_path)

    print("✅ Target class detection test passed")


def test_apply():

    augmenter = TargetedAugmentation()

    with tempfile.TemporaryDirectory() as tmp:

        tmp_dir = Path(tmp)

        image_path, label_path = create_test_sample(tmp_dir)

        output_image = tmp_dir / "sample_e2.jpg"
        output_label = tmp_dir / "sample_e2.txt"

        result = augmenter.apply(
            image_path=image_path,
            label_path=label_path,
            out_image=output_image,
            out_label=output_label,
        )

        assert result is True

        assert output_image.exists()
        assert output_label.exists()

        # Make sure generated image can actually be read
        image = cv2.imread(str(output_image))

        assert image is not None
        assert image.shape == (640, 640, 3)

        # Validate generated YOLO labels
        lines = output_label.read_text().splitlines()

        assert len(lines) > 0

        for line in lines:

            parts = line.split()

            assert len(parts) == 5

            cls, x, y, w, h = map(float, parts)

            assert int(cls) == 1

            assert 0 <= x <= 1
            assert 0 <= y <= 1
            assert 0 < w <= 1
            assert 0 < h <= 1

    print("✅ Augmentation application test passed")


if __name__ == "__main__":

    test_initialization()
    test_contains_target_class()
    test_apply()

    print()
    print("=" * 60)
    print("ALL TARGETED AUGMENTATION TESTS PASSED")
    print("=" * 60)