from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.e7.background_bank.validator import BackgroundValidator


TEST_ROOT = Path("tests/trash/E7_background_validator_test")


def _create_image(
    path: Path,
    width: int,
    height: int,
) -> None:
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Create texture and edges so the image
    # behaves more like a real background.
    for y in range(0, height, 40):
        cv2.line(
            image,
            (0, y),
            (width, y),
            (80, 80, 80),
            3,
        )

    for x in range(0, width, 40):
        cv2.line(
            image,
            (x, 0),
            (x, height),
            (180, 180, 180),
            3,
        )

    cv2.rectangle(
        image,
        (100, 100),
        (width - 100, height - 100),
        (220, 220, 220),
        -1,
    )

    success = cv2.imwrite(str(path), image)

    assert success


def test_background_validator() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    TEST_ROOT.mkdir(parents=True)

    valid_image = TEST_ROOT / "valid.jpg"
    small_image = TEST_ROOT / "small.jpg"

    _create_image(
        valid_image,
        width=1280,
        height=720,
    )

    _create_image(
        small_image,
        width=100,
        height=100,
    )

    validator = BackgroundValidator()

    valid_result = validator.validate(valid_image)

    small_result = validator.validate(small_image)

    assert valid_result is not None
    assert small_result is not None

    assert valid_result.width == 1280
    assert valid_result.height == 720

    assert valid_result.usable is True

    assert small_result.usable is False

    print("\nBackgroundValidator smoke test passed.")   