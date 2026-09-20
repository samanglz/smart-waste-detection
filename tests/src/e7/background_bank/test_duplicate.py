from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.e7.background_bank.duplicate import (
    BackgroundDuplicateDetector,
)


TEST_ROOT = Path(
    "tests/trash/E7_background_duplicate_test"
)


def _create_image(
    path: Path,
    width: int = 800,
    height: int = 600,
) -> None:
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Add stable visual structure.
    cv2.rectangle(
        image,
        (100, 100),
        (width - 100, height - 100),
        (180, 180, 180),
        -1,
    )

    cv2.circle(
        image,
        (width // 2, height // 2),
        100,
        (80, 80, 80),
        -1,
    )

    cv2.line(
        image,
        (0, 0),
        (width, height),
        (255, 255, 255),
        5,
    )

    success = cv2.imwrite(
        str(path),
        image,
    )

    assert success


def test_exact_duplicate() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    TEST_ROOT.mkdir(parents=True)

    image_a = TEST_ROOT / "image_a.jpg"
    image_b = TEST_ROOT / "image_b.jpg"

    _create_image(image_a)

    # Exact byte-for-byte copy.
    shutil.copy2(
        image_a,
        image_b,
    )

    detector = BackgroundDuplicateDetector()

    result_a = detector.analyze(image_a)
    result_b = detector.analyze(image_b)

    assert result_a.readable is True
    assert result_b.readable is True

    assert result_a.content_hash == result_b.content_hash

    assert result_a.perceptual_hash == result_b.perceptual_hash


def test_near_duplicate() -> None:
    image_a = TEST_ROOT / "original.jpg"
    image_b = TEST_ROOT / "resized.jpg"

    _create_image(
        image_a,
        width=800,
        height=600,
    )

    original = cv2.imread(str(image_a))

    resized = cv2.resize(
        original,
        (640, 480),
        interpolation=cv2.INTER_AREA,
    )

    success = cv2.imwrite(
        str(image_b),
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    assert success

    detector = BackgroundDuplicateDetector()

    result_a = detector.analyze(image_a)
    result_b = detector.analyze(image_b)

    assert result_a.readable is True
    assert result_b.readable is True

    # File bytes differ, so SHA-256 must differ.
    assert result_a.content_hash != result_b.content_hash

    # Visual structure should remain similar.
    distance = detector.hamming_distance(
        result_a.perceptual_hash,
        result_b.perceptual_hash,
    )

    assert distance <= 8

    assert detector.are_near_duplicates(
        result_a.perceptual_hash,
        result_b.perceptual_hash,
        max_distance=8,
    )


def test_different_images() -> None:
    image_a = TEST_ROOT / "first.jpg"
    image_b = TEST_ROOT / "second.jpg"

    _create_image(image_a)

    different = np.random.default_rng(42).integers(
        0,
        256,
        size=(600, 800, 3),
        dtype=np.uint8,
    )

    success = cv2.imwrite(
        str(image_b),
        different,
    )

    assert success

    detector = BackgroundDuplicateDetector()

    result_a = detector.analyze(image_a)
    result_b = detector.analyze(image_b)

    assert result_a.readable is True
    assert result_b.readable is True

    assert result_a.content_hash != result_b.content_hash

    distance = detector.hamming_distance(
        result_a.perceptual_hash,
        result_b.perceptual_hash,
    )

    assert distance > 8


def test_missing_file() -> None:
    detector = BackgroundDuplicateDetector()

    result = detector.analyze(
        TEST_ROOT / "does_not_exist.jpg"
    )

    assert result.readable is False
    assert result.content_hash is None
    assert result.perceptual_hash is None


def test_invalid_parameters() -> None:
    try:
        BackgroundDuplicateDetector(
            hash_size=0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        BackgroundDuplicateDetector(
            hash_size=32,
            dct_size=8,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass