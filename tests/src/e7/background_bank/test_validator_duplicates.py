from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.e7.background_bank.validator import BackgroundValidator


TEST_ROOT = Path(
    "tests/trash/E7_background_validator_duplicates_test"
)


def _create_base_image(
    path: Path,
    width: int = 800,
    height: int = 600,
) -> None:
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (80, 80),
        (width - 80, height - 80),
        (180, 180, 180),
        -1,
    )

    cv2.circle(
        image,
        (width // 2, height // 2),
        120,
        (70, 70, 70),
        -1,
    )

    cv2.line(
        image,
        (0, 0),
        (width, height),
        (255, 255, 255),
        5,
    )

    success = cv2.imwrite(str(path), image)

    assert success


def test_validator_detects_exact_duplicate() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    TEST_ROOT.mkdir(parents=True)

    image_a = TEST_ROOT / "image_a.jpg"
    image_b = TEST_ROOT / "image_b.jpg"

    _create_base_image(image_a)

    shutil.copy2(
        image_a,
        image_b,
    )

    validator = BackgroundValidator()

    first_result = validator.validate(image_a)

    assert first_result.usable is True
    assert first_result.content_hash is not None
    assert first_result.perceptual_hash is not None

    content_hashes = {
        first_result.content_hash,
    }

    perceptual_hashes = [
        first_result.perceptual_hash,
    ]

    second_result = validator.validate(
        image_b,
        existing_content_hashes=content_hashes,
        existing_perceptual_hashes=perceptual_hashes,
    )

    assert second_result.usable is False

    assert "exact_duplicate" in second_result.reasons
    assert "near_duplicate" in second_result.reasons


def test_validator_detects_near_duplicate() -> None:
    image_a = TEST_ROOT / "original.jpg"
    image_b = TEST_ROOT / "resized.jpg"

    _create_base_image(
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

    validator = BackgroundValidator()

    first_result = validator.validate(image_a)

    assert first_result.usable is True

    content_hashes = {
        first_result.content_hash,
    }

    perceptual_hashes = [
        first_result.perceptual_hash,
    ]

    second_result = validator.validate(
        image_b,
        existing_content_hashes=content_hashes,
        existing_perceptual_hashes=perceptual_hashes,
    )

    assert second_result.usable is False

    assert "near_duplicate" in second_result.reasons


def test_validator_accepts_unique_image() -> None:
    image_a = TEST_ROOT / "unique_a.jpg"
    image_b = TEST_ROOT / "unique_b.jpg"

    _create_base_image(image_a)

    rng = np.random.default_rng(42)

    different_image = rng.integers(
        0,
        256,
        size=(600, 800, 3),
        dtype=np.uint8,
    )

    success = cv2.imwrite(
        str(image_b),
        different_image,
    )

    assert success

    validator = BackgroundValidator()

    first_result = validator.validate(image_a)

    assert first_result.usable is True

    content_hashes = {
        first_result.content_hash,
    }

    perceptual_hashes = [
        first_result.perceptual_hash,
    ]

    second_result = validator.validate(
        image_b,
        existing_content_hashes=content_hashes,
        existing_perceptual_hashes=perceptual_hashes,
    )

    assert second_result.usable is True

    assert "exact_duplicate" not in second_result.reasons
    assert "near_duplicate" not in second_result.reasons