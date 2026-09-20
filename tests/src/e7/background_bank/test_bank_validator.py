from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.e7.background_bank.bank_validator import (
    BackgroundBankValidator,
)


TEST_ROOT = Path(
    "tests/trash/E7_background_bank_test"
)


def _create_image(
    path: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    image = rng.integers(
        0,
        256,
        size=(600, 800, 3),
        dtype=np.uint8,
    )

    success = cv2.imwrite(
        str(path),
        image,
    )

    assert success


def test_bank_validator_creates_manifests_and_report() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    images_dir = TEST_ROOT / "images"

    library_dir = images_dir / "library"
    home_dir = images_dir / "home"

    library_dir.mkdir(
        parents=True,
    )

    home_dir.mkdir(
        parents=True,
    )

    _create_image(
        library_dir / "library_01.jpg",
        seed=1,
    )

    _create_image(
        home_dir / "home_01.jpg",
        seed=2,
    )

    validator = BackgroundBankValidator(
        bank_dir=TEST_ROOT,
    )

    summary = validator.validate()

    assert summary["total_images"] == 2
    assert summary["approved_images"] == 2
    assert summary["rejected_images"] == 0
    assert summary["approval_rate"] == 1.0

    assert (
        TEST_ROOT
        / "manifests"
        / "all.json"
    ).exists()

    assert (
        TEST_ROOT
        / "manifests"
        / "approved.json"
    ).exists()

    assert (
        TEST_ROOT
        / "reports"
        / "validation_report.json"
    ).exists()


def test_bank_validator_rejects_duplicate() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    images_dir = TEST_ROOT / "images"
    images_dir.mkdir(
        parents=True,
    )

    original = images_dir / "original.jpg"
    duplicate = images_dir / "duplicate.jpg"

    _create_image(
        original,
        seed=42,
    )

    shutil.copy2(
        original,
        duplicate,
    )

    validator = BackgroundBankValidator(
        bank_dir=TEST_ROOT,
    )

    summary = validator.validate()

    assert summary["total_images"] == 2
    assert summary["approved_images"] == 1
    assert summary["rejected_images"] == 1

    report_path = (
        TEST_ROOT
        / "reports"
        / "validation_report.json"
    )

    assert report_path.exists()