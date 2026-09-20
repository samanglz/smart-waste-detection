from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.e7.place365_background.places365_collector import (
    Places365Collector,
)


TEST_ROOT = Path(
    "tests/trash/E7_places365_collector_test"
)


def _create_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    image = rng.integers(
        0,
        256,
        size=(600, 800, 3),
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(path),
        image,
    )


def test_collects_limited_images_per_category() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    places_root = TEST_ROOT / "places365"

    for category in [
        "library",
        "kitchen",
    ]:
        category_dir = places_root / category
        category_dir.mkdir(
            parents=True,
        )

        for index in range(5):
            _create_image(
                category_dir / f"{index:04d}.jpg",
                seed=index,
            )

    output_dir = TEST_ROOT / "background_bank"

    collector = Places365Collector(
        places365_root=places_root,
        output_dir=output_dir,
    )

    summary = collector.collect_categories(
        categories=[
            "library",
            "kitchen",
        ],
        images_per_category=3,
    )

    assert summary == {
        "library": 3,
        "kitchen": 3,
    }

    assert len(
        list(
            (output_dir / "images" / "library").glob("*.jpg")
        )
    ) == 3

    assert len(
        list(
            (output_dir / "images" / "kitchen").glob("*.jpg")
        )
    ) == 3


def test_missing_category_raises_error() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    places_root = TEST_ROOT / "places365"
    places_root.mkdir(parents=True)

    collector = Places365Collector(
        places365_root=places_root,
        output_dir=TEST_ROOT / "output",
    )

    try:
        collector.collect_categories(
            categories=["library"],
            images_per_category=10,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )