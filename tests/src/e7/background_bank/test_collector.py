from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.e7.background_bank.collector import BackgroundCollector


TEST_ROOT = Path("tests/trash/E7_background_collector_test")


def _create_test_image(path: Path, width: int, height: int) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)

    image[:, :] = (120, 150, 180)

    success = cv2.imwrite(str(path), image)

    assert success, f"Failed to create test image: {path}"


def test_background_collector() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)

    source_dir = TEST_ROOT / "source"
    output_dir = TEST_ROOT / "background_bank"

    source_dir.mkdir(parents=True)

    _create_test_image(
        source_dir / "library_01.jpg",
        width=800,
        height=600,
    )

    _create_test_image(
        source_dir / "library_02.jpg",
        width=1280,
        height=720,
    )

    collector = BackgroundCollector(
        output_dir=output_dir,
    )

    metadata_list = collector.collect(
        source_dir=source_dir,
        category="library",
        source="internet",
    )

    assert len(metadata_list) == 2

    for metadata in metadata_list:
        assert metadata.background_id.startswith("bg_")
        assert metadata.category == "library"
        assert metadata.source == "internet"

        output_path = Path(metadata.image_path)

        assert output_path.exists()
        assert output_path.parent == output_dir / "images" / "library"

        assert metadata.width > 0
        assert metadata.height > 0
        assert metadata.aspect_ratio > 0

    metadata_by_name = {
        Path(metadata.image_path).name: metadata
        for metadata in metadata_list
    }

    first = next(
        metadata
        for metadata in metadata_list
        if "library_01" in metadata.source_id
    )

    second = next(
        metadata
        for metadata in metadata_list
        if "library_02" in metadata.source_id
    )

    assert first.width == 800
    assert first.height == 600

    assert second.width == 1280
    assert second.height == 720

    print("\nBackgroundCollector smoke test passed.")