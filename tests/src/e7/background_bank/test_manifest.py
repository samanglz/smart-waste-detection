from __future__ import annotations

from pathlib import Path

from src.e7.background_bank.manifest import (
    BackgroundManifest,
)


TEST_MANIFEST = Path(
    "tests/trash/E7_background_manifest_test/approved.json"
)


def test_manifest_save_and_load() -> None:
    entries = [
        {
            "image_path": "images/library/bg_001.jpg",
            "category": "library",
            "usable": True,
        },
        {
            "image_path": "images/home/bg_002.jpg",
            "category": "home",
            "usable": True,
        },
        {
            "image_path": "images/library/bg_003.jpg",
            "category": "library",
            "usable": True,
        },
    ]

    manifest = BackgroundManifest(TEST_MANIFEST)

    manifest.save(entries)

    loaded = manifest.load()

    assert loaded == entries
    assert manifest.count() == 3

    assert manifest.categories() == {
        "library": 2,
        "home": 1,
    }   