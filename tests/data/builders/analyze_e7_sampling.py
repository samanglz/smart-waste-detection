"""
E7 Object Sampling Analysis.

This analysis does NOT generate a dataset.

It simulates the current random object-selection strategy
used by DatasetBuilder and reports:

    1. Object selection frequency
    2. Class selection distribution
    3. Small / Medium / Large distribution
    4. Expected number of synthetic images
    5. Most/least selected objects
    6. Reproducibility with a fixed seed

The Object Bank itself is never modified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OBJECT_BANK_DIR = (
    PROJECT_ROOT
    / "data"
    / "E7_object_bank"
)

MANIFEST_PATH = (
    OBJECT_BANK_DIR / "manifest.json"
)

NUM_BACKGROUND_IMAGES = 1673

SEED = 42

SMALL_THRESHOLD = 0.05
MEDIUM_THRESHOLD = 0.20


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


def load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), (
        f"Object Bank manifest not found: "
        f"{MANIFEST_PATH}"
    )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    assert isinstance(manifest, dict)

    assert manifest["experiment"] == "E7"
    assert manifest["stage"] == "object_bank"
    assert manifest["split"] == "train"

    return manifest


def build_object_pool(
    manifest: dict,
) -> list[dict]:

    objects = manifest.get("objects")

    assert isinstance(objects, list)
    assert objects

    pool = []

    for entry in objects:

        metadata_path = (
            OBJECT_BANK_DIR
            / entry["metadata_path"]
        )

        assert metadata_path.exists(), (
            f"Missing metadata: {metadata_path}"
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        area_pixels = int(
            metadata["mask"]["area_pixels"]
        )

        image_width = int(
            metadata["image"]["width"]
        )

        image_height = int(
            metadata["image"]["height"]
        )

        image_area = (
            image_width
            * image_height
        )

        area_ratio = (
            area_pixels / image_area
        )

        if area_ratio < SMALL_THRESHOLD:
            category = "small"
            generation_count = 3

        elif area_ratio < MEDIUM_THRESHOLD:
            category = "medium"
            generation_count = 2

        else:
            category = "large"
            generation_count = 1

        pool.append(
            {
                "object_id": entry["object_id"],
                "class_id": int(entry["class_id"]),
                "class_name": entry["class_name"],
                "area_ratio": area_ratio,
                "category": category,
                "generation_count": generation_count,
            }
        )

    return pool


def simulate_sampling(
    object_pool: list[dict],
    num_samples: int,
    seed: int,
) -> tuple[
    Counter,
    Counter,
    Counter,
    int,
]:

    rng = np.random.default_rng(seed)

    object_selection_counts = Counter()
    class_selection_counts = Counter()
    category_selection_counts = Counter()

    expected_synthetic_images = 0

    for _ in range(num_samples):

        index = int(
            rng.integers(
                0,
                len(object_pool),
            )
        )

        obj = object_pool[index]

        object_selection_counts[
            obj["object_id"]
        ] += 1

        class_selection_counts[
            obj["class_name"]
        ] += 1

        category_selection_counts[
            obj["category"]
        ] += 1

        expected_synthetic_images += (
            obj["generation_count"]
        )

    return (
        object_selection_counts,
        class_selection_counts,
        category_selection_counts,
        expected_synthetic_images,
    )


def print_object_bank_distribution(
    object_pool: list[dict],
) -> None:

    print("[1] Object Bank distribution...")

    class_counts = Counter(
        obj["class_name"]
        for obj in object_pool
    )

    category_counts = Counter(
        obj["category"]
        for obj in object_pool
    )

    print(
        f"✓ Total objects: {len(object_pool)}"
    )

    print()
    print("Class distribution:")

    for class_name in CLASS_NAMES:

        count = class_counts[class_name]

        percentage = (
            count
            / len(object_pool)
            * 100.0
        )

        print(
            f"  {class_name:<10} "
            f"{count:>4} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("Area category distribution:")

    for category in [
        "small",
        "medium",
        "large",
    ]:

        count = category_counts[category]

        percentage = (
            count
            / len(object_pool)
            * 100.0
        )

        print(
            f"  {category:<10} "
            f"{count:>4} "
            f"({percentage:6.2f}%)"
        )


def print_sampling_results(
    object_pool: list[dict],
    object_counts: Counter,
    class_counts: Counter,
    category_counts: Counter,
    expected_synthetic_images: int,
) -> None:

    print()
    print("[2] Sampling simulation...")
    print(
        f"✓ Samples: {NUM_BACKGROUND_IMAGES}"
    )

    print()
    print("Class selection distribution:")

    for class_name in CLASS_NAMES:

        count = class_counts[class_name]

        percentage = (
            count
            / NUM_BACKGROUND_IMAGES
            * 100.0
        )

        print(
            f"  {class_name:<10} "
            f"{count:>4} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("Area category distribution:")

    for category in [
        "small",
        "medium",
        "large",
    ]:

        count = category_counts[category]

        percentage = (
            count
            / NUM_BACKGROUND_IMAGES
            * 100.0
        )

        print(
            f"  {category:<10} "
            f"{count:>4} "
            f"({percentage:6.2f}%)"
        )

    print()
    print(
        "Expected synthetic images: "
        f"{expected_synthetic_images}"
    )


def print_object_frequency(
    object_pool: list[dict],
    object_counts: Counter,
) -> None:

    print()
    print("[3] Object selection frequency...")

    selected_counts = [
        object_counts[obj["object_id"]]
        for obj in object_pool
    ]

    never_selected = sum(
        count == 0
        for count in selected_counts
    )

    print(
        f"✓ Objects never selected: "
        f"{never_selected}"
    )

    print(
        f"✓ Minimum selections: "
        f"{min(selected_counts)}"
    )

    print(
        f"✓ Maximum selections: "
        f"{max(selected_counts)}"
    )

    print(
        f"✓ Mean selections: "
        f"{np.mean(selected_counts):.3f}"
    )

    print()
    print("Least selected objects:")

    least_selected = sorted(
        object_pool,
        key=lambda obj: (
            object_counts[obj["object_id"]],
            obj["object_id"],
        ),
    )[:10]

    for obj in least_selected:

        object_id = obj["object_id"]
        count = object_counts[object_id]

        print(
            f"  {object_id:<30} "
            f"{count:>3}"
        )

    print()
    print("Most selected objects:")

    most_selected = sorted(
        object_pool,
        key=lambda obj: (
            -object_counts[obj["object_id"]],
            obj["object_id"],
        ),
    )[:10]

    for obj in most_selected:

        object_id = obj["object_id"]
        count = object_counts[object_id]

        print(
            f"  {object_id:<30} "
            f"{count:>3}"
        )


def validate_reproducibility(
    object_pool: list[dict],
) -> None:

    print()
    print("[4] Checking reproducibility...")

    result_a = simulate_sampling(
        object_pool=object_pool,
        num_samples=NUM_BACKGROUND_IMAGES,
        seed=SEED,
    )

    result_b = simulate_sampling(
        object_pool=object_pool,
        num_samples=NUM_BACKGROUND_IMAGES,
        seed=SEED,
    )

    assert result_a == result_b

    print(
        f"✓ Fixed seed ({SEED}) "
        f"produces identical results"
    )


def print_sampling_interpretation(
    object_pool: list[dict],
    object_counts: Counter,
    class_counts: Counter,
    category_counts: Counter,
) -> None:

    print()
    print("[5] Sampling interpretation...")

    total_objects = len(object_pool)

    expected_selection_per_object = (
        NUM_BACKGROUND_IMAGES
        / total_objects
    )

    print(
        "Uniform expected selections/object: "
        f"{expected_selection_per_object:.3f}"
    )

    never_selected = sum(
        object_counts[obj["object_id"]] == 0
        for obj in object_pool
    )

    print(
        f"Objects with zero selections: "
        f"{never_selected}/{total_objects}"
    )

    print()
    print(
        "Important:"
    )

    print(
        "  Random sampling does NOT guarantee "
        "that every Object Bank object is used."
    )

    print(
        "  Class distribution follows the "
        "Object Bank distribution."
    )

    print(
        "  Small/medium/large generation counts "
        "change the final synthetic-image distribution."
    )


def main():
    print("=" * 70)
    print("E7 OBJECT SAMPLING ANALYSIS")
    print("=" * 70)

    manifest = load_manifest()

    object_pool = build_object_pool(
        manifest
    )

    print_object_bank_distribution(
        object_pool
    )

    (
        object_counts,
        class_counts,
        category_counts,
        expected_synthetic_images,
    ) = simulate_sampling(
        object_pool=object_pool,
        num_samples=NUM_BACKGROUND_IMAGES,
        seed=SEED,
    )

    print_sampling_results(
        object_pool=object_pool,
        object_counts=object_counts,
        class_counts=class_counts,
        category_counts=category_counts,
        expected_synthetic_images=expected_synthetic_images,
    )

    print_object_frequency(
        object_pool=object_pool,
        object_counts=object_counts,
    )

    validate_reproducibility(
        object_pool
    )

    print_sampling_interpretation(
        object_pool=object_pool,
        object_counts=object_counts,
        class_counts=class_counts,
        category_counts=category_counts,
    )

    print()
    print("=" * 70)
    print("E7 OBJECT SAMPLING ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()