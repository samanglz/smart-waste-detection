"""
E7 Sampling Strategy Comparison.

This module compares different object-sampling strategies
without generating any images or modifying the dataset.

Strategies:
    1. Random
    2. Balanced Object
    3. Class + Area Balanced

The simulation uses the real E7 Object Bank and the real
number of training backgrounds.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OBJECT_BANK_DIR = PROJECT_ROOT / "data" / "E7_object_bank"
MANIFEST_PATH = OBJECT_BANK_DIR / "manifest.json"

NUM_SAMPLES = 1673
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

AREA_CATEGORIES = [
    "small",
    "medium",
    "large",
]


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Object Bank manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Object Bank manifest must be a dictionary.")

    return manifest


def build_object_pool(manifest: dict) -> list[dict]:
    objects = manifest.get("objects")

    if not isinstance(objects, list) or not objects:
        raise ValueError("Object Bank contains no objects.")

    pool = []

    for entry in objects:
        metadata_path = OBJECT_BANK_DIR / entry["metadata_path"]

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata: {metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        area_pixels = int(metadata["mask"]["area_pixels"])
        image_width = int(metadata["image"]["width"])
        image_height = int(metadata["image"]["height"])

        image_area = image_width * image_height
        area_ratio = area_pixels / image_area

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


def random_sampling(
    pool: list[dict],
    num_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    indices = rng.integers(
        0,
        len(pool),
        size=num_samples,
    )

    return [pool[int(index)] for index in indices]


def balanced_object_sampling(
    pool: list[dict],
    num_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Cycle through a shuffled Object Bank.

    Every object gets used before any object is reused.
    """

    selected = []

    shuffled_pool = list(pool)

    while len(selected) < num_samples:
        rng.shuffle(shuffled_pool)

        remaining = num_samples - len(selected)

        selected.extend(
            shuffled_pool[:remaining]
        )

    return selected[:num_samples]


def class_area_balanced_sampling(
    pool: list[dict],
    num_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Balanced sampling across:

        class × area category

    There are 5 classes × 3 area categories = 15 groups.

    Each group receives approximately the same number
    of selections.

    Objects inside each group are sampled cyclically,
    so the strategy does not repeatedly select the same
    object while ignoring others.
    """

    groups = defaultdict(list)

    for obj in pool:
        key = (
            obj["class_name"],
            obj["category"],
        )
        groups[key].append(obj)

    group_keys = sorted(groups.keys())

    selected = []

    shuffled_groups = {}

    for key in group_keys:
        objects = list(groups[key])
        rng.shuffle(objects)
        shuffled_groups[key] = objects

    for index in range(num_samples):
        group_key = group_keys[index % len(group_keys)]

        group_objects = shuffled_groups[group_key]

        object_index = (
            index // len(group_keys)
        ) % len(group_objects)

        selected.append(
            group_objects[object_index]
        )

    return selected


def calculate_statistics(
    selected: list[dict],
    pool: list[dict],
) -> dict:

    object_counts = Counter(
        obj["object_id"]
        for obj in selected
    )

    class_counts = Counter(
        obj["class_name"]
        for obj in selected
    )

    area_counts = Counter(
        obj["category"]
        for obj in selected
    )

    generation_counts = Counter(
        obj["generation_count"]
        for obj in selected
    )

    total_synthetic_images = sum(
        obj["generation_count"]
        for obj in selected
    )

    selection_values = [
        object_counts[obj["object_id"]]
        for obj in pool
    ]

    return {
        "object_counts": object_counts,
        "class_counts": class_counts,
        "area_counts": area_counts,
        "generation_counts": generation_counts,
        "total_synthetic_images": total_synthetic_images,
        "never_selected": sum(
            count == 0
            for count in selection_values
        ),
        "min_object_selection": min(selection_values),
        "max_object_selection": max(selection_values),
        "mean_object_selection": float(
            np.mean(selection_values)
        ),
    }


def print_strategy_result(
    name: str,
    statistics: dict,
    pool_size: int,
) -> None:

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Never selected objects : "
        f"{statistics['never_selected']}"
    )

    print(
        f"Min selections/object  : "
        f"{statistics['min_object_selection']}"
    )

    print(
        f"Max selections/object  : "
        f"{statistics['max_object_selection']}"
    )

    print(
        f"Mean selections/object : "
        f"{statistics['mean_object_selection']:.3f}"
    )

    print(
        f"Synthetic images       : "
        f"{statistics['total_synthetic_images']}"
    )

    print()
    print("Class distribution:")

    for class_name in CLASS_NAMES:
        count = statistics["class_counts"][class_name]

        percentage = (
            count / NUM_SAMPLES * 100
        )

        print(
            f"  {class_name:<10} "
            f"{count:>4} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("Area distribution:")

    for category in AREA_CATEGORIES:
        count = statistics["area_counts"][category]

        percentage = (
            count / NUM_SAMPLES * 100
        )

        print(
            f"  {category:<10} "
            f"{count:>4} "
            f"({percentage:6.2f}%)"
        )

    coverage = (
        (pool_size - statistics["never_selected"])
        / pool_size
        * 100
    )

    print()
    print(
        f"Object Bank coverage : "
        f"{coverage:.2f}%"
    )


def print_group_distribution(
    selected: list[dict],
) -> None:

    print()
    print("Class × Area distribution:")

    counts = Counter(
        (
            obj["class_name"],
            obj["category"],
        )
        for obj in selected
    )

    print()

    header = (
        f"{'Class':<12}"
        f"{'Small':>10}"
        f"{'Medium':>10}"
        f"{'Large':>10}"
    )

    print(header)
    print("-" * 42)

    for class_name in CLASS_NAMES:

        values = []

        for category in AREA_CATEGORIES:
            values.append(
                counts[
                    (class_name, category)
                ]
            )

        print(
            f"{class_name:<12}"
            f"{values[0]:>10}"
            f"{values[1]:>10}"
            f"{values[2]:>10}"
        )


def compare_strategies(
    pool: list[dict],
) -> None:

    print("=" * 70)
    print("E7 SAMPLING STRATEGY COMPARISON")
    print("=" * 70)

    print()
    print(f"Object Bank size : {len(pool)}")
    print(f"Samples         : {NUM_SAMPLES}")
    print(f"Seed            : {SEED}")

    # ---------------------------------------------------------
    # Strategy 1: Random
    # ---------------------------------------------------------

    rng = np.random.default_rng(SEED)

    random_selected = random_sampling(
        pool,
        NUM_SAMPLES,
        rng,
    )

    random_stats = calculate_statistics(
        random_selected,
        pool,
    )

    print_strategy_result(
        "STRATEGY 1 — RANDOM",
        random_stats,
        len(pool),
    )

    print_group_distribution(
        random_selected
    )

    # ---------------------------------------------------------
    # Strategy 2: Balanced Object
    # ---------------------------------------------------------

    rng = np.random.default_rng(SEED)

    balanced_selected = balanced_object_sampling(
        pool,
        NUM_SAMPLES,
        rng,
    )

    balanced_stats = calculate_statistics(
        balanced_selected,
        pool,
    )

    print_strategy_result(
        "STRATEGY 2 — BALANCED OBJECT",
        balanced_stats,
        len(pool),
    )

    print_group_distribution(
        balanced_selected
    )

    # ---------------------------------------------------------
    # Strategy 3: Class + Area Balanced
    # ---------------------------------------------------------

    rng = np.random.default_rng(SEED)

    class_area_selected = class_area_balanced_sampling(
        pool,
        NUM_SAMPLES,
        rng,
    )

    class_area_stats = calculate_statistics(
        class_area_selected,
        pool,
    )

    print_strategy_result(
        "STRATEGY 3 — CLASS + AREA BALANCED",
        class_area_stats,
        len(pool),
    )

    print_group_distribution(
        class_area_selected
    )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Strategy':<30}"
        f"{'Unused':>10}"
        f"{'Coverage':>12}"
        f"{'Synthetic':>12}"
    )

    print("-" * 66)

    strategies = [
        (
            "Random",
            random_stats,
        ),
        (
            "Balanced Object",
            balanced_stats,
        ),
        (
            "Class + Area Balanced",
            class_area_stats,
        ),
    ]

    for name, stats in strategies:

        coverage = (
            (
                len(pool)
                - stats["never_selected"]
            )
            / len(pool)
            * 100
        )

        print(
            f"{name:<30}"
            f"{stats['never_selected']:>10}"
            f"{coverage:>11.2f}%"
            f"{stats['total_synthetic_images']:>12}"
        )

    print()
    print("=" * 70)
    print("E7 SAMPLING STRATEGY COMPARISON COMPLETED")
    print("=" * 70)


def main() -> None:

    manifest = load_manifest()

    pool = build_object_pool(
        manifest
    )

    compare_strategies(pool)


if __name__ == "__main__":
    main()