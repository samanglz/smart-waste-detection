from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

from src.e7.place365_background.places365_downloader import (
    Places365Downloader,
)


CATEGORIES = [
    "library/indoor",
    "living_room",
    "bedroom",
    "kitchen",
    "home_office",
    "office",
    "classroom",
    "dining_room",
    "coffee_shop",
    "cafeteria",
]

SEED = 123
NUM_BATCHES = 30
BATCH_SIZE = 100
DATASET_SIZE = 1_839_960


def test_sampling_distribution() -> None:
    downloader = Places365Downloader(
        output_dir=Path(
            "tests/trash/places365_diagnostic"
        ),
        seed=SEED,
        request_interval=1.2,
        max_api_retries=8,
        rate_limit_base_sleep=10.0,
    )

    rng = random.Random(SEED)

    offsets = rng.sample(
        range(
            0,
            DATASET_SIZE - BATCH_SIZE,
        ),
        NUM_BATCHES,
    )

    category_counts = Counter()
    label_counts = Counter()

    target_category_counts = Counter()
    target_label_counts = Counter()

    total_rows = 0
    parsed_rows = 0
    skipped_rows = 0

    print()
    print("=" * 90)
    print("PLACES365 SAMPLING DISTRIBUTION DIAGNOSTIC")
    print("=" * 90)
    print(
        f"Dataset rows : {DATASET_SIZE:,}"
    )
    print(
        f"Random batches : {NUM_BATCHES}"
    )
    print(
        f"Rows per batch: {BATCH_SIZE}"
    )
    print(
        f"Expected rows : "
        f"{NUM_BATCHES * BATCH_SIZE}"
    )
    print(
        f"Seed          : {SEED}"
    )
    print("=" * 90)

    for batch_index, offset in enumerate(
        offsets,
        start=1,
    ):
        data = downloader._request_rows(
            offset=offset,
            length=BATCH_SIZE,
        )

        rows = data.get(
            "rows",
            [],
        )

        total_rows += len(rows)

        print(
            f"Batch {batch_index:02d}/"
            f"{NUM_BATCHES} "
            f"| offset={offset:>8,} "
            f"| rows={len(rows):>3}"
        )

        for item in rows:
            parsed = downloader._parse_row(
                item
            )

            if parsed is None:
                skipped_rows += 1
                continue

            parsed_rows += 1

            category = parsed[
                "category"
            ]

            label = parsed.get(
                "label"
            )

            category_counts[
                category
            ] += 1

            if label is not None:
                label_counts[
                    int(label)
                ] += 1

            if category in CATEGORIES:
                target_category_counts[
                    category
                ] += 1

                if label is not None:
                    target_label_counts[
                        int(label)
                    ] += 1

    print()
    print("=" * 90)
    print("TARGET CATEGORY COUNTS")
    print("=" * 90)

    for category in CATEGORIES:
        count = target_category_counts[
            category
        ]

        print(
            f"{category:<20}: "
            f"{count:>5}"
        )

    print()
    print("=" * 90)
    print("TARGET CATEGORY LABELS")
    print("=" * 90)

    for category in CATEGORIES:
        expected_label = (
            downloader.CATEGORY_LABELS[
                category
            ]
        )

        observed = target_label_counts[
            expected_label
        ]

        category_count = (
            target_category_counts[
                category
            ]
        )

        print(
            f"{category:<20} "
            f"expected_label={expected_label:<3} "
            f"observed_label_count={observed:<5} "
            f"category_count={category_count:<5}"
        )

    print()
    print("=" * 90)
    print("TOP OBSERVED CATEGORIES")
    print("=" * 90)

    for category, count in (
        category_counts.most_common(30)
    ):
        print(
            f"{category:<35}: "
            f"{count:>5}"
        )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"Requested rows : "
        f"{NUM_BATCHES * BATCH_SIZE:,}"
    )
    print(
        f"Received rows  : "
        f"{total_rows:,}"
    )
    print(
        f"Parsed rows    : "
        f"{parsed_rows:,}"
    )
    print(
        f"Skipped rows   : "
        f"{skipped_rows:,}"
    )
    print(
        f"Unique categories: "
        f"{len(category_counts)}"
    )
    print("=" * 90)

    assert total_rows > 0
    assert parsed_rows > 0