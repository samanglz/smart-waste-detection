from __future__ import annotations

from collections import Counter

from src.e7.place365_background.places365_downloader import (
    Places365Downloader,
)


DATASET_SIZE = 1_839_960
BATCH_SIZE = 100

# در diagnostic قبلی classroom در این offset دیده شد.
CLASSROOM_OFFSET = 1_000_000

WINDOW_BEFORE = 2
WINDOW_AFTER = 2

EXPECTED_CATEGORY = "classroom"
EXPECTED_LABEL = 92


def test_classroom_category_block():
    downloader = Places365Downloader(
        seed=123,
        request_interval=1.0,
        timeout=60,
    )

    categories: list[str] = []
    labels: list[int] = []

    offsets = range(
        max(
            0,
            CLASSROOM_OFFSET
            - WINDOW_BEFORE * BATCH_SIZE,
        ),
        min(
            DATASET_SIZE,
            CLASSROOM_OFFSET
            + (
                WINDOW_AFTER + 1
            ) * BATCH_SIZE,
        ),
        BATCH_SIZE,
    )

    for offset in offsets:
        data = downloader._request_rows(
            offset=offset,
            length=BATCH_SIZE,
        )

        rows = data.get("rows", [])

        print(
            f"\noffset={offset}, "
            f"rows={len(rows)}"
        )

        for item in rows:
            parsed = downloader._parse_row(
                item
            )

            if parsed is None:
                continue

            categories.append(
                parsed["category"]
            )

            labels.append(
                int(parsed["label"])
            )

    category_counts = Counter(
        categories
    )

    print("\nCategory distribution:")

    for category, count in (
        category_counts.most_common()
    ):
        print(
            f"  {category:<30} {count}"
        )

    classroom_count = category_counts.get(
        EXPECTED_CATEGORY,
        0,
    )

    classroom_labels = [
        label
        for category, label in zip(
            categories,
            labels,
        )
        if category == EXPECTED_CATEGORY
    ]

    print(
        f"\n{EXPECTED_CATEGORY}: "
        f"{classroom_count}"
    )

    print(
        f"labels: "
        f"{Counter(classroom_labels)}"
    )

    assert classroom_count > 0, (
        "No classroom images were found "
        "in the inspected window."
    )

    assert all(
        label == EXPECTED_LABEL
        for label in classroom_labels
    )