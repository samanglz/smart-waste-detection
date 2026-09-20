from __future__ import annotations

import requests


API_URL = (
    "https://datasets-server.huggingface.co/parquet"
)

DATASET = "Andron00e/Places365-custom-train"


def test_places365_parquet_manifest():
    response = requests.get(
        API_URL,
        params={"dataset": DATASET},
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    parquet_files = data["parquet_files"]

    print(
        f"\nParquet files: {len(parquet_files)}"
    )

    total_size = 0

    for item in parquet_files:
        size = int(item["size"])
        total_size += size

        print(
            f"{item['filename']}: "
            f"{size / (1024 ** 3):.2f} GB"
        )

    print(
        f"\nTotal Parquet size: "
        f"{total_size / (1024 ** 3):.2f} GB"
    )

    assert parquet_files
    assert all(
        item["split"] == "train"
        for item in parquet_files
    )