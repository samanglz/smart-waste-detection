from __future__ import annotations

import requests


API_URL = (
    "https://datasets-server.huggingface.co/filter"
)

DATASET = "Andron00e/Places365-custom"
CONFIG = "default"
SPLIT = "train"


def test_filter_classroom():
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "where": '"labels"=92',
        "offset": 0,
        "length": 5,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    print("\npartial:", data.get("partial"))

    rows = data.get("rows", [])

    print("rows:", len(rows))

    for row in rows:
        print(
            row["row"]["image_file_path"],
            row["row"]["labels"],
        )

    assert rows

    for row in rows:
        assert row["row"]["labels"] == 92