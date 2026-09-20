import requests


DATASET = "Andron00e/Places365-custom"
CONFIG = "default"
SPLIT = "train"

API_URL = "https://datasets-server.huggingface.co/rows"

OFFSETS = [
    0,
    1_000,
    10_000,
    100_000,
    500_000,
    1_000_000,
    1_500_000,
    1_800_000,
]


def get_row(offset: int) -> dict:
    response = requests.get(
        API_URL,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": 1,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if not data["rows"]:
        raise RuntimeError(f"No row returned for offset={offset}")

    return data["rows"][0]


def test_places365_row_order() -> None:
    print()
    print("=" * 90)
    print("PLACES365 ROW ORDER PROBE")
    print("=" * 90)

    for offset in OFFSETS:
        item = get_row(offset)

        row = item["row"]

        image_path = row["image_file_path"]
        label = row["labels"]

        print()
        print(f"Offset : {offset:,}")
        print(f"Path   : {image_path}")
        print(f"Label  : {label}")

    print()
    print("=" * 90)
    print("SUCCESS: all requested offsets returned valid rows.")
    print("=" * 90)