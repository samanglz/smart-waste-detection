import requests


DATASET = "Andron00e/Places365-custom"
CONFIG = "default"
SPLIT = "train"

API_URL = "https://datasets-server.huggingface.co/rows"


def test_places365_rows_api() -> None:
    response = requests.get(
        API_URL,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": 0,
            "length": 1,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    assert "features" in data
    assert "rows" in data
    assert len(data["rows"]) == 1

    row = data["rows"][0]["row"]

    print()
    print("=" * 70)
    print("PLACES365 API SMOKE TEST")
    print("=" * 70)

    print("Image path:")
    print(row["image_file_path"])

    print()
    print("Label:")
    print(row["labels"])

    print()
    print("Image metadata:")
    print(row["image"])

    print("=" * 70)

    assert "image_file_path" in row
    assert "labels" in row
    assert "image" in row

    image = row["image"]

    assert "src" in image
    assert image["src"].startswith("http")

    print()
    print("SUCCESS: image URL is available.")