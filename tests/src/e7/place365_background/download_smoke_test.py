from pathlib import Path

from src.e7.place365_background.places365_downloader import (
    Places365Downloader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "E7_background_bank"
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

SEED = 42

IMAGES_PER_CATEGORY = 1


def main() -> None:
    downloader = Places365Downloader(
        seed=SEED,
        output_dir=OUTPUT_DIR,
    )

    counts = downloader.collect(
        categories=CATEGORIES,
        images_per_category=IMAGES_PER_CATEGORY,
        candidate_multiplier=2.0,
    )

    print()
    print("Smoke test completed.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()