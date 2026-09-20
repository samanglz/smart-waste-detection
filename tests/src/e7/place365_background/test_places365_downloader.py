from pathlib import Path

from src.e7.place365_background.places365_downloader import (
    Places365Downloader,
)


def test_random_selection_is_reproducible(tmp_path: Path) -> None:
    train_list = tmp_path / "places365_train_standard.txt"

    train_list.write_text(
        "\n".join(
            [
                "l/library/indoor/00000001.jpg 212",
                "l/library/indoor/00000002.jpg 212",
                "l/library/indoor/00000003.jpg 212",
                "l/library/indoor/00000004.jpg 212",
                "l/library/indoor/00000005.jpg 212",
                "o/office/00000001.jpg 244",
                "o/office/00000002.jpg 244",
                "o/office/00000003.jpg 244",
                "o/office/00000004.jpg 244",
                "o/office/00000005.jpg 244",
            ]
        ),
        encoding="utf-8",
    )

    downloader_1 = Places365Downloader(
        train_list_path=train_list,
        seed=42,
    )

    downloader_2 = Places365Downloader(
        train_list_path=train_list,
        seed=42,
    )

    result_1 = downloader_1.select_random_images(
        categories=["library/indoor", "office"],
        images_per_category=2,
    )

    result_2 = downloader_2.select_random_images(
        categories=["library/indoor", "office"],
        images_per_category=2,
    )

    selected_1 = {
        category: [image.image_path for image in images]
        for category, images in result_1.items()
    }

    selected_2 = {
        category: [image.image_path for image in images]
        for category, images in result_2.items()
    }

    assert selected_1 == selected_2


def test_category_parsing(tmp_path: Path) -> None:
    train_list = tmp_path / "places365_train_standard.txt"

    train_list.write_text(
        "l/library/indoor/00000001.jpg 212\n"
        "o/office/00000001.jpg 244\n",
        encoding="utf-8",
    )

    downloader = Places365Downloader(
        train_list_path=train_list,
    )

    images = downloader.load_image_list()

    assert images[0].category == "library/indoor"
    assert images[1].category == "office"