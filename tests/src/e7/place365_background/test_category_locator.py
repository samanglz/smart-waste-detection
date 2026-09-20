from pathlib import Path

from src.e7.place365_background.category_locator import (
    Places365CategoryLocator,
)


def create_train_list(
    path: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "l/library/indoor/00000001.jpg 212",
                "l/library/indoor/00000002.jpg 212",
                "l/library/indoor/00000003.jpg 212",
                "o/office/00000001.jpg 244",
                "o/office/00000002.jpg 244",
                "b/bedroom/00000001.jpg 52",
            ]
        ),
        encoding="utf-8",
    )


def test_category_locator_loads_categories(
    tmp_path: Path,
) -> None:
    train_list = (
        tmp_path
        / "places365_train_standard.txt"
    )

    create_train_list(train_list)

    locator = Places365CategoryLocator(
        train_list_path=train_list,
        seed=42,
    )

    locator.load()

    assert locator.categories() == [
        "bedroom",
        "library/indoor",
        "office",
    ]


def test_category_counts(
    tmp_path: Path,
) -> None:
    train_list = (
        tmp_path
        / "places365_train_standard.txt"
    )

    create_train_list(train_list)

    locator = Places365CategoryLocator(
        train_list_path=train_list,
    )

    locator.load()

    assert locator.count(
        "library/indoor"
    ) == 3

    assert locator.count(
        "office"
    ) == 2

    assert locator.count(
        "bedroom"
    ) == 1


def test_category_contains(
    tmp_path: Path,
) -> None:
    train_list = (
        tmp_path
        / "places365_train_standard.txt"
    )

    create_train_list(train_list)

    locator = Places365CategoryLocator(
        train_list_path=train_list,
    )

    locator.load()

    assert locator.contains(
        "library/indoor"
    )

    assert not locator.contains(
        "kitchen"
    )


def test_sample_is_reproducible(
    tmp_path: Path,
) -> None:
    train_list = (
        tmp_path
        / "places365_train_standard.txt"
    )

    train_list.write_text(
        "\n".join(
            [
                f"l/library/indoor/"
                f"{index:08d}.jpg 212"
                for index in range(20)
            ]
        ),
        encoding="utf-8",
    )

    locator_1 = Places365CategoryLocator(
        train_list_path=train_list,
        seed=42,
    )

    locator_2 = Places365CategoryLocator(
        train_list_path=train_list,
        seed=42,
    )

    selected_1 = locator_1.sample(
        "library/indoor",
        5,
    )

    selected_2 = locator_2.sample(
        "library/indoor",
        5,
    )

    assert selected_1 == selected_2


def test_sample_does_not_exceed_category_size(
    tmp_path: Path,
) -> None:
    train_list = (
        tmp_path
        / "places365_train_standard.txt"
    )

    create_train_list(train_list)

    locator = Places365CategoryLocator(
        train_list_path=train_list,
    )

    selected = locator.sample(
        "office",
        100,
    )

    assert len(selected) == 2


def test_category_label_validation(
    tmp_path: Path,
) -> None:
    train_list = (
        tmp_path
        / "places365_train_standard.txt"
    )

    create_train_list(train_list)

    locator = Places365CategoryLocator(
        train_list_path=train_list,
    )

    locator.load()

    assert locator.validate_category_label(
        "library/indoor",
        212,
    )

    assert not locator.validate_category_label(
        "library/indoor",
        244,
    )