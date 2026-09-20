from __future__ import annotations

from collections import Counter

from src.e7.background_bank.sampler import (
    BackgroundSampler,
)


def _create_backgrounds() -> list[dict]:
    return [
        {
            "image_path": "library_01.jpg",
            "category": "library",
        },
        {
            "image_path": "library_02.jpg",
            "category": "library",
        },
        {
            "image_path": "home_01.jpg",
            "category": "home",
        },
        {
            "image_path": "home_02.jpg",
            "category": "home",
        },
        {
            "image_path": "home_03.jpg",
            "category": "home",
        },
        {
            "image_path": "desk_01.jpg",
            "category": "desk",
        },
    ]


def test_sampler_uses_all_backgrounds_once() -> None:
    sampler = BackgroundSampler(
        _create_backgrounds(),
        seed=42,
    )

    sampled = [
        sampler.sample()
        for _ in range(6)
    ]

    paths = [
        item["image_path"]
        for item in sampled
    ]

    assert len(paths) == 6
    assert len(set(paths)) == 6
    assert sampler.remaining() == 0


def test_sampler_preserves_category_distribution() -> None:
    backgrounds = _create_backgrounds()

    sampler = BackgroundSampler(
        backgrounds,
        seed=42,
    )

    sampled = [
        sampler.sample()
        for _ in range(len(backgrounds))
    ]

    counts = Counter(
        item["category"]
        for item in sampled
    )

    assert counts == Counter(
        {
            "library": 2,
            "home": 3,
            "desk": 1,
        }
    )


def test_sampler_restarts_after_cycle() -> None:
    sampler = BackgroundSampler(
        _create_backgrounds(),
        seed=42,
    )

    first_cycle = [
        sampler.sample()
        for _ in range(6)
    ]

    assert sampler.remaining() == 0

    second_sample = sampler.sample()

    assert second_sample in first_cycle
    assert sampler.remaining() == 5


def test_sampler_is_reproducible() -> None:
    backgrounds = _create_backgrounds()

    sampler_a = BackgroundSampler(
        backgrounds,
        seed=42,
    )

    sampler_b = BackgroundSampler(
        backgrounds,
        seed=42,
    )

    sequence_a = [
        sampler_a.sample()["image_path"]
        for _ in range(6)
    ]

    sequence_b = [
        sampler_b.sample()["image_path"]
        for _ in range(6)
    ]

    assert sequence_a == sequence_b


def test_sampler_rejects_empty_pool() -> None:
    try:
        BackgroundSampler([])
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError for empty pool."
        )