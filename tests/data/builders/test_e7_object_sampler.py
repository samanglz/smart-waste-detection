"""
Tests for E7 object sampling strategies.
"""

from __future__ import annotations

from src.e7.object_sampling import BalancedObjectSampler


def create_test_pool(size: int) -> list[dict]:
    return [
        {
            "object_id": f"object_{index:03d}",
            "class_id": index % 5,
            "class_name": f"class_{index % 5}",
        }
        for index in range(size)
    ]


def test_empty_pool_rejected() -> None:
    sampler = BalancedObjectSampler(seed=42)

    try:
        sampler.sample([])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty object pool should raise ValueError."
        )


def test_no_repetition_within_cycle() -> None:
    pool = create_test_pool(10)

    sampler = BalancedObjectSampler(seed=42)

    selected = [
        sampler.sample(pool)
        for _ in range(10)
    ]

    selected_ids = [
        obj["object_id"]
        for obj in selected
    ]

    assert len(selected_ids) == 10
    assert len(set(selected_ids)) == 10


def test_second_cycle_starts_after_full_cycle() -> None:
    pool = create_test_pool(10)

    sampler = BalancedObjectSampler(seed=42)

    selected = [
        sampler.sample(pool)
        for _ in range(11)
    ]

    selected_ids = [
        obj["object_id"]
        for obj in selected
    ]

    first_cycle = selected_ids[:10]
    second_cycle_object = selected_ids[10]

    assert len(set(first_cycle)) == 10

    assert second_cycle_object in first_cycle


def test_reproducibility() -> None:
    pool = create_test_pool(20)

    sampler_a = BalancedObjectSampler(seed=42)
    sampler_b = BalancedObjectSampler(seed=42)

    selected_a = [
        sampler_a.sample(pool)
        for _ in range(20)
    ]

    selected_b = [
        sampler_b.sample(pool)
        for _ in range(20)
    ]

    ids_a = [
        obj["object_id"]
        for obj in selected_a
    ]

    ids_b = [
        obj["object_id"]
        for obj in selected_b
    ]

    assert ids_a == ids_b


def test_different_seed_changes_order() -> None:
    pool = create_test_pool(20)

    sampler_a = BalancedObjectSampler(seed=42)
    sampler_b = BalancedObjectSampler(seed=123)

    selected_a = [
        sampler_a.sample(pool)
        for _ in range(20)
    ]

    selected_b = [
        sampler_b.sample(pool)
        for _ in range(20)
    ]

    ids_a = [
        obj["object_id"]
        for obj in selected_a
    ]

    ids_b = [
        obj["object_id"]
        for obj in selected_b
    ]

    assert ids_a != ids_b


def test_partial_cycle_uses_unique_objects() -> None:
    pool = create_test_pool(100)

    sampler = BalancedObjectSampler(seed=42)

    selected = [
        sampler.sample(pool)
        for _ in range(50)
    ]

    selected_ids = [
        obj["object_id"]
        for obj in selected
    ]

    assert len(selected_ids) == 50
    assert len(set(selected_ids)) == 50


def main() -> None:
    print("=" * 70)
    print("E7 OBJECT SAMPLER TEST")
    print("=" * 70)

    tests = [
        (
            "Empty pool rejection",
            test_empty_pool_rejected,
        ),
        (
            "No repetition within cycle",
            test_no_repetition_within_cycle,
        ),
        (
            "Second cycle",
            test_second_cycle_starts_after_full_cycle,
        ),
        (
            "Reproducibility",
            test_reproducibility,
        ),
        (
            "Different seeds",
            test_different_seed_changes_order,
        ),
        (
            "Partial cycle",
            test_partial_cycle_uses_unique_objects,
        ),
    ]

    for index, (name, test) in enumerate(
        tests,
        start=1,
    ):
        test()
        print(f"[{index}] {name} ✓")

    print()
    print("ALL OBJECT SAMPLER TESTS PASSED")


if __name__ == "__main__":
    main()