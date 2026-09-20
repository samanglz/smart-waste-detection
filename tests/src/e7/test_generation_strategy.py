"""
Tests for E7 generation count strategies.

Run:
    python -m tests.src.e7.test_generation_strategy
"""

from src.e7.generation_strategy import AreaBasedGenerationStrategy


def main():
    print("=" * 70)
    print("E7 GENERATION STRATEGY TEST")
    print("=" * 70)

    strategy = AreaBasedGenerationStrategy(
        small_threshold=0.05,
        medium_threshold=0.20,
    )

    image_width = 100
    image_height = 100

    image_area = image_width * image_height

    # ---------------------------------------------------------
    # Small
    # ---------------------------------------------------------

    print("\n[1] Testing Small Object...")

    decision = strategy.decide(
        object_area_pixels=300,
        image_width=image_width,
        image_height=image_height,
    )

    assert decision.category == "small"
    assert decision.count == 3

    print(
        f"✓ Small: category={decision.category}, "
        f"count={decision.count}, "
        f"ratio={decision.area_ratio:.4f}"
    )
    # ---------------------------------------------------------
    # Medium
    # ---------------------------------------------------------

    print("[2] Testing Medium Object...")

    decision = strategy.decide(
        object_area_pixels=1000,
        image_width=100,
        image_height=100,
    )

    assert decision.category == "medium"
    assert decision.count == 2

    print(
        f"✓ Medium: category={decision.category}, "
        f"count={decision.count}, "
        f"ratio={decision.area_ratio:.4f}"
    )

    # ---------------------------------------------------------
    # Large
    # ---------------------------------------------------------

    print("[3] Testing Large Object...")

    decision = strategy.decide(
        object_area_pixels=3000,
        image_width=100,
        image_height=100,
    )

    assert decision.category == "large"
    assert decision.count == 1

    print(
        f"✓ Large: category={decision.category}, "
        f"count={decision.count}, "
        f"ratio={decision.area_ratio:.4f}"
    )

    # ---------------------------------------------------------
    # Invalid area
    # ---------------------------------------------------------

    print("[4] Testing Invalid Object Area...")

    try:
        strategy.decide(
            object_area_pixels=0,
            image_width=image_width,
            image_height=image_height,
        )
    except ValueError:
        print("✓ Invalid area rejected")
    else:
        raise AssertionError(
            "Expected ValueError for zero object area."
        )

    # ---------------------------------------------------------
    # Invalid image dimensions
    # ---------------------------------------------------------

    print("[5] Testing Invalid Image Dimensions...")

    try:
        strategy.decide(
            object_area_pixels=100,
            image_width=0,
            image_height=image_height,
        )
    except ValueError:
        print("✓ Invalid image dimensions rejected")
    else:
        raise AssertionError(
            "Expected ValueError for invalid image dimensions."
        )

    print("\n" + "=" * 70)
    print("ALL GENERATION STRATEGY TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()