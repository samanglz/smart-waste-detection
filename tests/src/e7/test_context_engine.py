"""
E7 Context Engine integration tests.

Tests:
    1. Scale
    2. Lighting
    3. Color Temperature
    4. Reflection
    5. Occlusion
    6. Compositor
    7. Full Context Engine
    8. Final object mask
"""

from pathlib import Path

import cv2
import numpy as np

from src.e7.context_engine import (
    ColorTemperatureEngine,
    Compositor,
    ContextEngine,
    LightingEngine,
    OcclusionEngine,
    RandomScaleStrategy,
    ReflectionEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "trash" /  "E7_context_engine_test"


def create_test_images():
    """Create deterministic dummy background and RGBA object."""

    background = np.full(
        (480, 640, 3),
        160,
        dtype=np.uint8,
    )

    object_rgba = np.zeros(
        (160, 120, 4),
        dtype=np.uint8,
    )

    # RGB object
    object_rgba[:, :, 0] = 80
    object_rgba[:, :, 1] = 160
    object_rgba[:, :, 2] = 220

    # Fully opaque object
    object_rgba[:, :, 3] = 255

    return background, object_rgba


def check_mask(mask: np.ndarray, placement) -> None:
    """Validate the final object mask."""

    assert isinstance(mask, np.ndarray)

    assert mask.ndim == 2

    assert mask.shape == (
        placement.height,
        placement.width,
    )

    assert mask.dtype == np.uint8

    unique_values = np.unique(mask)

    assert set(unique_values).issubset({0, 255})

    assert np.any(mask == 255), (
        "Object mask must contain visible object pixels."
    )

    assert np.any(mask == 0), (
        "Object mask should contain transparent/background pixels."
    )


def main():
    print("=" * 70)
    print("E7 CONTEXT ENGINE TEST")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    background, object_rgba = create_test_images()

    # ---------------------------------------------------------
    # Initialize components
    # ---------------------------------------------------------

    scale = RandomScaleStrategy(
        min_relative_height=0.10,
        max_relative_height=0.35,
    )

    lighting = LightingEngine()
    color_temperature = ColorTemperatureEngine()
    reflection = ReflectionEngine()
    occlusion = OcclusionEngine()
    compositor = Compositor()

    engine = ContextEngine(
        scale_strategy=scale,
        lighting=lighting,
        color_temperature=color_temperature,
        reflection=reflection,
        occlusion=occlusion,
        compositor=compositor,
        seed=42,
    )

    # ---------------------------------------------------------
    # 1. Scale
    # ---------------------------------------------------------

    print("[1] Testing Scale...")

    placement = scale.calculate(
        obj_w=object_rgba.shape[1],
        obj_h=object_rgba.shape[0],
        img_w=background.shape[1],
        img_h=background.shape[0],
        rng=np.random.default_rng(42),
    )

    assert placement.width > 0
    assert placement.height > 0

    assert placement.x >= 0
    assert placement.y >= 0

    assert (
        placement.x + placement.width
        <= background.shape[1]
    )

    assert (
        placement.y + placement.height
        <= background.shape[0]
    )

    print("✓ Scale")

    # ---------------------------------------------------------
    # 2. Lighting
    # ---------------------------------------------------------

    print("[2] Testing Lighting...")

    lighting_result = lighting.apply(
        rgba=object_rgba,
        brightness_factor=1.1,
    )

    assert lighting_result.shape == object_rgba.shape
    assert lighting_result.dtype == np.uint8
    assert np.array_equal(
        lighting_result[:, :, 3],
        object_rgba[:, :, 3],
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "lighting.png"),
        cv2.cvtColor(
            lighting_result,
            cv2.COLOR_RGBA2BGRA,
        ),
    )

    print("✓ Lighting")

    # ---------------------------------------------------------
    # 3. Color Temperature
    # ---------------------------------------------------------

    print("[3] Testing Color Temperature...")

    warm_result = color_temperature.apply(
        rgba=object_rgba,
        temperature=20,
    )

    assert warm_result.shape == object_rgba.shape
    assert warm_result.dtype == np.uint8
    assert np.array_equal(
        warm_result[:, :, 3],
        object_rgba[:, :, 3],
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "warm.png"),
        cv2.cvtColor(
            warm_result,
            cv2.COLOR_RGBA2BGRA,
        ),
    )

    print("✓ Color Temperature")

    # ---------------------------------------------------------
    # 4. Reflection
    # ---------------------------------------------------------

    print("[4] Testing Reflection...")

    reflection_result = reflection.apply(
        rgba=object_rgba,
        strength=0.25,
    )

    assert reflection_result.shape == object_rgba.shape
    assert reflection_result.dtype == np.uint8
    assert np.array_equal(
        reflection_result[:, :, 3],
        object_rgba[:, :, 3],
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "reflection.png"),
        cv2.cvtColor(
            reflection_result,
            cv2.COLOR_RGBA2BGRA,
        ),
    )

    print("✓ Reflection")

    # ---------------------------------------------------------
    # 5. Occlusion
    # ---------------------------------------------------------

    print("[5] Testing Occlusion...")

    occlusion_result = occlusion.apply(
        rgba=object_rgba,
        ratio=0.25,
    )

    assert occlusion_result.shape == object_rgba.shape
    assert occlusion_result.dtype == np.uint8

    original_alpha = object_rgba[:, :, 3]
    occluded_alpha = occlusion_result[:, :, 3]

    assert np.count_nonzero(
        occluded_alpha
    ) < np.count_nonzero(
        original_alpha
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "occlusion.png"),
        cv2.cvtColor(
            occlusion_result,
            cv2.COLOR_RGBA2BGRA,
        ),
    )

    print("✓ Occlusion")

    # ---------------------------------------------------------
    # 6. Compositor
    # ---------------------------------------------------------

    print("[6] Testing Compositor...")

    compositor_result = compositor.composite(
        background=background.copy(),
        object_rgba=object_rgba,
        placement=placement,
    )

    assert compositor_result.shape == background.shape
    assert compositor_result.dtype == np.uint8

    cv2.imwrite(
        str(OUTPUT_DIR / "compositor.png"),
        compositor_result,
    )

    print("✓ Compositor")

    # ---------------------------------------------------------
    # 7. Full Context Engine
    # ---------------------------------------------------------

    print("[7] Testing Full Context Engine...")

    context_result = engine.generate(
        background=background,
        object_rgba=object_rgba,
        enable_reflection=True,
        enable_occlusion=True,
    )

    assert context_result.image.shape == background.shape
    assert context_result.image.dtype == np.uint8

    assert context_result.placement.width > 0
    assert context_result.placement.height > 0

    cv2.imwrite(
        str(OUTPUT_DIR / "integration.png"),
        context_result.image,
    )

    print("✓ Full Context Engine")

    # ---------------------------------------------------------
    # 8. Final Object Mask
    # ---------------------------------------------------------

    print("[8] Testing Final Object Mask...")

    object_mask = context_result.object_mask
    placement = context_result.placement

    check_mask(
        mask=object_mask,
        placement=placement,
    )

    # Because occlusion is enabled, the visible mask should
    # contain fewer pixels than the full rectangular placement.
    visible_pixels = np.count_nonzero(
        object_mask == 255
    )

    placement_area = (
        placement.width
        * placement.height
    )

    assert visible_pixels > 0

    assert visible_pixels < placement_area

    cv2.imwrite(
        str(OUTPUT_DIR / "object_mask.png"),
        object_mask,
    )

    print(
        "✓ Object Mask: "
        f"shape={object_mask.shape}, "
        f"visible_pixels={visible_pixels}"
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("Results saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()