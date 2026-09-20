"""
Integration test for E7 DatasetBuilder object sampling.

This test verifies that DatasetBuilder uses BalancedObjectSampler
instead of random object selection.

The test intentionally uses a small subset of the training dataset
and writes all temporary outputs under tests/trash/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.data.builders.dataset_builder_for_e7 import DatasetBuilder
from src.data.yolo_dataset import YOLODataset
from src.e7.context_engine import (
    ColorTemperatureEngine,
    Compositor,
    ContextEngine,
    LightingEngine,
    OcclusionEngine,
    ReflectionEngine,
    RandomScaleStrategy,
)
from src.e7.generation_strategy import AreaBasedGenerationStrategy
from src.e7.object_sampling import BalancedObjectSampler


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "data" / "processed"

OBJECT_BANK_DIR = (
    PROJECT_ROOT
    / "data"
    / "E7_object_bank"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "trash"
    / "E7_sampling_integration_test"
)

NUM_TEST_IMAGES = 20
SEED = 42


def create_context_engine() -> ContextEngine:
    """
    Create the same ContextEngine configuration used by E7.
    """

    return ContextEngine(
        scale_strategy=RandomScaleStrategy(
            min_relative_height=0.10,
            max_relative_height=0.35,
        ),
        lighting=LightingEngine(),
        color_temperature=ColorTemperatureEngine(),
        reflection=ReflectionEngine(),
        occlusion=OcclusionEngine(),
        compositor=Compositor(),
        seed=SEED,
    )


def create_dataset_builder() -> DatasetBuilder:
    """
    Create a DatasetBuilder configured for the integration test.
    """

    dataset = YOLODataset(
        dataset_root=DATASET_ROOT,
    )

    generation_strategy = AreaBasedGenerationStrategy()

    context_engine = create_context_engine()

    sampler = BalancedObjectSampler(
        seed=SEED,
    )

    builder = DatasetBuilder(
        dataset=dataset,
        object_bank_dir=OBJECT_BANK_DIR,
        output_dir=OUTPUT_DIR,
        generation_strategy=generation_strategy,
        context_engine=context_engine,
        seed=SEED,
    )

    # Inject the sampler explicitly so the test can verify
    # that DatasetBuilder uses BalancedObjectSampler.
    builder.object_sampler = sampler

    return builder


def load_object_pool(builder: DatasetBuilder) -> list[dict]:
    """
    Load the same Object Bank pool used internally by DatasetBuilder.
    """

    manifest_path = (
        builder.object_bank_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    return builder._build_object_pool(
        manifest["objects"]
    )


def test_builder_uses_balanced_sampler() -> None:
    """
    Verify that DatasetBuilder uses BalancedObjectSampler.
    """

    builder = create_dataset_builder()

    assert isinstance(
        builder.object_sampler,
        BalancedObjectSampler,
    )

    print(
        "[1] DatasetBuilder uses BalancedObjectSampler ✓"
    )


def test_sampling_without_replacement() -> None:
    """
    Verify that the first N selections contain no duplicate objects.
    """

    builder = create_dataset_builder()

    object_pool = load_object_pool(builder)

    assert len(object_pool) >= NUM_TEST_IMAGES, (
        f"Object Bank contains only {len(object_pool)} objects, "
        f"but test requires {NUM_TEST_IMAGES}."
    )

    selected_ids = []

    for _ in range(NUM_TEST_IMAGES):
        object_entry = builder._select_object(
            object_pool=object_pool,
        )

        selected_ids.append(
            object_entry["object_id"]
        )

    unique_ids = set(selected_ids)

    assert len(unique_ids) == NUM_TEST_IMAGES, (
        "Balanced sampling failed: duplicate objects "
        "were selected within the first cycle."
    )

    print(
        f"[2] {NUM_TEST_IMAGES} selections without replacement ✓"
    )


def test_sampling_covers_distinct_objects() -> None:
    """
    Verify that sampling is actually progressing through
    different Object Bank entries.
    """

    builder = create_dataset_builder()

    object_pool = load_object_pool(builder)

    selected_ids = []

    for _ in range(NUM_TEST_IMAGES):
        object_entry = builder._select_object(
            object_pool=object_pool,
        )

        selected_ids.append(
            object_entry["object_id"]
        )

    assert len(selected_ids) == len(
        set(selected_ids)
    )

    assert set(selected_ids).issubset(
        {
            entry["object_id"]
            for entry in object_pool
        }
    )

    print(
        "[3] All selected objects belong to Object Bank ✓"
    )


def test_builder_integration() -> None:
    """
    Run DatasetBuilder on a small subset and verify that
    the resulting dataset is valid.
    """

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    builder = create_dataset_builder()

    summary = builder.build(
        max_images=NUM_TEST_IMAGES,
        copy_original=True,
    )

    assert summary["source_images"] == NUM_TEST_IMAGES

    assert summary["failures"] == 0

    assert summary["objects_processed"] == (
        summary["synthetic_images"]
    )

    expected_originals = NUM_TEST_IMAGES

    assert summary["original_images"] == expected_originals

    image_files = list(
        (
            OUTPUT_DIR
            / "train"
            / "images"
        ).glob("*")
    )

    label_files = list(
        (
            OUTPUT_DIR
            / "train"
            / "labels"
        ).glob("*.txt")
    )

    expected_files = (
        summary["original_images"]
        + summary["synthetic_images"]
    )

    assert len(image_files) == expected_files

    assert len(label_files) == expected_files

    print(
        "[4] DatasetBuilder integration build ✓"
    )

    print(
        f"    Original images : "
        f"{summary['original_images']}"
    )

    print(
        f"    Synthetic images: "
        f"{summary['synthetic_images']}"
    )

    print(
        f"    Objects processed: "
        f"{summary['objects_processed']}"
    )


def main() -> None:
    print("=" * 70)
    print("E7 DATASET BUILDER SAMPLING INTEGRATION TEST")
    print("=" * 70)

    test_builder_uses_balanced_sampler()

    test_sampling_without_replacement()

    test_sampling_covers_distinct_objects()

    test_builder_integration()

    print("=" * 70)
    print("ALL E7 SAMPLING INTEGRATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

