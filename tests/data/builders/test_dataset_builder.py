"""
Smoke test for the generic DatasetBuilder.
"""

from pathlib import Path
import json

from src.data.yolo_dataset import YOLODataset
from src.data.builders.dataset_builder_for_e7 import DatasetBuilder

from src.e7.context_engine import (
    ColorTemperatureEngine,
    Compositor,
    ContextEngine,
    LightingEngine,
    OcclusionEngine,
    RandomScaleStrategy,
    ReflectionEngine,
)

from src.e7.generation_strategy import (
    AreaBasedGenerationStrategy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed"
)

OBJECT_BANK_DIR = (
    PROJECT_ROOT / "data" / "E7_object_bank"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "tests" / "trash"/ "E7_dataset_test"
)


def build_context_engine() -> ContextEngine:
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
        seed=42,
    )


def main():
    print("=" * 70)
    print("E7 DATASET BUILDER SMOKE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = YOLODataset(
        dataset_root=DATASET_ROOT,
    )

    dataset.load()

    print(
        f"Train images available: "
        f"{len(dataset.get_train_data())}"
    )

    # ---------------------------------------------------------
    # Generation Strategy
    # ---------------------------------------------------------

    generation_strategy = AreaBasedGenerationStrategy(
        small_threshold=0.05,
        medium_threshold=0.20,
    )

    # ---------------------------------------------------------
    # Context Engine
    # ---------------------------------------------------------

    context_engine = build_context_engine()

    # ---------------------------------------------------------
    # Builder
    # ---------------------------------------------------------

    builder = DatasetBuilder(
        dataset=dataset,
        object_bank_dir=OBJECT_BANK_DIR,
        output_dir=OUTPUT_DIR,
        generation_strategy=generation_strategy,
        context_engine=context_engine,
        seed=42,
    )

    # ---------------------------------------------------------
    # Smoke Build
    # ---------------------------------------------------------

    print()
    print("[1] Building dataset from 5 images...")

    summary = builder.build(
        max_images=5,
        copy_original=True,
    )

    print("✓ Dataset build completed")

    # ---------------------------------------------------------
    # Summary validation
    # ---------------------------------------------------------

    print("[2] Validating build summary...")

    assert summary["source_images"] == 5
    assert summary["original_images"] == 5
    assert summary["synthetic_images"] > 0
    assert summary["objects_processed"] > 0
    assert summary["failures"] == 0

    print("✓ Build summary valid")

    # ---------------------------------------------------------
    # Output validation
    # ---------------------------------------------------------

    print("[3] Validating output files...")

    images_dir = (
        OUTPUT_DIR / "train" / "images"
    )

    labels_dir = (
        OUTPUT_DIR / "train" / "labels"
    )

    assert images_dir.exists()
    assert labels_dir.exists()

    image_files = list(
        images_dir.glob("*")
    )

    label_files = list(
        labels_dir.glob("*.txt")
    )

    assert len(image_files) > 0
    assert len(label_files) > 0

    print(
        f"✓ Images: {len(image_files)}"
    )

    print(
        f"✓ Labels: {len(label_files)}"
    )

    # ---------------------------------------------------------
    # Image-label matching
    # ---------------------------------------------------------

    print("[4] Validating image-label pairs...")

    image_stems = {
        path.stem
        for path in image_files
    }

    label_stems = {
        path.stem
        for path in label_files
    }

    assert image_stems == label_stems

    print("✓ Image/label pairs match")

    # ---------------------------------------------------------
    # YOLO label validation
    # ---------------------------------------------------------

    print("[5] Validating YOLO labels...")

    for label_path in label_files:

        lines = label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        assert lines

        for line in lines:

            values = line.split()

            assert len(values) == 5

            class_id = int(values[0])

            cx = float(values[1])
            cy = float(values[2])
            width = float(values[3])
            height = float(values[4])

            assert class_id >= 0

            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0

            assert 0.0 < width <= 1.0
            assert 0.0 < height <= 1.0

    print("✓ YOLO labels valid")

    # ---------------------------------------------------------
    # Build summary file
    # ---------------------------------------------------------

    print("[6] Validating build_summary.json...")

    summary_path = (
        OUTPUT_DIR / "build_summary.json"
    )

    assert summary_path.exists()

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_summary = json.load(file)

    assert saved_summary == summary

    print("✓ Build summary file valid")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("E7 DATASET BUILDER SMOKE TEST PASSED")
    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()