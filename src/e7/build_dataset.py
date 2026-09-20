"""
E7 dataset build runner.

This module orchestrates the E7 dataset generation pipeline.

E7 training source:
    data/e2_targeted_test/train

E7 evaluation source:
    data/processed/val
    data/processed/test

The validation and test datasets are never modified.
"""

from pathlib import Path

from src.e7.context_engine.scale import RandomScaleStrategy
from src.e7.context_engine.lighting import LightingEngine
from src.e7.context_engine.color_temperature import ColorTemperatureEngine
from src.e7.context_engine.reflection import ReflectionEngine
from src.e7.context_engine.occlusion import OcclusionEngine
from src.e7.context_engine.compositor import Compositor

from src.data.builders.dataset_builder_for_e7 import DatasetBuilder
from src.e7.config import (
    SOURCE_TRAIN_DIR,
    EVALUATION_DATASET_DIR,
    E7_DATASET_DIR,
    OBJECT_BANK_DIR,
    OBJECT_BANK_MANIFEST,
    COPY_ORIGINAL_TRAIN,
    MAX_IMAGES,
    SEED,
    SMALL_AREA_THRESHOLD,
    MEDIUM_AREA_THRESHOLD,
    MIN_OBJECT_HEIGHT_RATIO,
    MAX_OBJECT_HEIGHT_RATIO,
    ENABLE_REFLECTION,
    ENABLE_OCCLUSION,
)

from src.e7.context_engine.context_engine import ContextEngine
from src.e7.generation_strategy import (
    AreaBasedGenerationStrategy,
)
from src.e7.object_sampling import (
    BalancedObjectSampler,
)
from src.e7.dataset_source import E7DatasetSource


def build_dataset():
    """
    Build the E7 training dataset.

    E2 train is used as the source/background dataset.
    Original validation and test remain untouched.
    """

    print("=" * 70)
    print("E7 REAL-WORLD DOMAIN ADAPTATION")
    print("=" * 70)

    print(f"E2 train source : {SOURCE_TRAIN_DIR}")
    print(f"Evaluation data : {EVALUATION_DATASET_DIR}")
    print(f"Object Bank     : {OBJECT_BANK_DIR}")
    print(f"Output dataset  : {E7_DATASET_DIR}")
    print(f"Seed            : {SEED}")
    print(f"Max images      : {MAX_IMAGES}")
    print()

    # ------------------------------------------------------------------
    # Dataset source
    # ------------------------------------------------------------------

    dataset_source = E7DatasetSource(
        train_dataset_root=SOURCE_TRAIN_DIR,
        evaluation_dataset_root=EVALUATION_DATASET_DIR,
    )

    # ------------------------------------------------------------------
    # Context Engine
    # ------------------------------------------------------------------

    scale_strategy = RandomScaleStrategy(
        min_relative_height=MIN_OBJECT_HEIGHT_RATIO,
        max_relative_height=MAX_OBJECT_HEIGHT_RATIO,
    )

    lighting = LightingEngine()

    color_temperature = ColorTemperatureEngine()

    reflection = ReflectionEngine()

    occlusion = OcclusionEngine()

    compositor = Compositor()

    context_engine = ContextEngine(
        scale_strategy=scale_strategy,
        lighting=lighting,
        color_temperature=color_temperature,
        reflection=reflection,
        occlusion=occlusion,
        compositor=compositor,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Generation strategy
    # ------------------------------------------------------------------

    generation_strategy = AreaBasedGenerationStrategy(
        small_threshold=SMALL_AREA_THRESHOLD,
        medium_threshold=MEDIUM_AREA_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Dataset Builder
    # ------------------------------------------------------------------
    
    builder = DatasetBuilder(
        dataset=dataset_source,
        object_bank_dir=OBJECT_BANK_DIR,
        output_dir=E7_DATASET_DIR,
        generation_strategy=generation_strategy,
        context_engine=context_engine,
        seed=SEED,
    )

    # Explicitly use the balanced object sampler.
    builder.object_sampler = BalancedObjectSampler(
        seed=SEED
    )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    summary = builder.build(
        max_images=MAX_IMAGES,
        copy_original=COPY_ORIGINAL_TRAIN,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("E7 BUILD SUMMARY")
    print("=" * 70)

    print(
        f"Source images     : "
        f"{summary['source_images']}"
    )

    print(
        f"Original images   : "
        f"{summary['original_images']}"
    )

    print(
        f"Synthetic images  : "
        f"{summary['synthetic_images']}"
    )

    print(
        f"Objects processed : "
        f"{summary['objects_processed']}"
    )

    print(
        f"Failures          : "
        f"{summary['failures']}"
    )

    print("=" * 70)

    return summary


def main():
    """Run the E7 dataset build."""

    build_dataset()


if __name__ == "__main__":
    main()