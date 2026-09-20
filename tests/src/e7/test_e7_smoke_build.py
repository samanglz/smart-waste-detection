from pathlib import Path

from src.e7.config import (
    SOURCE_TRAIN_DIR,
    EVALUATION_DATASET_DIR,
    OBJECT_BANK_DIR,
    SEED,
    SMALL_AREA_THRESHOLD,
    MEDIUM_AREA_THRESHOLD,
    MIN_OBJECT_HEIGHT_RATIO,
    MAX_OBJECT_HEIGHT_RATIO,
    ENABLE_REFLECTION,
    ENABLE_OCCLUSION,
)

from src.e7.dataset_source import E7DatasetSource
from src.e7.context_engine.context_engine import ContextEngine
from src.e7.context_engine.scale import RandomScaleStrategy
from src.e7.context_engine.lighting import LightingEngine
from src.e7.context_engine.color_temperature import ColorTemperatureEngine
from src.e7.context_engine.reflection import ReflectionEngine
from src.e7.context_engine.occlusion import OcclusionEngine
from src.e7.context_engine.compositor import Compositor
from src.e7.generation_strategy import AreaBasedGenerationStrategy
from src.data.builders.dataset_builder_for_e7 import DatasetBuilder


def main():
    output_dir = Path("tests/trash/E7_smoke_build")

    print("=" * 70)
    print("E7 SMOKE BUILD")
    print("=" * 70)

    dataset_source = E7DatasetSource(
        train_dataset_root=SOURCE_TRAIN_DIR,
        evaluation_dataset_root=EVALUATION_DATASET_DIR,
    )

    context_engine = ContextEngine(
        scale_strategy=RandomScaleStrategy(
            min_relative_height=MIN_OBJECT_HEIGHT_RATIO,
            max_relative_height=MAX_OBJECT_HEIGHT_RATIO,
        ),
        lighting=LightingEngine(),
        color_temperature=ColorTemperatureEngine(),
        reflection=ReflectionEngine(),
        occlusion=OcclusionEngine(),
        compositor=Compositor(),
        seed=SEED,
    )

    generation_strategy = AreaBasedGenerationStrategy(
        small_threshold=SMALL_AREA_THRESHOLD,
        medium_threshold=MEDIUM_AREA_THRESHOLD,
    )

    builder = DatasetBuilder(
        dataset=dataset_source,
        object_bank_dir=OBJECT_BANK_DIR,
        output_dir=output_dir,
        generation_strategy=generation_strategy,
        context_engine=context_engine,
        seed=SEED,
    )

    summary = builder.build(
        max_images=5,
        copy_original=True,
    )

    print()
    print("=" * 70)
    print("SMOKE BUILD SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key:<20}: {value}")

    print("=" * 70)


if __name__ == "__main__":
    main()