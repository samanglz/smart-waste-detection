from pathlib import Path

from src.e7.config import (
    SOURCE_TRAIN_DIR,
    EVALUATION_DATASET_DIR,
    E7_DATASET_DIR,
    OBJECT_BANK_DIR,
    MIN_OBJECT_HEIGHT_RATIO,
    MAX_OBJECT_HEIGHT_RATIO,
    ENABLE_REFLECTION,
    ENABLE_OCCLUSION,
    SEED,
    SMALL_AREA_THRESHOLD,
    MEDIUM_AREA_THRESHOLD,
)

from src.e7.dataset_source import E7DatasetSource
from src.e7.context_engine.context_engine import ContextEngine
from src.e7.generation_strategy import AreaBasedGenerationStrategy

from src.e7.context_engine.scale import RandomScaleStrategy
from src.e7.context_engine.lighting import LightingEngine
from src.e7.context_engine.color_temperature import ColorTemperatureEngine
from src.e7.context_engine.reflection import ReflectionEngine
from src.e7.context_engine.occlusion import OcclusionEngine
from src.e7.context_engine.compositor import Compositor

def main():
    print("=" * 70)
    print("E7 PREFLIGHT CHECK")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    print("\n[1] PATHS")

    paths = {
        "E2 train": SOURCE_TRAIN_DIR,
        "Evaluation": EVALUATION_DATASET_DIR,
        "Object Bank": OBJECT_BANK_DIR,
    }

    for name, path in paths.items():
        status = "OK" if path.exists() else "FAIL"
        print(f"{name:<15}: {status} -> {path}")

        if not path.exists():
            raise FileNotFoundError(path)

    # ---------------------------------------------------------------
    # Dataset Source
    # ---------------------------------------------------------------

    print("\n[2] DATASET SOURCE")

    dataset = E7DatasetSource(
        train_dataset_root=SOURCE_TRAIN_DIR,
        evaluation_dataset_root=EVALUATION_DATASET_DIR,
    )

    train_count = len(dataset.get_train_data())
    val_count = len(dataset.get_val_data())
    test_count = len(dataset.get_test_data())

    print(f"E2 train : {train_count}")
    print(f"Val      : {val_count}")
    print(f"Test     : {test_count}")

    assert train_count == 2367
    assert val_count == 358
    assert test_count == 359

    print("Dataset source: OK")

    # ---------------------------------------------------------------
    # Classes
    # ---------------------------------------------------------------

    print("\n[3] CLASSES")

    class_names = dataset.get_class_names()

    print(f"Classes: {class_names}")
    print(f"Number : {dataset.get_num_classes()}")

    assert dataset.get_num_classes() == 5

    expected_classes = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    ]

    assert class_names == expected_classes

    print("Classes: OK")

    # ---------------------------------------------------------------
    # Object Bank
    # ---------------------------------------------------------------

    print("\n[4] OBJECT BANK")

    manifest_path = OBJECT_BANK_DIR / "manifest.json"

    assert manifest_path.exists()

    import json

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    object_count = len(manifest["objects"])

    print(f"Objects: {object_count}")

    assert object_count == 1677

    print("Object Bank: OK")

    # ---------------------------------------------------------------
    # Context Engine
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Context Engine
    # ---------------------------------------------------------------

    print("\n[5] CONTEXT ENGINE")

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

    print(
        f"Min object height: "
        f"{MIN_OBJECT_HEIGHT_RATIO}"
    )

    print(
        f"Max object height: "
        f"{MAX_OBJECT_HEIGHT_RATIO}"
    )

    print(
        f"Reflection       : "
        f"{ENABLE_REFLECTION}"
    )

    print(
        f"Occlusion        : "
        f"{ENABLE_OCCLUSION}"
    )

    assert context_engine is not None

    print("Context Engine: OK")

    # ---------------------------------------------------------------
    # Generation Strategy
    # ---------------------------------------------------------------

    print("\n[6] GENERATION STRATEGY")

    strategy = AreaBasedGenerationStrategy(
        small_threshold=SMALL_AREA_THRESHOLD,
        medium_threshold=MEDIUM_AREA_THRESHOLD,
    )

    print(f"Small threshold : {SMALL_AREA_THRESHOLD}")
    print(f"Medium threshold: {MEDIUM_AREA_THRESHOLD}")

    assert strategy is not None

    print("Generation Strategy: OK")

    # ---------------------------------------------------------------
    # Output Safety
    # ---------------------------------------------------------------

    print("\n[7] OUTPUT SAFETY")

    print(f"E7 output: {E7_DATASET_DIR}")

    if E7_DATASET_DIR.exists():
        print(
            "WARNING: E7 output directory already exists."
        )
        print(
            "No files will be deleted or modified by this test."
        )
    else:
        print("E7 output directory does not exist yet.")

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("E7 PREFLIGHT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()