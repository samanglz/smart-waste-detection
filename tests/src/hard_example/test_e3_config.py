from pathlib import Path

from configs.e3_hard_sampling_config import (
    E3HardSamplingConfig,
)


def main():

    config = E3HardSamplingConfig

    print("=" * 70)
    print("E3 CONFIG VALIDATION")
    print("=" * 70)

    # ============================================================
    # Required paths
    # ============================================================

    print("\nPATHS")
    print("-" * 70)

    required_paths = {
        "E2 best.pt": config.MODEL_PATH,
        "Dataset root": config.DATASET_ROOT,
        "Dataset YAML": config.DATASET_YAML,
        "Sampling weights": config.SAMPLING_WEIGHTS_PATH,
    }

    for name, path in required_paths.items():

        path = Path(path)

        print(f"{name:<20}: {path}")

        assert path.exists(), (
            f"{name} does not exist:\n{path}"
        )

        print(" " * 22 + "✓ exists")

    # ============================================================
    # Training configuration
    # ============================================================

    print("\nTRAINING")
    print("-" * 70)

    assert config.EPOCHS == 30
    print("EPOCHS             : 30 ✓")

    assert config.BATCH_SIZE == 2
    print("BATCH_SIZE         : 2 ✓")

    assert config.IMAGE_SIZE == 640
    print("IMAGE_SIZE         : 640 ✓")

    assert config.LR0 == 0.0005
    print("LR0                : 0.0005 ✓")

    assert config.OPTIMIZER == "AdamW"
    print("OPTIMIZER          : AdamW ✓")

    assert config.COS_LR is True
    print("COS_LR             : True ✓")

    assert config.PATIENCE == 30
    print("PATIENCE           : 30 ✓")

    # ============================================================
    # Online augmentation
    # ============================================================

    print("\nONLINE AUGMENTATION")
    print("-" * 70)

    expected_augmentation = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
    }

    assert (
        config.ONLINE_AUGMENTATION
        == expected_augmentation
    )

    print("All online augmentations disabled ✓")

    # ============================================================
    # Hard sampling
    # ============================================================

    print("\nHARD SAMPLING")
    print("-" * 70)

    assert config.SAMPLING_REPLACEMENT is True
    print("Replacement        : True ✓")

    assert (
        config.SAMPLING_SEED
        == 6148914691236517205
    )

    print(
        "Sampling seed      : "
        "6148914691236517205 ✓"
    )

    # ============================================================
    # Hardware
    # ============================================================

    print("\nHARDWARE")
    print("-" * 70)

    assert config.DEVICE == 0
    print("DEVICE             : 0 ✓")

    assert config.WORKERS == 0
    print("WORKERS            : 0 ✓")

    # ============================================================
    # Output
    # ============================================================

    print("\nOUTPUT")
    print("-" * 70)

    print(
        f"PROJECT_NAME       : {config.PROJECT_NAME}"
    )

    print(
        f"RUN_NAME           : {config.RUN_NAME}"
    )

    assert config.EXPERIMENT_NAME == "E3"
    print("EXPERIMENT_NAME    : E3 ✓")

    # ============================================================
    # Final
    # ============================================================

    print()
    print("=" * 70)
    print("✅ E3 CONFIG VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()