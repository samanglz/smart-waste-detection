from pathlib import Path


class E3HardSamplingConfig:
    """
    Static configuration for E3 - Hard Example Sampling.

    E3 changes only the training sampling strategy.

    Everything else is inherited from E2.
    """

    # ============================================================
    # MODEL
    # ============================================================

    # E3 starts from the E2 best checkpoint.
    MODEL_PATH = Path(
        r"runs\detect\runs\E2_targeted_augmentation\targeted_aug\weights\best.pt"
    )

    # ============================================================
    # DATASET
    # ============================================================

    DATASET_ROOT = Path(
        r"data\processed"
    )

    DATASET_YAML = DATASET_ROOT / "data.yaml"

    # ============================================================
    # HARD EXAMPLE SAMPLING
    # ============================================================

    SAMPLING_WEIGHTS_PATH = Path(
        r"outputs\hard_example_mining\E2\sampling_weights.json"
    )

    # ============================================================
    # TRAINING
    # Same as E2
    # ============================================================

    EPOCHS = 30

    BATCH_SIZE = 2

    IMAGE_SIZE = 640

    LR0 = 0.0005

    OPTIMIZER = "AdamW"

    COS_LR = True

    PATIENCE = 30

    # ============================================================
    # ONLINE AUGMENTATION
    # Same as E2:
    # disabled.
    #
    # E3 is supposed to change sampling,
    # NOT introduce additional augmentation.
    # ============================================================

    ONLINE_AUGMENTATION = {
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

    # ============================================================
    # OUTPUT
    # ============================================================

    PROJECT_NAME = (
        "runs/E3_hard_example_sampling"
    )

    RUN_NAME = "hard_sampling"

    SAVE_PERIOD = 10

    # ============================================================
    # HARD SAMPLING
    # ============================================================

    SAMPLING_REPLACEMENT = True

    SAMPLING_SEED = 6148914691236517205

    # ============================================================
    # DEVICE
    # ============================================================

    DEVICE = 0

    WORKERS = 0

    # ============================================================
    # EXPERIMENT METADATA
    # ============================================================

    EXPERIMENT_NAME = "E3"

    EXPERIMENT_DESCRIPTION = (
        "Hard Example Sampling using image-level "
        "sampling weights generated from E2 hard-example mining."
    )