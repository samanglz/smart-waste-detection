
from configs.e3_hard_sampling_config import E3HardSamplingConfig
from src.data.yolo_dataset import YOLODataset
from src.trainers.E3_hard_example_sampling_trainer import (
    YOLOHardSamplerTrainer,
)


def main():

    config = E3HardSamplingConfig

    print("=" * 70)
    print("E3 HARD EXAMPLE SAMPLING TRAINING")
    print("=" * 70)

    # ============================================================
    # 1. BUILD DATASET
    # ============================================================

    print("\n[1] BUILD DATASET")
    print("-" * 70)

    dataset = YOLODataset(
        dataset_root=config.DATASET_ROOT,
    )

    print(
        f"Dataset root : {config.DATASET_ROOT}"
    )

    print(
        f"Dataset YAML : {dataset.yaml_path}"
    )

    print(
        f"Train images : "
        f"{len(dataset.get_train_data())}"
    )

    print(
        f"Val images   : "
        f"{len(dataset.get_val_data())}"
    )

    print(
        f"Test images  : "
        f"{len(dataset.get_test_data())}"
    )

    print("✓ Dataset ready")

    # ============================================================
    # 2. BUILD E3 TRAINER
    # ============================================================

    print("\n[2] BUILD E3 TRAINER")
    print("-" * 70)

    trainer = YOLOHardSamplerTrainer(
        config=config,
        dataset=dataset,
    )

    print(
        "Trainer class : "
        f"{type(trainer).__name__}"
    )

    # ============================================================
    # 3. BUILD STARTING MODEL
    # ============================================================

    print("\n[3] BUILD STARTING MODEL")
    print("-" * 70)

    trainer.build_model()

    if trainer.model is None:
        raise RuntimeError(
            "Failed to load E2 best.pt."
        )

    print(
        f"Starting model : "
        f"{config.MODEL_PATH}"
    )

    print("✓ E2 best.pt loaded")

    # ============================================================
    # 4. START E3 TRAINING
    # ============================================================

    print("\n[4] START E3 TRAINING")
    print("-" * 70)

    print(
        "Experiment       : E3"
    )

    print(
        "Method           : Hard Example Sampling"
    )

    print(
        f"Epochs           : {config.EPOCHS}"
    )

    print(
        f"Batch size       : {config.BATCH_SIZE}"
    )

    print(
        f"Image size       : {config.IMAGE_SIZE}"
    )

    print(
        f"Learning rate    : {config.LR0}"
    )

    print(
        f"Optimizer        : {config.OPTIMIZER}"
    )

    print(
        f"Device           : {config.DEVICE}"
    )

    print(
        f"Sampling weights : "
        f"{config.SAMPLING_WEIGHTS_PATH}"
    )
    
    print(
        "\nOnline augmentation: DISABLED"
    )

    print(
        "\nStarting training..."
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # YOLOHardSamplerTrainer.train() is responsible for:
    #
    #   - building HardSamplingDetectionTrainer
    #   - loading sampling weights
    #   - enabling EpochWeightedRandomSampler
    #   - starting Ultralytics training
    #   - locating best.pt
    #   - loading the trained checkpoint
    #
    # Do NOT duplicate that logic here.
    # ------------------------------------------------------------

    trainer.train()

    # ============================================================
    # 5. VERIFY RESULT
    # ============================================================

    print("\n[5] VERIFY E3 RESULT")
    print("-" * 70)

    if trainer.model is None:
        raise RuntimeError(
            "E3 training finished but the trained "
            "model was not loaded."
        )

    print(
        "Final model : "
        f"{trainer.model}"
    )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 70)
    print("✅ E3 HARD EXAMPLE SAMPLING COMPLETED")
    print("=" * 70)

    print()
    print(
        "E3 training finished successfully."
    )

    print(
        "The best checkpoint has been loaded "
        "into YOLOHardSamplerTrainer."
    )


if __name__ == "__main__":
    main()

