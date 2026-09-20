from pathlib import Path

from ultralytics import YOLO

from configs.e3_hard_sampling_config import (
    E3HardSamplingConfig,
)

from src.data.yolo_dataset import YOLODataset

from src.trainers.E3_hard_example_sampling_trainer import (
    YOLOHardSamplerTrainer,
    HardSamplingDetectionTrainer,
)


def main():

    config = E3HardSamplingConfig

    print("=" * 70)
    print("E3 TRAINER SETUP TEST")
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

    print("✓ YOLODataset initialized")

    # ============================================================
    # 2. BUILD E3 PROJECT TRAINER
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

    assert isinstance(
        trainer,
        YOLOHardSamplerTrainer,
    )

    print("✓ YOLOHardSamplerTrainer initialized")

    # ============================================================
    # 3. BUILD MODEL
    # ============================================================

    print("\n[3] BUILD MODEL")
    print("-" * 70)

    trainer.build_model()

    assert trainer.model is not None

    print(
        f"Model path : {config.MODEL_PATH}"
    )

    print("✓ Model loaded")

    # ============================================================
    # 4. BUILD ULTRALYTICS TRAINER
    # ============================================================

    print("\n[4] BUILD ULTRALYTICS TRAINER")
    print("-" * 70)

    sampling_weights = Path(
        config.SAMPLING_WEIGHTS_PATH
    ).resolve()

    assert sampling_weights.exists(), (
        "Sampling weights file does not exist:\n"
        f"{sampling_weights}"
    )

    ultralytics_trainer = HardSamplingDetectionTrainer(
        overrides={

            "model": str(
                config.MODEL_PATH
            ),

            "data": str(
                dataset.yaml_path
            ),

            "epochs": config.EPOCHS,
            "imgsz": config.IMAGE_SIZE,
            "batch": config.BATCH_SIZE,

            "lr0": config.LR0,
            "optimizer": config.OPTIMIZER,
            "cos_lr": config.COS_LR,
            "patience": config.PATIENCE,

            "project": config.PROJECT_NAME,
            "name": config.RUN_NAME,
            "save_period": config.SAVE_PERIOD,

            "device": config.DEVICE,
            "workers": config.WORKERS,

            **config.ONLINE_AUGMENTATION,
        },

        sampling_weights_path=sampling_weights,
    )

    assert isinstance(
        ultralytics_trainer,
        HardSamplingDetectionTrainer,
    )

    print(
        "Ultralytics trainer : "
        f"{type(ultralytics_trainer).__name__}"
    )

    print("✓ Ultralytics trainer initialized")

    # ============================================================
    # IMPORTANT MODEL FIX
    #
    # Ultralytics get_dataloader() needs a real model because
    # build_dataset() accesses model.stride.
    #
    # The override contains the model path, but depending on the
    # Ultralytics version, the trainer model can still be a string.
    #
    # Therefore explicitly load the YOLO model here.
    # ============================================================

    print(
        "\nLoading real Ultralytics model "
        "for DataLoader construction..."
    )

    ultralytics_model = YOLO(
        str(config.MODEL_PATH)
    )

    ultralytics_trainer.model = (
        ultralytics_model.model
    )

    assert ultralytics_trainer.model is not None

    print(
        "Model object : "
        f"{type(ultralytics_trainer.model).__name__}"
    )

    assert hasattr(
        ultralytics_trainer.model,
        "stride",
    )

    print(
        f"Model stride : "
        f"{ultralytics_trainer.model.stride} ✓"
    )

    print(
        "✓ Real Ultralytics model attached"
    )

    # ============================================================
    # 5. VERIFY TRAINING ARGUMENTS
    # ============================================================

    print("\n[5] VERIFY TRAINING ARGUMENTS")
    print("-" * 70)

    args = ultralytics_trainer.args

    assert args.epochs == config.EPOCHS

    print(
        f"epochs    : {args.epochs} ✓"
    )

    assert args.imgsz == config.IMAGE_SIZE

    print(
        f"imgsz     : {args.imgsz} ✓"
    )

    assert args.batch == config.BATCH_SIZE

    print(
        f"batch     : {args.batch} ✓"
    )

    assert args.lr0 == config.LR0

    print(
        f"lr0       : {args.lr0} ✓"
    )

    assert args.optimizer == config.OPTIMIZER

    print(
        f"optimizer : {args.optimizer} ✓"
    )

    assert args.cos_lr == config.COS_LR

    print(
        f"cos_lr    : {args.cos_lr} ✓"
    )

    assert args.patience == config.PATIENCE

    print(
        f"patience  : {args.patience} ✓"
    )

    assert str(args.device) == str(config.DEVICE), (
        f"device mismatch: "
        f"expected {config.DEVICE!r}, "
        f"got {args.device!r}"
    )

    print(
        f"device    : {args.device} ✓"
    )
    
    assert args.workers == config.WORKERS

    print(
        f"workers   : {args.workers} ✓"
    )

    # ============================================================
    # 6. VERIFY ONLINE AUGMENTATION
    # ============================================================

    print("\n[6] VERIFY ONLINE AUGMENTATION")
    print("-" * 70)

    for name, value in (
        config.ONLINE_AUGMENTATION.items()
    ):

        actual = getattr(
            args,
            name,
        )

        assert actual == value, (
            f"{name}: expected {value}, "
            f"got {actual}"
        )

        print(
            f"{name:<12}: {actual} ✓"
        )

    print(
        "✓ All online augmentations disabled"
    )

    # ============================================================
    # 7. BUILD TRAIN DATALOADER
    # ============================================================

    print("\n[7] BUILD TRAIN DATALOADER")
    print("-" * 70)

    train_loader = (
        ultralytics_trainer.get_dataloader(
            dataset_path=str(
                dataset.train_img_dir
            ),
            batch_size=config.BATCH_SIZE,
            rank=-1,
            mode="train",
        )
    )

    assert train_loader is not None

    print(
        "Loader class : "
        f"{type(train_loader).__name__}"
    )

    print(
        "Sampler class: "
        f"{type(train_loader.sampler).__name__}"
    )

    print(
        f"Dataset size : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Batch size   : "
        f"{train_loader.batch_size}"
    )

    # ============================================================
    # 8. VERIFY HARD SAMPLER
    # ============================================================

    print("\n[8] VERIFY HARD SAMPLING")
    print("-" * 70)

    sampler_name = (
        type(train_loader.sampler).__name__
    )

    assert (
        sampler_name
        == "EpochWeightedRandomSampler"
    ), (
        "Expected EpochWeightedRandomSampler, "
        f"got {sampler_name}"
    )

    print(
        "Sampler : "
        "EpochWeightedRandomSampler ✓"
    )

    # ============================================================
    # 9. VERIFY SAMPLING WEIGHTS
    # ============================================================

    print("\n[9] VERIFY SAMPLING WEIGHTS")
    print("-" * 70)

    weights = (
        ultralytics_trainer.sampling_weights
    )

    print(
        f"Loaded sampling weights : "
        f"{len(weights)}"
    )

    assert len(weights) > 0

    print(
        "✓ Sampling weights loaded"
    )

    # ------------------------------------------------------------
    # Expected dataset size
    # ------------------------------------------------------------

    dataset_size = len(
        train_loader.dataset
    )

    assert dataset_size == 1673, (
        "Unexpected training dataset size: "
        f"{dataset_size}"
    )

    print(
        f"Training dataset size : "
        f"{dataset_size} ✓"
    )

    # ------------------------------------------------------------
    # Weight statistics
    # ------------------------------------------------------------

    hard_weights = [
        weight
        for weight in weights.values()
        if weight > 1.0
    ]

    normal_weights = [
        weight
        for weight in weights.values()
        if weight == 1.0
    ]

    print(
        f"Hard weights  : "
        f"{len(hard_weights)}"
    )

    print(
        f"Normal weights: "
        f"{len(normal_weights)}"
    )

    assert len(hard_weights) == 146

    assert len(normal_weights) == 1527

    print(
        "✓ Hard/normal weight distribution "
        "matches expected values"
    )

    # ============================================================
    # 10. VERIFY SAMPLER CONFIGURATION
    # ============================================================

    print("\n[10] VERIFY SAMPLER CONFIGURATION")
    print("-" * 70)

    sampler = train_loader.sampler

    assert (
    ultralytics_trainer.sampling_replacement
    == config.SAMPLING_REPLACEMENT
    )

    print(
    f"trainer replacement : "
    f"{ultralytics_trainer.sampling_replacement} ✓"
    )

    assert (
    ultralytics_trainer.sampling_seed
    == config.SAMPLING_SEED
    )

    print(
    f"trainer seed        : "
    f"{ultralytics_trainer.sampling_seed} ✓"
    )

    # ============================================================
    # 11. VERIFY SAMPLER ITERATION
    # ============================================================

    print("\n[11] TEST SAMPLER ITERATION")
    print("-" * 70)

    sampler.set_epoch(0)

    sampled_indices = list(
        iter(sampler)
    )

    assert len(sampled_indices) == dataset_size

    print(
        f"Sampled indices : "
        f"{len(sampled_indices)} ✓"
    )

    assert all(
        0 <= index < dataset_size
        for index in sampled_indices
    )

    print(
        "All sampled indices are valid ✓"
    )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 70)
    print("✅ E3 TRAINER SETUP TEST PASSED")
    print("=" * 70)

    print()
    print(
        "Training was NOT started."
    )

    print(
        "Hard-example sampling is ready."
    )


if __name__ == "__main__":
    main()