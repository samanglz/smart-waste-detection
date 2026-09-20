from pathlib import Path
import torch

from src.trainers.yolo_hard_trainer_sampler_trainer import (
    HardSamplingDetectionTrainer,
)


SAMPLING_WEIGHTS = Path(
    "outputs/hard_example_mining/E2/sampling_weights.json"
)

TRAIN_IMAGES = Path(
    "data/processed/train/images"
)


def main():

    trainer = HardSamplingDetectionTrainer(
        overrides={
            "model": "yolo11m.pt",
            "data": "data/processed/data.yaml",
            "batch": 16,
            "workers": 0,
            "device": 0,
        },
        sampling_weights_path=SAMPLING_WEIGHTS,
    )

    # ---------------------------------------------------------
    # Build YOLO model
    # ---------------------------------------------------------

    trainer.setup_model()

    # ---------------------------------------------------------
    # Mimic parsed dataset YAML
    # ---------------------------------------------------------

    trainer.data = {
        "train": str(TRAIN_IMAGES),
        "val": "data/processed/val/images",
        "names": {
            0: "cardboard",
            1: "glass",
            2: "metal",
            3: "paper",
            4: "plastic",
        },
        "nc": 5,
    }

    # ---------------------------------------------------------
    # Build TRAIN dataloader
    # ---------------------------------------------------------

    loader = trainer.get_dataloader(
        dataset_path=str(TRAIN_IMAGES),
        batch_size=16,
        rank=-1,
        mode="train",
    )

    print("=" * 60)
    print("TRAIN DATALOADER TEST")
    print("=" * 60)

    print("Loader class :", type(loader).__name__)
    print("Sampler class:", type(loader.sampler).__name__)
    print("Dataset size :", len(loader.dataset))
    print("Batch size   :", loader.batch_size)

    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Verify sampler type
    # ---------------------------------------------------------

    assert (
        type(loader.sampler).__name__
        == "EpochWeightedRandomSampler"
    ), (
        "Expected EpochWeightedRandomSampler, "
        f"got {type(loader.sampler).__name__}"
    )

    print("✅ WeightedRandomSampler is ACTIVE")

    # ---------------------------------------------------------
    # 2. Verify sampler weights
    # ---------------------------------------------------------

    sampler = loader.sampler

    weights = torch.as_tensor(
        sampler.weights,
        dtype=torch.float64,
    )

    hard_count = int(
        (weights > 1.0).sum().item()
    )

    normal_count = int(
        (weights == 1.0).sum().item()
    )

    print()
    print("=" * 60)
    print("SAMPLING WEIGHT VALIDATION")
    print("=" * 60)

    print("Total weights :", len(weights))
    print("Hard weights  :", hard_count)
    print("Normal weights:", normal_count)
    print("Min weight    :", weights.min().item())
    print("Max weight    :", weights.max().item())

    # ---------------------------------------------------------
    # 3. Validate expected E3 statistics
    # ---------------------------------------------------------

    assert len(weights) == 1673, (
        f"Expected 1673 weights, got {len(weights)}"
    )

    assert hard_count == 146, (
        f"Expected 146 hard images, got {hard_count}"
    )

    assert normal_count == 1527, (
        f"Expected 1527 normal images, got {normal_count}"
    )

    assert weights.min().item() == 1.0
    assert weights.max().item() == 3.0

    print()
    print("✅ 1673 images have sampling weights")
    print("✅ 146 hard images have weight > 1")
    print("✅ 1527 normal images have weight = 1")
    print("✅ Weight range is 1.0 → 3.0")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("✅ HARD SAMPLING DATALOADER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()