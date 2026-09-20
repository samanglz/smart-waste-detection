from pathlib import Path

from src.trainers.yolo_hard_trainer_sampler_trainer import (
    HardSamplingDetectionTrainer,
)


SAMPLING_WEIGHTS_PATH = Path(
    "outputs/hard_example_mining/E2/sampling_weights.json"
)


def main():
    print("=" * 70)
    print("HARD SAMPLING SMOKE TEST")
    print("=" * 70)

    trainer = HardSamplingDetectionTrainer(
        overrides={
            "model": "yolo11m.pt",
            "data": "data/processed/data.yaml",
            "batch": 16,
            "workers": 0,
            "device": 0,
        },
        sampling_weights_path=SAMPLING_WEIGHTS_PATH,
    )

    print("\nTrainer created successfully.")

    print(
        "Loaded sampling weights:",
        len(trainer.sampling_weights),
    )

    hard = sum(
        1
        for weight in trainer.sampling_weights.values()
        if weight > 1.0
    )

    normal = sum(
        1
        for weight in trainer.sampling_weights.values()
        if weight == 1.0
    )

    print("Hard images:", hard)
    print("Normal images:", normal)

    print("\nSmoke test configuration loaded successfully.")

    print("=" * 70)


if __name__ == "__main__":
    main()