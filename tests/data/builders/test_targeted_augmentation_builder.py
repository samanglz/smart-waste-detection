from pathlib import Path
import shutil

from src.augmentations.targeted_augmentation import (
    TargetedAugmentation,
)

from src.data.builders.targeted_augmentation_builder import (
    TargetedAugmentationDatasetBuilder,
)


def test_build_e2_dataset():

    source_root = Path("data/processed")
    output_root = Path("data/e2_targeted_test")

    # فقط output تست را پاک می‌کنیم.
    # به data/processed دست نمی‌زنیم.
    if output_root.exists():
        shutil.rmtree(output_root)

    augmenter = TargetedAugmentation()

    builder = TargetedAugmentationDatasetBuilder(
        source_root=source_root,
        output_root=output_root,
        augmenter=augmenter,
    )

    result = builder.build()

    # -------------------------------------------------
    # Exact E2 numbers (must match Colab)
    # -------------------------------------------------

    assert result["build"]["original"] == 1673
    assert result["build"]["target"] == 694
    assert result["build"]["augmented"] == 694
    assert result["build"]["total"] == 2367

    # -------------------------------------------------
    # Dataset validation
    # -------------------------------------------------

    validation = result["validation"]

    assert validation["images"] == 2367
    assert validation["labels"] == 2367
    assert validation["missing_labels"] == 0
    assert validation["invalid_boxes"] == 0

    print("E2 Builder PASSED")
    print(result)


if __name__ == "__main__":
    test_build_e2_dataset()