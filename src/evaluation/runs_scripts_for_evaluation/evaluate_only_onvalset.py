from pathlib import Path

from configs import config
from src.data.yolo_dataset import YOLODataset
from src.models.yolo.yolo_model import YOLOModel

from src.evaluation import ModelEvaluator_copy

from src.logging.logger import get_logger


logger = get_logger(__name__)


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


def main():

    # ---------------------------------------------------------
    # 1. Dataset
    # ---------------------------------------------------------

    dataset = YOLODataset(
        config.FINAL_DATASET_DIR
    )

    # ---------------------------------------------------------
    # 2. Model
    # ---------------------------------------------------------

    model_path = Path(
        "runs/detect/runs/E2_targeted_augmentation/"
        "targeted_aug/weights/best.pt"
    )

    model = YOLOModel(model_path)

    # ---------------------------------------------------------
    # 3. Evaluator
    # ---------------------------------------------------------

    evaluator = ModelEvaluator_copy(
        model=model,
        dataset=dataset,
    )

    # ---------------------------------------------------------
    # 4. Evaluation output directory
    # ---------------------------------------------------------

    evaluation_dir = Path(
        "outputs/yolo11m/E2_targeted_augmentation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 5. Standard TEST evaluation
    # ---------------------------------------------------------

    logger.info(
        "\n📊 Running TEST evaluation..."
    )

    evaluator.generate_report(
        output_path=evaluation_dir / "report.json",
        split="test",
    )

    # ---------------------------------------------------------
    # 6. VAL error analysis
    #
    # IMPORTANT:
    # VAL is used only for failure-pattern discovery.
    # No TEST data is involved in this stage.
    # ---------------------------------------------------------

    logger.info(
        "\n🔎 Running VAL error analysis "
        "for failure-pattern discovery..."
    )

    evaluator.analyze_errors(
        split="val",
        output_path=(
            evaluation_dir
            / "val_error_analysis.json"
        ),
    )

    logger.info(
        "\n✅ VAL error report generated:"
        "\n%s",
        evaluation_dir / "val_error_analysis.json",
    )

    logger.info(
        "\n✅ Evaluation preparation completed successfully!"
    )


if __name__ == "__main__":
    main()