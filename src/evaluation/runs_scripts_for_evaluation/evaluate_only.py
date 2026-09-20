# evaluate_only.py
'''
from pathlib import Path
from configs import config
from src.data.yolo_dataset import YOLODataset
from src.models.yolo.yolo_model import YOLOModel
from src.evaluation import ModelEvaluator
from src.evaluation import ModelEvaluator_copy
from src.evaluation import ErrorVisualizer

from src.logging.logger import get_logger

logger = get_logger(__name__)



class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]



def main():
    # بارگذاری دیتاست
    dataset = YOLODataset(config.FINAL_DATASET_DIR)

    # upload best trained model
    model_path = Path("runs/detect/runs/yolo11m/detect/weights/best.pt")
    model = YOLOModel(model_path)

    # evaluation
    evaluator = ModelEvaluator_copy(model, dataset)
    evaluator.generate_report(Path("outputs/yolo11m/evaluation/report.json"), split="test")


       # 2. Run error analysis
    logger.info("\n🔬 Running error analysis...")
    evaluator.analyze_errors(
        split="test",
        output_path=Path("outputs/yolo11m/evaluation/error_analysis.json")
    )

    logger.info("\n✅ All tasks completed successfully!")
    
    
    
if __name__ == "__main__":
    main()


'''



from pathlib import Path

from configs import config

from src.data.yolo_dataset import YOLODataset
from src.models.yolo.yolo_model import YOLOModel

from src.evaluation import (
    ModelEvaluator_copy,
    ErrorVisualizer,
)

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
        "runs/detect/runs/E7_plus/weights/best.pt"
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
    # 4. Standard evaluation
    # ---------------------------------------------------------

    evaluation_dir = Path(
        "outputs/yolo11m/E7_plus"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluator.generate_report(
        output_path=evaluation_dir / "report.json",
        split="test",
    )

    # ---------------------------------------------------------
    # 5. Error analysis
    # ---------------------------------------------------------

    logger.info(
        "\n🔬 Running error analysis..."
    )

    error_report = evaluator.analyze_errors(
        split="test",
        output_path=(
            evaluation_dir
            / "error_analysis.json"
        ),
    )



    # ---------------------------------------------------------
    # 6. Error prioritization
    # ---------------------------------------------------------

    logger.info(
        "\n🎯 Running error prioritization..."
    )

    prioritized_errors = evaluator.prioritize_errors(
        split="test",
        top_k=5,
    )

    logger.info(
        "\nPrioritized errors:\n%s",
        prioritized_errors,
    )



    # ---------------------------------------------------------
    # 7. Error visualizer
    # ---------------------------------------------------------

    logger.info(
        "\n🖼️ Running error visualization..."
    )

    visualizer = ErrorVisualizer(
        class_names=CLASS_NAMES
    )

    error_analysis = error_report[
        "error_analysis"
    ]

    visualization_dir = (
        evaluation_dir
        / "visualizations"
    )
    print(type(error_analysis["confusion"]))

    print(error_analysis["confusion"])
    # ---------------------------------------------------------
    # 8. Visualize prioritized confusions
    # ---------------------------------------------------------

    for item in prioritized_errors["confusions"]:

        visualizer.visualize_confusion(
            cases=error_analysis["confusion"],
            gt_class=item["gt_class"],
            pred_class=item["pred_class"],
            output_dir=visualization_dir,
            top_k=12,
        )

    # ---------------------------------------------------------
    # 9. Visualize prioritized false positives
    # ---------------------------------------------------------

    for item in prioritized_errors["false_positives"]:

        visualizer.visualize_false_positives(
            cases=error_analysis["false_positives"],
            class_name=item["class_name"],
            output_dir=visualization_dir,
            top_k=12,
        )

    # ---------------------------------------------------------
    # 10. Visualize prioritized false negatives
    # ---------------------------------------------------------

    for item in prioritized_errors["false_negatives"]:

        visualizer.visualize_false_negatives(
            cases=error_analysis["false_negatives"],
            class_name=item["class_name"],
            output_dir=visualization_dir,
            top_k=12,
        )
    logger.info(
        "\n✅ All evaluation tasks completed successfully!"
    )


if __name__ == "__main__":
    main()