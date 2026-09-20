from pathlib import Path
import json

from configs.e4_hard_case_analysis_config import E4HardCaseAnalysisConfig
from src.data.yolo_dataset import YOLODataset
from src.models.yolo.yolo_model import YOLOModel
from src.evaluation.model_evaluator_Copy import ModelEvaluator_copy
from src.analysis.hard_case_similarity_analyzer import (
    HardCaseSimilarityAnalyzer,
)


def main():

    config = E4HardCaseAnalysisConfig

    print("=" * 70)
    print("E4 HARD CASE SIMILARITY ANALYSIS")
    print("=" * 70)

    # ============================================================
    # 1. BUILD DATASET
    # ============================================================

    print("\n[1] BUILD DATASET")
    print("-" * 70)

    dataset = YOLODataset(
        dataset_root=config.DATASET_ROOT
    )

    print(f"Dataset root : {config.DATASET_ROOT}")
    print(f"Dataset YAML : {dataset.yaml_path}")
    print(f"Train images : {len(dataset.get_train_data())}")
    print(f"Val images   : {len(dataset.get_val_data())}")
    print(f"Test images  : {len(dataset.get_test_data())}")

    # ============================================================
    # 2. LOAD E3 MODEL
    # ============================================================

    print("\n[2] LOAD E3 MODEL")
    print("-" * 70)

    model = YOLOModel(
        config.MODEL_PATH
    )

    print(f"E3 model : {config.MODEL_PATH}")

    # ============================================================
    # 3. BUILD MODEL EVALUATOR
    # ============================================================

    print("\n[3] BUILD MODEL EVALUATOR")
    print("-" * 70)

    evaluator = ModelEvaluator_copy(
        model=model,
        dataset=dataset,
    )

    print("✓ ModelEvaluator ready")

    # ============================================================
    # 4. RUN ERROR ANALYSIS
    # ============================================================

    print("\n[4] RUN ERROR ANALYSIS")
    print("-" * 70)

    error_report = evaluator.analyze_errors(
        split="test",
        output_path=config.ERROR_ANALYSIS_PATH,
    )

    print(
        f"Error analysis saved to:\n"
        f"{config.ERROR_ANALYSIS_PATH}"
    )

    # ============================================================
    # 5. EXTRACT HARD CASES
    # ============================================================

    print("\n[5] EXTRACT HARD CASES")
    print("-" * 70)

    confusion = error_report[
        "error_analysis"
    ]["confusion"]

    cases = []

    for gt_class, predictions in confusion.items():

        for pred_class, data in predictions.items():

            class_cases = data.get("cases", [])

            for case in class_cases:

                cases.append(case)

    print(f"Total confusion cases : {len(cases)}")

    if not cases:
        print("⚠ No confusion cases found.")
        return

    # ============================================================
    # 6. BUILD SIMILARITY ANALYZER
    # ============================================================

    print("\n[6] BUILD HARD CASE SIMILARITY ANALYZER")
    print("-" * 70)

    analyzer = HardCaseSimilarityAnalyzer(
        train_images_dir=config.TRAIN_IMAGES_DIR,
        train_labels_dir=config.TRAIN_LABELS_DIR,
        class_names=dataset.get_class_names(),
        device=config.DEVICE,
    )

    print("✓ Similarity analyzer ready")

    # ============================================================
    # 7. BUILD TRAIN EMBEDDING INDEX
    # ============================================================

    print("\n[7] BUILD TRAIN EMBEDDING INDEX")
    print("-" * 70)

    analyzer.build_train_index(
        target_classes=config.TARGET_CLASSES
    )

    print("✓ Train embedding index created")

    # ============================================================
    # 8. ANALYZE HARD CASES
    # ============================================================

    print("\n[8] ANALYZE HARD CASES")
    print("-" * 70)

    results = analyzer.analyze_cases(
        cases=cases,
        top_k=config.TOP_K,
    )

    print(
        f"Analyzed cases : {len(results)}"
    )

    # ============================================================
    # 9. GENERATE REPORT
    # ============================================================

    print("\n[9] GENERATE E4 REPORT")
    print("-" * 70)

    report_path = analyzer.generate_report(
        results=results,
        output_dir=config.OUTPUT_DIR,
    )

    print(
        f"E4 report saved to:\n"
        f"{report_path}"
    )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 70)
    print("✅ E4 HARD CASE ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()