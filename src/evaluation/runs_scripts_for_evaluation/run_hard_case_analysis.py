"""

Not dynamically obtained from prioritization but also static
Run hard case similarity analysis for priority confusions.

This script:
    1. Loads the evaluation JSON file
    2. Extracts cases for priority confusions
    3. Runs HardCaseSimilarityAnalyzer for each
    4. Generates reports
"""

import json
from pathlib import Path

from src.evaluation.hard_case_detect.hard_case_similarity import HardCaseSimilarityAnalyzer
from src.logging.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Hard Case Similarity Analysis")
    logger.info("=" * 60)

    # ============================================================
    # Step 1: Load evaluation JSON
    # ============================================================
    json_path = Path("outputs/yolo11m/evaluation/error_analysis.json")
    if not json_path.exists():
        logger.error("File not found: %s", json_path)
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    confusion = data.get("error_analysis", {}).get("confusion", {})

    logger.info("Loaded confusion data from: %s", json_path)

    # ============================================================
    # Step 2: Define priority confusions
    # ============================================================
    priority_confusions = [
        ("glass", "plastic"),
        ("metal", "plastic"),
        ("cardboard", "paper"),
        ("metal", "glass"),
        ("plastic", "paper"),
    ]

    # ============================================================
    # Step 3: Initialize analyzer
    # ============================================================
    logger.info("Initializing HardCaseSimilarityAnalyzer...")
    analyzer = HardCaseSimilarityAnalyzer(
        train_images_dir=Path("data/processed/train/images"),
        train_labels_dir=Path("data/processed/train/labels"),
        class_names=["cardboard", "glass", "metal", "paper", "plastic"],
        device="cuda",
    )

    # Build index for all classes
    all_classes = ["cardboard", "glass", "metal", "paper", "plastic"]
    logger.info("Building train index for all classes...")
    analyzer.build_train_index(all_classes)
    logger.info("Train index built successfully")

    # ============================================================
    # Step 4: Analyze each confusion
    # ============================================================
    results_summary = {}

    for gt_class, pred_class in priority_confusions:
        logger.info("-" * 60)
        logger.info("Analyzing: %s → %s", gt_class, pred_class)

        # Extract cases
        cases = confusion.get(gt_class, {}).get(pred_class, {}).get("cases", [])
        if not cases:
            logger.warning("No cases found for %s → %s", gt_class, pred_class)
            continue

        logger.info("  Found %d cases", len(cases))

        # Analyze cases
        try:
            results = analyzer.analyze_cases(cases, top_k=5)
            output_dir = Path(f"outputs/hard_case_analysis/{gt_class}_to_{pred_class}")
            analyzer.generate_report(results, output_dir)
            logger.info("  Report saved to: %s", output_dir)

            # Store summary
            nearest_counts = {}
            for r in results:
                nearest = r.get("nearest_class")
                if nearest:
                    nearest_counts[nearest] = nearest_counts.get(nearest, 0) + 1

            results_summary[f"{gt_class}→{pred_class}"] = {
                "total": len(cases),
                "nearest_counts": nearest_counts,
            }

        except Exception as e:
            logger.error("Error analyzing %s → %s: %s", gt_class, pred_class, e)

    # ============================================================
    # Step 5: Summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, summary in results_summary.items():
        total = summary["total"]
        counts = summary["nearest_counts"]
        logger.info("%s:", name)
        logger.info("  Total: %d", total)
        for cls, count in counts.items():
            logger.info("  Nearest %s: %d (%.1f%%)", cls, count, count / total * 100)
        logger.info("")

    logger.info("=" * 60)
    logger.info("All analyses completed!")


if __name__ == "__main__":
    main()