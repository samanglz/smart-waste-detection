#!/usr/bin/env python
"""
Training script with a larger model (YOLO11m) and focus on Glass.

This script:
    1. Loads configuration with larger model
    2. Applies glass-focused augmentations
    3. Uses class weights for Glass
    4. Trains with more epochs
    5. Evaluates and exports the model
"""

from pathlib import Path
import sys

from configs import config
from src.data.yolo_dataset import YOLODataset
from src.trainers.yolo_trainer import YOLOTrainer
from src.pipelines.training_pipeline import TrainingPipeline
from src.evaluation import ModelEvaluator
from src.export import ModelExporter
from src.logging.logger import get_logger


logger = get_logger(__name__)


def main():
    print("main started - Training with Large Model (YOLO11m)")
    
    logger.info("=" * 60)
    logger.info("Smart Waste Detection - Training with Large Model")
    logger.info("=" * 60)
    
    # ============================================================
    # Step 0: Override configuration for large model training
    # ============================================================
    logger.info("Step 0: Configuring large model settings...")
    
    # Set model to YOLO11m (larger model)
    config.MODEL_PATH = "yolo11m.pt"
    logger.info("  Model: %s", config.MODEL_PATH)
    
    # Increase epochs for better convergence
    config.EPOCHS = 100
    logger.info("  Epochs: %d", config.EPOCHS)
    
    # Reduce batch size for memory management
    config.BATCH_SIZE = 4
    logger.info("  Batch size: %d", config.BATCH_SIZE)
    
    # Reduce workers for memory management
    config.WORKERS = 16
    logger.info("  Workers: %d", config.WORKERS)
    
    logger.info("  Configuration complete")
    
    # ============================================================
    # Step 1: Load dataset
    # ============================================================
    logger.info("Step 1: Loading dataset...")
    
    dataset = YOLODataset(config.FINAL_DATASET_DIR)
    
    logger.info("  Dataset loaded successfully:")
    logger.info("    - Classes: %s", dataset.get_class_names())
    logger.info("    - Num classes: %d", dataset.get_num_classes())
    logger.info("    - Train samples: %d", len(dataset.get_train_data()))
    logger.info("    - Val samples: %d", len(dataset.get_val_data()))
    logger.info("    - Test samples: %d", len(dataset.get_test_data()))
    
    # ============================================================
    # Step 2: Create trainer
    # ============================================================
    logger.info("Step 2: Creating trainer...")
    
    trainer = YOLOTrainer(config, dataset)
    logger.info("  Trainer created successfully")
    
    # ============================================================
    # Step 3: Run training pipeline
    # ============================================================
    logger.info("Step 3: Running training pipeline...")
    logger.info("  This will take longer with YOLO11m...")
    
    pipeline = TrainingPipeline(config, trainer)
    pipeline.run()
    
    logger.info("  Training completed successfully")
    
    # ============================================================
    # Step 4: Evaluate on test set
    # ============================================================
    logger.info("Step 4: Evaluating on test set...")
    
    evaluator = ModelEvaluator(trainer.model, dataset)
    test_metrics = evaluator.evaluate(split="test")

    
    logger.info("  Test metrics:")
    for key, value in test_metrics.items():
        if key != "speed":
            logger.info("    - %s: %.4f", key, value)
        else:
            logger.info("    - %s: %s", key, value)
        # Create export directory if it doesn't exist
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # Generate evaluation report
    report_path = config.OUTPUTS_DIR / "evaluation" / "report.json"
    evaluator.generate_report(report_path, split="test")
    logger.info("  Evaluation report saved to: %s", report_path)
    
    # Run error analysis
    logger.info("\n🔬 Running error analysis...")
    analyze_path = config.OUTPUTS_DIR / "evaluation" /  "error_analysis_large_model.json"
    evaluator.analyze_errors(
        split="test",
        output_path=analyze_path
    )
    
    logger.info("\n✅ All evaluation tasks completed successfully!")
    
    # ============================================================
    # Step 5: Export model
    # ============================================================
    logger.info("Step 5: Exporting model...")
    
    exporter = ModelExporter(trainer.model)
    export_dir = config.OUTPUTS_DIR / "export" / "export_large_model"
    
    # Create export directory if it doesn't exist
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export with default format from config
    exported_path = exporter.export_default(export_dir)
    logger.info("  Model exported to: %s", exported_path)
    
    # ============================================================
    # Step 6: Final summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("  - Model: %s", config.MODEL_PATH)
    logger.info("  - Model weights: %s", trainer.model.weight_path)
    logger.info("  - Evaluation report: %s", report_path)
    logger.info("  - Error analysis: %s", Path("outputs/evaluation/error_analysis_large_model.json"))
    logger.info("  - Exported model: %s", exported_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("Pipeline failed with error: %s", str(e))
        sys.exit(1)