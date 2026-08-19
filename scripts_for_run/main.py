"""
Main entry point for Smart Waste Detection project.

This script orchestrates the entire pipeline:
    1. Load configuration
    2. Load dataset
    3. Train model
    4. Evaluate on test set
    5. Export model
    6. Generate final report
"""

from pathlib import Path
import sys

# Add project root to path (if needed)
# sys.path.insert(0, str(Path(__file__).parent))

from configs import config
from src.data.yolo_dataset import YOLODataset
from src.trainers.yolo_trainer import YOLOTrainer
from src.pipelines.training_pipeline import TrainingPipeline
from src.evaluation import ModelEvaluator
from src.export import ModelExporter
from src.logging.logger import get_logger



logger = get_logger(__name__)

def main():
    print("main started")
    """
    Run the complete training and evaluation pipeline.
    
    """
    
    logger.info("=" * 60)
    logger.info("Smart Waste Detection - Training Pipeline")
    logger.info("=" * 60)    
    
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
    
    pipeline = TrainingPipeline(config, trainer)
    pipeline.run()
    
    logger.info("  Training completed successfully")
    
   # ============================================================
    # Step 4: Evaluate on test set
    # ============================================================
    logger.info("Step 4: Evaluating on test set...")
    
    
    evaluator = ModelEvaluator(trainer.model, dataset)
    test_metrics = evaluator.evaluate(split = "test")
    
    logger.info(" Test metrics: ")
    
    for key, value in test_metrics.items():
        if key != "speed":  # showing speed with defferent format
            logger.info("    - %s: %.4f", key, value)
        else:
            logger.info("    - %s: %s", key, value)
    
    # Generate evaluation report
    report_path = Path("outputs/evaluation/yolo11m/report.json")
    evaluator.generate_report(report_path, split="test")
    logger.info("  Evaluation report saved to: %s", report_path)
    
    
       # 2. Run error analysis
    logger.info("\n🔬 Running error analysis...")
    evaluator.analyze_errors(
        split="test", 
        output_path=Path("outputs/evaluation/yolo11m/error_analysis.json")
        
    )

    logger.info("\n✅ All evaluation tasks completed successfully!")
    
    
    
    # ============================================================
    # Step 5: Export model
    # ============================================================
    logger.info("Step 5: Exporting model...")
    
    exporter = ModelExporter(trainer.model)
    export_dir = Path("outputs/export")
    
    # Create export directory if it doesn't exist
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export with default format from config
    exported_path = exporter.export_default(export_dir)
    logger.info("  Model exported to: %s", exported_path)
    
    # Optional: Export to specific formats (commented out)
    # exporter.export_onnx(export_dir / "model.onnx")
    # exporter.export_tensorrt(export_dir / "model.engine")
    
    
    # ============================================================
    # Step 6: Final summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("  - Model weights: %s", trainer.model.weight_path)
    logger.info("  - Evaluation report: %s", report_path)
    logger.info("  - Exported model: %s", exported_path)
    logger.info("=" * 60)   

if __name__ == "__main__":
    try:
        
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by User")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("Pipeline failed with error: %s", str(e))
        sys.exit(1)