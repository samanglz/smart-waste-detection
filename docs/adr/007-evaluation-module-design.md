# ADR-007: Evaluation Module Design

## Status
Accepted

## Context
After training, models must be rigorously evaluated on held-out test data
to assess real-world performance. The evaluation should provide:
- Overall metrics (mAP, precision, recall)
- Per-class metrics for detailed analysis
- Reproducible JSON reports

The evaluation should be separate from training to maintain clean
separation of concerns and enable evaluation of pre-trained models.

## Decision

### 1. Create a Dedicated Evaluator Class
Implement `ModelEvaluator` class that:
- Takes a trained `YOLOModel` and `YOLODataset`
- Supports evaluation on `train`, `val`, and `test` splits
- Returns metrics in a standardized dictionary format

### 2. Support Per-Class Metrics
Provide `evaluate_per_class()` method that computes:
- Precision per class
- Recall per class
- Average Precision (AP) per class

### 3. Generate JSON Reports
The `generate_report()` method creates a comprehensive JSON report containing:
- Dataset information
- Overall metrics
- Per-class metrics
- Model information

### 4. Use YOLO's Built-in Validation
The evaluator leverages `model.val()` from Ultralytics YOLO,
extracting metrics from the results object using the private
`_extract_metrics()` method.

## Consequences

### Positive
- Clear separation between training and evaluation
- Consistent evaluation across model types
- Reproducible and shareable evaluation reports
- Easy to compare different models

### Negative
- Additional abstraction layer
- YOLO-specific metric extraction may need refactoring for other models

### Future Extensibility
A `BaseEvaluator` abstract class has been designed to support:
- `YOLOEvaluator`: Current implementation
- `MMDetEvaluator`: For MMDetection models (future)
- `GroundingDINOEvaluator`: For GroundingDINO models (future)

## Related ADRs
- ADR-002: Separation of Training Components
- ADR-003: Avoid Premature Abstractions
- ADR-004: Dataset Abstraction Implementation

## Notes
Evaluation on `test` split is the primary use case for final model assessment,
while `val` split evaluation is typically used for monitoring during training.