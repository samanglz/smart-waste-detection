# ADR-004: Dataset Abstraction Implementation

## Status
Accepted

## Context

The framework requires a unified interface for loading and accessing dataset
samples across different model implementations (YOLO, MMDetection, GroundingDINO).

The `BaseDataset` abstraction was defined in ADR-001 but remained incomplete.
The current implementation only provided `load()`, `validate()`, and
`get_metadata()` methods, which are insufficient for training pipelines that
need direct access to image-label pairs.

Additionally, the training pipeline expects a `get_dataset_config()` method
to pass framework-specific configuration to models (e.g., YOLO's `data.yaml`
structure).

## Decision

Extend the `BaseDataset` abstraction with the following required methods:

- `get_train_data() -> List[Dict[str, Any]]`
- `get_val_data() -> List[Dict[str, Any]]`
- `get_test_data() -> List[Dict[str, Any]]`
- `get_class_names() -> List[str]`
- `get_num_classes() -> int`
- `get_dataset_config() -> Dict[str, Any]`

Each dataset implementation (e.g., `YOLODataset`) encapsulates all
filesystem-specific logic and returns data in a framework-agnostic format.

The `YOLODataset` implementation specifically:
- Reads YOLO-formatted label files (.txt) containing `class_id cx cy w h`
- Loads class names from `data.yaml` located at the dataset root
- Provides a `get_dataset_config()` method that returns the dictionary
  expected by Ultralytics YOLO's `model.train(data=...)` parameter

## Consequences

### Positive

- Training pipelines become framework-agnostic.
- Each dataset format implementation encapsulates its own filesystem logic.
- Adding support for new dataset formats (COCO, Pascal VOC, etc.) requires
  only creating a new subclass of `BaseDataset`.
- The `data.yaml` structure is isolated within the YOLO implementation.

### Negative

- Additional methods increase the interface surface area.
- Dataset implementations must maintain consistency across all required methods.

## Related ADRs

- ADR-001: Unified Dataset Abstraction
- ADR-002: Separation of Training Components
- ADR-003: Avoid Premature Abstractions