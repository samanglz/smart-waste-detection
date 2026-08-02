# ADR-001: Unified Dataset Abstraction

## Status
Accepted

## Context

The framework should support multiple dataset formats such as YOLO and COCO.

Directly coupling the training pipeline to a specific dataset format would make
future extensions harder and violate the Open/Closed Principle.

## Decision

Introduce a `BaseDataset` abstraction as the common interface for all dataset
implementations.

Each dataset implementation must provide:

- `validate()`: Validate dataset structure and required files.
- `load()`: Load dataset metadata required by the framework.

Dataset-specific logic should remain inside the corresponding implementation
(e.g., `YOLODataset`, `COCODataset`).

Dataset lifecycle orchestration is handled by higher-level components such as
the training engine or trainer.

## Consequences

### Positive

- New dataset formats can be added without modifying existing implementations.
- Dataset-specific logic remains isolated.
- The framework follows the Open/Closed Principle.

### Negative

- Additional abstraction complexity is introduced.
- Interfaces must be carefully designed to avoid unnecessary constraints.