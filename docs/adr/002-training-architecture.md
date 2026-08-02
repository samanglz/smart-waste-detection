# ADR-002: Separation of Training Components

## Status
Accepted

## Context

The framework should support different training backends and models such as
YOLO, MMDetection, and GroundingDINO.

Placing all training logic inside a single trainer implementation would create
large, tightly coupled classes.

## Decision

Separate training responsibilities into independent components:

- `BaseTrainer`: Defines the common training lifecycle interface.
- Model-specific trainers: Implement framework-specific behavior.
- Models: Encapsulate model-specific operations.
- Engine components: Orchestrate the overall workflow.

The trainer is responsible for training behavior, while higher-level components
manage execution order and workflow.

## Consequences

### Positive

- Easier addition of new training approaches.
- Reduced coupling between models and training logic.
- Better maintainability and testability.

### Negative

- More files and abstractions compared to a simple script-based approach.