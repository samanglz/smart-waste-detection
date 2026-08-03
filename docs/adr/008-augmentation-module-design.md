# ADR-008: Augmentation Module Design

## Status
Accepted

## Context
Data augmentation is critical for improving model generalization,
especially with limited datasets. YOLO provides built-in augmentations
with configurable parameters.

The augmentation module should provide:
- Centralized configuration for all augmentation parameters
- Predefined profiles for common scenarios
- Clean integration with the training pipeline

## Decision

### 1. Create a Configuration Class
Implement `YOLOAugmentationConfig` dataclass that contains:
- All YOLO augmentation parameters as attributes
- Default values matching Ultralytics YOLO v11
- Type hints for clarity

### 2. Provide Predefined Profiles
The class includes methods for common scenarios:
- `disable_all()`: No augmentations (for validation)
- `enable_light_augmentations()`: Fine-tuning scenarios
- `enable_heavy_augmentations()`: Small datasets

### 3. Support Conversion to Dict
The `to_dict()` method converts the configuration to a dictionary
suitable for passing to `model.train()` using `**kwargs`.

### 4. Integration with Training Config
The `training_config.py` file defines `AUGMENTATION` variable
using `YOLOAugmentationConfig`, making it easily adjustable.

## Consequences

### Positive
- All augmentation settings are centralized and visible
- Easy to switch between augmentation profiles
- Clean separation between configuration and training logic
- Extensible for future models (MMDetection, GroundingDINO)

### Negative
- Additional abstraction layer
- YOLO-specific class may need adaptation for other models

### Future Extensibility
A `BaseAugmentationConfig` could be designed to support:
- `YOLOAugmentationConfig`: Current implementation
- `MMDetAugmentationConfig`: For MMDetection (future)
- `GroundingDINOAugmentationConfig`: For GroundingDINO (future)

## Related ADRs
- ADR-002: Separation of Training Components
- ADR-003: Avoid Premature Abstractions
- ADR-004: Dataset Abstraction Implementation

## Notes
The augmentation module follows the YAGNI principle, adding only
YOLO-specific augmentations now. When other models are added,
the module can be extended without modifying existing code.