

## Status
Accepted

## Context
After training a model, it needs to be deployed in production environments.
Different deployment scenarios require different model formats:
- **ONNX**: Cross-platform inference (CPU/GPU)
- **TensorRT**: Fastest inference on NVIDIA GPUs
- **TFLite**: Mobile and edge devices
- **TorchScript**: C++ deployment without Python

The export module should provide a unified interface for converting
trained models to these formats.

## Decision

### 1. Create a Dedicated Exporter Class
Implement `ModelExporter` class that:
- Takes a trained `YOLOModel` instance
- Provides separate methods for each export format
- Uses `configs/export_config.py` for default settings

### 2. Support Format-Specific Parameters
Each export method accepts format-specific parameters:
- ONNX: `simplify`, `dynamic`
- TensorRT: `workspace` (memory allocation)
- TFLite: `int8` (quantization)

### 3. Use Configuration for Defaults
The `export_config.py` file defines:
- `EXPORT_FORMAT`: Default format
- `SIMPLIFY`: Whether to simplify ONNX
- `DYNAMIC`: Whether to allow variable input sizes

### 4. Provide Convenience Method
The `export_default()` method exports to the format defined in config,
simplifying the most common use case.

## Consequences

### Positive
- Unified interface for all export formats
- Configuration-driven defaults reduce boilerplate
- Easy to add new formats (CoreML, OpenVINO, etc.)
- Clear separation between training and deployment

### Negative
- Additional abstraction layer
- Some format-specific parameters may be confusing

## Related ADRs
- ADR-002: Separation of Training Components
- ADR-003: Avoid Premature Abstractions

## Notes
When using exported models (ONNX, TensorRT), users must manually
normalize input images (divide by 255.0) in their inference code
as this is not automatically handled by the exported model.