# ADR-005: Inference Module Design

## Status
Accepted

## Context
The framework requires a clean interface for running inference (prediction)
on trained models. Users need to perform predictions on:
- Single images
- Batches of images
- Video files

The inference module should provide consistent output format regardless
of the underlying model implementation.

## Decision

### 1. Design a Dedicated Inference Class
Create a `YOLOInference` class that wraps the YOLO model's predict method
and provides:
- `predict_image()`: Single image prediction
- `predict_batch()`: Multiple images from a directory
- `predict_video()`: Video file prediction
- `predict_with_confidence()`: Prediction with custom confidence threshold

### 2. Standardize Output Format
All inference methods return predictions in a consistent dictionary format:
```python
{
    'class_id': int,
    'confidence': float,
    'bbox': [x1, y1, x2, y2],  # in pixel coordinates
    'image_path': str
}