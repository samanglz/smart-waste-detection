Introduction

Dataset
v1.0
752 manually annotated images

v1.1
Auto annotation added only for the rest of the trainset

v1.2
Human correction completed



Project Structure

# Smart Waste Detection

End-to-end computer vision pipeline for smart waste detection and classification.

## 🎯 Purpose
Detect and classify five types of waste:
- 🟫 Cardboard
- 🫙 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic

## 🧠 Models Supported
- ✅ YOLO (Ultralytics)
- ⏳ MMDetection (Coming Soon)
- ⏳ GroundingDINO (Coming Soon)

## 🏗️ Architecture
- Modular design with dependency injection
- Support for multiple dataset formats
- Unified interfaces for training, evaluation, and inference
- Clean separation of concerns
- Comprehensive logging and error handling

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python -m main
```

## 📁 Project Structure
```
smart-waste-detection/
├── src/
│   ├── data/          # Dataset handling
│   ├── models/        # Model implementations
│   ├── trainers/      # Training logic
│   ├── pipelines/     # Workflow orchestration
│   ├── evaluation/    # Model evaluation
│   ├── export/        # Model export
│   ├── inference/     # Prediction
│   ├── augmentation/  # Data augmentation
│   └── logging/       # Logging utilities
├── configs/           # Configuration files
├── data/              # Dataset directory
├── outputs/           # Output artifacts
└── docs/              # Documentation
```



Training

Evaluation

Results

Docker

License