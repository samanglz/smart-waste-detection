# Smart Waste Detection

An end-to-end, modular computer vision pipeline for **waste object detection and classification**, with a focus on reproducible experimentation, real-world domain adaptation, model evaluation, and deployment.

The project detects five categories of waste:

* 🟫 Cardboard
* 🫙 Glass
* 🥫 Metal
* 📄 Paper
* 🧴 Plastic

---

## 🎯 Project Overview

The goal of this project is to build a production-oriented waste detection system while systematically studying how different data-centric and model-centric approaches affect detection performance.

The project follows an experiment-driven workflow:

**Dataset → Training → Evaluation → Error Analysis → Domain Adaptation → Deployment**

The current development focuses on **YOLO-based object detection**, with the architecture designed to support additional detection frameworks in the future.

---

## 📊 Dataset

The dataset was progressively developed through multiple annotation stages:

| Version  | Description                                             |
| -------- | ------------------------------------------------------- |
| **v1.0** | 752 manually annotated images                           |
| **v1.1** | Auto-annotation added for the remaining training images |
| **v1.2** | Human correction of the auto-annotated data completed   |

The dataset contains five waste categories:

```text
cardboard
glass
metal
paper
plastic
```

The dataset is organized into separate training, validation, and test splits.

### Evaluation Protocol

The validation and test sets are kept unchanged across the main experiments to provide a consistent basis for comparison.

This is particularly important for the domain-adaptation experiments, where changes are applied to the **training data only**.

---

# 🧠 Models

## Current

* ✅ YOLO / Ultralytics  -> YOLO11m as baseline

The project does not rely on the Ultralytics training pipeline as the main application architecture. Instead, model-specific functionality is integrated into a modular project-level framework.

---

# 🏗️ Architecture

The project is designed around modular components and separation of concerns.

Core components include:

* Dataset management
* Model abstraction
* Training
* Evaluation
* Error analysis
* Data augmentation
* Domain adaptation
* Inference
* Model export
* Deployment benchmarking
* API serving
* Configuration management
* Logging
* Visualization

The architecture uses reusable interfaces, dependency injection, and extensible components to make experiments easier to reproduce and extend.

---

# 🧪 Experiments

The project uses controlled experiments to evaluate different approaches.

## E0 — Baseline

Initial baseline experiment used to establish the reference performance.

---

## E1 — Lower Learning Rate Fine-Tuning

A fine-tuning experiment using a lower learning rate to investigate whether additional optimization could improve the baseline model.

---

## E2 — Targeted Augmentation

The main benchmark baseline for subsequent experiments.

Targeted augmentation was applied to underrepresented classes, particularly:

* Glass
* Plastic

E2 is currently used as the primary benchmark when comparing subsequent experimental approaches.

---

## E3 — Hard Example Sampling

An experiment investigating hard-example-based sampling.

The goal was to increase the contribution of difficult training examples.

---

## E4 — Similarity-Based Analysis

An experiment involving visual similarity analysis using deep image embeddings.

A ResNet-based representation was used to investigate relationships between difficult examples and the training data.

---

## E5 — Focal Loss

An experiment replacing the classification BCE loss with Focal Loss.

The experiment investigated whether focusing more strongly on difficult classification examples could improve detection performance.

The results did not outperform the E2 benchmark.

---

## E6 — Failure-Pattern-Guided Hard Mining

E6 combined failure-pattern analysis with similarity mining.

The experiment resulted in a regression compared with E2 and is therefore retained as a documented ablation rather than being continued as the main development direction.

---

## E7 — Real-World Domain Adaptation

E7 introduced a data-centric domain adaptation approach.

The objective was to reduce the distribution gap between the training data and real-world deployment conditions.

The validation and test sets remained unchanged.

---

# 🧩 E7 Plus

The current domain-adaptation pipeline extends E7 into a modular synthetic-data generation system.

### Pipeline

```text
                    ┌─────────────────┐
                    │   Source Data   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Object Bank   │
                    │                 │
                    │ Object Extractor│
                    │ Mask Generator  │
                    │ Metadata        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Context Engine  │
                    │                 │
                    │ Composition     │
                    │ Scale           │
                    │ Lighting        │
                    │ Occlusion       │
                    │ Reflection      │
                    │ Color Temperature│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Dataset Builder │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Training Dataset│
                    └─────────────────┘
```

### Object Bank

The Object Bank stores reusable object representations extracted from source images.

Objects are represented using RGBA crops together with associated metadata.

```text
src/e7/object_bank/
├── extractor.py
├── mask_generator.py
└── metadata.py
```

### Context Engine

The Context Engine places extracted objects into new visual contexts while controlling environmental properties.

```text
src/e7/context_engine/
├── compositor.py
├── scale.py
├── lighting.py
├── occlusion.py
├── reflection.py
└── color_temperature.py
```

### Dataset Builder

The Dataset Builder combines objects and generated contexts into a training dataset while preserving the required detection annotations.

The generated data is used only for training.

The original validation and test datasets remain unchanged.

---

# 📈 Evaluation

Evaluation is performed independently from training and includes:

* mAP@50
* mAP@75
* mAP@50:95
* Precision
* Recall
* Per-class metrics
* Error analysis
* Visualization

The project also includes experiment-specific evaluation reports and visualizations.

Example evaluation artifacts include:

```text
outputs/
└── yolo11m/
    └── E7_plus/
        ├── report.json
        ├── error_analysis.json
        └── visualizations/
```

---

# 🏆 Current Benchmark

The current benchmark model is based on **YOLO11m**.

### E7 Plus Test Performance

| Metric    |     Result |
| --------- | ---------: |
| mAP@50    | **0.9569** |
| mAP@75    | **0.9536** |
| mAP@50:95 | **0.8765** |

Test evaluation is performed on the fixed test split to maintain comparability between experiments.

---

# 🔍 Error Analysis

The project includes dedicated error-analysis workflows for investigating:

* False positives
* False negatives
* Per-class performance
* Difficult samples
* Class-specific failure patterns
* Similarity between difficult samples and training examples

These analyses are used to guide subsequent experiments rather than relying only on aggregate mAP values.

---

# 🚀 Training

Training is configuration-driven and separated from dataset, model, and evaluation logic.

Example workflow:

```text
Configuration
     │
     ▼
Dataset
     │
     ▼
Model
     │
     ▼
Trainer
     │
     ▼
Checkpoint
     │
     ▼
Evaluation
```

Experiment outputs are stored separately from source code and dataset files.

---

# 📦 Model Export

Trained checkpoints can be exported to deployment-oriented formats through the model exporter.

Supported export targets include:

* ONNX
* TensorRT
* TFLite
* TorchScript

The current deployment workflow primarily uses **ONNX**.

Example:

```text
YOLOModel
    │
    ▼
ModelExporter
    ├── ONNX
    ├── TensorRT
    ├── TFLite
    └── TorchScript
```

The current ONNX configuration uses a fixed `640×640` input size.

---

# ⚡ Deployment Benchmark

The project includes a deployment benchmarking workflow for comparing inference performance across runtimes and devices.

The benchmark measures:

* Latency
* FPS
* Warm-up behavior
* Runtime performance

Current benchmark comparisons include:

* PyTorch CPU
* PyTorch GPU
* ONNX CPU
* ONNX GPU

Example benchmark results for E7 Plus:

| Runtime | Device |    Latency |    FPS |
| ------- | ------ | ---------: | -----: |
| PyTorch | CPU    | 235.785 ms |  4.241 |
| PyTorch | GPU    |  24.139 ms | 41.428 |
| ONNX    | CPU    | 153.091 ms |  6.532 |
| ONNX    | GPU    |  28.288 ms | 35.351 |

These measurements are intended as deployment benchmarks for the current hardware and software environment rather than universal performance claims.

---

# 🌐 API

The project includes a FastAPI-based inference API for serving the exported detection model.

The API provides a deployment interface for:

* Image inference
* Model loading
* Detection results
* Structured responses

The API is designed to use the exported model rather than coupling inference directly to the training workflow.

---

# 🐳 Docker

Docker support is part of the deployment workflow and is intended to provide a reproducible runtime environment for the application.

The Docker setup separates the application environment from the local development environment.

---

# 📁 Project Structure

```text
smart-waste-detection/
│
├── src/
│   ├── data/                  # Dataset handling
│   ├── models/                # Model abstractions and implementations
│   ├── trainers/              # Training logic
│   ├── pipelines/             # Workflow orchestration
│   ├── evaluation/            # Evaluation and metrics
│   ├── export/                # Model export
│   ├── inference/             # Inference
│   ├── augmentation/          # Data augmentation
│   ├── logging/               # Logging utilities
│   │
│   └── e7/
│       ├── object_bank/
│       │   ├── extractor.py
│       │   ├── mask_generator.py
│       │   └── metadata.py
│       │
│       ├── context_engine/
│       │   ├── compositor.py
│       │   ├── scale.py
│       │   ├── lighting.py
│       │   ├── occlusion.py
│       │   ├── reflection.py
│       │   └── color_temperature.py
│       │
│       ├── dataset_builder.py
│       └── config.py
│
├── configs/                   # Experiment and runtime configurations
├── data/                      # Dataset
├── runs/                      # Training runs and checkpoints
├── outputs/                   # Evaluation and visualization artifacts
├── reports/                   # Benchmark and reporting artifacts
├── tests/                     # Tests and temporary test artifacts
├── docs/                      # Project documentation
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔁 Reproducibility

Reproducibility is an important design goal of the project.

Experiments are organized independently and use explicit configurations for:

* Dataset configuration
* Model configuration
* Training parameters
* Augmentation
* Evaluation
* Export
* Deployment benchmarking

Validation and test data are kept fixed across the primary experiments to ensure that performance differences can be attributed to changes in the training pipeline as consistently as possible.

---

# 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/samanglz/smart-waste-detection.git
cd smart-waste-detection

pip install -r requirements.txt
```

Python **3.11** is used for the current development environment.

---

# 🚀 Quick Start

Run the project entry point:

```bash
python -m main
```

Training, evaluation, export, and inference workflows can be executed through their respective project modules and configurations.

---

# 📚 Documentation

Additional documentation is available under:

```text
docs/
```

The documentation covers experiment methodology, architecture, evaluation, deployment, and reproducibility.

---

# 🔬 Research Direction

The project currently focuses on the intersection of:

* Object Detection
* Data-Centric AI
* Real-World Domain Adaptation
* Synthetic Data Generation
* Error Analysis
* Model Deployment
* Reproducible Computer Vision Experiments

The long-term architecture is designed to support additional detection frameworks and more advanced domain-adaptation methods.

---

# 📄 License

This project is licensed under the terms specified in the repository's license file.
