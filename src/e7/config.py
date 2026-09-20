"""
E7 experiment configuration.

This module contains configuration for the E7
Real-World Domain Adaptation pipeline.

E7 modifies only the training split.
Validation and test remain unchanged.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Dataset sources
# ---------------------------------------------------------------------------

# E2 train is the source/background dataset for E7.
#
# E2 train contains:
#   - original training images
#   - targeted augmentation from E2
#
# This dataset is used only as the training source for E7.
SOURCE_DATASET_DIR = (
    DATA_DIR / "e2_targeted_test"
)

SOURCE_TRAIN_DIR = (
    SOURCE_DATASET_DIR / "train"
)


# Original dataset is kept unchanged for evaluation.
#
# IMPORTANT:
# Validation and test must remain identical across experiments.
EVALUATION_DATASET_DIR = (
    DATA_DIR / "processed"
)

EVALUATION_VAL_DIR = (
    EVALUATION_DATASET_DIR / "val"
)

EVALUATION_TEST_DIR = (
    EVALUATION_DATASET_DIR / "test"
)


# ---------------------------------------------------------------------------
# E7 output dataset
# ---------------------------------------------------------------------------

E7_DATASET_DIR = (
    DATA_DIR / "E7_dataset"
)

E7_TRAIN_DIR = (
    E7_DATASET_DIR / "train"
)

E7_TRAIN_IMAGES_DIR = (
    E7_TRAIN_DIR / "images"
)

E7_TRAIN_LABELS_DIR = (
    E7_TRAIN_DIR / "labels"
)


# ---------------------------------------------------------------------------
# Object Bank
# ---------------------------------------------------------------------------

OBJECT_BANK_DIR = (
    DATA_DIR / "E7_object_bank"
)

OBJECT_BANK_MANIFEST = (
    OBJECT_BANK_DIR / "manifest.json"
)


# ---------------------------------------------------------------------------
# Background Bank
# ---------------------------------------------------------------------------

BACKGROUND_BANK_DIR = (
    DATA_DIR / "E7_background_bank"
)

BACKGROUND_BANK_MANIFEST = (
    BACKGROUND_BANK_DIR / "manifests" / "approved.json"
)

# Ratio of synthetic images generated on external
# Background Bank images.
#
# 0.0 = only E2 backgrounds
# 0.5 = 50% E2 / 50% external backgrounds
# 1.0 = only external backgrounds
BACKGROUND_BANK_RATIO = 0.5



# ---------------------------------------------------------------------------
# Dataset split names
# ---------------------------------------------------------------------------

TRAIN_SPLIT = "train"

VAL_SPLIT = "val"

TEST_SPLIT = "test"


# ---------------------------------------------------------------------------
# E7 generation
# ---------------------------------------------------------------------------

SEED = 42

COPY_ORIGINAL_TRAIN = True

# None = process the complete E2 training set.
#
# During development this can temporarily be changed to
# a small value such as 5 or 20.
MAX_IMAGES = None


# ---------------------------------------------------------------------------
# Object generation strategy
# ---------------------------------------------------------------------------

# Object area ratio:
#
#   small  : area < 0.05  -> 3 generations
#   medium : area < 0.20  -> 2 generations
#   large  : area >= 0.20 -> 1 generation
#
# These values are consumed by AreaBasedGenerationStrategy.

SMALL_AREA_THRESHOLD = 0.05

MEDIUM_AREA_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------

MIN_OBJECT_HEIGHT_RATIO = 0.10

MAX_OBJECT_HEIGHT_RATIO = 0.35

ENABLE_REFLECTION = True

ENABLE_OCCLUSION = False


# ---------------------------------------------------------------------------
# E7 experiment metadata
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "E7"

EXPERIMENT_DESCRIPTION = (
    "Real-World Domain Adaptation"
)

BUILD_SUMMARY_FILENAME = (
    "build_summary.json"
)

DATASET_YAML_FILENAME = (
    "data.yaml"
)