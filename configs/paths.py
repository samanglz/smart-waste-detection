from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

FINAL_DATASET_DIR = DATA_DIR / "processed"

TRAIN_DIR = FINAL_DATASET_DIR / "train"

VAL_DIR = FINAL_DATASET_DIR / "val"

TEST_DIR = FINAL_DATASET_DIR / "test"

RUNS_DIR = PROJECT_ROOT / "runs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / " yolo"