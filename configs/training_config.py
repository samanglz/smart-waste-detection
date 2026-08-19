from src.augmentations import YOLOAugmentationConfig


EPOCHS = 50

BATCH_SIZE = 2

ACCUMULATE = 4

EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATE  

IMAGE_SIZE = 640

DEVICE = 0

WORKERS = 5

PATIENCE = 50

PROJECT_NAME = "runs/yolo11m"

RUN_NAME = "detect"

GRADIENT_CLIP = 10.0

OPTIMIZER = "AdamW"



AUGMENTATION = YOLOAugmentationConfig()


