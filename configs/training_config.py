from src.augmentations import YOLOAugmentationConfig


EPOCHS = 50

BATCH_SIZE = 4

IMAGE_SIZE = 640

DEVICE = 0

WORKERS = 2

PATIENCE = 50

PROJECT_NAME = "smart_waste"

RUN_NAME = "yolo_baseline"


AUGMENTATION = YOLOAugmentationConfig()
