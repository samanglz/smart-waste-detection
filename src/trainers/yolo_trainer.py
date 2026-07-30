from ultralytics import YOLO

from src.trainers.base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):

    def __init__(self, config):

        self.config = config
        self.model = None

    def build_model(self):

        self.model = YOLO(self.config.MODEL_PATH)

    def train(self):

        self.model.train()

    def validate(self):

        self.model.val()

    def save_checkpoint(self):

        pass

    def load_checkpoint(self):

        pass