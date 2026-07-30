from src.trainers.base_trainer import BaseTrainer
from src.models.yolo.model import YOLOModel

class YOLOTrainer(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)

    def build_model(self):

        self.model = YOLOModel(self.config.MODEL_PATH)

    def train(self):

        self.model.train()

    def validate(self):

        self.model.val()

    def save_checkpoint(self):

        pass

    def load_checkpoint(self):

        pass