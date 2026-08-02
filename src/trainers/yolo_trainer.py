from src.trainers.base_trainer import BaseTrainer
from src.models.yolo.yolo_model import YOLOModel

class YOLOTrainer(BaseTrainer):

    def __init__(self, config, dataset):

        super().__init__(config)
        self.dataset = dataset

    def build_model(self):

        self.model = YOLOModel(self.config.MODEL_PATH)

    def train(self):
        
        self.model.train(

            data=self.dataset.get_dataset_config(),

            epochs=self.config.EPOCHS,

            imgsz=self.config.IMAGE_SIZE,

            batch=self.config.BATCH_SIZE,
        )
    def validate(self):

        self.model.val()

    def save_checkpoint(self):

        pass

    def load_checkpoint(self):

        pass