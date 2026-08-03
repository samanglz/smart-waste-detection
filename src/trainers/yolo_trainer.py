from src.trainers.base_trainer import BaseTrainer
from src.models.yolo.yolo_model import YOLOModel



class YOLOTrainer(BaseTrainer):

    def __init__(self, config, dataset):

        super().__init__(config)
        self.config = config 
        self.dataset = dataset

    def build_model(self):

        self.model = YOLOModel(self.config.MODEL_PATH)

    def train(self):
        
        aug_params = self.config.AUGMENTATION.to_dict(),
        
        self.model.train(
            

            data=self.dataset.get_dataset_config(),

            epochs=self.config.EPOCHS,

            imgsz=self.config.IMAGE_SIZE,

            batch=self.config.BATCH_SIZE,
            **aug_params,
        )
    def validate(self):

        self.model.val()

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        self.model.model.save(str(path))

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        self.model = YOLOModel(path)