from src.trainers.yolo_trainer import YOLOTrainer


class TrainingPipeline:

    def __init__(self, config):

        self.config = config
        self.trainer = YOLOTrainer(config)

    def run(self):

        self.trainer.build_model()

        self.trainer.train()

        self.trainer.validate()