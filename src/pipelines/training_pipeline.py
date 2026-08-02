from src.logging.logger import get_logger



class TrainingPipeline:

    def __init__(self, config, trainer):

        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.trainer = trainer

    def build(self):

        self.logger.info("Building model...")
        self.trainer.build_model()

    def train(self):

        self.logger.info("Training started...")
        self.trainer.train()

    def validate(self):

        self.logger.info("Validation started...")
        self.trainer.validate()

    def run(self):

        self.build()
        self.train()
        self.validate()

        self.logger.info("Pipeline finished.")