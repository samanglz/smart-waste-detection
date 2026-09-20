from pathlib import Path
import os


class Settings:
    MODEL_PATH = Path(
        os.getenv("MODEL_PATH", "exports/E7/best.onnx")
    )

    INPUT_SIZE = int(
        os.getenv("INPUT_SIZE", "640")
    )

    CONF_THRESHOLD = float(
        os.getenv("CONF_THRESHOLD", "0.25")
    )

    IOU_THRESHOLD = float(
        os.getenv("IOU_THRESHOLD", "0.45")
    )


settings = Settings()