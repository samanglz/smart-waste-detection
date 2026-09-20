from pathlib import Path

import torch

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


SAM2_PACKAGE_DIR = Path(
    r"C:\Users\ASUS\AppData\Local\Programs\Python\Python311\Lib\site-packages\sam2"
)

CHECKPOINT = (
    Path(r"C:\Users\ASUS\Desktop\smart-waste-detection")
    / "sam2"
    / "checkpoints"
    / "sam2.1_hiera_small.pt"
)

CONFIG_DIR = SAM2_PACKAGE_DIR / "configs"
CONFIG_NAME = "sam2.1/sam2.1_hiera_s.yaml"


def main():
    print("Loading SAM 2.1 Small...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM 2.1 checkpoint not found: {CHECKPOINT}"
        )

    config_path = CONFIG_DIR / CONFIG_NAME

    if not config_path.exists():
        raise FileNotFoundError(
            f"SAM 2.1 config not found: {config_path}"
        )

    print(f"Config: {config_path}")
    print(f"Checkpoint: {CHECKPOINT}")

    # build_sam2() initializes Hydra itself.
    # We only clear any previous Hydra state.
    GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base="1.2",
        config_dir=str(CONFIG_DIR),
    ):
        model = build_sam2(
            config_file=CONFIG_NAME,
            ckpt_path=str(CHECKPOINT),
            device=device,
        )

    predictor = SAM2ImagePredictor(model)

    print("SAM 2.1 model: OK")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Predictor: OK")


if __name__ == "__main__":
    main()