from pathlib import Path
import argparse

from src.models.yolo.yolo_model import YOLOModel
from src.export import ModelExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained YOLO model to a deployment format."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained .pt checkpoint.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name, e.g. E2, E5, E6.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = args.checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    if checkpoint.suffix.lower() != ".pt":
        raise ValueError(
            f"Expected a .pt checkpoint, got: {checkpoint}"
        )

    output_dir = Path("exports") / args.experiment
    output_path = output_dir / "best.onnx"

    print("=" * 80)
    print("MODEL EXPORT")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint}")
    print(f"Experiment : {args.experiment}")
    print(f"Format     : ONNX")
    print(f"Image size : {args.imgsz}")
    print(f"Output     : {output_path}")
    print("=" * 80)

    model = YOLOModel(weight_path=checkpoint)

    exporter = ModelExporter(model=model)

    exported_path = exporter.export_onnx(
        output_path=output_path,
        imgsz=args.imgsz,
    )

    print()
    print("=" * 80)
    print("EXPORT SUCCESSFUL")
    print("=" * 80)
    print(f"ONNX model: {exported_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()