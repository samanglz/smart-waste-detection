from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from src.deployment.benchmark_core import DeploymentBenchmark


class BenchmarkReportGenerator:
    """
    Generate JSON, CSV and TXT benchmark reports.

    This class is intentionally separated from benchmark.py.
    Benchmark performs measurements.
    BenchmarkReportGenerator serializes results.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, result: dict) -> Path:
        path = self.output_dir / "benchmark_report.json"

        payload = {
            "created_at": datetime.now().isoformat(),
            **result,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return path

    def save_csv(self, result: dict) -> Path:
        path = self.output_dir / "benchmark_summary.csv"

        rows = [
            ["metric", "pytorch", "onnx"],
            ["mean_ms", result["pytorch"]["mean_ms"], result["onnx"]["mean_ms"]],
            ["median_ms", result["pytorch"]["median_ms"], result["onnx"]["median_ms"]],
            ["std_ms", result["pytorch"]["std_ms"], result["onnx"]["std_ms"]],
            ["min_ms", result["pytorch"]["min_ms"], result["onnx"]["min_ms"]],
            ["max_ms", result["pytorch"]["max_ms"], result["onnx"]["max_ms"]],
            ["p95_ms", result["pytorch"]["p95_ms"], result["onnx"]["p95_ms"]],
            ["fps", result["pytorch"]["fps"], result["onnx"]["fps"]],
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return path

    def save_txt(self, result: dict) -> Path:
        path = self.output_dir / "benchmark_report.txt"

        c = result["configuration"]
        p = result["pytorch"]
        o = result["onnx"]
        s = result["speedup"]

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("DEPLOYMENT BENCHMARK REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Created : {datetime.now().isoformat()}\n")
            f.write(f"Device  : {c['device']}\n")
            f.write(f"Images  : {c['num_images']}\n")
            f.write(f"ImgSize : {c['imgsz']}\n\n")

            f.write("PYTORCH\n")
            f.write("-" * 30 + "\n")
            for k, v in p.items():
                f.write(f"{k:12}: {v}\n")

            f.write("\nONNX\n")
            f.write("-" * 30 + "\n")
            for k, v in o.items():
                f.write(f"{k:12}: {v}\n")

            f.write("\nCOMPARISON\n")
            f.write("-" * 30 + "\n")
            f.write(f"Speedup : {s['onnx_vs_pytorch']:.4f}x\n")
            f.write(f"Result  : {s['interpretation']}\n")

        return path

    def generate(self, result: dict) -> dict:
        json_path = self.save_json(result)
        csv_path = self.save_csv(result)
        txt_path = self.save_txt(result)

        return {
            "json": json_path,
            "csv": csv_path,
            "txt": txt_path,
        }


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", required=True)

    parser.add_argument("--pytorch-model", required=True)

    parser.add_argument("--onnx-model", required=True)

    parser.add_argument("--image-dir", required=True)

    parser.add_argument("--device", default="cpu")

    parser.add_argument("--num-images", type=int, default=30)

    parser.add_argument("--imgsz", type=int, default=640)

    return parser


def main():
    args = build_parser().parse_args()

    benchmark = DeploymentBenchmark(
        pytorch_model_path=Path(args.pytorch_model),
        onnx_model_path=Path(args.onnx_model),
        image_dir=Path(args.image_dir),
        imgsz=args.imgsz,
        num_images=args.num_images,
        device=args.device,
    )

    result = benchmark.run()

    output_dir = (
        Path("reports")
        / "deployment"
        / args.experiment
    )

    generator = BenchmarkReportGenerator(output_dir)

    files = generator.generate(result)

    print("\n" + "=" * 70)
    print("REPORTS GENERATED")
    print("=" * 70)

    for name, path in files.items():
        print(f"{name.upper():5} : {path}")


if __name__ == "__main__":
    main()