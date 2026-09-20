from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.deployment.benchmark_core.artifacts import get_artifact_info
from src.deployment.benchmark_core.config import BenchmarkConfig
from src.deployment.benchmark_core.engine import BenchmarkEngine
from src.deployment.benchmark_core.environment import collect_environment_info
from src.deployment.benchmark_core.schema import (
    BenchmarkMetadata,
    CanonicalBenchmarkResult,
    DatasetInfo,
    RuntimeResult,
)
from src.deployment.runners.onnx_runner import ONNXRunner
from src.deployment.runners.pytorch_runner import PyTorchRunner
from src.models.yolo.yolo_model import YOLOModel


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

DEFAULT_IMAGE_SIZE = 640
DEFAULT_NUM_IMAGES = 30
DEFAULT_WARMUP_RUNS = 10
DEFAULT_BENCHMARK_RUNS = 50
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("reports/deployment_benchmark")


class DeploymentBenchmark:
    """
    Orchestrates deployment benchmarking across supported runtimes.

    Runtime-specific inference logic belongs to Runner classes.
    Benchmark timing and statistics belong to BenchmarkEngine.
    """

    def __init__(
        self,
        pytorch_model_path: str | Path,
        onnx_model_path: str | Path,
        image_dir: str | Path,
        config: BenchmarkConfig,
        output_dir: str | Path = DEFAULT_OUTPUT,
    ) -> None:
        self.pytorch_model_path = Path(pytorch_model_path)
        self.onnx_model_path = Path(onnx_model_path)
        self.image_dir = Path(image_dir)
        self.config = config
        self.output_dir = Path(output_dir)

        self._validate_configuration()

        self.environment = collect_environment_info()

        self.pytorch_cpu_runner: PyTorchRunner | None = None
        self.pytorch_gpu_runner: PyTorchRunner | None = None
        self.onnx_cpu_runner: ONNXRunner | None = None
        self.onnx_gpu_runner: ONNXRunner | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if not self.pytorch_model_path.exists():
            raise FileNotFoundError(
                f"PyTorch model not found: {self.pytorch_model_path}"
            )

        if not self.pytorch_model_path.is_file():
            raise ValueError(
                f"PyTorch model is not a file: {self.pytorch_model_path}"
            )

        if not self.onnx_model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_model_path}"
            )

        if not self.onnx_model_path.is_file():
            raise ValueError(
                f"ONNX model is not a file: {self.onnx_model_path}"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.image_dir.is_dir():
            raise ValueError(
                f"Image path is not a directory: {self.image_dir}"
            )

        if self.config.batch_size != 1:
            raise ValueError(
                "Benchmark Protocol 1.0 requires batch_size=1."
            )

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def discover_images(self) -> list[Path]:
        """Discover all supported images in deterministic sorted order."""
        images = [
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        images.sort(key=lambda path: path.name)

        if not images:
            raise ValueError(
                f"No supported images found in: {self.image_dir}"
            )

        return images

    def select_images(self) -> list[Path]:
        """
        Select a deterministic subset of images using the protocol seed.
        """
        images = self.discover_images()

        if self.config.num_images > len(images):
            raise ValueError(
                f"Requested {self.config.num_images} images, "
                f"but only {len(images)} images are available."
            )

        rng = random.Random(self.config.seed)
        selected = rng.sample(images, self.config.num_images)

        selected.sort(key=lambda path: path.name)

        return selected

    # ------------------------------------------------------------------
    # Runtime initialization
    # ------------------------------------------------------------------

    def _create_pytorch_runners(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the GPU benchmark, "
                "but torch.cuda.is_available() is False."
            )

        print("\nInitializing PyTorch CPU runner...")

        cpu_model = YOLOModel(self.pytorch_model_path)

        self.pytorch_cpu_runner = PyTorchRunner(
            model=cpu_model.model.model,
            device="cpu",
            model_path=self.pytorch_model_path,
        )

        print("PyTorch CPU runner ready.")

        print("\nInitializing PyTorch CUDA runner...")

        gpu_model = YOLOModel(self.pytorch_model_path)

        self.pytorch_gpu_runner = PyTorchRunner(
            model=gpu_model.model.model,
            device="cuda",
            model_path=self.pytorch_model_path,
        )

        print("PyTorch CUDA runner ready.")

    def _create_onnx_runners(self) -> None:
        print("\nInitializing ONNX CPU runner...")

        self.onnx_cpu_runner = ONNXRunner(
            model_path=self.onnx_model_path,
            providers=["CPUExecutionProvider"],
            imgsz=self.config.image_size,
        )

        print(
            "ONNX CPU providers:",
            self.onnx_cpu_runner.actual_providers,
        )

        if "CPUExecutionProvider" not in (
            self.onnx_cpu_runner.actual_providers or []
        ):
            raise RuntimeError(
                "ONNX CPU runner did not initialize "
                "CPUExecutionProvider."
            )

        print("\nInitializing ONNX CUDA runner...")

        self.onnx_gpu_runner = ONNXRunner(
            model_path=self.onnx_model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            imgsz=self.config.image_size,
        )

        print(
            "ONNX CUDA providers:",
            self.onnx_gpu_runner.actual_providers,
        )

        if "CUDAExecutionProvider" not in (
            self.onnx_gpu_runner.actual_providers or []
        ):
            raise RuntimeError(
                "ONNX CUDA runner did not initialize "
                "CUDAExecutionProvider."
            )

    def initialize_runtimes(self) -> None:
        self._create_pytorch_runners()
        self._create_onnx_runners()

    # ------------------------------------------------------------------
    # Benchmark helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_benchmark_runs(
        runs: list[Any],
    ) -> list[dict[str, Any]]:
        return [run.to_dict() for run in runs]

    @staticmethod
    def _serialize_statistics(
        statistics: Any,
    ) -> dict[str, Any]:
        return statistics.to_dict()

    @staticmethod
    def _calculate_speedup(
        baseline_ms: float,
        comparison_ms: float,
    ) -> float:
        if comparison_ms <= 0:
            return 0.0

        return baseline_ms / comparison_ms

    def _build_runtime_result(
        self,
        runs: list[Any],
        aggregate_statistics: Any,
    ) -> RuntimeResult:
        return RuntimeResult(
            statistics=self._serialize_statistics(
                aggregate_statistics
            ),
            per_image=self._serialize_benchmark_runs(runs),
        )

    # ------------------------------------------------------------------
    # Main benchmark
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        print("=" * 80)
        print("DEPLOYMENT BENCHMARK")
        print("=" * 80)

        print(f"Protocol:       {self.config.protocol_version}")
        print(f"Split:          {self.config.split}")
        print(f"Image size:     {self.config.image_size}")
        print(f"Images:         {self.config.num_images}")
        print(f"Warmup runs:    {self.config.warmup_runs}")
        print(f"Benchmark runs: {self.config.benchmark_runs}")
        print(f"Batch size:     {self.config.batch_size}")
        print(f"Seed:           {self.config.seed}")
        print(f"Timing:         {self.config.timing_mode}")
        print("=" * 80)

        selected_images = self.select_images()

        print(
            f"\nSelected {len(selected_images)} images "
            f"from {self.image_dir}"
        )

        self.initialize_runtimes()

        assert self.pytorch_cpu_runner is not None
        assert self.pytorch_gpu_runner is not None
        assert self.onnx_cpu_runner is not None
        assert self.onnx_gpu_runner is not None

        engine = BenchmarkEngine(self.config)

        print("\nPreparing images...")

        prepared_images = engine.prepare_images(
            selected_images
        )

        print("Image preparation complete.")

        # --------------------------------------------------------------
        # PyTorch CPU
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("Benchmarking PyTorch CPU")
        print("-" * 80)

        pytorch_cpu_runs = engine.benchmark_runner(
            self.pytorch_cpu_runner,
            prepared_images,
        )

        pytorch_cpu_stats = engine.aggregate_results(
            pytorch_cpu_runs
        )

        # --------------------------------------------------------------
        # PyTorch GPU
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("Benchmarking PyTorch GPU")
        print("-" * 80)

        pytorch_gpu_runs = engine.benchmark_runner(
            self.pytorch_gpu_runner,
            prepared_images,
        )

        pytorch_gpu_stats = engine.aggregate_results(
            pytorch_gpu_runs
        )

        # --------------------------------------------------------------
        # ONNX CPU
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("Benchmarking ONNX CPU")
        print("-" * 80)

        onnx_cpu_runs = engine.benchmark_runner(
            self.onnx_cpu_runner,
            prepared_images,
        )

        onnx_cpu_stats = engine.aggregate_results(
            onnx_cpu_runs
        )

        # --------------------------------------------------------------
        # ONNX GPU
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("Benchmarking ONNX GPU")
        print("-" * 80)

        onnx_gpu_runs = engine.benchmark_runner(
            self.onnx_gpu_runner,
            prepared_images,
        )

        onnx_gpu_stats = engine.aggregate_results(
            onnx_gpu_runs
        )

        # --------------------------------------------------------------
        # Runtime results
        # --------------------------------------------------------------

        results = {
            "pytorch_cpu": self._build_runtime_result(
                pytorch_cpu_runs,
                pytorch_cpu_stats,
            ),
            "pytorch_gpu": self._build_runtime_result(
                pytorch_gpu_runs,
                pytorch_gpu_stats,
            ),
            "onnx_cpu": self._build_runtime_result(
                onnx_cpu_runs,
                onnx_cpu_stats,
            ),
            "onnx_gpu": self._build_runtime_result(
                onnx_gpu_runs,
                onnx_gpu_stats,
            ),
        }

        # --------------------------------------------------------------
        # Speedups
        # --------------------------------------------------------------

        speedup = {
            "pytorch_gpu_vs_cpu": self._calculate_speedup(
                pytorch_cpu_stats.mean_ms,
                pytorch_gpu_stats.mean_ms,
            ),
            "onnx_gpu_vs_cpu": self._calculate_speedup(
                onnx_cpu_stats.mean_ms,
                onnx_gpu_stats.mean_ms,
            ),
            "onnx_cpu_vs_pytorch_cpu": self._calculate_speedup(
                pytorch_cpu_stats.mean_ms,
                onnx_cpu_stats.mean_ms,
            ),
            "onnx_gpu_vs_pytorch_gpu": self._calculate_speedup(
                pytorch_gpu_stats.mean_ms,
                onnx_gpu_stats.mean_ms,
            ),
        }

        # --------------------------------------------------------------
        # Provenance
        # --------------------------------------------------------------

        model_artifacts = {
            "pytorch": get_artifact_info(
                self.pytorch_model_path
            ).to_dict(),
            "onnx": get_artifact_info(
                self.onnx_model_path
            ).to_dict(),
        }

        selected_filenames = [
            path.name
            for path in selected_images
        ]

        # --------------------------------------------------------------
        # Canonical result
        # --------------------------------------------------------------

        canonical_result = CanonicalBenchmarkResult(
            benchmark=BenchmarkMetadata(
                timestamp_utc=datetime.now(
                    timezone.utc
                ).isoformat(),
                protocol=self.config.to_dict(),
            ),
            models={
                "pytorch": str(
                    self.pytorch_model_path.resolve()
                ),
                "onnx": str(
                    self.onnx_model_path.resolve()
                ),
            },
            artifacts=model_artifacts,
            dataset=DatasetInfo(
                split=self.config.split,
                image_dir=str(
                    self.image_dir.resolve()
                ),
                available_images=len(
                    self.discover_images()
                ),
                selected_images=len(
                    selected_images
                ),
                seed=self.config.seed,
                filenames=selected_filenames,
            ),
            environment=self.environment.to_dict(),
            runtimes={
                "pytorch_cpu": (
                    self.pytorch_cpu_runner.metadata()
                ),
                "pytorch_gpu": (
                    self.pytorch_gpu_runner.metadata()
                ),
                "onnx_cpu": (
                    self.onnx_cpu_runner.metadata()
                ),
                "onnx_gpu": (
                    self.onnx_gpu_runner.metadata()
                ),
            },
            results=results,
            speedup=speedup,
        )

        return canonical_result.to_dict()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(
        self,
        result: dict[str, Any],
    ) -> Path:
        output_path = self.output_dir

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
        
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path


# ==========================================================================
# CLI
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deployment benchmark for PyTorch and ONNX."
        )
    )

    parser.add_argument(
        "--pytorch-model",
        required=True,
        help="Path to PyTorch checkpoint (.pt).",
    )

    parser.add_argument(
        "--onnx-model",
        required=True,
        help="Path to ONNX model (.onnx).",
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing benchmark images.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=(
            f"Input image size. Default: "
            f"{DEFAULT_IMAGE_SIZE}"
        ),
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=(
            f"Number of images. Default: "
            f"{DEFAULT_NUM_IMAGES}"
        ),
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=(
            f"Warmup runs. Default: "
            f"{DEFAULT_WARMUP_RUNS}"
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_BENCHMARK_RUNS,
        help=(
            f"Measured runs per image. "
            f"Default: {DEFAULT_BENCHMARK_RUNS}"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed. Default: "
            f"{DEFAULT_SEED}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            f"Output JSON file. Default: "
            f"{DEFAULT_OUTPUT / 'E2_benchmark_results.json'}"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = BenchmarkConfig(
        image_size=args.imgsz,
        num_images=args.num_images,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.runs,
        seed=args.seed,
    )

    benchmark = DeploymentBenchmark(
        pytorch_model_path=args.pytorch_model,
        onnx_model_path=args.onnx_model,
        image_dir=args.image_dir,
        config=config,
        output_dir=args.output,
    )

    result = benchmark.run()
    output_path = benchmark.save_results(result)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_path}")

    for runtime_name, runtime_result in (
        result["results"].items()
    ):
        stats = runtime_result["statistics"]

        print(
            f"{runtime_name:24s} "
            f"mean={stats['mean_ms']:.3f} ms "
            f"FPS={stats['fps']:.3f}"
        )

    print("\nSpeedups:")

    for name, value in result["speedup"].items():
        print(
            f"{name:32s}: {value:.3f}x"
        )


if __name__ == "__main__":
    main()