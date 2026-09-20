from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.deployment.benchmark_core.config import BenchmarkConfig
from src.deployment.benchmark_core.statistics import (
    RuntimeStatistics,
    calculate_statistics,
)
from src.deployment.runners.base_runner import BaseRunner
from src.deployment.utils.image_preprocess import preprocess_image_path


@dataclass(frozen=True)
class BenchmarkRun:
    image: str
    runtime: str
    timings_ms: list[float]
    statistics: RuntimeStatistics

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "runtime": self.runtime,
            "timings_ms": self.timings_ms,
            "statistics": self.statistics.to_dict(),
        }


class BenchmarkEngine:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def prepare_images(
        self,
        image_paths: list[str | Path],
    ) -> list[tuple[Path, np.ndarray]]:
        """
        Prepare image preprocessing once.

        Image decoding, resizing, letterboxing, RGB conversion,
        normalization, and tensor layout conversion happen outside
        the benchmark timing section.
        """
        prepared: list[tuple[Path, np.ndarray]] = []

        for image_path in image_paths:
            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

            if not path.is_file():
                raise ValueError(
                    f"Image path is not a file: {path}"
                )

            tensor = preprocess_image_path(
                str(path),
                size=self.config.image_size,
            )

            prepared.append((path, tensor))

        return prepared

    def prepare_runner_inputs(
        self,
        runner: BaseRunner,
        prepared_images: list[tuple[Path, np.ndarray]],
    ) -> list[tuple[Path, Any]]:
        """
        Prepare runtime-specific inputs outside timing.

        Examples:
            PyTorch:
                NumPy -> torch.Tensor -> CPU/GPU

            ONNX:
                NumPy -> contiguous float32 NumPy
        """
        if not prepared_images:
            raise ValueError(
                "prepared_images must not be empty."
            )

        prepared_inputs: list[tuple[Path, Any]] = []

        for image_path, input_array in prepared_images:
            prepared_input = runner.prepare_input(
                input_array
            )

            prepared_inputs.append(
                (image_path, prepared_input)
            )

        return prepared_inputs

    def _run_warmup(
        self,
        runner: BaseRunner,
        prepared_input: Any,
    ) -> None:
        """
        Execute warmup runs.

        Warmup is never included in measured timings.
        """
        for _ in range(self.config.warmup_runs):
            runner.synchronize()
            runner.predict_tensor(prepared_input)
            runner.synchronize()

    def _run_timed_inference(
        self,
        runner: BaseRunner,
        prepared_input: Any,
    ) -> float:
        """
        Measure inference only.

        Runtime-specific input preparation must already be complete.
        """
        runner.synchronize()

        start = time.perf_counter()

        runner.predict_tensor(prepared_input)

        runner.synchronize()

        end = time.perf_counter()

        return (end - start) * 1000.0

    def benchmark_runner(
        self,
        runner: BaseRunner,
        prepared_images: list[tuple[Path, np.ndarray]],
    ) -> list[BenchmarkRun]:
        """
        Benchmark one runtime over prepared images.

        Runtime-specific input preparation occurs once per image,
        outside the measured inference loop.
        """
        if not prepared_images:
            raise ValueError(
                "prepared_images must not be empty."
            )

        prepared_inputs = self.prepare_runner_inputs(
            runner,
            prepared_images,
        )

        results: list[BenchmarkRun] = []

        for image_path, prepared_input in prepared_inputs:
            self._run_warmup(
                runner,
                prepared_input,
            )

            timings_ms: list[float] = []

            for _ in range(self.config.benchmark_runs):
                latency_ms = self._run_timed_inference(
                    runner,
                    prepared_input,
                )

                timings_ms.append(latency_ms)

            statistics = calculate_statistics(
                timings_ms
            )

            results.append(
                BenchmarkRun(
                    image=image_path.name,
                    runtime=runner.runtime_name,
                    timings_ms=timings_ms,
                    statistics=statistics,
                )
            )

        return results

    def benchmark_runner_with_tensors(
        self,
        runner: BaseRunner,
        tensors: list[tuple[str, Any]],
    ) -> list[BenchmarkRun]:
        """
        Benchmark already-prepared runtime inputs.

        Useful for tests or callers that have already completed
        runtime-specific input preparation.
        """
        if not tensors:
            raise ValueError(
                "tensors must not be empty."
            )

        results: list[BenchmarkRun] = []

        for image_name, prepared_input in tensors:
            if not isinstance(image_name, str):
                raise TypeError(
                    "image_name must be a string."
                )

            if not image_name.strip():
                raise ValueError(
                    "image_name must not be empty."
                )

            self._run_warmup(
                runner,
                prepared_input,
            )

            timings_ms: list[float] = []

            for _ in range(self.config.benchmark_runs):
                timings_ms.append(
                    self._run_timed_inference(
                        runner,
                        prepared_input,
                    )
                )

            statistics = calculate_statistics(
                timings_ms
            )

            results.append(
                BenchmarkRun(
                    image=image_name,
                    runtime=runner.runtime_name,
                    timings_ms=timings_ms,
                    statistics=statistics,
                )
            )

        return results

    @staticmethod
    def aggregate_results(
        results: list[BenchmarkRun],
    ) -> RuntimeStatistics:
        """
        Aggregate image-level results according to Protocol 1.0.

        First calculate each image's mean latency.
        Then calculate aggregate statistics over those
        per-image means.
        """
        if not results:
            raise ValueError(
                "results must not be empty."
            )

        per_image_means = [
            result.statistics.mean_ms
            for result in results
        ]

        return calculate_statistics(
            per_image_means
        )