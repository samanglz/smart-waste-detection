from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkMetadata:
    timestamp_utc: str
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetInfo:
    split: str
    image_dir: str
    available_images: int
    selected_images: int
    seed: int
    filenames: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeResult:
    statistics: dict[str, Any]
    per_image: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalBenchmarkResult:
    """
    Canonical deployment benchmark result.

    This schema describes benchmark performance and provenance only.
    Model accuracy and error analysis are intentionally kept outside
    the deployment benchmark.
    """

    benchmark: BenchmarkMetadata
    models: dict[str, str]
    artifacts: dict[str, dict[str, Any]]
    dataset: DatasetInfo
    environment: dict[str, Any]
    runtimes: dict[str, dict[str, Any]]
    results: dict[str, RuntimeResult]
    speedup: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark.to_dict(),
            "models": self.models,
            "artifacts": self.artifacts,
            "dataset": self.dataset.to_dict(),
            "environment": self.environment,
            "runtimes": self.runtimes,
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            "speedup": self.speedup,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "CanonicalBenchmarkResult":
        """
        Validate and reconstruct a canonical benchmark result
        from a dictionary.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "Canonical benchmark result must be a dictionary."
            )

        required_keys = {
            "benchmark",
            "models",
            "artifacts",
            "dataset",
            "environment",
            "runtimes",
            "results",
            "speedup",
        }

        missing = required_keys - data.keys()

        if missing:
            raise ValueError(
                "Canonical benchmark result is missing keys: "
                + ", ".join(sorted(missing))
            )

        benchmark_data = data["benchmark"]

        if not isinstance(benchmark_data, dict):
            raise TypeError(
                "'benchmark' must be a dictionary."
            )

        dataset_data = data["dataset"]

        if not isinstance(dataset_data, dict):
            raise TypeError(
                "'dataset' must be a dictionary."
            )

        results_data = data["results"]

        if not isinstance(results_data, dict):
            raise TypeError(
                "'results' must be a dictionary."
            )

        results = {}

        for runtime_name, runtime_data in results_data.items():
            if not isinstance(runtime_data, dict):
                raise TypeError(
                    f"Result for runtime '{runtime_name}' "
                    "must be a dictionary."
                )

            if "statistics" not in runtime_data:
                raise ValueError(
                    f"Runtime '{runtime_name}' is missing "
                    "'statistics'."
                )

            if "per_image" not in runtime_data:
                raise ValueError(
                    f"Runtime '{runtime_name}' is missing "
                    "'per_image'."
                )

            results[runtime_name] = RuntimeResult(
                statistics=runtime_data["statistics"],
                per_image=runtime_data["per_image"],
            )

        return cls(
            benchmark=BenchmarkMetadata(
                timestamp_utc=benchmark_data["timestamp_utc"],
                protocol=benchmark_data["protocol"],
            ),
            models=data["models"],
            artifacts=data["artifacts"],
            dataset=DatasetInfo(
                split=dataset_data["split"],
                image_dir=dataset_data["image_dir"],
                available_images=dataset_data["available_images"],
                selected_images=dataset_data["selected_images"],
                seed=dataset_data["seed"],
                filenames=dataset_data["filenames"],
            ),
            environment=data["environment"],
            runtimes=data["runtimes"],
            results=results,
            speedup=data["speedup"],
        )