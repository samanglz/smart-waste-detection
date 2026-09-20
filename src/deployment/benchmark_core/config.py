from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Immutable configuration for the deployment benchmark protocol.

    This configuration is runtime-agnostic. It defines HOW a benchmark
    should be conducted, not which inference runtime is being used.

    Protocol Version 1.0 is intentionally aligned with the historical
    E2 deployment benchmark so that the future canonical E2 benchmark
    can be compared fairly with the historical result.
    """

    protocol_version: str = "1.0"

    image_size: int = 640
    num_images: int = 30

    warmup_runs: int = 10
    benchmark_runs: int = 50

    batch_size: int = 1
    seed: int = 42

    split: str = "val"

    timing_mode: str = "inference_only"
    preprocessing_timed: bool = False
    postprocessing_timed: bool = False

    aggregation: str = "per_image_mean"

    def __post_init__(self) -> None:
        """Validate benchmark protocol configuration."""

        if not self.protocol_version.strip():
            raise ValueError(
                "protocol_version must be a non-empty string."
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        if self.num_images <= 0:
            raise ValueError(
                "num_images must be greater than zero."
            )

        if self.warmup_runs < 0:
            raise ValueError(
                "warmup_runs cannot be negative."
            )

        if self.benchmark_runs <= 0:
            raise ValueError(
                "benchmark_runs must be greater than zero."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if not self.split.strip():
            raise ValueError(
                "split must be a non-empty string."
            )

        if not self.timing_mode.strip():
            raise ValueError(
                "timing_mode must be a non-empty string."
            )

        if not self.aggregation.strip():
            raise ValueError(
                "aggregation must be a non-empty string."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration into a JSON-serializable dictionary.
        """

        return asdict(self)