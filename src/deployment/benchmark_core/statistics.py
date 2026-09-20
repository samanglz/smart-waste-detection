from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RuntimeStatistics:
    """
    Statistical summary of inference latency measurements.

    All latency values are expressed in milliseconds.
    """

    runs: int

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float

    fps: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def calculate_statistics(
    timings: Sequence[float],
) -> RuntimeStatistics:
    """
    Calculate latency statistics from measured inference timings.

    Parameters
    ----------
    timings:
        Sequence of inference latency measurements in milliseconds.

    Returns
    -------
    RuntimeStatistics
        Statistical summary of the provided timings.

    Notes
    -----
    Standard deviation uses population standard deviation (ddof=0).
    This preserves compatibility with the historical benchmark.

    FPS is calculated as:

        FPS = 1000 / mean_ms
    """

    if not timings:
        raise ValueError(
            "timings must contain at least one measurement."
        )

    values = np.asarray(timings, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            "timings must be a one-dimensional sequence."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "timings must contain only finite numeric values."
        )

    if np.any(values < 0):
        raise ValueError(
            "timings cannot contain negative values."
        )

    mean_ms = float(np.mean(values))

    if mean_ms <= 0:
        fps = 0.0
    else:
        fps = 1000.0 / mean_ms

    return RuntimeStatistics(
        runs=int(values.size),
        mean_ms=mean_ms,
        median_ms=float(np.median(values)),
        std_ms=float(np.std(values)),
        min_ms=float(np.min(values)),
        max_ms=float(np.max(values)),
        p95_ms=float(np.percentile(values, 95)),
        fps=fps,
    )


def aggregate_per_image_means(
    image_results: Sequence[dict[str, Any]],
    runtime_key: str,
) -> RuntimeStatistics:
    """
    Aggregate benchmark results using per-image means.

    The benchmark protocol intentionally uses a two-stage aggregation:

    1. Calculate the mean latency for each image.
    2. Calculate the final benchmark statistics across those
       per-image means.

    This reproduces the aggregation methodology used by the
    historical E2 benchmark.

    Expected input format:

        [
            {
                "image": "img_001.jpg",
                "runtimes": {
                    "pytorch_gpu": {
                        "timings_ms": [...]
                    }
                }
            },
            ...
        ]

    Parameters
    ----------
    image_results:
        Per-image benchmark results.

    runtime_key:
        Runtime identifier whose timings should be aggregated.

    Returns
    -------
    RuntimeStatistics
        Statistics calculated from the mean latency of each image.
    """

    if not image_results:
        raise ValueError(
            "image_results must contain at least one image."
        )

    per_image_means: list[float] = []

    for index, image_result in enumerate(image_results):
        if not isinstance(image_result, dict):
            raise TypeError(
                f"image_results[{index}] must be a dictionary."
            )

        runtimes = image_result.get("runtimes")

        if not isinstance(runtimes, dict):
            raise ValueError(
                f"image_results[{index}] is missing a valid "
                "'runtimes' dictionary."
            )

        runtime_result = runtimes.get(runtime_key)

        if not isinstance(runtime_result, dict):
            raise ValueError(
                f"Runtime '{runtime_key}' is missing for "
                f"image_results[{index}]."
            )

        timings = runtime_result.get("timings_ms")

        if not timings:
            raise ValueError(
                f"Runtime '{runtime_key}' has no timings for "
                f"image_results[{index}]."
            )

        timings_array = np.asarray(
            timings,
            dtype=np.float64,
        )

        if timings_array.ndim != 1:
            raise ValueError(
                f"Timings for '{runtime_key}' in "
                f"image_results[{index}] must be one-dimensional."
            )

        if not np.all(np.isfinite(timings_array)):
            raise ValueError(
                f"Timings for '{runtime_key}' in "
                f"image_results[{index}] must be finite."
            )

        if np.any(timings_array < 0):
            raise ValueError(
                f"Timings for '{runtime_key}' in "
                f"image_results[{index}] cannot be negative."
            )

        per_image_means.append(
            float(np.mean(timings_array))
        )

    return calculate_statistics(per_image_means)


def statistics_to_dict(
    stats: RuntimeStatistics,
) -> dict[str, Any]:
    """
    Convert RuntimeStatistics into a JSON-serializable dictionary.
    """

    return stats.to_dict()