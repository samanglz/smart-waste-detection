from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


RUNTIME_ORDER = [
    "pytorch_cpu",
    "pytorch_gpu",
    "onnx_cpu",
    "onnx_gpu",
]


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object: {path}")

    return data


def extract_statistics(runtime_data: Any) -> dict[str, float]:
    """
    Extract mean latency and FPS from a runtime result.

    Supports both:

        {
            "mean_ms": ...,
            "fps": ...
        }

    and:

        {
            "statistics": {
                "mean_ms": ...,
                "fps": ...
            }
        }
    """

    if not isinstance(runtime_data, dict):
        raise TypeError("Runtime result must be a JSON object.")

    statistics = runtime_data.get("statistics", runtime_data)

    if not isinstance(statistics, dict):
        raise TypeError("Runtime statistics must be a JSON object.")

    mean_ms = statistics.get("mean_ms")
    fps = statistics.get("fps")

    if mean_ms is None:
        raise KeyError(
            "Could not find 'mean_ms' in runtime result."
        )

    mean_ms = float(mean_ms)

    if fps is None:
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    else:
        fps = float(fps)

    return {
        "mean_ms": mean_ms,
        "fps": fps,
    }

def find_runtime_results(
    data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Find runtime benchmark results from both:

    Historical schema:
        results:
            pytorch:
                cpu:
                gpu:
            onnx:
                cpu:
                gpu:

    Canonical schema:
        results:
            pytorch_cpu:
            pytorch_gpu:
            onnx_cpu:
            onnx_gpu:
    """

    results = data.get("results")

    if not isinstance(results, dict):
        raise ValueError(
            "Benchmark JSON does not contain a valid 'results' object."
        )

    extracted: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Historical E2 schema
    # ------------------------------------------------------------------
    #
    # results = {
    #     "pytorch": {
    #         "cpu": {...},
    #         "gpu": {...},
    #     },
    #     "onnx": {
    #         "cpu": {...},
    #         "gpu": {...},
    #     },
    # }
    #
    if (
        isinstance(results.get("pytorch"), dict)
        or isinstance(results.get("onnx"), dict)
    ):
        for runtime in ("pytorch", "onnx"):

            runtime_data = results.get(runtime)

            if not isinstance(runtime_data, dict):
                continue

            for device in ("cpu", "gpu"):

                device_data = runtime_data.get(device)

                if not isinstance(device_data, dict):
                    continue

                try:
                    extracted[
                        f"{runtime}_{device}"
                    ] = extract_statistics(device_data)

                except (
                    TypeError,
                    KeyError,
                    ValueError,
                ):
                    continue

        if extracted:
            return extracted

    # ------------------------------------------------------------------
    # Canonical schema
    # ------------------------------------------------------------------
    #
    # results = {
    #     "pytorch_cpu": {...},
    #     "pytorch_gpu": {...},
    #     "onnx_cpu": {...},
    #     "onnx_gpu": {...},
    # }
    #
    for runtime in RUNTIME_ORDER:

        if runtime not in results:
            continue

        try:
            extracted[runtime] = extract_statistics(
                results[runtime]
            )

        except (
            TypeError,
            KeyError,
            ValueError,
        ):
            continue

    if extracted:
        return extracted

    raise ValueError(
        "Could not find benchmark runtime results. "
        "Expected either the historical schema "
        "(results.pytorch.cpu / gpu, results.onnx.cpu / gpu) "
        "or the canonical schema "
        "(results.pytorch_cpu / pytorch_gpu / onnx_cpu / onnx_gpu)."
    )


def environment_summary(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Extract useful environment information."""

    environment = data.get("environment", {})

    if not isinstance(environment, dict):
        return {}

    keys = [
        "python_version",
        "platform",
        "os",
        "pytorch_version",
        "onnxruntime_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
        "gpu_memory_gb",
    ]

    return {
        key: environment[key]
        for key in keys
        if key in environment
    }


def benchmark_summary(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Extract benchmark protocol information."""

    benchmark = data.get("benchmark", {})

    if not isinstance(benchmark, dict):
        return {}

    protocol = benchmark.get("protocol", {})

    if not isinstance(protocol, dict):
        protocol = {}

    return {
        "timestamp_utc": benchmark.get("timestamp_utc"),
        "protocol_version": protocol.get(
            "protocol_version"
        ),
        "image_size": protocol.get("image_size"),
        "num_images": protocol.get("num_images"),
        "warmup_runs": protocol.get("warmup_runs"),
        "benchmark_runs": protocol.get("benchmark_runs"),
        "batch_size": protocol.get("batch_size"),
        "seed": protocol.get("seed"),
        "split": protocol.get("split"),
        "timing_mode": protocol.get("timing_mode"),
        "aggregation": protocol.get("aggregation"),
    }


def compare(
    historical: dict[str, dict[str, float]],
    canonical: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Compare historical and canonical benchmark results."""

    rows = []

    for runtime in RUNTIME_ORDER:

        if runtime not in historical:
            continue

        if runtime not in canonical:
            continue

        historical_mean = historical[runtime]["mean_ms"]
        canonical_mean = canonical[runtime]["mean_ms"]

        historical_fps = historical[runtime]["fps"]
        canonical_fps = canonical[runtime]["fps"]

        latency_change_ms = (
            canonical_mean - historical_mean
        )

        if historical_mean != 0:
            latency_change_pct = (
                latency_change_ms
                / historical_mean
                * 100.0
            )
        else:
            latency_change_pct = None

        fps_change = (
            canonical_fps - historical_fps
        )

        if historical_fps != 0:
            fps_change_pct = (
                fps_change
                / historical_fps
                * 100.0
            )
        else:
            fps_change_pct = None

        rows.append(
            {
                "runtime": runtime,
                "historical_mean_ms": historical_mean,
                "canonical_mean_ms": canonical_mean,
                "latency_change_ms": latency_change_ms,
                "latency_change_pct": latency_change_pct,
                "historical_fps": historical_fps,
                "canonical_fps": canonical_fps,
                "fps_change": fps_change,
                "fps_change_pct": fps_change_pct,
            }
        )

    return rows


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save comparison table as CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "No comparable runtime results found."
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    path: Path,
    historical_path: Path,
    canonical_path: Path,
    historical_data: dict[str, Any],
    canonical_data: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Save concise comparison summary as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "comparison": {
            "historical_file": str(
                historical_path
            ),
            "canonical_file": str(
                canonical_path
            ),
            "historical_protocol": benchmark_summary(
                historical_data
            ),
            "canonical_protocol": benchmark_summary(
                canonical_data
            ),
        },
        "historical_environment": environment_summary(
            historical_data
        ),
        "canonical_environment": environment_summary(
            canonical_data
        ),
        "results": rows,
        "interpretation": (
            "The historical and canonical E2 benchmarks "
            "were executed under different "
            "software/hardware environments. Therefore, "
            "latency/FPS changes describe benchmark-result "
            "differences but must not be interpreted as "
            "model or runtime regressions without "
            "controlling the execution environment."
        ),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_latency_chart(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Create a clear historical-vs-canonical latency chart."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtimes = [
        row["runtime"]
        for row in rows
    ]

    historical = [
        row["historical_mean_ms"]
        for row in rows
    ]

    canonical = [
        row["canonical_mean_ms"]
        for row in rows
    ]

    x = list(range(len(runtimes)))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        historical,
        width=width,
        label="Historical",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        canonical,
        width=width,
        label="Canonical",
    )

    ax.set_title(
        "E2 Benchmark Comparison — "
        "Mean Inference Latency"
    )

    ax.set_ylabel(
        "Mean latency (ms)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        runtimes,
        rotation=15,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(
        historical
    ):
        ax.text(
            index - width / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for index, value in enumerate(
        canonical
    ):
        ax.text(
            index + width / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare historical and canonical "
            "E2 benchmark results."
        )
    )

    parser.add_argument(
        "--historical",
        type=Path,
        default=Path(
            "reports/deployment_benchmark/"
            "E2_benchmark.json"
        ),
        help="Historical benchmark JSON.",
    )

    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path(
            "reports/deployment_benchmark/"
            "E2_benchmark_results.json"
        ),
        help="Canonical benchmark JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/deployment_benchmark"
        ),
        help="Directory for comparison outputs.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("E2 BENCHMARK COMPARISON")
    print("=" * 80)

    print()
    print(
        f"Historical: {args.historical}"
    )

    print(
        f"Canonical : {args.canonical}"
    )

    historical_data = load_json(
        args.historical
    )

    canonical_data = load_json(
        args.canonical
    )

    historical_results = (
        find_runtime_results(
            historical_data
        )
    )

    canonical_results = (
        find_runtime_results(
            canonical_data
        )
    )

    rows = compare(
        historical_results,
        canonical_results,
    )

    if not rows:
        raise RuntimeError(
            "No common runtimes were found "
            "between the two benchmark files."
        )

    summary_path = (
        args.output_dir
        / "E2_benchmark_comparison_summary.json"
    )

    csv_path = (
        args.output_dir
        / "E2_benchmark_comparison.csv"
    )

    chart_path = (
        args.output_dir
        / "E2_benchmark_comparison_latency.png"
    )

    save_summary(
        summary_path,
        args.historical,
        args.canonical,
        historical_data,
        canonical_data,
        rows,
    )

    save_csv(
        rows,
        csv_path,
    )

    save_latency_chart(
        rows,
        chart_path,
    )

    print()
    print(
        f"{'Runtime':<16}"
        f"{'Historical ms':>16}"
        f"{'Canonical ms':>16}"
        f"{'Latency Δ %':>14}"
        f"{'FPS Δ %':>12}"
    )

    print("-" * 74)

    for row in rows:

        latency_pct = (
            row["latency_change_pct"]
        )

        fps_pct = (
            row["fps_change_pct"]
        )

        print(
            f"{row['runtime']:<16}"
            f"{row['historical_mean_ms']:>16.3f}"
            f"{row['canonical_mean_ms']:>16.3f}"
            f"{latency_pct:>13.2f}%"
            f"{fps_pct:>11.2f}%"
        )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"Summary : {summary_path}"
    )

    print(
        f"CSV     : {csv_path}"
    )

    print(
        f"Chart   : {chart_path}"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()