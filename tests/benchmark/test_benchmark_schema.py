from __future__ import annotations

import json

from src.deployment.benchmark_core.schema import (
    BenchmarkMetadata,
    CanonicalBenchmarkResult,
    DatasetInfo,
    RuntimeResult,
)


def main() -> None:
    result = CanonicalBenchmarkResult(
        benchmark=BenchmarkMetadata(
            timestamp_utc="2026-09-07T00:00:00+00:00",
            protocol={
                "protocol_version": "1.0",
                "image_size": 640,
                "num_images": 2,
                "warmup_runs": 1,
                "benchmark_runs": 2,
                "batch_size": 1,
                "seed": 42,
                "split": "val",
                "timing_mode": "inference_only",
                "preprocessing_timed": False,
                "postprocessing_timed": False,
                "aggregation": "per_image_mean",
            },
        ),
        models={
            "pytorch": "model.pt",
            "onnx": "model.onnx",
        },
        artifacts={
            "pytorch": {
                "path": "model.pt",
                "sha256": "dummy-sha256",
                "size_bytes": 123,
            },
            "onnx": {
                "path": "model.onnx",
                "sha256": "dummy-sha256",
                "size_bytes": 456,
            },
        },
        dataset=DatasetInfo(
            split="val",
            image_dir="data/processed/val/images",
            available_images=358,
            selected_images=2,
            seed=42,
            filenames=[
                "img_000001.jpg",
                "img_000002.jpg",
            ],
        ),
        environment={
            "python_version": "3.x",
        },
        runtimes={
            "pytorch_cpu": {
                "runtime": "pytorch",
                "device": "cpu",
            },
            "pytorch_gpu": {
                "runtime": "pytorch",
                "device": "cuda",
            },
            "onnx_cpu": {
                "runtime": "onnx",
                "device": "cpu",
            },
            "onnx_gpu": {
                "runtime": "onnx",
                "device": "cuda",
            },
        },
        results={
            "pytorch_cpu": RuntimeResult(
                statistics={
                    "runs": 2,
                    "mean_ms": 100.0,
                    "median_ms": 100.0,
                    "std_ms": 0.0,
                    "min_ms": 100.0,
                    "max_ms": 100.0,
                    "p95_ms": 100.0,
                    "fps": 10.0,
                },
                per_image=[],
            ),
        },
        speedup={
            "pytorch_gpu_vs_cpu": 2.0,
        },
    )

    # --------------------------------------------------------------
    # Object -> dict
    # --------------------------------------------------------------

    data = result.to_dict()

    assert isinstance(data, dict)
    assert "benchmark" in data
    assert "models" in data
    assert "artifacts" in data
    assert "dataset" in data
    assert "environment" in data
    assert "runtimes" in data
    assert "results" in data
    assert "speedup" in data

    # --------------------------------------------------------------
    # Dict -> JSON
    # --------------------------------------------------------------

    json_text = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )

    assert isinstance(json_text, str)

    # --------------------------------------------------------------
    # JSON -> dict
    # --------------------------------------------------------------

    restored_dict = json.loads(json_text)

    # --------------------------------------------------------------
    # Dict -> Object
    # --------------------------------------------------------------

    restored = CanonicalBenchmarkResult.from_dict(
        restored_dict
    )

    assert isinstance(
        restored,
        CanonicalBenchmarkResult,
    )

    # --------------------------------------------------------------
    # Verify important fields
    # --------------------------------------------------------------

    assert (
        restored.benchmark.protocol["protocol_version"]
        == "1.0"
    )

    assert restored.dataset.split == "val"

    assert (
        restored.dataset.selected_images
        == 2
    )

    assert (
        restored.results[
            "pytorch_cpu"
        ].statistics["mean_ms"]
        == 100.0
    )

    assert (
        restored.speedup[
            "pytorch_gpu_vs_cpu"
        ]
        == 2.0
    )

    print("=" * 70)
    print("BENCHMARK SCHEMA SMOKE TEST")
    print("=" * 70)
    print("Object -> dict:       PASS")
    print("dict -> JSON:         PASS")
    print("JSON -> dict:         PASS")
    print("dict -> object:       PASS")
    print("Field validation:     PASS")
    print("=" * 70)
    print("SCHEMA SMOKE TEST PASSED")


if __name__ == "__main__":
    main()