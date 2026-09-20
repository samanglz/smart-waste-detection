from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.deployment.benchmark import DeploymentBenchmark
from src.deployment.benchmark_core.config import BenchmarkConfig
from src.deployment.benchmark_core.schema import (
    CanonicalBenchmarkResult,
)


def main() -> None:
    print("=" * 70)
    print("BENCHMARK INTEGRATION SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Create temporary benchmark assets
    # --------------------------------------------------------------

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        pytorch_model = tmp_path / "model.pt"
        onnx_model = tmp_path / "model.onnx"
        image_dir = tmp_path / "images"
        output_dir = tmp_path / "output"

        pytorch_model.write_bytes(b"dummy pytorch model")
        onnx_model.write_bytes(b"dummy onnx model")
        image_dir.mkdir()

        # Create deterministic fake image filenames.
        for index in range(10):
            (image_dir / f"image_{index:03d}.jpg").write_bytes(
                b"dummy image"
            )

        # ----------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------

        config = BenchmarkConfig(
            image_size=640,
            num_images=5,
            warmup_runs=1,
            benchmark_runs=2,
            batch_size=1,
            seed=42,
            split="val",
        )

        # ----------------------------------------------------------
        # Instantiate DeploymentBenchmark
        # ----------------------------------------------------------

        benchmark = DeploymentBenchmark(
            pytorch_model_path=pytorch_model,
            onnx_model_path=onnx_model,
            image_dir=image_dir,
            config=config,
            output_dir=output_dir,
        )

        print("DeploymentBenchmark init: PASS")

        # ----------------------------------------------------------
        # Image discovery
        # ----------------------------------------------------------

        discovered = benchmark.discover_images()

        assert len(discovered) == 10
        assert discovered == sorted(
            discovered,
            key=lambda path: path.name,
        )

        print("Image discovery:          PASS")

        # ----------------------------------------------------------
        # Deterministic selection
        # ----------------------------------------------------------

        selected_1 = benchmark.select_images()
        selected_2 = benchmark.select_images()

        assert selected_1 == selected_2
        assert len(selected_1) == 5

        print("Deterministic selection:  PASS")

        # ----------------------------------------------------------
        # Verify filenames
        # ----------------------------------------------------------

        selected_names = [
            path.name
            for path in selected_1
        ]

        assert len(selected_names) == 5
        assert len(set(selected_names)) == 5

        print("Selected image metadata:  PASS")

        # ----------------------------------------------------------
        # Test canonical schema independently
        # ----------------------------------------------------------

        fake_result = {
            "benchmark": {
                "timestamp_utc": (
                    "2026-09-07T00:00:00+00:00"
                ),
                "protocol": config.to_dict(),
            },
            "models": {
                "pytorch": str(pytorch_model.resolve()),
                "onnx": str(onnx_model.resolve()),
            },
            "artifacts": {},
            "dataset": {
                "split": config.split,
                "image_dir": str(image_dir.resolve()),
                "available_images": 10,
                "selected_images": 5,
                "seed": config.seed,
                "filenames": selected_names,
            },
            "environment": {},
            "runtimes": {},
            "results": {},
            "speedup": {},
        }

        canonical = CanonicalBenchmarkResult.from_dict(
            fake_result
        )

        assert isinstance(
            canonical,
            CanonicalBenchmarkResult,
        )

        print("Canonical schema integration: PASS")

        # ----------------------------------------------------------
        # JSON serialization
        # ----------------------------------------------------------

        json_data = canonical.to_dict()

        json_text = json.dumps(
            json_data,
            indent=2,
            ensure_ascii=False,
        )

        restored = json.loads(json_text)

        assert restored["benchmark"]["protocol"][
            "protocol_version"
        ] == "1.0"

        assert restored["dataset"]["selected_images"] == 5

        print("JSON serialization:       PASS")

        # ----------------------------------------------------------
        # Save result
        # ----------------------------------------------------------

        output_path = benchmark.save_results(
            json_data
        )

        assert output_path.exists()
        assert output_path.is_file()

        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            saved = json.load(file)

        assert saved["benchmark"]["protocol"][
            "protocol_version"
        ] == "1.0"

        assert saved["dataset"]["selected_images"] == 5

        print("Result file generation:   PASS")

    print("=" * 70)
    print("BENCHMARK INTEGRATION SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()