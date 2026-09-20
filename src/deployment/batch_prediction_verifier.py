from pathlib import Path

import numpy as np

from src.deployment.prediction_comparator import (
    PredictionComparator,
)


class BatchPredictionVerifier:
    """
    Run PyTorch vs ONNX prediction-level verification
    over a batch of validation images.

    This class reuses PredictionComparator and aggregates
    prediction-level equivalence statistics.

    Responsibilities:
        - Discover validation images
        - Select a deterministic subset
        - Run comparison on each image
        - Aggregate results
        - Report summary statistics

    It does NOT:
        - Modify the test set
        - Train models
        - Calculate mAP
        - Perform dataset evaluation
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        pytorch_model_path: Path,
        onnx_model_path: Path,
        image_dir: Path,
        num_images: int = 30,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        match_iou_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.image_dir = Path(image_dir)
        self.num_images = num_images
        self.seed = seed

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Validation image directory not found: "
                f"{self.image_dir}"
            )

        self.comparator = PredictionComparator(
            pytorch_model_path=pytorch_model_path,
            onnx_model_path=onnx_model_path,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        self.match_iou_threshold = (
            match_iou_threshold
        )

    # ------------------------------------------------------------------
    # Image discovery
    # ------------------------------------------------------------------

    def discover_images(self) -> list[Path]:
        """
        Discover supported image files in validation directory.
        """

        images = [
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in self.SUPPORTED_EXTENSIONS
        ]

        if not images:
            raise RuntimeError(
                f"No supported images found in: "
                f"{self.image_dir}"
            )

        images.sort()

        return images

    # ------------------------------------------------------------------
    # Deterministic sampling
    # ------------------------------------------------------------------

    def select_images(
        self,
        images: list[Path],
    ) -> list[Path]:
        """
        Select a deterministic subset of validation images.

        If the directory contains fewer images than requested,
        all available images are used.
        """

        if len(images) <= self.num_images:
            return images

        rng = np.random.default_rng(
            self.seed
        )

        indices = rng.choice(
            len(images),
            size=self.num_images,
            replace=False,
        )

        indices = np.sort(indices)

        return [
            images[index]
            for index in indices
        ]

    # ------------------------------------------------------------------
    # Batch verification
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run prediction comparison on the selected
        validation images.
        """

        all_images = self.discover_images()

        selected_images = self.select_images(
            all_images
        )

        results = []
        errors = []

        total = len(selected_images)

        print()
        print("=" * 72)
        print("BATCH PYTORCH vs ONNX VERIFICATION")
        print("=" * 72)

        print(
            f"Validation directory : "
            f"{self.image_dir}"
        )

        print(
            f"Available images     : "
            f"{len(all_images)}"
        )

        print(
            f"Selected images      : "
            f"{len(selected_images)}"
        )

        print(
            f"Random seed          : "
            f"{self.seed}"
        )

        print("=" * 72)
        print()

        for index, image_path in enumerate(
            selected_images,
            start=1,
        ):
            print(
                f"[{index:02d}/{total:02d}] "
                f"{image_path.name}",
                end=" ... ",
            )

            try:
                result = self.comparator.compare(
                    image_path
                )

                results.append(result)

                print(
                    f"PT={result['pytorch_count']} "
                    f"ONNX={result['onnx_count']} "
                    f"MATCH={result['matched_count']} "
                    f"IoU={result['mean_iou']:.4f}"
                )

            except Exception as exc:
                error = {
                    "image": str(image_path),
                    "error": str(exc),
                }

                errors.append(error)

                print(
                    f"ERROR: {exc}"
                )

        return self.aggregate(
            results=results,
            errors=errors,
            selected_images=selected_images,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        results: list[dict],
        errors: list[dict],
        selected_images: list[Path],
    ) -> dict:
        """
        Aggregate prediction-level comparison results.
        """

        processed_count = len(results)
        error_count = len(errors)

        if processed_count == 0:
            raise RuntimeError(
                "No images were successfully processed."
            )

        exact_detection_count = 0
        perfect_match_count = 0

        total_pytorch_detections = 0
        total_onnx_detections = 0
        total_matched = 0
        total_unmatched_pytorch = 0
        total_unmatched_onnx = 0

        image_ious = []
        confidence_differences = []

        min_iou = None
        max_conf_difference = None

        for result in results:
            pytorch_count = result[
                "pytorch_count"
            ]

            onnx_count = result[
                "onnx_count"
            ]

            matched_count = result[
                "matched_count"
            ]

            unmatched_pytorch = result[
                "unmatched_pytorch"
            ]

            unmatched_onnx = result[
                "unmatched_onnx"
            ]

            mean_iou = result[
                "mean_iou"
            ]

            mean_conf_difference = result[
                "mean_conf_difference"
            ]

            total_pytorch_detections += (
                pytorch_count
            )

            total_onnx_detections += (
                onnx_count
            )

            total_matched += matched_count

            total_unmatched_pytorch += (
                unmatched_pytorch
            )

            total_unmatched_onnx += (
                unmatched_onnx
            )

            if (
                pytorch_count
                == onnx_count
                == matched_count
                and unmatched_pytorch == 0
                and unmatched_onnx == 0
            ):
                exact_detection_count += 1

            if (
                matched_count > 0
                and mean_iou >= 0.999
                and mean_conf_difference
                <= 1e-5
            ):
                perfect_match_count += 1

            if matched_count > 0:
                image_ious.append(
                    mean_iou
                )

                confidence_differences.append(
                    mean_conf_difference
                )

                if (
                    min_iou is None
                    or mean_iou < min_iou
                ):
                    min_iou = mean_iou

                if (
                    max_conf_difference is None
                    or mean_conf_difference
                    > max_conf_difference
                ):
                    max_conf_difference = (
                        mean_conf_difference
                    )

        total_images = len(
            selected_images
        )

        average_iou = (
            float(np.mean(image_ious))
            if image_ious
            else 0.0
        )

        average_conf_difference = (
            float(
                np.mean(
                    confidence_differences
                )
            )
            if confidence_differences
            else 0.0
        )

        detection_count_difference = (
            total_onnx_detections
            - total_pytorch_detections
        )

        total_detection_pairs = (
            total_matched
            + total_unmatched_pytorch
            + total_unmatched_onnx
        )

        match_rate = (
            total_matched
            / total_detection_pairs
            if total_detection_pairs > 0
            else 1.0
        )

        return {
            "total_images": total_images,
            "processed_images": processed_count,
            "error_images": error_count,
            "exact_detection_images": (
                exact_detection_count
            ),
            "perfect_match_images": (
                perfect_match_count
            ),
            "exact_detection_rate": (
                exact_detection_count
                / processed_count
            ),
            "perfect_match_rate": (
                perfect_match_count
                / processed_count
            ),
            "total_pytorch_detections": (
                total_pytorch_detections
            ),
            "total_onnx_detections": (
                total_onnx_detections
            ),
            "detection_count_difference": (
                detection_count_difference
            ),
            "total_matched_detections": (
                total_matched
            ),
            "total_unmatched_pytorch": (
                total_unmatched_pytorch
            ),
            "total_unmatched_onnx": (
                total_unmatched_onnx
            ),
            "detection_match_rate": (
                match_rate
            ),
            "average_matched_iou": (
                average_iou
            ),
            "minimum_matched_iou": (
                min_iou
            ),
            "average_confidence_difference": (
                average_conf_difference
            ),
            "maximum_confidence_difference": (
                max_conf_difference
            ),
            "errors": errors,
            "results": results,
        }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_report(summary: dict) -> None:
    """
    Print aggregate batch verification report.
    """

    print()
    print("=" * 72)
    print("BATCH VERIFICATION SUMMARY")
    print("=" * 72)

    print()
    print("IMAGE COVERAGE")
    print("-" * 72)

    print(
        f"Total selected images       : "
        f"{summary['total_images']}"
    )

    print(
        f"Successfully processed     : "
        f"{summary['processed_images']}"
    )

    print(
        f"Images with errors         : "
        f"{summary['error_images']}"
    )

    print()
    print("PREDICTION AGREEMENT")
    print("-" * 72)

    print(
        f"Exact detection images     : "
        f"{summary['exact_detection_images']}"
    )

    print(
        f"Exact detection rate       : "
        f"{summary['exact_detection_rate']:.2%}"
    )

    print(
        f"Near-perfect match images  : "
        f"{summary['perfect_match_images']}"
    )

    print(
        f"Near-perfect match rate    : "
        f"{summary['perfect_match_rate']:.2%}"
    )

    print()
    print("DETECTION COUNTS")
    print("-" * 72)

    print(
        f"Total PyTorch detections   : "
        f"{summary['total_pytorch_detections']}"
    )

    print(
        f"Total ONNX detections      : "
        f"{summary['total_onnx_detections']}"
    )

    print(
        f"Detection count difference : "
        f"{summary['detection_count_difference']:+d}"
    )

    print()
    print("MATCHING")
    print("-" * 72)

    print(
        f"Matched detections         : "
        f"{summary['total_matched_detections']}"
    )

    print(
        f"Unmatched PyTorch          : "
        f"{summary['total_unmatched_pytorch']}"
    )

    print(
        f"Unmatched ONNX             : "
        f"{summary['total_unmatched_onnx']}"
    )

    print(
        f"Detection match rate       : "
        f"{summary['detection_match_rate']:.2%}"
    )

    print()
    print("NUMERICAL AGREEMENT")
    print("-" * 72)

    print(
        f"Average matched IoU        : "
        f"{summary['average_matched_iou']:.8f}"
    )

    if summary["minimum_matched_iou"] is not None:
        print(
            f"Minimum matched IoU        : "
            f"{summary['minimum_matched_iou']:.8f}"
        )
    else:
        print(
            "Minimum matched IoU        : N/A"
        )

    print(
        f"Average confidence diff    : "
        f"{summary['average_confidence_difference']:.10f}"
    )

    if (
        summary["maximum_confidence_difference"]
        is not None
    ):
        print(
            f"Maximum confidence diff    : "
            f"{summary['maximum_confidence_difference']:.10f}"
        )
    else:
        print(
            "Maximum confidence diff    : N/A"
        )

    if summary["errors"]:
        print()
        print("ERRORS")
        print("-" * 72)

        for error in summary["errors"]:
            print(
                f"{error['image']} -> "
                f"{error['error']}"
            )

    print()
    print("=" * 72)