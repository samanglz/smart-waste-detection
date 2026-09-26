from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

E7_ROOT = PROJECT_ROOT / "outputs" / "yolo11m" / "E7_plus"
E7_REPORT = E7_ROOT / "report.json"
E7_ERROR_ANALYSIS = E7_ROOT / "error_analysis.json"
E7_VISUALIZATIONS = E7_ROOT / "visualizations"

BENCHMARK_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "deployment_benchmark"
    / "E7_plus_benchmark_results.json"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation" / "E7_plus"


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[OK] {output_path}")


def load_e7_report() -> dict:
    return load_json(E7_REPORT)


def find_e2_report() -> Path:
    candidates = [
        PROJECT_ROOT
        / "outputs"
        / "yolo11m"
        / "E2_targeted_augmentation"
        / "report.json",
        PROJECT_ROOT
        / "outputs"
        / "yolo11m"
        / "E2"
        / "report.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find E2 report.json. "
        "Expected one of:\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_e2_report() -> dict:
    path = find_e2_report()
    print(f"[INFO] E2 report: {path}")
    return load_json(path)


def chart_01_per_class_ap(e7_report: dict) -> None:
    metrics = e7_report["per_class_metrics"]

    values = [
        metrics[class_name]["ap"]
        for class_name in CLASS_NAMES
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(CLASS_NAMES, values)

    ax.set_title("E7 Plus — Per-Class AP")
    ax.set_xlabel("Class")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.0)

    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in values],
        padding=3,
    )

    save_figure(fig, "01_per_class_ap.png")


def chart_02_precision_recall(e7_report: dict) -> None:
    metrics = e7_report["per_class_metrics"]

    precision = [
        metrics[class_name]["precision"]
        for class_name in CLASS_NAMES
    ]

    recall = [
        metrics[class_name]["recall"]
        for class_name in CLASS_NAMES
    ]

    x = np.arange(len(CLASS_NAMES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    precision_bars = ax.bar(
        x - width / 2,
        precision,
        width,
        label="Precision",
    )

    recall_bars = ax.bar(
        x + width / 2,
        recall,
        width,
        label="Recall",
    )

    ax.set_title("E7 Plus — Precision and Recall")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend()

    ax.bar_label(
        precision_bars,
        labels=[f"{value:.3f}" for value in precision],
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        recall_bars,
        labels=[f"{value:.3f}" for value in recall],
        padding=3,
        fontsize=8,
    )

    save_figure(fig, "02_precision_recall.png")


def chart_03_e2_vs_e7_map(
    e2_report: dict,
    e7_report: dict,
) -> None:
    e2_metrics = e2_report["metrics"]
    e7_metrics = e7_report["metrics"]

    metric_names = [
        "mAP@0.5",
        "mAP@0.75",
        "mAP@0.5:0.95",
    ]

    e2_values = [
        e2_metrics["mAP_0.5"],
        e2_metrics["mAP_0.75"],
        e2_metrics["mAP_0.5_0.95"],
    ]

    e7_values = [
        e7_metrics["mAP_0.5"],
        e7_metrics["mAP_0.75"],
        e7_metrics["mAP_0.5_0.95"],
    ]

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    e2_bars = ax.bar(
        x - width / 2,
        e2_values,
        width,
        label="E2",
    )

    e7_bars = ax.bar(
        x + width / 2,
        e7_values,
        width,
        label="E7 Plus",
    )

    ax.set_title("E2 vs E7 Plus — Test mAP")
    ax.set_xlabel("Metric")
    ax.set_ylabel("mAP")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.legend()

    ax.bar_label(
        e2_bars,
        labels=[f"{value:.3f}" for value in e2_values],
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        e7_bars,
        labels=[f"{value:.3f}" for value in e7_values],
        padding=3,
        fontsize=8,
    )

    save_figure(fig, "03_e2_vs_e7_plus_map.png")


def chart_04_delta_ap(
    e2_report: dict,
    e7_report: dict,
) -> None:
    e2_metrics = e2_report["per_class_metrics"]
    e7_metrics = e7_report["per_class_metrics"]

    deltas = [
        e7_metrics[class_name]["ap"]
        - e2_metrics[class_name]["ap"]
        for class_name in CLASS_NAMES
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(CLASS_NAMES, deltas)

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title("E2 vs E7 Plus — ΔAP by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("ΔAP (E7 Plus − E2)")

    ax.bar_label(
        bars,
        labels=[f"{value:+.3f}" for value in deltas],
        padding=3,
    )

    save_figure(fig, "04_e2_vs_e7_plus_delta_ap.png")


def build_confusion_matrix(error_analysis: dict) -> np.ndarray:
    matrix = np.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=int,
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(CLASS_NAMES)
    }

    confusion = error_analysis["error_analysis"]["confusion"]

    # JSON structure:
    # outer key  = predicted class
    # inner key  = ground-truth class
    #
    # Matrix:
    # rows    = ground truth
    # columns = prediction

    for predicted_class, ground_truths in confusion.items():
        if predicted_class not in class_to_index:
            continue

        predicted_index = class_to_index[predicted_class]

        for ground_truth_class, info in ground_truths.items():
            if ground_truth_class not in class_to_index:
                continue

            ground_truth_index = class_to_index[ground_truth_class]

            matrix[
                ground_truth_index,
                predicted_index,
            ] += int(info["count"])

    return matrix


def chart_05_confusion_matrix(error_analysis: dict) -> None:
    matrix = build_confusion_matrix(error_analysis)

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(matrix)

    ax.set_title("E7 Plus — Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))

    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]

            if value > 0:
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                )

    fig.colorbar(image, ax=ax, label="Error Count")

    save_figure(fig, "05_confusion_matrix.png")


def chart_06_error_distribution(error_analysis: dict) -> None:
    confusion = error_analysis["error_analysis"]["confusion"]

    pairs = []

    for predicted_class, ground_truths in confusion.items():
        for ground_truth_class, info in ground_truths.items():
            count = int(info["count"])

            pairs.append(
                (
                    f"{ground_truth_class} → {predicted_class}",
                    count,
                )
            )

    pairs.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    labels = [item[0] for item in pairs]
    counts = [item[1] for item in pairs]

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(labels))

    bars = ax.barh(y, counts)

    ax.set_title("E7 Plus — Confusion Error Distribution")
    ax.set_xlabel("Number of Errors")
    ax.set_ylabel("GT → Prediction")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    ax.bar_label(
        bars,
        padding=3,
    )

    save_figure(fig, "06_error_distribution.png")


def load_error_summary(
    error_type: str,
) -> dict[str, dict]:
    root = E7_VISUALIZATIONS / error_type

    results = {}

    if not root.exists():
        return results

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        summary_path = class_dir / "summary.json"

        if not summary_path.exists():
            continue

        summary = load_json(summary_path)

        results[class_dir.name] = summary

    return results


def chart_07_false_positives() -> None:
    summaries = load_error_summary("false_positives")

    counts = [
        int(summaries.get(class_name, {}).get("count", 0))
        for class_name in CLASS_NAMES
    ]

    confidences = [
        summaries.get(class_name, {}).get(
            "mean_confidence",
            0.0,
        )
        for class_name in CLASS_NAMES
    ]

    x = np.arange(len(CLASS_NAMES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    count_bars = ax.bar(
        x - width / 2,
        counts,
        width,
        label="FP count",
    )

    confidence_bars = ax.bar(
        x + width / 2,
        confidences,
        width,
        label="Mean confidence",
    )

    ax.set_title("E7 Plus — False Positives")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Count / Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()

    ax.bar_label(
        count_bars,
        labels=[str(value) for value in counts],
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        confidence_bars,
        labels=[f"{value:.3f}" for value in confidences],
        padding=3,
        fontsize=8,
    )

    save_figure(fig, "07_false_positives.png")


def chart_08_false_negatives() -> None:
    summaries = load_error_summary("false_negatives")

    counts = [
        int(summaries.get(class_name, {}).get("count", 0))
        for class_name in CLASS_NAMES
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(CLASS_NAMES, counts)

    ax.set_title("E7 Plus — False Negatives")
    ax.set_xlabel("Ground Truth Class")
    ax.set_ylabel("FN Count")

    ax.bar_label(
        bars,
        labels=[str(value) for value in counts],
        padding=3,
    )

    save_figure(fig, "08_false_negatives.png")


def chart_09_deployment_performance() -> None:
    benchmark = load_json(BENCHMARK_REPORT)

    results = benchmark["results"]

    runtime_order = [
        "pytorch_cpu",
        "pytorch_gpu",
        "onnx_cpu",
        "onnx_gpu",
    ]

    labels = []
    latency = []
    fps = []

    for runtime in runtime_order:
        if runtime not in results:
            continue

        labels.append(runtime.replace("_", " ").title())

        result = results[runtime]

        latency.append(float(result["statistics"]["mean_ms"]))
        fps.append(float(result["statistics"]["fps"]))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, latency)

    ax.set_title("E7 Plus — Deployment Latency")
    ax.set_xlabel("Runtime")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(
        bars,
        labels=[f"{value:.2f} ms" for value in latency],
        padding=3,
    )

    save_figure(fig, "09_deployment_performance.png")

    # Keep FPS as a separate textual report because latency and FPS
    # use different units/scales.
    fps_path = OUTPUT_DIR / "09_deployment_fps.txt"

    with fps_path.open("w", encoding="utf-8") as file:
        file.write("E7 Plus Deployment FPS\n")
        file.write("======================\n\n")

        for label, value in zip(labels, fps):
            file.write(f"{label}: {value:.3f} FPS\n")

    print(f"[OK] {fps_path}")


def main() -> None:
    
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Generate E7 Plus evaluation visualizations."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated charts.",
    )

    args = parser.parse_args()

    
    OUTPUT_DIR = args.output_dir

    print("=" * 70)
    print("E7 PLUS VISUALIZATION REPORT")
    print("=" * 70)

    print("\n[1/9] Loading reports...")

    e7_report = load_e7_report()
    e2_report = load_e2_report()
    error_analysis = load_json(E7_ERROR_ANALYSIS)

    print("\n[2/9] Per-class AP")
    chart_01_per_class_ap(e7_report)

    print("\n[3/9] Precision / Recall")
    chart_02_precision_recall(e7_report)

    print("\n[4/9] E2 vs E7 Plus mAP")
    chart_03_e2_vs_e7_map(
        e2_report,
        e7_report,
    )

    print("\n[5/9] E2 vs E7 Plus ΔAP")
    chart_04_delta_ap(
        e2_report,
        e7_report,
    )

    print("\n[6/9] Confusion Matrix")
    chart_05_confusion_matrix(error_analysis)

    print("\n[7/9] Error Distribution")
    chart_06_error_distribution(error_analysis)

    print("\n[8/9] False Positives / False Negatives")
    chart_07_false_positives()
    chart_08_false_negatives()

    print("\n[9/9] Deployment Performance")
    chart_09_deployment_performance()

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()