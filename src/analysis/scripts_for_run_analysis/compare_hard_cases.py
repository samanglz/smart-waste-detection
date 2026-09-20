import json
from pathlib import Path

E2_PATH = Path("outputs/hard_case_analysis/E2/similarity_analysis.json")
E5_PATH = Path("outputs/hard_case_analysis/E5/similarity_analysis.json")

OUTPUT_DIR = Path("outputs/hard_case_analysis/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}

    for item in data["results"]:
        mapping[item["image_path"]] = item

    return mapping, data


def main():

    e2_cases, e2_data = load_results(E2_PATH)
    e5_cases, e5_data = load_results(E5_PATH)

    common = sorted(set(e2_cases.keys()) & set(e5_cases.keys()))

    comparisons = []

    improved = 0
    regressed = 0
    unchanged = 0

    e2_margins = []
    e5_margins = []

    for img in common:

        a = e2_cases[img]
        b = e5_cases[img]

        e2_margin = a["margin"]
        e5_margin = b["margin"]

        e2_margins.append(e2_margin)
        e5_margins.append(e5_margin)

        if e5_margin < e2_margin:
            status = "improved"
            improved += 1
        elif e5_margin > e2_margin:
            status = "regressed"
            regressed += 1
        else:
            status = "unchanged"
            unchanged += 1

        comparisons.append({
            "image_path": img,
            "gt_class": a["gt_class"],

            "pred_class_e2": a["pred_class"],
            "pred_class_e5": b["pred_class"],

            "prediction_changed": (
                a["pred_class"] != b["pred_class"]
            ),

            "margin_e2": e2_margin,
            "margin_e5": e5_margin,

            "margin_delta": round(
                e5_margin - e2_margin,
                4
            ),

            "status": status,
        })

    avg_e2 = sum(e2_margins) / len(e2_margins)
    avg_e5 = sum(e5_margins) / len(e5_margins)

    summary = {
        "matched_cases": len(common),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "avg_margin_e2": round(avg_e2, 4),
        "avg_margin_e5": round(avg_e5, 4),
        "delta_margin": round(avg_e5 - avg_e2, 4),
    }

    with open(OUTPUT_DIR / "hard_case_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")
        f.write("E2 vs E5 HARD CASE COMPARISON\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Matched hard cases : {len(common)}\n\n")

        f.write(f"Improved  : {improved}\n")
        f.write(f"Regressed : {regressed}\n")
        f.write(f"Unchanged : {unchanged}\n\n")

        f.write("Average margin\n")
        f.write(f"  E2 : {avg_e2:.4f}\n")
        f.write(f"  E5 : {avg_e5:.4f}\n")
        f.write(f"\nMargin difference : {avg_e5-avg_e2:.4f}\n")

    print(summary)
    
    
    


if __name__ == "__main__":
    main()