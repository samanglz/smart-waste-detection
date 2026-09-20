from pathlib import Path
import json
from typing import Dict, Any


class ErrorAnalysisComparator:
    """
    Compare error_analysis.json files from two experiments
    (e.g. E2 vs E5).

    Produces:
        - confusion matrix comparison
        - per-confusion delta
        - improved / regressed / unchanged pairs
        - new errors
        - removed errors
        - total confusion counts
    """

    def __init__(
        self,
        baseline_path: Path,
        experiment_path: Path,
        baseline_name: str = "E2",
        experiment_name: str = "E5",
    ):
        self.baseline_path = Path(baseline_path)
        self.experiment_path = Path(experiment_path)

        self.baseline_name = baseline_name
        self.experiment_name = experiment_name

        self.baseline = self._load_json(self.baseline_path)
        self.experiment = self._load_json(self.experiment_path)

    # ============================================================
    # LOAD
    # ============================================================

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ============================================================
    # EXTRACT CONFUSION
    # ============================================================

    @staticmethod
    def _extract_confusion(data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """
        Extract:

            GT class -> predicted class -> count
        """

        confusion = (
            data
            .get("error_analysis", {})
            .get("confusion", {})
        )

        result = {}

        for gt_class, predictions in confusion.items():

            if not isinstance(predictions, dict):
                continue

            result[gt_class] = {}

            for pred_class, info in predictions.items():

                if not isinstance(info, dict):
                    continue

                count = info.get("count", 0)

                result[gt_class][pred_class] = int(count)

        return result

    # ============================================================
    # FLATTEN
    # ============================================================

    @staticmethod
    def _flatten(
        confusion: Dict[str, Dict[str, int]]
    ) -> Dict[str, int]:

        flat = {}

        for gt_class, predictions in confusion.items():

            for pred_class, count in predictions.items():

                key = f"{gt_class} -> {pred_class}"
                flat[key] = count

        return flat

    # ============================================================
    # COMPARE
    # ============================================================

    def compare(self) -> Dict[str, Any]:

        e2_confusion = self._extract_confusion(self.baseline)
        e5_confusion = self._extract_confusion(self.experiment)

        e2_flat = self._flatten(e2_confusion)
        e5_flat = self._flatten(e5_confusion)

        all_pairs = sorted(
            set(e2_flat.keys()) |
            set(e5_flat.keys())
        )

        comparison = []

        improved = []
        regressed = []
        unchanged = []
        new_errors = []
        removed_errors = []

        for pair in all_pairs:

            e2_count = e2_flat.get(pair, 0)
            e5_count = e5_flat.get(pair, 0)

            delta = e5_count - e2_count

            if e2_count == 0 and e5_count > 0:
                status = "new_error"
                new_errors.append(pair)

            elif e2_count > 0 and e5_count == 0:
                status = "removed_error"
                removed_errors.append(pair)

            elif e5_count < e2_count:
                status = "improved"
                improved.append(pair)

            elif e5_count > e2_count:
                status = "regressed"
                regressed.append(pair)

            else:
                status = "unchanged"
                unchanged.append(pair)

            comparison.append({
                "confusion": pair,
                self.baseline_name: e2_count,
                self.experiment_name: e5_count,
                "delta": delta,
                "status": status,
            })

        total_e2 = sum(e2_flat.values())
        total_e5 = sum(e5_flat.values())

        return {
            "experiments": {
                "baseline": self.baseline_name,
                "experiment": self.experiment_name,
            },

            "total_confusions": {
                self.baseline_name: total_e2,
                self.experiment_name: total_e5,
                "delta": total_e5 - total_e2,
            },

            "summary": {
                "improved": len(improved),
                "regressed": len(regressed),
                "unchanged": len(unchanged),
                "new_errors": len(new_errors),
                "removed_errors": len(removed_errors),
            },

            "improved_confusions": improved,
            "regressed_confusions": regressed,
            "new_errors": new_errors,
            "removed_errors": removed_errors,
            "unchanged_confusions": unchanged,

            "comparison": comparison,
        }

    # ============================================================
    # SAVE JSON
    # ============================================================

    def save_json(
        self,
        output_path: Path,
        result: Dict[str, Any],
    ) -> None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

    # ============================================================
    # SAVE TEXT REPORT
    # ============================================================

    def save_text_report(
        self,
        output_path: Path,
        result: Dict[str, Any],
    ) -> None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        summary = result["summary"]
        totals = result["total_confusions"]

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("=" * 75 + "\n")
            f.write(
                f"{self.baseline_name} → "
                f"{self.experiment_name} ERROR ANALYSIS COMPARISON\n"
            )
            f.write("=" * 75 + "\n\n")

            # ----------------------------------------------------
            # TOTAL
            # ----------------------------------------------------

            f.write("TOTAL CONFUSIONS\n")
            f.write("-" * 75 + "\n")

            f.write(
                f"{self.baseline_name}: "
                f"{totals[self.baseline_name]}\n"
            )

            f.write(
                f"{self.experiment_name}: "
                f"{totals[self.experiment_name]}\n"
            )

            f.write(
                f"Delta: "
                f"{totals['delta']:+d}\n\n"
            )

            # ----------------------------------------------------
            # SUMMARY
            # ----------------------------------------------------

            f.write("SUMMARY\n")
            f.write("-" * 75 + "\n")

            f.write(
                f"Improved confusion pairs : "
                f"{summary['improved']}\n"
            )

            f.write(
                f"Regressed confusion pairs: "
                f"{summary['regressed']}\n"
            )

            f.write(
                f"Unchanged pairs          : "
                f"{summary['unchanged']}\n"
            )

            f.write(
                f"New errors               : "
                f"{summary['new_errors']}\n"
            )

            f.write(
                f"Removed errors           : "
                f"{summary['removed_errors']}\n\n"
            )

            # ----------------------------------------------------
            # DETAILED COMPARISON
            # ----------------------------------------------------

            f.write("CONFUSION COMPARISON\n")
            f.write("-" * 75 + "\n")

            f.write(
                f"{'Confusion':30}"
                f"{self.baseline_name:>8}"
                f"{self.experiment_name:>8}"
                f"{'Delta':>10}"
                f"  Status\n"
            )

            f.write("-" * 75 + "\n")

            for item in result["comparison"]:

                f.write(
                    f"{item['confusion']:30}"
                    f"{item[self.baseline_name]:8d}"
                    f"{item[self.experiment_name]:8d}"
                    f"{item['delta']:+10d}"
                    f"  {item['status']}\n"
                )

            # ----------------------------------------------------
            # IMPROVED
            # ----------------------------------------------------

            f.write("\n")
            f.write("=" * 75 + "\n")
            f.write("IMPROVED CONFUSIONS\n")
            f.write("=" * 75 + "\n")

            if result["improved_confusions"]:

                for pair in result["improved_confusions"]:
                    f.write(f"  ✓ {pair}\n")

            else:
                f.write("  None\n")

            # ----------------------------------------------------
            # REGRESSED
            # ----------------------------------------------------

            f.write("\n")
            f.write("=" * 75 + "\n")
            f.write("REGRESSED CONFUSIONS\n")
            f.write("=" * 75 + "\n")

            if result["regressed_confusions"]:

                for pair in result["regressed_confusions"]:
                    f.write(f"  ✗ {pair}\n")

            else:
                f.write("  None\n")

            # ----------------------------------------------------
            # NEW
            # ----------------------------------------------------

            f.write("\n")
            f.write("=" * 75 + "\n")
            f.write("NEW ERRORS IN EXPERIMENT\n")
            f.write("=" * 75 + "\n")

            if result["new_errors"]:

                for pair in result["new_errors"]:
                    f.write(f"  + {pair}\n")

            else:
                f.write("  None\n")

            # ----------------------------------------------------
            # REMOVED
            # ----------------------------------------------------

            f.write("\n")
            f.write("=" * 75 + "\n")
            f.write("ERRORS REMOVED IN EXPERIMENT\n")
            f.write("=" * 75 + "\n")

            if result["removed_errors"]:

                for pair in result["removed_errors"]:
                    f.write(f"  - {pair}\n")

            else:
                f.write("  None\n")

            f.write("\n")
            f.write("=" * 75 + "\n")


# =================================================================
# MAIN
# =================================================================

def main():

    # ============================================================
    # PATHS
    # ============================================================

    E2_ERROR_PATH = Path(
        "outputs/hard_case_analysis/E2/error_analysis.json"
    )

    E5_ERROR_PATH = Path(
        "outputs/hard_case_analysis/E5/error_analysis.json"
    )

    OUTPUT_DIR = Path(
        "outputs/hard_case_analysis/E2_vs_E5"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # COMPARE
    # ============================================================

    print("=" * 75)
    print("E2 vs E5 ERROR ANALYSIS COMPARISON")
    print("=" * 75)

    comparator = ErrorAnalysisComparator(
        baseline_path=E2_ERROR_PATH,
        experiment_path=E5_ERROR_PATH,
        baseline_name="E2",
        experiment_name="E5",
    )

    result = comparator.compare()

    # ============================================================
    # SAVE
    # ============================================================

    json_path = OUTPUT_DIR / "e2_vs_e5_error_comparison.json"
    txt_path = OUTPUT_DIR / "e2_vs_e5_error_comparison.txt"

    comparator.save_json(
        json_path,
        result
    )

    comparator.save_text_report(
        txt_path,
        result
    )

    # ============================================================
    # CONSOLE SUMMARY
    # ============================================================

    summary = result["summary"]
    totals = result["total_confusions"]

    print("\nRESULT")
    print("-" * 75)

    print(
        f"Total confusions:"
        f" E2={totals['E2']}"
        f" | E5={totals['E5']}"
        f" | Delta={totals['delta']:+d}"
    )

    print(
        f"Improved : {summary['improved']}"
    )

    print(
        f"Regressed: {summary['regressed']}"
    )

    print(
        f"Unchanged: {summary['unchanged']}"
    )

    print(
        f"New      : {summary['new_errors']}"
    )

    print(
        f"Removed  : {summary['removed_errors']}"
    )

    print("\nFiles saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {txt_path}")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()