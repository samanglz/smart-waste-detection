
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ErrorCase:
    image_path: str
    gt_class: str
    pred_class: str
    confidence: float
    iou: float


class ErrorAnalysisComparator:
    """
    Compare two evaluation.json files (E2 vs E6).

    Outputs:
        - improved cases
        - regressed cases
        - unchanged errors
        - per-class summary
    """

    def __init__(self, e2_json: Dict, e6_json: Dict):
        self.e2 = e2_json
        self.e6 = e6_json

    def _index_errors(self, data: Dict):
        index = {}

        for category in [
            "confusion_cases",
            "false_positive_cases",
            "false_negative_cases",
        ]:
            cases = data.get("error_analysis", {}).get(category, [])

            for c in cases:
                key = (
                    c["image_path"],
                    c.get("gt_class"),
                    c.get("pred_class"),
                )
                index[key] = c

        return index

    def compare(self):
        e2_errors = self._index_errors(self.e2)
        e6_errors = self._index_errors(self.e6)

        improved = []
        regressed = []
        unchanged = []

        keys = set(e2_errors) | set(e6_errors)

        for k in keys:

            in_e2 = k in e2_errors
            in_e6 = k in e6_errors

            if in_e2 and not in_e6:
                improved.append(e2_errors[k])

            elif (not in_e2) and in_e6:
                regressed.append(e6_errors[k])

            else:
                unchanged.append(e6_errors[k])

        return {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        }