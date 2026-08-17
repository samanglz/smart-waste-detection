from typing import Any, Dict

from src.evaluation.prioritization.criteria.base_criterion import (
    PriorityCriterion,
)


class APCriterion(PriorityCriterion):
    """
    Prioritize classes according to their Average Precision (AP).

    This criterion treats classes with lower AP as higher-priority
    error cases because a lower AP indicates weaker overall
    detection performance for the corresponding class.

    The criterion is intended to operate on structured per-class
    evaluation results produced by ``ModelEvaluator``.

    Expected item structure:

        {
            "class_name": "plastic",
            "ap": 0.719
        }

    Ranking behavior:

        lower AP -> higher priority

    Therefore, a class with AP=0.50 receives higher priority than
    a class with AP=0.85.
    """

    @property
    def name(self) -> str:
        """
        Return the unique identifier of this prioritization criterion.

        Returns:
            str:
                Criterion name used for registration, configuration,
                logging, and criterion selection.
        """
        return "ap"

    @property
    def lower_is_worse(self) -> bool:
        """
        Indicate that lower AP represents worse performance.

        Returns:
            bool:
                ``True`` because classes with lower AP should be
                ranked with higher priority.
        """
        return True

    def score(self, item: Dict[str, Any]) -> float:
        """
        Extract the AP score from a structured evaluation item.

        Args:
            item:
                Per-class evaluation result containing an ``ap`` field.

        Returns:
            float:
                Average Precision value used for prioritization.

        Raises:
            KeyError:
                If the evaluation item does not contain ``ap``.

            TypeError:
                If the AP value cannot be converted to ``float``.
        """
        return float(item["ap"])