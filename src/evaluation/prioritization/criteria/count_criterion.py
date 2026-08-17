from typing import Any, Dict

from src.evaluation.prioritization.criteria.base_criterion import (
    PriorityCriterion,
)


class CountCriterion(PriorityCriterion):
    """
    Prioritize evaluation items according to their error count.

    Higher counts represent worse performance and therefore
    receive higher priority.
    """

    @property
    def name(self) -> str:
        return "count"

    @property
    def lower_is_worse(self) -> bool:
        return False

    def score(
        self,
        item: Dict[str, Any],
    ) -> float:
        return float(item["count"])