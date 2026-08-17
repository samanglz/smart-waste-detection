from typing import Any, Dict, List

from src.evaluation.prioritization.criteria import PriorityCriterion


class PriorityPolicy:
    """
    Define how evaluation items are prioritized using
    one or more priority criteria.
    """

    def __init__(
        self,
        criteria: List[PriorityCriterion],
    ):
        if not criteria:
            raise ValueError(
                "At least one priority criterion is required."
            )

        self.criteria = criteria

    def score(
        self,
        item: Dict[str, Any],
    ) -> float:
        """
        Calculate the priority score of an evaluation item.

        The current policy uses the arithmetic mean
        of normalized criterion scores.

        This method will be extended when multi-criterion
        weighting is introduced.
        """

        scores = [
            criterion.score(item)
            for criterion in self.criteria
        ]

        return sum(scores) / len(scores)