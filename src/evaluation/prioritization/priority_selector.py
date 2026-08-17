from typing import Any, Dict, List, Optional

from src.evaluation.prioritization.criteria import PriorityCriterion


class PrioritySelector:
    """
    Rank evaluation items according to a prioritization criterion.
    """

    def __init__(
        self,
        criterion: PriorityCriterion,
    ):
        self.criterion = criterion

    def rank(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rank structured evaluation items from worst to best.

        Args:
            items:
                List of structured evaluation items.

                Example:
                    [
                        {
                            "name": "plastic",
                            "ap": 0.719,
                        },
                        {
                            "name": "glass",
                            "ap": 0.744,
                        },
                    ]

        Returns:
            Ranked evaluation items.

            Each item contains the original data plus
            a computed priority_score.
        """

        ranked_items = []

        for item in items:

            ranked_item = {
                **item,
                "priority_score": self.criterion.score(item),
            }

            ranked_items.append(ranked_item)

        ranked_items.sort(
            key=lambda item: item["priority_score"],
            reverse=not self.criterion.lower_is_worse,
        )

        return ranked_items

    def select(
        self,
        items: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select the top-k highest-priority evaluation items.

        Args:
            items:
                Structured evaluation items.

            top_k:
                Maximum number of items to return.
                If None, all ranked items are returned.

        Returns:
            Selected ranked evaluation items.
        """

        if top_k is not None and top_k <= 0:
            return []

        ranked_items = self.rank(items)

        if top_k is not None:
            return ranked_items[:top_k]

        return ranked_items