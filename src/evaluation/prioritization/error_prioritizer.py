from typing import Dict, Any, Optional, List

from src.evaluation.prioritization.priority_selector import PrioritySelector
from src.evaluation.prioritization.criterion_factory import CriterionFactory


class ErrorPrioritizer:
    """
    Prioritize detection errors according to configurable criteria.
    """

    def _build_selector(
        self,
        criterion: str,
    ) -> PrioritySelector:
        """
        Build a PrioritySelector using the requested criterion.
        """

        return PrioritySelector(
            criterion=CriterionFactory.create(
                criterion
            )
        )

    def rank_classes(
        self,
        metrics: Dict[str, Dict[str, Any]],
        criterion: str = "ap",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank classes according to the selected criterion.
        """

        items = [
            {
                "name": name,
                **data,
            }
            for name, data in metrics.items()
        ]

        selector = self._build_selector(
            criterion
        )

        return selector.select(
            items,
            top_k,
        )

    def rank_confusions(
        self,
        confusion: Dict[str, Dict[str, Dict[str, Any]]],
        criterion: str = "count",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank confusion pairs according to the selected criterion.
        """

        items = []

        for gt_class, predictions in confusion.items():

            for pred_class, data in predictions.items():

                items.append(
                    {
                        "gt_class": gt_class,
                        "pred_class": pred_class,
                        **data,
                    }
                )

        selector = self._build_selector(
            criterion
        )

        return selector.select(
            items,
            top_k,
        )

    def rank_false_positives(
        self,
        false_positives: Dict[str, Dict[str, Any]],
        criterion: str = "count",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank false-positive classes according to the selected criterion.
        """

        items = [
            {
                "class_name": class_name,
                **data,
            }
            for class_name, data in false_positives.items()
        ]

        selector = self._build_selector(
            criterion
        )

        return selector.select(
            items,
            top_k,
        )

    def rank_false_negatives(
        self,
        false_negatives: Dict[str, Dict[str, Any]],
        criterion: str = "count",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank false-negative classes according to the selected criterion.
        """

        items = [
            {
                "class_name": class_name,
                **data,
            }
            for class_name, data in false_negatives.items()
        ]

        selector = self._build_selector(
            criterion
        )

        return selector.select(
            items,
            top_k,
        )
        
        
        
        

        
        
        
        
        