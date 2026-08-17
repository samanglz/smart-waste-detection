from src.evaluation.prioritization.criteria import PriorityCriterion
from src.evaluation.prioritization.criteria.registry import (
    CRITERION_REGISTRY
)


class CriterionFactory:
    """
    Create criterion instances by name.
    """


    @staticmethod
    def create(
        name: str,
    ) -> PriorityCriterion:

        if name not in CRITERION_REGISTRY:
            raise ValueError(
                f"Unknown criterion: {name}"
            )


        criterion_class = CRITERION_REGISTRY[name]


        return criterion_class()