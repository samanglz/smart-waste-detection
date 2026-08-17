from typing import Dict, Type

from src.evaluation.prioritization.criteria.base_criterion import (
    PriorityCriterion
)

from src.evaluation.prioritization.criteria.ap_criterion import (
    APCriterion
)

from src.evaluation.prioritization.criteria.count_criterion import CountCriterion


CRITERION_REGISTRY: Dict[str, Type[PriorityCriterion]] = {

    "ap": APCriterion,
    "count" : CountCriterion,

}