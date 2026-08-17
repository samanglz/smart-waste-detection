import pytest

from src.evaluation.prioritization import PriorityPolicy
from src.evaluation.prioritization.criteria import APCriterion


def test_priority_policy_with_single_criterion():

    policy = PriorityPolicy(
        criteria=[
            APCriterion(),
        ]
    )

    item = {
        "ap": 0.719,
    }

    score = policy.score(item)

    assert score == pytest.approx(0.719)


def test_priority_policy_requires_criterion():

    with pytest.raises(ValueError):
        PriorityPolicy(
            criteria=[]
        )