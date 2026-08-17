import pytest
from src.evaluation.prioritization.criteria import APCriterion


def test_ap_criterion():
    criterion = APCriterion()

    item = {
        "class_name": "plastic",
        "ap": 0.719,
    }

    assert criterion.name == "ap"
    assert criterion.lower_is_worse is True
    assert criterion.score(item) == pytest.approx(0.719)