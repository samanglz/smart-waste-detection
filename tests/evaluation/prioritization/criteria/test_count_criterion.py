from src.evaluation.prioritization.criteria import CountCriterion


def test_count_criterion():

    criterion = CountCriterion()

    item = {
        "count": 42,
    }

    assert criterion.name == "count"
    assert criterion.lower_is_worse is False
    assert criterion.score(item) == 42.0