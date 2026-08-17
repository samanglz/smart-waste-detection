from src.evaluation.prioritization import PrioritySelector
from src.evaluation.prioritization.criteria import APCriterion


def test_rank_by_ap():
    
    items = [
        {
            "name": "cardboard",
            "ap": 0.889,
        },
        {
            "name": "glass",
            "ap": 0.744,
        },
        {
            "name": "metal",
            "ap": 0.834,
        },
        {
            "name": "paper",
            "ap": 0.863,
        },
        {
            "name": "plastic",
            "ap": 0.719,
        },
    ]

    selector = PrioritySelector(
        criterion=APCriterion()
    )

    ranked = selector.rank(items)

    assert ranked[0]["name"] == "plastic"
    assert ranked[1]["name"] == "glass"
    assert ranked[2]["name"] == "metal"
    assert ranked[3]["name"] == "paper"
    assert ranked[4]["name"] == "cardboard"



def test_select_top_k():
    
    items = [
        {
            "name": "cardboard",
            "ap": 0.889,
        },
        {
            "name": "glass",
            "ap": 0.744,
        },
        {
            "name": "metal",
            "ap": 0.834,
        },
        {
            "name": "paper",
            "ap": 0.863,
        },
        {
            "name": "plastic",
            "ap": 0.719,
        },
    ]

    selector = PrioritySelector(
        criterion=APCriterion()
    )

    selected = selector.select(
        items,
        top_k=2,
    )

    assert len(selected) == 2

    assert selected[0]["name"] == "plastic"
    assert selected[1]["name"] == "glass"

def test_select_zero_items():
    items = {
        "plastic": {"ap": 0.719},
    }

    selector = PrioritySelector(
        criterion=APCriterion()
    )

    assert selector.select(
        items,
        top_k=0,
    ) == []
    
    
from src.evaluation.prioritization import PrioritySelector
from src.evaluation.prioritization.criteria import (
    APCriterion,
    CountCriterion,
)


def test_rank_by_count():

    items = [
        {
            "gt_class": "glass",
            "pred_class": "metal",
            "count": 17,
        },
        {
            "gt_class": "metal",
            "pred_class": "glass",
            "count": 23,
        },
        {
            "gt_class": "plastic",
            "pred_class": "glass",
            "count": 11,
        },
    ]

    selector = PrioritySelector(
        criterion=CountCriterion()
    )

    ranked = selector.rank(items)

    assert ranked[0]["gt_class"] == "metal"
    assert ranked[0]["pred_class"] == "glass"
    assert ranked[0]["priority_score"] == 23

    assert ranked[1]["gt_class"] == "glass"
    assert ranked[1]["pred_class"] == "metal"
    assert ranked[1]["priority_score"] == 17

    assert ranked[2]["gt_class"] == "plastic"
    assert ranked[2]["pred_class"] == "glass"
    assert ranked[2]["priority_score"] == 11