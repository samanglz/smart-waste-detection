import pytest


from src.evaluation.prioritization.error_prioritizer import (
    ErrorPrioritizer,
)


def test_rank_classes_by_ap():

    metrics = {
        "plastic": {
            "ap": 0.719,
            "precision": 0.68,
            "recall": 0.83,
        },
        "glass": {
            "ap": 0.744,
            "precision": 0.80,
            "recall": 0.76,
        },
        "cardboard": {
            "ap": 0.889,
            "precision": 0.90,
            "recall": 0.94,
        },
    }

    prioritizer = ErrorPrioritizer()

    ranked = prioritizer.rank_classes(
        metrics=metrics,
        criterion="ap",
    )

    assert ranked[0]["name"] == "plastic"
    assert ranked[1]["name"] == "glass"
    assert ranked[2]["name"] == "cardboard"

    assert ranked[0]["priority_score"] == 0.719
    assert ranked[1]["priority_score"] == 0.744
    assert ranked[2]["priority_score"] == 0.889
    
    
    
from src.evaluation.prioritization.error_prioritizer import (
    ErrorPrioritizer,
)


def test_rank_confusions_by_count():
    
    confusion = {
        "glass": {
            "metal": {
                "count": 17,
            },
            "plastic": {
                "count": 5,
            },
        },
        "metal": {
            "glass": {
                "count": 23,
            },
            "plastic": {
                "count": 8,
            },
        },
        "plastic": {
            "glass": {
                "count": 11,
            },
        },
    }

    prioritizer = ErrorPrioritizer()

    ranked = prioritizer.rank_confusions(
        confusion=confusion,
        criterion="count",
    )

    assert ranked[0]["gt_class"] == "metal"
    assert ranked[0]["pred_class"] == "glass"
    assert ranked[0]["count"] == 23
    assert ranked[0]["priority_score"] == 23

    assert ranked[1]["gt_class"] == "glass"
    assert ranked[1]["pred_class"] == "metal"
    assert ranked[1]["count"] == 17
    assert ranked[1]["priority_score"] == 17
    
def test_rank_confusions_with_top_k():
    
    confusion = {
        "glass": {
            "metal": {
                "count": 17,
            },
            "plastic": {
                "count": 5,
            },
        },
        "metal": {
            "glass": {
                "count": 23,
            },
            "plastic": {
                "count": 8,
            },
        },
        "plastic": {
            "glass": {
                "count": 11,
            },
        },
    }

    prioritizer = ErrorPrioritizer()

    ranked = prioritizer.rank_confusions(
        confusion=confusion,
        criterion="count",
        top_k=2,
    )

    assert len(ranked) == 2

    assert ranked[0]["count"] == 23
    assert ranked[1]["count"] == 17
    
    



def test_rank_confusions_with_unknown_criterion():

    confusion = {
        "glass": {
            "metal": {
                "count": 17,
            },
        },
    }

    prioritizer = ErrorPrioritizer()

    with pytest.raises(ValueError):

        prioritizer.rank_confusions(
            confusion=confusion,
            criterion="banana",
        )
        
        
        
def test_rank_false_positives_by_count():
    false_positives = {
        "plastic": {"count": 12},
        "glass": {"count": 27},
        "metal": {"count": 8},
    }

    prioritizer = ErrorPrioritizer()

    ranked = prioritizer.rank_false_positives(
        false_positives=false_positives,
        criterion="count",
    )

    assert ranked[0]["class_name"] == "glass"
    assert ranked[0]["count"] == 27
    assert ranked[1]["class_name"] == "plastic"
    assert ranked[1]["count"] == 12
    assert ranked[2]["class_name"] == "metal"
    assert ranked[2]["count"] == 8
    
    
    
def test_rank_false_negatives_by_count():
    false_negatives = {
        "plastic": {"count": 9},
        "glass": {"count": 21},
        "metal": {"count": 14},
    }

    prioritizer = ErrorPrioritizer()

    ranked = prioritizer.rank_false_negatives(
        false_negatives=false_negatives,
        criterion="count",
    )

    assert ranked[0]["class_name"] == "glass"
    assert ranked[0]["count"] == 21
    assert ranked[1]["class_name"] == "metal"
    assert ranked[1]["count"] == 14
    assert ranked[2]["class_name"] == "plastic"
    assert ranked[2]["count"] == 9