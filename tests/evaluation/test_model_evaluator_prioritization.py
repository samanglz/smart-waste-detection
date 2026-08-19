from unittest.mock import MagicMock

from src.evaluation.model_evaluator_Copy import ModelEvaluator_copy


def test_prioritize_errors():
    evaluator = ModelEvaluator_copy(
        model=MagicMock(),
        dataset=MagicMock(),
    )

    evaluator.analyze_errors = MagicMock(
        return_value={
            "per_class_metrics": {
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
            },
            "error_analysis": {
                "confusion": {
                    "metal": {
                        "glass": {
                            "count": 23,
                        },
                    },
                    "glass": {
                        "metal": {
                            "count": 17,
                        },
                    },
                },
                "false_positives": {
                    "glass": {
                        "count": 10,
                    },
                },
                "false_negatives": {
                    "plastic": {
                        "count": 8,
                    },
                },
            },
        }
    )

    result = evaluator.prioritize_errors(
        split="test",
        top_k=2,
    )

    assert result["classes"][0]["name"] == "plastic"
    assert result["classes"][1]["name"] == "glass"

    assert result["confusions"][0]["count"] == 23
    assert result["confusions"][1]["count"] == 17

    assert result["false_positives"][0]["class_name"] == "glass"
    assert result["false_positives"][0]["count"] == 10

    assert result["false_negatives"][0]["class_name"] == "plastic"
    assert result["false_negatives"][0]["count"] == 8

    evaluator.analyze_errors.assert_called_once_with(
        split="test",
    )