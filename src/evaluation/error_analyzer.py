from collections import defaultdict
from typing import Dict, Any, List


class ErrorAnalyzer:
    """
    Analyze detection errors produced by ModelEvaluator.
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def analyze_confusion(
        self,
        matches,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze class confusion from matched predictions.
        """

        confusion = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "count": 0,
                    "mean_iou": 0.0,
                    "mean_confidence": 0.0,
                    "images": [],
                    "cases" : [],
                }
            )
        )

        for match in matches:

            pred_class = match["pred_class"]
            gt_class = match["gt_class"]

            if pred_class == gt_class:
                continue

            gt_name = self._get_class_name(gt_class)
            pred_name = self._get_class_name(pred_class)

            confusion_data = confusion[gt_name][pred_name]

            confusion_data["count"] += 1
            confusion_data["mean_iou"] += match["iou"]

            confidence = match["prediction"].get(
                "confidence",
                0.0,
            )

            confusion_data["mean_confidence"] += confidence

            image_path = match.get("image_path")

            if image_path is not None:
                confusion_data["images"].append(image_path)

                confusion_data["cases"].append({
                    "image_path": image_path,
                    "gt_class": gt_name,
                    "pred_class": pred_name,
                    "gt_bbox": match["ground_truth"]["bbox"],
                    "pred_bbox": match["prediction"]["bbox"],
                    "iou": match["iou"],
                    "confidence": match["prediction"].get(
                        "confidence",
                        0.0,
                    ),
                })

        for predictions in confusion.values():

            for data in predictions.values():

                count = data["count"]

                if count > 0:
                    data["mean_iou"] /= count
                    data["mean_confidence"] /= count

        return {
            gt_name: dict(predictions)
            for gt_name, predictions in confusion.items()
        }



    def analyze_false_positives(
        self,
        unmatched_predictions,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze false-positive predictions.
        """

        false_positives = defaultdict(
            lambda: {
                "count": 0,
                "mean_confidence": 0.0,
                "images": [],
                "cases" : [],
            }
        )

        for item in unmatched_predictions:

            prediction = item["prediction"]

            class_id = prediction["class_id"]
            class_name = self._get_class_name(class_id)

            data = false_positives[class_name]

            data["count"] += 1

            confidence = prediction.get(
                "confidence",
                0.0,
            )

            data["mean_confidence"] += confidence

            image_path = item.get("image_path")

            if image_path is not None:
                data["images"].append(image_path)
                

                data["cases"].append({
                    "image_path": image_path,
                    "pred_class": class_name,
                    "pred_bbox": prediction["bbox"],
                    "confidence": prediction.get(
                        "confidence",
                        0.0,
                    ),
                })

        for data in false_positives.values():

            if data["count"] > 0:
                data["mean_confidence"] /= data["count"]

        return dict(false_positives)

    def analyze_false_negatives(
        self,
        unmatched_ground_truths,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze false-negative ground-truth objects.
        """

        false_negatives = defaultdict(
            lambda: {
                "count": 0,
                "images": [],
                "cases" : [],
            }
        )

        for item in unmatched_ground_truths:

            ground_truth = item["ground_truth"]

            class_id = ground_truth["class_id"]
            class_name = self._get_class_name(class_id)

            data = false_negatives[class_name]

            data["count"] += 1

            image_path = item.get("image_path")

            if image_path is not None:
                data["images"].append(image_path)
                

                data["cases"].append({
                    "image_path": image_path,
                    "gt_class": class_name,
                    "gt_bbox": ground_truth["bbox"],
                })

        return dict(false_negatives)

    def _get_class_name(self, class_id: int) -> str:
        """
        Convert class ID to class name safely.
        """

        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]

        return f"class_{class_id}"