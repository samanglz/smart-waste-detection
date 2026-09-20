from typing import List

import cv2
import numpy as np

from src.inference.schemas import Detection


class PostProcessor:
    """
    YOLO ONNX post-processing.

    Converts raw YOLO ONNX outputs into Detection objects
    and adds production diagnostics for each bounding box.
    """

    def __init__(
        self,
        class_names: List[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process(
        self,
        output,
        ratio,
        pad,
        image_width: int,
        image_height: int,
    ) -> List[Detection]:

        predictions = np.squeeze(output).T

        boxes = []
        scores = []
        class_ids = []

        pad_x, pad_y = pad

        for pred in predictions:

            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            cx, cy, w, h = pred[:4]

            # Convert from letterboxed coordinates
            # back to original image coordinates.
            x1 = (cx - w / 2 - pad_x) / ratio
            y1 = (cy - h / 2 - pad_y) / ratio
            x2 = (cx + w / 2 - pad_x) / ratio
            y2 = (cy + h / 2 - pad_y) / ratio

            boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ]
            )

            scores.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.conf_threshold,
            self.iou_threshold,
        )

        detections: List[Detection] = []

        if len(indices) == 0:
            return detections

        for idx in indices.flatten():

            x, y, w, h = boxes[idx]

            # ---------------------------------------------------------
            # Clip bounding box to the original image boundaries.
            # ---------------------------------------------------------
            x1 = max(0.0, min(float(x), float(image_width)))
            y1 = max(0.0, min(float(y), float(image_height)))
            x2 = max(0.0, min(float(x + w), float(image_width)))
            y2 = max(0.0, min(float(y + h), float(image_height)))

            # ---------------------------------------------------------
            # Diagnostic geometry metrics.
            # ---------------------------------------------------------
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)

            image_area = float(image_width * image_height)

            area_ratio = (
                (box_width * box_height) / image_area
                if image_area > 0
                else 0.0
            )

            width_ratio = (
                box_width / image_width
                if image_width > 0
                else 0.0
            )

            height_ratio = (
                box_height / image_height
                if image_height > 0
                else 0.0
            )

            # A small tolerance avoids floating-point boundary issues.
            touches_border = (
                x1 <= 1.0
                or y1 <= 1.0
                or x2 >= image_width - 1.0
                or y2 >= image_height - 1.0
            )

            detections.append(
                Detection(
                    class_id=class_ids[idx],
                    class_name=self.class_names[class_ids[idx]],
                    confidence=round(float(scores[idx]), 4),
                    bbox=[
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                    area_ratio=round(area_ratio, 4),
                    width_ratio=round(width_ratio, 4),
                    height_ratio=round(height_ratio, 4),
                    touches_border=touches_border,
                )
            )

        return detections