from pathlib import Path

import cv2

from src.inference.predictor import ONNXPredictor


class VideoProcessor:
    def __init__(self, predictor: ONNXPredictor):
        self.predictor = predictor

    def process(
        self,
        input_path: Path,
        output_path: Path,
    ) -> Path:

        # ---------------------------
        # VideoCapture
        # ---------------------------
        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise RuntimeError("Cannot open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ---------------------------
        # VideoWriter
        # ---------------------------
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                # ---------------------------
                # ONNXPredictor
                # ---------------------------
                result = self.predictor.predict_frame(frame)

                # ---------------------------
                # Draw Bounding Boxes
                # ---------------------------
                annotated = self._draw(frame, result)

                writer.write(annotated)

        finally:
            cap.release()
            writer.release()

        return output_path

    def _draw(self, frame, result):
        image = frame.copy()

        for det in result.predictions:

            h, w = image.shape[:2]

            x1 = max(0, min(int(det.bbox[0]), w - 1))
            y1 = max(0, min(int(det.bbox[1]), h - 1))
            x2 = max(0, min(int(det.bbox[2]), w - 1))
            y2 = max(0, min(int(det.bbox[3]), h - 1))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"{det.class_name} {det.confidence:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            cv2.rectangle(
                image,
                (x1, max(0, y1 - th - 8)),
                (x1 + tw + 6, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                image,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return image