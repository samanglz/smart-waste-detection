from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import time

from src.inference.schemas import PredictionResult, Performance
from src.inference.preprocess import ImagePreprocessor
from src.inference.postprocess import PostProcessor


class ONNXPredictor:
    """
    ONNX Runtime inference wrapper.

    Handles image, frame, and batch inference through
    a shared inference + post-processing pipeline.
    """

    def __init__(
        self,
        model_path: Path,
        class_names: List[str],
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        self.preprocessor = ImagePreprocessor(input_size)

        self.postprocessor = PostProcessor(
            class_names=class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    def predict_image(
        self,
        image_path: Path,
    ) -> PredictionResult:
        """
        Run inference on an image file.
        """

        tensor, _, ratio, pad = self.preprocessor.preprocess(
            image_path
        )

        # Read original image dimensions.
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        image_height, image_width = image.shape[:2]

        return self._predict_tensor(
            tensor=tensor,
            ratio=ratio,
            pad=pad,
            image_width=image_width,
            image_height=image_height,
        )

    def predict_frame(
        self,
        frame: np.ndarray,
    ) -> PredictionResult:
        """
        Run inference directly on an OpenCV BGR frame.

        Used by video and live-camera inference.
        """

        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame.")

        image_height, image_width = frame.shape[:2]

        tensor, ratio, pad = self.preprocessor.preprocess_frame(
            frame
        )

        return self._predict_tensor(
            tensor=tensor,
            ratio=ratio,
            pad=pad,
            image_width=image_width,
            image_height=image_height,
        )

    def predict_batch(
        self,
        image_paths: List[Path],
    ) -> List[PredictionResult]:
        """
        Run inference on multiple image files.
        """

        results = []

        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)

        return results

    def _predict_tensor(
        self,
        tensor: np.ndarray,
        ratio: float,
        pad,
        image_width: int,
        image_height: int,
    ) -> PredictionResult:
        """
        Shared ONNX inference + post-processing path.
        """

        start = time.perf_counter()

        outputs = self.session.run(
            None,
            {
                self.input_name: tensor
            },
        )

        inference_ms = (
            time.perf_counter() - start
        ) * 1000.0

        detections = self.postprocessor.process(
            output=outputs[0],
            ratio=ratio,
            pad=pad,
            image_width=image_width,
            image_height=image_height,
        )

        fps = (
            1000.0 / inference_ms
            if inference_ms > 0
            else 0.0
        )

        return PredictionResult(
            predictions=detections,
            performance=Performance(
                inference_time_ms=round(
                    inference_ms,
                    2,
                ),
                fps=round(
                    fps,
                    2,
                ),
                num_detections=len(detections),
            ),
        )