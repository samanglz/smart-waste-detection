from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.deployment.runners.base_runner import BaseRunner
from src.deployment.utils.image_preprocess import preprocess_image


class ONNXRunner(BaseRunner):
    """
    ONNX Runtime inference runner.

    Runtime-specific responsibilities such as provider selection,
    inference execution, and runtime synchronization are kept inside
    this class so that BenchmarkEngine remains runtime-agnostic.
    """

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = imgsz

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        if not self.model_path.is_file():
            raise ValueError(
                f"ONNX model path is not a file: {self.model_path}"
            )

        if imgsz <= 0:
            raise ValueError(
                f"imgsz must be greater than zero, got: {imgsz}"
            )

        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        self._requested_providers = list(providers)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self._requested_providers,
        )

        self.input_name = self.session.get_inputs()[0].name

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

    @property
    def runtime_name(self) -> str:
        """Return the canonical runtime identifier."""

        return "onnx"

    @property
    def device(self) -> str:
        """
        Determine the effective execution device from the active
        ONNX Runtime providers.
        """

        providers = self.providers

        if "CUDAExecutionProvider" in providers:
            return "cuda"

        if "TensorrtExecutionProvider" in providers:
            return "tensorrt"

        return "cpu"

    @property
    def requested_providers(self) -> list[str]:
        """Return providers requested during session creation."""

        return list(self._requested_providers)

    @property
    def actual_providers(self) -> list[str]:
        """Return providers reported by the active ONNX session."""

        return list(self.session.get_providers())

    @property
    def providers(self) -> list[str]:
        """
        Backward-compatible alias for actual providers.

        Existing deployment verification code can continue using
        runner.providers.
        """

        return self.actual_providers

    @property
    def input_shape(self):
        """Return the ONNX model input shape."""

        return self.session.get_inputs()[0].shape


    @property
    def output_shapes(self):
        """Return ONNX model output shapes."""

        return [
            output.shape
            for output in self.session.get_outputs()
        ]
        
        
    def predict(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Preprocess an image and run inference.
        """

        input_tensor = preprocess_image(
            image,
            size=self.imgsz,
        )

        return self.predict_tensor(input_tensor)

    def predict_tensor(
        self,
        input_tensor: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Run inference on an already-preprocessed tensor.
        """

        if not isinstance(input_tensor, np.ndarray):
            raise TypeError(
                "input_tensor must be a numpy.ndarray."
            )

        if input_tensor.ndim != 4:
            raise ValueError(
                "Expected input tensor with shape "
                "(B, C, H, W)."
            )

        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)

        input_tensor = np.ascontiguousarray(input_tensor)

        return self.session.run(
            self.output_names,
            {
                self.input_name: input_tensor,
            },
        )

    def synchronize(self) -> None:
        """
        ONNX Runtime session.run() is synchronous from the Python API
        perspective, so no explicit synchronization is required here.
        """

        return None

    def metadata(self) -> dict[str, Any]:
        """
        Return ONNX-specific metadata for benchmark reporting.
        """

        metadata = super().metadata()

        metadata.update(
            {
                "model_path": str(self.model_path.resolve()),
                "imgsz": self.imgsz,
                "input_name": self.input_name,
                "input_shape": list(self.input_shape),
                "output_names": self.output_names,
                "output_shapes": [
                    list(shape)
                    for shape in self.output_shapes
                ],
            }
        )

        return metadata